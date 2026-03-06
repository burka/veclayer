//! HTTP REST API server with Streamable HTTP MCP transport.

use std::collections::HashMap;
use std::result::Result as StdResult;
use std::sync::{Arc, Mutex};

use axum::extract::rejection::JsonRejection;
use axum::extract::{Extension, State};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::Method;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

use crate::auth::capability::Capability;
use crate::auth::middleware::{auth_middleware, AuthState};
use crate::auth::oauth::{oauth_router, OAuthState};
use crate::auth::token_store::TokenStore;
use crate::blob_store::BlobStore;
use crate::embedder;
use crate::store::StoreBackend;
use crate::{Config, Embedder, Result, VectorStore};

use super::handler::McpHandler;
use super::tools;
use super::types::*;

// ─── Auth setup ───────────────────────────────────────────────────────────────

/// Pre-built auth state passed into [`build_app`] when authentication is
/// configured.  Created in [`run_http`] from the keystore and serve options.
#[derive(Clone)]
pub struct AuthSetup {
    pub auth_state: AuthState,
    pub oauth_state: OAuthState,
}

// ─── Application state ────────────────────────────────────────────────────────

/// Shared application state for both REST API handlers and MCP session factory.
#[derive(Clone)]
pub struct AppState {
    pub store: Arc<StoreBackend>,
    pub embedder: Arc<dyn Embedder + Send + Sync>,
    pub blob_store: Arc<BlobStore>,
    pub data_dir: std::path::PathBuf,
    pub project: Option<String>,
    pub branch: Option<String>,
    /// Present when authentication is enabled.
    pub auth: Option<AuthSetup>,
    /// Git memory store for persisting entries to the memory branch, if configured.
    pub git_store: Option<Arc<crate::git::memory_store::MemoryStore>>,
    /// Push mode for git storage.
    pub push_mode: crate::git::branch_config::PushMode,
}

// ─── Error types ──────────────────────────────────────────────────────────────

/// HTTP error response with proper status codes.
struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }

    fn forbidden(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            message: msg.into(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({ "error": self.message });
        (self.status, Json(body)).into_response()
    }
}

impl From<crate::Error> for AppError {
    fn from(err: crate::Error) -> Self {
        let status = match &err {
            crate::Error::NotFound(_) => StatusCode::NOT_FOUND,
            crate::Error::Parse(_)
            | crate::Error::InvalidOperation(_)
            | crate::Error::Config(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self {
            status,
            message: err.to_string(),
        }
    }
}

impl From<JsonRejection> for AppError {
    fn from(rejection: JsonRejection) -> Self {
        Self::bad_request(rejection.body_text())
    }
}

/// Log a warning and convert a domain error into an HTTP error response.
fn warn_and_convert(context: &str) -> impl FnOnce(crate::Error) -> AppError + '_ {
    move |e| {
        tracing::warn!("{context} failed: {e}");
        AppError::from(e)
    }
}

// ─── Capability helpers ───────────────────────────────────────────────────────

/// Return a 403 `AppError` for an insufficient capability.
fn insufficient(required: Capability) -> AppError {
    AppError::forbidden(format!("Insufficient permission: need {required}"))
}

// ─── Router builder ───────────────────────────────────────────────────────────

/// Build the application router.
///
/// When `state.auth` is `Some`, OAuth endpoints and the auth middleware are
/// wired in.  When it is `None` the server runs fully open (backward-compatible
/// mode for local single-agent use).
///
/// # Security
///
/// **Open mode** (`auth = None`): any process that can reach the socket has
/// full Admin access.  Mitigated by localhost-only binding.
///
/// **Auth mode** (`auth = Some`): Bearer JWT tokens are required for all
/// `/api/*` and `/mcp/*` routes.  `/health` and OAuth endpoints remain public.
///
/// # Rate limiting
///
/// No request throttling.  For multi-tenant or remote use, add
/// `tower_governor` or a similar layer before deploying.
pub fn build_app(state: AppState) -> Router {
    let cors = build_cors(state.auth.as_ref().and_then(|a| {
        let url = &a.oauth_state.server_url;
        if url.starts_with("http://localhost") || url.starts_with("http://127.0.0.1") {
            None
        } else {
            Some(url.clone())
        }
    }));

    // MCP sessions inherit Admin: once a request has passed the auth middleware
    // (which rejects unauthenticated or under-privileged callers), the session
    // factory closure has no access to per-request extensions.  The middleware
    // already acts as the enforcement boundary for /mcp/*.
    let mcp_capability = Capability::Admin;

    let mcp_state = state.clone();
    let mcp_service = StreamableHttpService::new(
        move || {
            let instructions = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(super::compute_instructions(
                    mcp_state.store.as_ref(),
                    &mcp_state.data_dir,
                    mcp_state.project.as_deref(),
                    mcp_state.branch.as_deref(),
                ))
            });
            Ok(McpHandler::new(
                Arc::clone(&mcp_state.store),
                Arc::clone(&mcp_state.embedder),
                Arc::clone(&mcp_state.blob_store),
                mcp_state.data_dir.clone(),
                mcp_state.project.clone(),
                mcp_state.branch.clone(),
                instructions,
                mcp_capability,
                mcp_state.git_store.clone(),
                mcp_state.push_mode,
            ))
        },
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );

    // Routes that require authorization.
    let protected: Router<AppState> = Router::new()
        .route("/api/recall", post(api_recall))
        .route("/api/focus", post(api_focus))
        .route("/api/store", post(api_store))
        .route("/api/think", post(api_think))
        .route("/api/share", post(api_share))
        .route("/api/stats", get(api_stats))
        .route("/api/identity", get(api_identity))
        .route("/api/priming", get(api_priming))
        .nest_service("/mcp", mcp_service);

    let base: Router<AppState> = Router::new().route("/health", get(|| async { "OK" }));

    let app: Router<AppState> = match state.auth.clone() {
        Some(auth_setup) => {
            // Inject Admin capability for open paths (health).  Auth middleware
            // will inject the token capability for protected paths.
            let open_with_cap = base.layer(axum::middleware::from_fn(inject_admin_capability));

            let guarded = protected.layer(axum::middleware::from_fn_with_state(
                auth_setup.auth_state,
                auth_middleware,
            ));

            let oauth: Router<AppState> = oauth_router(auth_setup.oauth_state).with_state(());

            Router::new()
                .merge(oauth)
                .merge(open_with_cap)
                .merge(guarded)
        }
        None => {
            // No auth: inject Admin for all requests so handlers can read the
            // Capability extension uniformly.
            let all_routes = base.merge(protected);
            all_routes.layer(axum::middleware::from_fn(inject_admin_capability))
        }
    };

    app.layer(TraceLayer::new_for_http())
        .layer(RequestBodyLimitLayer::new(4 * 1024 * 1024)) // 4 MiB
        .layer(cors)
        .with_state(state)
}

/// Middleware that injects [`Capability::Admin`] into request extensions.
///
/// Used on routes that are always public (e.g. `/health`) so that handlers
/// can read capability extensions uniformly regardless of auth mode.
async fn inject_admin_capability(
    mut request: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> Response {
    request.extensions_mut().insert(Capability::Admin);
    next.run(request).await
}

/// Build the CORS layer.
///
/// Always allows localhost origins.  When a remote `server_url` is provided
/// it is added as an allowed origin alongside `claude.ai` origins.
/// The `Authorization` header is always included in allowed headers.
fn build_cors(server_url: Option<String>) -> CorsLayer {
    let extra_origin = server_url.clone();
    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(move |origin, _| {
            let s = origin.as_bytes();
            if s.starts_with(b"http://localhost")
                || s.starts_with(b"http://127.0.0.1")
                || s == b"https://claude.ai"
                || (s.starts_with(b"https://") && s.ends_with(b".claude.ai"))
            {
                return true;
            }
            if let Some(url) = &extra_origin {
                let origin_str = std::str::from_utf8(s).unwrap_or_default();
                if origin_str == url.as_str() {
                    return true;
                }
            }
            false
        }))
        // DELETE is required by the MCP Streamable HTTP spec for session termination.
        .allow_methods([Method::GET, Method::POST, Method::DELETE])
        .allow_headers([CONTENT_TYPE, AUTHORIZATION])
}

// ─── Server startup ───────────────────────────────────────────────────────────

/// Run the HTTP REST API server.
pub async fn run_http(config: Config) -> Result<()> {
    let embedder: Arc<dyn Embedder + Send + Sync> =
        Arc::from(embedder::from_config(&config.embedder)?);
    let dimension = embedder.dimension();
    let store = StoreBackend::open(&config.data_dir, dimension, config.read_only).await?;
    let blob_store = BlobStore::open(&config.data_dir)?;

    let store = Arc::new(store);
    let blob_store = Arc::new(blob_store);

    if !config.read_only {
        let _worker = super::embed_worker::spawn(
            Arc::clone(&store),
            Arc::clone(&embedder),
            Arc::clone(&blob_store),
        );
    }

    let auth = build_auth_setup(&config)?;

    let push_mode = config.push_mode;
    let git_store = if push_mode.uses_git() {
        open_git_store(&config)
    } else {
        None
    };

    let state = AppState {
        store,
        embedder,
        blob_store,
        data_dir: config.data_dir.clone(),
        project: config.project.clone(),
        branch: config.branch.clone(),
        auth,
        git_store,
        push_mode,
    };

    let app = build_app(state);

    let addr: std::net::SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .map_err(|e| crate::Error::config(format!("Invalid address: {}", e)))?;

    info!("VecLayer HTTP server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(crate::Error::Io)?;

    axum::serve(listener, app)
        .await
        .map_err(|e| crate::Error::InvalidOperation(format!("Server error: {}", e)))?;

    Ok(())
}

/// Try to open the git memory store for the current project.
///
/// Returns `None` when git storage is not configured or the git dir cannot be found.
pub(super) fn open_git_store(
    config: &Config,
) -> Option<Arc<crate::git::memory_store::MemoryStore>> {
    if config.storage.as_deref() != Some("git") {
        return None;
    }

    let cwd = std::env::current_dir().ok()?;
    let git_dir = crate::git::detect::find_git_dir(&cwd)?;

    match crate::git::memory_store::MemoryStore::open(&git_dir, None) {
        Ok(store) => {
            tracing::info!("Git memory store opened for MCP sessions");
            Some(Arc::new(store))
        }
        Err(e) => {
            tracing::warn!("Failed to open git memory store: {e}");
            None
        }
    }
}

/// Load the keystore and build auth state, or return `None` when auth is
/// disabled and no keystore exists.
///
/// Errors when `auth_required=true` but no identity has been initialised.
fn build_auth_setup(config: &Config) -> Result<Option<AuthSetup>> {
    use crate::crypto::keypair;
    use crate::crypto::keystore;

    let keystore_path = keystore::keystore_path(&config.data_dir);
    let auth_required = config.auth.auth_required;

    if !keystore::exists(&keystore_path) {
        if auth_required {
            return Err(crate::Error::Config(
                "Authentication is required but no server identity exists. \
                 Run `veclayer identity init` first."
                    .to_owned(),
            ));
        }
        // No identity and auth not required — run fully open.
        return Ok(None);
    }

    let passphrase = std::env::var("VECLAYER_PASSPHRASE").unwrap_or_default();
    let signing_key = keystore::load(&passphrase, &keystore_path)
        .map_err(|e| crate::Error::Config(format!("Failed to load identity: {e}")))?;

    let verifying_key = signing_key.verifying_key();
    let did = keypair::to_did(&verifying_key);

    let server_url = config
        .auth
        .server_url
        .clone()
        .unwrap_or_else(|| format!("http://{}:{}", config.host, config.port));

    let mut token_store = TokenStore::open(&config.data_dir)
        .map_err(|e| crate::Error::Config(format!("Failed to open token store: {e}")))?;
    token_store.purge_expired();

    let oauth_state = OAuthState {
        token_store: Arc::new(Mutex::new(token_store)),
        signing_key: Arc::new(signing_key),
        server_did: did.clone(),
        server_url,
        token_expiry_secs: config.auth.token_expiry_secs,
        refresh_expiry_secs: config.auth.refresh_expiry_secs,
        auto_approve: config.auth.auto_approve,
        device_codes: Arc::new(Mutex::new(HashMap::new())),
    };

    let auth_state = AuthState {
        verifying_key,
        server_did: did,
        auth_required,
    };

    Ok(Some(AuthSetup {
        auth_state,
        oauth_state,
    }))
}

// ─── REST API handlers ────────────────────────────────────────────────────────

async fn api_recall(
    State(state): State<AppState>,
    Extension(cap): Extension<Capability>,
    body: StdResult<Json<RecallInput>, JsonRejection>,
) -> StdResult<Json<Vec<SearchResultResponse>>, AppError> {
    if !cap.permits(Capability::Read) {
        return Err(insufficient(Capability::Read));
    }
    let Json(input) = body?;
    let results = tools::execute_recall(
        &state.store,
        &state.embedder,
        input,
        state.project.as_deref(),
        state.branch.as_deref(),
    )
    .await
    .map_err(warn_and_convert("Recall"))?;
    Ok(Json(results))
}

async fn api_focus(
    State(state): State<AppState>,
    Extension(cap): Extension<Capability>,
    body: StdResult<Json<FocusInput>, JsonRejection>,
) -> StdResult<Json<FocusResponse>, AppError> {
    if !cap.permits(Capability::Read) {
        return Err(insufficient(Capability::Read));
    }
    let Json(input) = body?;
    let response = tools::execute_focus(
        &state.store,
        &state.embedder,
        input,
        state.project.as_deref(),
        state.branch.as_deref(),
    )
    .await
    .map_err(warn_and_convert("Focus"))?;
    Ok(Json(response))
}

async fn api_store(
    State(state): State<AppState>,
    Extension(cap): Extension<Capability>,
    body: StdResult<Json<StoreInput>, JsonRejection>,
) -> StdResult<Json<String>, AppError> {
    if !cap.permits(Capability::Write) {
        return Err(insufficient(Capability::Write));
    }
    let Json(input) = body?;
    if input.content.is_empty() {
        return Err(AppError::bad_request("content is required"));
    }
    let effective_git_store = if state.push_mode.auto_stages() {
        state.git_store.as_deref()
    } else {
        None
    };
    let result = tools::execute_store(
        &state.store,
        &state.embedder,
        &state.blob_store,
        input,
        state.project.as_deref(),
        state.branch.as_deref(),
        effective_git_store,
        state.push_mode,
    )
    .await
    .map_err(warn_and_convert("Store"))?;
    Ok(Json(result.as_str().unwrap_or_default().to_string()))
}

async fn api_think(
    State(state): State<AppState>,
    Extension(cap): Extension<Capability>,
    body: StdResult<Json<ThinkInput>, JsonRejection>,
) -> StdResult<Json<String>, AppError> {
    if !cap.permits(Capability::Write) {
        return Err(insufficient(Capability::Write));
    }
    let Json(input) = body?;
    let text = tools::execute_think(
        &state.store,
        &state.data_dir,
        &state.blob_store,
        input,
        state.project.as_deref(),
        state.branch.as_deref(),
        state.git_store.as_deref(),
        Some(state.push_mode),
    )
    .await
    .map_err(warn_and_convert("Think"))?;
    Ok(Json(text))
}

async fn api_share(
    Extension(cap): Extension<Capability>,
    Json(input): Json<ShareInput>,
) -> StdResult<Json<serde_json::Value>, AppError> {
    if !cap.permits(Capability::Write) {
        return Err(insufficient(Capability::Write));
    }
    Ok(Json(tools::build_share_token(input)))
}

async fn api_stats(
    State(state): State<AppState>,
    Extension(cap): Extension<Capability>,
) -> StdResult<Json<serde_json::Value>, AppError> {
    if !cap.permits(Capability::Read) {
        return Err(insufficient(Capability::Read));
    }
    let stats = state
        .store
        .stats()
        .await
        .map_err(warn_and_convert("Stats"))?;
    Ok(Json(serde_json::json!({
        "total_chunks": stats.total_chunks,
        "chunks_by_level": stats.chunks_by_level,
        "source_files": stats.source_files,
        "pending_embeddings": stats.pending_embeddings,
    })))
}

/// Identity endpoint — mirrors stdio auto-priming on MCP initialize.
async fn api_identity(
    State(state): State<AppState>,
    Extension(cap): Extension<Capability>,
) -> StdResult<Json<serde_json::Value>, AppError> {
    if !cap.permits(Capability::Read) {
        return Err(insufficient(Capability::Read));
    }
    let snapshot = crate::identity::compute_identity(
        state.store.as_ref(),
        &state.data_dir,
        state.project.as_deref(),
        state.branch.as_deref(),
    )
    .await
    .map_err(warn_and_convert("Identity computation"))?;
    let priming = crate::identity::generate_priming(&snapshot);
    let instructions = super::build_priming_text(&priming);
    Ok(Json(serde_json::json!({
        "instructions": instructions,
        "core_entries": snapshot.core_entries.len(),
        "open_threads": snapshot.open_threads.len(),
        "perspectives": snapshot.centroids.iter()
            .map(|c| serde_json::json!({
                "perspective": c.perspective,
                "entry_count": c.entry_count,
            }))
            .collect::<Vec<_>>(),
    })))
}

/// Priming endpoint — returns the full MCP instructions + identity briefing as plain text.
///
/// Agents connecting via HTTP can GET this endpoint on startup to receive the same
/// identity briefing that stdio agents receive on `initialize`.
async fn api_priming(
    State(state): State<AppState>,
    Extension(cap): Extension<Capability>,
) -> Response {
    if !cap.permits(Capability::Read) {
        return (StatusCode::FORBIDDEN, "Insufficient permission: need read").into_response();
    }
    match crate::identity::compute_identity(
        state.store.as_ref(),
        &state.data_dir,
        state.project.as_deref(),
        state.branch.as_deref(),
    )
    .await
    {
        Ok(snapshot) => {
            let priming = crate::identity::generate_priming(&snapshot);
            let text = super::build_priming_text(&priming);
            (
                StatusCode::OK,
                [("content-type", "text/plain; charset=utf-8")],
                text,
            )
                .into_response()
        }
        Err(e) => {
            tracing::warn!("Priming computation failed: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                [("content-type", "text/plain; charset=utf-8")],
                format!("Priming failed: {e}"),
            )
                .into_response()
        }
    }
}
