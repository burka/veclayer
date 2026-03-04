//! HTTP REST API server with Streamable HTTP MCP transport.

use std::result::Result as StdResult;
use std::sync::Arc;

use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
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

use crate::blob_store::BlobStore;
use crate::embedder;
use crate::store::StoreBackend;
use crate::{Config, Embedder, Result, VectorStore};

use super::handler::McpHandler;
use super::tools;
use super::types::*;

/// Shared application state for both REST API handlers and MCP session factory.
#[derive(Clone)]
pub struct AppState {
    pub store: Arc<StoreBackend>,
    pub embedder: Arc<dyn Embedder + Send + Sync>,
    pub blob_store: Arc<BlobStore>,
    pub data_dir: std::path::PathBuf,
    pub project: Option<String>,
    pub branch: Option<String>,
}

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

/// Build the application router (public for integration tests).
///
/// # Security
///
/// **No authentication.** Any process that can reach the bound socket can
/// read/write the entire knowledge store. Currently mitigated by:
/// - Localhost-only binding (default `127.0.0.1`)
/// - CORS restricted to `http://localhost*` and `http://127.0.0.1*`
///
/// Add token-based auth (e.g. `Authorization: Bearer`) before exposing
/// to untrusted networks.
///
/// **No rate limiting.** There is no request throttling on any endpoint.
/// For localhost single-agent use this is acceptable. Add
/// `tower_http::limit::RateLimitLayer` or similar middleware before
/// exposing to multi-tenant or remote environments.
pub fn build_app(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, _| {
            let s = origin.as_bytes();
            s.starts_with(b"http://localhost") || s.starts_with(b"http://127.0.0.1")
        }))
        // DELETE is required by the MCP Streamable HTTP spec for session termination.
        .allow_methods([Method::GET, Method::POST, Method::DELETE])
        .allow_headers([CONTENT_TYPE]);

    // Streamable HTTP MCP transport: one McpHandler per session.
    // Each session computes fresh identity priming so the agent gets
    // up-to-date knowledge context on connect.
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
            ))
        },
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );

    Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/api/recall", post(api_recall))
        .route("/api/focus", post(api_focus))
        .route("/api/store", post(api_store))
        .route("/api/think", post(api_think))
        .route("/api/share", post(api_share))
        .route("/api/stats", get(api_stats))
        .route("/api/identity", get(api_identity))
        .route("/api/priming", get(api_priming))
        .nest_service("/mcp", mcp_service)
        .layer(TraceLayer::new_for_http())
        .layer(RequestBodyLimitLayer::new(4 * 1024 * 1024)) // 4 MiB
        .layer(cors)
        .with_state(state)
}

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

    let state = AppState {
        store,
        embedder,
        blob_store,
        data_dir: config.data_dir.clone(),
        project: config.project.clone(),
        branch: config.branch.clone(),
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

async fn api_recall(
    State(state): State<AppState>,
    body: StdResult<Json<RecallInput>, JsonRejection>,
) -> StdResult<Json<Vec<SearchResultResponse>>, AppError> {
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
    body: StdResult<Json<FocusInput>, JsonRejection>,
) -> StdResult<Json<FocusResponse>, AppError> {
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
    body: StdResult<Json<StoreInput>, JsonRejection>,
) -> StdResult<Json<String>, AppError> {
    let Json(input) = body?;
    if input.content.is_empty() {
        return Err(AppError::bad_request("content is required"));
    }
    let chunk_id = tools::execute_store(
        &state.store,
        &state.embedder,
        &state.blob_store,
        input,
        state.project.as_deref(),
        state.branch.as_deref(),
    )
    .await
    .map_err(warn_and_convert("Store"))?;
    Ok(Json(format!("Stored. ID: {}", chunk_id)))
}

async fn api_think(
    State(state): State<AppState>,
    body: StdResult<Json<ThinkInput>, JsonRejection>,
) -> StdResult<Json<String>, AppError> {
    let Json(input) = body?;
    let text = tools::execute_think(
        &state.store,
        &state.data_dir,
        &state.blob_store,
        input,
        state.project.as_deref(),
        state.branch.as_deref(),
    )
    .await
    .map_err(warn_and_convert("Think"))?;
    Ok(Json(text))
}

async fn api_share(Json(input): Json<ShareInput>) -> Json<serde_json::Value> {
    Json(tools::build_share_token(input))
}

async fn api_stats(State(state): State<AppState>) -> StdResult<Json<serde_json::Value>, AppError> {
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
) -> StdResult<Json<serde_json::Value>, AppError> {
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
async fn api_priming(State(state): State<AppState>) -> Response {
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
