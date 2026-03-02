//! HTTP REST API server.

use std::sync::Arc;

use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::http::Method;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tower_http::cors::{AllowOrigin, CorsLayer};
use tracing::info;

use crate::blob_store::BlobStore;
use crate::embedder::FastEmbedder;
use crate::store::StoreBackend;
use crate::{Config, Embedder, Result, VectorStore};

use super::tools;
use super::types::*;

/// Shared application state
#[derive(Clone)]
struct AppState {
    store: Arc<StoreBackend>,
    embedder: Arc<FastEmbedder>,
    blob_store: Arc<BlobStore>,
    data_dir: std::path::PathBuf,
    project: Option<String>,
    branch: Option<String>,
}

/// Run the HTTP REST API server.
pub async fn run_http(config: Config) -> Result<()> {
    let embedder = FastEmbedder::new()?;
    let dimension = embedder.dimension();
    let store = StoreBackend::open(&config.data_dir, dimension, config.read_only).await?;
    let blob_store = BlobStore::open(&config.data_dir)?;

    let state = AppState {
        store: Arc::new(store),
        embedder: Arc::new(embedder),
        blob_store: Arc::new(blob_store),
        data_dir: config.data_dir.clone(),
        project: config.project.clone(),
        branch: config.branch.clone(),
    };

    // TODO(security): The HTTP API has no authentication. Any process that can reach
    // the bound socket can read/write the entire knowledge store. Currently mitigated
    // by localhost-only binding + restricted CORS. Add token-based auth before
    // exposing to untrusted networks. Tracked for a future release.
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, _| {
            let s = origin.as_bytes();
            s.starts_with(b"http://localhost") || s.starts_with(b"http://127.0.0.1")
        }))
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([CONTENT_TYPE]);

    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/api/recall", post(api_recall))
        .route("/api/focus", post(api_focus))
        .route("/api/store", post(api_store))
        .route("/api/think", post(api_think))
        .route("/api/share", post(api_share))
        .route("/api/stats", get(api_stats))
        .route("/api/identity", get(api_identity))
        .route("/api/priming", get(api_priming))
        .layer(cors)
        .with_state(state);

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
    Json(input): Json<RecallInput>,
) -> Json<ApiResponse<Vec<SearchResultResponse>>> {
    match tools::execute_recall(
        &state.store,
        &state.embedder,
        input,
        state.project.as_deref(),
        state.branch.as_deref(),
    )
    .await
    {
        Ok(results) => Json(ApiResponse::Success(results)),
        Err(e) => {
            tracing::warn!("Recall failed: {e}");
            Json(ApiResponse::Error {
                error: "Recall operation failed.".to_string(),
            })
        }
    }
}

async fn api_focus(
    State(state): State<AppState>,
    Json(input): Json<FocusInput>,
) -> Json<ApiResponse<FocusResponse>> {
    match tools::execute_focus(
        &state.store,
        &state.embedder,
        input,
        state.project.as_deref(),
        state.branch.as_deref(),
    )
    .await
    {
        Ok(response) => Json(ApiResponse::Success(response)),
        Err(e) => {
            tracing::warn!("Focus failed: {e}");
            Json(ApiResponse::Error {
                error: "Focus operation failed.".to_string(),
            })
        }
    }
}

async fn api_store(
    State(state): State<AppState>,
    Json(input): Json<StoreInput>,
) -> Json<ApiResponse<String>> {
    if input.content.is_empty() {
        return Json(ApiResponse::Error {
            error: "content is required".to_string(),
        });
    }
    match tools::execute_store(
        &state.store,
        &state.embedder,
        &state.blob_store,
        input,
        state.project.as_deref(),
        state.branch.as_deref(),
    )
    .await
    {
        Ok(chunk_id) => Json(ApiResponse::Success(format!("Stored. ID: {}", chunk_id))),
        Err(e) => {
            tracing::warn!("Store failed: {e}");
            Json(ApiResponse::Error {
                error: "Store operation failed.".to_string(),
            })
        }
    }
}

async fn api_think(
    State(state): State<AppState>,
    Json(input): Json<ThinkInput>,
) -> Json<ApiResponse<String>> {
    match tools::execute_think(
        &state.store,
        &state.data_dir,
        &state.blob_store,
        input,
        state.project.as_deref(),
        state.branch.as_deref(),
    )
    .await
    {
        Ok(text) => Json(ApiResponse::Success(text)),
        Err(e) => {
            tracing::warn!("Think failed: {e}");
            Json(ApiResponse::Error {
                error: "Think operation failed.".to_string(),
            })
        }
    }
}

async fn api_share(Json(input): Json<ShareInput>) -> Json<ApiResponse<serde_json::Value>> {
    Json(ApiResponse::Success(tools::build_share_token(input)))
}

async fn api_stats(State(state): State<AppState>) -> Json<ApiResponse<serde_json::Value>> {
    match state.store.stats().await {
        Ok(stats) => Json(ApiResponse::Success(serde_json::json!({
            "total_chunks": stats.total_chunks,
            "chunks_by_level": stats.chunks_by_level,
            "source_files": stats.source_files,
        }))),
        Err(e) => {
            tracing::warn!("Stats failed: {e}");
            Json(ApiResponse::Error {
                error: "Stats retrieval failed.".to_string(),
            })
        }
    }
}

/// Identity endpoint — mirrors stdio auto-priming on MCP initialize.
async fn api_identity(State(state): State<AppState>) -> Json<ApiResponse<serde_json::Value>> {
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
            let instructions = super::build_priming_text(&priming);
            Json(ApiResponse::Success(serde_json::json!({
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
        Err(e) => {
            tracing::warn!("Identity computation failed: {e}");
            Json(ApiResponse::Error {
                error: "Identity computation temporarily unavailable.".to_string(),
            })
        }
    }
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
                "Identity briefing temporarily unavailable.".to_string(),
            )
                .into_response()
        }
    }
}
