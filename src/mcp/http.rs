//! HTTP REST API server.

use std::sync::Arc;

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

use crate::embedder::FastEmbedder;
use crate::store::LanceStore;
use crate::{Config, Embedder, Result, VectorStore};

use super::tools;
use super::types::*;

/// Shared application state
#[derive(Clone)]
struct AppState {
    store: Arc<LanceStore>,
    embedder: Arc<FastEmbedder>,
    data_dir: std::path::PathBuf,
}

/// Run the HTTP REST API server.
pub async fn run_http(config: Config) -> Result<()> {
    let embedder = FastEmbedder::new()?;
    let dimension = embedder.dimension();
    let store = LanceStore::open(&config.data_dir, dimension, config.read_only).await?;

    let state = AppState {
        store: Arc::new(store),
        embedder: Arc::new(embedder),
        data_dir: config.data_dir.clone(),
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/api/recall", post(api_recall))
        .route("/api/focus", post(api_focus))
        .route("/api/store", post(api_store))
        .route("/api/think", post(api_think))
        .route("/api/share", post(api_share))
        .route("/api/stats", get(api_stats))
        .route("/api/identity", get(api_identity))
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
    match tools::execute_recall(&state.store, &state.embedder, input).await {
        Ok(results) => Json(ApiResponse::Success(results)),
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
    }
}

async fn api_focus(
    State(state): State<AppState>,
    Json(input): Json<FocusInput>,
) -> Json<ApiResponse<FocusResponse>> {
    match tools::execute_focus(&state.store, &state.embedder, input).await {
        Ok(response) => Json(ApiResponse::Success(response)),
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
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
    match tools::execute_store(&state.store, &state.embedder, input).await {
        Ok(chunk_id) => Json(ApiResponse::Success(format!("Stored. ID: {}", chunk_id))),
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
    }
}

async fn api_think(
    State(state): State<AppState>,
    Json(input): Json<ThinkInput>,
) -> Json<ApiResponse<String>> {
    match tools::execute_think(&state.store, &state.data_dir, input).await {
        Ok(text) => Json(ApiResponse::Success(text)),
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
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
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
    }
}

/// Identity endpoint — mirrors stdio auto-priming on MCP initialize.
async fn api_identity(State(state): State<AppState>) -> Json<ApiResponse<serde_json::Value>> {
    match crate::identity::compute_identity(state.store.as_ref(), &state.data_dir).await {
        Ok(snapshot) => {
            let priming = crate::identity::generate_priming(&snapshot);
            let instructions = if priming.len() > 50 {
                format!("{}\n\n---\n\n{}", super::MCP_INSTRUCTIONS, priming)
            } else {
                super::MCP_INSTRUCTIONS.to_string()
            };
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
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
    }
}
