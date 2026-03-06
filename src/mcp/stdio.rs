//! MCP stdio transport — rmcp-based.

use std::sync::Arc;

use rmcp::{transport::stdio, ServiceExt};
use tracing::info;

use crate::auth::capability::Capability;
use crate::blob_store::BlobStore;
use crate::embedder;
use crate::store::StoreBackend;
use crate::{Config, Embedder, Result};

use super::handler::McpHandler;

/// Run the MCP server on stdio.
pub async fn run_stdio(config: Config) -> Result<()> {
    info!("Starting MCP stdio server...");

    // Build shared state (same initialization as before)
    let embedder: Arc<dyn Embedder + Send + Sync> =
        Arc::from(embedder::from_config(&config.embedder)?);
    let dimension = embedder.dimension();
    let store = StoreBackend::open(&config.data_dir, dimension, config.read_only).await?;
    let store = Arc::new(store);
    let blob_store = BlobStore::open(&config.data_dir)?;
    let blob_store = Arc::new(blob_store);

    // Spawn background embedding worker
    if !config.read_only {
        let _worker = super::embed_worker::spawn(
            Arc::clone(&store),
            Arc::clone(&embedder),
            Arc::clone(&blob_store),
        );
    }

    let instructions = super::compute_instructions(
        store.as_ref(),
        &config.data_dir,
        config.project.as_deref(),
        config.branch.as_deref(),
    )
    .await;

    let push_mode = config.push_mode;
    let git_store = if push_mode.uses_git() {
        super::http::open_git_store(&config)
    } else {
        None
    };

    // Create handler and serve via rmcp stdio transport.
    // Stdio transport is trusted (local process), so it always gets Admin capability.
    let handler = McpHandler::new(
        store,
        embedder,
        blob_store,
        config.data_dir.clone(),
        config.project.clone(),
        config.branch.clone(),
        instructions,
        Capability::Admin,
        git_store,
        push_mode,
    );

    let service = handler
        .serve(stdio())
        .await
        .map_err(|e| crate::Error::InvalidOperation(format!("MCP stdio error: {}", e)))?;

    // Block until the client disconnects
    service
        .waiting()
        .await
        .map_err(|e| crate::Error::InvalidOperation(format!("MCP stdio error: {}", e)))?;

    Ok(())
}
