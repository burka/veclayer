//! Background embedding worker — polls pending entries and computes embeddings.
//!
//! Spawned as a `tokio::spawn` task in both stdio and HTTP server modes.
//! Polls `get_pending_embeddings`, embeds in batches, updates via `update_embedding`.
//! All errors are logged with `tracing::warn`, never panics.

use std::sync::Arc;

use tracing::warn;

use crate::blob_store::BlobStore;
use crate::store::StoreBackend;
use crate::{Embedder, VectorStore};

const BATCH_SIZE: usize = 32;
const POLL_INTERVAL_IDLE: std::time::Duration = std::time::Duration::from_secs(10);
const POLL_INTERVAL_BUSY: std::time::Duration = std::time::Duration::from_secs(2);
/// Approximate embed time per batch (inference + overhead).
const EMBED_TIME_SECS: u64 = 2;

/// Conservative ETA for processing `pending` entries.
pub(crate) fn eta_seconds(pending: usize) -> u64 {
    let batches = pending.div_ceil(BATCH_SIZE) as u64;
    batches * (POLL_INTERVAL_BUSY.as_secs() + EMBED_TIME_SECS)
}

/// Spawn the background embedding worker. Returns the `JoinHandle` for the task.
pub fn spawn(
    store: Arc<StoreBackend>,
    embedder: Arc<dyn Embedder + Send + Sync>,
    blob_store: Arc<BlobStore>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            match process_batch(&store, &embedder, &blob_store).await {
                Ok(0) => {
                    // No pending entries — sleep longer
                    tokio::time::sleep(POLL_INTERVAL_IDLE).await;
                }
                Ok(n) => {
                    tracing::debug!("Embedded {} pending entries", n);
                    // More may be waiting — poll again soon
                    tokio::time::sleep(POLL_INTERVAL_BUSY).await;
                }
                Err(e) => {
                    warn!("Embedding worker error: {e}");
                    tokio::time::sleep(POLL_INTERVAL_IDLE).await;
                }
            }
        }
    })
}

/// Process one batch of pending entries. Returns the number processed.
async fn process_batch(
    store: &Arc<StoreBackend>,
    embedder: &Arc<dyn Embedder + Send + Sync>,
    blob_store: &Arc<BlobStore>,
) -> crate::Result<usize> {
    let pending = store.get_pending_embeddings(BATCH_SIZE).await?;
    if pending.is_empty() {
        return Ok(0);
    }

    // CPU-bound embedding — run off the async executor
    let embedder_clone = Arc::clone(embedder);
    let texts: Vec<String> = pending.iter().map(|c| c.content.clone()).collect();
    let embeddings = tokio::task::spawn_blocking(move || {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        embedder_clone.embed(&refs)
    })
    .await
    .map_err(|e| crate::Error::embedding(format!("Embedding task panicked: {e}")))??;

    if embeddings.len() != pending.len() {
        return Err(crate::Error::embedding(format!(
            "Embedding count mismatch: expected {}, got {}",
            pending.len(),
            embeddings.len()
        )));
    }

    let count = pending.len();

    // Build batch update: (chunk_id, embedding) pairs
    let updates: Vec<(String, Vec<f32>)> = pending
        .iter()
        .zip(embeddings.iter())
        .map(|(chunk, emb)| (chunk.id.clone(), emb.clone()))
        .collect();

    if let Err(e) = store.batch_update_embeddings(updates).await {
        warn!("Batch embedding update failed: {e}");
        return Ok(0);
    }

    // Update blob store for each embedded entry
    let embedder_name = embedder.name();
    for (chunk, embedding) in pending.iter().zip(embeddings.into_iter()) {
        let mut updated = chunk.clone();
        updated.embedding = Some(embedding);
        let blob = crate::entry::StoredBlob::from_chunk_and_embedding(&updated, embedder_name);
        if let Err(e) = blob_store.put(&blob) {
            warn!("Failed to update blob for {}: {e}", chunk.id);
        }
    }

    Ok(count)
}
