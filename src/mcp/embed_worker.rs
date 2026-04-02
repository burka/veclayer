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
        let mut chunk_with_embedding = chunk.clone();
        chunk_with_embedding.embedding = Some(embedding);
        let blob = crate::entry::StoredBlob::from_chunk_and_embedding(
            &chunk_with_embedding,
            embedder_name,
        );
        if let Err(e) = blob_store.put(&blob) {
            warn!("Failed to update blob for {}: {e}", chunk.id);
        }
    }

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create pending chunks (embedding = None) for batch processing tests.
    fn make_pending_chunks(count: usize, prefix: &str) -> Vec<crate::HierarchicalChunk> {
        (0..count)
            .map(|i| {
                let id = format!("{}{:0>width$}", prefix, i, width = 64 - prefix.len());
                let content = format!("content {i}");
                let mut chunk = crate::test_helpers::make_test_chunk(&id, &content);
                chunk.embedding = None;
                chunk
            })
            .collect()
    }

    // ── eta_seconds ──────────────────────────────────────────────────────

    #[test]
    fn eta_seconds_zero_pending_returns_zero() {
        assert_eq!(eta_seconds(0), 0);
    }

    #[test]
    fn eta_seconds_one_batch_equals_single_batch_cost() {
        let expected = POLL_INTERVAL_BUSY.as_secs() + EMBED_TIME_SECS;
        assert_eq!(eta_seconds(1), expected);
    }

    #[test]
    fn eta_seconds_exactly_one_full_batch() {
        let expected = POLL_INTERVAL_BUSY.as_secs() + EMBED_TIME_SECS;
        assert_eq!(eta_seconds(BATCH_SIZE), expected);
    }

    #[test]
    fn eta_seconds_one_over_batch_size_rounds_up() {
        let expected = 2 * (POLL_INTERVAL_BUSY.as_secs() + EMBED_TIME_SECS);
        assert_eq!(eta_seconds(BATCH_SIZE + 1), expected);
    }

    #[test]
    fn eta_seconds_two_full_batches() {
        let expected = 2 * (POLL_INTERVAL_BUSY.as_secs() + EMBED_TIME_SECS);
        assert_eq!(eta_seconds(BATCH_SIZE * 2), expected);
    }

    #[test]
    fn eta_seconds_scales_linearly_with_batches() {
        let cost_per_batch = POLL_INTERVAL_BUSY.as_secs() + EMBED_TIME_SECS;
        for batches in 1usize..=5 {
            let pending = batches * BATCH_SIZE;
            assert_eq!(
                eta_seconds(pending),
                batches as u64 * cost_per_batch,
                "failed for {batches} batch(es)"
            );
        }
    }

    #[test]
    fn batch_size_constant_is_positive() {
        const {
            assert!(BATCH_SIZE > 0);
        }
    }

    #[test]
    fn poll_interval_idle_is_longer_than_busy() {
        assert!(POLL_INTERVAL_IDLE > POLL_INTERVAL_BUSY);
    }

    // ── process_batch ─────────────────────────────────────────────────────

    struct FixedEmbedder;

    impl crate::Embedder for FixedEmbedder {
        fn embed(&self, texts: &[&str]) -> crate::Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.5f32; 384]).collect())
        }

        fn dimension(&self) -> usize {
            384
        }

        fn name(&self) -> &str {
            "fixed-embedder"
        }
    }

    async fn make_store_and_blobs(dir: &std::path::Path) -> (Arc<StoreBackend>, Arc<BlobStore>) {
        let store = StoreBackend::open(dir, 384, false).await.unwrap();
        let blob_store = BlobStore::open(dir).unwrap();
        (Arc::new(store), Arc::new(blob_store))
    }

    /// Shared harness for process_batch tests with FixedEmbedder.
    async fn batch_harness_fixed(
        dir: &std::path::Path,
    ) -> (
        Arc<StoreBackend>,
        Arc<BlobStore>,
        Arc<dyn crate::Embedder + Send + Sync>,
    ) {
        let (store, blob_store) = make_store_and_blobs(dir).await;
        let embedder: Arc<dyn crate::Embedder + Send + Sync> = Arc::new(FixedEmbedder);
        (store, blob_store, embedder)
    }

    #[tokio::test]
    async fn process_batch_empty_store_returns_zero() {
        let dir = tempfile::tempdir().unwrap();
        let (store, blob_store, embedder) = batch_harness_fixed(dir.path()).await;

        let count = process_batch(&store, &embedder, &blob_store).await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn process_batch_embeds_pending_entries() {
        let dir = tempfile::tempdir().unwrap();
        let (store, blob_store, embedder) = batch_harness_fixed(dir.path()).await;

        let mut chunk = crate::test_helpers::make_test_chunk(
            "embed001deadbeef1234567890abcdef12345678",
            "embed me please",
        );
        chunk.embedding = None;
        store.insert_chunks(vec![chunk]).await.unwrap();

        let count = process_batch(&store, &embedder, &blob_store).await.unwrap();
        assert_eq!(count, 1, "should have embedded 1 pending entry");
    }

    #[tokio::test]
    async fn process_batch_already_embedded_returns_zero() {
        let dir = tempfile::tempdir().unwrap();
        let (store, blob_store, embedder) = batch_harness_fixed(dir.path()).await;

        // make_test_chunk sets embedding = Some(vec![0.0; 384])
        let chunk = crate::test_helpers::make_test_chunk(
            "embedded1deadbeef1234567890abcdef12345678",
            "already has embedding",
        );
        assert!(chunk.embedding.is_some());
        store.insert_chunks(vec![chunk]).await.unwrap();

        let count = process_batch(&store, &embedder, &blob_store).await.unwrap();
        assert_eq!(count, 0, "no pending entries to embed");
    }

    #[tokio::test]
    async fn process_batch_processes_at_most_batch_size() {
        let dir = tempfile::tempdir().unwrap();
        let (store, blob_store, embedder) = batch_harness_fixed(dir.path()).await;

        let chunks = make_pending_chunks(BATCH_SIZE + 5, "pend");
        store.insert_chunks(chunks).await.unwrap();

        let count = process_batch(&store, &embedder, &blob_store).await.unwrap();
        assert_eq!(count, BATCH_SIZE, "should process exactly {BATCH_SIZE}");
    }

    struct MismatchEmbedder;

    impl crate::Embedder for MismatchEmbedder {
        fn embed(&self, texts: &[&str]) -> crate::Result<Vec<Vec<f32>>> {
            // Return fewer embeddings than texts to trigger count mismatch
            if texts.len() > 1 {
                Ok(vec![vec![0.1f32; 384]])
            } else {
                Ok(texts.iter().map(|_| vec![0.1f32; 384]).collect())
            }
        }

        fn dimension(&self) -> usize {
            384
        }

        fn name(&self) -> &str {
            "mismatch-embedder"
        }
    }

    #[tokio::test]
    async fn process_batch_returns_error_on_embedding_count_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let (store, blob_store) = make_store_and_blobs(dir.path()).await;
        let embedder: Arc<dyn crate::Embedder + Send + Sync> = Arc::new(MismatchEmbedder);

        // Insert 2 pending chunks so embedder returns fewer than expected
        let chunks = make_pending_chunks(2, "mismatch");
        store.insert_chunks(chunks).await.unwrap();

        let result = process_batch(&store, &embedder, &blob_store).await;
        assert!(result.is_err(), "should fail with count mismatch error");
        let err_str = result.unwrap_err().to_string();
        assert!(
            err_str.contains("mismatch"),
            "error should mention mismatch: {err_str}"
        );
    }
}
