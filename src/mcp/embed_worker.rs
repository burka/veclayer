//! Background embedding worker — polls pending entries and computes embeddings.
//!
//! Spawned as a `tokio::spawn` task in both stdio and HTTP server modes.
//! Polls `get_pending_embeddings`, embeds in batches, updates via `update_embedding`.
//! All errors are logged with `tracing::warn`, never panics.

use std::sync::Arc;

use tracing::warn;

use crate::blob_store::BlobStore;
use crate::store::StoreBackend;
use crate::Embedder;

const BATCH_SIZE: usize = 32;
const POLL_INTERVAL_IDLE: std::time::Duration = std::time::Duration::from_secs(10);
const POLL_INTERVAL_BUSY: std::time::Duration = std::time::Duration::from_secs(2);
/// Approximate embed time per batch (inference + overhead). Benchmarked at
/// ~129ms for a batch of 32 on CPU (tests/embedder_bench.rs); rounded up to 1s
/// to stay conservative on slower machines without grossly over-reporting ETAs.
const EMBED_TIME_SECS: u64 = 1;

// Regression guard: the prior 2s value over-reported ETAs ~15x versus the
// measured ~129ms batch-32 cost. Keep the budget at most 1s — bumping it back
// up should fail the build, not silently inflate user-facing ETAs.
const _: () = assert!(
    EMBED_TIME_SECS <= 1,
    "EMBED_TIME_SECS is too pessimistic; batch-32 measures ~129ms (tests/embedder_bench.rs)"
);

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
            match process_batch(store.as_ref(), &embedder, &blob_store).await {
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
                    // Retry on busy cadence — an error does not mean the queue
                    // is empty; sleeping the idle interval would drain ~5x slower.
                    tokio::time::sleep(POLL_INTERVAL_BUSY).await;
                }
            }
        }
    })
}

/// Process one batch of pending entries. Returns the number processed.
///
/// Generic over any [`VectorStore`] so the iteration logic can be exercised in
/// unit tests with a lightweight mock without spinning up a real database.
async fn process_batch(
    store: &impl crate::VectorStore,
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

    store.batch_update_embeddings(updates).await.map_err(|e| {
        warn!("Batch embedding update failed: {e}");
        e
    })?;

    // Update blob store for each embedded entry
    let embedder_name = embedder.name();
    for (chunk, embedding) in pending.iter().zip(embeddings) {
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
    use crate::VectorStore as _;

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

        let count = process_batch(store.as_ref(), &embedder, &blob_store)
            .await
            .unwrap();
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

        let count = process_batch(store.as_ref(), &embedder, &blob_store)
            .await
            .unwrap();
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

        let count = process_batch(store.as_ref(), &embedder, &blob_store)
            .await
            .unwrap();
        assert_eq!(count, 0, "no pending entries to embed");
    }

    #[tokio::test]
    async fn process_batch_processes_at_most_batch_size() {
        let dir = tempfile::tempdir().unwrap();
        let (store, blob_store, embedder) = batch_harness_fixed(dir.path()).await;

        let chunks = make_pending_chunks(BATCH_SIZE + 5, "pend");
        store.insert_chunks(chunks).await.unwrap();

        let count = process_batch(store.as_ref(), &embedder, &blob_store)
            .await
            .unwrap();
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

        let result = process_batch(store.as_ref(), &embedder, &blob_store).await;
        assert!(result.is_err(), "should fail with count mismatch error");
        let err_str = result.unwrap_err().to_string();
        assert!(
            err_str.contains("mismatch"),
            "error should mention mismatch: {err_str}"
        );
    }

    // ── MockStore — lightweight VectorStore for write-failure tests ──────────

    use std::sync::Mutex;

    /// In-memory VectorStore whose `batch_update_embeddings` can be configured
    /// to fail, letting us prove the bug: the old code returned `Ok(0)` (looks
    /// like "queue empty"), the fixed code returns `Err`.
    struct MockStore {
        pending: Vec<crate::HierarchicalChunk>,
        /// When `Some(msg)`, `batch_update_embeddings` returns `Err(Store(msg))`.
        batch_update_error: Option<String>,
        /// Records how many times `batch_update_embeddings` was called.
        batch_update_calls: Mutex<usize>,
    }

    impl MockStore {
        fn with_pending(chunks: Vec<crate::HierarchicalChunk>) -> Self {
            Self {
                pending: chunks,
                batch_update_error: None,
                batch_update_calls: Mutex::new(0),
            }
        }

        fn with_pending_and_write_error(
            chunks: Vec<crate::HierarchicalChunk>,
            msg: impl Into<String>,
        ) -> Self {
            Self {
                pending: chunks,
                batch_update_error: Some(msg.into()),
                batch_update_calls: Mutex::new(0),
            }
        }

        fn batch_update_call_count(&self) -> usize {
            *self.batch_update_calls.lock().unwrap()
        }
    }

    impl crate::VectorStore for MockStore {
        async fn insert_chunks(&self, _chunks: Vec<crate::HierarchicalChunk>) -> crate::Result<()> {
            Ok(())
        }

        async fn search(
            &self,
            _query_embedding: &[f32],
            _limit: usize,
            _level_filter: Option<crate::ChunkLevel>,
            _perspectives: &[&str],
        ) -> crate::Result<Vec<crate::store::SearchResult>> {
            Ok(vec![])
        }

        async fn get_children(
            &self,
            _parent_id: &str,
        ) -> crate::Result<Vec<crate::HierarchicalChunk>> {
            Ok(vec![])
        }

        async fn get_by_id(&self, _id: &str) -> crate::Result<Option<crate::HierarchicalChunk>> {
            Ok(None)
        }

        async fn get_by_id_prefix(
            &self,
            _prefix: &str,
        ) -> crate::Result<Option<crate::HierarchicalChunk>> {
            Ok(None)
        }

        async fn get_by_source(
            &self,
            _source_file: &str,
        ) -> crate::Result<Vec<crate::HierarchicalChunk>> {
            Ok(vec![])
        }

        async fn delete_by_source(&self, _source_file: &str) -> crate::Result<usize> {
            Ok(0)
        }

        async fn stats(&self) -> crate::Result<crate::store::StoreStats> {
            Ok(crate::store::StoreStats::default())
        }

        async fn update_access_profiles(
            &self,
            _updates: Vec<(String, crate::AccessProfile)>,
        ) -> crate::Result<()> {
            Ok(())
        }

        async fn update_visibility(&self, _chunk_id: &str, _visibility: &str) -> crate::Result<()> {
            Ok(())
        }

        async fn add_relation(
            &self,
            _chunk_id: &str,
            _relation: crate::ChunkRelation,
        ) -> crate::Result<()> {
            Ok(())
        }

        async fn get_hot_chunks(
            &self,
            _limit: usize,
        ) -> crate::Result<Vec<crate::HierarchicalChunk>> {
            Ok(vec![])
        }

        async fn get_stale_chunks(
            &self,
            _stale_seconds: i64,
            _limit: usize,
        ) -> crate::Result<Vec<crate::HierarchicalChunk>> {
            Ok(vec![])
        }

        async fn search_text(
            &self,
            _query: &str,
            _perspectives: &[&str],
            _since: Option<i64>,
            _until: Option<i64>,
            _limit: usize,
        ) -> crate::Result<Vec<crate::HierarchicalChunk>> {
            Ok(vec![])
        }

        async fn list_entries(
            &self,
            _perspectives: &[&str],
            _since: Option<i64>,
            _until: Option<i64>,
            _limit: usize,
        ) -> crate::Result<Vec<crate::HierarchicalChunk>> {
            Ok(vec![])
        }

        async fn get_pending_embeddings(
            &self,
            limit: usize,
        ) -> crate::Result<Vec<crate::HierarchicalChunk>> {
            Ok(self.pending.iter().take(limit).cloned().collect())
        }

        async fn batch_update_embeddings(
            &self,
            _updates: Vec<(String, Vec<f32>)>,
        ) -> crate::Result<()> {
            *self.batch_update_calls.lock().unwrap() += 1;
            match &self.batch_update_error {
                Some(msg) => Err(crate::Error::store(msg.clone())),
                None => Ok(()),
            }
        }

        async fn count_pending_embeddings(&self) -> crate::Result<usize> {
            Ok(self.pending.len())
        }
    }

    // ── process_batch: write-failure error path ──────────────────────────────

    /// THE BUG TEST: when `batch_update_embeddings` fails the old code returned
    /// `Ok(0)`, which the worker loop mistook for "queue empty" and slept the
    /// idle interval. The fix must return `Err` so the loop retries on the busy
    /// cadence and the failure is observable.
    #[tokio::test]
    async fn process_batch_returns_err_when_batch_update_fails() {
        let dir = tempfile::tempdir().unwrap();
        let blob_store = Arc::new(BlobStore::open(dir.path()).unwrap());
        let embedder: Arc<dyn crate::Embedder + Send + Sync> = Arc::new(FixedEmbedder);

        let pending = make_pending_chunks(1, "wfail");
        let store = MockStore::with_pending_and_write_error(pending, "simulated write failure");

        let result = process_batch(&store, &embedder, &blob_store).await;

        assert!(
            result.is_err(),
            "write failure must propagate as Err, not Ok(0) — \
             old code masked the error and caused idle-interval retry"
        );
        let err_str = result.unwrap_err().to_string();
        assert!(
            err_str.contains("simulated write failure"),
            "error should carry the original message: {err_str}"
        );
        assert_eq!(
            store.batch_update_call_count(),
            1,
            "batch_update_embeddings should have been called exactly once"
        );
    }

    /// GREEN path: an empty queue still returns `Ok(0)` — the idle-sleep decision
    /// for truly empty queues must not be broken by the error-propagation fix.
    #[tokio::test]
    async fn process_batch_returns_ok_zero_on_empty_queue() {
        let dir = tempfile::tempdir().unwrap();
        let blob_store = Arc::new(BlobStore::open(dir.path()).unwrap());
        let embedder: Arc<dyn crate::Embedder + Send + Sync> = Arc::new(FixedEmbedder);

        let store = MockStore::with_pending(vec![]);

        let result = process_batch(&store, &embedder, &blob_store).await;

        assert_eq!(
            result.unwrap(),
            0,
            "empty queue must still return Ok(0) so the loop takes the idle path"
        );
        assert_eq!(
            store.batch_update_call_count(),
            0,
            "batch_update_embeddings must not be called for an empty queue"
        );
    }

    /// PARTIAL BATCH: a write failure after a partial set of embeddings are
    /// computed must also surface as `Err`, not silently succeed with `Ok(0)`.
    #[tokio::test]
    async fn process_batch_returns_err_on_partial_batch_write_failure() {
        let dir = tempfile::tempdir().unwrap();
        let blob_store = Arc::new(BlobStore::open(dir.path()).unwrap());
        let embedder: Arc<dyn crate::Embedder + Send + Sync> = Arc::new(FixedEmbedder);

        // Use a partial batch (< BATCH_SIZE) to confirm the code path is not
        // gated on batch fullness.
        let pending = make_pending_chunks(3, "pfail");
        let store =
            MockStore::with_pending_and_write_error(pending, "disk full during partial batch");

        let result = process_batch(&store, &embedder, &blob_store).await;

        assert!(
            result.is_err(),
            "partial-batch write failure must propagate as Err"
        );
        assert_eq!(store.batch_update_call_count(), 1);
    }
}
