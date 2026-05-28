//! Vector store trait and backend implementations.
//!
//! [`VectorStore`] defines the async interface for all storage backends.
//! [`StoreBackend`] is the dispatch enum — adding a new backend requires one
//! new file and one new variant. The current production backend is LanceDB.

#[cfg(feature = "store-lance")]
mod lancedb_impl;
pub(crate) mod lock;
#[cfg(feature = "store-sqlite")]
mod sqlite_impl;

#[cfg(feature = "store-lance")]
pub(crate) use lancedb_impl::LanceStore;
#[cfg(feature = "cli")]
pub(crate) use lancedb_impl::TABLE_NAME;
pub(crate) use lock::FileLock;
#[cfg(feature = "store-sqlite")]
pub(crate) use sqlite_impl::SqliteStore;

use crate::{AccessProfile, ChunkLevel, ChunkRelation, HierarchicalChunk, Result};
use std::future::Future;
use std::path::Path;

/// Embedding status values shared across backends.
pub(crate) const EMBEDDING_STATUS_EMBEDDED: &str = "embedded";
pub(crate) const EMBEDDING_STATUS_PENDING: &str = "pending";

/// Search result from the vector store
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: HierarchicalChunk,
    pub score: f32,
}

/// Outcome of a compact + prune pass. Zeroed when nothing was eligible.
#[derive(Debug, Default, Clone, Copy)]
pub struct CompactStats {
    /// Old version manifests removed.
    pub versions_removed: u64,
    /// Bytes freed by version pruning (manifest + orphaned data files).
    pub bytes_reclaimed: u64,
    /// Fragments rewritten (deletions materialized, small files merged).
    pub fragments_removed: u64,
    /// Fragments produced by the rewrite.
    pub fragments_added: u64,
    /// Data + deletion files removed by compaction.
    pub files_removed: u64,
    /// Data files added by compaction.
    pub files_added: u64,
}

/// Trait for vector storage backends.
/// All operations are async to support both local and remote backends.
pub trait VectorStore: Send + Sync {
    /// Insert chunks into the store. Chunks without embeddings are stored as pending.
    fn insert_chunks(
        &self,
        chunks: Vec<HierarchicalChunk>,
    ) -> impl Future<Output = Result<()>> + Send;

    /// Search for similar chunks using a query embedding.
    /// Optionally filter by perspectives (entries tagged with any of the given perspectives).
    /// Empty slice = no filter.
    fn search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        level_filter: Option<ChunkLevel>,
        perspectives: &[&str],
    ) -> impl Future<Output = Result<Vec<SearchResult>>> + Send;

    /// Get all children of a chunk by parent ID.
    fn get_children(
        &self,
        parent_id: &str,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send;

    /// Get a chunk by its ID.
    fn get_by_id(&self, id: &str)
        -> impl Future<Output = Result<Option<HierarchicalChunk>>> + Send;

    /// Get a chunk by ID prefix (short ID). Tries exact match first,
    /// then falls back to prefix scan. Returns error if prefix is ambiguous.
    fn get_by_id_prefix(
        &self,
        prefix: &str,
    ) -> impl Future<Output = Result<Option<HierarchicalChunk>>> + Send;

    /// Get all chunks from a source file.
    fn get_by_source(
        &self,
        source_file: &str,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send;

    /// Delete all chunks from a source file.
    fn delete_by_source(&self, source_file: &str) -> impl Future<Output = Result<usize>> + Send;

    /// Get statistics about the store.
    fn stats(&self) -> impl Future<Output = Result<StoreStats>> + Send;

    /// Update access profiles for multiple chunks (used after search).
    fn update_access_profiles(
        &self,
        updates: Vec<(String, AccessProfile)>,
    ) -> impl Future<Output = Result<()>> + Send;

    /// Update the visibility of a chunk (for promote/demote).
    ///
    /// Returns `Err` if no chunk with `chunk_id` exists (a zero-row update),
    /// consistent across all backends and with [`Self::add_relation`].
    fn update_visibility(
        &self,
        chunk_id: &str,
        visibility: &str,
    ) -> impl Future<Output = Result<()>> + Send;

    /// Add a relation to a chunk.
    fn add_relation(
        &self,
        chunk_id: &str,
        relation: ChunkRelation,
    ) -> impl Future<Output = Result<()>> + Send;

    /// Get chunks with highest access totals (most popular).
    fn get_hot_chunks(
        &self,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send;

    /// Get chunks that haven't been accessed within the given number of seconds.
    /// Only returns chunks with visibility "normal" or "always" (candidates for degradation).
    fn get_stale_chunks(
        &self,
        stale_seconds: i64,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send;

    /// Search entries by keyword (SQL LIKE on content column).
    /// Fallback when vector embeddings are unavailable.
    /// Results ordered by recency (most recent first).
    fn search_text(
        &self,
        query: &str,
        perspectives: &[&str],
        since: Option<i64>,
        until: Option<i64>,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send;

    /// List entries without vector search, optionally filtered by perspectives and time range.
    /// Empty slice = no filter.
    fn list_entries(
        &self,
        perspectives: &[&str],
        since: Option<i64>,
        until: Option<i64>,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send;

    /// Get chunks whose embeddings are still pending (zero-vector placeholders).
    fn get_pending_embeddings(
        &self,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send;

    /// Replace zero-vector placeholders with real embeddings and set status to "embedded".
    /// Each tuple is (chunk_id, embedding). Performed as a single batch delete+reinsert.
    fn batch_update_embeddings(
        &self,
        updates: Vec<(String, Vec<f32>)>,
    ) -> impl Future<Output = Result<()>> + Send;

    /// Count how many chunks are still awaiting embeddings.
    fn count_pending_embeddings(&self) -> impl Future<Output = Result<usize>> + Send;
}

/// Statistics about the vector store
#[derive(Debug, Clone, Default)]
pub struct StoreStats {
    pub total_chunks: usize,
    pub chunks_by_level: std::collections::HashMap<u8, usize>,
    pub source_files: Vec<String>,
    pub pending_embeddings: usize,
}

// Ensure at least one storage backend is enabled at compile time.
#[cfg(not(any(feature = "store-lance", feature = "store-sqlite")))]
compile_error!(
    "At least one storage backend must be enabled. \
     Use `store-lance`, `store-sqlite`, or both."
);

/// Dispatch enum for storage backends.
///
/// Adding a new backend is: one new file, one new variant here, done.
/// Follows the same pattern as `LlmBackend`.
#[non_exhaustive]
#[allow(private_interfaces)]
pub enum StoreBackend {
    #[cfg(feature = "store-lance")]
    Lance(LanceStore),
    #[cfg(feature = "store-sqlite")]
    Sqlite(SqliteStore),
}

impl StoreBackend {
    /// Open the default backend.
    ///
    /// When both backends are compiled, prefers LanceDB.
    /// When only one is compiled, uses that one.
    pub async fn open(path: impl AsRef<Path>, dimension: usize, read_only: bool) -> Result<Self> {
        #[cfg(feature = "store-lance")]
        {
            Ok(Self::Lance(
                LanceStore::open(path, dimension, read_only).await?,
            ))
        }
        #[cfg(not(feature = "store-lance"))]
        {
            Self::open_sqlite(path, dimension, read_only).await
        }
    }

    #[cfg(feature = "store-lance")]
    pub async fn open_metadata(path: impl AsRef<Path>, read_only: bool) -> Result<Self> {
        Ok(Self::Lance(
            LanceStore::open_metadata(path, read_only).await?,
        ))
    }

    #[cfg(feature = "store-sqlite")]
    pub async fn open_sqlite(
        path: impl AsRef<Path>,
        dimension: usize,
        read_only: bool,
    ) -> Result<Self> {
        Ok(Self::Sqlite(
            SqliteStore::open(path, dimension, read_only).await?,
        ))
    }

    /// Run auto-compaction if version count exceeds the threshold.
    /// No-op for non-Lance backends. Returns the actual stats so callers can
    /// log what was reclaimed.
    #[cfg(feature = "store-lance")]
    pub async fn auto_compact_if_needed(&self) -> Result<CompactStats> {
        match self {
            Self::Lance(s) => s.auto_compact_if_needed().await,
            #[cfg(feature = "store-sqlite")]
            Self::Sqlite(_) => Ok(CompactStats::default()),
        }
    }

    /// Force a compact + prune pass regardless of thresholds. Returns stats.
    /// No-op for non-Lance backends.
    #[cfg(feature = "store-lance")]
    pub async fn force_compact(&self) -> Result<CompactStats> {
        match self {
            Self::Lance(s) => s.force_compact().await,
            #[cfg(feature = "store-sqlite")]
            Self::Sqlite(_) => Ok(CompactStats::default()),
        }
    }
}

/// Dispatch a method call to the active backend variant.
///
/// Uses an async block so all arms resolve to the same opaque future type,
/// which is required when more than one backend is compiled.
macro_rules! dispatch {
    ($self:expr, $method:ident ( $($arg:expr),* $(,)? )) => {
        async move {
            match $self {
                #[cfg(feature = "store-lance")]
                StoreBackend::Lance(s) => s.$method($($arg),*).await,
                #[cfg(feature = "store-sqlite")]
                StoreBackend::Sqlite(s) => s.$method($($arg),*).await,
            }
        }
    };
}

#[allow(clippy::manual_async_fn)]
impl VectorStore for StoreBackend {
    fn insert_chunks(
        &self,
        chunks: Vec<HierarchicalChunk>,
    ) -> impl Future<Output = Result<()>> + Send {
        dispatch!(self, insert_chunks(chunks))
    }

    fn search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        level_filter: Option<ChunkLevel>,
        perspectives: &[&str],
    ) -> impl Future<Output = Result<Vec<SearchResult>>> + Send {
        dispatch!(
            self,
            search(query_embedding, limit, level_filter, perspectives)
        )
    }

    fn get_children(
        &self,
        parent_id: &str,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        dispatch!(self, get_children(parent_id))
    }

    fn get_by_id(
        &self,
        id: &str,
    ) -> impl Future<Output = Result<Option<HierarchicalChunk>>> + Send {
        dispatch!(self, get_by_id(id))
    }

    fn get_by_id_prefix(
        &self,
        prefix: &str,
    ) -> impl Future<Output = Result<Option<HierarchicalChunk>>> + Send {
        dispatch!(self, get_by_id_prefix(prefix))
    }

    fn get_by_source(
        &self,
        source_file: &str,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        dispatch!(self, get_by_source(source_file))
    }

    fn delete_by_source(&self, source_file: &str) -> impl Future<Output = Result<usize>> + Send {
        dispatch!(self, delete_by_source(source_file))
    }

    fn stats(&self) -> impl Future<Output = Result<StoreStats>> + Send {
        dispatch!(self, stats())
    }

    fn update_access_profiles(
        &self,
        updates: Vec<(String, AccessProfile)>,
    ) -> impl Future<Output = Result<()>> + Send {
        dispatch!(self, update_access_profiles(updates))
    }

    fn update_visibility(
        &self,
        chunk_id: &str,
        visibility: &str,
    ) -> impl Future<Output = Result<()>> + Send {
        dispatch!(self, update_visibility(chunk_id, visibility))
    }

    fn add_relation(
        &self,
        chunk_id: &str,
        relation: ChunkRelation,
    ) -> impl Future<Output = Result<()>> + Send {
        dispatch!(self, add_relation(chunk_id, relation))
    }

    fn get_hot_chunks(
        &self,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        dispatch!(self, get_hot_chunks(limit))
    }

    fn get_stale_chunks(
        &self,
        stale_seconds: i64,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        dispatch!(self, get_stale_chunks(stale_seconds, limit))
    }

    fn search_text(
        &self,
        query: &str,
        perspectives: &[&str],
        since: Option<i64>,
        until: Option<i64>,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        dispatch!(self, search_text(query, perspectives, since, until, limit))
    }

    fn list_entries(
        &self,
        perspectives: &[&str],
        since: Option<i64>,
        until: Option<i64>,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        dispatch!(self, list_entries(perspectives, since, until, limit))
    }

    fn get_pending_embeddings(
        &self,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        dispatch!(self, get_pending_embeddings(limit))
    }

    fn batch_update_embeddings(
        &self,
        updates: Vec<(String, Vec<f32>)>,
    ) -> impl Future<Output = Result<()>> + Send {
        dispatch!(self, batch_update_embeddings(updates))
    }

    fn count_pending_embeddings(&self) -> impl Future<Output = Result<usize>> + Send {
        dispatch!(self, count_pending_embeddings())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// When both backends are compiled (the default test configuration),
    /// `StoreBackend::open` must prefer LanceDB. This locks in the documented
    /// dispatch policy at lines 209-210.
    #[cfg(all(feature = "store-lance", feature = "store-sqlite"))]
    #[tokio::test]
    async fn open_prefers_lance_when_both_features_compiled() {
        let dir = TempDir::new().unwrap();
        let backend = StoreBackend::open(dir.path(), 384, false).await.unwrap();
        assert!(
            matches!(backend, StoreBackend::Lance(_)),
            "open() must pick Lance when both backends are compiled"
        );
    }

    /// `open_sqlite` is the explicit escape hatch when callers want SQLite even
    /// though Lance is compiled. Verify it actually produces the Sqlite variant.
    #[cfg(feature = "store-sqlite")]
    #[tokio::test]
    async fn open_sqlite_returns_sqlite_variant() {
        let dir = TempDir::new().unwrap();
        let backend = StoreBackend::open_sqlite(dir.path(), 384, false)
            .await
            .unwrap();
        assert!(
            matches!(backend, StoreBackend::Sqlite(_)),
            "open_sqlite must produce the Sqlite variant"
        );
    }

    /// `auto_compact_if_needed` is a Lance-only optimisation. When both
    /// backends are compiled, callers can call it on either variant; on the
    /// SQLite arm it must succeed as a no-op (zeroed `CompactStats`).
    #[cfg(all(feature = "store-lance", feature = "store-sqlite"))]
    #[tokio::test]
    async fn auto_compact_if_needed_is_noop_on_sqlite() {
        let dir = TempDir::new().unwrap();
        let backend = StoreBackend::open_sqlite(dir.path(), 384, false)
            .await
            .unwrap();
        let stats = backend.auto_compact_if_needed().await.unwrap();
        assert_eq!(stats.versions_removed, 0);
        assert_eq!(stats.bytes_reclaimed, 0);
        assert_eq!(stats.fragments_removed, 0);
        assert_eq!(stats.fragments_added, 0);
        assert_eq!(stats.files_removed, 0);
        assert_eq!(stats.files_added, 0);
    }

    /// `force_compact` follows the same no-op-on-SQLite contract when both
    /// backends are compiled.
    #[cfg(all(feature = "store-lance", feature = "store-sqlite"))]
    #[tokio::test]
    async fn force_compact_is_noop_on_sqlite() {
        let dir = TempDir::new().unwrap();
        let backend = StoreBackend::open_sqlite(dir.path(), 384, false)
            .await
            .unwrap();
        let stats = backend.force_compact().await.unwrap();
        assert_eq!(stats.versions_removed, 0);
        assert_eq!(stats.bytes_reclaimed, 0);
        assert_eq!(stats.fragments_removed, 0);
        assert_eq!(stats.fragments_added, 0);
        assert_eq!(stats.files_removed, 0);
        assert_eq!(stats.files_added, 0);
    }

    /// `auto_compact_if_needed` on a fresh Lance store should succeed and
    /// return zero stats (no versions to prune, no fragments to compact yet).
    /// Asserts all six fields so a regression in any one of them surfaces.
    #[cfg(feature = "store-lance")]
    #[tokio::test]
    async fn auto_compact_if_needed_on_fresh_lance_returns_zero_stats() {
        let dir = TempDir::new().unwrap();
        let backend = StoreBackend::open(dir.path(), 384, false).await.unwrap();
        let stats = backend.auto_compact_if_needed().await.unwrap();
        assert_eq!(stats.versions_removed, 0);
        assert_eq!(stats.bytes_reclaimed, 0);
        assert_eq!(stats.fragments_removed, 0);
        assert_eq!(stats.fragments_added, 0);
        assert_eq!(stats.files_removed, 0);
        assert_eq!(stats.files_added, 0);
    }

    /// `open_metadata` opens an existing Lance table for metadata-only reads
    /// (e.g. listing scopes during sync). The pre-`open` + `drop` is
    /// load-bearing: `open_metadata` does not create the table, so the table
    /// must already exist on disk. Removing the pre-open would make this test
    /// fail for the wrong reason.
    #[cfg(feature = "store-lance")]
    #[tokio::test]
    async fn open_metadata_returns_lance_variant() {
        let dir = TempDir::new().unwrap();
        let _full = StoreBackend::open(dir.path(), 384, false).await.unwrap();
        drop(_full);
        let backend = StoreBackend::open_metadata(dir.path(), true).await.unwrap();
        assert!(matches!(backend, StoreBackend::Lance(_)));
    }
}

// Implement VectorStore for Arc<T> where T: VectorStore
crate::arc_impl!(VectorStore {
    fn insert_chunks(&self, chunks: Vec<HierarchicalChunk>) -> impl Future<Output = Result<()>> + Send;
    fn search(&self, query_embedding: &[f32], limit: usize, level_filter: Option<ChunkLevel>, perspectives: &[&str]) -> impl Future<Output = Result<Vec<SearchResult>>> + Send;
    fn get_children(&self, parent_id: &str) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send;
    fn get_by_id(&self, id: &str) -> impl Future<Output = Result<Option<HierarchicalChunk>>> + Send;
    fn get_by_id_prefix(&self, prefix: &str) -> impl Future<Output = Result<Option<HierarchicalChunk>>> + Send;
    fn get_by_source(&self, source_file: &str) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send;
    fn delete_by_source(&self, source_file: &str) -> impl Future<Output = Result<usize>> + Send;
    fn stats(&self) -> impl Future<Output = Result<StoreStats>> + Send;
    fn update_access_profiles(&self, updates: Vec<(String, AccessProfile)>) -> impl Future<Output = Result<()>> + Send;
    fn update_visibility(&self, chunk_id: &str, visibility: &str) -> impl Future<Output = Result<()>> + Send;
    fn add_relation(&self, chunk_id: &str, relation: ChunkRelation) -> impl Future<Output = Result<()>> + Send;
    fn get_hot_chunks(&self, limit: usize) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send;
    fn get_stale_chunks(&self, stale_seconds: i64, limit: usize) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send;
    fn search_text(&self, query: &str, perspectives: &[&str], since: Option<i64>, until: Option<i64>, limit: usize) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send;
    fn list_entries(&self, perspectives: &[&str], since: Option<i64>, until: Option<i64>, limit: usize) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send;
    fn get_pending_embeddings(&self, limit: usize) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send;
    fn batch_update_embeddings(&self, updates: Vec<(String, Vec<f32>)>) -> impl Future<Output = Result<()>> + Send;
    fn count_pending_embeddings(&self) -> impl Future<Output = Result<usize>> + Send;
});
