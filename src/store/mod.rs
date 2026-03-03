//! Vector store trait and backend implementations.
//!
//! [`VectorStore`] defines the async interface for all storage backends.
//! [`StoreBackend`] is the dispatch enum — adding a new backend requires one
//! new file and one new variant. The current production backend is LanceDB.

mod lancedb_impl;
pub(crate) mod lock;

pub(crate) use lancedb_impl::LanceStore;
pub(crate) use lancedb_impl::TABLE_NAME;
pub(crate) use lock::FileLock;

use crate::{AccessProfile, ChunkLevel, ChunkRelation, HierarchicalChunk, Result};
use std::future::Future;
use std::path::Path;

/// Search result from the vector store
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: HierarchicalChunk,
    pub score: f32,
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
    fn search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        level_filter: Option<ChunkLevel>,
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

    /// Search with an optional perspective filter.
    /// Returns only entries that belong to the given perspective.
    fn search_by_perspective(
        &self,
        query_embedding: &[f32],
        limit: usize,
        perspective: &str,
    ) -> impl Future<Output = Result<Vec<SearchResult>>> + Send;

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

    /// List entries without vector search, optionally filtered by perspective and time range.
    fn list_entries(
        &self,
        perspective: Option<&str>,
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

/// Dispatch enum for storage backends.
///
/// Adding a new backend is: one new file, one new variant here, done.
/// Follows the same pattern as `LlmBackend`.
#[non_exhaustive]
pub enum StoreBackend {
    Lance(LanceStore),
}

impl StoreBackend {
    pub async fn open(path: impl AsRef<Path>, dimension: usize, read_only: bool) -> Result<Self> {
        Ok(Self::Lance(
            LanceStore::open(path, dimension, read_only).await?,
        ))
    }

    pub async fn open_metadata(path: impl AsRef<Path>, read_only: bool) -> Result<Self> {
        Ok(Self::Lance(
            LanceStore::open_metadata(path, read_only).await?,
        ))
    }
}

impl VectorStore for StoreBackend {
    fn insert_chunks(
        &self,
        chunks: Vec<HierarchicalChunk>,
    ) -> impl Future<Output = Result<()>> + Send {
        match self {
            Self::Lance(s) => s.insert_chunks(chunks),
        }
    }

    fn search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        level_filter: Option<ChunkLevel>,
    ) -> impl Future<Output = Result<Vec<SearchResult>>> + Send {
        match self {
            Self::Lance(s) => s.search(query_embedding, limit, level_filter),
        }
    }

    fn get_children(
        &self,
        parent_id: &str,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        match self {
            Self::Lance(s) => s.get_children(parent_id),
        }
    }

    fn get_by_id(
        &self,
        id: &str,
    ) -> impl Future<Output = Result<Option<HierarchicalChunk>>> + Send {
        match self {
            Self::Lance(s) => s.get_by_id(id),
        }
    }

    fn get_by_id_prefix(
        &self,
        prefix: &str,
    ) -> impl Future<Output = Result<Option<HierarchicalChunk>>> + Send {
        match self {
            Self::Lance(s) => s.get_by_id_prefix(prefix),
        }
    }

    fn get_by_source(
        &self,
        source_file: &str,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        match self {
            Self::Lance(s) => s.get_by_source(source_file),
        }
    }

    fn delete_by_source(&self, source_file: &str) -> impl Future<Output = Result<usize>> + Send {
        match self {
            Self::Lance(s) => s.delete_by_source(source_file),
        }
    }

    fn stats(&self) -> impl Future<Output = Result<StoreStats>> + Send {
        match self {
            Self::Lance(s) => s.stats(),
        }
    }

    fn update_access_profiles(
        &self,
        updates: Vec<(String, AccessProfile)>,
    ) -> impl Future<Output = Result<()>> + Send {
        match self {
            Self::Lance(s) => s.update_access_profiles(updates),
        }
    }

    fn update_visibility(
        &self,
        chunk_id: &str,
        visibility: &str,
    ) -> impl Future<Output = Result<()>> + Send {
        match self {
            Self::Lance(s) => s.update_visibility(chunk_id, visibility),
        }
    }

    fn add_relation(
        &self,
        chunk_id: &str,
        relation: ChunkRelation,
    ) -> impl Future<Output = Result<()>> + Send {
        match self {
            Self::Lance(s) => s.add_relation(chunk_id, relation),
        }
    }

    fn search_by_perspective(
        &self,
        query_embedding: &[f32],
        limit: usize,
        perspective: &str,
    ) -> impl Future<Output = Result<Vec<SearchResult>>> + Send {
        match self {
            Self::Lance(s) => s.search_by_perspective(query_embedding, limit, perspective),
        }
    }

    fn get_hot_chunks(
        &self,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        match self {
            Self::Lance(s) => s.get_hot_chunks(limit),
        }
    }

    fn get_stale_chunks(
        &self,
        stale_seconds: i64,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        match self {
            Self::Lance(s) => s.get_stale_chunks(stale_seconds, limit),
        }
    }

    fn list_entries(
        &self,
        perspective: Option<&str>,
        since: Option<i64>,
        until: Option<i64>,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        match self {
            Self::Lance(s) => s.list_entries(perspective, since, until, limit),
        }
    }

    fn get_pending_embeddings(
        &self,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        match self {
            Self::Lance(s) => s.get_pending_embeddings(limit),
        }
    }

    fn batch_update_embeddings(
        &self,
        updates: Vec<(String, Vec<f32>)>,
    ) -> impl Future<Output = Result<()>> + Send {
        match self {
            Self::Lance(s) => s.batch_update_embeddings(updates),
        }
    }

    fn count_pending_embeddings(&self) -> impl Future<Output = Result<usize>> + Send {
        match self {
            Self::Lance(s) => s.count_pending_embeddings(),
        }
    }
}

// Implement VectorStore for Arc<T> where T: VectorStore
crate::arc_impl!(VectorStore {
    fn insert_chunks(&self, chunks: Vec<HierarchicalChunk>) -> impl Future<Output = Result<()>> + Send;
    fn search(&self, query_embedding: &[f32], limit: usize, level_filter: Option<ChunkLevel>) -> impl Future<Output = Result<Vec<SearchResult>>> + Send;
    fn get_children(&self, parent_id: &str) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send;
    fn get_by_id(&self, id: &str) -> impl Future<Output = Result<Option<HierarchicalChunk>>> + Send;
    fn get_by_id_prefix(&self, prefix: &str) -> impl Future<Output = Result<Option<HierarchicalChunk>>> + Send;
    fn get_by_source(&self, source_file: &str) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send;
    fn delete_by_source(&self, source_file: &str) -> impl Future<Output = Result<usize>> + Send;
    fn stats(&self) -> impl Future<Output = Result<StoreStats>> + Send;
    fn update_access_profiles(&self, updates: Vec<(String, AccessProfile)>) -> impl Future<Output = Result<()>> + Send;
    fn update_visibility(&self, chunk_id: &str, visibility: &str) -> impl Future<Output = Result<()>> + Send;
    fn add_relation(&self, chunk_id: &str, relation: ChunkRelation) -> impl Future<Output = Result<()>> + Send;
    fn search_by_perspective(&self, query_embedding: &[f32], limit: usize, perspective: &str) -> impl Future<Output = Result<Vec<SearchResult>>> + Send;
    fn get_hot_chunks(&self, limit: usize) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send;
    fn get_stale_chunks(&self, stale_seconds: i64, limit: usize) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send;
    fn list_entries(&self, perspective: Option<&str>, since: Option<i64>, until: Option<i64>, limit: usize) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send;
    fn get_pending_embeddings(&self, limit: usize) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send;
    fn batch_update_embeddings(&self, updates: Vec<(String, Vec<f32>)>) -> impl Future<Output = Result<()>> + Send;
    fn count_pending_embeddings(&self) -> impl Future<Output = Result<usize>> + Send;
});
