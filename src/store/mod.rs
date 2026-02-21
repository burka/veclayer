mod lancedb_impl;

pub use lancedb_impl::LanceStore;

use crate::{AccessProfile, ChunkLevel, ChunkRelation, HierarchicalChunk, Result};
use std::future::Future;

/// Search result from the vector store
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: HierarchicalChunk,
    pub score: f32,
}

/// Trait for vector storage backends.
/// All operations are async to support both local and remote backends.
pub trait VectorStore: Send + Sync {
    /// Insert chunks into the store. Chunks must have embeddings.
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
}

/// Statistics about the vector store
#[derive(Debug, Clone, Default)]
pub struct StoreStats {
    pub total_chunks: usize,
    pub chunks_by_level: std::collections::HashMap<u8, usize>,
    pub source_files: Vec<String>,
}

/// Box type for dynamic dispatch of stores
pub type BoxedStore = Box<dyn VectorStore>;

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
});
