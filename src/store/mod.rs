mod lancedb_impl;

pub use lancedb_impl::LanceStore;

use crate::{ChunkLevel, HierarchicalChunk, Result};
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

    /// Get all chunks from a source file.
    fn get_by_source(
        &self,
        source_file: &str,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send;

    /// Delete all chunks from a source file.
    fn delete_by_source(&self, source_file: &str) -> impl Future<Output = Result<usize>> + Send;

    /// Get statistics about the store.
    fn stats(&self) -> impl Future<Output = Result<StoreStats>> + Send;
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
impl<T: VectorStore> VectorStore for std::sync::Arc<T> {
    fn insert_chunks(
        &self,
        chunks: Vec<HierarchicalChunk>,
    ) -> impl Future<Output = Result<()>> + Send {
        (**self).insert_chunks(chunks)
    }

    fn search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        level_filter: Option<ChunkLevel>,
    ) -> impl Future<Output = Result<Vec<SearchResult>>> + Send {
        (**self).search(query_embedding, limit, level_filter)
    }

    fn get_children(
        &self,
        parent_id: &str,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        (**self).get_children(parent_id)
    }

    fn get_by_id(
        &self,
        id: &str,
    ) -> impl Future<Output = Result<Option<HierarchicalChunk>>> + Send {
        (**self).get_by_id(id)
    }

    fn get_by_source(
        &self,
        source_file: &str,
    ) -> impl Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        (**self).get_by_source(source_file)
    }

    fn delete_by_source(&self, source_file: &str) -> impl Future<Output = Result<usize>> + Send {
        (**self).delete_by_source(source_file)
    }

    fn stats(&self) -> impl Future<Output = Result<StoreStats>> + Send {
        (**self).stats()
    }
}
