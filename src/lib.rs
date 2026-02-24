#![recursion_limit = "256"]

pub mod access_profile;
pub mod aging;
pub mod blob_store;
pub mod chunk;
#[cfg(feature = "llm")]
pub mod cluster;
pub mod commands;
pub mod config;
pub mod embedder;
pub mod entry;
pub mod error;
pub mod identity;
#[cfg(feature = "llm")]
pub mod llm;
pub mod macros;
pub mod mcp;
pub mod parser;
pub mod perspective;
pub mod relations;
pub mod resolve;
pub mod salience;
pub mod search;
pub mod store;
#[cfg(feature = "llm")]
pub mod summarizer;
#[cfg(feature = "sync")]
pub mod sync;
#[cfg(test)]
pub mod test_helpers;
#[cfg(feature = "llm")]
pub mod think;

pub use blob_store::BlobStore;
pub use chunk::{
    content_hash, relation, short_id, visibility, AccessProfile, ChunkLevel, ChunkRelation,
    ClusterMembership, EntryType, HierarchicalChunk, RecencyWindow, STANDARD_VISIBLE,
};
#[cfg(feature = "llm")]
pub use cluster::{ClusterPipeline, SoftClusterer};
pub use config::Config;
pub use embedder::Embedder;
pub use entry::{EmbeddingCache, Entry, StoredBlob};
pub use error::{Error, Result};
#[cfg(feature = "llm")]
pub use llm::{LlmBackend, LlmProvider};
pub use parser::DocumentParser;
pub use search::HierarchicalSearch;
pub use store::StoreBackend;
pub use store::VectorStore;
#[cfg(feature = "llm")]
pub use summarizer::{OllamaSummarizer, Summarizer};
#[cfg(feature = "sync")]
pub use sync::{NameResolver, SyncBackend};
