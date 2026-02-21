pub mod access_profile;
pub mod aging;
pub mod chunk;
#[cfg(feature = "llm")]
pub mod cluster;
pub mod commands;
pub mod config;
pub mod embedder;
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
#[cfg(test)]
pub mod test_helpers;
#[cfg(feature = "llm")]
pub mod think;

pub use chunk::{
    content_hash, relation, short_id, visibility, AccessProfile, ChunkLevel, ChunkRelation,
    ClusterMembership, EntryType, HierarchicalChunk, RecencyWindow, STANDARD_VISIBLE,
};
#[cfg(feature = "llm")]
pub use cluster::{ClusterPipeline, SoftClusterer};
pub use config::Config;
pub use embedder::Embedder;
pub use error::{Error, Result};
#[cfg(feature = "llm")]
pub use llm::{LlmBackend, LlmProvider};
pub use parser::DocumentParser;
pub use search::HierarchicalSearch;
pub use store::VectorStore;
#[cfg(feature = "llm")]
pub use summarizer::{OllamaSummarizer, Summarizer};
