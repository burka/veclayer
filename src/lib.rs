//! # VecLayer
//!
//! Hierarchical vector indexing for documents with perspective-based memory.
//!
//! VecLayer provides semantic search over documents with hierarchical context,
//! a persistent memory layer for AI agents, and automatic knowledge aging —
//! important knowledge stays, unused knowledge fades.
//!
//! ## Core Concepts
//!
//! - **Hierarchical chunks**: Documents are split into chunks organized at
//!   different levels (document, section, paragraph, fragment)
//! - **Perspectives**: Different views of the same knowledge store (e.g.,
//!   decisions, learnings, intentions)
//! - **Salience scoring**: Retrieval blends semantic similarity, recency,
//!   access frequency, and reinforcement
//! - **Aging**: Entries naturally decay unless accessed or reinforced
//!
//! ## Feature Flags
//!
//! - `llm` (default): Enables LLM-powered summarization and clustering
//! - `sync`: Enables cross-store synchronization

#![recursion_limit = "256"]

use std::path::PathBuf;

pub mod access_profile;
pub mod aging;
pub mod auth;
pub mod blob_store;
pub mod chunk;
#[cfg(feature = "llm")]
#[doc(hidden)]
pub mod cluster;
#[doc(hidden)]
pub mod commands;
pub mod config;
pub mod crypto;
pub mod embedder;
pub mod entry;
pub mod error;
#[doc(hidden)]
pub mod git;
pub mod identity;
#[cfg(feature = "llm")]
#[doc(hidden)]
pub mod llm;
pub(crate) mod macros;
#[doc(hidden)]
pub mod mcp;
pub mod parser;
pub mod perspective;
pub mod relations;
pub mod resolve;
pub mod salience;
pub mod search;
pub mod store;
#[cfg(feature = "llm")]
#[doc(hidden)]
pub mod summarizer;
#[cfg(feature = "sync")]
pub mod sync;
#[cfg(test)]
pub(crate) mod test_helpers;
#[cfg(feature = "llm")]
#[doc(hidden)]
pub mod think;
pub mod util;

/// Platform-appropriate default data directory for VecLayer.
///
/// Returns `~/.local/share/veclayer` on Linux, `~/Library/Application Support/veclayer`
/// on macOS, `AppData\Local\veclayer` on Windows. Falls back to `.veclayer` if
/// platform directories cannot be determined.
pub fn default_data_dir() -> PathBuf {
    directories::ProjectDirs::from("", "", "veclayer")
        .map(|dirs| dirs.data_local_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from(".veclayer"))
}

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
pub use salience::SalienceWeights;
pub use search::{HierarchicalSearch, HierarchicalSearchResult, SearchConfig};
pub use store::StoreBackend;
pub use store::VectorStore;
#[cfg(feature = "llm")]
pub use summarizer::{OllamaSummarizer, Summarizer};
#[cfg(feature = "sync")]
pub use sync::{NameResolver, SyncBackend};
