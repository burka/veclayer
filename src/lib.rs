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
//! ### Storage backends (at least one required)
//! - `store-lance`: LanceDB backend with ANN vector search (`requires` `lancedb`, `arrow-*`, `futures`)
//! - `store-sqlite`: Lightweight SQLite backend with brute-force cosine similarity (`requires` `rusqlite`)
//!
//! ### Other features
//! - `parser`: Enables `DocumentParser` and Markdown parsing (requires `pulldown-cmark`)
//! - `config`: Enables config file discovery, user config, and `git` module (requires `toml`, `glob`, `directories`, `shellexpand`, `regex`, `serde_yml`, `walkdir`)
//! - `cli`: Enables the CLI binary, implies `store-lance`, `mcp`, `config`, and `parser`
//! - `mcp`: Enables the MCP server module, implies `config`
//! - `embedding-local`: Enables local embedding via `FastEmbed` (`ONNX`)
//! - `llm`: Enables LLM-powered summarization and clustering
//! - `http`: Enables HTTP REST API, Streamable HTTP MCP transport, and OAuth; implies `auth` and `mcp`
//! - `auth`: Enables cryptographic identity (Ed25519), JWT tokens, and keystore
//! - `full`: Enables `cli`, `store-sqlite`, `llm`, `http`, and `auth`
//! - `full-local`: Enables `full` plus `embedding-local`
//!
//! For lightweight consumers, use `store-sqlite` to avoid ~185 transitive crates from `LanceDB`:
//! ```toml
//! veclayer = { version = "0.1", default-features = false, features = ["store-sqlite"] }
//! ```
//!
//! Without `parser`, the `DocumentParser` trait and Markdown parser are unavailable.
//! Without `config`, config file discovery and the `git` module are unavailable; `Config` types are still available.
//! Without `mcp`, the MCP server module is unavailable.
//! Without `http`, only the stdio MCP transport is available — no network listener.
//! Without `auth`, identity/token CLI commands are unavailable.
//! Without `embedding-local`, only the Ollama embedder (requires `llm`) is available.

#![recursion_limit = "256"]

#[cfg(feature = "config")]
use std::path::PathBuf;

pub mod access_profile;
pub mod aging;
pub mod auth;
pub mod blob_store;
pub mod chunk;
#[cfg(feature = "llm")]
#[doc(hidden)]
pub mod cluster;
#[cfg(feature = "cli")]
#[doc(hidden)]
pub mod commands;
pub mod config;
#[cfg(feature = "auth")]
pub mod crypto;
pub mod embedder;
pub mod entry;
pub mod error;
pub mod facade;
#[cfg(feature = "config")]
#[doc(hidden)]
pub mod git;
pub mod identity;
#[cfg(feature = "llm")]
#[doc(hidden)]
pub mod llm;
pub(crate) mod macros;
#[cfg(feature = "mcp")]
#[doc(hidden)]
pub mod mcp;
#[cfg(feature = "llm")]
pub mod ollama_discover;
#[cfg(feature = "llm")]
pub mod openai_compat_discover;
#[cfg(feature = "parser")]
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
#[cfg(feature = "config")]
pub fn default_data_dir() -> PathBuf {
    directories::ProjectDirs::from("", "", "veclayer")
        .map(|dirs| dirs.data_local_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from(".veclayer"))
}

/// Returns the platform cache directory for VecLayer.
///
/// Returns `~/.cache/veclayer` on Linux, `~/Library/Caches/veclayer` on macOS,
/// `AppData\Local\veclayer\cache` on Windows. Falls back to `.veclayer/cache` if
/// platform directories cannot be determined.
#[cfg(feature = "config")]
pub fn default_cache_dir() -> PathBuf {
    directories::ProjectDirs::from("", "", "veclayer")
        .map(|dirs| dirs.cache_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from(".veclayer").join("cache"))
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
#[cfg(feature = "embedding-local")]
pub use embedder::FastEmbedder;
pub use entry::{EmbeddingCache, Entry, StoredBlob};
pub use error::{Error, Result};
pub use facade::{FocusResult, StoreOptions, VecLayer};
#[cfg(feature = "llm")]
pub use llm::{DynLlmProvider, LlmBackend, LlmProvider};
#[cfg(feature = "parser")]
pub use parser::DocumentParser;
pub use salience::SalienceWeights;
pub use search::{HierarchicalSearch, HierarchicalSearchResult, SearchConfig};
pub use store::StoreBackend;
pub use store::VectorStore;
#[cfg(feature = "llm")]
pub use summarizer::{OllamaSummarizer, Summarizer};
