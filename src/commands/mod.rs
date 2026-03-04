//! Command implementations for VecLayer CLI and library use.
//!
//! This module provides clean, testable command implementations that can be used
//! both from the CLI and programmatically as a library.

use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

use owo_colors::{OwoColorize, Stream};
#[cfg(feature = "llm")]
use tracing::info;
use tracing::{debug, warn};

use crate::blob_store::BlobStore;
use crate::chunk::{short_id, EntryType};
#[cfg(feature = "llm")]
use crate::cluster::ClusterPipeline;
use crate::embedder;
use crate::parser::MarkdownParser;
use crate::search::{HierarchicalSearch, SearchConfig};
use crate::store::StoreBackend;
#[cfg(feature = "llm")]
use crate::summarizer::OllamaSummarizer;
use crate::{Config, Embedder, Result, VectorStore};

pub mod add;
pub mod auth;
pub mod data;
pub mod identity;
pub mod merge;
pub mod perspective_ops;
pub mod reflect;
pub mod search;
pub mod serve;
pub mod store_ops;
pub mod think;

// Over-fetch when temporal filters are active, then filter client-side
use crate::search::TEMPORAL_PREFETCH_FACTOR;

// --- Infrastructure helpers ---

/// Create a config + embedder + store + blob store quad.  Centralises the init
/// sequence that was previously repeated in every command that needs embeddings.
pub async fn open_store(
    data_dir: &Path,
) -> Result<(
    Config,
    Box<dyn Embedder + Send + Sync>,
    StoreBackend,
    BlobStore,
)> {
    let config = Config::new().with_data_dir(data_dir);
    let embedder = embedder::from_config(&config.embedder)?;
    let dimension = embedder.dimension();
    let store = StoreBackend::open(data_dir, dimension, false).await?;
    let blob_store = BlobStore::open(data_dir)?;
    Ok((config, embedder, store, blob_store))
}

// --- Output helpers ---

/// Color a visibility string for CLI display.
pub fn vis_color(vis: &str) -> String {
    match vis {
        "always" => vis
            .if_supports_color(Stream::Stdout, |s| s.green())
            .to_string(),
        "normal" => vis.to_string(),
        "deep_only" => vis
            .if_supports_color(Stream::Stdout, |s| s.dimmed())
            .to_string(),
        "expiring" => vis
            .if_supports_color(Stream::Stdout, |s| s.yellow())
            .to_string(),
        _ => vis
            .if_supports_color(Stream::Stdout, |s| s.red())
            .to_string(),
    }
}

/// Truncate content to `max` chars, replacing newlines with spaces.
pub fn preview(s: &str, max: usize) -> String {
    let clean = s.replace('\n', " ");
    if clean.len() <= max {
        clean
    } else {
        let end = clean.floor_char_boundary(max);
        format!("{}...", &clean[..end])
    }
}

// --- Option types ---

/// Options for adding knowledge (files, directories, or inline text)
#[derive(Debug, Clone)]
pub struct AddOptions {
    pub recursive: bool,
    pub follow_links: bool,
    pub summarize: bool,
    pub model: String,
    pub visibility: Option<String>,
    pub entry_type: String,
    pub perspectives: Vec<String>,
    pub parent_id: Option<String>,
    pub heading: Option<String>,
    pub impression_hint: Option<String>,
    pub impression_strength: f32,
    pub rel_supersedes: Vec<String>,
    pub rel_summarizes: Vec<String>,
    pub rel_to: Vec<String>,
    pub rel_derived_from: Vec<String>,
    pub rel_version_of: Vec<String>,
    pub rel_custom: Vec<String>,
}

impl Default for AddOptions {
    fn default() -> Self {
        Self {
            recursive: true,
            follow_links: false,
            summarize: true,
            model: "llama3.2".to_string(),
            visibility: None,
            entry_type: "raw".to_string(),
            perspectives: Vec::new(),
            parent_id: None,
            heading: None,
            impression_hint: None,
            impression_strength: 1.0,
            rel_supersedes: Vec::new(),
            rel_summarizes: Vec::new(),
            rel_to: Vec::new(),
            rel_derived_from: Vec::new(),
            rel_version_of: Vec::new(),
            rel_custom: Vec::new(),
        }
    }
}

/// Options for semantic search
#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub top_k: usize,
    pub show_path: bool,
    pub subtree: Option<String>,
    pub deep: bool,
    pub recent: Option<String>,
    pub perspective: Option<String>,
    pub similar_to: Option<String>,
    pub min_salience: Option<f32>,
    pub min_score: Option<f32>,
    pub since: Option<String>,
    pub until: Option<String>,
    pub ongoing: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            top_k: 5,
            show_path: false,
            subtree: None,
            deep: false,
            recent: None,
            perspective: None,
            similar_to: None,
            min_salience: None,
            min_score: None,
            since: None,
            until: None,
            ongoing: false,
        }
    }
}

/// Options for focus command
#[derive(Debug, Clone)]
pub struct FocusOptions {
    pub question: Option<String>,
    pub limit: usize,
}

impl Default for FocusOptions {
    fn default() -> Self {
        Self {
            question: None,
            limit: 10,
        }
    }
}

/// Options for the MCP server
#[derive(Debug, Clone)]
pub struct ServeOptions {
    pub host: String,
    pub port: u16,
    pub read_only: bool,
    pub mcp_stdio: bool,
    pub project: Option<String>,
    pub branch: Option<String>,
    pub auth_required: bool,
    pub server_url: Option<String>,
    pub auto_approve: bool,
    pub token_expiry_secs: u64,
    pub refresh_expiry_secs: u64,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            read_only: false,
            mcp_stdio: false,
            project: None,
            branch: None,
            auth_required: false,
            server_url: None,
            auto_approve: false,
            token_expiry_secs: 3600,
            refresh_expiry_secs: 2_592_000,
        }
    }
}

/// Options for exporting entries to JSONL
#[derive(Debug, Clone, Default)]
pub struct ExportOptions {
    pub perspective: Option<String>,
}

/// Options for importing entries from JSONL
#[derive(Debug, Clone, Default)]
pub struct ImportOptions {
    pub path: String,
}

/// Options for merging another store's blobs into this one
#[derive(Debug, Clone, Default)]
pub struct MergeOptions {
    pub project: Option<String>,
    pub dry_run: bool,
    pub force: bool,
}

/// Options for the compact command.
#[derive(Debug, Clone)]
pub struct CompactOptions {
    pub limit: usize,
    pub archive_threshold: f32,
}

impl Default for CompactOptions {
    fn default() -> Self {
        Self {
            limit: 20,
            archive_threshold: 0.1,
        }
    }
}

/// Result of an import operation
#[derive(Debug)]
pub struct ImportResult {
    pub imported: usize,
    pub skipped: usize,
}

// --- Result types ---

/// Result of an add/ingest operation
#[derive(Debug)]
pub struct AddResult {
    pub total_entries: usize,
    pub summary_entries: usize,
    pub files_processed: usize,
}

pub type IngestOptions = AddOptions;
pub type IngestResult = AddResult;

/// Result of a search/query operation
#[derive(Debug)]
pub struct SearchResult {
    pub chunk: crate::HierarchicalChunk,
    pub score: f32,
    pub hierarchy_path: Vec<crate::HierarchicalChunk>,
    pub relevant_children: Vec<SearchResult>,
}

/// Store statistics
#[derive(Debug)]
pub struct StatsResult {
    pub total_chunks: usize,
    pub chunks_by_level: std::collections::HashMap<u8, usize>,
    pub source_files: Vec<String>,
}

/// Resolve a potentially short ID to a full entry.
pub async fn resolve_entry(store: &impl VectorStore, id: &str) -> Result<crate::HierarchicalChunk> {
    crate::resolve::resolve_entry(store, id).await
}

// --- Re-exports for external API ---

pub use add::{add, ingest};
pub use auth::{auth_login, auth_status, auth_token, cache_path, CachedToken};
pub use data::{export_entries, import_entries, rebuild_index};
pub use identity::{identity_init, identity_show};
pub use merge::merge;
pub use perspective_ops::{perspective_add, perspective_list, perspective_remove};
pub use reflect::{compact, reflect, CompactAction};
pub use search::{browse, focus, search, search_results};
pub use serve::serve;
pub use store_ops::{
    archive, history, init, orientation, print_sources, show_config, sources, stats, status,
};
#[cfg(feature = "llm")]
pub use think::think;
pub use think::{
    think_aging_apply, think_aging_configure, think_demote, think_discover, think_promote,
    think_relate,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preview_short_content() {
        assert_eq!(preview("hello world", 50), "hello world");
    }

    #[test]
    fn test_preview_truncates() {
        assert_eq!(preview("hello world", 5), "hello...");
    }

    #[test]
    fn test_preview_replaces_newlines() {
        assert_eq!(preview("line1\nline2\nline3", 50), "line1 line2 line3");
    }

    #[test]
    fn test_preview_multibyte_utf8() {
        let result = preview("日本語テスト", 5);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 9);
    }

    #[test]
    fn test_preview_empty() {
        assert_eq!(preview("", 10), "");
    }

    #[test]
    fn test_preview_exact_boundary() {
        assert_eq!(preview("abcde", 5), "abcde");
        assert_eq!(preview("abcdef", 5), "abcde...");
    }
}
