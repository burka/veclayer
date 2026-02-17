//! Command implementations for VecLayer CLI and library use.
//!
//! This module provides clean, testable command implementations that can be used
//! both from the CLI and programmatically as a library.

use std::path::{Path, PathBuf};

use tracing::info;

use crate::cluster::ClusterPipeline;
use crate::embedder::FastEmbedder;
use crate::parser::MarkdownParser;
use crate::search::{HierarchicalSearch, SearchConfig};
use crate::store::LanceStore;
use crate::summarizer::OllamaSummarizer;
use crate::{Config, DocumentParser, Embedder, Result, VectorStore};

/// Options for document ingestion
#[derive(Debug, Clone)]
pub struct IngestOptions {
    /// Enable recursive directory processing
    pub recursive: bool,
    /// Enable cluster summarization
    pub summarize: bool,
    /// Ollama model to use for summarization
    pub model: String,
    /// Visibility to assign to ingested chunks (default: "normal")
    pub visibility: Option<String>,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            recursive: true,
            summarize: true,
            model: "llama3.2".to_string(),
            visibility: None,
        }
    }
}

/// Options for search queries
#[derive(Debug, Clone)]
pub struct QueryOptions {
    /// Number of top results to return
    pub top_k: usize,
    /// Show full hierarchy path in results
    pub show_path: bool,
    /// Search within a specific subtree (parent chunk ID)
    pub subtree: Option<String>,
    /// Deep search: include all visibilities (deep_only, expired, custom)
    pub deep: bool,
    /// Recency window for relevancy boosting: "24h", "7d", "30d"
    pub recent: Option<String>,
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            top_k: 5,
            show_path: false,
            subtree: None,
            deep: false,
            recent: None,
        }
    }
}

/// Options for the MCP server
#[derive(Debug, Clone)]
pub struct ServeOptions {
    /// Host to bind to
    pub host: String,
    /// Port to listen on
    pub port: u16,
    /// Run in read-only mode
    pub read_only: bool,
    /// Enable MCP stdio transport (for Claude integration)
    pub mcp_stdio: bool,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            read_only: false,
            mcp_stdio: false,
        }
    }
}

/// Result of an ingestion operation
#[derive(Debug)]
pub struct IngestResult {
    /// Total number of chunks processed
    pub total_chunks: usize,
    /// Number of summary chunks created (if summarization was enabled)
    pub summary_chunks: usize,
    /// Number of files processed
    pub files_processed: usize,
}

/// Result of a query operation
#[derive(Debug)]
pub struct QueryResult {
    /// The matched chunk
    pub chunk: crate::HierarchicalChunk,
    /// Similarity score
    pub score: f32,
    /// Full hierarchy path
    pub hierarchy_path: Vec<crate::HierarchicalChunk>,
    /// Relevant children chunks
    pub relevant_children: Vec<QueryResult>,
}

/// Store statistics
#[derive(Debug)]
pub struct StatsResult {
    /// Total number of chunks
    pub total_chunks: usize,
    /// Chunks organized by level
    pub chunks_by_level: std::collections::HashMap<u8, usize>,
    /// List of source files
    pub source_files: Vec<String>,
}

/// Ingest documents into the vector store
///
/// This function processes files from the specified path, generates embeddings,
/// and optionally performs cluster summarization.
pub async fn ingest(data_dir: &Path, path: &Path, options: &IngestOptions) -> Result<IngestResult> {
    info!("Initializing embedder...");
    let embedder = FastEmbedder::new()?;
    let dimension = embedder.dimension();

    info!("Opening vector store at {:?}...", data_dir);
    let store = LanceStore::open(data_dir, dimension).await?;

    let parser = MarkdownParser::new();

    let files = collect_files(path, options.recursive, &parser)?;
    info!("Found {} files to process", files.len());

    let mut all_chunks = Vec::new();

    for file in &files {
        info!("Processing {:?}...", file);

        // Delete existing chunks from this file
        let deleted = store.delete_by_source(&file.to_string_lossy()).await?;
        if deleted > 0 {
            info!("  Removed {} existing chunks", deleted);
        }

        // Parse the file
        let mut chunks = parser.parse_file(file)?;
        info!("  Parsed {} chunks", chunks.len());

        if chunks.is_empty() {
            continue;
        }

        // Apply visibility if specified
        if let Some(ref vis) = options.visibility {
            for chunk in &mut chunks {
                chunk.visibility = vis.clone();
            }
        }

        // Generate embeddings
        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        let embeddings = embedder.embed(&texts)?;

        // Attach embeddings to chunks
        for (chunk, embedding) in chunks.iter_mut().zip(embeddings.into_iter()) {
            chunk.embedding = Some(embedding);
        }

        // Insert into store
        store.insert_chunks(chunks.clone()).await?;
        info!("  Indexed successfully");

        all_chunks.extend(chunks);
    }

    let total_chunks = all_chunks.len();
    let mut summary_chunks = 0;

    // Phase 2: Cluster and summarize if enabled
    if options.summarize && !all_chunks.is_empty() {
        info!(
            "Starting cluster summarization with model '{}'...",
            options.model
        );

        // Create a new embedder for the pipeline (to embed summaries)
        let summary_embedder = FastEmbedder::new()?;
        let summarizer = OllamaSummarizer::new().with_model(&options.model);

        let pipeline = ClusterPipeline::with_summarizer(summary_embedder, summarizer)
            .with_min_cluster_size(2)
            .with_cluster_range(2, 10);

        match pipeline.process(all_chunks).await {
            Ok((updated_chunks, summary_chunk_list)) => {
                // Update existing chunks with cluster memberships
                for chunk in updated_chunks {
                    if !chunk.cluster_memberships.is_empty() {
                        // Re-insert to update cluster memberships
                        // Note: In production, we'd want an upsert operation
                        store.insert_chunks(vec![chunk]).await?;
                    }
                }

                // Insert summary chunks
                if !summary_chunk_list.is_empty() {
                    info!(
                        "Inserting {} cluster summaries...",
                        summary_chunk_list.len()
                    );
                    summary_chunks = summary_chunk_list.len();
                    store.insert_chunks(summary_chunk_list).await?;
                }
            }
            Err(e) => {
                info!(
                    "Cluster summarization failed: {} - continuing without summaries",
                    e
                );
            }
        }
    }

    info!("Ingestion complete!");

    Ok(IngestResult {
        total_chunks,
        summary_chunks,
        files_processed: files.len(),
    })
}

/// Query the vector store with hierarchical search
///
/// Returns matching results with their hierarchy paths and relevant children.
pub async fn query(
    data_dir: &Path,
    query: &str,
    options: &QueryOptions,
) -> Result<Vec<QueryResult>> {
    let embedder = FastEmbedder::new()?;
    let dimension = embedder.dimension();
    let store = LanceStore::open(data_dir, dimension).await?;

    let recency_window = options
        .recent
        .as_deref()
        .and_then(crate::RecencyWindow::from_str_opt);

    let config = SearchConfig {
        top_k: options.top_k,
        children_k: 3,
        max_depth: 3,
        min_score: 0.0,
        deep: options.deep,
        recency_window,
        recency_alpha: SearchConfig::alpha_for_window(recency_window),
    };

    let search = HierarchicalSearch::new(store, embedder).with_config(config);

    let results = if let Some(ref parent_id) = options.subtree {
        search.search_subtree(query, parent_id).await?
    } else {
        search.search(query).await?
    };

    // Convert internal search results to our QueryResult type
    Ok(results
        .into_iter()
        .map(|r| QueryResult {
            chunk: r.chunk,
            score: r.score,
            hierarchy_path: r.hierarchy_path,
            relevant_children: r
                .relevant_children
                .into_iter()
                .map(|c| QueryResult {
                    chunk: c.chunk,
                    score: c.score,
                    hierarchy_path: vec![],
                    relevant_children: vec![],
                })
                .collect(),
        })
        .collect())
}

/// Start the MCP server (HTTP or stdio transport)
pub async fn serve(data_dir: &Path, options: &ServeOptions) -> Result<()> {
    let config = Config::default()
        .with_data_dir(data_dir)
        .with_host(&options.host)
        .with_port(options.port)
        .with_read_only(options.read_only);

    if options.mcp_stdio {
        info!("Starting MCP server on stdio...");
        crate::mcp::run_stdio(config).await
    } else {
        info!(
            "Starting HTTP server on {}:{}...",
            options.host, options.port
        );
        crate::mcp::run_http(config).await
    }
}

/// Show statistics about the vector store
pub async fn stats(data_dir: &Path) -> Result<StatsResult> {
    let embedder = FastEmbedder::new()?;
    let dimension = embedder.dimension();
    let store = LanceStore::open(data_dir, dimension).await?;

    let store_stats = store.stats().await?;

    Ok(StatsResult {
        total_chunks: store_stats.total_chunks,
        chunks_by_level: store_stats.chunks_by_level,
        source_files: store_stats.source_files,
    })
}

/// List all indexed source files
pub async fn sources(data_dir: &Path) -> Result<Vec<String>> {
    let embedder = FastEmbedder::new()?;
    let dimension = embedder.dimension();
    let store = LanceStore::open(data_dir, dimension).await?;

    let store_stats = store.stats().await?;

    Ok(store_stats.source_files)
}

/// Collect files from a path, optionally recursively
///
/// This function respects the parser's can_parse method to filter files.
pub fn collect_files(
    path: &Path,
    recursive: bool,
    parser: &impl DocumentParser,
) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    if path.is_file() {
        if parser.can_parse(path) {
            files.push(path.to_path_buf());
        }
    } else if path.is_dir() {
        if recursive {
            for entry in walkdir::WalkDir::new(path)
                .follow_links(true)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let entry_path = entry.path();
                if entry_path.is_file() && parser.can_parse(entry_path) {
                    files.push(entry_path.to_path_buf());
                }
            }
        } else {
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let entry_path = entry.path();
                if entry_path.is_file() && parser.can_parse(&entry_path) {
                    files.push(entry_path);
                }
            }
        }
    }

    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_ingest_options_default() {
        let opts = IngestOptions::default();
        assert!(opts.recursive);
        assert!(opts.summarize);
        assert_eq!(opts.model, "llama3.2");
    }

    #[test]
    fn test_query_options_default() {
        let opts = QueryOptions::default();
        assert_eq!(opts.top_k, 5);
        assert!(!opts.show_path);
        assert!(opts.subtree.is_none());
    }

    #[test]
    fn test_serve_options_default() {
        let opts = ServeOptions::default();
        assert_eq!(opts.host, "127.0.0.1");
        assert_eq!(opts.port, 8080);
        assert!(!opts.read_only);
        assert!(!opts.mcp_stdio);
    }

    #[test]
    fn test_collect_files_single_file() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("test.md");
        fs::write(&file_path, "# Test")?;

        let parser = MarkdownParser::new();
        let files = collect_files(&file_path, false, &parser)?;

        assert_eq!(files.len(), 1);
        assert_eq!(files[0], file_path);

        Ok(())
    }

    #[test]
    fn test_collect_files_single_non_markdown() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "Test content")?;

        let parser = MarkdownParser::new();
        let files = collect_files(&file_path, false, &parser)?;

        // MarkdownParser should not parse non-.md files
        assert_eq!(files.len(), 0);

        Ok(())
    }

    #[test]
    fn test_collect_files_directory_non_recursive() -> Result<()> {
        let temp_dir = TempDir::new()?;
        fs::write(temp_dir.path().join("file1.md"), "# File 1")?;
        fs::write(temp_dir.path().join("file2.md"), "# File 2")?;
        fs::write(temp_dir.path().join("ignore.txt"), "Text file")?;

        let parser = MarkdownParser::new();
        let files = collect_files(temp_dir.path(), false, &parser)?;

        assert_eq!(files.len(), 2);
        assert!(files.iter().all(|f| f.extension().unwrap() == "md"));

        Ok(())
    }

    #[test]
    fn test_collect_files_directory_recursive() -> Result<()> {
        let temp_dir = TempDir::new()?;
        fs::write(temp_dir.path().join("file1.md"), "# File 1")?;

        let subdir = temp_dir.path().join("subdir");
        fs::create_dir(&subdir)?;
        fs::write(subdir.join("file2.md"), "# File 2")?;

        let parser = MarkdownParser::new();

        // Non-recursive should find only 1 file
        let files_non_recursive = collect_files(temp_dir.path(), false, &parser)?;
        assert_eq!(files_non_recursive.len(), 1);

        // Recursive should find both files
        let files_recursive = collect_files(temp_dir.path(), true, &parser)?;
        assert_eq!(files_recursive.len(), 2);

        Ok(())
    }

    #[test]
    fn test_collect_files_empty_directory() -> Result<()> {
        let temp_dir = TempDir::new()?;

        let parser = MarkdownParser::new();
        let files = collect_files(temp_dir.path(), true, &parser)?;

        assert_eq!(files.len(), 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_stats_empty_store() -> Result<()> {
        let temp_dir = TempDir::new()?;

        let result = stats(temp_dir.path()).await?;

        assert_eq!(result.total_chunks, 0);
        assert_eq!(result.source_files.len(), 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_sources_empty_store() -> Result<()> {
        let temp_dir = TempDir::new()?;

        let result = sources(temp_dir.path()).await?;

        assert!(result.is_empty());

        Ok(())
    }
}
