//! Command implementations for VecLayer CLI and library use.
//!
//! This module provides clean, testable command implementations that can be used
//! both from the CLI and programmatically as a library.

use std::path::{Path, PathBuf};

use tracing::{debug, info};

use crate::chunk::{short_id, EntryType};
#[cfg(feature = "llm")]
use crate::cluster::ClusterPipeline;
use crate::embedder::FastEmbedder;
use crate::parser::MarkdownParser;
use crate::search::{HierarchicalSearch, SearchConfig};
use crate::store::LanceStore;
#[cfg(feature = "llm")]
use crate::summarizer::OllamaSummarizer;
use crate::{Config, DocumentParser, Embedder, Result, VectorStore};

// Over-fetch when temporal filters are active, then filter client-side
const TEMPORAL_PREFETCH_FACTOR: usize = 3;

// --- Infrastructure helpers ---

/// Create an embedder + store pair.  Centralises the 3-line init sequence
/// that was previously repeated in every command that needs embeddings.
async fn open_store(data_dir: &Path) -> Result<(FastEmbedder, LanceStore)> {
    let embedder = FastEmbedder::new()?;
    let dimension = embedder.dimension();
    let store = LanceStore::open(data_dir, dimension).await?;
    Ok((embedder, store))
}

// --- Output helpers ---

/// Truncate content to `max` chars, replacing newlines with spaces.
fn preview(s: &str, max: usize) -> String {
    let clean = s.replace('\n', " ");
    if clean.len() <= max {
        clean
    } else {
        format!("{}...", &clean[..max])
    }
}

// --- Option types ---

/// Options for adding knowledge (files, directories, or inline text)
#[derive(Debug, Clone)]
pub struct AddOptions {
    /// Enable recursive directory processing
    pub recursive: bool,
    /// Enable cluster summarization
    pub summarize: bool,
    /// Ollama model to use for summarization
    pub model: String,
    /// Visibility to assign to entries (default: "normal")
    pub visibility: Option<String>,
    /// Entry type: "raw", "meta", "impression"
    pub entry_type: String,
    /// Perspectives to tag entries with (e.g. "decisions", "learnings")
    pub perspectives: Vec<String>,
    /// Relation: this entry summarizes the given ID
    pub summarizes: Option<String>,
    /// Relation: this entry supersedes the given ID
    pub supersedes: Option<String>,
    /// Relation: this is a version of the given ID
    pub version_of: Option<String>,
    /// Parent entry ID for hierarchy placement
    pub parent_id: Option<String>,
    /// Heading/title for the entry
    pub heading: Option<String>,
    /// Relation: this entry is related to the given ID (bidirectional)
    pub related_to: Option<String>,
    /// Relation: this entry is derived from the given ID
    pub derived_from: Option<String>,
}

/// Backwards-compatible alias
pub type IngestOptions = AddOptions;

impl Default for AddOptions {
    fn default() -> Self {
        Self {
            recursive: true,
            summarize: true,
            model: "llama3.2".to_string(),
            visibility: None,
            entry_type: "raw".to_string(),
            perspectives: Vec::new(),
            summarizes: None,
            supersedes: None,
            version_of: None,
            parent_id: None,
            heading: None,
            related_to: None,
            derived_from: None,
        }
    }
}

/// Options for semantic search
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Number of top results to return
    pub top_k: usize,
    /// Show full hierarchy path in results
    pub show_path: bool,
    /// Search within a specific subtree (entry ID)
    pub subtree: Option<String>,
    /// Deep search: include all visibilities
    pub deep: bool,
    /// Recency window: "24h", "7d", "30d"
    pub recent: Option<String>,
    /// Filter by perspective (e.g. "decisions", "learnings")
    pub perspective: Option<String>,
    /// Minimum salience (entries below this excluded from salience boosting)
    pub min_salience: Option<f32>,
    /// Minimum search score (entries below this are filtered out)
    pub min_score: Option<f32>,
    /// Only entries created after this ISO 8601 date or epoch seconds
    pub since: Option<String>,
    /// Only entries created before this ISO 8601 date or epoch seconds
    pub until: Option<String>,
}

/// Backwards-compatible alias
pub type QueryOptions = SearchOptions;

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            top_k: 5,
            show_path: false,
            subtree: None,
            deep: false,
            recent: None,
            perspective: None,
            min_salience: None,
            min_score: None,
            since: None,
            until: None,
        }
    }
}

/// Options for focus command
#[derive(Debug, Clone)]
pub struct FocusOptions {
    /// Optional question to rerank children by relevance
    pub question: Option<String>,
    /// Max children to display
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

// --- Result types ---

/// Result of an add/ingest operation
#[derive(Debug)]
pub struct AddResult {
    /// Total number of entries processed
    pub total_entries: usize,
    /// Number of summary entries created
    pub summary_entries: usize,
    /// Number of files processed
    pub files_processed: usize,
}

/// Backwards-compatible alias
pub type IngestResult = AddResult;

/// Result of a search/query operation
#[derive(Debug)]
pub struct SearchResult {
    /// The matched entry
    pub chunk: crate::HierarchicalChunk,
    /// Similarity score
    pub score: f32,
    /// Full hierarchy path
    pub hierarchy_path: Vec<crate::HierarchicalChunk>,
    /// Relevant children
    pub relevant_children: Vec<SearchResult>,
}

/// Backwards-compatible alias
pub type QueryResult = SearchResult;

/// Store statistics
#[derive(Debug)]
pub struct StatsResult {
    /// Total number of entries
    pub total_chunks: usize,
    /// Entries organized by level
    pub chunks_by_level: std::collections::HashMap<u8, usize>,
    /// List of source files
    pub source_files: Vec<String>,
}

// --- Command implementations ---

/// Initialize a new VecLayer store in the given directory.
pub fn init(data_dir: &Path) -> Result<()> {
    if data_dir.exists() {
        println!("VecLayer store already exists at {}", data_dir.display());
        println!("  use `veclayer add` to add knowledge");
    } else {
        std::fs::create_dir_all(data_dir)?;
        println!("Initialized VecLayer store at {}", data_dir.display());
    }
    // Initialize perspectives (idempotent — won't overwrite existing)
    crate::perspective::init(data_dir)?;
    println!("\nNext steps:");
    println!("  veclayer add ./docs       # Add files");
    println!("  veclayer add \"text\"        # Add inline text");
    println!("  veclayer search \"query\"    # Search");
    Ok(())
}

/// Add knowledge to the store (files, directories, or inline text).
///
/// If `input` is a path to a file or directory, it parses and ingests documents.
/// Otherwise, it treats the input as inline text content.
pub async fn add(data_dir: &Path, input: &str, options: &AddOptions) -> Result<AddResult> {
    // Validate perspectives early (fail fast at CLI boundary)
    if !options.perspectives.is_empty() {
        crate::perspective::validate_ids(data_dir, &options.perspectives)?;
    }

    let input_path = Path::new(input);

    if input_path.exists() {
        // Input is a file or directory -- ingest it
        add_files(data_dir, input_path, options).await
    } else {
        // Input is inline text -- store as a single entry
        add_text(data_dir, input, options).await
    }
}

/// Backwards-compatible alias
pub async fn ingest(data_dir: &Path, path: &Path, options: &AddOptions) -> Result<AddResult> {
    add_files(data_dir, path, options).await
}

/// Add files from a path to the store.
async fn add_files(data_dir: &Path, path: &Path, options: &AddOptions) -> Result<AddResult> {
    debug!("Opening store at {:?}...", data_dir);
    let (embedder, store) = open_store(data_dir).await?;

    let parser = MarkdownParser::new();

    let files = collect_files(path, options.recursive, &parser)?;
    debug!("Found {} files to process", files.len());

    let mut all_chunks = Vec::new();

    for file in &files {
        debug!("Processing {:?}...", file);

        let deleted = store.delete_by_source(&file.to_string_lossy()).await?;
        if deleted > 0 {
            debug!("  Removed {} existing entries", deleted);
        }

        let mut chunks = parser.parse_file(file)?;
        debug!("  Parsed {} entries", chunks.len());

        if chunks.is_empty() {
            continue;
        }

        if let Some(ref vis) = options.visibility {
            for chunk in &mut chunks {
                chunk.visibility = vis.clone();
            }
        }

        if !options.perspectives.is_empty() {
            for chunk in &mut chunks {
                chunk.perspectives = options.perspectives.clone();
            }
        }

        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        let embeddings = embedder.embed(&texts)?;

        for (chunk, embedding) in chunks.iter_mut().zip(embeddings.into_iter()) {
            chunk.embedding = Some(embedding);
        }

        store.insert_chunks(chunks.clone()).await?;
        debug!("  Indexed successfully");

        all_chunks.extend(chunks);
    }

    let total_entries = all_chunks.len();
    let mut summary_entries = 0;

    #[cfg(feature = "llm")]
    if options.summarize && !all_chunks.is_empty() {
        info!(
            "Starting cluster summarization with model '{}'...",
            options.model
        );

        let summary_embedder = FastEmbedder::new()?;
        let summarizer = OllamaSummarizer::new().with_model(&options.model);

        let pipeline = ClusterPipeline::with_summarizer(summary_embedder, summarizer)
            .with_min_cluster_size(2)
            .with_cluster_range(2, 10);

        match pipeline.process(all_chunks).await {
            Ok((updated_chunks, summary_chunk_list)) => {
                for chunk in updated_chunks {
                    if !chunk.cluster_memberships.is_empty() {
                        store.insert_chunks(vec![chunk]).await?;
                    }
                }

                if !summary_chunk_list.is_empty() {
                    info!(
                        "Inserting {} cluster summaries...",
                        summary_chunk_list.len()
                    );
                    summary_entries = summary_chunk_list.len();
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

    println!(
        "Added {} entries ({} summaries) from {} files",
        total_entries,
        summary_entries,
        files.len()
    );

    Ok(AddResult {
        total_entries,
        summary_entries,
        files_processed: files.len(),
    })
}

/// Add inline text as a single entry.
async fn add_text(data_dir: &Path, text: &str, options: &AddOptions) -> Result<AddResult> {
    let (embedder, store) = open_store(data_dir).await?;

    let entry_type = match options.entry_type.as_str() {
        "meta" => crate::chunk::EntryType::Meta,
        "impression" => crate::chunk::EntryType::Impression,
        "summary" => crate::chunk::EntryType::Summary,
        _ => crate::chunk::EntryType::Raw,
    };

    // Resolve parent ID and compute level/path from parent
    let (level, path, resolved_parent_id) = if let Some(ref pid) = options.parent_id {
        let parent = resolve_entry(&store, pid).await?;
        (
            crate::chunk::ChunkLevel(parent.level.0 + 1),
            format!("{}/agent", parent.path),
            Some(parent.id),
        )
    } else {
        (crate::ChunkLevel::CONTENT, String::new(), None)
    };

    let mut chunk = crate::HierarchicalChunk::new(
        text.to_string(),
        level,
        resolved_parent_id,
        path,
        "[inline]".to_string(),
    )
    .with_entry_type(entry_type)
    .with_perspectives(options.perspectives.clone());

    // Set heading if provided
    if let Some(ref heading) = options.heading {
        chunk.heading = Some(heading.clone());
    }

    if let Some(ref vis) = options.visibility {
        chunk.visibility = vis.clone();
    }

    // Add relations from options
    if let Some(ref target) = options.summarizes {
        chunk
            .relations
            .push(crate::ChunkRelation::summarized_by(target));
    }
    if let Some(ref target) = options.supersedes {
        chunk
            .relations
            .push(crate::ChunkRelation::superseded_by(target));
    }
    if let Some(ref target) = options.version_of {
        chunk
            .relations
            .push(crate::ChunkRelation::new("version_of", target));
    }
    if let Some(ref target) = options.related_to {
        chunk
            .relations
            .push(crate::ChunkRelation::related_to(target));
    }
    if let Some(ref target) = options.derived_from {
        chunk
            .relations
            .push(crate::ChunkRelation::new("derived_from", target));
    }

    let embeddings = embedder.embed(&[text])?;
    chunk.embedding = Some(
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| crate::Error::embedding("Failed to generate embedding"))?,
    );

    let id = chunk.id.clone();
    store.insert_chunks(vec![chunk]).await?;

    // Process bidirectional relation for related_to (mirror MCP logic)
    if let Some(ref target) = options.related_to {
        let target_id = crate::helpers::resolve_entry(&store, target).await?.id;
        let backward = crate::ChunkRelation::related_to(&id);
        store.add_relation(&target_id, backward).await?;
    }

    println!("Added entry {} ({})", short_id(&id), entry_type);

    Ok(AddResult {
        total_entries: 1,
        summary_entries: 0,
        files_processed: 0,
    })
}

/// Semantic search with hierarchical results (prints output).
pub async fn search(data_dir: &Path, query_str: &str, options: &SearchOptions) -> Result<()> {
    let results = search_results(data_dir, query_str, options).await?;

    if results.is_empty() {
        println!("No results found.");
        return Ok(());
    }

    for (i, result) in results.iter().enumerate() {
        if i > 0 {
            println!();
        }

        let heading = result
            .chunk
            .heading
            .as_deref()
            .unwrap_or_else(|| result.chunk.content.lines().next().unwrap_or("(untitled)"));
        let tier = crate::mcp::types::relevance_tier(result.score);
        println!("{}. {} ({}, {:.2})", i + 1, heading, tier, result.score,);

        // Compact metadata
        let mut meta = vec![short_id(&result.chunk.id).to_string()];
        if result.chunk.entry_type != EntryType::Raw {
            meta.push(result.chunk.entry_type.to_string());
        }
        if options.show_path && !result.hierarchy_path.is_empty() {
            let path: Vec<&str> = result
                .hierarchy_path
                .iter()
                .filter_map(|c| c.heading.as_deref())
                .collect();
            meta.push(path.join(" > "));
        }
        println!("   {}", meta.join(" | "));

        // Content preview
        println!("   {}", preview(&result.chunk.content, 200));

        if !result.relevant_children.is_empty() {
            for child in &result.relevant_children {
                let child_heading = child
                    .chunk
                    .heading
                    .as_deref()
                    .unwrap_or_else(|| child.chunk.content.lines().next().unwrap_or("..."));
                println!(
                    "     > {} [{}]",
                    preview(child_heading, 60),
                    short_id(&child.chunk.id)
                );
            }
        }
    }

    println!(
        "\n{} result(s). `veclayer focus <id>` to drill in.",
        results.len()
    );

    Ok(())
}

/// Run search and return structured results (for programmatic use).
pub async fn search_results(
    data_dir: &Path,
    query_str: &str,
    options: &SearchOptions,
) -> Result<Vec<SearchResult>> {
    let since_epoch = options
        .since
        .as_deref()
        .and_then(crate::helpers::parse_temporal);
    let until_epoch = options
        .until
        .as_deref()
        .and_then(crate::helpers::parse_temporal);

    let (embedder, store) = open_store(data_dir).await?;

    // Fetch more results when temporal filtering will reduce the set
    let fetch_limit = if since_epoch.is_some() || until_epoch.is_some() {
        options.top_k * TEMPORAL_PREFETCH_FACTOR
    } else {
        options.top_k
    };

    let config = SearchConfig::for_query(fetch_limit, options.deep, options.recent.as_deref())
        .with_perspective(options.perspective.clone())
        .with_min_salience(options.min_salience)
        .with_min_score(options.min_score);

    let search_engine = HierarchicalSearch::new(store, embedder).with_config(config);

    let results = if let Some(ref parent_id) = options.subtree {
        search_engine.search_subtree(query_str, parent_id).await?
    } else {
        search_engine.search(query_str).await?
    };

    let filtered = results
        .into_iter()
        .filter(|r| {
            let created = r.chunk.access_profile.created_at;
            since_epoch.is_none_or(|s| created >= s) && until_epoch.is_none_or(|u| created <= u)
        })
        .take(options.top_k);

    Ok(filtered
        .map(|r| SearchResult {
            chunk: r.chunk,
            score: r.score,
            hierarchy_path: r.hierarchy_path,
            relevant_children: r
                .relevant_children
                .into_iter()
                .map(|c| SearchResult {
                    chunk: c.chunk,
                    score: c.score,
                    hierarchy_path: vec![],
                    relevant_children: vec![],
                })
                .collect(),
        })
        .collect())
}

/// Backwards-compatible alias
pub async fn query(
    data_dir: &Path,
    query_str: &str,
    options: &SearchOptions,
) -> Result<Vec<SearchResult>> {
    search_results(data_dir, query_str, options).await
}

/// Focus on an entry: show details and children.
pub async fn focus(data_dir: &Path, id: &str, options: &FocusOptions) -> Result<()> {
    let (embedder, store) = open_store(data_dir).await?;

    let entry = store
        .get_by_id_prefix(id)
        .await?
        .ok_or_else(|| crate::Error::not_found(format!("Entry {} not found", id)))?;

    // Display entry details
    println!("Entry {}", short_id(&entry.id));
    println!("{}", "=".repeat(50));
    println!("  Type: {}", entry.entry_type);
    println!("  Level: {}", entry.level);
    println!("  Visibility: {}", entry.visibility);
    println!("  Source: {}", entry.source_file);
    if let Some(ref heading) = entry.heading {
        println!("  Heading: {}", heading);
    }
    if !entry.relations.is_empty() {
        println!("  Relations:");
        for rel in &entry.relations {
            println!("    {} -> {}", rel.kind, short_id(&rel.target_id));
        }
    }
    println!("\n{}\n", entry.content);

    // Get and display children
    let mut children = store.get_children(&entry.id).await?;

    if !children.is_empty() {
        // Optionally rerank by question
        if let Some(ref question) = options.question {
            let query_emb = embedder.embed(&[question.as_str()])?;
            if let Some(query_vec) = query_emb.into_iter().next() {
                children.sort_by(|a, b| {
                    let score_a = a
                        .embedding
                        .as_ref()
                        .map(|e| crate::search::cosine_similarity(&query_vec, e))
                        .unwrap_or(0.0);
                    let score_b = b
                        .embedding
                        .as_ref()
                        .map(|e| crate::search::cosine_similarity(&query_vec, e))
                        .unwrap_or(0.0);
                    score_b
                        .partial_cmp(&score_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        let shown = children.iter().take(options.limit).count();
        println!("Children ({}/{}):", shown, children.len());
        for child in children.iter().take(options.limit) {
            println!(
                "  {} [{}] {}",
                short_id(&child.id),
                child.entry_type,
                preview(&child.content, 100)
            );
        }

        println!("\nUse `veclayer focus <child-id>` to drill deeper.");
    } else {
        println!("(no children)");
    }

    Ok(())
}

/// Print store status (statistics).
pub async fn status(data_dir: &Path) -> Result<()> {
    let result = stats(data_dir).await?;

    println!("VecLayer Status");
    println!("{}", "=".repeat(40));
    println!("Store: {}", data_dir.display());
    println!("Total entries: {}", result.total_chunks);
    println!("\nEntries by level:");
    for level in 1..=7 {
        if let Some(count) = result.chunks_by_level.get(&level) {
            let level_name = if level <= 6 {
                format!("H{}", level)
            } else {
                "Content".to_string()
            };
            println!("  {}: {}", level_name, count);
        }
    }
    println!("\nSource files: {}", result.source_files.len());

    if !result.source_files.is_empty() {
        println!("\nNext: `veclayer search \"query\"` or `veclayer add <path>`");
    } else {
        println!("\nStore is empty. Use `veclayer add <path>` to add knowledge.");
    }

    Ok(())
}

/// Show statistics about the store (returns structured data).
pub async fn stats(data_dir: &Path) -> Result<StatsResult> {
    let store = LanceStore::open_metadata(data_dir).await?;

    let store_stats = store.stats().await?;

    Ok(StatsResult {
        total_chunks: store_stats.total_chunks,
        chunks_by_level: store_stats.chunks_by_level,
        source_files: store_stats.source_files,
    })
}

/// Print indexed source files.
pub async fn print_sources(data_dir: &Path) -> Result<()> {
    let result = sources(data_dir).await?;

    if result.is_empty() {
        println!("No files indexed. Use `veclayer add <path>` to add knowledge.");
    } else {
        println!("Indexed source files:");
        for file in &result {
            println!("  {}", file);
        }
    }

    Ok(())
}

/// List all indexed source files (returns data).
pub async fn sources(data_dir: &Path) -> Result<Vec<String>> {
    let store = LanceStore::open_metadata(data_dir).await?;

    let store_stats = store.stats().await?;

    Ok(store_stats.source_files)
}

/// Start the MCP/HTTP server.
pub async fn serve(data_dir: &Path, options: &ServeOptions) -> Result<()> {
    let config = Config::new()
        .with_data_dir(data_dir)
        .with_host(&options.host)
        .with_port(options.port)
        .with_read_only(options.read_only);

    if options.mcp_stdio {
        crate::mcp::run_stdio(config).await
    } else {
        crate::mcp::run_http(config).await
    }
}

// --- Perspective commands ---

/// List all perspectives.
pub fn perspective_list(data_dir: &Path) -> Result<()> {
    let perspectives = crate::perspective::load(data_dir)?;
    if perspectives.is_empty() {
        println!("No perspectives defined.");
        return Ok(());
    }
    for p in &perspectives {
        let tag = if p.builtin { " [builtin]" } else { "" };
        println!("  {} -- {}{}", p.id, p.hint, tag);
    }
    println!("\n{} perspective(s)", perspectives.len());
    Ok(())
}

/// Add a custom perspective.
pub fn perspective_add(data_dir: &Path, id: &str, name: &str, hint: &str) -> Result<()> {
    crate::perspective::add(
        data_dir,
        crate::perspective::Perspective::new(id, name, hint),
    )?;
    println!("Added perspective '{}'", id);
    Ok(())
}

/// Remove a custom perspective.
pub fn perspective_remove(data_dir: &Path, id: &str) -> Result<()> {
    crate::perspective::remove(data_dir, id)?;
    println!("Removed perspective '{}'", id);
    Ok(())
}

// --- History command ---

/// Show version/relation history of an entry.
pub async fn history(data_dir: &Path, id: &str) -> Result<()> {
    let store = LanceStore::open_metadata(data_dir).await?;

    // Resolve short IDs by prefix search
    let chunk = resolve_entry(&store, id).await?;

    println!("Entry {} ({})", short_id(&chunk.id), chunk.entry_type);
    if let Some(ref heading) = chunk.heading {
        println!("  Heading: {}", heading);
    }
    println!("  Content: {}", preview(&chunk.content, 80));

    if !chunk.perspectives.is_empty() {
        println!("  Perspectives: {}", chunk.perspectives.join(", "));
    }

    if chunk.relations.is_empty() {
        println!("  No relations.");
    } else {
        println!("  Relations:");
        for rel in &chunk.relations {
            println!("    {} -> {}", rel.kind, short_id(&rel.target_id));
        }
    }

    // Note: reverse relation lookup (entries that point TO this entry)
    // is deferred to Phase 3 (requires index or full scan).

    Ok(())
}

// --- Archive command ---

/// Archive entries by demoting them to deep_only visibility.
pub async fn archive(data_dir: &Path, ids: &[String]) -> Result<()> {
    let (_embedder, store) = open_store(data_dir).await?;

    for id in ids {
        let chunk = resolve_entry(&store, id).await?;
        store
            .update_visibility(&chunk.id, crate::chunk::visibility::DEEP_ONLY)
            .await?;
        println!(
            "Archived {} (was: {})",
            short_id(&chunk.id),
            chunk.visibility
        );
    }

    Ok(())
}

// --- Compact command ---

/// Options for the compact command.
#[derive(Debug, Clone)]
pub struct CompactOptions {
    /// Max entries to show for salience/archive-candidates reports.
    pub limit: usize,
    /// Salience threshold below which entries are archive candidates.
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

/// Compact sub-operations.
#[derive(Debug, Clone, Copy)]
pub enum CompactAction {
    /// Roll access-profile buckets for all entries.
    Rotate,
    /// Compute and display salience for top entries.
    Salience,
    /// Show entries that are candidates for archival (low salience).
    ArchiveCandidates,
}

/// Run a compact sub-action.
pub async fn compact(
    data_dir: &Path,
    action: CompactAction,
    options: &CompactOptions,
) -> Result<()> {
    match action {
        CompactAction::Rotate => compact_rotate(data_dir).await,
        CompactAction::Salience => compact_salience(data_dir, options).await,
        CompactAction::ArchiveCandidates => compact_archive_candidates(data_dir, options).await,
    }
}

/// Rotate: roll access-profile buckets and apply aging rules.
async fn compact_rotate(data_dir: &Path) -> Result<()> {
    let (_embedder, store) = open_store(data_dir).await?;

    let aging_config = crate::aging::AgingConfig::load(data_dir);
    let aging_result = crate::aging::apply_aging(&store, &aging_config).await?;

    println!("Compact: rotate");
    println!(
        "  Aging config: degrade after {} days",
        aging_config.degrade_after_days
    );
    println!(
        "  Degraded {} entries to '{}'",
        aging_result.degraded_count, aging_config.degrade_to
    );
    for id in &aging_result.degraded_ids {
        println!("    {}", short_id(id));
    }

    Ok(())
}

/// Salience: compute and display salience scores for the most/least salient entries.
async fn compact_salience(data_dir: &Path, options: &CompactOptions) -> Result<()> {
    let store = LanceStore::open_metadata(data_dir).await?;

    // Get hot chunks (most accessed) as a proxy for "all interesting entries"
    let hot = store.get_hot_chunks(options.limit * 2).await?;

    if hot.is_empty() {
        println!("No entries to analyze.");
        return Ok(());
    }

    let weights = crate::salience::SalienceWeights::default();
    let top = crate::salience::top_salient(&hot, &weights, options.limit);

    println!("Salience report (top {}):", top.len());
    println!("{}", "=".repeat(60));
    for (idx, score) in &top {
        let chunk = &hot[*idx];
        println!(
            "  {} [{:.3}] inter={:.2} persp={:.2} rev={:.2}  {}",
            short_id(&chunk.id),
            score.composite,
            score.interaction,
            score.perspective,
            score.revision,
            preview(&chunk.content, 60)
        );
    }

    Ok(())
}

/// Archive candidates: entries with low salience that could be archived.
async fn compact_archive_candidates(data_dir: &Path, options: &CompactOptions) -> Result<()> {
    let store = LanceStore::open_metadata(data_dir).await?;
    let aging_config = crate::aging::AgingConfig::load(data_dir);

    // Get stale chunks as the candidate pool
    let stale = store
        .get_stale_chunks(aging_config.stale_seconds(), options.limit * 2)
        .await?;

    if stale.is_empty() {
        println!("No archive candidates found.");
        return Ok(());
    }

    let weights = crate::salience::SalienceWeights::default();
    let candidates: Vec<_> = stale
        .iter()
        .filter(|c| {
            crate::salience::is_archive_candidate(
                c,
                &weights,
                options.archive_threshold,
                &aging_config.degrade_from,
            )
        })
        .take(options.limit)
        .collect();

    if candidates.is_empty() {
        println!(
            "No archive candidates below threshold {:.2}.",
            options.archive_threshold
        );
        return Ok(());
    }

    println!(
        "Archive candidates ({}, threshold {:.2}):",
        candidates.len(),
        options.archive_threshold
    );
    println!("{}", "=".repeat(60));
    for chunk in &candidates {
        let score = crate::salience::compute(chunk, &weights);
        println!(
            "  {} [salience={:.3}, vis={}]  {}",
            short_id(&chunk.id),
            score.composite,
            chunk.visibility,
            preview(&chunk.content, 60)
        );
    }
    println!("\nUse `veclayer archive <id>...` to archive selected entries.");

    Ok(())
}

// --- Reflect command ---

/// Generate a comprehensive reflection/identity report.
pub async fn reflect(data_dir: &Path) -> Result<()> {
    let store = LanceStore::open_metadata(data_dir).await?;
    let snapshot = crate::identity::compute_identity(&store, data_dir).await?;
    let priming = crate::identity::generate_priming(&snapshot);
    println!("{}", priming);
    Ok(())
}

// --- Think command (requires LLM) ---

/// Run one think cycle: reflect → LLM → add → compact.
///
/// This is the sleep cycle: VecLayer gathers context, the LLM generates
/// consolidations and learnings, VecLayer writes them back and cleans up.
#[cfg(feature = "llm")]
pub async fn think(data_dir: &Path) -> Result<()> {
    let (embedder, store) = open_store(data_dir).await?;

    let config = crate::Config::new().with_data_dir(data_dir);
    let llm = crate::llm::LlmBackend::from_config(&config.llm);

    println!(
        "Think: starting sleep cycle (LLM: {} via {})",
        config.llm.model, config.llm.provider
    );

    let result = crate::think::execute(&store, &embedder, &llm, data_dir).await?;

    if result.entries_created.is_empty() {
        println!("\nNothing to consolidate. Memory is either empty or already well-organized.");
        return Ok(());
    }

    println!("\nThink cycle complete:");

    if let Some(ref id) = result.narrative_id {
        println!("  Narrative: {}", short_id(id));
    }

    if result.consolidations_added > 0 {
        println!(
            "  Consolidations: {} summaries created",
            result.consolidations_added
        );
    }

    if result.learnings_added > 0 {
        println!(
            "  Learnings: {} meta-entries extracted",
            result.learnings_added
        );
    }

    println!("\nEntries created:");
    for entry in &result.entries_created {
        let persp = if entry.perspectives.is_empty() {
            String::new()
        } else {
            format!(" [{}]", entry.perspectives.join(", "))
        };
        println!(
            "  {} ({}{}) {}",
            short_id(&entry.id),
            entry.entry_type,
            persp,
            entry.content_preview
        );
    }

    println!("\nAging applied. Run `veclayer reflect` to see updated identity.");

    Ok(())
}

// --- Orientation command (default, no args) ---

/// Quick orientation: "Who am I, what's on my mind?"
pub async fn orientation(data_dir: &Path) -> Result<()> {
    let store = LanceStore::open_metadata(data_dir).await?;

    let store_stats = store.stats().await?;
    if store_stats.total_chunks == 0 {
        println!("VecLayer is empty. Get started:");
        println!("  veclayer add \"Your first piece of knowledge\"");
        println!("  veclayer add ./notes/");
        println!("  veclayer search \"What do I know about X?\"");
        return Ok(());
    }

    let snapshot = crate::identity::compute_identity(&store, data_dir).await?;

    println!(
        "VecLayer — {} entries from {} sources",
        store_stats.total_chunks,
        store_stats.source_files.len()
    );

    // Perspective summary (one line)
    if !snapshot.centroids.is_empty() {
        let persp_summary: Vec<String> = snapshot
            .centroids
            .iter()
            .map(|c| format!("{} ({})", c.perspective, c.entry_count))
            .collect();
        println!("Perspectives: {}", persp_summary.join(", "));
    }

    // Top 5 core
    if !snapshot.core_entries.is_empty() {
        println!("\nMost important:");
        for entry in snapshot.core_entries.iter().take(5) {
            let heading = entry.heading.as_deref().unwrap_or("(untitled)");
            println!("  {} {}", short_id(&entry.id), heading);
        }
    }

    // Open threads (brief)
    if !snapshot.open_threads.is_empty() {
        println!(
            "\n{} open thread(s) need attention. Run `veclayer reflect` for details.",
            snapshot.open_threads.len()
        );
    }

    // Hints
    println!("\nTry: search, reflect, think, reflect salience");

    Ok(())
}

// --- Browse command (#29) ---

/// Browse entries without vector search (list by perspective/recency).
pub async fn browse(data_dir: &Path, options: &SearchOptions) -> Result<()> {
    let since_epoch = options
        .since
        .as_deref()
        .and_then(crate::helpers::parse_temporal);
    let until_epoch = options
        .until
        .as_deref()
        .and_then(crate::helpers::parse_temporal);

    let store = LanceStore::open_metadata(data_dir).await?;
    let entries = store
        .list_entries(
            options.perspective.as_deref(),
            since_epoch,
            until_epoch,
            options.top_k,
        )
        .await?;

    if entries.is_empty() {
        println!("No entries found.");
        return Ok(());
    }

    for (i, chunk) in entries.iter().enumerate() {
        if i > 0 {
            println!();
        }
        let heading = chunk
            .heading
            .as_deref()
            .unwrap_or_else(|| chunk.content.lines().next().unwrap_or("(untitled)"));
        println!("{}. {}", i + 1, heading);

        let mut meta = vec![short_id(&chunk.id).to_string()];
        if chunk.entry_type != EntryType::Raw {
            meta.push(chunk.entry_type.to_string());
        }
        if !chunk.perspectives.is_empty() {
            meta.push(chunk.perspectives.join(", "));
        }
        meta.push(chunk.visibility.clone());
        println!("   {}", meta.join(" | "));
        println!("   {}", preview(&chunk.content, 200));
    }

    println!(
        "\n{} entry(ies). `veclayer focus <id>` to drill in.",
        entries.len()
    );
    Ok(())
}

// --- Think subcommands (CLI parity with MCP) ---

/// Set an entry's visibility and print a labeled confirmation.
async fn set_visibility(data_dir: &Path, id: &str, visibility: &str, label: &str) -> Result<()> {
    let store = LanceStore::open_metadata(data_dir).await?;
    let store = std::sync::Arc::new(store);
    let chunk_id = crate::helpers::resolve_id(&store, id).await?;
    store.update_visibility(&chunk_id, visibility).await?;
    println!("{} {} to visibility '{}'", label, short_id(&chunk_id), visibility);
    Ok(())
}

/// Promote an entry's visibility (CLI for MCP think(promote)).
pub async fn think_promote(data_dir: &Path, id: &str, visibility: &str) -> Result<()> {
    set_visibility(data_dir, id, visibility, "Promoted").await
}

/// Demote an entry's visibility (CLI for MCP think(demote)).
pub async fn think_demote(data_dir: &Path, id: &str, visibility: &str) -> Result<()> {
    set_visibility(data_dir, id, visibility, "Demoted").await
}

/// Add a relation between two entries (CLI for MCP think(relate)).
pub async fn think_relate(data_dir: &Path, source: &str, target: &str, kind: &str) -> Result<()> {
    let (_embedder, store) = open_store(data_dir).await?;
    let store = std::sync::Arc::new(store);
    let source_id = crate::helpers::resolve_id(&store, source).await?;
    let target_id = crate::helpers::resolve_id(&store, target).await?;

    let relation = crate::ChunkRelation::new(kind, &target_id);
    store.add_relation(&source_id, relation).await?;

    // For bidirectional relations, add the reverse
    if kind == "related_to" {
        let backward = crate::ChunkRelation::new("related_to", &source_id);
        store.add_relation(&target_id, backward).await?;
    }

    println!(
        "Added relation '{}' from {} to {}",
        kind,
        short_id(&source_id),
        short_id(&target_id)
    );
    Ok(())
}

/// Apply aging rules (CLI for MCP think(apply_aging)).
pub async fn think_aging_apply(data_dir: &Path) -> Result<()> {
    let (_embedder, store) = open_store(data_dir).await?;
    let config = crate::aging::AgingConfig::load(data_dir);
    let result = crate::aging::apply_aging(&store, &config).await?;

    if result.degraded_count == 0 {
        println!("No entries needed aging. All knowledge is fresh.");
    } else {
        println!(
            "Aged {} entries (degraded to '{}'):",
            result.degraded_count, config.degrade_to
        );
        for id in &result.degraded_ids {
            println!("  {}", short_id(id));
        }
    }
    Ok(())
}

/// Configure aging parameters (CLI for MCP think(configure_aging)).
pub async fn think_aging_configure(
    data_dir: &Path,
    days: Option<u32>,
    to: Option<&str>,
) -> Result<()> {
    let mut config = crate::aging::AgingConfig::load(data_dir);
    if let Some(days) = days {
        config.degrade_after_days = days;
    }
    if let Some(to) = to {
        config.degrade_to = to.to_string();
    }
    config.save(data_dir)?;
    println!(
        "Aging configured: degrade {} -> '{}' after {} days without access",
        config.degrade_from.join(", "),
        config.degrade_to,
        config.degrade_after_days
    );
    Ok(())
}

/// Resolve a potentially short ID to a full entry.
///
/// Delegates to `helpers::resolve_entry`.
async fn resolve_entry(store: &LanceStore, id: &str) -> Result<crate::HierarchicalChunk> {
    crate::helpers::resolve_entry(store, id).await
}

/// Collect files from a path, optionally recursively.
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
    fn test_add_options_default() {
        let opts = AddOptions::default();
        assert!(opts.recursive);
        assert!(opts.summarize);
        assert_eq!(opts.model, "llama3.2");
        assert_eq!(opts.entry_type, "raw");
        assert!(opts.parent_id.is_none());
        assert!(opts.heading.is_none());
        assert!(opts.related_to.is_none());
        assert!(opts.derived_from.is_none());
    }

    #[test]
    fn test_search_options_default() {
        let opts = SearchOptions::default();
        assert_eq!(opts.top_k, 5);
        assert!(!opts.show_path);
        assert!(opts.subtree.is_none());
        assert!(opts.since.is_none());
        assert!(opts.until.is_none());
    }

    #[test]
    fn test_focus_options_default() {
        let opts = FocusOptions::default();
        assert!(opts.question.is_none());
        assert_eq!(opts.limit, 10);
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
    fn test_init_creates_directory() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let store_dir = temp_dir.path().join("new-store");

        init(&store_dir)?;

        assert!(store_dir.exists());
        Ok(())
    }

    #[test]
    fn test_init_existing_directory() -> Result<()> {
        let temp_dir = TempDir::new()?;

        // Should not error on existing directory
        init(temp_dir.path())?;
        Ok(())
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

        let files_non_recursive = collect_files(temp_dir.path(), false, &parser)?;
        assert_eq!(files_non_recursive.len(), 1);

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
