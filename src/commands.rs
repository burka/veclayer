//! Command implementations for VecLayer CLI and library use.
//!
//! This module provides clean, testable command implementations that can be used
//! both from the CLI and programmatically as a library.

use std::path::{Path, PathBuf};

use tracing::info;

use crate::chunk::short_id;
#[cfg(feature = "llm")]
use crate::cluster::ClusterPipeline;
use crate::embedder::FastEmbedder;
use crate::parser::MarkdownParser;
use crate::search::{HierarchicalSearch, SearchConfig};
use crate::store::LanceStore;
#[cfg(feature = "llm")]
use crate::summarizer::OllamaSummarizer;
use crate::{Config, DocumentParser, Embedder, Result, VectorStore};

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
    info!("Opening store at {:?}...", data_dir);
    let (embedder, store) = open_store(data_dir).await?;

    let parser = MarkdownParser::new();

    let files = collect_files(path, options.recursive, &parser)?;
    info!("Found {} files to process", files.len());

    let mut all_chunks = Vec::new();

    for file in &files {
        info!("Processing {:?}...", file);

        let deleted = store.delete_by_source(&file.to_string_lossy()).await?;
        if deleted > 0 {
            info!("  Removed {} existing entries", deleted);
        }

        let mut chunks = parser.parse_file(file)?;
        info!("  Parsed {} entries", chunks.len());

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
        info!("  Indexed successfully");

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

    let mut chunk = crate::HierarchicalChunk::new(
        text.to_string(),
        crate::ChunkLevel::CONTENT,
        None,
        String::new(),
        "[inline]".to_string(),
    )
    .with_entry_type(entry_type)
    .with_perspectives(options.perspectives.clone());

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

    let embeddings = embedder.embed(&[text])?;
    chunk.embedding = Some(
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| crate::Error::embedding("Failed to generate embedding"))?,
    );

    let id = chunk.id.clone();
    store.insert_chunks(vec![chunk]).await?;

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

    println!("\nSearch results for: \"{}\"\n", query_str);
    println!("{}", "=".repeat(60));

    for (i, result) in results.iter().enumerate() {
        println!(
            "\n{}. [Score: {:.3}] {} ({})",
            i + 1,
            result.score,
            result.chunk.level,
            result.chunk.entry_type,
        );

        if options.show_path && !result.hierarchy_path.is_empty() {
            let path: Vec<&str> = result
                .hierarchy_path
                .iter()
                .filter_map(|c| c.heading.as_deref())
                .collect();
            println!("   Path: {}", path.join(" > "));
        }

        println!("   Source: {}", result.chunk.source_file);

        if let Some(ref heading) = result.chunk.heading {
            println!("   Heading: {}", heading);
        }

        let content = preview(&result.chunk.content, 200);
        println!("   Content: {}", content.replace('\n', " "));

        if !result.relevant_children.is_empty() {
            println!("   Children:");
            for child in &result.relevant_children {
                println!(
                    "     - [Score: {:.3}] {}",
                    child.score,
                    preview(&child.chunk.content, 80)
                );
            }
        }

        println!("   ID: {}", short_id(&result.chunk.id));
    }

    println!(
        "\n{} results. Use `veclayer focus <id>` to drill in.",
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
    let (embedder, store) = open_store(data_dir).await?;

    let config = SearchConfig::for_query(options.top_k, options.deep, options.recent.as_deref())
        .with_perspective(options.perspective.clone())
        .with_min_salience(options.min_salience)
        .with_min_score(options.min_score);

    let search_engine = HierarchicalSearch::new(store, embedder).with_config(config);

    let results = if let Some(ref parent_id) = options.subtree {
        search_engine.search_subtree(query_str, parent_id).await?
    } else {
        search_engine.search(query_str).await?
    };

    Ok(results
        .into_iter()
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
        .get_by_id(id)
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

// --- Identity command ---

/// Show a compact identity summary.
pub async fn identity(data_dir: &Path) -> Result<()> {
    let store = LanceStore::open_metadata(data_dir).await?;
    let snapshot = crate::identity::compute_identity(&store, data_dir).await?;

    println!("VecLayer Identity");
    println!("{}", "=".repeat(40));

    // Perspective coverage
    if snapshot.centroids.is_empty() {
        println!("\nNo perspective data yet. Add entries with --perspective flags.");
    } else {
        println!("\nPerspectives:");
        for c in &snapshot.centroids {
            let bar_len = (c.avg_salience * 20.0).round() as usize;
            let bar = "#".repeat(bar_len.min(20));
            println!(
                "  {:12} {:>3} entries  salience {:.2}  {}",
                c.perspective, c.entry_count, c.avg_salience, bar
            );
        }
    }

    // Core knowledge
    if !snapshot.core_entries.is_empty() {
        println!("\nCore Knowledge (top {}):", snapshot.core_entries.len());
        for entry in &snapshot.core_entries {
            let heading = entry.heading.as_deref().unwrap_or("(untitled)");
            println!(
                "  {} [{:.2}] {}",
                short_id(&entry.id),
                entry.salience,
                heading
            );
        }
    }

    // Open threads
    if !snapshot.open_threads.is_empty() {
        println!("\nOpen Threads ({}):", snapshot.open_threads.len());
        for thread in &snapshot.open_threads {
            let heading = thread.heading.as_deref().unwrap_or("(untitled)");
            println!("  {} {}: {}", short_id(&thread.id), heading, thread.reason);
        }
    }

    // Recent learnings
    if !snapshot.recent_learnings.is_empty() {
        println!("\nRecent Learnings:");
        for learning in &snapshot.recent_learnings {
            let heading = learning.heading.as_deref().unwrap_or("(untitled)");
            println!("  {} {}", short_id(&learning.id), heading);
        }
    }

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
    println!("\nTry: search, reflect, compact salience, id");

    Ok(())
}

/// Resolve a potentially short ID to a full entry.
///
/// Tries exact match first. If the input looks like a short ID (hex, <64 chars),
/// the error message hints the user to use the full ID.
async fn resolve_entry(store: &LanceStore, id: &str) -> Result<crate::HierarchicalChunk> {
    if let Some(chunk) = store.get_by_id(id).await? {
        return Ok(chunk);
    }

    let is_short_hex = id.len() < 64 && id.chars().all(|c| c.is_ascii_hexdigit());
    if is_short_hex {
        Err(crate::Error::not_found(format!(
            "Entry '{}' not found. Short ID prefix lookup is not yet supported — use the full 64-char hash.",
            id
        )))
    } else {
        Err(crate::Error::not_found(format!("Entry '{}' not found", id)))
    }
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
    }

    #[test]
    fn test_search_options_default() {
        let opts = SearchOptions::default();
        assert_eq!(opts.top_k, 5);
        assert!(!opts.show_path);
        assert!(opts.subtree.is_none());
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
