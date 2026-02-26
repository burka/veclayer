//! Command implementations for VecLayer CLI and library use.
//!
//! This module provides clean, testable command implementations that can be used
//! both from the CLI and programmatically as a library.

use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use owo_colors::{OwoColorize, Stream};
use tracing::{debug, info, warn};

use crate::blob_store::BlobStore;
use crate::chunk::{short_id, EntryType};
#[cfg(feature = "llm")]
use crate::cluster::ClusterPipeline;
use crate::embedder::FastEmbedder;
use crate::parser::MarkdownParser;
use crate::search::{HierarchicalSearch, SearchConfig};
use crate::store::StoreBackend;
#[cfg(feature = "llm")]
use crate::summarizer::OllamaSummarizer;
use crate::{Config, DocumentParser, Embedder, Result, VectorStore};

// Over-fetch when temporal filters are active, then filter client-side
use crate::search::TEMPORAL_PREFETCH_FACTOR;

// --- Infrastructure helpers ---

/// Create an embedder + store + blob store triple.  Centralises the init
/// sequence that was previously repeated in every command that needs embeddings.
async fn open_store(data_dir: &Path) -> Result<(FastEmbedder, StoreBackend, BlobStore)> {
    let embedder = FastEmbedder::new()?;
    let dimension = embedder.dimension();
    let store = StoreBackend::open(data_dir, dimension, false).await?;
    let blob_store = BlobStore::open(data_dir)?;
    Ok((embedder, store, blob_store))
}

// --- Output helpers ---

/// Color a visibility string for CLI display.
fn vis_color(vis: &str) -> String {
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
fn preview(s: &str, max: usize) -> String {
    let clean = s.replace('\n', " ");
    if clean.len() <= max {
        clean
    } else {
        // Find a char boundary at or before `max` to avoid panicking on multi-byte UTF-8
        let end = clean.floor_char_boundary(max);
        format!("{}...", &clean[..end])
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
    /// Parent entry ID for hierarchy placement
    pub parent_id: Option<String>,
    /// Heading/title for the entry
    pub heading: Option<String>,
    /// Impression hint: qualitative label (e.g. "uncertain", "confident")
    pub impression_hint: Option<String>,
    /// Impression strength: 0.0–1.0 (default 1.0)
    pub impression_strength: f32,
    /// --rel-supersedes targets (auto-demotes, inverse superseded_by)
    pub rel_supersedes: Vec<String>,
    /// --rel-summarizes targets (inverse summarized_by)
    pub rel_summarizes: Vec<String>,
    /// --rel-to targets (bidirectional related_to)
    pub rel_to: Vec<String>,
    /// --rel-derived-from targets (forward only)
    pub rel_derived_from: Vec<String>,
    /// --rel-version-of targets (auto-demotes, inverse superseded_by)
    pub rel_version_of: Vec<String>,
    /// -R / --rel KIND:ID (custom forward on self)
    pub rel_custom: Vec<String>,
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
    /// Search for entries similar to this entry ID (uses its embedding as query)
    pub similar_to: Option<String>,
    /// Minimum salience (entries below this excluded from salience boosting)
    pub min_salience: Option<f32>,
    /// Minimum search score (entries below this are filtered out)
    pub min_score: Option<f32>,
    /// Only entries created after this ISO 8601 date or epoch seconds
    pub since: Option<String>,
    /// Only entries created before this ISO 8601 date or epoch seconds
    pub until: Option<String>,
    /// Filter to open threads (unresolved items)
    pub ongoing: bool,
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
    /// Project scope for memory isolation
    pub project: Option<String>,
    /// Git branch for branch-scoped entries
    pub branch: Option<String>,
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
        }
    }
}

/// Options for exporting entries to JSONL
#[derive(Debug, Clone, Default)]
pub struct ExportOptions {
    /// Filter by perspective (e.g. "decisions", "learnings")
    pub perspective: Option<String>,
}

/// Options for importing entries from JSONL
#[derive(Debug, Clone, Default)]
pub struct ImportOptions {
    /// Path to JSONL file, or "-" for stdin
    pub path: String,
}

/// Options for merging another store's blobs into this one
#[derive(Debug, Clone, Default)]
pub struct MergeOptions {
    /// Project scope to tag merged entries with (e.g. "veclayer")
    pub project: Option<String>,
    /// Preview only, don't actually merge
    pub dry_run: bool,
    /// Merge without project scope, suppress warning
    pub force: bool,
}

/// Result of an import operation
#[derive(Debug)]
pub struct ImportResult {
    /// Number of entries successfully imported
    pub imported: usize,
    /// Number of entries skipped (already exist)
    pub skipped: usize,
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
    println!("  veclayer store ./docs      # Store files");
    println!("  veclayer store \"text\"      # Store inline text");
    println!("  veclayer recall \"query\"    # Recall knowledge");
    Ok(())
}

/// Add knowledge to the store (files, directories, or inline text).
///
/// If `input` is a path to a file or directory, it parses and ingests documents.
/// Otherwise, it treats the input as inline text content.
pub async fn add(data_dir: &Path, input: &str, mut options: AddOptions) -> Result<AddResult> {
    // Expand comma-separated perspectives (e.g. "knowledge,decisions" → ["knowledge", "decisions"])
    options.perspectives = options
        .perspectives
        .iter()
        .flat_map(|p| p.split(',').map(|s| s.trim().to_string()))
        .filter(|s| !s.is_empty())
        .collect();

    // Validate perspectives early (fail fast at CLI boundary)
    if !options.perspectives.is_empty() {
        crate::perspective::validate_ids(data_dir, &options.perspectives)?;
    }

    let input_path = Path::new(input);

    if input_path.exists() {
        // Input is a file or directory -- ingest it
        add_files(data_dir, input_path, &options).await
    } else {
        // Input is inline text -- store as a single entry
        add_text(data_dir, input, &options).await
    }
}

/// Backwards-compatible alias
pub async fn ingest(data_dir: &Path, path: &Path, options: &AddOptions) -> Result<AddResult> {
    add_files(data_dir, path, options).await
}

/// Add files from a path to the store.
async fn add_files(data_dir: &Path, path: &Path, options: &AddOptions) -> Result<AddResult> {
    debug!("Opening store at {:?}...", data_dir);
    let (embedder, store, blob_store) = open_store(data_dir).await?;

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

        // Persist to blob store
        for chunk in &chunks {
            let blob = crate::entry::StoredBlob::from_chunk_and_embedding(chunk, embedder.name());
            blob_store.put(&blob)?;
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
                    for chunk in &summary_chunk_list {
                        let blob = crate::entry::StoredBlob::from_chunk_and_embedding(
                            chunk,
                            embedder.name(),
                        );
                        blob_store.put(&blob)?;
                    }
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
    let (embedder, store, blob_store) = open_store(data_dir).await?;

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

    // Impression metadata
    if let Some(ref hint) = options.impression_hint {
        chunk.impression_hint = Some(hint.clone());
    }
    chunk.impression_strength = options.impression_strength;

    let embeddings = embedder.embed(&[text])?;
    chunk.embedding = Some(
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| crate::Error::embedding("Failed to generate embedding"))?,
    );

    // Persist to blob store
    let blob = crate::entry::StoredBlob::from_chunk_and_embedding(&chunk, embedder.name());
    blob_store.put(&blob)?;

    let id = chunk.id.clone();
    let store = std::sync::Arc::new(store);
    store.insert_chunks(vec![chunk]).await?;

    // Collect all raw relations from --rel-* flags
    let mut raw_relations = Vec::new();
    for target in &options.rel_supersedes {
        raw_relations.push(crate::relations::RawRelation {
            kind: "supersedes".to_string(),
            target_id: target.clone(),
        });
    }
    for target in &options.rel_summarizes {
        raw_relations.push(crate::relations::RawRelation {
            kind: "summarizes".to_string(),
            target_id: target.clone(),
        });
    }
    for target in &options.rel_to {
        raw_relations.push(crate::relations::RawRelation {
            kind: "related_to".to_string(),
            target_id: target.clone(),
        });
    }
    for target in &options.rel_derived_from {
        raw_relations.push(crate::relations::RawRelation {
            kind: "derived_from".to_string(),
            target_id: target.clone(),
        });
    }
    for target in &options.rel_version_of {
        raw_relations.push(crate::relations::RawRelation {
            kind: "version_of".to_string(),
            target_id: target.clone(),
        });
    }
    // Parse --rel / -R KIND:ID custom relations
    for spec in &options.rel_custom {
        if let Some((kind, target_id)) = spec.split_once(':') {
            if kind.is_empty() || target_id.is_empty() {
                return Err(crate::Error::parse(format!(
                    "Invalid --rel format '{}': expected KIND:ID",
                    spec
                )));
            }
            crate::relations::validate_relation_kind(kind)?;
            raw_relations.push(crate::relations::RawRelation {
                kind: kind.to_string(),
                target_id: target_id.to_string(),
            });
        } else {
            return Err(crate::Error::parse(format!(
                "Invalid --rel format '{}': expected KIND:ID",
                spec
            )));
        }
    }

    crate::relations::process_relations(&store, &id, raw_relations).await?;

    println!("Added entry {} ({})", short_id(&id), entry_type);

    Ok(AddResult {
        total_entries: 1,
        summary_entries: 0,
        files_processed: 0,
    })
}

/// Semantic search with hierarchical results (prints output).
pub async fn search(data_dir: &Path, query_str: &str, options: &SearchOptions) -> Result<()> {
    if options.similar_to.is_some() && !query_str.is_empty() {
        return Err(crate::Error::InvalidOperation(
            "--similar-to and a text query are mutually exclusive. Use one or the other."
                .to_string(),
        ));
    }

    let results = search_results(data_dir, query_str, options).await?;

    if results.is_empty() {
        if let Some(ref target_id) = options.similar_to {
            println!("No similar entries found for: {}", short_id(target_id));
        } else {
            println!("No results found.");
        }
        return Ok(());
    }

    if let Some(ref target_id) = options.similar_to {
        println!("\nSimilar entries to: {}\n", short_id(target_id));
    } else {
        println!("\nSearch results for: \"{}\"\n", query_str);
    }
    println!("{}", "=".repeat(60));

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
        println!(
            "{}  {} {:.2}",
            format!("{}. {}", i + 1, heading).if_supports_color(Stream::Stdout, |s| s.bold()),
            tier.if_supports_color(Stream::Stdout, |s| s.dimmed()),
            result
                .score
                .if_supports_color(Stream::Stdout, |s| s.dimmed()),
        );

        // Compact metadata
        let mut meta = vec![short_id(&result.chunk.id)
            .if_supports_color(Stream::Stdout, |s| s.cyan())
            .to_string()];
        if result.chunk.entry_type != EntryType::Raw {
            meta.push(
                result
                    .chunk
                    .entry_type
                    .to_string()
                    .if_supports_color(Stream::Stdout, |s| s.yellow())
                    .to_string(),
            );
        }
        if !result.chunk.perspectives.is_empty() {
            meta.push(
                result
                    .chunk
                    .perspectives
                    .join(", ")
                    .if_supports_color(Stream::Stdout, |s| s.magenta())
                    .to_string(),
            );
        }
        if result.chunk.visibility != "normal" {
            meta.push(
                result
                    .chunk
                    .visibility
                    .if_supports_color(Stream::Stdout, |s| s.red())
                    .to_string(),
            );
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
        println!(
            "   {}",
            preview(&result.chunk.content, 200).if_supports_color(Stream::Stdout, |s| s.dimmed())
        );

        if !result.relevant_children.is_empty() {
            for child in &result.relevant_children {
                let child_heading = child
                    .chunk
                    .heading
                    .as_deref()
                    .unwrap_or_else(|| child.chunk.content.lines().next().unwrap_or("..."));
                println!(
                    "     {} {} [{}]",
                    ">".if_supports_color(Stream::Stdout, |s| s.dimmed()),
                    preview(child_heading, 60),
                    short_id(&child.chunk.id).if_supports_color(Stream::Stdout, |s| s.cyan())
                );
            }
        }
    }

    println!(
        "\n{} `veclayer focus <id>` to drill in.",
        format!("{} result(s).", results.len()).if_supports_color(Stream::Stdout, |s| s.bold())
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
        .and_then(crate::resolve::parse_temporal);
    let until_epoch = options
        .until
        .as_deref()
        .and_then(crate::resolve::parse_temporal);

    let (embedder, store, _blob_store) = open_store(data_dir).await?;

    let open_thread_ids = crate::identity::resolve_ongoing_filter(&store, options.ongoing).await?;

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

    let results = if let Some(ref target_id) = options.similar_to {
        search_engine
            .search_by_embedding(target_id, fetch_limit)
            .await?
    } else if let Some(ref parent_id) = options.subtree {
        search_engine.search_subtree(query_str, parent_id).await?
    } else {
        search_engine.search(query_str).await?
    };

    let filtered = results
        .into_iter()
        .filter(|r| {
            let created = r.chunk.access_profile.created_at;
            since_epoch.is_none_or(|s| created >= s)
                && until_epoch.is_none_or(|u| created <= u)
                && crate::identity::passes_ongoing_filter(&open_thread_ids, &r.chunk.id)
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
    let (embedder, store, _blob_store) = open_store(data_dir).await?;

    let entry = store
        .get_by_id_prefix(id)
        .await?
        .ok_or_else(|| crate::Error::not_found(format!("Entry {} not found", id)))?;

    // Display entry details
    println!(
        "{}",
        format!("Entry {}", short_id(&entry.id)).if_supports_color(Stream::Stdout, |s| s.bold())
    );
    println!(
        "{}",
        "=".repeat(50)
            .if_supports_color(Stream::Stdout, |s| s.dimmed())
    );
    println!(
        "  {}  {}",
        "Type:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
        entry
            .entry_type
            .to_string()
            .if_supports_color(Stream::Stdout, |s| s.yellow())
    );
    println!(
        "  {}  {}",
        "Level:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
        entry.level
    );
    println!(
        "  {}  {}",
        "Vis:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
        vis_color(&entry.visibility)
    );
    println!(
        "  {}  {}",
        "Source:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
        entry.source_file
    );
    if let Some(ref heading) = entry.heading {
        println!(
            "  {}  {}",
            "Heading:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
            heading
        );
    }
    if !entry.perspectives.is_empty() {
        println!(
            "  {}  {}",
            "Perspectives:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
            entry
                .perspectives
                .join(", ")
                .if_supports_color(Stream::Stdout, |s| s.magenta())
        );
    }
    if !entry.relations.is_empty() {
        println!(
            "  {}",
            "Relations:".if_supports_color(Stream::Stdout, |s| s.dimmed())
        );
        for rel in &entry.relations {
            println!(
                "    {} {} {}",
                rel.kind.if_supports_color(Stream::Stdout, |s| s.yellow()),
                "->".if_supports_color(Stream::Stdout, |s| s.dimmed()),
                short_id(&rel.target_id).if_supports_color(Stream::Stdout, |s| s.cyan())
            );
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
        println!(
            "{}",
            format!("Children ({}/{}):", shown, children.len())
                .if_supports_color(Stream::Stdout, |s| s.bold())
        );
        for child in children.iter().take(options.limit) {
            println!(
                "  {} [{}] {}",
                short_id(&child.id).if_supports_color(Stream::Stdout, |s| s.cyan()),
                child
                    .entry_type
                    .to_string()
                    .if_supports_color(Stream::Stdout, |s| s.yellow()),
                preview(&child.content, 100).if_supports_color(Stream::Stdout, |s| s.dimmed())
            );
        }

        println!("\nUse `veclayer focus <child-id>` to drill deeper.");
    } else {
        println!(
            "{}",
            "(no children)".if_supports_color(Stream::Stdout, |s| s.dimmed())
        );
    }

    Ok(())
}

/// Print store status (statistics).
pub async fn status(data_dir: &Path) -> Result<()> {
    let result = stats(data_dir).await?;

    println!(
        "{}",
        "VecLayer Status".if_supports_color(Stream::Stdout, |s| s.bold())
    );
    println!(
        "{}",
        "=".repeat(40)
            .if_supports_color(Stream::Stdout, |s| s.dimmed())
    );
    println!(
        "{}  {}",
        "Store:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
        data_dir.display()
    );
    println!(
        "{}  {}",
        "Total entries:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
        result
            .total_chunks
            .if_supports_color(Stream::Stdout, |s| s.bold())
    );
    println!(
        "\n{}",
        "Entries by level:".if_supports_color(Stream::Stdout, |s| s.dimmed())
    );
    for level in 1..=7 {
        if let Some(count) = result.chunks_by_level.get(&level) {
            let level_name = if level <= 6 {
                format!("H{}", level)
            } else {
                "Content".to_string()
            };
            println!(
                "  {}  {}",
                level_name.if_supports_color(Stream::Stdout, |s| s.cyan()),
                count
            );
        }
    }
    println!("\nSource files: {}", result.source_files.len());

    if !result.source_files.is_empty() {
        println!("\nNext: `veclayer recall \"query\"` or `veclayer store <path>`");
    } else {
        println!("\nStore is empty. Use `veclayer store <path>` to add knowledge.");
    }

    Ok(())
}

/// Show statistics about the store (returns structured data).
pub async fn stats(data_dir: &Path) -> Result<StatsResult> {
    let store = StoreBackend::open_metadata(data_dir, true).await?;

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
        println!("No files indexed. Use `veclayer store <path>` to add knowledge.");
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
    let store = StoreBackend::open_metadata(data_dir, true).await?;

    let store_stats = store.stats().await?;

    Ok(store_stats.source_files)
}

/// Start the MCP/HTTP server.
pub async fn serve(data_dir: &Path, options: &ServeOptions) -> Result<()> {
    // Auto-create data directory with restrictive permissions (contains personal memory)
    if !data_dir.exists() {
        std::fs::create_dir_all(data_dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(data_dir, std::fs::Permissions::from_mode(0o700))?;
        }
    }

    let config = Config::new()
        .with_data_dir(data_dir)
        .with_host(&options.host)
        .with_port(options.port)
        .with_read_only(options.read_only)
        .with_project(options.project.clone())
        .with_branch(options.branch.clone());

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
    let store = StoreBackend::open_metadata(data_dir, true).await?;

    // Resolve short IDs by prefix search
    let chunk = resolve_entry(&store, id).await?;

    println!(
        "{} ({})",
        format!("Entry {}", short_id(&chunk.id)).if_supports_color(Stream::Stdout, |s| s.bold()),
        chunk
            .entry_type
            .to_string()
            .if_supports_color(Stream::Stdout, |s| s.yellow())
    );
    if let Some(ref heading) = chunk.heading {
        println!(
            "  {}  {}",
            "Heading:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
            heading
        );
    }
    println!(
        "  {}  {}",
        "Content:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
        preview(&chunk.content, 80).if_supports_color(Stream::Stdout, |s| s.dimmed())
    );

    if !chunk.perspectives.is_empty() {
        println!(
            "  {}  {}",
            "Perspectives:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
            chunk
                .perspectives
                .join(", ")
                .if_supports_color(Stream::Stdout, |s| s.magenta())
        );
    }

    if chunk.relations.is_empty() {
        println!(
            "  {}",
            "No relations.".if_supports_color(Stream::Stdout, |s| s.dimmed())
        );
    } else {
        println!(
            "  {}",
            "Relations:".if_supports_color(Stream::Stdout, |s| s.dimmed())
        );
        for rel in &chunk.relations {
            println!(
                "    {} {} {}",
                rel.kind.if_supports_color(Stream::Stdout, |s| s.yellow()),
                "->".if_supports_color(Stream::Stdout, |s| s.dimmed()),
                short_id(&rel.target_id).if_supports_color(Stream::Stdout, |s| s.cyan())
            );
        }
    }

    // Note: reverse relation lookup (entries that point TO this entry)
    // is deferred to Phase 3 (requires index or full scan).

    Ok(())
}

// --- Archive command ---

/// Archive entries by demoting them to deep_only visibility.
pub async fn archive(data_dir: &Path, ids: &[String]) -> Result<()> {
    if ids.is_empty() {
        return Err(crate::Error::InvalidOperation(
            "No entry IDs provided. Usage: veclayer archive <ID>...".into(),
        ));
    }

    let (_embedder, store, _blob_store) = open_store(data_dir).await?;

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
    let (_embedder, store, _blob_store) = open_store(data_dir).await?;

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
    let store = StoreBackend::open_metadata(data_dir, true).await?;

    // Get hot chunks (most accessed) as a proxy for "all interesting entries"
    let hot = store.get_hot_chunks(options.limit * 2).await?;

    if hot.is_empty() {
        println!("No entries to analyze.");
        return Ok(());
    }

    let weights = crate::salience::SalienceWeights::default();
    let top = crate::salience::top_salient(&hot, &weights, options.limit);

    println!(
        "{}",
        format!("Salience report (top {}):", top.len())
            .if_supports_color(Stream::Stdout, |s| s.bold())
    );
    println!(
        "{}",
        "=".repeat(60)
            .if_supports_color(Stream::Stdout, |s| s.dimmed())
    );
    for (idx, score) in &top {
        let chunk = &hot[*idx];
        println!(
            "  {} [{}] inter={:.2} persp={:.2} rev={:.2}  {}",
            short_id(&chunk.id).if_supports_color(Stream::Stdout, |s| s.cyan()),
            format!("{:.3}", score.composite).if_supports_color(Stream::Stdout, |s| s.green()),
            score.interaction,
            score.perspective,
            score.revision,
            preview(&chunk.content, 60).if_supports_color(Stream::Stdout, |s| s.dimmed())
        );
    }

    Ok(())
}

/// Archive candidates: entries with low salience that could be archived.
async fn compact_archive_candidates(data_dir: &Path, options: &CompactOptions) -> Result<()> {
    let store = StoreBackend::open_metadata(data_dir, true).await?;
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
        "{}",
        format!(
            "Archive candidates ({}, threshold {:.2}):",
            candidates.len(),
            options.archive_threshold
        )
        .if_supports_color(Stream::Stdout, |s| s.bold())
    );
    println!(
        "{}",
        "=".repeat(60)
            .if_supports_color(Stream::Stdout, |s| s.dimmed())
    );
    for chunk in &candidates {
        let score = crate::salience::compute(chunk, &weights);
        println!(
            "  {} [salience={}, vis={}]  {}",
            short_id(&chunk.id).if_supports_color(Stream::Stdout, |s| s.cyan()),
            format!("{:.3}", score.composite).if_supports_color(Stream::Stdout, |s| s.red()),
            vis_color(&chunk.visibility),
            preview(&chunk.content, 60).if_supports_color(Stream::Stdout, |s| s.dimmed())
        );
    }
    println!("\nUse `veclayer archive <id>...` to archive selected entries.");

    Ok(())
}

// --- Config command ---

/// Show resolved configuration for the current working directory.
pub fn show_config(
    cwd: &Path,
    user_config: &crate::config::UserConfig,
    resolved: &crate::config::ResolvedConfig,
    git_remote: Option<&str>,
) -> Result<()> {
    use owo_colors::OwoColorize;

    println!("Configuration for: {}", cwd.display().cyan());

    let cwd_str = cwd.to_str().unwrap_or("");

    println!("\n{}", "[User Config]".bold());
    println!("  Match overrides: {}", user_config.matches.len());
    for (idx, m) in user_config.matches.iter().enumerate() {
        let path_matched = m.path_matches(cwd_str);
        let remote_matched = m.remote_matches(git_remote);
        let any_matched = path_matched || remote_matched;

        println!(
            "  {}",
            if any_matched {
                format!("[{}]", idx).bold().to_string()
            } else {
                format!("[{}]", idx).dimmed().to_string()
            }
        );

        if let Some(ref pat) = m.path {
            let marker = if path_matched {
                " [matched]".green().to_string()
            } else {
                " [no match]".dimmed().to_string()
            };
            println!("    path=\"{}\"{}", pat.as_str(), marker);
        }

        if let Some(ref re) = m.git_remote {
            let marker = if remote_matched {
                " [matched]".green().to_string()
            } else {
                " [no match]".dimmed().to_string()
            };
            println!("    git-remote=/{}/{}", re.as_str(), marker);
        }

        if let Some(ref p) = m.project {
            println!("    → project: {}", p.yellow());
        }
        if let Some(ref d) = m.data_dir {
            println!("    → data_dir: {}", d.yellow());
        }
        if let Some(ref h) = m.host {
            println!("    → host: {}", h.yellow());
        }
        if let Some(p) = m.port {
            println!("    → port: {}", p.to_string().yellow());
        }
        if let Some(ro) = m.read_only {
            let val = if ro {
                "true".red().to_string()
            } else {
                "false".green().to_string()
            };
            println!("    → read_only: {}", val);
        }
    }

    println!("\n{}", "[Resolved Config]".bold());
    println!(
        "  project: {}",
        resolved.project.as_deref().unwrap_or("(none)").yellow()
    );
    println!(
        "  data_dir: {}",
        resolved.data_dir.as_deref().unwrap_or("(default)").cyan()
    );
    println!(
        "  host: {}",
        resolved.host.as_deref().unwrap_or("(default)").cyan()
    );
    println!(
        "  port: {}",
        resolved
            .port
            .map(|p| p.to_string())
            .unwrap_or_else(|| "(default)".to_string())
            .cyan()
    );
    println!(
        "  read_only: {}",
        match resolved.read_only {
            Some(true) => "true".red().to_string(),
            Some(false) => "false".green().to_string(),
            None => "(default)".to_string().dimmed().to_string(),
        }
    );

    Ok(())
}

// --- Reflect command ---

/// Generate a comprehensive reflection/identity report.
pub async fn reflect(data_dir: &Path) -> Result<()> {
    let store = StoreBackend::open_metadata(data_dir, true).await?;
    let snapshot = crate::identity::compute_identity(&store, data_dir, None, None).await?;
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
    let (embedder, store, blob_store) = open_store(data_dir).await?;

    let config = crate::Config::new().with_data_dir(data_dir);
    let llm = crate::llm::LlmBackend::from_config(&config.llm);

    println!(
        "Think: starting sleep cycle (LLM: {} via {})",
        config.llm.model, config.llm.provider
    );

    let result =
        crate::think::execute(&store, &embedder, &llm, data_dir, Some(&blob_store)).await?;

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
    let store = StoreBackend::open_metadata(data_dir, true).await?;

    let store_stats = store.stats().await?;
    if store_stats.total_chunks == 0 {
        println!("VecLayer is empty. Get started:");
        println!("  veclayer store \"Your first piece of knowledge\"");
        println!("  veclayer store ./notes/");
        println!("  veclayer recall \"What do I know about X?\"");
        return Ok(());
    }

    let snapshot = crate::identity::compute_identity(&store, data_dir, None, None).await?;

    println!(
        "{} {} entries from {} sources",
        "VecLayer".if_supports_color(Stream::Stdout, |s| s.bold()),
        store_stats
            .total_chunks
            .if_supports_color(Stream::Stdout, |s| s.bold()),
        store_stats.source_files.len()
    );

    // Perspective summary (one line)
    if !snapshot.centroids.is_empty() {
        let persp_summary: Vec<String> = snapshot
            .centroids
            .iter()
            .map(|c| {
                format!(
                    "{} ({})",
                    c.perspective
                        .if_supports_color(Stream::Stdout, |s| s.magenta()),
                    c.entry_count
                )
            })
            .collect();
        println!(
            "{} {}",
            "Perspectives:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
            persp_summary.join(", ")
        );
    }

    // Top 5 core
    if !snapshot.core_entries.is_empty() {
        println!(
            "\n{}",
            "Most important:".if_supports_color(Stream::Stdout, |s| s.bold())
        );
        for entry in snapshot.core_entries.iter().take(5) {
            let heading = entry.heading.as_deref().unwrap_or("(untitled)");
            println!(
                "  {} {}",
                short_id(&entry.id).if_supports_color(Stream::Stdout, |s| s.cyan()),
                heading
            );
        }
    }

    // Open threads (brief)
    if !snapshot.open_threads.is_empty() {
        println!(
            "\n{} Run `veclayer reflect` for details.",
            format!(
                "{} open thread(s) need attention.",
                snapshot.open_threads.len()
            )
            .if_supports_color(Stream::Stdout, |s| s.yellow())
        );
    }

    // Hints
    println!("\nTry: recall, reflect, think, reflect salience");

    Ok(())
}

// --- Browse command (#29) ---

/// Print one entry line: number + heading, compact metadata, content preview.
fn print_entry_line(index: usize, chunk: &crate::HierarchicalChunk) {
    let heading = chunk
        .heading
        .as_deref()
        .unwrap_or_else(|| chunk.content.lines().next().unwrap_or("(untitled)"));
    println!(
        "{}",
        format!("{}. {}", index + 1, heading).if_supports_color(Stream::Stdout, |s| s.bold())
    );

    let mut meta = vec![short_id(&chunk.id)
        .if_supports_color(Stream::Stdout, |s| s.cyan())
        .to_string()];
    if chunk.entry_type != EntryType::Raw {
        meta.push(
            chunk
                .entry_type
                .to_string()
                .if_supports_color(Stream::Stdout, |s| s.yellow())
                .to_string(),
        );
    }
    if !chunk.perspectives.is_empty() {
        meta.push(
            chunk
                .perspectives
                .join(", ")
                .if_supports_color(Stream::Stdout, |s| s.magenta())
                .to_string(),
        );
    }
    if chunk.visibility != "normal" {
        meta.push(vis_color(&chunk.visibility));
    }
    println!("   {}", meta.join(" | "));
    println!(
        "   {}",
        preview(&chunk.content, 200).if_supports_color(Stream::Stdout, |s| s.dimmed())
    );
}

/// Browse entries without vector search (list by perspective/recency).
pub async fn browse(data_dir: &Path, options: &SearchOptions) -> Result<()> {
    let since_epoch = options
        .since
        .as_deref()
        .and_then(crate::resolve::parse_temporal);
    let until_epoch = options
        .until
        .as_deref()
        .and_then(crate::resolve::parse_temporal);

    let store = StoreBackend::open_metadata(data_dir, true).await?;

    let open_thread_ids = crate::identity::resolve_ongoing_filter(&store, options.ongoing).await?;

    let fetch_limit = if open_thread_ids.is_some() {
        usize::MAX
    } else {
        options.top_k
    };

    let all_entries = store
        .list_entries(
            options.perspective.as_deref(),
            since_epoch,
            until_epoch,
            fetch_limit,
        )
        .await?;

    let entries: Vec<_> = all_entries
        .into_iter()
        .filter(|chunk| crate::identity::passes_ongoing_filter(&open_thread_ids, &chunk.id))
        .take(options.top_k)
        .collect();

    if entries.is_empty() {
        println!("No entries found.");
        return Ok(());
    }

    for (i, chunk) in entries.iter().enumerate() {
        if i > 0 {
            println!();
        }
        print_entry_line(i, chunk);
    }

    println!(
        "\n{} `veclayer focus <id>` to drill in.",
        format!("{} entry(ies).", entries.len()).if_supports_color(Stream::Stdout, |s| s.bold())
    );
    Ok(())
}

// --- Export / Import ---

/// Export all entries (or filtered by perspective) to JSONL on stdout.
///
/// Each line is a JSON-serialized `HierarchicalChunk` with the `embedding`
/// field stripped.  Output is sorted by `id` for deterministic, diffable output.
pub async fn export_entries(data_dir: &Path, options: &ExportOptions) -> Result<()> {
    let store = StoreBackend::open_metadata(data_dir, true).await?;
    let mut entries = store
        .list_entries(options.perspective.as_deref(), None, None, usize::MAX)
        .await?;

    entries.sort_by(|a, b| a.id.cmp(&b.id));

    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for chunk in &entries {
        let serializable = chunk.clone().without_embedding();
        let line = serde_json::to_string(&serializable)?;
        writeln!(out, "{}", line)?;
    }

    eprintln!("Exported {} entries.", entries.len());
    Ok(())
}

/// Import entries from a JSONL file (or stdin when path is "-").
///
/// Each line is deserialized to a `HierarchicalChunk`, re-embedded, and
/// inserted.  Entries whose `id` already exists in the store are skipped.
/// A single bad line is logged and skipped without aborting the whole import.
pub async fn import_entries(data_dir: &Path, options: &ImportOptions) -> Result<ImportResult> {
    let (embedder, store, blob_store) = open_store(data_dir).await?;

    let lines = read_jsonl_lines(&options.path)?;

    let mut imported = 0usize;
    let mut skipped = 0usize;

    for (line_number, line) in lines.into_iter().enumerate() {
        match import_one_entry(&embedder, &store, &blob_store, &line).await {
            Ok(true) => imported += 1,
            Ok(false) => skipped += 1,
            Err(e) => {
                warn!("Skipping line {}: {}", line_number + 1, e);
                skipped += 1;
            }
        }
    }

    eprintln!(
        "Imported {} entries, {} skipped (already exist).",
        imported, skipped
    );
    Ok(ImportResult { imported, skipped })
}

/// Rebuild the Lance vector index from the blob store.
///
/// Reads every blob, re-embeds if the cached embedding model doesn't match,
/// and inserts into a fresh Lance index.
pub async fn rebuild_index(data_dir: &Path) -> Result<()> {
    let blob_store = BlobStore::open(data_dir)?;
    let blob_count = blob_store.count()?;
    if blob_count == 0 {
        println!("No blobs found — nothing to rebuild.");
        return Ok(());
    }

    // Drop the existing Lance table so we start fresh (idempotent rebuild).
    let lance_table = data_dir.join(format!("{}.lance", crate::store::TABLE_NAME));
    if lance_table.exists() {
        std::fs::remove_dir_all(&lance_table)?;
        debug!("Removed existing Lance table at {:?}", lance_table);
    }

    let embedder = FastEmbedder::new()?;
    let dimension = embedder.dimension();
    let store = StoreBackend::open(data_dir, dimension, false).await?;

    let model_name = embedder.name();
    let mut count = 0;

    for hash_result in blob_store.iter_hashes() {
        let hash = hash_result?;
        if let Some(blob) = blob_store.get(&hash)? {
            let embedding = match blob.embedding_for_model(model_name) {
                Some(cached) => cached.to_vec(),
                None => {
                    let vecs = embedder.embed(&[blob.entry.content.as_str()])?;
                    vecs.into_iter().next().unwrap_or_default()
                }
            };
            let chunk = crate::HierarchicalChunk::from_entry(&blob.entry, embedding);
            store.insert_chunks(vec![chunk]).await?;
            count += 1;
        }
    }

    println!("Rebuilt index: {count} entries from blob store");
    Ok(())
}

/// Read all non-empty lines from a JSONL file path or stdin ("-").
fn read_jsonl_lines(path: &str) -> Result<Vec<String>> {
    if path == "-" {
        let reader = BufReader::new(io::stdin());
        collect_non_empty_lines(reader)
    } else {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);
        collect_non_empty_lines(reader)
    }
}

fn collect_non_empty_lines(reader: impl BufRead) -> Result<Vec<String>> {
    let lines = reader
        .lines()
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .filter(|l| !l.trim().is_empty())
        .collect();
    Ok(lines)
}

/// Attempt to import one JSONL line.  Returns `Ok(true)` if inserted,
/// `Ok(false)` if the entry already exists.
async fn import_one_entry(
    embedder: &impl crate::Embedder,
    store: &impl crate::store::VectorStore,
    blob_store: &BlobStore,
    line: &str,
) -> Result<bool> {
    let mut chunk: crate::HierarchicalChunk = serde_json::from_str(line)?;

    if store.get_by_id(&chunk.id).await?.is_some() {
        return Ok(false);
    }

    let embeddings = embedder.embed(&[chunk.content.as_str()])?;
    chunk.embedding = embeddings.into_iter().next();

    // Persist to blob store
    let blob = crate::entry::StoredBlob::from_chunk_and_embedding(&chunk, embedder.name());
    blob_store.put(&blob)?;

    store.insert_chunks(vec![chunk]).await?;
    Ok(true)
}

// --- Think subcommands (CLI parity with MCP) ---

/// Set an entry's visibility and print a labeled confirmation.
async fn set_visibility(data_dir: &Path, id: &str, visibility: &str, label: &str) -> Result<()> {
    let store = StoreBackend::open_metadata(data_dir, false).await?;
    let store = std::sync::Arc::new(store);
    let chunk_id = crate::resolve::resolve_id(&store, id).await?;
    store.update_visibility(&chunk_id, visibility).await?;
    println!(
        "{} {} to visibility '{}'",
        label,
        short_id(&chunk_id),
        visibility
    );
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
    let store = StoreBackend::open_metadata(data_dir, false).await?;
    let store = std::sync::Arc::new(store);
    let source_id = crate::resolve::resolve_id(&store, source).await?;
    let target_id = crate::resolve::resolve_id(&store, target).await?;

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
    let store = StoreBackend::open_metadata(data_dir, false).await?;
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

/// Discover similar-but-unlinked entries (CLI for MCP think(discover)).
pub async fn think_discover(data_dir: &Path, limit: usize) -> Result<()> {
    let store = StoreBackend::open_metadata(data_dir, true).await?;
    let store = std::sync::Arc::new(store);
    let blob_store = std::sync::Arc::new(BlobStore::open(data_dir)?);

    let input = crate::mcp::types::ThinkInput {
        action: Some("discover".to_string()),
        hot_limit: Some(limit),
        stale_limit: None,
        id: None,
        visibility: None,
        source_id: None,
        target_id: None,
        kind: None,
        degrade_after_days: None,
        degrade_to: None,
        degrade_from: None,
    };

    let report =
        crate::mcp::tools::execute_think(&store, data_dir, &blob_store, input, None, None).await?;
    println!("{}", report);
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
/// Delegates to `resolve::resolve_entry`.
async fn resolve_entry(store: &impl VectorStore, id: &str) -> Result<crate::HierarchicalChunk> {
    crate::resolve::resolve_entry(store, id).await
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

// --- Merge ---

/// Merge blobs from a source store into the target store.
///
/// Copies content-addressed blobs, optionally tagging them with a project
/// perspective, merges custom perspectives, and rebuilds the Lance index.
pub async fn merge(data_dir: &Path, source: &Path, options: &MergeOptions) -> Result<()> {
    // Step 1: Validate source
    if same_directory(source, data_dir) {
        return Err(crate::Error::config(
            "Source and target are the same directory".to_string(),
        ));
    }

    let source_objects = source.join("objects");
    if !source_objects.is_dir() {
        return Err(crate::Error::config(format!(
            "Source has no objects/ directory: {}",
            source.display()
        )));
    }
    let source_blob_store = BlobStore::open(source)?;
    let source_count = source_blob_store.count()?;
    if source_count == 0 {
        println!("Source store is empty — nothing to merge.");
        return Ok(());
    }

    // Step 2: Resolve project scope
    let cwd = std::env::current_dir().unwrap_or_else(|_| data_dir.to_path_buf());
    let git_info = crate::git_detect::detect(&cwd);
    let user_config = crate::config::UserConfig::discover();
    let resolved = user_config.resolve(&cwd, git_info.remote.as_deref());

    let project = resolve_merge_project(
        source,
        &options.project,
        &resolved,
        git_info.remote.as_deref(),
    );

    if project.is_none() && !options.force {
        eprintln!("Warning: No project scope detected for merged entries.");
        eprintln!("They will be unscoped — invisible when serving with --project.\n");
        eprintln!("Suggestions:");
        eprintln!("  veclayer merge {} -p <project-name>\n", source.display());
        eprintln!("  Or add a [[match]] to ~/.config/veclayer/config.toml:");
        eprintln!("    [[match]]");
        eprintln!("    git-remote = \"org/repo\"");
        eprintln!("    project = \"my-project\"\n");
        eprintln!("Use --force to merge without project scope.");
        return Err(crate::Error::config(
            "No project scope — use -p or --force".to_string(),
        ));
    }

    let project_tag = project.as_deref().map(|p| format!("project:{p}"));

    // Step 3: Copy blobs
    let (new_count, dup_count) =
        copy_blobs(&source_blob_store, data_dir, options.dry_run, &project_tag)?;

    // Step 4: Merge perspectives
    let (persp_added, _persp_skipped) = if options.dry_run {
        (0, 0)
    } else {
        crate::perspective::merge_from(data_dir, source)?
    };

    // Step 5: Rebuild index
    if !options.dry_run && new_count > 0 {
        rebuild_index(data_dir).await?;
    }

    // Step 6: Update user config (when -p given and no existing match covers source)
    if !options.dry_run {
        if let Some(ref proj) = options.project {
            if let Err(e) =
                try_update_user_config(source, proj, &user_config, git_info.remote.as_deref())
            {
                warn!("Could not update user config: {}", e);
            }
        }
    }

    // Step 7: Summary
    let label = if options.dry_run { " (dry run)" } else { "" };
    let project_label = match project.as_deref() {
        Some(p) => format!(" (project: {p})"),
        None => String::new(),
    };

    println!("Merged: {new_count} new{project_label}, {dup_count} skipped{label}.");

    if persp_added > 0 {
        println!("Perspectives: {persp_added} added.");
    }

    Ok(())
}

/// Check if two paths refer to the same directory (canonicalized).
fn same_directory(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(ca), Ok(cb)) => ca == cb,
        _ => false,
    }
}

/// Copy blobs from source to target, optionally adding a project tag.
/// Returns `(new, duplicates)`.
fn copy_blobs(
    source: &BlobStore,
    target_dir: &Path,
    dry_run: bool,
    project_tag: &Option<String>,
) -> Result<(usize, usize)> {
    let target = BlobStore::open(target_dir)?;
    let mut new_count = 0usize;
    let mut dup_count = 0usize;

    for hash_result in source.iter_hashes() {
        let hash = hash_result?;

        if target.has(&hash) {
            dup_count += 1;
            continue;
        }

        if dry_run {
            new_count += 1;
            continue;
        }

        let Some(mut blob) = source.get(&hash)? else {
            warn!("Source blob missing for hash {}", hash.to_hex());
            continue;
        };

        if let Some(ref tag) = project_tag {
            if !blob.entry.perspectives.contains(tag) {
                blob.entry.perspectives.push(tag.clone());
            }
        }

        target.put(&blob)?;
        new_count += 1;
    }

    Ok((new_count, dup_count))
}

/// Resolve project scope for merge: flag → source config → target resolved → git remote.
fn resolve_merge_project(
    source: &Path,
    flag_project: &Option<String>,
    resolved: &crate::config::ResolvedConfig,
    git_remote: Option<&str>,
) -> Option<String> {
    // 1. Explicit -p flag
    if let Some(p) = flag_project {
        return Some(p.clone());
    }

    // 2. Source's .veclayer/config.toml project field
    if source.join("config.toml").exists() {
        let project = crate::config::discover_project(source.parent().unwrap_or(source))
            .and_then(|(_, pc)| pc.project);
        if project.is_some() {
            return project;
        }
    }

    // 3. Target's resolved project (already computed by caller)
    if resolved.project.is_some() {
        return resolved.project.clone();
    }

    // 4. Git remote as fallback
    git_remote.map(String::from)
}

/// Append a `[[match]]` entry to user config if no existing match covers the source.
/// Canonicalizes the source path before writing the glob pattern.
fn try_update_user_config(
    source: &Path,
    project: &str,
    user_config: &crate::config::UserConfig,
    git_remote: Option<&str>,
) -> Result<()> {
    let source_canonical = source
        .parent()
        .unwrap_or(source)
        .canonicalize()
        .unwrap_or_else(|_| source.parent().unwrap_or(source).to_path_buf());

    let cwd_str = source_canonical.to_str().unwrap_or("");
    let already_matched = user_config
        .matches
        .iter()
        .any(|m| m.matches(cwd_str, git_remote) && m.project.as_deref() == Some(project));

    if already_matched {
        return Ok(());
    }

    let path_glob = source_canonical.to_str().map(|p| format!("{p}*"));
    let path_glob_ref = path_glob.as_deref();

    if git_remote.is_none() && path_glob_ref.is_none() {
        return Ok(());
    }

    let config_path =
        crate::config::append_match_to_user_config(git_remote, path_glob_ref, project)?;

    println!("\nUpdated {}:\n", config_path.display());
    println!("  [[match]]");
    if let Some(remote) = git_remote {
        println!("  git-remote = \"{}\"", remote);
    }
    if let Some(glob) = path_glob_ref {
        println!("  path = \"{}\"", glob);
    }
    println!("  project = \"{}\"\n", project);
    println!("Verify with: veclayer config");

    Ok(())
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
        assert!(opts.rel_supersedes.is_empty());
        assert!(opts.rel_summarizes.is_empty());
        assert!(opts.rel_to.is_empty());
        assert!(opts.rel_derived_from.is_empty());
        assert!(opts.rel_version_of.is_empty());
        assert!(opts.rel_custom.is_empty());
        assert!(opts.impression_hint.is_none());
        assert!((opts.impression_strength - 1.0).abs() < f32::EPSILON);
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

    // --- Test infrastructure ---

    use crate::test_helpers::make_test_chunk;

    async fn seed_store(dir: &Path) -> StoreBackend {
        let store = StoreBackend::open(dir, 384, false).await.unwrap();
        store
            .insert_chunks(vec![
                make_test_chunk("aaa111", "First entry about architecture"),
                make_test_chunk("bbb222", "Second entry about testing"),
            ])
            .await
            .unwrap();
        store
    }

    // --- think_promote tests ---

    #[tokio::test]
    async fn test_think_promote_changes_visibility() -> Result<()> {
        let dir = TempDir::new()?;
        seed_store(dir.path()).await;

        think_promote(dir.path(), "aaa111", "always").await?;

        let store = StoreBackend::open_metadata(dir.path(), false).await?;
        let entry = store.get_by_id("aaa111").await?.unwrap();
        assert_eq!(entry.visibility, "always");
        Ok(())
    }

    #[tokio::test]
    async fn test_think_promote_resolves_prefix() -> Result<()> {
        let dir = TempDir::new()?;
        seed_store(dir.path()).await;

        think_promote(dir.path(), "aaa", "always").await?;

        let store = StoreBackend::open_metadata(dir.path(), false).await?;
        let entry = store.get_by_id("aaa111").await?.unwrap();
        assert_eq!(entry.visibility, "always");
        Ok(())
    }

    #[tokio::test]
    async fn test_think_promote_not_found() {
        let dir = TempDir::new().unwrap();
        seed_store(dir.path()).await;

        let result = think_promote(dir.path(), "zzz999", "always").await;
        assert!(result.is_err());
    }

    // --- think_demote tests ---

    #[tokio::test]
    async fn test_think_demote_changes_visibility() -> Result<()> {
        let dir = TempDir::new()?;
        seed_store(dir.path()).await;

        think_demote(dir.path(), "aaa111", "deep_only").await?;

        let store = StoreBackend::open_metadata(dir.path(), false).await?;
        let entry = store.get_by_id("aaa111").await?.unwrap();
        assert_eq!(entry.visibility, "deep_only");
        Ok(())
    }

    // --- think_relate tests ---

    #[tokio::test]
    async fn test_think_relate_adds_forward_relation() -> Result<()> {
        let dir = TempDir::new()?;
        seed_store(dir.path()).await;

        think_relate(dir.path(), "aaa111", "bbb222", "derived_from").await?;

        let store = StoreBackend::open_metadata(dir.path(), false).await?;
        let source = store.get_by_id("aaa111").await?.unwrap();
        assert_eq!(source.relations.len(), 1);
        assert_eq!(source.relations[0].kind, "derived_from");
        assert_eq!(source.relations[0].target_id, "bbb222");
        Ok(())
    }

    #[tokio::test]
    async fn test_think_relate_bidirectional_for_related_to() -> Result<()> {
        let dir = TempDir::new()?;
        seed_store(dir.path()).await;

        think_relate(dir.path(), "aaa111", "bbb222", "related_to").await?;

        let store = StoreBackend::open_metadata(dir.path(), false).await?;
        let source = store.get_by_id("aaa111").await?.unwrap();
        assert_eq!(source.relations.len(), 1);
        assert_eq!(source.relations[0].kind, "related_to");
        assert_eq!(source.relations[0].target_id, "bbb222");

        // Backward link
        let target = store.get_by_id("bbb222").await?.unwrap();
        assert_eq!(target.relations.len(), 1);
        assert_eq!(target.relations[0].kind, "related_to");
        assert_eq!(target.relations[0].target_id, "aaa111");
        Ok(())
    }

    #[tokio::test]
    async fn test_think_relate_no_backward_for_derived_from() -> Result<()> {
        let dir = TempDir::new()?;
        seed_store(dir.path()).await;

        think_relate(dir.path(), "aaa111", "bbb222", "derived_from").await?;

        let store = StoreBackend::open_metadata(dir.path(), false).await?;
        let target = store.get_by_id("bbb222").await?.unwrap();
        assert!(
            target.relations.is_empty(),
            "non-related_to should not add backward link"
        );
        Ok(())
    }

    // --- think_aging tests ---

    #[tokio::test]
    async fn test_think_aging_apply_empty_store() -> Result<()> {
        let dir = TempDir::new()?;
        // No seeding — empty store
        StoreBackend::open_metadata(dir.path(), false).await?;
        think_aging_apply(dir.path()).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_think_aging_configure_saves() -> Result<()> {
        let dir = TempDir::new()?;
        std::fs::create_dir_all(dir.path())?;

        think_aging_configure(dir.path(), Some(7), Some("archived")).await?;

        let config = crate::aging::AgingConfig::load(dir.path());
        assert_eq!(config.degrade_after_days, 7);
        assert_eq!(config.degrade_to, "archived");
        Ok(())
    }

    #[tokio::test]
    async fn test_think_aging_configure_partial_update() -> Result<()> {
        let dir = TempDir::new()?;
        std::fs::create_dir_all(dir.path())?;

        // Set initial
        think_aging_configure(dir.path(), Some(14), Some("deep_only")).await?;
        // Update only days
        think_aging_configure(dir.path(), Some(3), None).await?;

        let config = crate::aging::AgingConfig::load(dir.path());
        assert_eq!(config.degrade_after_days, 3);
        assert_eq!(config.degrade_to, "deep_only"); // unchanged
        Ok(())
    }

    // --- browse tests ---

    #[tokio::test]
    async fn test_browse_empty_store() -> Result<()> {
        let dir = TempDir::new()?;
        StoreBackend::open_metadata(dir.path(), false).await?;
        browse(dir.path(), &SearchOptions::default()).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_browse_returns_entries() -> Result<()> {
        let dir = TempDir::new()?;
        seed_store(dir.path()).await;

        // browse prints to stdout — verify it doesn't error
        browse(dir.path(), &SearchOptions::default()).await?;
        Ok(())
    }

    // --- preview tests ---

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
        // "日本語" is 9 bytes (3 bytes per char); truncating at 5 should not panic
        let result = preview("日本語テスト", 5);
        assert!(result.ends_with("..."));
        // Should cut at a valid char boundary (either 3 or 6 bytes, not 5)
        assert!(result.len() <= 9); // at most 3 bytes + "..."
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

    // --- export / import tests ---

    #[test]
    fn test_export_options_default() {
        let opts = ExportOptions::default();
        assert!(opts.perspective.is_none());
    }

    #[test]
    fn test_import_options_default() {
        let opts = ImportOptions::default();
        assert_eq!(opts.path, "");
    }

    #[tokio::test]
    async fn test_export_empty_store() -> Result<()> {
        let dir = TempDir::new()?;
        StoreBackend::open_metadata(dir.path(), false).await?;

        // export should succeed with an empty store without error
        let opts = ExportOptions::default();
        export_entries(dir.path(), &opts).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_import_skips_existing_entries() -> Result<()> {
        let dir = TempDir::new()?;
        seed_store(dir.path()).await;

        // Build JSONL from existing entries (open read-only to avoid lock conflict)
        let jsonl_file = dir.path().join("export.jsonl");
        let entry_count = {
            let store = StoreBackend::open_metadata(dir.path(), true).await?;
            let entries = store.list_entries(None, None, None, usize::MAX).await?;
            let jsonl: String = entries
                .iter()
                .map(|c| serde_json::to_string(&c.clone().without_embedding()).unwrap() + "\n")
                .collect();
            fs::write(&jsonl_file, &jsonl)?;
            entries.len()
        }; // store dropped here, releasing its (read-only) handle

        let opts = ImportOptions {
            path: jsonl_file.to_string_lossy().to_string(),
        };
        let result = import_entries(dir.path(), &opts).await?;

        assert_eq!(result.imported, 0, "all entries already exist");
        assert_eq!(result.skipped, entry_count);
        Ok(())
    }

    #[tokio::test]
    async fn test_export_import_roundtrip() -> Result<()> {
        let source_dir = TempDir::new()?;
        let target_dir = TempDir::new()?;
        let jsonl_file = source_dir.path().join("roundtrip.jsonl");

        // Insert two entries and build JSONL in a scoped block so the lock is released
        {
            let store = StoreBackend::open(source_dir.path(), 384, false).await?;
            store
                .insert_chunks(vec![
                    make_test_chunk("export001", "Export roundtrip entry one"),
                    make_test_chunk("export002", "Export roundtrip entry two"),
                ])
                .await?;

            let entries = store.list_entries(None, None, None, usize::MAX).await?;
            let mut sorted = entries.clone();
            sorted.sort_by(|a, b| a.id.cmp(&b.id));
            let jsonl: String = sorted
                .iter()
                .map(|c| serde_json::to_string(&c.clone().without_embedding()).unwrap() + "\n")
                .collect();
            fs::write(&jsonl_file, &jsonl)?;
        } // store + lock dropped here

        // Import into fresh store
        let opts = ImportOptions {
            path: jsonl_file.to_string_lossy().to_string(),
        };
        let result = import_entries(target_dir.path(), &opts).await?;

        assert_eq!(result.imported, 2);
        assert_eq!(result.skipped, 0);

        // Verify entries are present in target store
        {
            let target_store = StoreBackend::open_metadata(target_dir.path(), true).await?;
            let imported_entries = target_store
                .list_entries(None, None, None, usize::MAX)
                .await?;
            assert_eq!(imported_entries.len(), 2);
        } // target_store dropped here

        // Verify idempotency: importing again should skip all
        let result2 = import_entries(target_dir.path(), &opts).await?;
        assert_eq!(result2.imported, 0);
        assert_eq!(result2.skipped, 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_import_skips_bad_lines() -> Result<()> {
        let dir = TempDir::new()?;

        // Write JSONL with one valid and one invalid line
        let valid_chunk = make_test_chunk("badline001", "Valid entry for bad line test");
        let valid_json = serde_json::to_string(&valid_chunk.clone().without_embedding()).unwrap();
        let jsonl_content = format!("{}\n{{invalid json}}\n", valid_json);

        let jsonl_file = dir.path().join("bad_lines.jsonl");
        fs::write(&jsonl_file, &jsonl_content)?;

        let opts = ImportOptions {
            path: jsonl_file.to_string_lossy().to_string(),
        };
        let result = import_entries(dir.path(), &opts).await?;

        // The valid line should import, the bad line should be skipped
        assert_eq!(result.imported, 1);
        assert_eq!(result.skipped, 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_export_perspective_filter() -> Result<()> {
        let dir = TempDir::new()?;
        crate::perspective::init(dir.path())?;

        // Insert entries with and without a perspective
        let store = StoreBackend::open(dir.path(), 384, false).await?;
        let mut chunk_with_perspective =
            make_test_chunk("persp001", "Entry with decisions perspective");
        chunk_with_perspective.perspectives = vec!["decisions".to_string()];
        let chunk_no_perspective = make_test_chunk("persp002", "Entry without perspective");
        store
            .insert_chunks(vec![chunk_with_perspective, chunk_no_perspective])
            .await?;

        // Export with perspective filter — should not error
        let opts = ExportOptions {
            perspective: Some("decisions".to_string()),
        };
        export_entries(dir.path(), &opts).await?;
        Ok(())
    }

    // --- merge tests ---

    fn test_blob(content: &str) -> crate::entry::StoredBlob {
        crate::entry::StoredBlob {
            entry: crate::entry::Entry {
                content: content.to_string(),
                entry_type: crate::chunk::EntryType::Raw,
                source: "test".to_string(),
                created_at: 0,
                perspectives: vec![],
                relations: vec![],
                summarizes: vec![],
                heading: None,
                parent_id: None,
                impression_hint: None,
                impression_strength: 1.0,
                expires_at: None,
                visibility: "normal".to_string(),
                level: crate::chunk::ChunkLevel(7),
                path: String::new(),
            },
            embeddings: vec![crate::entry::EmbeddingCache {
                model: "BAAI/bge-small-en-v1.5".to_string(),
                dimensions: 384,
                vector: vec![0.0f32; 384],
            }],
        }
    }

    fn seed_blob_store(dir: &Path, contents: &[&str]) -> BlobStore {
        let store = BlobStore::open(dir).unwrap();
        for content in contents {
            store.put(&test_blob(content)).unwrap();
        }
        store
    }

    #[test]
    fn test_same_directory_detects_identical_paths() {
        let dir = TempDir::new().unwrap();
        assert!(same_directory(dir.path(), dir.path()));
    }

    #[test]
    fn test_same_directory_false_for_different_paths() {
        let a = TempDir::new().unwrap();
        let b = TempDir::new().unwrap();
        assert!(!same_directory(a.path(), b.path()));
    }

    #[tokio::test]
    async fn test_merge_rejects_same_directory() {
        let dir = TempDir::new().unwrap();
        seed_blob_store(dir.path(), &["entry one"]);

        let opts = MergeOptions {
            force: true,
            ..Default::default()
        };
        let result = merge(dir.path(), dir.path(), &opts).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("same directory"),
            "Expected 'same directory' error, got: {err}"
        );
    }

    #[tokio::test]
    async fn test_merge_rejects_missing_objects() {
        let source = TempDir::new().unwrap();
        let target = TempDir::new().unwrap();
        // source has no objects/ dir

        let opts = MergeOptions {
            force: true,
            ..Default::default()
        };
        let result = merge(target.path(), source.path(), &opts).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("objects/"));
    }

    #[tokio::test]
    async fn test_merge_empty_source() -> Result<()> {
        let source = TempDir::new().unwrap();
        let target = TempDir::new().unwrap();
        // Create empty objects dir
        fs::create_dir_all(source.path().join("objects")).unwrap();

        let opts = MergeOptions {
            force: true,
            ..Default::default()
        };
        merge(target.path(), source.path(), &opts).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_merge_copies_blobs_with_force() -> Result<()> {
        let source = TempDir::new().unwrap();
        let target = TempDir::new().unwrap();

        seed_blob_store(source.path(), &["alpha", "beta"]);

        let opts = MergeOptions {
            force: true,
            ..Default::default()
        };
        merge(target.path(), source.path(), &opts).await?;

        let target_store = BlobStore::open(target.path())?;
        assert_eq!(target_store.count()?, 2);
        Ok(())
    }

    #[tokio::test]
    async fn test_merge_skips_duplicates() -> Result<()> {
        let source = TempDir::new().unwrap();
        let target = TempDir::new().unwrap();

        // Same blob in both
        seed_blob_store(source.path(), &["shared", "unique"]);
        seed_blob_store(target.path(), &["shared"]);

        let opts = MergeOptions {
            force: true,
            ..Default::default()
        };
        merge(target.path(), source.path(), &opts).await?;

        let target_store = BlobStore::open(target.path())?;
        assert_eq!(target_store.count()?, 2); // shared + unique
        Ok(())
    }

    #[tokio::test]
    async fn test_merge_dry_run_does_not_write() -> Result<()> {
        let source = TempDir::new().unwrap();
        let target = TempDir::new().unwrap();

        seed_blob_store(source.path(), &["alpha", "beta"]);

        let opts = MergeOptions {
            force: true,
            dry_run: true,
            ..Default::default()
        };
        merge(target.path(), source.path(), &opts).await?;

        let target_store = BlobStore::open(target.path())?;
        assert_eq!(target_store.count()?, 0, "dry run should not write blobs");
        Ok(())
    }

    #[tokio::test]
    async fn test_merge_tags_with_project() -> Result<()> {
        let source = TempDir::new().unwrap();
        let target = TempDir::new().unwrap();

        seed_blob_store(source.path(), &["entry with project tag"]);

        let opts = MergeOptions {
            project: Some("myproject".to_string()),
            force: false,
            ..Default::default()
        };
        merge(target.path(), source.path(), &opts).await?;

        // Read back the blob and check perspectives
        let target_store = BlobStore::open(target.path())?;
        let hash = target_store.iter_hashes().next().unwrap()?;
        let blob = target_store.get(&hash)?.unwrap();
        assert!(
            blob.entry
                .perspectives
                .contains(&"project:myproject".to_string()),
            "Expected project:myproject in perspectives, got: {:?}",
            blob.entry.perspectives
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_merge_warns_without_project_scope() {
        let source = TempDir::new().unwrap();
        let target = TempDir::new().unwrap();

        seed_blob_store(source.path(), &["entry"]);

        // No project, no force → should fail with warning
        let opts = MergeOptions::default();
        let result = merge(target.path(), source.path(), &opts).await;

        // May succeed if git auto-detection provides a project. If it fails,
        // it should be the "No project scope" error.
        if let Err(e) = result {
            assert!(
                e.to_string().contains("project scope"),
                "Expected project scope error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_resolve_merge_project_flag_wins() {
        let resolved = crate::config::ResolvedConfig::default();
        let project = resolve_merge_project(
            Path::new("/tmp"),
            &Some("explicit".to_string()),
            &resolved,
            Some("github.com/org/repo"),
        );
        assert_eq!(project.as_deref(), Some("explicit"));
    }

    #[test]
    fn test_resolve_merge_project_falls_back_to_resolved() {
        let resolved = crate::config::ResolvedConfig {
            project: Some("from-config".to_string()),
            ..Default::default()
        };
        let project = resolve_merge_project(Path::new("/tmp"), &None, &resolved, None);
        assert_eq!(project.as_deref(), Some("from-config"));
    }

    #[test]
    fn test_resolve_merge_project_falls_back_to_git_remote() {
        let resolved = crate::config::ResolvedConfig::default();
        let project = resolve_merge_project(
            Path::new("/tmp"),
            &None,
            &resolved,
            Some("github.com/org/repo"),
        );
        assert_eq!(project.as_deref(), Some("github.com/org/repo"));
    }

    #[test]
    fn test_resolve_merge_project_none_when_nothing_available() {
        let resolved = crate::config::ResolvedConfig::default();
        let project = resolve_merge_project(Path::new("/tmp"), &None, &resolved, None);
        assert!(project.is_none());
    }

    #[test]
    fn test_copy_blobs_basic() -> Result<()> {
        let source = TempDir::new().unwrap();
        let target = TempDir::new().unwrap();

        let source_store = BlobStore::open(source.path())?;
        source_store.put(&test_blob("one"))?;
        source_store.put(&test_blob("two"))?;

        let (new, dup) = copy_blobs(&source_store, target.path(), false, &None)?;
        assert_eq!(new, 2);
        assert_eq!(dup, 0);

        let target_store = BlobStore::open(target.path())?;
        assert_eq!(target_store.count()?, 2);
        Ok(())
    }

    #[test]
    fn test_copy_blobs_with_tag() -> Result<()> {
        let source = TempDir::new().unwrap();
        let target = TempDir::new().unwrap();

        let source_store = BlobStore::open(source.path())?;
        source_store.put(&test_blob("tagged entry"))?;

        let tag = Some("project:test".to_string());
        let (new, _dup) = copy_blobs(&source_store, target.path(), false, &tag)?;
        assert_eq!(new, 1);

        let target_store = BlobStore::open(target.path())?;
        let hash = target_store.iter_hashes().next().unwrap()?;
        let blob = target_store.get(&hash)?.unwrap();
        assert!(blob
            .entry
            .perspectives
            .contains(&"project:test".to_string()));
        Ok(())
    }

    #[test]
    fn test_copy_blobs_dry_run() -> Result<()> {
        let source = TempDir::new().unwrap();
        let target = TempDir::new().unwrap();

        let source_store = BlobStore::open(source.path())?;
        source_store.put(&test_blob("dry"))?;

        let (new, dup) = copy_blobs(&source_store, target.path(), true, &None)?;
        assert_eq!(new, 1);
        assert_eq!(dup, 0);

        let target_store = BlobStore::open(target.path())?;
        assert_eq!(target_store.count()?, 0);
        Ok(())
    }

    #[test]
    fn test_copy_blobs_deduplicates() -> Result<()> {
        let source = TempDir::new().unwrap();
        let target = TempDir::new().unwrap();

        let source_store = BlobStore::open(source.path())?;
        source_store.put(&test_blob("shared"))?;
        source_store.put(&test_blob("only-in-source"))?;

        // Pre-populate target with the shared blob
        let target_store = BlobStore::open(target.path())?;
        target_store.put(&test_blob("shared"))?;
        drop(target_store);

        let (new, dup) = copy_blobs(&source_store, target.path(), false, &None)?;
        assert_eq!(new, 1);
        assert_eq!(dup, 1);
        Ok(())
    }
}
