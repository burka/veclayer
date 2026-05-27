//! Add/ingest commands and related helpers.

use super::*;
use crate::parser::DocumentParser;

/// Backwards-compatible alias
pub async fn ingest(data_dir: &Path, path: &Path, options: &AddOptions) -> Result<AddResult> {
    add_files(data_dir, path, options, None).await
}

/// Add knowledge to the store (files, directories, or inline text).
pub async fn add(
    data_dir: &Path,
    input: &str,
    mut options: AddOptions,
    git_store: Option<&crate::git::memory_store::MemoryStore>,
) -> Result<AddResult> {
    options.perspectives = options
        .perspectives
        .iter()
        .flat_map(|p| p.split(',').map(|s| s.trim().to_string()))
        .filter(|s| !s.is_empty())
        .collect();

    if !options.perspectives.is_empty() {
        crate::perspective::validate_ids(data_dir, &options.perspectives)?;
    }

    let input_path = Path::new(input);

    if input_path.exists() {
        add_files(data_dir, input_path, &options, git_store).await
    } else {
        add_text(data_dir, input, &options, git_store).await
    }
}

/// Add files from a path to the store.
async fn add_files(
    data_dir: &Path,
    path: &Path,
    options: &AddOptions,
    git_store: Option<&crate::git::memory_store::MemoryStore>,
) -> Result<AddResult> {
    debug!("Opening store at {:?}...", data_dir);
    let (_config, embedder, store, blob_store) = super::open_store(data_dir).await?;

    let parser = MarkdownParser::new();

    let files = collect_files(path, options.recursive, options.follow_links, &parser)?;
    debug!("Found {} files to process", files.len());

    let mut all_chunks = Vec::new();
    let mut git_warnings: Vec<String> = Vec::new();

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
        let embeddings = embedder.embed(&texts).await?;

        for (chunk, embedding) in chunks.iter_mut().zip(embeddings) {
            chunk.embedding = Some(embedding);
        }

        for chunk in &chunks {
            let blob = crate::entry::StoredBlob::from_chunk_and_embedding(chunk, embedder.name());
            blob_store.put(&blob)?;
        }

        store.insert_chunks(chunks.clone()).await?;
        debug!("  Indexed successfully");

        if let Some(gs) = git_store {
            for chunk in &chunks {
                let entry = crate::entry::Entry::from_chunk(chunk);
                if let Err(e) = gs.store_entry(&entry) {
                    git_warnings.push(format!("failed to stage entry in git: {e}"));
                } else if let Some(emb) = chunk.embedding.as_deref() {
                    if let Err(e) = gs.store_embedding(&entry, embedder.name(), emb) {
                        git_warnings.push(format!("failed to cache embedding in git: {e}"));
                    }
                }
            }
        }

        all_chunks.extend(chunks);
    }

    let total_entries = all_chunks.len();
    #[allow(unused_mut)]
    let mut summary_entries = 0;

    #[cfg(feature = "llm")]
    if options.summarize && !all_chunks.is_empty() {
        info!(
            "Starting cluster summarization with model '{}'...",
            options.model
        );

        let summary_embedder = crate::embedder::from_config(&_config.embedder)?;
        let summarizer = OllamaSummarizer::new()?.with_model(&options.model);

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
        git_warnings,
    })
}

/// Add inline text as a single entry.
async fn add_text(
    data_dir: &Path,
    text: &str,
    options: &AddOptions,
    git_store: Option<&crate::git::memory_store::MemoryStore>,
) -> Result<AddResult> {
    let (_config, embedder, store, blob_store) = super::open_store(data_dir).await?;

    let entry_type: crate::chunk::EntryType = options.entry_type.parse().map_err(|_| {
        crate::Error::parse(format!("invalid entry type: '{}'", options.entry_type))
    })?;

    let (level, path, resolved_parent_id) = if let Some(ref pid) = options.parent_id {
        let parent = resolve_entry(&store, pid).await?;
        (
            parent.level.child(),
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

    let embeddings = embedder.embed(&[text]).await?;
    chunk.embedding = Some(
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| crate::Error::embedding("Failed to generate embedding"))?,
    );

    let blob = crate::entry::StoredBlob::from_chunk_and_embedding(&chunk, embedder.name());
    blob_store.put(&blob)?;

    let id = chunk.id.clone();
    let git_entry = crate::entry::Entry::from_chunk(&chunk);
    let git_embedding = chunk.embedding.clone();
    let embedder_name = embedder.name().to_string();
    let store = std::sync::Arc::new(store);
    store.insert_chunks(vec![chunk]).await?;

    let mut git_warnings: Vec<String> = Vec::new();
    if let Some(gs) = git_store {
        if let Err(e) = gs.store_entry(&git_entry) {
            git_warnings.push(format!("failed to stage entry in git: {e}"));
        } else if let Some(emb) = git_embedding.as_deref() {
            if let Err(e) = gs.store_embedding(&git_entry, &embedder_name, emb) {
                git_warnings.push(format!("failed to cache embedding in git: {e}"));
            }
        }
    }

    let mut raw_relations = crate::relations::RawRelation::from_typed_options(
        &options.rel_supersedes,
        &options.rel_summarizes,
        &options.rel_to,
        &options.rel_derived_from,
        &options.rel_version_of,
    );
    raw_relations.extend(crate::relations::RawRelation::parse_custom(
        &options.rel_custom,
    )?);

    crate::relations::process_relations(&store, &id, raw_relations).await?;

    println!("Added entry {} ({})", short_id(&id), entry_type);

    Ok(AddResult {
        total_entries: 1,
        summary_entries: 0,
        files_processed: 0,
        git_warnings,
    })
}

/// Collect files from a path, optionally recursively.
pub fn collect_files(
    path: &Path,
    recursive: bool,
    follow_links: bool,
    parser: &impl DocumentParser,
) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    if path.is_file() {
        if parser.can_parse(path) {
            files.push(path.to_path_buf());
        } else {
            let supported = parser.supported_extensions().join(", .");
            eprintln!(
                "Skipped '{}': unsupported file type (supported: .{}). \
                 Use inline text instead: veclayer store \"$(cat '{}')\"",
                path.display(),
                supported,
                path.display()
            );
        }
    } else if path.is_dir() {
        if recursive {
            for entry in walkdir::WalkDir::new(path)
                .follow_links(follow_links)
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
        assert!(!opts.follow_links);
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
    fn test_collect_files_single_file() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("test.md");
        fs::write(&file_path, "# Test")?;

        let parser = MarkdownParser::new();
        let files = collect_files(&file_path, false, false, &parser)?;

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
        let files = collect_files(&file_path, false, false, &parser)?;

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
        let files = collect_files(temp_dir.path(), false, false, &parser)?;

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

        let files_non_recursive = collect_files(temp_dir.path(), false, false, &parser)?;
        assert_eq!(files_non_recursive.len(), 1);

        let files_recursive = collect_files(temp_dir.path(), true, false, &parser)?;
        assert_eq!(files_recursive.len(), 2);

        Ok(())
    }

    #[test]
    fn test_collect_files_empty_directory() -> Result<()> {
        let temp_dir = TempDir::new()?;

        let parser = MarkdownParser::new();
        let files = collect_files(temp_dir.path(), true, false, &parser)?;

        assert_eq!(files.len(), 0);

        Ok(())
    }

    // ── AddOptions construction ───────────────────────────────────────────────

    #[test]
    fn test_add_options_custom_fields() {
        let opts = AddOptions {
            recursive: false,
            follow_links: true,
            summarize: false,
            model: "mistral".to_string(),
            visibility: Some("deep_only".to_string()),
            entry_type: "meta".to_string(),
            perspectives: vec!["decisions".to_string()],
            parent_id: Some("abc123".to_string()),
            heading: Some("My heading".to_string()),
            impression_hint: Some("hint".to_string()),
            impression_strength: 0.5,
            rel_supersedes: vec!["old-id".to_string()],
            rel_summarizes: vec!["sum-id".to_string()],
            rel_to: vec!["related-id".to_string()],
            rel_derived_from: vec!["source-id".to_string()],
            rel_version_of: vec!["v0-id".to_string()],
            rel_custom: vec!["custom:target-id".to_string()],
        };
        assert!(!opts.recursive);
        assert!(opts.follow_links);
        assert!(!opts.summarize);
        assert_eq!(opts.model, "mistral");
        assert_eq!(opts.visibility.as_deref(), Some("deep_only"));
        assert_eq!(opts.entry_type, "meta");
        assert_eq!(opts.perspectives, vec!["decisions"]);
        assert_eq!(opts.parent_id.as_deref(), Some("abc123"));
        assert_eq!(opts.heading.as_deref(), Some("My heading"));
        assert_eq!(opts.impression_hint.as_deref(), Some("hint"));
        assert!((opts.impression_strength - 0.5).abs() < f32::EPSILON);
        assert_eq!(opts.rel_supersedes, vec!["old-id"]);
        assert_eq!(opts.rel_summarizes, vec!["sum-id"]);
        assert_eq!(opts.rel_to, vec!["related-id"]);
        assert_eq!(opts.rel_derived_from, vec!["source-id"]);
        assert_eq!(opts.rel_version_of, vec!["v0-id"]);
        assert_eq!(opts.rel_custom, vec!["custom:target-id"]);
    }

    // ── collect_files: nonexistent path ──────────────────────────────────────

    #[test]
    fn test_collect_files_nonexistent_path_is_not_file_or_dir() -> Result<()> {
        let parser = MarkdownParser::new();
        let nonexistent = std::path::Path::new("/tmp/__veclayer_does_not_exist_12345__");
        // Neither is_file() nor is_dir() — returns empty without error
        let files = collect_files(nonexistent, false, false, &parser)?;
        assert!(files.is_empty());
        Ok(())
    }

    // ── collect_files: multiple markdown extensions ───────────────────────────

    #[test]
    fn test_collect_files_filters_out_non_markdown_in_dir() -> Result<()> {
        let temp_dir = TempDir::new()?;
        fs::write(temp_dir.path().join("doc.md"), "# doc")?;
        fs::write(temp_dir.path().join("image.png"), [0u8; 4])?;
        fs::write(temp_dir.path().join("archive.zip"), [0u8; 4])?;
        fs::write(temp_dir.path().join("notes.txt"), "plain text")?;

        let parser = MarkdownParser::new();
        let files = collect_files(temp_dir.path(), false, false, &parser)?;

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].file_name().unwrap(), "doc.md");
        Ok(())
    }

    // ── collect_files: deeply nested recursive ────────────────────────────────

    #[test]
    fn test_collect_files_recursive_deep_nesting() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let deep = temp_dir.path().join("a").join("b").join("c");
        fs::create_dir_all(&deep)?;
        fs::write(deep.join("deep.md"), "# Deep")?;

        let parser = MarkdownParser::new();
        let files = collect_files(temp_dir.path(), true, false, &parser)?;

        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("deep.md"));
        Ok(())
    }

    // ── add: perspective comma splitting ─────────────────────────────────────

    #[tokio::test]
    async fn test_add_text_inline_with_perspective_validation() {
        // A perspective that doesn't exist in the store should fail validation.
        // We call add() with a non-existent path (inline text path) and a
        // perspective that hasn't been registered.
        let temp_dir = TempDir::new().unwrap();

        // Initialize perspective store so validation runs
        crate::perspective::init(temp_dir.path()).unwrap();

        let opts = AddOptions {
            perspectives: vec!["nonexistent-perspective".to_string()],
            ..Default::default()
        };

        let err = super::super::add::add(temp_dir.path(), "some text", opts, None)
            .await
            .unwrap_err();

        // Perspective validation should catch this
        assert!(
            err.to_string().contains("nonexistent-perspective")
                || err
                    .to_string()
                    .to_lowercase()
                    .contains("unknown perspective")
                || err.to_string().to_lowercase().contains("invalid"),
            "expected perspective error, got: {err}"
        );
    }

    // ── entry_type: invalid value returns an error ────────────────────────────

    #[tokio::test]
    async fn test_add_text_invalid_entry_type_returns_error() {
        let temp_dir = TempDir::new().unwrap();

        let opts = AddOptions {
            entry_type: "not_a_real_type".to_string(),
            ..Default::default()
        };

        let err = super::super::add::add(temp_dir.path(), "some text", opts, None)
            .await
            .unwrap_err();

        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("invalid entry type") || msg.contains("not_a_real_type"),
            "expected invalid-entry-type error, got: {err}"
        );
    }

    // ── add: perspective comma splitting logic ────────────────────────────────

    /// Split comma-separated perspective strings, trimming whitespace and removing empty parts.
    fn split_perspectives(raw: &[String]) -> Vec<String> {
        raw.iter()
            .flat_map(|p| p.split(',').map(|s| s.trim().to_string()))
            .filter(|s| !s.is_empty())
            .collect()
    }

    #[test]
    fn test_add_options_perspectives_can_be_comma_separated_conceptually() {
        // The add() function splits comma-separated perspectives.
        // We test the splitting logic itself by verifying the expected result
        // of what add() does to perspectives before validation.
        let raw = ["decisions,knowledge".to_string(), "learnings".to_string()];
        let split = split_perspectives(&raw);
        assert_eq!(split, vec!["decisions", "knowledge", "learnings"]);
    }

    #[test]
    fn test_add_options_perspectives_comma_splits_with_spaces() {
        let raw = ["  decisions , knowledge  ".to_string()];
        let split = split_perspectives(&raw);
        assert_eq!(split, vec!["decisions", "knowledge"]);
    }

    #[test]
    fn test_add_options_perspectives_empty_after_split_removed() {
        let raw = [",,,".to_string()];
        let split = split_perspectives(&raw);
        assert!(split.is_empty());
    }
}
