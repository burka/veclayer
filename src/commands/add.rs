//! Add/ingest commands and related helpers.

use super::*;
use crate::parser::DocumentParser;

/// Backwards-compatible alias
pub async fn ingest(data_dir: &Path, path: &Path, options: &AddOptions) -> Result<AddResult> {
    add_files(data_dir, path, options).await
}

/// Add knowledge to the store (files, directories, or inline text).
pub async fn add(data_dir: &Path, input: &str, mut options: AddOptions) -> Result<AddResult> {
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
        add_files(data_dir, input_path, &options).await
    } else {
        add_text(data_dir, input, &options).await
    }
}

/// Add files from a path to the store.
async fn add_files(data_dir: &Path, path: &Path, options: &AddOptions) -> Result<AddResult> {
    debug!("Opening store at {:?}...", data_dir);
    let (config, embedder, store, blob_store) = super::open_store(data_dir).await?;

    let parser = MarkdownParser::new();

    let files = collect_files(path, options.recursive, options.follow_links, &parser)?;
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

        for chunk in &chunks {
            let blob = crate::entry::StoredBlob::from_chunk_and_embedding(chunk, embedder.name());
            blob_store.put(&blob)?;
        }

        store.insert_chunks(chunks.clone()).await?;
        debug!("  Indexed successfully");

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

        let summary_embedder = crate::embedder::from_config(&config.embedder)?;
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
    let (_config, embedder, store, blob_store) = super::open_store(data_dir).await?;

    let entry_type = match options.entry_type.as_str() {
        "meta" => crate::chunk::EntryType::Meta,
        "impression" => crate::chunk::EntryType::Impression,
        "summary" => crate::chunk::EntryType::Summary,
        _ => crate::chunk::EntryType::Raw,
    };

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

    let blob = crate::entry::StoredBlob::from_chunk_and_embedding(&chunk, embedder.name());
    blob_store.put(&blob)?;

    let id = chunk.id.clone();
    let store = std::sync::Arc::new(store);
    store.insert_chunks(vec![chunk]).await?;

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
}
