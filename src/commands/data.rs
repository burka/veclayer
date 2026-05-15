//! Data import, export, and index rebuild operations.

use std::io::Write;

use super::*;

/// Export all entries (or filtered by perspective) to JSONL on stdout.
pub async fn export_entries(data_dir: &Path, options: &ExportOptions) -> Result<()> {
    let store = StoreBackend::open_metadata(data_dir, true).await?;
    let perspective_refs: Vec<&str> = options.perspectives.iter().map(String::as_str).collect();
    let mut entries = store
        .list_entries(&perspective_refs, None, None, 10_000)
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
pub async fn import_entries(data_dir: &Path, options: &ImportOptions) -> Result<ImportResult> {
    let (_config, embedder, store, blob_store) = open_store(data_dir).await?;

    let lines = read_jsonl_lines(&options.path)?;

    let mut imported = 0usize;
    let mut skipped_duplicates = 0usize;
    let mut skipped_parse_errors = 0usize;
    let mut skipped_import_errors = 0usize;

    for (line_number, line) in lines.into_iter().enumerate() {
        let chunk = match serde_json::from_str::<crate::HierarchicalChunk>(&line) {
            Ok(chunk) => chunk,
            Err(e) => {
                warn!("Skipping line {}: invalid JSON: {}", line_number + 1, e);
                skipped_parse_errors += 1;
                continue;
            }
        };
        match import_parsed_entry(&embedder, &store, &blob_store, chunk).await {
            Ok(true) => imported += 1,
            Ok(false) => skipped_duplicates += 1,
            Err(e) => {
                warn!("Skipping line {}: {}", line_number + 1, e);
                skipped_import_errors += 1;
            }
        }
    }

    let skipped = skipped_duplicates + skipped_parse_errors + skipped_import_errors;
    let skip_detail =
        build_skip_detail(skipped_parse_errors, skipped_import_errors, skipped_duplicates);
    eprintln!(
        "Imported {} entries, {} skipped{}.",
        imported, skipped, skip_detail
    );
    Ok(ImportResult { imported, skipped })
}

/// Build a parenthetical breakdown of skip reasons for the import summary.
///
/// Distinguishes three causes so the summary is not misleading:
/// - `parse_errors`: a line was not valid JSON for an entry,
/// - `import_errors`: the entry parsed but could not be embedded or stored,
/// - `duplicates`: the entry already exists in the target store.
///
/// Returns an empty string when there are no skips.
pub(crate) fn build_skip_detail(
    parse_errors: usize,
    import_errors: usize,
    duplicates: usize,
) -> String {
    let plural = |n: usize| if n == 1 { "" } else { "s" };
    let mut parts = Vec::new();
    if parse_errors > 0 {
        parts.push(format!("{parse_errors} parse error{}", plural(parse_errors)));
    }
    if import_errors > 0 {
        parts.push(format!(
            "{import_errors} import error{}",
            plural(import_errors)
        ));
    }
    if duplicates > 0 {
        parts.push(format!("{duplicates} already exist"));
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!(" ({})", parts.join(", "))
    }
}

/// Rebuild the Lance vector index from the blob store.
pub async fn rebuild_index(data_dir: &Path) -> Result<()> {
    let blob_store = BlobStore::open(data_dir)?;
    let blob_count = blob_store.count()?;
    if blob_count == 0 {
        println!("No blobs found — nothing to rebuild.");
        return Ok(());
    }

    let lance_table = data_dir.join(format!("{}.lance", crate::store::TABLE_NAME));
    if lance_table.exists() {
        std::fs::remove_dir_all(&lance_table)?;
        debug!("Removed existing Lance table at {:?}", lance_table);
    }

    let config = crate::Config::new().with_data_dir(data_dir);
    let embedder = crate::embedder::from_config(&config.embedder)?;
    let dimension = embedder.dimension();
    let store = StoreBackend::open(data_dir, dimension, false).await?;

    let model_name = embedder.name();
    let mut count = 0;
    let mut skipped = 0;

    for hash_result in blob_store.iter_hashes() {
        let hash = hash_result?;
        if let Some(blob) = blob_store.get(&hash)? {
            let embedding = match blob.embedding_for_model(model_name) {
                Some(cached) if cached.len() == dimension => cached.to_vec(),
                Some(cached) => {
                    println!(
                        "Skipping '{}': cached embedding is {}d but current model uses {}d",
                        short_id(&blob.entry.content_id()),
                        cached.len(),
                        dimension
                    );
                    skipped += 1;
                    continue;
                }
                None => {
                    // Try to re-embed; if it fails (e.g., too large for Ollama context),
                    // skip the entry so the index build doesn't fail.
                    match embedder.embed(&[blob.entry.content.as_str()]) {
                        Ok(mut vecs) => vecs.pop().unwrap_or_default(),
                        Err(e) => {
                            println!(
                                "Skipping '{}': re-embedding failed ({})",
                                short_id(&blob.entry.content_id()),
                                e
                            );
                            skipped += 1;
                            continue;
                        }
                    }
                }
            };
            let chunk = crate::HierarchicalChunk::from_entry(&blob.entry, embedding);
            store.insert_chunks(vec![chunk]).await?;
            count += 1;
        }
    }

    println!("Rebuilt index: {count} entries");
    if skipped > 0 {
        println!("Skipped: {skipped}");
    }
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

/// Embed and store one already-parsed entry.
///
/// Returns `Ok(false)` when the entry is a duplicate already present in the
/// store. Errors here are import failures (embedding, blob, or store writes) —
/// distinct from JSON parse failures, which the caller handles separately.
async fn import_parsed_entry(
    embedder: &impl crate::Embedder,
    store: &impl crate::store::VectorStore,
    blob_store: &BlobStore,
    mut chunk: crate::HierarchicalChunk,
) -> Result<bool> {
    if store.get_by_id(&chunk.id).await?.is_some() {
        return Ok(false);
    }

    let embeddings = embedder.embed(&[chunk.content.as_str()])?;
    chunk.embedding = embeddings.into_iter().next();

    let blob = crate::entry::StoredBlob::from_chunk_and_embedding(&chunk, embedder.name());
    blob_store.put(&blob)?;

    store.insert_chunks(vec![chunk]).await?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    async fn seed_store(dir: &Path) -> StoreBackend {
        let dim = crate::test_helpers::config_dimension();
        let store = StoreBackend::open(dir, dim, false).await.unwrap();
        store
            .insert_chunks(vec![
                crate::test_helpers::make_test_chunk_dim(
                    "aaa111",
                    "First entry about architecture",
                    dim,
                ),
                crate::test_helpers::make_test_chunk_dim(
                    "bbb222",
                    "Second entry about testing",
                    dim,
                ),
            ])
            .await
            .unwrap();
        store
    }

    #[test]
    fn test_export_options_default() {
        let opts = ExportOptions::default();
        assert!(opts.perspectives.is_empty());
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

        let opts = ExportOptions::default();
        export_entries(dir.path(), &opts).await?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_import_skips_existing_entries() -> Result<()> {
        let dir = TempDir::new()?;
        seed_store(dir.path()).await;

        let jsonl_file = dir.path().join("export.jsonl");
        let entry_count = {
            let store = StoreBackend::open_metadata(dir.path(), true).await?;
            let entries = store.list_entries(&[], None, None, 10_000).await?;
            let jsonl: String = entries
                .iter()
                .map(|c| serde_json::to_string(&c.clone().without_embedding()).unwrap() + "\n")
                .collect();
            fs::write(&jsonl_file, &jsonl)?;
            entries.len()
        };

        let opts = ImportOptions {
            path: jsonl_file.to_string_lossy().to_string(),
        };
        let result = import_entries(dir.path(), &opts).await?;

        assert_eq!(result.imported, 0);
        assert_eq!(result.skipped, entry_count);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_export_import_roundtrip() -> Result<()> {
        let source_dir = TempDir::new()?;
        let target_dir = TempDir::new()?;
        let jsonl_file = source_dir.path().join("roundtrip.jsonl");

        {
            let dim = crate::test_helpers::config_dimension();
            let store = StoreBackend::open(source_dir.path(), dim, false).await?;
            store
                .insert_chunks(vec![
                    crate::test_helpers::make_test_chunk_dim(
                        "export001",
                        "Export roundtrip entry one",
                        dim,
                    ),
                    crate::test_helpers::make_test_chunk_dim(
                        "export002",
                        "Export roundtrip entry two",
                        dim,
                    ),
                ])
                .await?;

            let entries = store.list_entries(&[], None, None, 10_000).await?;
            let mut sorted = entries.clone();
            sorted.sort_by(|a, b| a.id.cmp(&b.id));
            let jsonl: String = sorted
                .iter()
                .map(|c| serde_json::to_string(&c.clone().without_embedding()).unwrap() + "\n")
                .collect();
            fs::write(&jsonl_file, &jsonl)?;
        }

        let opts = ImportOptions {
            path: jsonl_file.to_string_lossy().to_string(),
        };
        let result = import_entries(target_dir.path(), &opts).await?;

        assert_eq!(result.imported, 2);
        assert_eq!(result.skipped, 0);

        {
            let target_store = StoreBackend::open_metadata(target_dir.path(), true).await?;
            let imported_entries = target_store
                .list_entries(&[], None, None, usize::MAX)
                .await?;
            assert_eq!(imported_entries.len(), 2);
        }

        let result2 = import_entries(target_dir.path(), &opts).await?;
        assert_eq!(result2.imported, 0);
        assert_eq!(result2.skipped, 2);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_import_skips_bad_lines() -> Result<()> {
        let dir = TempDir::new()?;

        let valid_chunk =
            crate::test_helpers::make_test_chunk("badline001", "Valid entry for bad line test");
        let valid_json = serde_json::to_string(&valid_chunk.clone().without_embedding()).unwrap();
        let jsonl_content = format!("{}\n{{invalid json}}\n", valid_json);

        let jsonl_file = dir.path().join("bad_lines.jsonl");
        fs::write(&jsonl_file, &jsonl_content)?;

        let opts = ImportOptions {
            path: jsonl_file.to_string_lossy().to_string(),
        };
        let result = import_entries(dir.path(), &opts).await?;

        assert_eq!(result.imported, 1);
        assert_eq!(result.skipped, 1);
        Ok(())
    }

    #[test]
    fn test_build_skip_detail_no_skips() {
        assert_eq!(build_skip_detail(0, 0, 0), "");
    }

    #[test]
    fn test_build_skip_detail_duplicates_only() {
        assert_eq!(build_skip_detail(0, 0, 3), " (3 already exist)");
    }

    #[test]
    fn test_build_skip_detail_parse_errors_only() {
        assert_eq!(build_skip_detail(2, 0, 0), " (2 parse errors)");
    }

    #[test]
    fn test_build_skip_detail_import_errors_only() {
        assert_eq!(build_skip_detail(0, 2, 0), " (2 import errors)");
    }

    #[test]
    fn test_build_skip_detail_singular_uses_no_plural_suffix() {
        assert_eq!(
            build_skip_detail(1, 1, 0),
            " (1 parse error, 1 import error)"
        );
    }

    #[test]
    fn test_build_skip_detail_mixed() {
        assert_eq!(
            build_skip_detail(2, 1, 3),
            " (2 parse errors, 1 import error, 3 already exist)"
        );
    }

    #[tokio::test]
    async fn test_export_perspective_filter() -> Result<()> {
        let dir = TempDir::new()?;
        crate::perspective::init(dir.path())?;

        let store = StoreBackend::open(dir.path(), 384, false).await?;
        let mut chunk_with_perspective =
            crate::test_helpers::make_test_chunk("persp001", "Entry with decisions perspective");
        chunk_with_perspective.perspectives = vec!["decisions".to_string()];
        let chunk_no_perspective =
            crate::test_helpers::make_test_chunk("persp002", "Entry without perspective");
        store
            .insert_chunks(vec![chunk_with_perspective, chunk_no_perspective])
            .await?;

        let opts = ExportOptions {
            perspectives: vec!["decisions".to_string()],
        };
        export_entries(dir.path(), &opts).await?;
        Ok(())
    }
}
