//! Project-aware blob merge between stores.

use super::*;

/// Merge blobs from a source store into the target store.
pub async fn merge(data_dir: &Path, source: &Path, options: &MergeOptions) -> Result<()> {
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

    let cwd = std::env::current_dir().unwrap_or_else(|_| data_dir.to_path_buf());
    let git_info = crate::git::detect::detect(&cwd);
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

    let (new_count, dup_count) =
        copy_blobs(&source_blob_store, data_dir, options.dry_run, &project_tag)?;

    let (persp_added, _persp_skipped) = if options.dry_run {
        (0, 0)
    } else {
        crate::perspective::merge_from(data_dir, source)?
    };

    if !options.dry_run && new_count > 0 {
        rebuild_index(data_dir).await?;
    }

    if !options.dry_run {
        if let Some(ref proj) = options.project {
            if let Err(e) =
                try_update_user_config(source, proj, &user_config, git_info.remote.as_deref())
            {
                warn!("Could not update user config: {}", e);
            }
        }
    }

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

/// Resolve project scope for merge.
fn resolve_merge_project(
    source: &Path,
    flag_project: &Option<String>,
    resolved: &crate::config::ResolvedConfig,
    git_remote: Option<&str>,
) -> Option<String> {
    if let Some(p) = flag_project {
        return Some(p.clone());
    }

    if source.join("config.toml").exists() {
        let project = crate::config::discover_project(source.parent().unwrap_or(source))
            .and_then(|(_, pc)| pc.project);
        if project.is_some() {
            return project;
        }
    }

    if resolved.project.is_some() {
        return resolved.project.clone();
    }

    git_remote.map(String::from)
}

/// Append a `[[match]]` entry to user config if no existing match covers the source.
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

        seed_blob_store(source.path(), &["shared", "unique"]);
        seed_blob_store(target.path(), &["shared"]);

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
        assert_eq!(target_store.count()?, 0);
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

        let opts = MergeOptions::default();
        let result = merge(target.path(), source.path(), &opts).await;

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

        let target_store = BlobStore::open(target.path())?;
        target_store.put(&test_blob("shared"))?;
        drop(target_store);

        let (new, dup) = copy_blobs(&source_store, target.path(), false, &None)?;
        assert_eq!(new, 1);
        assert_eq!(dup, 1);
        Ok(())
    }
}
