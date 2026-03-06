//! Store lifecycle and inspection operations.

use super::*;

/// Initialize a new VecLayer store in the given directory.
///
/// When `share` is `true`, also creates a git memory branch and configures
/// `storage = "git"` in the project config, enabling team memory sharing.
pub fn init(cwd: &Path, data_dir: &Path, share: bool) -> Result<()> {
    if data_dir.exists() {
        println!("VecLayer store already exists at {}", data_dir.display());
        println!("  use `veclayer add` to add knowledge");
    } else {
        std::fs::create_dir_all(data_dir)?;
        println!("Initialized VecLayer store at {}", data_dir.display());
    }
    crate::perspective::init(data_dir)?;

    if share {
        init_share(cwd)?;
        return Ok(());
    }

    println!("\nNext steps:");
    println!("  veclayer store ./docs      # Store files");
    println!("  veclayer store \"text\"      # Store inline text");
    println!("  veclayer recall \"query\"    # Recall knowledge");
    Ok(())
}

/// Set up git-based memory sharing: create orphan branch and write project config.
fn init_share(cwd: &Path) -> Result<()> {
    let git_dir = crate::git::detect::find_git_dir(cwd).ok_or_else(|| {
        crate::Error::InvalidOperation(
            "Not a git project. Run from inside a git repository.".into(),
        )
    })?;

    let branch = crate::git::GitMemoryBranch::open(&git_dir, None).map_err(|e| {
        crate::Error::InvalidOperation(format!("Failed to open memory branch: {e}"))
    })?;
    branch.create_orphan_branch().map_err(|e| {
        crate::Error::InvalidOperation(format!("Failed to create orphan branch: {e}"))
    })?;

    println!("  Created memory branch 'veclayer-memory'");

    let project_veclayer_dir = cwd.join(".veclayer");
    std::fs::create_dir_all(&project_veclayer_dir)?;
    write_git_storage_config(&project_veclayer_dir)?;
    println!("  Updated .veclayer/config.toml");
    println!();
    println!("Commit .veclayer/config.toml to share with your team.");
    println!("Team members run: veclayer sync");

    Ok(())
}

/// Write `storage = "git"` and `push = "auto"` into `.veclayer/config.toml`.
///
/// If the file already exists, adds the fields without overwriting existing content.
/// If it does not exist, creates it with both fields.
fn write_git_storage_config(data_dir: &Path) -> Result<()> {
    let config_path = data_dir.join("config.toml");

    if config_path.exists() {
        let mut content = std::fs::read_to_string(&config_path)?;
        if !content.contains("storage") {
            content.push_str("\nstorage = \"git\"\n");
        }
        if !content.contains("push") {
            content.push_str("push = \"review\"\n");
        }
        std::fs::write(&config_path, content)?;
    } else {
        std::fs::write(&config_path, "storage = \"git\"\npush = \"review\"\n")?;
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

/// List all indexed source files (returns data).
pub async fn sources(data_dir: &Path) -> Result<Vec<String>> {
    let store = StoreBackend::open_metadata(data_dir, true).await?;

    let store_stats = store.stats().await?;

    Ok(store_stats.source_files)
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

/// Archive entries by demoting them to deep_only visibility.
pub async fn archive(data_dir: &Path, ids: &[String]) -> Result<()> {
    if ids.is_empty() {
        return Err(crate::Error::InvalidOperation(
            "No entry IDs provided. Usage: veclayer archive <ID>...".into(),
        ));
    }

    let store = StoreBackend::open_metadata(data_dir, false).await?;

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

/// Show version/relation history of an entry.
pub async fn history(data_dir: &Path, id: &str) -> Result<()> {
    let store = StoreBackend::open_metadata(data_dir, true).await?;

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

    Ok(())
}

/// Show resolved configuration for the current working directory.
pub fn show_config(
    cwd: &Path,
    user_config: &crate::config::UserConfig,
    resolved: &crate::config::ResolvedConfig,
    git_remote: Option<&str>,
    git_branch: Option<&str>,
) -> Result<()> {
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

    println!("\n{}", "[Git]".bold());
    println!(
        "  remote: {}",
        git_remote
            .unwrap_or("(none)")
            .if_supports_color(Stream::Stdout, |s| s.cyan())
    );
    println!(
        "  branch: {}",
        git_branch
            .unwrap_or("(none)")
            .if_supports_color(Stream::Stdout, |s| s.cyan())
    );

    if !resolved.scopes.is_empty() {
        println!("\n{}", "[Scopes]".bold());
        for scope in &resolved.scopes {
            println!(
                "  {:<18} {:<30} {}",
                scope.name.if_supports_color(Stream::Stdout, |s| s.yellow()),
                scope
                    .storage
                    .if_supports_color(Stream::Stdout, |s| s.cyan()),
                scope.push.if_supports_color(Stream::Stdout, |s| s.dimmed()),
            );
        }
    }

    Ok(())
}

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

    println!("\nTry: recall, reflect, think, reflect salience");

    let cwd = std::env::current_dir().unwrap_or_default();
    if crate::git::detect::find_git_dir(&cwd).is_some() {
        let project_config_path = cwd.join(".veclayer/config.toml");
        let has_git_storage = project_config_path.exists() && {
            let content = std::fs::read_to_string(&project_config_path).unwrap_or_default();
            content.contains("storage") && content.contains("git")
        };

        if !has_git_storage {
            println!();
            println!(
                "{}",
                "This is a git project. Store memory in git:"
                    .if_supports_color(Stream::Stdout, |s| s.dimmed())
            );
            println!("  veclayer init --share");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_init_creates_directory() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let store_dir = temp_dir.path().join("new-store");

        init(temp_dir.path(), &store_dir, false)?;

        assert!(store_dir.exists());
        Ok(())
    }

    #[test]
    fn test_init_existing_directory() -> Result<()> {
        let temp_dir = TempDir::new()?;

        init(temp_dir.path(), temp_dir.path(), false)?;
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
