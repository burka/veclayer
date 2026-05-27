//! Store lifecycle and inspection operations.

use super::*;

/// Initialize a new VecLayer store in the given directory.
///
/// When `share` is `true`, also creates a git memory branch and configures
/// `storage = "git"` in the project config, enabling team memory sharing.
pub fn init(cwd: &Path, data_dir: &Path, share: bool) -> Result<()> {
    if data_dir.exists() {
        // When `--share` is requested the user is setting up team-sharing for the
        // current project.  The global platform store pre-existing is expected and
        // irrelevant — printing "already exists" for it misleads the user into
        // thinking the project store was already configured.
        let is_global_store = data_dir == crate::default_data_dir();
        if !(share && is_global_store) {
            println!("VecLayer store already exists at {}", data_dir.display());
            println!("  use `veclayer add` to add knowledge");
        }
    } else {
        std::fs::create_dir_all(data_dir)?;
        println!("Initialized VecLayer store at {}", data_dir.display());
    }
    crate::perspective::init(data_dir)?;

    if share {
        init_share(cwd)?;
        return Ok(());
    }

    #[cfg(feature = "llm")]
    {
        use crate::ollama_discover;
        use crate::util::DEFAULT_OLLAMA_EMBED_MODEL;
        if let Some(info) = ollama_discover::detect_ollama() {
            if let Some(model) = info.best_embedding_model() {
                println!("\nEmbedding: Ollama ({model} @ {})", info.base_url);
            } else {
                println!("\nEmbedding: Ollama running but no embedding model found.");
                println!("  Recommended:  ollama pull {DEFAULT_OLLAMA_EMBED_MODEL}");
            }
        } else {
            println!("\nEmbedding: No Ollama detected at localhost:11434");
            println!("  Install Ollama and pull an embedding model:");
            println!("    ollama pull {DEFAULT_OLLAMA_EMBED_MODEL}");
        }
    }

    println!("\nNext steps:");
    println!("  veclayer store ./docs      # Store files");
    println!("  veclayer store \"text\"      # Store inline text");
    println!("  veclayer recall \"query\"    # Recall knowledge");
    Ok(())
}

/// Set up git-based memory sharing: create orphan branch and write project config.
///
/// If the memory branch already exists on the remote (a teammate already shared),
/// we track that branch instead of creating a new orphan — this avoids divergence.
fn init_share(cwd: &Path) -> Result<()> {
    let git_dir = crate::git::detect::find_git_dir(cwd).ok_or_else(|| {
        crate::Error::InvalidOperation(
            "Not a git project. Run from inside a git repository.".into(),
        )
    })?;

    // Probe for an existing branch before MemoryStore::open creates or tracks it,
    // so we can emit the correct message afterward. This probe-then-open is not
    // atomic: a concurrent `init` could create the branch in between. That race
    // is benign — it only affects which cosmetic message we print, never the
    // store's correctness (MemoryStore::open is idempotent w.r.t. the branch).
    let branch_existed = detect_existing_memory_branch(&git_dir);

    // Use MemoryStore::open which handles remote tracking branch detection.
    crate::git::memory_store::MemoryStore::open(&git_dir, None).map_err(|e| {
        crate::Error::InvalidOperation(format!("Failed to open memory branch: {e}"))
    })?;

    if branch_existed {
        println!("  Using existing memory branch 'veclayer-memory'");
    } else {
        println!("  Created memory branch 'veclayer-memory'");
    }

    let project_veclayer_dir = cwd.join(".veclayer");
    std::fs::create_dir_all(&project_veclayer_dir)?;
    write_git_storage_config(&project_veclayer_dir)?;
    println!("  Updated .veclayer/config.toml");
    println!();
    println!("Commit .veclayer/config.toml to share with your team.");
    println!("Team members run: veclayer sync");

    Ok(())
}

/// Returns `true` when the `veclayer-memory` branch (or its remote counterpart)
/// already exists, so `init_share` can emit "Using existing …" vs "Created …".
fn detect_existing_memory_branch(git_dir: &std::path::Path) -> bool {
    use crate::git::GitMemoryBranch;

    let Ok(branch) = GitMemoryBranch::open(git_dir, None) else {
        return false;
    };

    // Local branch present?
    if branch.branch_exists().unwrap_or(false) {
        return true;
    }

    // Remote tracking ref present? (refs/remotes/origin/veclayer-memory)
    // Use the same remote name as the rest of the codebase ("origin").
    let remote_ref = format!(
        "refs/remotes/{}/{}",
        crate::git::REMOTE,
        crate::git::DEFAULT_BRANCH
    );
    let output = std::process::Command::new("git")
        .arg("--git-dir")
        .arg(git_dir)
        .args(["rev-parse", "--verify", &remote_ref])
        .output();
    output.map(|o| o.status.success()).unwrap_or(false)
}

/// Write `storage = "git"` and `push = "review"` into `.veclayer/config.toml`.
///
/// If the file already exists, adds the fields without overwriting existing content.
/// If it does not exist, creates it with both fields.
fn write_git_storage_config(data_dir: &Path) -> Result<()> {
    let config_path = data_dir.join("config.toml");

    if config_path.exists() {
        let mut content = std::fs::read_to_string(&config_path)?;
        let existing: toml::Value = toml::from_str(&content).map_err(|e| {
            crate::Error::InvalidOperation(format!(
                "failed to parse existing {}: {e}",
                config_path.display()
            ))
        })?;
        if existing.get("storage").is_none() {
            content.push_str("\nstorage = \"git\"\n");
        }
        if existing.get("push").is_none() {
            content.push_str("push = \"review\"\n");
        }
        std::fs::write(&config_path, content)?;
    } else {
        std::fs::write(&config_path, "storage = \"git\"\npush = \"review\"\n")?;
    }

    Ok(())
}

/// Format a model selection line showing the selected model and available alternatives.
///
/// Output examples:
/// - `qwen3.5:9b (selected) | also: llama3.2:3b, mistral`
/// - `(none detected — run: ollama pull nomic-embed-text)`
#[cfg(feature = "llm")]
fn format_model_line(selected: Option<&str>, all: &[String], pull_hint: &str) -> String {
    match selected {
        None => format!("(none detected — run: {pull_hint})"),
        Some(best) => {
            let others: Vec<&str> = all
                .iter()
                .filter(|m| m.as_str() != best)
                .map(String::as_str)
                .collect();
            if others.is_empty() {
                format!("{best} (selected)")
            } else {
                format!("{best} (selected) | also: {}", others.join(", "))
            }
        }
    }
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
    println!(
        "{}  {}",
        "Sources:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
        result.source_files.len()
    );

    // Ollama availability
    #[cfg(feature = "llm")]
    {
        use owo_colors::OwoColorize;

        println!(
            "\n{}",
            "[Integrations]".if_supports_color(Stream::Stdout, |s| s.bold())
        );

        match crate::ollama_discover::detect_ollama() {
            Some(info) => {
                println!(
                    "  {}  detected {}",
                    "Ollama:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
                    info.base_url
                );

                // Chat model line
                let chat_line = format_model_line(
                    info.best_chat_model(),
                    &info.chat_models,
                    "ollama pull qwen3",
                );
                println!(
                    "    {}  {}",
                    "chat:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
                    chat_line
                );

                // Embedding model line
                let embed_line = format_model_line(
                    info.best_embedding_model(),
                    &info.embedding_models,
                    "ollama pull nomic-embed-text",
                );
                println!(
                    "    {}  {}",
                    "embed:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
                    embed_line
                );
            }
            None => {
                println!(
                    "  {}  not detected",
                    "Ollama:".if_supports_color(Stream::Stdout, |s| s.dimmed())
                );
            }
        }
    }

    // Usage guide
    println!(
        "\n{}",
        "MCP tools (via Claude Code / MCP client):".if_supports_color(Stream::Stdout, |s| s.bold())
    );
    println!("  store(content, heading, perspectives)  Save a decision or learning");
    println!("  recall(query)                          Search memory semantically");
    println!("  think(action=\"prepare\")                Reflect without LLM");
    println!("  think(action=\"consolidate\")            Full LLM-powered reflection");
    println!("  focus(id)                              Drill into an entry");

    println!(
        "\n{}",
        "CLI (shell):".if_supports_color(Stream::Stdout, |s| s.bold())
    );
    println!("  veclayer store \"text\" --heading \"...\" -P decisions");
    println!("  veclayer recall \"query\"");
    println!("  veclayer think prepare");
    println!("  veclayer context --brief");

    if result.source_files.is_empty() {
        println!("\nStore is empty. Run `veclayer setup claude --apply` to get started.");
    } else {
        println!("\nPersist decisions and learnings as you work. Run think between tasks.");
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

/// Controls the output format for `stale`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum OutputMode {
    Text,
    LlmNudge,
}

impl std::str::FromStr for OutputMode {
    type Err = crate::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "text" => Ok(Self::Text),
            "llm-nudge" => Ok(Self::LlmNudge),
            other => Err(crate::Error::InvalidOperation(format!(
                "Unknown output mode '{other}'. Valid modes: text, llm-nudge"
            ))),
        }
    }
}

/// Check whether any knowledge was stored recently.
///
/// `since` is a duration string (e.g. "15min", "1h") parsed by `parse_temporal`.
/// `output` controls the output mode: "text" (human-readable) or "llm-nudge"
/// Return the OS-temp-dir path of the per-`data_dir` stale-check marker file.
fn stale_marker_path(data_dir: &Path) -> std::path::PathBuf {
    use sha2::{Digest, Sha256};
    // Canonicalize the path so symlinks don't cause collisions.
    let path_str = data_dir
        .canonicalize()
        .unwrap_or_else(|_| data_dir.to_path_buf())
        .to_string_lossy()
        .to_string();
    let hash = Sha256::digest(path_str.as_bytes());
    // First 8 bytes as hex = 16 chars, ample entropy, no extra deps.
    std::env::temp_dir().join(format!(".veclayer_stale_{}", hex_encode(&hash[..8])))
}

/// Encode bytes as lowercase hex string.
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// (machine-friendly, returns exit code 2 when stale).
/// `hooks_enabled` mirrors the `hooks_enabled` config field — when `false`, exits 0 silently.
pub async fn stale(data_dir: &Path, since: &str, output: &str, hooks_enabled: bool) -> Result<i32> {
    let output_mode: OutputMode = output.parse()?;
    // Config-level opt-out (veclayer.toml: hooks_enabled = false, or VECLAYER_HOOKS_ENABLED=false)
    if !hooks_enabled {
        return Ok(0);
    }
    // Allow opt-out via VECLAYER_STALE=off (e.g. in project .claude/settings.json env block)
    if std::env::var("VECLAYER_STALE").ok().as_deref() == Some("off") {
        return Ok(0);
    }
    // Deduplicate: if another stale check ran within the last 5 seconds for the
    // same data_dir, skip silently. This prevents duplicate output when both global
    // and project-level hooks invoke `veclayer stale`.
    let marker = stale_marker_path(data_dir);
    if let Ok(meta) = std::fs::metadata(&marker) {
        if let Ok(modified) = meta.modified() {
            if modified.elapsed().unwrap_or_default() < std::time::Duration::from_secs(5) {
                return Ok(0);
            }
        }
    }
    // Touch the marker file
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&marker);

    let threshold_epoch = crate::resolve::parse_temporal(since)
        .ok_or_else(|| crate::Error::InvalidOperation(format!("Invalid duration: {since}")))?;

    let store = StoreBackend::open_metadata(data_dir, true).await?;
    // list_entries returns newest-first; one entry is enough to check freshness.
    let recent = store.list_entries(&[], None, None, 1).await?;

    let now = crate::chunk::now_epoch_secs();

    let exit_code = match recent.first() {
        None => print_stale_result(&output_mode, None, now),
        Some(chunk) => {
            let created_at = chunk.access_profile.created_at;
            let is_fresh = created_at >= threshold_epoch;
            print_stale_result(&output_mode, Some((created_at, is_fresh)), now)
        }
    };
    Ok(exit_code)
}

fn print_stale_result(output: &OutputMode, entry: Option<(i64, bool)>, now: i64) -> i32 {
    match output {
        OutputMode::LlmNudge => print_llm_nudge(entry, now),
        OutputMode::Text => {
            print_text(entry, now);
            0
        }
    }
}

fn minutes_ago(created_at: i64, now: i64) -> i64 {
    (now - created_at).max(0) / 60
}

fn print_text(entry: Option<(i64, bool)>, now: i64) {
    match entry {
        None => println!("Memory is stale. No entries in store."),
        Some((created_at, true)) => {
            println!(
                "Memory is fresh. Last store was {} minutes ago.",
                minutes_ago(created_at, now)
            );
        }
        Some((created_at, false)) => {
            println!(
                "Memory is stale. Last store was {} minutes ago.",
                minutes_ago(created_at, now)
            );
        }
    }
}

fn print_llm_nudge(entry: Option<(i64, bool)>, now: i64) -> i32 {
    match entry {
        Some((created_at, true)) => {
            let m = minutes_ago(created_at, now);
            eprintln!("Memory fresh ({m}min ago). OK to stop.");
            0
        }
        Some((created_at, false)) => {
            let m = minutes_ago(created_at, now);
            eprintln!("Last store was {m} minutes ago. Before stopping:");
            eprintln!("- Store decisions, learnings, conventions, or bugs found this session (e.g. \"Switched to X because Y — 2x faster, same recall\").");
            eprintln!("- Run `think` to curate memory if needed.");
            eprintln!("- Then stop again.");
            2
        }
        None => {
            eprintln!("No entries in store. Before stopping:");
            eprintln!("- Store decisions, learnings, conventions, or bugs found this session (e.g. \"Switched to X because Y — 2x faster, same recall\").");
            eprintln!("- Run `think` to curate memory if needed.");
            eprintln!("- Then stop again.");
            2
        }
    }
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
    data_dir: &Path,
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
    println!("  data_dir: {}", data_dir.display().to_string().cyan());
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
    if let Some(ref storage) = resolved.storage {
        println!("  storage: {}", storage.cyan());
    }
    if let Some(ref push) = resolved.push {
        println!("  push: {}", push.cyan());
    }

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

    // Ollama integration status
    #[cfg(feature = "llm")]
    {
        println!("\n{}", "[Integrations]".bold());
        match crate::ollama_discover::detect_ollama() {
            Some(info) => {
                println!(
                    "  Ollama:  {} {}",
                    "detected".if_supports_color(Stream::Stdout, |s| s.green()),
                    info.base_url
                        .if_supports_color(Stream::Stdout, |s| s.dimmed())
                );
                let chat_line = format_model_line(
                    info.best_chat_model(),
                    &info.chat_models,
                    "ollama pull qwen3.5",
                );
                println!(
                    "    {}  {}",
                    "chat:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
                    chat_line
                );
                let embed_line = format_model_line(
                    info.best_embedding_model(),
                    &info.embedding_models,
                    "ollama pull nomic-embed-text",
                );
                println!(
                    "    {}  {}",
                    "embed:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
                    embed_line
                );
            }
            None => {
                println!(
                    "  Ollama:  {}",
                    "not detected".if_supports_color(Stream::Stdout, |s| s.dimmed())
                );
            }
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
            let parsed: toml::Value =
                toml::from_str(&content).unwrap_or(toml::Value::Table(Default::default()));
            parsed.get("storage").and_then(|v| v.as_str()) == Some("git")
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

    #[tokio::test]
    async fn test_stale_dedup_second_call_is_noop() -> Result<()> {
        let temp_dir = TempDir::new()?;

        // First call should run (returns exit code based on store state)
        let code1 = stale(temp_dir.path(), "15min", "llm-nudge", true).await?;

        // Second call within 5 seconds should be deduped (exit 0)
        let code2 = stale(temp_dir.path(), "15min", "llm-nudge", true).await?;
        assert_eq!(code2, 0, "second stale call should be deduped to exit 0");

        // First call should have produced a non-zero exit (empty store = stale)
        assert_ne!(
            code1, 0,
            "first stale call on empty store should be non-zero"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_stale_dedup_no_marker_runs_normally() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let marker = stale_marker_path(temp_dir.path());

        // Ensure no marker exists (clean temp dir)
        let _ = std::fs::remove_file(&marker);
        assert!(!marker.exists());

        // Should run normally (no marker = no dedup)
        let code = stale(temp_dir.path(), "15min", "llm-nudge", true).await?;
        assert_ne!(code, 0, "stale call without marker should run normally");

        // Marker should now exist
        assert!(marker.exists(), "marker should be created after stale runs");

        Ok(())
    }

    #[tokio::test]
    async fn test_stale_hooks_disabled_skips() -> Result<()> {
        let temp_dir = TempDir::new()?;

        let code = stale(temp_dir.path(), "15min", "llm-nudge", false).await?;
        assert_eq!(code, 0, "hooks_enabled=false should exit 0");

        // Marker should NOT be created when hooks are disabled
        assert!(
            !stale_marker_path(temp_dir.path()).exists(),
            "no marker when hooks disabled"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_stale_different_data_dirs_not_deduped() -> Result<()> {
        let dir1 = TempDir::new()?;
        let dir2 = TempDir::new()?;

        // First call on dir1
        let code1 = stale(dir1.path(), "15min", "llm-nudge", true).await?;
        assert_ne!(code1, 0);

        // Call on dir2 should NOT be deduped (different data_dir)
        let code2 = stale(dir2.path(), "15min", "llm-nudge", true).await?;
        assert_ne!(code2, 0, "different data_dir should not be deduped");

        Ok(())
    }

    // ── write_git_storage_config ──────────────────────────────────────────────

    #[test]
    fn test_write_git_storage_config_creates_new_file() -> Result<()> {
        let temp_dir = TempDir::new()?;
        write_git_storage_config(temp_dir.path())?;

        let config_path = temp_dir.path().join("config.toml");
        assert!(config_path.exists());
        let content = std::fs::read_to_string(&config_path)?;
        assert!(content.contains("storage = \"git\""));
        assert!(content.contains("push = \"review\""));
        Ok(())
    }

    #[test]
    fn test_write_git_storage_config_existing_file_without_fields() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config_path = temp_dir.path().join("config.toml");
        std::fs::write(&config_path, "project = \"test\"\n")?;

        write_git_storage_config(temp_dir.path())?;

        let content = std::fs::read_to_string(&config_path)?;
        assert!(content.contains("project = \"test\""));
        assert!(content.contains("storage = \"git\""));
        assert!(content.contains("push = \"review\""));
        Ok(())
    }

    #[test]
    fn test_write_git_storage_config_does_not_duplicate_existing_fields() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config_path = temp_dir.path().join("config.toml");
        std::fs::write(&config_path, "storage = \"lancedb\"\npush = \"manual\"\n")?;

        write_git_storage_config(temp_dir.path())?;

        let content = std::fs::read_to_string(&config_path)?;
        // Original values preserved — function must not overwrite existing keys
        assert!(content.contains("storage = \"lancedb\""));
        assert!(content.contains("push = \"manual\""));
        // Must not have added a second storage or push line
        assert_eq!(content.matches("storage").count(), 1);
        assert_eq!(content.matches("push").count(), 1);
        Ok(())
    }

    // ── minutes_ago ───────────────────────────────────────────────────────────

    #[test]
    fn test_minutes_ago_basic() {
        assert_eq!(minutes_ago(0, 3600), 60);
        assert_eq!(minutes_ago(0, 60), 1);
        assert_eq!(minutes_ago(0, 0), 0);
    }

    #[test]
    fn test_minutes_ago_clamps_negative() {
        // future created_at relative to now → should return 0, not negative
        assert_eq!(minutes_ago(1000, 500), 0);
    }

    // ── OutputMode::from_str ──────────────────────────────────────────────────

    #[test]
    fn test_output_mode_parses_text() {
        let mode: OutputMode = "text".parse().unwrap();
        assert_eq!(mode, OutputMode::Text);
    }

    #[test]
    fn test_output_mode_parses_llm_nudge() {
        let mode: OutputMode = "llm-nudge".parse().unwrap();
        assert_eq!(mode, OutputMode::LlmNudge);
    }

    #[test]
    fn test_output_mode_rejects_unknown_with_helpful_message() {
        let err = "bogus".parse::<OutputMode>().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Unknown output mode 'bogus'"),
            "expected unknown-mode message, got: {msg}"
        );
        assert!(
            msg.contains("text") && msg.contains("llm-nudge"),
            "error should list valid modes, got: {msg}"
        );
    }

    // ── print_stale_result ────────────────────────────────────────────────────

    #[test]
    fn test_print_stale_result_text_returns_zero() {
        let code = print_stale_result(&OutputMode::Text, None, 1000);
        assert_eq!(code, 0);
    }

    #[test]
    fn test_print_stale_result_text_fresh_returns_zero() {
        let code = print_stale_result(&OutputMode::Text, Some((900, true)), 1000);
        assert_eq!(code, 0);
    }

    #[test]
    fn test_print_stale_result_text_stale_returns_zero() {
        // text mode always returns 0 regardless of freshness
        let code = print_stale_result(&OutputMode::Text, Some((100, false)), 1000);
        assert_eq!(code, 0);
    }

    #[test]
    fn test_print_stale_result_llm_nudge_no_entries_returns_two() {
        let code = print_stale_result(&OutputMode::LlmNudge, None, 1000);
        assert_eq!(code, 2);
    }

    #[test]
    fn test_print_stale_result_llm_nudge_fresh_returns_zero() {
        let code = print_stale_result(&OutputMode::LlmNudge, Some((900, true)), 1000);
        assert_eq!(code, 0);
    }

    #[test]
    fn test_print_stale_result_llm_nudge_stale_returns_two() {
        let code = print_stale_result(&OutputMode::LlmNudge, Some((100, false)), 1000);
        assert_eq!(code, 2);
    }

    // ── archive error path ────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_archive_empty_ids_returns_error() {
        let temp_dir = TempDir::new().unwrap();
        let err = archive(temp_dir.path(), &[]).await.unwrap_err();
        assert!(
            err.to_string().contains("No entry IDs"),
            "expected 'No entry IDs', got: {err}"
        );
    }

    #[tokio::test]
    async fn test_archive_unknown_id_returns_not_found() {
        let temp_dir = TempDir::new().unwrap();
        let err = archive(temp_dir.path(), &["deadbeef".to_string()])
            .await
            .unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("not found")
                || err.to_string().contains("deadbeef"),
            "expected not-found error, got: {err}"
        );
    }

    // ── stale invalid duration ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_stale_invalid_duration_returns_error() {
        let temp_dir = TempDir::new().unwrap();
        let err = stale(temp_dir.path(), "not-a-duration", "text", true)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("Invalid duration"),
            "expected 'Invalid duration', got: {err}"
        );
    }

    // ── status/print_sources on empty store ───────────────────────────────────

    #[tokio::test]
    async fn test_status_empty_store() -> Result<()> {
        let temp_dir = TempDir::new()?;
        status(temp_dir.path()).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_print_sources_empty_store() -> Result<()> {
        let temp_dir = TempDir::new()?;
        print_sources(temp_dir.path()).await?;
        Ok(())
    }

    // ── detect_existing_memory_branch ────────────────────────────────────────

    /// Create a bare-minimum git repo in `dir` (no commits needed).
    fn init_git_repo(dir: &std::path::Path) {
        let status = std::process::Command::new("git")
            .args(["init", "--initial-branch=main", dir.to_str().unwrap()])
            .output()
            .expect("git init");
        assert!(status.status.success(), "git init failed");
    }

    /// Returns the `.git/` directory inside `dir`.
    fn git_dir(repo_root: &std::path::Path) -> std::path::PathBuf {
        repo_root.join(".git")
    }

    /// Returns `false` when no memory branch exists in a fresh repo.
    #[test]
    fn test_detect_existing_branch_fresh_repo_returns_false() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());
        let result = detect_existing_memory_branch(&git_dir(temp.path()));
        assert!(!result, "fresh repo must report no existing branch");
    }

    /// Returns `true` when the memory branch was already created.
    #[test]
    fn test_detect_existing_branch_after_branch_created_returns_true() {
        let temp = TempDir::new().unwrap();
        init_git_repo(temp.path());
        let repo_root = temp.path();

        // Create an empty orphan veclayer-memory branch so the branch exists.
        let _ = std::process::Command::new("git")
            .args([
                "--git-dir",
                &git_dir(repo_root).to_string_lossy(),
                "symbolic-ref",
                "HEAD",
                "refs/heads/veclayer-memory",
            ])
            .output()
            .expect("symbolic-ref");

        // Write a tree object and a commit so the branch ref is actually created.
        let tree_out = std::process::Command::new("git")
            .args([
                "--git-dir",
                &git_dir(repo_root).to_string_lossy(),
                "hash-object",
                "-t",
                "tree",
                "--stdin",
            ])
            .stdin(std::process::Stdio::null())
            .output()
            .expect("hash-object tree");
        let tree_sha = String::from_utf8_lossy(&tree_out.stdout).trim().to_string();

        let commit_out = std::process::Command::new("git")
            .args([
                "--git-dir",
                &git_dir(repo_root).to_string_lossy(),
                "commit-tree",
                &tree_sha,
                "-m",
                "init",
            ])
            .output()
            .expect("commit-tree");
        let commit_sha = String::from_utf8_lossy(&commit_out.stdout)
            .trim()
            .to_string();

        // Point the branch ref at the commit.
        let _ = std::process::Command::new("git")
            .args([
                "--git-dir",
                &git_dir(repo_root).to_string_lossy(),
                "update-ref",
                "refs/heads/veclayer-memory",
                &commit_sha,
            ])
            .output()
            .expect("update-ref");

        let result = detect_existing_memory_branch(&git_dir(repo_root));
        assert!(result, "must report existing branch after it was created");
    }
}
