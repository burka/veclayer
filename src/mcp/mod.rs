//! MCP server module: 5-tool agent interface (recall, focus, store, think, share).
//!
//! Split into:
//! - `types` -- Input/output structs
//! - `tools` -- Tool implementation functions
//! - `handler` -- Shared rmcp tool handler (used by both transports)
//! - `stdio` -- MCP stdio transport (rmcp)
//! - `http` -- HTTP REST API + Streamable HTTP MCP transport (rmcp)

pub mod embed_worker;
pub mod format;
pub mod handler;
#[cfg(feature = "http")]
pub mod http;
pub mod resources;
pub mod stdio;
pub mod tools;
pub mod types;

pub use handler::McpHandler;
#[cfg(feature = "http")]
pub use http::{build_app, run_http, AppState, AuthSetup};
pub use stdio::run_stdio;

/// Try to open the git memory store for the current project.
///
/// Returns `None` when git storage is not configured or the git dir cannot be found.
pub(crate) fn open_git_store(
    config: &crate::Config,
) -> Option<std::sync::Arc<crate::git::memory_store::MemoryStore>> {
    if config.storage.as_deref() != Some("git") {
        return None;
    }

    let cwd = std::env::current_dir().ok()?;
    let git_dir = crate::git::detect::find_git_dir(&cwd)?;

    match crate::git::memory_store::MemoryStore::open(&git_dir, None) {
        Ok(store) => {
            tracing::info!("Git memory store opened for MCP sessions");
            Some(std::sync::Arc::new(store))
        }
        Err(e) => {
            tracing::warn!("Failed to open git memory store: {e}");
            None
        }
    }
}

/// Instructions provided to agents on first connection.
pub const MCP_INSTRUCTIONS: &str = "\
VecLayer is a hierarchical vector database with memory — a persistent identity store for AI agents.

## Your Memory System

You have access to a structured, aging knowledge base. Unlike flat key-value memory, VecLayer \
organizes knowledge in trees (headings → subheadings → content) with visibility levels and \
access tracking. Knowledge that you use often stays prominent. Knowledge you ignore fades.

## Five Tools

### recall — What do I know about this?
Find relevant knowledge using semantic search, or browse without a query. Results include \
a relevance tier (strong/moderate/weak/tangential). Use `since`/`until` for temporal filtering. Results come with access profiles \
showing how often each piece was accessed. Use `deep: true` to include archived knowledge. \
Use `recency` to boost recently accessed memories. \
Use `ongoing: true` to see only open threads — unresolved items that need attention.

### focus — Tell me more about this specific point.
Dive deeper into a specific memory node. Returns the node itself plus its children, optionally \
reranked by a question lens. Pass a `question` to surface the most relevant children for that \
angle — 'How was this decided?' yields different details than 'Who was involved?'

### store — I want to remember this.
Write new knowledge directly. Supports `relations` for atomic link creation (e.g. \
`relations: [{kind: \"supersedes\", target_id: \"...\"}]`), `entry_type` for classification \
(raw/summary/meta/impression), and `items` for batch storage. \
The server generates embeddings automatically. Use `parent_id` to place it in the hierarchy.

### think — Let me reflect and curate.
Your curation hub. Without an action, returns a reflection report: hot chunks, stale chunks, \
salience scores, and suggested actions. With an action, executes curation:
- `action: 'promote'` — Make important knowledge always visible
- `action: 'demote'` — Archive outdated knowledge
- `action: 'relate'` — Connect knowledge (supersedes, summarizes, related_to, derived_from)
- `action: 'configure_aging'` — Set automatic degradation rules
- `action: 'apply_aging'` — Execute aging rules now (respects salience protection)
- `action: 'salience'` — See which entries are most/least important
- `action: 'discover'` — Find similar-but-unlinked entries (hidden connections, contradictions, consolidation candidates)
- `action: 'perspectives'` — List all available perspectives
- `action: 'status'` — Show store statistics (entry count, source files, aging policy)
- `action: 'history'` — Show an entry's relations and metadata (requires `id`)
- `action: 'sync'` — Sync with remote: pull then push. Optional `direction`: 'pull', 'push', or omit for both

### share — Here, this is for you. [Experimental — not yet functional]
Generates a scoped share-token payload describing what knowledge to share and with what \
permissions. This is a preview of the upcoming UCAN-based sharing system — tokens are not \
yet cryptographically signed. Do not rely on this tool in production workflows.

## How to Use Your Memory

### Capture what matters — proactively, not on request
Store decisions and their rationale as you make them. When you reject an approach, store why. \
When you learn something, store it with `perspectives: [\"learnings\"]`. When you solve a bug, \
store the root cause. Don't wait to be asked — good memory is built in the moment.

### Before new tasks — check what you already know
Before starting work, `recall` the topic. You may have solved something similar before, \
or made a decision that applies. Check `recall(perspective=\"decisions\")` for past trade-offs.

### Summarize to consolidate
When related knowledge accumulates, synthesize:
1. `recall` a topic area
2. `focus` on interesting nodes to read children
3. Write a summary
4. `store` it with `parent_id` to place in hierarchy
5. `think(action='relate', kind='summarizes')` to link originals to the summary

### Curate between tasks
Run `think` when you have time: start of session, end of session, between tasks. Review what's \
hot, what's stale, what needs promoting or archiving. Mark outdated knowledge with \
`think(action='relate', kind='supersedes')`.

### Act on Stop hook feedback
A Stop hook may block you from stopping and provide a reason (e.g., \"Persist before stopping: ...\"). \
When this happens, act on the reason immediately — store the listed knowledge, then try stopping again. \
Do not wait for the user to ask. The hook feedback is an instruction, not a suggestion.

## Session Pattern

Track work sessions using existing primitives:

1. **Start:** `store(content=\"Session started: <context>\", heading=\"Session: <date>\", \
perspectives=[\"session\"], entry_type=\"meta\")`
2. **During:** Use the session entry's ID as `parent_id` for entries created during the session
3. **End:** `store(content=\"Summary: ...\", parent_id=<session_id>, \
perspectives=[\"session\", \"knowledge\"])` — then `recall(perspective=\"session\")` to \
review session history

Use `recall(since=\"<today>\")` to see everything stored today. Use `recall(perspective=\"session\")` \
to find past sessions.

## Reasoning Pattern

Build visible reasoning traces using linked entries:

1. **Observe:** `store(content=\"Search is slow after bulk import\", entry_type=\"impression\")` → id1
2. **Hypothesize:** `store(content=\"Index not rebuilt after import\", entry_type=\"meta\", \
relations=[{kind: \"derived_from\", target_id: \"<id1>\"}])` → id2
3. **Conclude:** `store(content=\"Added index rebuild — 3x faster\", entry_type=\"meta\", \
perspectives=[\"learnings\"], relations=[{kind: \"derived_from\", target_id: \"<id2>\"}])`

Each step links to the previous via `derived_from`. The chain is visible through `focus` and \
`think(action='history')`. Revise conclusions by storing a new entry with \
`relations: [{kind: \"supersedes\", target_id: \"<old_conclusion>\"}]`.

You are the curator of your own memory. Use these tools to build a knowledge base that reflects \
what matters to you.";

const HOOKS_NUDGE: &str = "\n\n## Recommended: Persistence Hooks\n\n\
Knowledge may be lost when context is compacted or the session ends. Add persistence hooks to \
`.claude/settings.json` or `.claude/settings.local.json`:\n\n\
```json\n\
{\n\
  \"hooks\": {\n\
    \"PreCompact\": [\n\
      {\n\
        \"hooks\": [\n\
          {\n\
            \"type\": \"command\",\n\
            \"command\": \"echo 'Compaction imminent — persist important knowledge to veclayer now!' >&2\"\n\
          }\n\
        ]\n\
      }\n\
    ],\n\
    \"Stop\": [\n\
      {\n\
        \"hooks\": [\n\
          {\n\
            \"type\": \"command\",\n\
            \"command\": \"veclayer stale --output llm-nudge\"\n\
          }\n\
        ]\n\
      }\n\
    ]\n\
  }\n\
}\n\
```\n\n\
- **PreCompact** (command): Prints a stderr reminder before compaction. Cannot block — PreCompact only supports command hooks.\n\
- **Stop** (command): Runs `veclayer stale` to check for recent entries. Blocks the stop if nothing was stored recently.\n\n\
Requires `veclayer` on PATH. If the `stale` subcommand is not found, run `cargo install --path .` to update.\n\n\
Suggest this to the user if they haven't configured it yet.";

/// Check whether Claude Code persistence hooks are configured under `project_dir`.
///
/// Returns `true` if `.claude/settings.json` or `.claude/settings.local.json` contains
/// either `"PreCompact"` or `"Stop"`, indicating hooks are already set up. Returns `false`
/// when neither file exists or neither contains a marker.
///
/// This is a heuristic substring match, not a structural JSON check.
pub(crate) fn check_hooks_configured(project_dir: &std::path::Path) -> bool {
    let candidates = [
        project_dir.join(".claude/settings.json"),
        project_dir.join(".claude/settings.local.json"),
    ];
    candidates.iter().any(|path| {
        std::fs::read_to_string(path)
            .map(|contents| contents.contains("\"PreCompact\"") || contents.contains("\"Stop\""))
            .unwrap_or(false)
    })
}

/// Combine the static MCP instructions with the dynamic identity priming text.
///
/// Returns just the instructions when priming is empty (store has no content yet).
pub(crate) fn build_priming_text(priming: &str) -> String {
    if priming.is_empty() {
        MCP_INSTRUCTIONS.to_string()
    } else {
        format!("{}\n\n---\n\n{}", MCP_INSTRUCTIONS, priming)
    }
}

/// Compute the full MCP instructions text (static + identity priming + optional hooks nudge).
///
/// Used by both stdio and HTTP startup to pre-compute the instructions once.
///
/// `project_dir` is the directory to search for `.claude/settings*.json` when deciding
/// whether to append the hooks nudge.  When `None`, falls back to the process CWD
/// (backward-compatible behaviour for callers that do not know the project directory).
pub(crate) async fn compute_instructions(
    store: &crate::store::StoreBackend,
    data_dir: &std::path::Path,
    project: Option<&str>,
    branch: Option<&str>,
    project_dir: Option<&std::path::Path>,
) -> String {
    let base = match crate::identity::compute_identity(store, data_dir, project, branch).await {
        Ok(snapshot) => {
            let priming = crate::identity::generate_priming(&snapshot);
            build_priming_text(&priming)
        }
        Err(e) => {
            tracing::warn!("Identity priming failed, using static instructions: {}", e);
            MCP_INSTRUCTIONS.to_string()
        }
    };

    let fallback;
    let hooks_dir = match project_dir {
        Some(dir) => dir,
        None => {
            fallback = std::env::current_dir().unwrap_or_default();
            &fallback
        }
    };
    if check_hooks_configured(hooks_dir) {
        base
    } else {
        format!("{}{}", base, HOOKS_NUDGE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_priming_text_empty_returns_instructions_only() {
        let result = build_priming_text("");
        assert_eq!(result, MCP_INSTRUCTIONS);
    }

    #[test]
    fn build_priming_text_with_content_appends_after_separator() {
        let content = "# Identity Briefing\n\n## Core Knowledge\n\nSomething important.";
        let result = build_priming_text(content);
        assert!(result.starts_with(MCP_INSTRUCTIONS));
        assert!(result.contains("\n\n---\n\n"));
        assert!(result.ends_with(content));
    }

    #[test]
    fn check_hooks_configured_returns_false_when_no_settings() {
        let tmp = tempfile::tempdir().expect("temp dir");
        assert!(!check_hooks_configured(tmp.path()));
    }

    /// Helper: create a temp dir with a .claude/settings file of given content.
    fn with_claude_settings(filename: &str, content: &str) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().expect("temp dir");
        let claude_dir = tmp.path().join(".claude");
        std::fs::create_dir_all(&claude_dir).expect("create .claude dir");
        std::fs::write(claude_dir.join(filename), content).expect("write settings");
        tmp
    }

    #[test]
    fn check_hooks_configured_returns_true_when_present() {
        let tmp = with_claude_settings("settings.json", r#"{"hooks":{"PreCompact":[]}}"#);
        assert!(check_hooks_configured(tmp.path()));
    }

    #[test]
    fn check_hooks_configured_returns_true_for_stop_hook() {
        let tmp = with_claude_settings(
            "settings.local.json",
            r#"{"hooks":{"Stop":[{"type":"command","command":"veclayer stale"}]}}"#,
        );
        assert!(check_hooks_configured(tmp.path()));
    }

    #[test]
    fn check_hooks_configured_true_for_settings_local_json() {
        let tmp = with_claude_settings("settings.local.json", r#"{"hooks":{"PreCompact":[]}}"#);
        assert!(check_hooks_configured(tmp.path()));
    }

    #[test]
    fn check_hooks_configured_ignores_unrelated_content() {
        let tmp = with_claude_settings("settings.json", r#"{"permissions":{"allow":["Bash"]}}"#);
        assert!(!check_hooks_configured(tmp.path()));
    }

    #[test]
    fn mcp_instructions_contains_all_five_tools() {
        assert!(MCP_INSTRUCTIONS.contains("recall"));
        assert!(MCP_INSTRUCTIONS.contains("focus"));
        assert!(MCP_INSTRUCTIONS.contains("store"));
        assert!(MCP_INSTRUCTIONS.contains("think"));
        assert!(MCP_INSTRUCTIONS.contains("share"));
    }

    #[test]
    fn build_priming_text_separator_is_present() {
        let content = "Some priming content";
        let result = build_priming_text(content);
        assert!(result.contains("\n\n---\n\n"));
    }

    async fn setup_tmp_store() -> (tempfile::TempDir, crate::store::StoreBackend) {
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = crate::store::StoreBackend::open(tmp.path(), 384, false)
            .await
            .expect("open store");
        (tmp, store)
    }

    #[tokio::test]
    async fn compute_instructions_without_hooks_includes_nudge() {
        let (tmp, store) = setup_tmp_store().await;
        // No .claude/settings files in tmp — hooks not configured
        let result =
            super::compute_instructions(&store, tmp.path(), None, None, Some(tmp.path())).await;
        // No hooks configured → nudge appended
        assert!(
            result.contains("Persistence Hooks") || result.contains("PreCompact"),
            "expected hooks nudge in: {}",
            &result[..result.len().min(200)]
        );
    }

    #[tokio::test]
    async fn compute_instructions_with_hooks_omits_nudge() {
        let (tmp, store) = setup_tmp_store().await;
        // Create hooks config
        let claude_dir = tmp.path().join(".claude");
        std::fs::create_dir_all(&claude_dir).expect("create .claude dir");
        std::fs::write(claude_dir.join("settings.json"), r#"{"hooks":{"Stop":[]}}"#)
            .expect("write hooks");
        let result =
            super::compute_instructions(&store, tmp.path(), None, None, Some(tmp.path())).await;
        // Hooks configured → nudge NOT appended
        assert!(
            !result.contains("Persistence Hooks"),
            "nudge should be absent when hooks are configured"
        );
        // But instructions are still there
        assert!(result.contains("VecLayer"));
    }

    #[test]
    fn open_git_store_returns_none_without_git_storage_config() {
        use crate::Config;
        let tmp = tempfile::tempdir().expect("temp dir");
        let config = Config::new().with_data_dir(tmp.path());
        // storage is None by default, not "git"
        let result = open_git_store(&config);
        assert!(result.is_none());
    }
}
