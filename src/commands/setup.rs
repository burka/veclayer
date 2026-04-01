//! `veclayer setup` — configure integrations (Claude Code, etc.).

use std::fs;
use std::path::Path;

use serde_json::{json, Value};

use crate::util::{DEFAULT_OLLAMA_DIMENSION, DEFAULT_OLLAMA_EMBED_MODEL, DEFAULT_OLLAMA_URL};
use crate::Result;

/// Alias so existing local references continue to compile.
const DEFAULT_OLLAMA_BASE_URL: &str = DEFAULT_OLLAMA_URL;

// ---------------------------------------------------------------------------
// Claude Code configuration constants
// ---------------------------------------------------------------------------

const VECLAYER_MCP_KEY: &str = "veclayer";

const SESSION_START_COMMAND: &str = "veclayer context --brief";
const POST_TOOL_USE_COMMAND: &str = "veclayer observe";
const PRE_COMPACT_COMMAND: &str = "veclayer stale --output llm-nudge";
const POST_COMPACT_COMMAND: &str = "veclayer context --brief";
const STOP_COMMAND: &str = "veclayer stale --output llm-nudge";

/// Build the canonical Claude Code MCP + hooks configuration block.
fn claude_config() -> Value {
    json!({
        "mcpServers": {
            "veclayer": {
                "command": "veclayer",
                "args": ["serve", "--mcp-stdio"]
            }
        },
        "hooks": {
            "SessionStart": [
                {
                    "matcher": "startup|resume",
                    "hooks": [
                        {
                            "type": "command",
                            "command": SESSION_START_COMMAND,
                            "timeout": 10
                        }
                    ]
                }
            ],
            "PostToolUse": [
                {
                    "matcher": "Bash|Write|Edit|WebFetch|WebSearch|Agent",
                    "hooks": [
                        {
                            "type": "command",
                            "command": POST_TOOL_USE_COMMAND,
                            "timeout": 10
                        }
                    ]
                }
            ],
            "PreCompact": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": PRE_COMPACT_COMMAND
                        }
                    ]
                }
            ],
            "PostCompact": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": POST_COMPACT_COMMAND,
                            "timeout": 10
                        }
                    ]
                }
            ],
            "Stop": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": STOP_COMMAND
                        }
                    ]
                }
            ]
        }
    })
}

// ---------------------------------------------------------------------------
// Top-level: veclayer setup
// ---------------------------------------------------------------------------

/// Print an overview of available setup targets.
pub fn setup() {
    println!("VecLayer Setup\n");
    println!("Available integrations:\n");
    println!("  veclayer setup ollama");
    println!("      Print local Ollama embedding setup and config.");
    println!();
    println!("  veclayer setup ollama --apply");
    println!("      Write .veclayer/config.toml with an external Ollama embedder.");
    println!("      Existing non-embedder settings are preserved.");
    println!();
    println!("  veclayer setup ollama --apply --global");
    println!("      Write ~/.config/veclayer/config.toml (or VECLAYER_USER_CONFIG path).");
    println!();
    println!("  veclayer setup claude");
    println!("      Print the Claude Code configuration snippet (MCP server + 4 hooks).");
    println!();
    println!("  veclayer setup claude --apply");
    println!("      Merge configuration into .claude/settings.json (project only).");
    println!("      Safe to run on an existing file — preserves all current settings.");
    println!();
    println!("  veclayer setup claude --apply --global");
    println!("      Merge configuration into ~/.claude/settings.json (all projects).");
    println!();
    println!("Run `veclayer setup <target> --help` for details on what gets configured.");
}

fn ollama_embedder_toml(model: &str, base_url: &str, dimension: usize) -> String {
    format!(
        "[embedder]\n\
type = \"ollama\"\n\
model = \"{model}\"\n\
base_url = \"{base_url}\"\n\
dimension = {dimension}\n"
    )
}

pub fn setup_ollama() {
    println!("## Local Ollama Embedding Setup\n");
    println!("VecLayer already supports external embedding providers.");
    println!(
        "This setup keeps VecLayer working out of the box by leaving the default build unchanged,"
    );
    println!(
        "but configures this installation to use a local Ollama server instead of FastEmbed.\n"
    );
    println!("1. Install and start Ollama.");
    println!("2. Pull the embedding model:");
    println!("   ollama pull {DEFAULT_OLLAMA_EMBED_MODEL}\n");
    println!("3. Write this config to .veclayer/config.toml (or use --apply):\n");
    println!("```toml");
    print!(
        "{}",
        ollama_embedder_toml(
            DEFAULT_OLLAMA_EMBED_MODEL,
            DEFAULT_OLLAMA_BASE_URL,
            DEFAULT_OLLAMA_DIMENSION
        )
    );
    println!("```\n");
    println!("Environment-variable equivalent:\n");
    println!("  VECLAYER_EMBEDDER=ollama");
    println!("  VECLAYER_OLLAMA_MODEL={DEFAULT_OLLAMA_EMBED_MODEL}");
    println!("  VECLAYER_OLLAMA_URL={DEFAULT_OLLAMA_BASE_URL}");
    println!("  VECLAYER_OLLAMA_DIMENSION={DEFAULT_OLLAMA_DIMENSION}\n");
    println!("To apply automatically:");
    println!("  veclayer setup ollama --apply");
    println!("  veclayer setup ollama --apply --global");
}

fn ollama_config_path(cwd: &Path, global: bool) -> std::path::PathBuf {
    if global {
        crate::config::user_config_path()
    } else {
        cwd.join(".veclayer/config.toml")
    }
}

fn read_existing_toml(path: &Path) -> Result<toml::Value> {
    if !path.exists() {
        return Ok(toml::Value::Table(toml::map::Map::new()));
    }

    let content = fs::read_to_string(path)?;
    toml::from_str(&content).map_err(|e| {
        crate::Error::config(format!(
            "Cannot parse {}: {}. Fix the TOML syntax before running --apply.",
            path.display(),
            e
        ))
    })
}

fn merge_ollama_embedder_config(
    existing: toml::Value,
    model: &str,
    base_url: &str,
    dimension: usize,
) -> toml::Value {
    let mut root = match existing {
        toml::Value::Table(table) => table,
        _ => toml::map::Map::new(),
    };

    let embedder = root
        .entry("embedder")
        .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
    let embedder_table = embedder.as_table_mut().expect("embedder must be a table");

    embedder_table.insert(
        "type".to_string(),
        toml::Value::String("ollama".to_string()),
    );
    embedder_table.insert("model".to_string(), toml::Value::String(model.to_string()));
    embedder_table.insert(
        "base_url".to_string(),
        toml::Value::String(base_url.to_string()),
    );
    embedder_table.insert(
        "dimension".to_string(),
        toml::Value::Integer(dimension as i64),
    );

    toml::Value::Table(root)
}

pub fn setup_ollama_apply(
    cwd: &Path,
    global: bool,
    model: Option<&str>,
    base_url: Option<&str>,
    dimension: Option<usize>,
) -> Result<()> {
    let model = model.unwrap_or(DEFAULT_OLLAMA_EMBED_MODEL);
    let base_url = base_url.unwrap_or(DEFAULT_OLLAMA_BASE_URL);
    let dimension = dimension.unwrap_or(DEFAULT_OLLAMA_DIMENSION);

    let config_path = ollama_config_path(cwd, global);
    let existing = read_existing_toml(&config_path)?;
    let merged = merge_ollama_embedder_config(existing, model, base_url, dimension);
    let rendered = toml::to_string_pretty(&merged)
        .map_err(|e| crate::Error::config(format!("TOML serialization failed: {e}")))?;

    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&config_path, format!("{rendered}\n"))?;

    println!("Ollama embedder configured in {}:", config_path.display());
    println!("  model = {model}");
    println!("  base_url = {base_url}");
    println!("  dimension = {dimension}");
    println!();
    println!("Next step:");
    println!("  ollama pull {model}");

    Ok(())
}

// ---------------------------------------------------------------------------
// veclayer setup claude (no --apply)
// ---------------------------------------------------------------------------

/// Print the Claude Code configuration snippet with explanation.
pub fn setup_claude() {
    let config = claude_config();
    let json_str = serde_json::to_string_pretty(&config).expect("serialisation is infallible");

    println!("## Claude Code Integration\n");
    println!("Configures the following in .claude/settings.json:\n");
    println!("  MCP server:  veclayer serve --mcp-stdio");
    println!("               Exposes recall, store, focus, think and other tools via MCP.\n");
    println!("  SessionStart → {SESSION_START_COMMAND}");
    println!("               Injects a memory briefing at the start of each session.\n");
    println!("  PostToolUse  → {POST_TOOL_USE_COMMAND}");
    println!("               Captures compact observations for Bash/Write/Edit/WebFetch/WebSearch/Agent.\n");
    println!("  PreCompact   → {PRE_COMPACT_COMMAND}");
    println!("               Nudges the agent to persist knowledge before compaction.\n");
    println!("  PostCompact  → {POST_COMPACT_COMMAND}");
    println!("               Re-injects memory briefing after context compaction.\n");
    println!("  Stop         → veclayer stale");
    println!(
        "               Warns the agent at session end if memory hasn't been updated recently.\n"
    );
    println!("JSON snippet (add to .claude/settings.json manually, or use --apply):\n");
    println!("```json");
    println!("{json_str}");
    println!("```\n");
    println!("To apply automatically:");
    println!("  veclayer setup claude --apply           (project: .claude/settings.json)");
    println!("  veclayer setup claude --apply --global  (global:  ~/.claude/settings.json)");
}

// ---------------------------------------------------------------------------
// veclayer setup claude --apply
// ---------------------------------------------------------------------------

/// Actions taken (or skipped) during an apply operation — used for reporting and testing.
#[derive(Debug, Default)]
pub struct ApplyReport {
    pub mcp_server_added: bool,
    pub mcp_server_skipped: bool,
    pub session_start_added: bool,
    pub session_start_skipped: bool,
    pub post_tool_use_added: bool,
    pub post_tool_use_skipped: bool,
    pub pre_compact_added: bool,
    pub pre_compact_skipped: bool,
    pub post_compact_added: bool,
    pub post_compact_skipped: bool,
    pub stop_added: bool,
    pub stop_skipped: bool,
}

/// Merge veclayer configuration into an existing settings value.
///
/// Returns the merged value and a report of what was added vs skipped.
/// Pure function — no file I/O — so it can be unit-tested directly.
pub fn merge_claude_config(existing: Value) -> (Value, ApplyReport) {
    let mut root = match existing {
        Value::Object(map) => map,
        _ => serde_json::Map::new(),
    };

    let mut report = ApplyReport::default();

    // --- MCP server ---
    let mcp_servers = root
        .entry("mcpServers")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .expect("mcpServers must be an object");

    if mcp_servers.contains_key(VECLAYER_MCP_KEY) {
        report.mcp_server_skipped = true;
    } else {
        mcp_servers.insert(
            VECLAYER_MCP_KEY.to_string(),
            json!({ "command": "veclayer", "args": ["serve", "--mcp-stdio"] }),
        );
        report.mcp_server_added = true;
    }

    // --- Hooks ---
    let hooks = root
        .entry("hooks")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .expect("hooks must be an object");

    merge_hook(
        hooks,
        "SessionStart",
        SESSION_START_COMMAND,
        || {
            json!({
                "matcher": "startup|resume",
                "hooks": [{ "type": "command", "command": SESSION_START_COMMAND, "timeout": 10 }]
            })
        },
        &mut report.session_start_added,
        &mut report.session_start_skipped,
    );

    merge_hook(
        hooks,
        "PostToolUse",
        POST_TOOL_USE_COMMAND,
        || {
            json!({
                "matcher": "Bash|Write|Edit|WebFetch|WebSearch|Agent",
                "hooks": [{ "type": "command", "command": POST_TOOL_USE_COMMAND, "timeout": 10 }]
            })
        },
        &mut report.post_tool_use_added,
        &mut report.post_tool_use_skipped,
    );

    merge_hook(
        hooks,
        "PreCompact",
        PRE_COMPACT_COMMAND,
        || {
            json!({
                "hooks": [{ "type": "command", "command": PRE_COMPACT_COMMAND }]
            })
        },
        &mut report.pre_compact_added,
        &mut report.pre_compact_skipped,
    );

    merge_hook(
        hooks,
        "PostCompact",
        POST_COMPACT_COMMAND,
        || {
            json!({
                "hooks": [{ "type": "command", "command": POST_COMPACT_COMMAND }]
            })
        },
        &mut report.post_compact_added,
        &mut report.post_compact_skipped,
    );

    merge_hook(
        hooks,
        "Stop",
        STOP_COMMAND,
        || {
            json!({
                "hooks": [{ "type": "command", "command": STOP_COMMAND }]
            })
        },
        &mut report.stop_added,
        &mut report.stop_skipped,
    );

    (Value::Object(root), report)
}

/// Merge a single hook entry into the hooks map for a given event type.
///
/// Checks every entry in the existing array for any hook whose `command`
/// field contains `marker`. If found, marks as skipped; otherwise appends.
fn merge_hook(
    hooks: &mut serde_json::Map<String, Value>,
    event: &str,
    marker: &str,
    build_entry: impl FnOnce() -> Value,
    added: &mut bool,
    skipped: &mut bool,
) {
    let array = hooks
        .entry(event)
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .expect("hook event value must be an array");

    if hook_array_has_command(array, marker) {
        *skipped = true;
    } else {
        array.push(build_entry());
        *added = true;
    }
}

/// Return true if any entry in the hook array already has a `command` containing `needle`.
fn hook_array_has_command(array: &[Value], needle: &str) -> bool {
    array
        .iter()
        .any(|entry| entry_contains_command(entry, needle))
}

/// Walk a hook entry (which may nest `hooks: [{command: "..."}]`) looking for `needle`.
fn entry_contains_command(entry: &Value, needle: &str) -> bool {
    // Direct command field on the entry itself
    if let Some(cmd) = entry.get("command").and_then(|v| v.as_str()) {
        if cmd.contains(needle) {
            return true;
        }
    }
    // Inner hooks array
    if let Some(inner) = entry.get("hooks").and_then(|v| v.as_array()) {
        for hook in inner {
            if let Some(cmd) = hook.get("command").and_then(|v| v.as_str()) {
                if cmd.contains(needle) {
                    return true;
                }
            }
        }
    }
    false
}

/// Resolve the `.claude` directory based on whether `--global` was requested.
///
/// Global mode uses `$HOME/.claude`; project mode uses `cwd/.claude`.
/// Pure function — no I/O — so it can be unit-tested without touching the filesystem.
fn settings_dir(cwd: &Path, global: bool) -> Result<std::path::PathBuf> {
    if global {
        let home = std::env::var("HOME").map_err(|_| {
            crate::Error::config("HOME not set — cannot determine global settings path")
        })?;
        Ok(std::path::PathBuf::from(home).join(".claude"))
    } else {
        Ok(cwd.join(".claude"))
    }
}

/// Apply Claude Code configuration to `.claude/settings.json`.
///
/// When `global` is true, writes to `~/.claude/settings.json` (all projects).
/// When `global` is false, writes to `cwd/.claude/settings.json` (project only).
pub fn setup_claude_apply(cwd: &Path, global: bool) -> Result<()> {
    let settings_dir = settings_dir(cwd, global)?;
    let settings_path = settings_dir.join("settings.json");

    let existing = read_existing_settings(&settings_path)?;
    let (merged, report) = merge_claude_config(existing);
    let json_str = serde_json::to_string_pretty(&merged).expect("serialisation is infallible");

    fs::create_dir_all(&settings_dir)?;
    fs::write(&settings_path, format!("{json_str}\n"))?;

    print_apply_report(&settings_path.display().to_string(), &report);

    Ok(())
}

/// Read and parse an existing settings file, returning `{}` if absent.
///
/// Returns an error if the file exists but contains invalid JSON — we never
/// silently overwrite a file we cannot parse.
fn read_existing_settings(path: &Path) -> Result<Value> {
    if !path.exists() {
        return Ok(json!({}));
    }
    let content = fs::read_to_string(path)?;
    serde_json::from_str(&content).map_err(|e| {
        crate::Error::config(format!(
            "Cannot parse {}: {}. Fix the JSON syntax before running --apply.",
            path.display(),
            e
        ))
    })
}

/// Print the human-readable apply report.
fn print_apply_report(path: &str, report: &ApplyReport) {
    println!("Claude Code integration configured in {path}:");

    print_action(
        "MCP server: veclayer",
        report.mcp_server_added,
        report.mcp_server_skipped,
    );
    print_action(
        "Hook: SessionStart → veclayer context --brief",
        report.session_start_added,
        report.session_start_skipped,
    );
    print_action(
        "Hook: PostToolUse → veclayer observe",
        report.post_tool_use_added,
        report.post_tool_use_skipped,
    );
    print_action(
        "Hook: PreCompact → veclayer stale",
        report.pre_compact_added,
        report.pre_compact_skipped,
    );
    print_action(
        "Hook: PostCompact → veclayer context --brief",
        report.post_compact_added,
        report.post_compact_skipped,
    );
    print_action(
        "Hook: Stop → veclayer stale",
        report.stop_added,
        report.stop_skipped,
    );
}

fn print_action(label: &str, added: bool, skipped: bool) {
    if added {
        println!("  + {label}");
    } else if skipped {
        println!("  ~ {label} (already configured, skipped)");
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn empty() -> Value {
        json!({})
    }

    fn with_existing_mcp() -> Value {
        json!({
            "mcpServers": {
                "other-tool": { "command": "other", "args": [] }
            }
        })
    }

    fn with_veclayer_mcp() -> Value {
        json!({
            "mcpServers": {
                "veclayer": { "command": "veclayer", "args": ["serve", "--mcp-stdio"] }
            }
        })
    }

    fn with_existing_stop_hook() -> Value {
        json!({
            "hooks": {
                "Stop": [
                    {
                        "hooks": [
                            { "type": "command", "command": "veclayer stale --output llm-nudge" }
                        ]
                    }
                ]
            }
        })
    }

    fn with_partial_hooks() -> Value {
        json!({
            "hooks": {
                "Stop": [
                    {
                        "hooks": [
                            { "type": "command", "command": "veclayer stale --output llm-nudge" }
                        ]
                    }
                ],
                "SessionStart": [
                    {
                        "matcher": "startup|resume",
                        "hooks": [
                            { "type": "command", "command": "veclayer context --brief", "timeout": 10 }
                        ]
                    }
                ]
            }
        })
    }

    // --- test_merge_into_empty_file ---

    #[test]
    fn test_merge_into_empty_file() {
        let (result, report) = merge_claude_config(empty());

        // MCP server present
        assert!(result["mcpServers"]["veclayer"].is_object());
        assert_eq!(result["mcpServers"]["veclayer"]["command"], "veclayer");

        // All four hook types present
        assert!(result["hooks"]["SessionStart"].is_array());
        assert!(result["hooks"]["PostToolUse"].is_array());
        assert!(result["hooks"]["PostCompact"].is_array());
        assert!(result["hooks"]["Stop"].is_array());

        // Everything was added
        assert!(report.mcp_server_added);
        assert!(report.session_start_added);
        assert!(report.post_tool_use_added);
        assert!(report.post_compact_added);
        assert!(report.stop_added);

        // Nothing was skipped
        assert!(!report.mcp_server_skipped);
        assert!(!report.session_start_skipped);
        assert!(!report.post_tool_use_skipped);
        assert!(!report.post_compact_skipped);
        assert!(!report.stop_skipped);
    }

    // --- test_merge_preserves_existing_settings ---

    #[test]
    fn test_merge_preserves_existing_settings() {
        let existing = json!({
            "permissions": { "allow": ["Bash"] },
            "mcpServers": {
                "other-tool": { "command": "other", "args": [] }
            },
            "hooks": {
                "PostToolUse": [
                    {
                        "matcher": "CustomTool",
                        "hooks": [{ "type": "command", "command": "my-custom-hook" }]
                    }
                ]
            }
        });

        let (result, report) = merge_claude_config(existing);

        // Existing settings preserved
        assert_eq!(result["permissions"]["allow"][0], "Bash");

        // Existing MCP server preserved
        assert!(result["mcpServers"]["other-tool"].is_object());

        // Veclayer MCP added alongside
        assert!(result["mcpServers"]["veclayer"].is_object());
        assert!(report.mcp_server_added);

        // Existing hook preserved
        let post_tool_use = result["hooks"]["PostToolUse"].as_array().unwrap();
        assert_eq!(post_tool_use.len(), 2, "existing hook + veclayer hook");
        let empty = vec![];
        let commands: Vec<&str> = post_tool_use
            .iter()
            .flat_map(|entry| {
                entry["hooks"]
                    .as_array()
                    .unwrap_or(&empty)
                    .iter()
                    .filter_map(|h| h["command"].as_str())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(commands.contains(&"my-custom-hook"));
        assert!(commands.contains(&POST_TOOL_USE_COMMAND));
    }

    // --- test_merge_skips_duplicate_hooks ---

    #[test]
    fn test_merge_skips_duplicate_hooks() {
        // First apply
        let (after_first, _) = merge_claude_config(empty());

        // Second apply — should skip everything
        let (after_second, report) = merge_claude_config(after_first.clone());

        assert!(report.mcp_server_skipped);
        assert!(report.session_start_skipped);
        assert!(report.post_tool_use_skipped);
        assert!(report.post_compact_skipped);
        assert!(report.stop_skipped);

        assert!(!report.mcp_server_added);
        assert!(!report.session_start_added);
        assert!(!report.post_tool_use_added);
        assert!(!report.post_compact_added);
        assert!(!report.stop_added);

        // Result is structurally identical
        assert_eq!(after_first, after_second);
    }

    // --- test_merge_skips_existing_veclayer_mcp ---

    #[test]
    fn test_merge_skips_existing_veclayer_mcp() {
        let (_, report) = merge_claude_config(with_veclayer_mcp());
        assert!(report.mcp_server_skipped);
        assert!(!report.mcp_server_added);
    }

    // --- test_merge_adds_missing_hooks ---

    #[test]
    fn test_merge_adds_missing_hooks() {
        // Config has Stop and SessionStart; PostToolUse and PreCompact are missing.
        let (result, report) = merge_claude_config(with_partial_hooks());

        // Stop and SessionStart were already there → skipped
        assert!(report.stop_skipped);
        assert!(report.session_start_skipped);
        assert!(!report.stop_added);
        assert!(!report.session_start_added);

        // PostToolUse and PreCompact should be added
        assert!(report.post_tool_use_added);
        assert!(report.post_compact_added);

        // Both arrays exist now
        assert!(result["hooks"]["PostToolUse"].is_array());
        assert!(result["hooks"]["PostCompact"].is_array());
    }

    // --- test_merge_detects_existing_stop_hook ---

    #[test]
    fn test_merge_detects_existing_stop_hook() {
        let (_, report) = merge_claude_config(with_existing_stop_hook());
        assert!(report.stop_skipped);
        assert!(!report.stop_added);
    }

    // --- test_merge_adds_veclayer_to_existing_mcp ---

    #[test]
    fn test_merge_adds_veclayer_to_existing_mcp() {
        let (result, report) = merge_claude_config(with_existing_mcp());
        assert!(
            result["mcpServers"]["other-tool"].is_object(),
            "other tool preserved"
        );
        assert!(
            result["mcpServers"]["veclayer"].is_object(),
            "veclayer added"
        );
        assert!(report.mcp_server_added);
    }

    // --- apply report helpers ---

    #[test]
    fn test_print_apply_report_all_added() {
        // Just verify it doesn't panic with all-added state
        let report = ApplyReport {
            mcp_server_added: true,
            session_start_added: true,
            post_tool_use_added: true,
            post_compact_added: true,
            stop_added: true,
            ..Default::default()
        };
        print_apply_report(".claude/settings.json", &report);
    }

    #[test]
    fn test_print_apply_report_all_skipped() {
        let report = ApplyReport {
            mcp_server_skipped: true,
            session_start_skipped: true,
            post_tool_use_skipped: true,
            post_compact_skipped: true,
            stop_skipped: true,
            ..Default::default()
        };
        print_apply_report(".claude/settings.json", &report);
    }

    // --- file I/O integration test ---

    // --- settings_dir helper ---

    #[test]
    fn test_settings_dir_project() {
        use std::path::{Path, PathBuf};
        let dir = Path::new("/tmp/myproject");
        let result = settings_dir(dir, false).unwrap();
        assert_eq!(result, PathBuf::from("/tmp/myproject/.claude"));
    }

    #[test]
    fn test_settings_dir_global() {
        // HOME is always set in CI/dev environments; verify path ends with .claude
        let result = settings_dir(std::path::Path::new("/unused"), true);
        assert!(result.is_ok());
        let path = result.unwrap();
        assert!(
            path.to_str().unwrap().ends_with(".claude"),
            "expected path ending in .claude, got: {}",
            path.display()
        );
    }

    // --- file I/O integration tests ---

    #[test]
    fn test_setup_claude_apply_creates_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let result = setup_claude_apply(dir.path(), false);
        assert!(result.is_ok());

        let settings_path = dir.path().join(".claude").join("settings.json");
        assert!(settings_path.exists());

        let content = fs::read_to_string(&settings_path).unwrap();
        let parsed: Value = serde_json::from_str(&content).unwrap();
        assert!(parsed["mcpServers"]["veclayer"].is_object());
        assert!(parsed["hooks"]["Stop"].is_array());
    }

    #[test]
    fn test_setup_claude_apply_idempotent() {
        let dir = tempfile::TempDir::new().unwrap();
        let settings_path = dir.path().join(".claude").join("settings.json");

        // First apply
        setup_claude_apply(dir.path(), false).unwrap();
        let content_first = fs::read_to_string(&settings_path).unwrap();

        // Second apply
        setup_claude_apply(dir.path(), false).unwrap();
        let content_second = fs::read_to_string(&settings_path).unwrap();

        assert_eq!(content_first, content_second, "apply must be idempotent");
    }

    #[test]
    fn test_setup_claude_apply_rejects_malformed_json() {
        let dir = tempfile::TempDir::new().unwrap();
        let settings_dir_path = dir.path().join(".claude");
        fs::create_dir_all(&settings_dir_path).unwrap();
        fs::write(
            settings_dir_path.join("settings.json"),
            "{ invalid json, trailing comma, }",
        )
        .unwrap();

        let result = setup_claude_apply(dir.path(), false);
        assert!(result.is_err(), "should reject malformed JSON");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Fix the JSON syntax"),
            "error should guide the user, got: {msg}"
        );
    }

    #[test]
    fn test_setup_ollama_apply_creates_project_config() {
        let dir = tempfile::TempDir::new().unwrap();

        setup_ollama_apply(
            dir.path(),
            false,
            Some("nomic-embed-text"),
            Some(crate::util::DEFAULT_OLLAMA_URL),
            Some(768),
        )
        .unwrap();

        let config_path = dir.path().join(".veclayer").join("config.toml");
        let content = fs::read_to_string(&config_path).unwrap();
        assert!(content.contains("type = \"ollama\""));
        assert!(content.contains("model = \"nomic-embed-text\""));
        assert!(content.contains("base_url = \"http://localhost:11434\""));
        assert!(content.contains("dimension = 768"));
    }

    #[test]
    fn test_setup_ollama_apply_preserves_existing_settings() {
        let dir = tempfile::TempDir::new().unwrap();
        let veclayer_dir = dir.path().join(".veclayer");
        fs::create_dir_all(&veclayer_dir).unwrap();
        fs::write(
            veclayer_dir.join("config.toml"),
            "project = \"demo\"\n[auth]\nauth_required = true\n",
        )
        .unwrap();

        setup_ollama_apply(dir.path(), false, None, None, None).unwrap();

        let content = fs::read_to_string(veclayer_dir.join("config.toml")).unwrap();
        assert!(content.contains("project = \"demo\""));
        assert!(content.contains("[auth]"));
        assert!(content.contains("auth_required = true"));
        assert!(content.contains("[embedder]"));
        assert!(content.contains("type = \"ollama\""));
    }
}
