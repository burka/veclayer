//! Hook-integration commands: observe (PostToolUse) and context (SessionStart).

use std::path::Path;

use serde::Deserialize;

use crate::chunk::EntryType;
use crate::{ChunkLevel, HierarchicalChunk, Result, VectorStore};

// ---------------------------------------------------------------------------
// Payload types
// ---------------------------------------------------------------------------

/// Claude Code PostToolUse hook payload.
#[derive(Debug, Deserialize)]
pub struct PostToolUsePayload {
    pub tool_name: String,
    #[serde(default)]
    pub tool_input: serde_json::Value,
    #[serde(default)]
    pub tool_response: Option<serde_json::Value>,
    pub session_id: Option<String>,
}

impl PostToolUsePayload {
    /// Extract success status from tool_response.success (defaults to true if missing).
    fn success(&self) -> bool {
        self.tool_response
            .as_ref()
            .and_then(|r| r.get("success"))
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    /// Extract output text from tool_response (tries .output as string, falls back to compact JSON).
    fn output_text(&self) -> String {
        let Some(resp) = &self.tool_response else {
            return String::new();
        };
        // Try .output as string first
        if let Some(s) = resp.get("output").and_then(|v| v.as_str()) {
            return s.trim().to_string();
        }
        // Try the whole response as a string
        if let Some(s) = resp.as_str() {
            return s.trim().to_string();
        }
        // For complex objects, return empty (don't serialize entire JSON blobs)
        String::new()
    }
}

// ---------------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------------

/// Tools that produce only exploration noise — skip them.
const NOISE_TOOLS: &[&str] = &["Read", "Glob", "Grep", "LS"];

fn is_noise(payload: &PostToolUsePayload) -> bool {
    if !payload.success() {
        return true;
    }
    if payload.tool_name.starts_with("mcp__") {
        return true;
    }
    NOISE_TOOLS.contains(&payload.tool_name.as_str())
}

// ---------------------------------------------------------------------------
// Summary formatting
// ---------------------------------------------------------------------------

/// Extract a short input description appropriate for the tool.
fn input_summary(payload: &PostToolUsePayload) -> String {
    let input = &payload.tool_input;
    match payload.tool_name.as_str() {
        "Bash" => input
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        "Write" | "Edit" => input
            .get("file_path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        "Agent" => input
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        _ => {
            let s = input.to_string();
            truncate_output(&s).to_string()
        }
    }
}

const MAX_OUTPUT_CHARS: usize = 200;

/// Truncate `output` to at most `MAX_OUTPUT_CHARS` characters.
fn truncate_output(output: &str) -> &str {
    if output.len() <= MAX_OUTPUT_CHARS {
        output
    } else {
        // Walk back to a char boundary.
        let mut end = MAX_OUTPUT_CHARS;
        while !output.is_char_boundary(end) {
            end -= 1;
        }
        &output[..end]
    }
}

/// Build the compact one-line summary stored as the chunk content.
pub fn build_summary(payload: &PostToolUsePayload) -> String {
    let inp = input_summary(payload);
    let output = payload.output_text();
    let out_trunc = truncate_output(&output);

    if out_trunc.is_empty() {
        format!("[tool:{}] {}", payload.tool_name, inp)
    } else {
        format!("[tool:{}] {} → {}", payload.tool_name, inp, out_trunc)
    }
}

// ---------------------------------------------------------------------------
// observe command
// ---------------------------------------------------------------------------

/// Process a single PostToolUse payload and store it when it is not noise.
///
/// Separated from stdin reading so that it can be unit-tested directly.
pub async fn process_payload(data_dir: &Path, payload: PostToolUsePayload) -> Result<()> {
    if is_noise(&payload) {
        return Ok(());
    }

    let summary = build_summary(&payload);
    let heading = format!("[observe:{}]", payload.tool_name);

    // Use lightweight metadata-only open (no ONNX model loading)
    let store = crate::store::StoreBackend::open_metadata(data_dir, false).await?;

    let mut chunk = HierarchicalChunk::new(
        summary,
        ChunkLevel::CONTENT,
        None,
        String::new(),
        "[hook:post-tool-use]".to_string(),
    )
    .with_entry_type(EntryType::Impression)
    .with_perspectives(vec!["observations".to_string()]);

    chunk.heading = Some(heading);
    // No embedding — deferred until next recall/rebuild-index

    store.insert_chunks(vec![chunk]).await?;

    Ok(())
}

/// `veclayer observe` — read a PostToolUse hook payload from stdin and store a compact observation.
///
/// Always exits 0; errors are reported to stderr so the hook never blocks the session.
pub async fn observe(data_dir: &Path) -> Result<()> {
    let payload: PostToolUsePayload = match serde_json::from_reader(std::io::stdin()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("veclayer observe: failed to parse stdin JSON: {e}");
            return Ok(());
        }
    };

    if let Err(e) = process_payload(data_dir, payload).await {
        eprintln!("veclayer observe: failed to store observation: {e}");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// context command
// ---------------------------------------------------------------------------

/// `veclayer context` — output an identity briefing for SessionStart hook injection.
///
/// Prints to stdout. If the store is empty or any error occurs, prints nothing.
pub async fn context(data_dir: &Path) -> Result<()> {
    let store = match crate::store::StoreBackend::open_metadata(data_dir, true).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("veclayer context: failed to open store: {e}");
            return Ok(());
        }
    };

    let snapshot = match crate::identity::compute_identity(&store, data_dir, None, None).await {
        Ok(s) => s,
        Err(_) => return Ok(()),
    };

    let priming = crate::identity::generate_priming(&snapshot);
    if !priming.is_empty() {
        print!("{}", priming);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_payload(
        tool_name: &str,
        tool_input: serde_json::Value,
        tool_response: Option<serde_json::Value>,
    ) -> PostToolUsePayload {
        PostToolUsePayload {
            tool_name: tool_name.to_string(),
            tool_input,
            tool_response,
            session_id: None,
        }
    }

    fn bash_payload(command: &str, output: &str) -> PostToolUsePayload {
        make_payload(
            "Bash",
            serde_json::json!({"command": command}),
            Some(serde_json::json!({"output": output, "success": true})),
        )
    }

    // --- filter tests ---

    #[test]
    fn test_observe_filters_read_tool() {
        let p = make_payload(
            "Read",
            serde_json::json!({}),
            Some(serde_json::json!({"output": "file content", "success": true})),
        );
        assert!(is_noise(&p), "Read should be filtered as noise");
    }

    #[test]
    fn test_observe_filters_grep_tool() {
        let p = make_payload(
            "Grep",
            serde_json::json!({}),
            Some(serde_json::json!({"output": "matches", "success": true})),
        );
        assert!(is_noise(&p), "Grep should be filtered as noise");
    }

    #[test]
    fn test_observe_filters_mcp_tool() {
        let p = make_payload(
            "mcp__veclayer__store",
            serde_json::json!({}),
            Some(serde_json::json!({"output": "stored", "success": true})),
        );
        assert!(is_noise(&p), "mcp__ tools should be filtered as noise");
    }

    #[test]
    fn test_observe_filters_failed_tool() {
        let p = make_payload(
            "Bash",
            serde_json::json!({"command": "cargo test"}),
            Some(serde_json::json!({"output": "", "success": false})),
        );
        assert!(is_noise(&p), "failed tools should be filtered as noise");
    }

    // --- format tests ---

    #[test]
    fn test_observe_stores_bash_tool() {
        let p = bash_payload("cargo test", "42 passed, 0 failed");
        assert!(!is_noise(&p));
        let summary = build_summary(&p);
        assert!(summary.contains("[tool:Bash]"));
        assert!(summary.contains("cargo test"));
        assert!(summary.contains("42 passed"));
    }

    #[test]
    fn test_observe_stores_edit_tool() {
        let p = make_payload(
            "Edit",
            serde_json::json!({"file_path": "src/main.rs", "old_string": "x", "new_string": "y"}),
            Some(serde_json::json!({"output": "", "success": true})),
        );
        assert!(!is_noise(&p));
        let summary = build_summary(&p);
        assert!(summary.contains("[tool:Edit]"));
        assert!(summary.contains("src/main.rs"));
    }

    #[test]
    fn test_observe_truncates_long_output() {
        let long_output = "x".repeat(500);
        let p = bash_payload("echo x", &long_output);
        let summary = build_summary(&p);
        // The output portion after → must be at most MAX_OUTPUT_CHARS chars
        let after_arrow = summary.split(" → ").nth(1).unwrap_or("");
        assert!(
            after_arrow.len() <= MAX_OUTPUT_CHARS,
            "output portion should be truncated to {MAX_OUTPUT_CHARS} chars, got {}",
            after_arrow.len()
        );
    }

    #[test]
    fn test_observe_summary_format() {
        let p = bash_payload("cargo test", "42 passed, 0 failed");
        let summary = build_summary(&p);
        assert_eq!(summary, "[tool:Bash] cargo test → 42 passed, 0 failed");
    }

    // --- deserialization tests ---

    #[test]
    fn test_deserialize_real_hook_payload() {
        let json = r#"{
            "session_id": "s1",
            "tool_name": "Bash",
            "tool_input": {"command": "cargo test", "description": "Run tests"},
            "tool_response": {"output": "42 passed, 0 failed", "success": true},
            "tool_use_id": "tu_123",
            "tool_execution_time_ms": 5000,
            "cwd": "/home/user/project"
        }"#;
        let p: PostToolUsePayload = serde_json::from_str(json).unwrap();
        assert_eq!(p.tool_name, "Bash");
        assert!(p.success());
        assert_eq!(p.output_text(), "42 passed, 0 failed");
    }

    #[test]
    fn test_deserialize_failed_tool_payload() {
        let json = r#"{
            "tool_name": "Bash",
            "tool_input": {"command": "false"},
            "tool_response": {"output": "error", "success": false}
        }"#;
        let p: PostToolUsePayload = serde_json::from_str(json).unwrap();
        assert!(!p.success());
        assert!(is_noise(&p));
    }

    #[test]
    fn test_deserialize_missing_tool_response() {
        let json = r#"{
            "tool_name": "Write",
            "tool_input": {"file_path": "foo.rs", "content": "..."}
        }"#;
        let p: PostToolUsePayload = serde_json::from_str(json).unwrap();
        assert!(p.success()); // defaults to true
        assert!(p.output_text().is_empty());
    }

    // --- context tests ---

    #[tokio::test]
    async fn test_context_empty_store() -> Result<()> {
        let dir = TempDir::new()?;
        // Should succeed with no output (empty store → no priming)
        context(dir.path()).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_context_with_entries() -> Result<()> {
        use crate::store::StoreBackend;
        use crate::test_helpers::make_test_chunk;

        let dir = TempDir::new()?;
        let store = StoreBackend::open(dir.path(), 384, false).await?;
        store
            .insert_chunks(vec![
                make_test_chunk("aaa111bbb222ccc333", "Architecture decision: use SQLite"),
                make_test_chunk("ddd444eee555fff666", "Learning: keep functions small"),
            ])
            .await?;
        drop(store);

        // context() writes to stdout; we just ensure it does not error out.
        context(dir.path()).await?;
        Ok(())
    }
}
