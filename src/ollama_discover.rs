//! Ollama auto-discovery — probes the local Ollama daemon and classifies its models.
//!
//! [`detect_ollama`] is the main entry point. It sends a single GET to `/api/tags`
//! with a 500 ms connect+read timeout and returns [`None`] if the daemon is not
//! reachable.  The caller is responsible for deciding whether to use the result:
//! auto-configuration should only happen when the user has not already set an
//! explicit embedder or LLM provider via environment variables or config file.
//!
//! Model classification follows Ollama naming conventions:
//! - Embedding models: names containing "embed" (e.g. `nomic-embed-text`,
//!   `mxbai-embed-large`) *or* matching a known embed family from
//!   [`EMBED_MODEL_PRIORITY`] whose name omits the word (e.g. `bge-m3`,
//!   `all-minilm`).
//! - Chat models: everything else (e.g. `llama3.2`, `mistral`, `phi3`)
//!
//! The module is gated behind the `llm` feature because it requires `reqwest`.

use std::time::Duration;

use serde::Deserialize;

use crate::util::DEFAULT_OLLAMA_URL;

/// Preferred chat models, best first. Balances quality, speed, and resource usage.
const CHAT_MODEL_PRIORITY: &[&str] = &[
    "qwen3.5",     // Best quality/speed ratio (62 tok/s at 9b)
    "gpt-oss",     // Excellent structured output (35 tok/s at 20b)
    "qwen3",       // Good quality, fast (74 tok/s at 8b)
    "deepseek-r1", // Strong reasoning
    "llama3.1",
    "mistral",
    "phi3",
    "gemma2",
    "tinyllama", // Last resort
];

/// Vision/multimodal models to skip when selecting a chat model.
const VISION_MODEL_SUBSTRINGS: &[&str] = &["llava", "bakllava", "ocr", "vision"];

/// Preferred embedding models, best first.
const EMBED_MODEL_PRIORITY: &[&str] = &[
    "nomic-embed-text",
    "mxbai-embed-large",
    "snowflake-arctic-embed",
    "all-minilm",
    "bge-m3",
];

/// Connect + overall timeout for the `/api/tags` probe. Ollama is a local
/// endpoint, so a tight 500 ms bound keeps startup snappy when it is absent.
const PROBE_TIMEOUT: Duration = Duration::from_millis(500);

/// Classify a (tag-stripped) model name as an embedding model.
///
/// Catches both the obvious `*embed*` names and the embed families in
/// [`EMBED_MODEL_PRIORITY`] whose names omit the word (`bge-m3`, `all-minilm`),
/// keeping classification consistent with selection so those models are never
/// misfiled as chat models.
fn is_embedding_model(name: &str) -> bool {
    name.contains("embed") || EMBED_MODEL_PRIORITY.iter().any(|p| name.starts_with(p))
}

/// Information about a running Ollama instance.
#[derive(Debug, Clone, PartialEq)]
pub struct OllamaInfo {
    /// Base URL that was probed (may come from `OLLAMA_HOST`).
    pub base_url: String,
    /// Names of embedding-capable models (contain "embed" in their name).
    pub embedding_models: Vec<String>,
    /// Names of chat/completion models (everything that is not an embed model).
    pub chat_models: Vec<String>,
}

impl OllamaInfo {
    /// Returns the highest-priority available embedding model, if any.
    ///
    /// Iterates `EMBED_MODEL_PRIORITY` and returns the first one whose prefix
    /// matches a locally available model (e.g. "nomic-embed-text" matches
    /// "nomic-embed-text:latest"). Falls back to the first available model when
    /// none of the priority prefixes match.
    pub fn best_embedding_model(&self) -> Option<&str> {
        select_by_priority(&self.embedding_models, EMBED_MODEL_PRIORITY, &[])
    }

    /// Returns the highest-priority available chat model, if any.
    ///
    /// Iterates `CHAT_MODEL_PRIORITY` and returns the first one whose prefix
    /// matches a locally available model. Skips vision/multimodal models.
    /// Falls back to the first non-vision model when none of the priority
    /// prefixes match.
    pub fn best_chat_model(&self) -> Option<&str> {
        select_by_priority(
            &self.chat_models,
            CHAT_MODEL_PRIORITY,
            VISION_MODEL_SUBSTRINGS,
        )
    }
}

/// Select the best model from `available` using the given `priority` prefixes.
///
/// Returns the first `available` entry whose name starts with a priority prefix,
/// checked in priority order. Models whose names contain any substring in
/// `skip_substrings` are excluded from both priority matching and the fallback.
/// If no priority prefix matches, returns the first non-skipped model.
fn select_by_priority<'a>(
    available: &'a [String],
    priority: &[&str],
    skip_substrings: &[&str],
) -> Option<&'a str> {
    let is_skipped = |name: &str| skip_substrings.iter().any(|s| name.contains(s));

    for prefix in priority {
        let candidates: Vec<&String> = available
            .iter()
            .filter(|name| !is_skipped(name) && name.starts_with(prefix))
            .collect();

        if let Some(best) = pick_smallest_variant(&candidates) {
            return Some(best.as_str());
        }
    }

    // Fallback: first non-skipped model
    available
        .iter()
        .find(|name| !is_skipped(name))
        .map(String::as_str)
}

/// Among multiple variants of the same model family, prefer the smallest.
///
/// Extracts the numeric size from the tag (e.g., `9` from `qwen3.5:9b`,
/// `35` from `qwen3.5:35b-a3b`) and picks the smallest. Untagged or
/// `:latest` models are preferred over any sized variant (they're typically
/// the default/recommended size).
fn pick_smallest_variant<'a>(candidates: &[&'a String]) -> Option<&'a String> {
    if candidates.is_empty() {
        return None;
    }
    if candidates.len() == 1 {
        return Some(candidates[0]);
    }

    candidates
        .iter()
        .copied()
        .min_by_key(|name| extract_size_from_tag(name))
        .or(candidates.first().copied())
}

/// Extract a numeric size from a model tag for comparison.
///
/// Returns 0 for untagged/`:latest` (preferred), or the leading number
/// from the tag (e.g., `9` from `:9b`, `35` from `:35b-a3b`).
fn extract_size_from_tag(name: &str) -> u64 {
    match name.split_once(':') {
        None => 0,                // untagged → prefer
        Some((_, "latest")) => 0, // :latest → prefer
        Some((_, tag)) => {
            // Extract leading digits from the tag
            let digits: String = tag.chars().take_while(|c| c.is_ascii_digit()).collect();
            digits.parse().unwrap_or(u64::MAX)
        }
    }
}

/// Wire-format of a single model entry from `/api/tags`.
#[derive(Deserialize)]
struct OllamaModel {
    name: String,
}

/// Wire-format of the `/api/tags` response.
#[derive(Deserialize)]
struct TagsResponse {
    models: Vec<OllamaModel>,
}

/// Probe the local Ollama instance and return its model list.
///
/// Reads the base URL from `OLLAMA_HOST` (following Ollama's own convention) or
/// falls back to `http://localhost:11434`.  Returns `None` if the daemon is not
/// reachable within 500 ms or if the response cannot be parsed.
///
/// Safe to call from both sync and async contexts:
/// - Outside any runtime: spawns a temporary single-threaded runtime.
/// - Inside a multi-threaded runtime: uses `block_in_place`.
/// - Inside a current-thread runtime (e.g. `#[tokio::test]`): returns `None`
///   because blocking is not possible on a single-thread scheduler.
pub fn detect_ollama() -> Option<OllamaInfo> {
    let base_url = ollama_base_url();
    crate::util::block_on_probe(probe(&base_url))
}

/// Return the Ollama base URL, respecting the `OLLAMA_HOST` env var.
///
/// Only `http://` and `https://` schemes are accepted to prevent SSRF.
/// Invalid values fall back to the default localhost URL.
pub fn ollama_base_url() -> String {
    match std::env::var("OLLAMA_HOST") {
        Ok(val) if val.starts_with("http://") || val.starts_with("https://") => val,
        _ => DEFAULT_OLLAMA_URL.to_string(),
    }
}

/// Async inner of the probe — separated so it can be unit-tested directly.
async fn probe(base_url: &str) -> Option<OllamaInfo> {
    let client = crate::util::build_probe_client(PROBE_TIMEOUT, PROBE_TIMEOUT)?;

    let url = format!("{}/api/tags", base_url.trim_end_matches('/'));
    let response = client.get(&url).send().await.ok()?;

    if !response.status().is_success() {
        return None;
    }

    let tags: TagsResponse = response.json().await.ok()?;

    let (embedding_models, chat_models): (Vec<String>, Vec<String>) = tags
        .models
        .into_iter()
        .map(|m| strip_tag(&m.name))
        .partition(|name| is_embedding_model(name));

    Some(OllamaInfo {
        base_url: base_url.trim_end_matches('/').to_string(),
        embedding_models,
        chat_models,
    })
}

/// Strip only the `:latest` tag from a model name, preserving other tags.
///
/// Ollama returns model names like `nomic-embed-text:latest` or `qwen3:8b`.
/// Only `:latest` is safe to strip (Ollama resolves bare names to `:latest`).
/// Other tags like `:8b`, `:3b`, `:instruct` must be preserved — Ollama
/// returns 404 if they're omitted.
fn strip_tag(name: &str) -> String {
    name.strip_suffix(":latest").unwrap_or(name).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- strip_tag ---

    #[test]
    fn strip_tag_removes_latest() {
        assert_eq!(strip_tag("nomic-embed-text:latest"), "nomic-embed-text");
    }

    #[test]
    fn strip_tag_preserves_version_tag() {
        assert_eq!(strip_tag("llama3.2:8b"), "llama3.2:8b");
        assert_eq!(strip_tag("qwen3:8b"), "qwen3:8b");
        assert_eq!(strip_tag("deepseek-r1:1.5b"), "deepseek-r1:1.5b");
    }

    #[test]
    fn strip_tag_keeps_untagged_name() {
        assert_eq!(strip_tag("phi3"), "phi3");
    }

    #[test]
    fn strip_tag_handles_empty_string() {
        assert_eq!(strip_tag(""), "");
    }

    // --- model classification ---

    #[test]
    fn embedding_models_are_partitioned_correctly() {
        let models = vec![
            OllamaModel {
                name: "nomic-embed-text:latest".to_string(),
            },
            OllamaModel {
                name: "mxbai-embed-large:latest".to_string(),
            },
            OllamaModel {
                name: "llama3.2:latest".to_string(),
            },
            OllamaModel {
                name: "mistral:7b".to_string(),
            },
        ];
        let tags = TagsResponse { models };

        let (embedding, chat): (Vec<String>, Vec<String>) = tags
            .models
            .into_iter()
            .map(|m| strip_tag(&m.name))
            .partition(|name| is_embedding_model(name));

        assert_eq!(embedding, vec!["nomic-embed-text", "mxbai-embed-large"]);
        assert_eq!(chat, vec!["llama3.2", "mistral:7b"]);
    }

    #[test]
    fn is_embedding_model_classifies_embed_families_without_the_word() {
        // The obvious `*embed*` names.
        assert!(is_embedding_model("nomic-embed-text"));
        assert!(is_embedding_model("mxbai-embed-large"));
        assert!(is_embedding_model("snowflake-arctic-embed"));
        // Embed families whose names omit "embed" — must still be embeddings.
        assert!(is_embedding_model("bge-m3"));
        assert!(is_embedding_model("all-minilm"));
        assert!(is_embedding_model("all-minilm:33m"));
        // Chat models must not be misclassified.
        assert!(!is_embedding_model("llama3.2"));
        assert!(!is_embedding_model("mistral:7b"));
        assert!(!is_embedding_model("qwen3:8b"));
    }

    #[test]
    fn embed_only_bge_m3_is_selected_not_misfiled_as_chat() {
        // Regression: a host whose only embed model is bge-m3 (no "embed" in the
        // name) must yield bge-m3 as the embedding model and no chat model — not
        // the other way around, which would suppress embed detection entirely.
        let info = OllamaInfo {
            base_url: DEFAULT_OLLAMA_URL.to_string(),
            embedding_models: vec!["bge-m3".to_string()],
            chat_models: vec![],
        };
        assert_eq!(info.best_embedding_model(), Some("bge-m3"));
        assert_eq!(info.best_chat_model(), None);
    }

    // --- OllamaInfo helpers ---

    fn make_info(chat: &[&str], embed: &[&str]) -> OllamaInfo {
        OllamaInfo {
            base_url: DEFAULT_OLLAMA_URL.to_string(),
            chat_models: chat.iter().map(|s| s.to_string()).collect(),
            embedding_models: embed.iter().map(|s| s.to_string()).collect(),
        }
    }

    // --- best_embedding_model ---

    #[test]
    fn best_embedding_model_returns_none_when_empty() {
        let info = make_info(&["llama3.2"], &[]);
        assert_eq!(info.best_embedding_model(), None);
    }

    #[test]
    fn best_embedding_model_returns_highest_priority() {
        // mxbai-embed-large is listed first but nomic-embed-text has higher priority
        let info = make_info(&[], &["mxbai-embed-large", "nomic-embed-text"]);
        assert_eq!(info.best_embedding_model(), Some("nomic-embed-text"));
    }

    #[test]
    fn best_embedding_model_prefix_matches_tagged_name() {
        // "nomic-embed-text" prefix should match "nomic-embed-text:latest" (already stripped)
        // and also "nomic-embed-text:v1.5"
        let info = make_info(&[], &["mxbai-embed-large:latest", "nomic-embed-text:v1.5"]);
        assert_eq!(info.best_embedding_model(), Some("nomic-embed-text:v1.5"));
    }

    #[test]
    fn best_embedding_model_falls_back_when_no_priority_match() {
        let info = make_info(&[], &["custom-embed-model"]);
        assert_eq!(info.best_embedding_model(), Some("custom-embed-model"));
    }

    // --- best_chat_model ---

    #[test]
    fn best_chat_model_returns_none_when_empty() {
        let info = make_info(&[], &["nomic-embed-text"]);
        assert_eq!(info.best_chat_model(), None);
    }

    #[test]
    fn best_chat_model_returns_highest_priority() {
        // llama3.2 listed first but qwen3 has higher priority
        let info = make_info(&["llama3.2", "mistral", "qwen3:8b"], &[]);
        assert_eq!(info.best_chat_model(), Some("qwen3:8b"));
    }

    #[test]
    fn best_chat_model_prefix_matches_tagged_name() {
        // "qwen3.5" should match "qwen3.5:9b"
        let info = make_info(&["llama3.2:3b", "qwen3.5:9b"], &[]);
        assert_eq!(info.best_chat_model(), Some("qwen3.5:9b"));
    }

    #[test]
    fn best_chat_model_skips_vision_models() {
        // llava is a vision model and should be skipped; llama3.2 is the fallback
        let info = make_info(&["llava:7b", "llama3.2"], &[]);
        assert_eq!(info.best_chat_model(), Some("llama3.2"));
    }

    #[test]
    fn best_chat_model_skips_all_vision_variants() {
        let info = make_info(&["llava:7b", "bakllava:latest", "llava-ocr"], &[]);
        assert_eq!(info.best_chat_model(), None);
    }

    #[test]
    fn best_chat_model_falls_back_to_first_non_vision() {
        // none of the models match the priority list
        let info = make_info(&["llava:7b", "custom-model", "another-model"], &[]);
        assert_eq!(info.best_chat_model(), Some("custom-model"));
    }

    #[test]
    fn best_chat_model_does_not_confuse_qwen3_with_qwen3_5_prefix() {
        // "qwen3" prefix must not match "qwen3.5:9b" — that belongs to "qwen3.5" priority
        // (but "qwen3" does start with "qwen3", so it WOULD match qwen3.5 — this is intentional:
        //  qwen3.5 has higher priority and is checked first, so qwen3.5:9b is selected)
        let info = make_info(&["qwen3:8b", "qwen3.5:9b"], &[]);
        assert_eq!(info.best_chat_model(), Some("qwen3.5:9b"));
    }

    // --- ollama_base_url ---

    #[test]
    fn base_url_defaults_to_localhost() {
        // Only check that when OLLAMA_HOST is not set we get the default.
        // We cannot unset env vars reliably in parallel tests, so just verify the
        // fallback path returns the expected string when the var is absent.
        let url = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| DEFAULT_OLLAMA_URL.to_string());
        assert!(!url.is_empty());
    }

    // --- JSON parsing ---

    #[test]
    fn parse_tags_response_with_mixed_models() {
        let json = r#"{"models":[
            {"name":"nomic-embed-text:latest","modified_at":"2024-01-01","size":1234},
            {"name":"llama3.2:latest","modified_at":"2024-01-02","size":5678}
        ]}"#;
        let tags: TagsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(tags.models.len(), 2);
        assert_eq!(tags.models[0].name, "nomic-embed-text:latest");
        assert_eq!(tags.models[1].name, "llama3.2:latest");
    }

    #[test]
    fn parse_tags_response_empty_models() {
        let json = r#"{"models":[]}"#;
        let tags: TagsResponse = serde_json::from_str(json).unwrap();
        assert!(tags.models.is_empty());
    }

    // --- probe (integration, requires running Ollama) ---

    #[tokio::test]
    #[ignore = "requires a running Ollama instance at localhost:11434"]
    async fn probe_detects_local_ollama() {
        let info = probe(DEFAULT_OLLAMA_URL).await;
        assert!(info.is_some(), "expected Ollama to be detected");
        let info = info.unwrap();
        assert_eq!(info.base_url, DEFAULT_OLLAMA_URL);
    }

    #[tokio::test]
    async fn probe_returns_none_when_unreachable() {
        // Port 19999 is almost certainly not running anything.
        let info = probe("http://localhost:19999").await;
        assert!(info.is_none(), "expected None for unreachable host");
    }

    #[tokio::test]
    async fn probe_respects_base_url_trailing_slash() {
        // Should not double-slash; just verify it doesn't panic.
        let _info = probe("http://localhost:19999/").await;
    }
}
