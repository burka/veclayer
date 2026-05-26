//! OpenAI-compatible embedding-service auto-discovery (vLLM, HuggingFace TEI,
//! LM Studio, llama.cpp server, …).
//!
//! Counterpart to [`crate::ollama_discover`]. Where Ollama exposes `/api/tags`,
//! OpenAI-compatible servers expose `/v1/models`. This module probes that
//! endpoint, classifies the served models by name, and — because `/v1/models`
//! does not report embedding dimensions — makes one `/v1/embeddings` call to
//! learn the true vector dimension before VecLayer creates a store with it.
//!
//! The pure helpers (`normalize_base`, `pick_embed_model`, `first_embedding_len`)
//! are unit-tested without a live server; end-to-end probing is covered by an
//! `#[ignore]`d integration test. Gated behind the `llm` feature (needs reqwest).

use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Base URLs probed when no explicit endpoint is configured.
/// vLLM serves OpenAI-compat on :8000; HuggingFace TEI on :8080.
const DEFAULT_BASE_URLS: &[&str] = &["http://localhost:8000", "http://localhost:8080"];

/// Connect timeout — bounds the common "nothing listening" case so startup is
/// not delayed when no OpenAI-compat server is running.
const PROBE_CONNECT_TIMEOUT: Duration = Duration::from_millis(500);
/// Overall request timeout — generous enough for a present-but-slow server to
/// answer the dimension-probe embedding call (which may trigger a model load).
const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// Case-insensitive substrings that mark a model id as an embedding model.
/// Any match qualifies; chat/instruct models match none of these.
const EMBED_MODEL_SUBSTRINGS: &[&str] = &[
    "embed",
    "bge",
    "e5",
    "gte",
    "nomic",
    "mxbai",
    "arctic",
    "mpnet",
    "minilm",
    "instructor",
];

/// A discovered OpenAI-compatible embedding endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenAiCompatInfo {
    /// Normalised base URL (no trailing `/` or `/v1`).
    pub base_url: String,
    /// The chosen embedding model id.
    pub embed_model: String,
    /// The true embedding dimension, learned via a `/v1/embeddings` probe.
    pub dimension: usize,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(Serialize)]
struct EmbedProbeRequest<'a> {
    model: &'a str,
    input: [&'a str; 1],
}

#[derive(Deserialize)]
struct EmbeddingEntry {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingEntry>,
}

/// Strip a trailing `/` and optional `/v1` suffix so known paths can be appended.
///
/// vLLM's `OPENAI_API_BASE` is conventionally `http://host:8000/v1`; this lets a
/// user paste that value and still get a correct `{base}/v1/models` URL.
fn normalize_base(url: &str) -> String {
    let trimmed = url.trim_end_matches('/');
    trimmed
        .strip_suffix("/v1")
        .unwrap_or(trimmed)
        .trim_end_matches('/')
        .to_string()
}

/// Pick the first model id that looks like an embedding model, or `None`.
fn pick_embed_model(model_ids: &[String]) -> Option<&str> {
    model_ids
        .iter()
        .find(|id| {
            let lower = id.to_lowercase();
            EMBED_MODEL_SUBSTRINGS.iter().any(|s| lower.contains(s))
        })
        .map(String::as_str)
}

/// Length of the first embedding vector in a response, if present.
fn first_embedding_len(resp: &EmbeddingsResponse) -> Option<usize> {
    resp.data.first().map(|e| e.embedding.len())
}

/// Candidate base URLs to probe, honouring an explicit env override.
///
/// `VECLAYER_OPENAI_BASE_URL`, `OPENAI_BASE_URL`, then `OPENAI_API_BASE` are
/// checked in order; the first http(s) value wins and is probed alone. With no
/// override, the well-known vLLM/TEI defaults are probed.
fn candidate_base_urls() -> Vec<String> {
    for var in [
        "VECLAYER_OPENAI_BASE_URL",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
    ] {
        if let Ok(val) = std::env::var(var) {
            if val.starts_with("http://") || val.starts_with("https://") {
                return vec![normalize_base(&val)];
            }
            tracing::debug!("Ignoring {var}: must start with http:// or https://");
        }
    }
    DEFAULT_BASE_URLS
        .iter()
        .map(|u| normalize_base(u))
        .collect()
}

/// Probe a single base URL: list models, pick an embed model, learn its dimension.
async fn probe(client: &Client, base_url: &str) -> Option<OpenAiCompatInfo> {
    let models_url = format!("{base_url}/v1/models");
    let resp = client.get(&models_url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let models: ModelsResponse = resp.json().await.ok()?;
    let ids: Vec<String> = models.data.into_iter().map(|m| m.id).collect();
    let embed_model = pick_embed_model(&ids)?.to_string();

    let dimension = probe_dimension(client, base_url, &embed_model).await?;

    Some(OpenAiCompatInfo {
        base_url: base_url.to_string(),
        embed_model,
        dimension,
    })
}

/// Make one `/v1/embeddings` call to learn the model's vector dimension.
async fn probe_dimension(client: &Client, base_url: &str, model: &str) -> Option<usize> {
    let url = format!("{base_url}/v1/embeddings");
    let body = EmbedProbeRequest {
        model,
        input: ["dimension probe"],
    };
    let resp = client.post(&url).json(&body).send().await.ok()?;
    if !resp.status().is_success() {
        tracing::debug!(
            "OpenAI-compat dimension probe failed: HTTP {}",
            resp.status()
        );
        return None;
    }
    let parsed: EmbeddingsResponse = resp.json().await.ok()?;
    let dim = first_embedding_len(&parsed)?;
    (dim > 0).then_some(dim)
}

/// Detect a local OpenAI-compatible embedding service.
///
/// Returns the first reachable endpoint that serves an embedding model, or
/// `None`. Safe to call from sync or async contexts (see
/// [`crate::util::block_on_probe`]).
pub fn detect() -> Option<OpenAiCompatInfo> {
    crate::util::block_on_probe(detect_async())
}

async fn detect_async() -> Option<OpenAiCompatInfo> {
    let client = crate::util::build_probe_client(PROBE_CONNECT_TIMEOUT, PROBE_TIMEOUT)?;

    for base_url in candidate_base_urls() {
        if let Some(info) = probe(&client, &base_url).await {
            return Some(info);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- normalize_base ---

    #[test]
    fn normalize_base_strips_trailing_slash() {
        assert_eq!(
            normalize_base("http://localhost:8000/"),
            "http://localhost:8000"
        );
    }

    #[test]
    fn normalize_base_strips_v1_suffix() {
        assert_eq!(
            normalize_base("http://localhost:8000/v1"),
            "http://localhost:8000"
        );
    }

    #[test]
    fn normalize_base_strips_v1_with_trailing_slash() {
        assert_eq!(
            normalize_base("http://localhost:8000/v1/"),
            "http://localhost:8000"
        );
    }

    #[test]
    fn normalize_base_leaves_plain_url_untouched() {
        assert_eq!(
            normalize_base("http://localhost:8000"),
            "http://localhost:8000"
        );
    }

    // --- pick_embed_model ---

    fn ids(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn pick_embed_model_finds_bge() {
        let models = ids(&["BAAI/bge-small-en-v1.5"]);
        assert_eq!(pick_embed_model(&models), Some("BAAI/bge-small-en-v1.5"));
    }

    #[test]
    fn pick_embed_model_finds_explicit_embed_substring() {
        let models = ids(&["nomic-ai/nomic-embed-text-v1.5"]);
        assert_eq!(
            pick_embed_model(&models),
            Some("nomic-ai/nomic-embed-text-v1.5")
        );
    }

    #[test]
    fn pick_embed_model_skips_chat_models() {
        let models = ids(&[
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        ]);
        assert_eq!(pick_embed_model(&models), None);
    }

    #[test]
    fn pick_embed_model_picks_embed_among_mixed() {
        let models = ids(&["meta-llama/Llama-3.1-8B-Instruct", "intfloat/e5-large-v2"]);
        assert_eq!(pick_embed_model(&models), Some("intfloat/e5-large-v2"));
    }

    #[test]
    fn pick_embed_model_is_case_insensitive() {
        let models = ids(&["BAAI/BGE-M3"]);
        assert_eq!(pick_embed_model(&models), Some("BAAI/BGE-M3"));
    }

    #[test]
    fn pick_embed_model_returns_none_for_empty() {
        assert_eq!(pick_embed_model(&[]), None);
    }

    // --- first_embedding_len + JSON parsing ---

    #[test]
    fn parse_models_response() {
        let json = r#"{"object":"list","data":[
            {"id":"BAAI/bge-small-en-v1.5","object":"model"},
            {"id":"meta-llama/Llama-3.1-8B","object":"model"}
        ]}"#;
        let parsed: ModelsResponse = serde_json::from_str(json).unwrap();
        let ids: Vec<String> = parsed.data.into_iter().map(|m| m.id).collect();
        assert_eq!(
            ids,
            vec!["BAAI/bge-small-en-v1.5", "meta-llama/Llama-3.1-8B"]
        );
    }

    #[test]
    fn parse_embeddings_response_and_measure_dimension() {
        let json = r#"{"object":"list","data":[
            {"object":"embedding","index":0,"embedding":[0.1,0.2,0.3,0.4]}
        ]}"#;
        let parsed: EmbeddingsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(first_embedding_len(&parsed), Some(4));
    }

    #[test]
    fn first_embedding_len_none_for_empty_data() {
        let parsed = EmbeddingsResponse { data: vec![] };
        assert_eq!(first_embedding_len(&parsed), None);
    }

    // --- candidate_base_urls ---

    #[test]
    #[serial_test::serial]
    fn candidate_urls_default_to_vllm_and_tei() {
        std::env::remove_var("VECLAYER_OPENAI_BASE_URL");
        std::env::remove_var("OPENAI_BASE_URL");
        std::env::remove_var("OPENAI_API_BASE");
        let urls = candidate_base_urls();
        assert_eq!(urls, vec!["http://localhost:8000", "http://localhost:8080"]);
    }

    #[test]
    #[serial_test::serial]
    fn candidate_urls_honour_env_override_and_normalise() {
        std::env::remove_var("OPENAI_BASE_URL");
        std::env::remove_var("OPENAI_API_BASE");
        std::env::set_var("VECLAYER_OPENAI_BASE_URL", "http://gpu-box:8000/v1/");
        let urls = candidate_base_urls();
        std::env::remove_var("VECLAYER_OPENAI_BASE_URL");
        assert_eq!(urls, vec!["http://gpu-box:8000"]);
    }

    #[test]
    #[serial_test::serial]
    fn candidate_urls_ignore_non_http_override() {
        std::env::remove_var("OPENAI_BASE_URL");
        std::env::remove_var("OPENAI_API_BASE");
        std::env::set_var("VECLAYER_OPENAI_BASE_URL", "ftp://nope");
        let urls = candidate_base_urls();
        std::env::remove_var("VECLAYER_OPENAI_BASE_URL");
        assert_eq!(urls, vec!["http://localhost:8000", "http://localhost:8080"]);
    }

    // --- probe (unreachable) ---

    #[tokio::test]
    async fn probe_returns_none_when_unreachable() {
        let client = Client::builder()
            .connect_timeout(Duration::from_millis(200))
            .timeout(Duration::from_millis(200))
            .build()
            .unwrap();
        // Port 1 is never an OpenAI-compat server.
        assert!(probe(&client, "http://127.0.0.1:1").await.is_none());
    }

    // --- detect (live integration) ---

    #[test]
    #[ignore = "requires a running vLLM/TEI server serving an embedding model"]
    fn detect_finds_running_service() {
        let info = detect();
        assert!(info.is_some(), "expected an OpenAI-compat embed service");
        let info = info.unwrap();
        assert!(info.dimension > 0);
        assert!(!info.embed_model.is_empty());
    }
}
