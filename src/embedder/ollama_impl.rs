use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::runtime::Handle;

use super::Embedder;
use crate::{Error, Result};

const FORMAT_UNKNOWN: u8 = 0;
const FORMAT_OLLAMA: u8 = 1;
const FORMAT_OPENAI: u8 = 2;

/// HTTP-based embedder that calls external embedding services.
/// Supports Ollama (`/api/embed`) and OpenAI-compatible APIs (`/v1/embeddings`).
/// On the first call, probes both formats and caches the working one to avoid
/// the double round-trip on every subsequent request.
pub struct OllamaEmbedder {
    client: Client,
    model: String,
    base_url: String,
    dimension: usize,
    /// Cached API format discovered after the first successful call.
    api_format: AtomicU8,
}

/// Unified request body used by both Ollama and OpenAI-compatible endpoints.
#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
}

#[derive(Deserialize)]
struct OllamaResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct OpenAiEmbedding {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Deserialize)]
struct OpenAiResponse {
    data: Vec<OpenAiEmbedding>,
}

impl OllamaEmbedder {
    /// Create a new OllamaEmbedder.
    pub fn new(
        model: impl Into<String>,
        base_url: impl Into<String>,
        dimension: usize,
    ) -> Result<Self> {
        // TODO: extract a shared HTTP client builder (with consistent timeouts) used by
        // embedder, src/llm/ollama.rs, and src/llm/openai.rs.
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|e| Error::embedding(format!("Failed to build HTTP client: {}", e)))?;
        Ok(Self {
            client,
            model: model.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            dimension,
            api_format: AtomicU8::new(FORMAT_UNKNOWN),
        })
    }

    async fn try_ollama(&self, texts: &[&str]) -> Result<Option<Vec<Vec<f32>>>> {
        let url = format!("{}/api/embed", self.base_url);
        let body = EmbedRequest {
            model: &self.model,
            input: texts,
        };

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::embedding(format!("HTTP request failed: {}", e)))?;

        if response.status().as_u16() == 404 {
            return Ok(None);
        }

        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| Error::embedding(format!("Failed to read response: {}", e)))?;

        if !status.is_success() {
            return Err(Error::embedding(format!(
                "Ollama API error {}: {}",
                status,
                String::from_utf8_lossy(&bytes)
            )));
        }

        let parsed: OllamaResponse = serde_json::from_slice(&bytes)
            .map_err(|e| Error::embedding(format!("Failed to parse Ollama response: {}", e)))?;
        Ok(Some(parsed.embeddings))
    }

    async fn try_openai(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/v1/embeddings", self.base_url);
        let body = EmbedRequest {
            model: &self.model,
            input: texts,
        };

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::embedding(format!("HTTP request failed: {}", e)))?;

        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| Error::embedding(format!("Failed to read response: {}", e)))?;

        if !status.is_success() {
            return Err(Error::embedding(format!(
                "OpenAI-compatible API error {}: {}",
                status,
                String::from_utf8_lossy(&bytes)
            )));
        }

        let mut parsed: OpenAiResponse = serde_json::from_slice(&bytes)
            .map_err(|e| Error::embedding(format!("Failed to parse OpenAI response: {}", e)))?;

        parsed.data.sort_by_key(|e| e.index);
        Ok(parsed.data.into_iter().map(|e| e.embedding).collect())
    }

    // TODO: chunk large batches (e.g. >64 texts) into smaller sub-batches to bound
    // memory and avoid hitting the request timeout on slow backends.
    async fn embed_async(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        match self.api_format.load(Ordering::Relaxed) {
            FORMAT_OLLAMA => self.try_ollama(texts).await?.ok_or_else(|| {
                Error::embedding("Ollama endpoint returned 404 after format was cached")
            }),
            FORMAT_OPENAI => self.try_openai(texts).await,
            _ => {
                // Unknown: probe Ollama first, then fall back to OpenAI.
                if let Some(result) = self.try_ollama(texts).await? {
                    self.api_format.store(FORMAT_OLLAMA, Ordering::Relaxed);
                    Ok(result)
                } else {
                    let result = self.try_openai(texts).await?;
                    self.api_format.store(FORMAT_OPENAI, Ordering::Relaxed);
                    Ok(result)
                }
            }
        }
    }
}

impl Embedder for OllamaEmbedder {
    // TODO: block_in_place panics on current_thread runtimes and requires an active
    // tokio Handle. Consider making the Embedder trait async or constructing a dedicated
    // runtime in new() to decouple from the caller's executor.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        tokio::task::block_in_place(|| Handle::current().block_on(self.embed_async(texts)))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "requires a running Ollama or TEI service at localhost:11434"]
    async fn test_ollama_embed() {
        let embedder =
            OllamaEmbedder::new("nomic-embed-text", "http://localhost:11434", 768).unwrap();
        let texts = vec!["Hello world", "This is a test"];
        let embeddings = embedder.embed(&texts).unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 768);
    }

    #[tokio::test]
    #[ignore = "requires a running Ollama or TEI service at localhost:11434"]
    async fn test_ollama_embed_empty() {
        let embedder =
            OllamaEmbedder::new("nomic-embed-text", "http://localhost:11434", 768).unwrap();
        let embeddings = embedder.embed(&[]).unwrap();
        assert!(embeddings.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires a running OpenAI-compatible service at localhost:8080"]
    async fn test_openai_compatible_embed() {
        let embedder =
            OllamaEmbedder::new("BAAI/bge-small-en-v1.5", "http://localhost:8080", 384).unwrap();
        let texts = vec!["Hello world"];
        let embeddings = embedder.embed(&texts).unwrap();

        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 384);
    }

    #[test]
    fn test_new_returns_ok() {
        let result = OllamaEmbedder::new("model", "http://localhost:11434", 384);
        assert!(result.is_ok());
    }
}
