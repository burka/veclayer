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

/// Maximum number of texts sent in a single HTTP request to the embedding backend.
/// Larger batches are split into sequential sub-batches of this size to avoid
/// server-side memory limits and request timeouts.
const MAX_BATCH: usize = 64;

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
    /// Low-level helper: POST a JSON body to `url` and return (status, bytes).
    async fn http_post_json(
        &self,
        url: &str,
        body: &impl Serialize,
        err_prefix: &str,
    ) -> Result<(u16, Vec<u8>)> {
        let response = self
            .client
            .post(url)
            .json(body)
            .send()
            .await
            .map_err(|e| Error::embedding(format!("{err_prefix} HTTP request failed: {e}")))?;

        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| Error::embedding(format!("{err_prefix} Failed to read response: {e}")))?;

        Ok((status.as_u16(), bytes.into()))
    }

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

        let (status, bytes) = self.http_post_json(&url, &body, "Ollama").await?;

        if status == 404 {
            return Ok(None);
        }

        if !(200..300).contains(&status) {
            return Err(Error::embedding(format!(
                "Ollama API error {}: {}\nHint: ensure the model '{}' is pulled — run `ollama pull {}`",
                status,
                String::from_utf8_lossy(&bytes),
                self.model,
                self.model,
            )));
        }

        let parsed: OllamaResponse = serde_json::from_slice(&bytes)
            .map_err(|e| Error::embedding(format!("Failed to parse Ollama response: {e}")))?;
        Ok(Some(parsed.embeddings))
    }

    async fn try_openai(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/v1/embeddings", self.base_url);
        let body = EmbedRequest {
            model: &self.model,
            input: texts,
        };

        let (status, bytes) = self
            .http_post_json(&url, &body, "OpenAI-compatible")
            .await?;

        if !(200..300).contains(&status) {
            return Err(Error::embedding(format!(
                "OpenAI-compatible API error {}: {}",
                status,
                String::from_utf8_lossy(&bytes)
            )));
        }

        let mut parsed: OpenAiResponse = serde_json::from_slice(&bytes)
            .map_err(|e| Error::embedding(format!("Failed to parse OpenAI response: {e}")))?;

        parsed.data.sort_by_key(|e| e.index);
        Ok(parsed.data.into_iter().map(|e| e.embedding).collect())
    }

    /// Dispatch a single chunk (already ≤ MAX_BATCH) to the right endpoint,
    /// probing on the first call and caching the discovered format.
    async fn embed_chunk(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
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

    /// Split `texts` into sequential sub-batches of at most [`MAX_BATCH`] items,
    /// send each as a separate HTTP request, and concatenate the results in order.
    /// Returns immediately with an `Err` if any chunk fails.
    async fn embed_async(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let mut all = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(MAX_BATCH) {
            let embeddings = self.embed_chunk(chunk).await?;
            all.extend(embeddings);
        }
        Ok(all)
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
    use std::sync::{Arc, Mutex};

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;
    use crate::util::DEFAULT_OLLAMA_URL;

    // ── mock server helpers ────────────────────────────────────────────────────

    /// Bind a TCP listener on an ephemeral port and return it with the base URL.
    async fn mock_listener() -> (TcpListener, String) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        (listener, format!("http://127.0.0.1:{port}"))
    }

    /// Build the Ollama-format JSON response for `n` embeddings of dimension 1.
    /// Each embedding is a single-element vec `[i as f32]` so callers can verify order.
    fn ollama_response_json(n: usize, offset: usize) -> String {
        let vecs: Vec<String> = (offset..offset + n).map(|i| format!("[{}.0]", i)).collect();
        format!(r#"{{"embeddings":[{}]}}"#, vecs.join(","))
    }

    /// Spawn a task that accepts one connection per entry in `chunk_sizes`,
    /// replies to each with an Ollama-format response (canned per-chunk), and
    /// records the raw request bodies.  Returns the shared request-log.
    fn serve_n_chunks(
        listener: TcpListener,
        // For each request index, how many embeddings to include in the response.
        chunk_sizes: Vec<usize>,
    ) -> Arc<Mutex<Vec<Vec<u8>>>> {
        let log: Arc<Mutex<Vec<Vec<u8>>>> = Arc::new(Mutex::new(Vec::new()));
        let log_clone = log.clone();

        tokio::spawn(async move {
            let mut offset = 0usize;
            for &n_embeddings in &chunk_sizes {
                let (mut stream, _) = listener.accept().await.unwrap();

                // Read the entire HTTP request (headers + body).  A real server
                // would parse Content-Length; for tests a large buffer is enough.
                let mut buf = vec![0u8; 65536];
                let n = stream.read(&mut buf).await.unwrap();
                buf.truncate(n);
                log_clone.lock().unwrap().push(buf);

                let body = ollama_response_json(n_embeddings, offset);
                offset += n_embeddings;

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
                    body.len(),
                    body,
                );
                stream.write_all(response.as_bytes()).await.unwrap();
            }
        });

        log
    }

    /// Like `serve_n_chunks` but every connection gets a non-2xx error reply.
    fn serve_error_once(listener: TcpListener, status: u16) {
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let _ = stream.read(&mut buf).await;
            let body = r#"{"error":"oops"}"#;
            let response = format!(
                "HTTP/1.1 {status} Error\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{body}",
                body.len(),
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        });
    }

    fn embedder_at(base_url: &str) -> OllamaEmbedder {
        let e = OllamaEmbedder::new("test-model", base_url, 1).unwrap();
        // Pre-cache the Ollama format so tests don't need a 404 probe path.
        e.api_format.store(FORMAT_OLLAMA, Ordering::Relaxed);
        e
    }

    // ── edge cases ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn empty_input_returns_empty_without_request() {
        // No server started; any HTTP attempt would fail — proving no request is sent.
        let (listener, base_url) = mock_listener().await;
        drop(listener);

        let embedder = embedder_at(&base_url);
        let result = embedder.embed_async(&[]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn exactly_max_batch_sends_one_request() {
        let texts: Vec<&str> = vec!["x"; MAX_BATCH];
        let (listener, base_url) = mock_listener().await;
        let log = serve_n_chunks(listener, vec![MAX_BATCH]);

        let embedder = embedder_at(&base_url);
        let result = embedder.embed_async(&texts).await.unwrap();

        assert_eq!(result.len(), MAX_BATCH, "wrong embedding count");
        assert_eq!(
            log.lock().unwrap().len(),
            1,
            "expected exactly 1 HTTP request for {} texts",
            MAX_BATCH
        );
    }

    #[tokio::test]
    async fn one_over_max_batch_sends_two_requests() {
        let n = MAX_BATCH + 1;
        let texts: Vec<&str> = vec!["x"; n];
        let (listener, base_url) = mock_listener().await;
        // chunk 1: MAX_BATCH embeddings, chunk 2: 1 embedding
        let log = serve_n_chunks(listener, vec![MAX_BATCH, 1]);

        let embedder = embedder_at(&base_url);
        let result = embedder.embed_async(&texts).await.unwrap();

        assert_eq!(result.len(), n, "wrong total embedding count");
        assert_eq!(
            log.lock().unwrap().len(),
            2,
            "expected 2 HTTP requests for {} texts (MAX_BATCH={})",
            n,
            MAX_BATCH
        );
    }

    // ── green: large batch chunks correctly, result count and order preserved ──

    #[tokio::test]
    async fn large_batch_splits_into_ceil_chunks_and_preserves_order() {
        const N: usize = 150;
        // ceil(150 / 64) = 3 chunks: 64, 64, 22
        let chunk_sizes = vec![64, 64, 22];
        let expected_requests = chunk_sizes.len();

        let texts: Vec<&str> = vec!["t"; N];
        let (listener, base_url) = mock_listener().await;
        let log = serve_n_chunks(listener, chunk_sizes.clone());

        let embedder = embedder_at(&base_url);
        let result = embedder.embed_async(&texts).await.unwrap();

        // Total count
        assert_eq!(
            result.len(),
            N,
            "expected {N} embeddings, got {}",
            result.len()
        );

        // Request count proves chunking happened
        assert_eq!(
            log.lock().unwrap().len(),
            expected_requests,
            "expected {} HTTP requests (ceil({N}/{MAX_BATCH})), got {}",
            expected_requests,
            log.lock().unwrap().len()
        );

        // Order: mock assigns offset-based values so result[i] == [i as f32]
        for (i, emb) in result.iter().enumerate() {
            assert_eq!(
                emb,
                &vec![i as f32],
                "embedding at index {i} is out of order"
            );
        }
    }

    // ── error: mid-stream chunk failure propagates as Err ─────────────────────

    #[tokio::test]
    async fn error_on_second_chunk_returns_err() {
        const N: usize = MAX_BATCH + 10; // needs 2 chunks
        let texts: Vec<&str> = vec!["t"; N];

        let (listener, base_url) = mock_listener().await;

        // Serve the first chunk successfully, then serve an error for the second.
        tokio::spawn(async move {
            // chunk 1 — success
            {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut buf = vec![0u8; 65536];
                let n = stream.read(&mut buf).await.unwrap();
                buf.truncate(n);
                let body = ollama_response_json(MAX_BATCH, 0);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
                    body.len(),
                    body,
                );
                stream.write_all(response.as_bytes()).await.unwrap();
            }
            // chunk 2 — server error
            {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut buf = vec![0u8; 4096];
                let _ = stream.read(&mut buf).await;
                let body = r#"{"error":"quota exceeded"}"#;
                let response = format!(
                    "HTTP/1.1 500 Error\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{body}",
                    body.len(),
                );
                stream.write_all(response.as_bytes()).await.unwrap();
            }
        });

        let embedder = embedder_at(&base_url);
        let result = embedder.embed_async(&texts).await;

        assert!(
            result.is_err(),
            "expected Err when a mid-stream chunk fails, got Ok"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("500") || msg.contains("API error"),
            "error message should mention 500 status; got: {msg}"
        );
    }

    // ── error: malformed body on first chunk ──────────────────────────────────

    #[tokio::test]
    async fn malformed_response_body_returns_err() {
        let (listener, base_url) = mock_listener().await;

        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let _ = stream.read(&mut buf).await;
            // 200 OK but body is not valid Ollama JSON
            let body = r#"not json at all"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{body}",
                body.len(),
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        });

        let embedder = embedder_at(&base_url);
        let result = embedder.embed_async(&["text"]).await;

        assert!(result.is_err(), "expected parse error, got Ok");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("parse") || msg.contains("JSON") || msg.contains("json"),
            "error should mention parse/JSON; got: {msg}"
        );
    }

    // ── non-2xx on first chunk ────────────────────────────────────────────────

    #[tokio::test]
    async fn non_2xx_on_first_chunk_returns_err() {
        let (listener, base_url) = mock_listener().await;
        serve_error_once(listener, 503);

        let embedder = embedder_at(&base_url);
        let result = embedder.embed_async(&["text"]).await;

        assert!(result.is_err(), "expected Err for 503, got Ok");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("503") || msg.contains("API error"),
            "error should mention 503; got: {msg}"
        );
    }

    // ── integration smoke tests (require live server, skipped by default) ──────

    #[tokio::test]
    #[ignore = "requires a running Ollama or TEI service at localhost:11434"]
    async fn test_ollama_embed() {
        let embedder = OllamaEmbedder::new("nomic-embed-text", DEFAULT_OLLAMA_URL, 768).unwrap();
        let texts = vec!["Hello world", "This is a test"];
        let embeddings = embedder.embed(&texts).unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 768);
    }

    #[tokio::test]
    #[ignore = "requires a running Ollama or TEI service at localhost:11434"]
    async fn test_ollama_embed_empty() {
        let embedder = OllamaEmbedder::new("nomic-embed-text", DEFAULT_OLLAMA_URL, 768).unwrap();
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
        let result = OllamaEmbedder::new("model", DEFAULT_OLLAMA_URL, 384);
        assert!(result.is_ok());
    }
}
