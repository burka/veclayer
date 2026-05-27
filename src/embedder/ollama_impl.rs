use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::Embedder;
use crate::util::{build_hardened_client, read_capped_body, MAX_HTTP_BODY_BYTES};
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
    ///
    /// Body is read via the shared capped reader; responses over
    /// [`MAX_HTTP_BODY_BYTES`] are rejected before buffering completes.
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
        let bytes = read_capped_body(response, MAX_HTTP_BODY_BYTES)
            .await
            .map_err(|e| Error::embedding(format!("{err_prefix} Failed to read response: {e}")))?;

        Ok((status.as_u16(), bytes))
    }

    /// Create a new OllamaEmbedder.
    pub fn new(
        model: impl Into<String>,
        base_url: impl Into<String>,
        dimension: usize,
    ) -> Result<Self> {
        let client = build_hardened_client(Duration::from_secs(10), Duration::from_secs(120))
            .ok_or_else(|| Error::embedding("Failed to build HTTP client"))?;
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

        let got = parsed.embeddings.len();
        if got != texts.len() {
            return Err(Error::embedding(format!(
                "embedding service returned {} vectors for {} inputs",
                got,
                texts.len()
            )));
        }

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
        let embeddings: Vec<Vec<f32>> = parsed.data.into_iter().map(|e| e.embedding).collect();

        let got = embeddings.len();
        if got != texts.len() {
            return Err(Error::embedding(format!(
                "embedding service returned {} vectors for {} inputs",
                got,
                texts.len()
            )));
        }

        Ok(embeddings)
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
    fn embed<'a>(
        &'a self,
        texts: &'a [&'a str],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + 'a>>
    {
        Box::pin(self.embed_async(texts))
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
    use crate::test_helpers::{mock_listener, serve_once};
    use crate::util::DEFAULT_OLLAMA_URL;

    // ── JSON response builders + multi-connection mock server ──────────────────
    // Single-connection helpers live in `crate::test_helpers` (mock_listener,
    // serve_once); `serve_n_chunks` below is embedder-specific (batched requests).

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
        serve_once(listener, 503, r#"{"error":"oops"}"#);

        let embedder = embedder_at(&base_url);
        let result = embedder.embed_async(&["text"]).await;

        assert!(result.is_err(), "expected Err for 503, got Ok");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("503") || msg.contains("API error"),
            "error should mention 503; got: {msg}"
        );
    }

    // ── security: redirect rejection ──────────────────────────────────────────

    /// A 301 redirect to a different host must NOT be followed; the client should
    /// surface the redirect as an error (or return the 3xx status), never silently
    /// follow it to a second endpoint.
    #[tokio::test]
    async fn redirect_301_is_not_followed() {
        let (listener, base_url) = mock_listener().await;

        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let _ = stream.read(&mut buf).await;
            // Reply with a 301 pointing to an evil pivot host.
            let response = "HTTP/1.1 301 Moved Permanently\r\nLocation: http://10.0.0.1:9999/steal\r\nContent-Length: 0\r\n\r\n";
            stream.write_all(response.as_bytes()).await.unwrap();
        });

        let embedder = embedder_at(&base_url);
        let result = embedder.embed_async(&["text"]).await;

        // Must NOT succeed — the redirect must surface as an error, not be followed.
        assert!(
            result.is_err(),
            "expected Err when server sends 301, got Ok (redirect was silently followed)"
        );
        let msg = result.unwrap_err().to_string();
        // Error should mention 301 or redirect, NOT a parse error from the pivot host.
        assert!(
            msg.contains("301") || msg.contains("redirect") || msg.contains("HTTP error"),
            "error should surface the 3xx, got: {msg}"
        );
    }

    // ── security: body cap enforced ───────────────────────────────────────────

    /// A response body larger than the cap must abort with an error rather than
    /// buffering the whole payload.  We test this by sending a body that is
    /// slightly over a small test-specific cap via `http_post_json_capped`.
    #[tokio::test]
    async fn oversized_body_returns_error() {
        use crate::util::MAX_HTTP_BODY_BYTES;

        let (listener, base_url) = mock_listener().await;

        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let _ = stream.read(&mut buf).await;
            // Send a body that is 1 byte over the cap.
            let body_size = MAX_HTTP_BODY_BYTES + 1;
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {body_size}\r\n\r\n"
            );
            stream.write_all(header.as_bytes()).await.unwrap();
            // Stream the body in chunks to avoid allocating a huge buffer.
            let chunk = vec![b'x'; 8192];
            let mut written = 0usize;
            while written < body_size {
                let to_write = chunk.len().min(body_size - written);
                if stream.write_all(&chunk[..to_write]).await.is_err() {
                    break;
                }
                written += to_write;
            }
        });

        let embedder = embedder_at(&base_url);
        let result = embedder.embed_async(&["text"]).await;

        assert!(
            result.is_err(),
            "expected Err when response body exceeds cap, got Ok"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("too large")
                || msg.contains("cap")
                || msg.contains("limit")
                || msg.contains("body"),
            "error should mention body size limit, got: {msg}"
        );
    }

    // ── TDD regression: embed must not panic on a current_thread runtime ────────

    /// RED before this fix: the old sync `embed` used block_in_place, which
    /// panics on a current_thread runtime. GREEN now that embed is async.
    #[tokio::test(flavor = "current_thread")]
    async fn embed_on_current_thread_runtime_succeeds() {
        let texts = vec!["a", "b"];
        let (listener, base_url) = mock_listener().await;
        let log = serve_n_chunks(listener, vec![2]);
        let embedder = embedder_at(&base_url);
        let result = embedder
            .embed(&texts)
            .await
            .expect("embed must not panic on current_thread");
        assert_eq!(result.len(), 2);
        assert_eq!(log.lock().unwrap().len(), 1);
    }

    // ── OpenAI-format mock helpers ────────────────────────────────────────────

    /// Build an OpenAI-format JSON response with `n` embeddings of dimension 1.
    fn openai_response_json(n: usize) -> String {
        let data: Vec<String> = (0..n)
            .map(|i| format!(r#"{{"embedding":[{}.0],"index":{}}}"#, i, i))
            .collect();
        format!(r#"{{"data":[{}]}}"#, data.join(","))
    }

    fn embedder_openai_at(base_url: &str) -> OllamaEmbedder {
        let e = OllamaEmbedder::new("test-model", base_url, 1).unwrap();
        e.api_format.store(FORMAT_OPENAI, Ordering::Relaxed);
        e
    }

    // ── count mismatch: Ollama format ─────────────────────────────────────────

    /// RED→GREEN: server returns 1 embedding for a 2-input request → Err.
    #[tokio::test]
    async fn ollama_count_mismatch_returns_err() {
        let (listener, base_url) = mock_listener().await;
        // Return only 1 embedding even though we send 2 inputs.
        serve_once(listener, 200, ollama_response_json(1, 0));

        let embedder = embedder_at(&base_url);
        let result = embedder.embed_async(&["text1", "text2"]).await;

        assert!(
            result.is_err(),
            "expected Err when server returns fewer embeddings than inputs"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("1") && msg.contains("2"),
            "error should mention counts; got: {msg}"
        );
    }

    /// GREEN: server returns exactly 2 embeddings for 2 inputs → Ok.
    #[tokio::test]
    async fn ollama_count_match_returns_ok() {
        let (listener, base_url) = mock_listener().await;
        serve_once(listener, 200, ollama_response_json(2, 0));

        let embedder = embedder_at(&base_url);
        let result = embedder.embed_async(&["text1", "text2"]).await;

        assert!(
            result.is_ok(),
            "expected Ok on matching count; got: {:?}",
            result
        );
        assert_eq!(result.unwrap().len(), 2);
    }

    // ── count mismatch: OpenAI format ─────────────────────────────────────────

    /// RED→GREEN: server returns 1 embedding for a 2-input request (OpenAI format) → Err.
    #[tokio::test]
    async fn openai_count_mismatch_returns_err() {
        let (listener, base_url) = mock_listener().await;
        serve_once(listener, 200, openai_response_json(1));

        let embedder = embedder_openai_at(&base_url);
        let result = embedder.embed_async(&["text1", "text2"]).await;

        assert!(
            result.is_err(),
            "expected Err when OpenAI server returns fewer embeddings than inputs"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("1") && msg.contains("2"),
            "error should mention counts; got: {msg}"
        );
    }

    /// GREEN: server returns exactly 2 embeddings for 2 inputs (OpenAI format) → Ok.
    #[tokio::test]
    async fn openai_count_match_returns_ok() {
        let (listener, base_url) = mock_listener().await;
        serve_once(listener, 200, openai_response_json(2));

        let embedder = embedder_openai_at(&base_url);
        let result = embedder.embed_async(&["text1", "text2"]).await;

        assert!(
            result.is_ok(),
            "expected Ok on matching count; got: {:?}",
            result
        );
        assert_eq!(result.unwrap().len(), 2);
    }

    // ── integration smoke tests (require live server, skipped by default) ──────

    #[tokio::test]
    #[ignore = "requires a running Ollama or TEI service at localhost:11434"]
    async fn test_ollama_embed() {
        let embedder = OllamaEmbedder::new("nomic-embed-text", DEFAULT_OLLAMA_URL, 768).unwrap();
        let texts = vec!["Hello world", "This is a test"];
        let embeddings = embedder.embed(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 768);
    }

    #[tokio::test]
    #[ignore = "requires a running Ollama or TEI service at localhost:11434"]
    async fn test_ollama_embed_empty() {
        let embedder = OllamaEmbedder::new("nomic-embed-text", DEFAULT_OLLAMA_URL, 768).unwrap();
        let embeddings = embedder.embed(&[]).await.unwrap();
        assert!(embeddings.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires a running OpenAI-compatible service at localhost:8080"]
    async fn test_openai_compatible_embed() {
        let embedder =
            OllamaEmbedder::new("BAAI/bge-small-en-v1.5", "http://localhost:8080", 384).unwrap();
        let texts = vec!["Hello world"];
        let embeddings = embedder.embed(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 384);
    }

    #[test]
    fn test_new_returns_ok() {
        let result = OllamaEmbedder::new("model", DEFAULT_OLLAMA_URL, 384);
        assert!(result.is_ok());
    }
}
