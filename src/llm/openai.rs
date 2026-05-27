//! OpenAI-compatible LLM provider.
//!
//! Works with OpenAI, Azure OpenAI, LM Studio, and any API that implements
//! the /v1/chat/completions endpoint.

use secrecy::{ExposeSecret, SecretString};

use super::{make_standard_http_client, LlmConfig, LlmProvider, Message};
use crate::util::{read_capped_body, MAX_HTTP_BODY_BYTES};
use reqwest::Client;

pub struct OpenAiLlm {
    client: Client,
    model: String,
    base_url: String,
    api_key: SecretString,
    temperature: f32,
    max_tokens: usize,
}

impl OpenAiLlm {
    pub fn new(config: &LlmConfig) -> crate::Result<Self> {
        Ok(Self {
            client: make_standard_http_client()?,
            model: config.model.clone(),
            base_url: config.base_url.clone(),
            api_key: config
                .api_key
                .clone()
                .unwrap_or_else(|| SecretString::from(String::new())),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
        })
    }
}

impl LlmProvider for OpenAiLlm {
    async fn complete(&self, messages: &[Message]) -> crate::Result<String> {
        let msgs = Message::to_json_values(messages);

        let resp = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header(
                "Authorization",
                format!("Bearer {}", self.api_key.expose_secret()),
            )
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": self.model,
                "messages": msgs,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }))
            .send()
            .await
            .map_err(|e| crate::Error::llm(format!("OpenAI request failed: {}", e)))?;

        if !resp.status().is_success() {
            return Err(crate::llm::http_error("OpenAI API", resp).await);
        }

        let bytes = read_capped_body(resp, MAX_HTTP_BODY_BYTES)
            .await
            .map_err(|e| crate::Error::llm(format!("OpenAI response read failed: {}", e)))?;
        let body: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| crate::Error::llm(format!("OpenAI response parse failed: {}", e)))?;

        body["choices"][0]["message"]["content"]
            .as_str()
            .map(String::from)
            .ok_or_else(|| crate::Error::llm("OpenAI response missing choices[0].message.content"))
    }

    fn name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::LlmProvider;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// Spawn a single-shot mock HTTP server. Accepts one connection, reads the
    /// full request into a string, writes `response`, then closes. Returns the
    /// bound port and a future that resolves to the raw request text once the
    /// connection is handled.
    async fn spawn_mock(response: String) -> (u16, tokio::task::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        let handle = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let n = stream.read(&mut buf).await.unwrap_or(0);
            let request = String::from_utf8_lossy(&buf[..n]).into_owned();
            stream.write_all(response.as_bytes()).await.unwrap();
            request
        });

        (port, handle)
    }

    fn make_config(port: u16, api_key: Option<&str>) -> LlmConfig {
        LlmConfig {
            provider: "openai".to_string(),
            model: "gpt-test".to_string(),
            base_url: format!("http://127.0.0.1:{port}"),
            api_key: api_key.map(|s| SecretString::from(s.to_string())),
            temperature: 0.0,
            max_tokens: 16,
        }
    }

    // 1. Green path: 200 with well-formed choices → Ok("answer")
    #[tokio::test]
    async fn complete_returns_content_on_200() {
        let body = r#"{"choices":[{"message":{"content":"answer"}}]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let (port, mock) = spawn_mock(response).await;

        let llm = OpenAiLlm::new(&make_config(port, Some("sk-test"))).expect("client build");
        let result = llm.complete(&[Message::user("hi")]).await;

        let request = mock.await.unwrap();
        assert_eq!(result.unwrap(), "answer");
        // Bonus: verify the Authorization header was sent correctly.
        // reqwest lowercases header names on the wire, so check case-insensitively.
        let request_lower = request.to_lowercase();
        assert!(
            request_lower.contains("authorization: bearer sk-test"),
            "Authorization header missing or wrong in request:\n{request}"
        );
    }

    // 2. Error path: non-2xx → Err (http_error path)
    #[tokio::test]
    async fn complete_errors_on_401() {
        let body = r#"{"error":"Unauthorized"}"#;
        let response = format!(
            "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let (port, mock) = spawn_mock(response).await;

        let llm = OpenAiLlm::new(&make_config(port, Some("bad-key"))).expect("client build");
        let result = llm.complete(&[Message::user("hi")]).await;

        mock.await.unwrap();
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("401"),
            "Expected 401 in error message, got: {msg}"
        );
    }

    // 2b. Error path: 500 also triggers http_error
    #[tokio::test]
    async fn complete_errors_on_500() {
        let body = r#"{"error":"Internal Server Error"}"#;
        let response = format!(
            "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let (port, mock) = spawn_mock(response).await;

        let llm = OpenAiLlm::new(&make_config(port, None)).expect("client build");
        let result = llm.complete(&[Message::user("hi")]).await;

        mock.await.unwrap();
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("500"),
            "Expected 500 in error message, got: {msg}"
        );
    }

    // 3a. Edge: 200 with empty choices array → Err (missing-content path)
    #[tokio::test]
    async fn complete_errors_on_empty_choices() {
        let body = r#"{"choices":[]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let (port, mock) = spawn_mock(response).await;

        let llm = OpenAiLlm::new(&make_config(port, None)).expect("client build");
        let result = llm.complete(&[Message::user("hi")]).await;

        mock.await.unwrap();
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("choices[0].message.content"),
            "Expected missing-content error, got: {err}"
        );
    }

    // 3b. Edge: 200 with message missing the content field → Err (missing-content path)
    #[tokio::test]
    async fn complete_errors_on_missing_content_field() {
        let body = r#"{"choices":[{"message":{}}]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let (port, mock) = spawn_mock(response).await;

        let llm = OpenAiLlm::new(&make_config(port, None)).expect("client build");
        let result = llm.complete(&[Message::user("hi")]).await;

        mock.await.unwrap();
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("choices[0].message.content"),
            "Expected missing-content error, got: {err}"
        );
    }
}
