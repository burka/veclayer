//! Ollama LLM provider using the /api/chat endpoint.

use super::{make_standard_http_client, LlmConfig, LlmProvider, Message};
use crate::util::{read_capped_body, MAX_HTTP_BODY_BYTES};
use reqwest::Client;

pub struct OllamaLlm {
    client: Client,
    model: String,
    base_url: String,
    temperature: f32,
}

impl OllamaLlm {
    pub fn new(config: &LlmConfig) -> crate::Result<Self> {
        Ok(Self {
            client: make_standard_http_client()?,
            model: config.model.clone(),
            base_url: config.base_url.clone(),
            temperature: config.temperature,
        })
    }
}

impl LlmProvider for OllamaLlm {
    async fn complete(&self, messages: &[Message]) -> crate::Result<String> {
        let msgs = Message::to_json_values(messages);

        let resp = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&serde_json::json!({
                "model": self.model,
                "messages": msgs,
                "stream": false,
                "options": {
                    "temperature": self.temperature,
                }
            }))
            .send()
            .await
            .map_err(|e| crate::Error::llm(format!("Ollama request failed: {}", e)))?;

        if !resp.status().is_success() {
            return Err(crate::llm::http_error("Ollama", resp).await);
        }

        let bytes = read_capped_body(resp, MAX_HTTP_BODY_BYTES)
            .await
            .map_err(|e| crate::Error::llm(format!("Ollama response read failed: {}", e)))?;
        let body: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| crate::Error::llm(format!("Ollama response parse failed: {}", e)))?;

        body["message"]["content"]
            .as_str()
            .map(String::from)
            .ok_or_else(|| crate::Error::llm("Ollama response missing message.content"))
    }

    fn name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LlmConfig;
    use crate::test_helpers::{mock_listener, serve_once};

    fn ollama_at(base_url: &str) -> OllamaLlm {
        OllamaLlm::new(&LlmConfig {
            provider: "ollama".to_string(),
            model: "llama3.2".to_string(),
            base_url: base_url.to_string(),
            api_key: None,
            temperature: 0.0,
            max_tokens: 256,
        })
        .expect("client build should succeed")
    }

    // --- green path ---

    #[tokio::test]
    async fn complete_returns_content_on_200() {
        let (listener, base_url) = mock_listener().await;
        serve_once(listener, 200, r#"{"message":{"content":"hello world"}}"#);

        let llm = ollama_at(&base_url);
        let result = llm.complete(&[Message::user("ping")]).await;

        assert_eq!(result.unwrap(), "hello world");
    }

    // --- error path: non-2xx response ---

    #[tokio::test]
    async fn complete_returns_err_on_500() {
        let (listener, base_url) = mock_listener().await;
        serve_once(listener, 500, r#"{"error":"internal server error"}"#);

        let llm = ollama_at(&base_url);
        let result = llm.complete(&[Message::user("ping")]).await;

        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("500"),
            "expected HTTP status in error, got: {msg}"
        );
    }

    #[tokio::test]
    async fn complete_returns_err_on_404() {
        let (listener, base_url) = mock_listener().await;
        serve_once(listener, 404, r#"not found"#);

        let llm = ollama_at(&base_url);
        let result = llm.complete(&[Message::user("ping")]).await;

        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("404"),
            "expected HTTP 404 in error, got: {msg}"
        );
    }

    // --- edge path: 200 but missing message.content ---

    #[tokio::test]
    async fn complete_returns_err_when_content_field_absent() {
        let (listener, base_url) = mock_listener().await;
        // `message` object exists but `content` key is missing.
        serve_once(listener, 200, r#"{"message":{}}"#);

        let llm = ollama_at(&base_url);
        let result = llm.complete(&[Message::user("ping")]).await;

        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("missing message.content"),
            "expected missing-content error, got: {err}"
        );
    }

    #[tokio::test]
    async fn complete_returns_err_when_message_field_absent() {
        let (listener, base_url) = mock_listener().await;
        // Top-level `message` key is absent entirely.
        serve_once(listener, 200, r#"{"response":"unexpected shape"}"#);

        let llm = ollama_at(&base_url);
        let result = llm.complete(&[Message::user("ping")]).await;

        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("missing message.content"),
            "expected missing-content error, got: {err}"
        );
    }

    // --- connection-refused path ---

    #[tokio::test]
    async fn complete_returns_err_on_connection_refused() {
        // Bind and immediately drop to ensure the port is not listening.
        let (listener, base_url) = mock_listener().await;
        drop(listener);

        let llm = ollama_at(&base_url);
        let result = llm.complete(&[Message::user("ping")]).await;

        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("request failed"),
            "expected request-failed error, got: {err}"
        );
    }
}
