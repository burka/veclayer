//! Ollama LLM provider using the /api/chat endpoint.

use std::time::Duration;

use reqwest::Client;

use super::{LlmConfig, LlmProvider, Message, Role};

pub struct OllamaLlm {
    client: Client,
    model: String,
    base_url: String,
    temperature: f32,
}

impl OllamaLlm {
    pub fn new(config: &LlmConfig) -> Self {
        Self {
            client: Client::builder()
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(120))
                .build()
                .expect("reqwest client"),
            model: config.model.clone(),
            base_url: config.base_url.clone(),
            temperature: config.temperature,
        }
    }
}

impl LlmProvider for OllamaLlm {
    async fn complete(&self, messages: &[Message]) -> crate::Result<String> {
        let msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": match m.role {
                        Role::System => "system",
                        Role::User => "user",
                        Role::Assistant => "assistant",
                    },
                    "content": m.content,
                })
            })
            .collect();

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
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(crate::Error::llm(format!(
                "Ollama returned {}: {}",
                status, body
            )));
        }

        let body: serde_json::Value = resp
            .json()
            .await
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
