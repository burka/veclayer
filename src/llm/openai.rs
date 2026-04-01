//! OpenAI-compatible LLM provider.
//!
//! Works with OpenAI, Azure OpenAI, LM Studio, and any API that implements
//! the /v1/chat/completions endpoint.

use super::{make_standard_http_client, LlmConfig, LlmProvider, Message};
use reqwest::Client;

pub struct OpenAiLlm {
    client: Client,
    model: String,
    base_url: String,
    api_key: String,
    temperature: f32,
    max_tokens: usize,
}

impl OpenAiLlm {
    pub fn new(config: &LlmConfig) -> Self {
        Self {
            client: make_standard_http_client(),
            model: config.model.clone(),
            base_url: config.base_url.clone(),
            api_key: config.api_key.clone().unwrap_or_default(),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
        }
    }
}

impl LlmProvider for OpenAiLlm {
    async fn complete(&self, messages: &[Message]) -> crate::Result<String> {
        let msgs = Message::to_json_values(messages);

        let resp = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
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
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(crate::Error::llm(format!(
                "OpenAI API returned {}: {}",
                status, body
            )));
        }

        let body: serde_json::Value = resp
            .json()
            .await
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
