use std::time::Duration;

use super::Summarizer;
use crate::util::{
    build_hardened_client, read_capped_body, DEFAULT_OLLAMA_URL, MAX_HTTP_BODY_BYTES,
};
use crate::{Error, Result};

/// Summarizer using Ollama for local LLM inference.
/// Free and runs locally - perfect for RAPTOR-style summarization.
pub struct OllamaSummarizer {
    model: String,
    base_url: String,
    client: reqwest::Client,
}

impl OllamaSummarizer {
    /// Create a new Ollama summarizer with default settings.
    /// Uses llama3.2 model and localhost:11434 endpoint.
    ///
    /// Fallible (hence no `Default` impl) because building the hardened HTTP
    /// client can fail.
    pub fn new() -> Result<Self> {
        Ok(Self {
            model: "llama3.2".to_string(),
            base_url: DEFAULT_OLLAMA_URL.to_string(),
            client: build_hardened_client(Duration::from_secs(10), Duration::from_secs(120))
                .ok_or_else(|| Error::summarization("failed to build reqwest client"))?,
        })
    }

    /// Set the model in place, reusing the existing HTTP client.
    pub fn set_model(&mut self, model: impl Into<String>) {
        self.model = model.into();
    }

    /// Use a specific model
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.set_model(model);
        self
    }

    /// Use a specific Ollama endpoint
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Build the summarization prompt
    fn build_prompt(&self, texts: &[&str]) -> String {
        let combined = texts.join("\n\n---\n\n");
        format!(
            r#"You are a summarization assistant. Your task is to create a concise summary that captures the key themes and information from the following text chunks.

These chunks may come from different documents but share semantic similarity. Focus on:
1. Common themes and topics
2. Key facts and information
3. Important relationships between concepts

TEXT CHUNKS:
{combined}

Provide a clear, comprehensive summary in 2-4 sentences that would help someone understand what these texts are about without reading them in full. Do not use phrases like "The texts discuss" or "These chunks cover" - write directly about the content."#
        )
    }
}

#[derive(serde::Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(serde::Deserialize)]
struct OllamaResponse {
    response: String,
}

impl Summarizer for OllamaSummarizer {
    async fn summarize(&self, texts: &[&str]) -> Result<String> {
        if texts.is_empty() {
            return Ok(String::new());
        }

        let prompt = self.build_prompt(texts);
        let url = format!("{}/api/generate", self.base_url);

        let request = OllamaRequest {
            model: self.model.clone(),
            prompt,
            stream: false,
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::summarization(format!("Failed to connect to Ollama: {}", e)))?;

        let status = response.status();
        let bytes = read_capped_body(response, MAX_HTTP_BODY_BYTES)
            .await
            .map_err(|e| Error::summarization(format!("Failed to read Ollama response: {}", e)))?;

        if !status.is_success() {
            return Err(Error::summarization(format!(
                "Ollama request failed ({}): {}",
                status,
                String::from_utf8_lossy(&bytes)
            )));
        }

        let ollama_response: OllamaResponse = serde_json::from_slice(&bytes)
            .map_err(|e| Error::summarization(format!("Failed to parse Ollama response: {}", e)))?;

        Ok(ollama_response.response.trim().to_string())
    }

    fn name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_building() {
        let summarizer = OllamaSummarizer::new().expect("client build should succeed");
        let prompt = summarizer.build_prompt(&["Text one", "Text two"]);
        assert!(prompt.contains("Text one"));
        assert!(prompt.contains("Text two"));
        assert!(prompt.contains("---"));
    }

    #[test]
    fn test_new_returns_ok() {
        let result = OllamaSummarizer::new();
        assert!(result.is_ok(), "OllamaSummarizer::new() should return Ok");
        assert_eq!(result.unwrap().name(), "llama3.2");
    }
}
