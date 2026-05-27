//! LLM providers for the think/sleep cycle.
//!
//! VecLayer uses LLMs only in the think cycle: reflect → LLM → add → compact.
//! Everything else works without an LLM.

pub mod ollama;
pub mod openai;

pub use ollama::OllamaLlm;
pub use openai::OpenAiLlm;

use reqwest::Client;
use std::time::Duration;

use crate::util::{build_hardened_client, read_capped_body, MAX_HTTP_BODY_BYTES};

/// Build a hardened reqwest `Client` with standard timeouts (10s connect, 120s
/// overall) and redirects disabled.
pub fn make_standard_http_client() -> crate::Result<Client> {
    build_hardened_client(Duration::from_secs(10), Duration::from_secs(120))
        .ok_or_else(|| crate::Error::llm("failed to build reqwest client"))
}

/// Read the body from a non-success HTTP response and format it as an LLM error.
///
/// Uses the capped reader so a lying server cannot cause OOM via error responses.
pub async fn http_error(service_name: &str, resp: reqwest::Response) -> crate::Error {
    let status = resp.status();
    let body = read_capped_body(resp, MAX_HTTP_BODY_BYTES)
        .await
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default();
    crate::Error::llm(format!("{service_name} returned {status}: {body}"))
}

/// A message in a chat conversation.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }

    /// Convert a slice of messages to the JSON array format used by most LLM APIs.
    pub fn to_json_values(messages: &[Message]) -> Vec<serde_json::Value> {
        messages
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
            .collect()
    }
}

/// Trait for LLM text generation.
///
/// Implementations call an LLM API and return the response text.
/// VecLayer uses this only in the think/sleep cycle.
pub trait LlmProvider: Send + Sync {
    fn complete(
        &self,
        messages: &[Message],
    ) -> impl std::future::Future<Output = crate::Result<String>> + Send;

    fn name(&self) -> &str;
}

/// Object-safe version of [`LlmProvider`] for type-erased storage.
///
/// The main `LlmProvider` trait uses RPITIT (`-> impl Future`) which prevents
/// `dyn LlmProvider`. This wrapper uses boxed futures for object safety,
/// allowing `Box<dyn DynLlmProvider>` in the facade.
pub trait DynLlmProvider: Send + Sync {
    fn complete(
        &self,
        messages: &[Message],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<String>> + Send + '_>>;

    fn dyn_name(&self) -> &str;
}

/// Wrapper that adapts any [`LlmProvider`] into a [`DynLlmProvider`].
///
/// Use this to box a concrete `LlmProvider` for storage in the facade.
pub struct DynLlmProviderWrapper<T>(pub T);

impl<T: LlmProvider> DynLlmProvider for DynLlmProviderWrapper<T> {
    fn complete(
        &self,
        messages: &[Message],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<String>> + Send + '_>>
    {
        // Clone messages into an owned Vec so the future doesn't borrow the slice.
        let owned: Vec<Message> = messages.to_vec();
        Box::pin(async move { LlmProvider::complete(&self.0, &owned).await })
    }

    fn dyn_name(&self) -> &str {
        LlmProvider::name(&self.0)
    }
}

// LlmConfig lives in config.rs (not feature-gated). Re-export for convenience.
pub use crate::config::LlmConfig;

/// Enum-based dispatch for LLM providers.
/// Avoids trait objects while supporting multiple backends.
pub enum LlmBackend {
    Ollama(OllamaLlm),
    OpenAi(OpenAiLlm),
}

impl LlmProvider for LlmBackend {
    async fn complete(&self, messages: &[Message]) -> crate::Result<String> {
        match self {
            Self::Ollama(o) => o.complete(messages).await,
            Self::OpenAi(o) => o.complete(messages).await,
        }
    }

    fn name(&self) -> &str {
        match self {
            Self::Ollama(o) => o.name(),
            Self::OpenAi(o) => o.name(),
        }
    }
}

impl LlmBackend {
    /// Create an LLM backend from config.
    pub fn from_config(config: &LlmConfig) -> crate::Result<Self> {
        match config.provider.as_str() {
            "openai" => Ok(Self::OpenAi(OpenAiLlm::new(config)?)),
            _ => Ok(Self::Ollama(OllamaLlm::new(config)?)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_constructors() {
        let sys = Message::system("you are helpful");
        assert_eq!(sys.role, Role::System);
        assert_eq!(sys.content, "you are helpful");

        let usr = Message::user("hello");
        assert_eq!(usr.role, Role::User);

        let asst = Message::assistant("hi there");
        assert_eq!(asst.role, Role::Assistant);
    }

    #[test]
    fn test_llm_config_default() {
        let config = LlmConfig::default();
        assert_eq!(config.provider, "ollama");
        assert_eq!(config.model, "llama3.2");
        assert!(config.api_key.is_none());
    }

    #[test]
    fn test_llm_backend_from_config_ollama() {
        let config = LlmConfig::default();
        let backend = LlmBackend::from_config(&config).expect("client build should succeed");
        assert_eq!(backend.name(), "llama3.2");
    }

    #[test]
    fn test_llm_backend_from_config_openai() {
        let config = LlmConfig {
            provider: "openai".to_string(),
            model: "gpt-4o".to_string(),
            base_url: "https://api.openai.com".to_string(),
            api_key: Some("sk-test".to_string()),
            ..Default::default()
        };
        let backend = LlmBackend::from_config(&config).expect("client build should succeed");
        assert_eq!(backend.name(), "gpt-4o");
    }

    #[test]
    fn test_make_standard_http_client_returns_ok() {
        let result = make_standard_http_client();
        assert!(result.is_ok(), "expected Ok from make_standard_http_client");
    }
}
