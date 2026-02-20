//! LLM providers for the think/sleep cycle.
//!
//! VecLayer uses LLMs only in the think cycle: reflect → LLM → add → compact.
//! Everything else works without an LLM.

pub mod ollama;
pub mod openai;

pub use ollama::OllamaLlm;
pub use openai::OpenAiLlm;

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
    pub fn from_config(config: &LlmConfig) -> Self {
        match config.provider.as_str() {
            "openai" => Self::OpenAi(OpenAiLlm::new(config)),
            _ => Self::Ollama(OllamaLlm::new(config)),
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
        let backend = LlmBackend::from_config(&config);
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
        let backend = LlmBackend::from_config(&config);
        assert_eq!(backend.name(), "gpt-4o");
    }
}
