//! Embedding trait and backend implementations.
//!
//! [`Embedder`] defines the interface for converting text batches into
//! fixed-size float vectors. [`FastEmbedder`] is the production backend,
//! running models locally via the `fastembed` crate. [`OllamaEmbedder`] calls
//! external HTTP embedding services (Ollama, TEI, OpenAI-compatible).
//! All implementations must be `Send + Sync` for concurrent use across async tasks.

mod fastembed_impl;
#[cfg(feature = "llm")]
mod ollama_impl;

pub use fastembed_impl::FastEmbedder;
#[cfg(feature = "llm")]
pub use ollama_impl::OllamaEmbedder;

use crate::config::EmbedderConfig;
use crate::Result;

/// Trait for embedding text into vectors.
/// Implementations should be thread-safe for concurrent embedding.
pub trait Embedder: Send + Sync {
    /// Embed a batch of texts into vectors.
    /// Returns a vector of embeddings, one per input text.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Get the dimension of the embedding vectors produced by this embedder.
    fn dimension(&self) -> usize;

    /// Get the name/identifier of this embedder.
    fn name(&self) -> &str;
}

// Implement Embedder for Arc<T> where T: Embedder
crate::arc_impl!(Embedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
    fn name(&self) -> &str;
});

impl<T: Embedder + ?Sized> Embedder for Box<T> {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        (**self).embed(texts)
    }

    fn dimension(&self) -> usize {
        (**self).dimension()
    }

    fn name(&self) -> &str {
        (**self).name()
    }
}

/// Create an embedder from configuration.
pub fn from_config(config: &EmbedderConfig) -> Result<Box<dyn Embedder + Send + Sync>> {
    match config {
        EmbedderConfig::FastEmbed { model } => {
            // Try parsing the model name; fall back to the default model if unrecognised.
            // This preserves backward compatibility with config values like "BAAI/bge-small-en-v1.5"
            // that pre-date the Xenova model code convention used by fastembed.
            let embedder = match model.parse::<fastembed::EmbeddingModel>() {
                Ok(m) => FastEmbedder::with_model(m)?,
                Err(_) => {
                    tracing::warn!(
                        "Unrecognised fastembed model '{}', falling back to default",
                        model
                    );
                    FastEmbedder::new()?
                }
            };
            Ok(Box::new(embedder))
        }
        #[cfg(feature = "llm")]
        EmbedderConfig::Ollama {
            model,
            base_url,
            dimension,
        } => Ok(Box::new(OllamaEmbedder::new(model, base_url, *dimension)?)),
        #[cfg(not(feature = "llm"))]
        EmbedderConfig::Ollama { .. } => Err(crate::Error::config(
            "Ollama embedder requires the 'llm' feature flag",
        )),
    }
}
