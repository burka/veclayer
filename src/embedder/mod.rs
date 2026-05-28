//! Embedding trait and backend implementations.
//!
//! [`Embedder`] defines the interface for converting text batches into
//! fixed-size float vectors. [`FastEmbedder`] is the production backend,
//! running models locally via the `fastembed` crate. [`OllamaEmbedder`] calls
//! external HTTP embedding services (Ollama, TEI, OpenAI-compatible).
//! All implementations must be `Send + Sync` for concurrent use across async tasks.

#[cfg(feature = "embedding-local")]
mod fastembed_impl;
#[cfg(feature = "llm")]
mod ollama_impl;

#[cfg(feature = "embedding-local")]
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
    ///
    /// Returns a boxed future rather than `-> impl Future` because `dyn Embedder` is used
    /// widely across the codebase and RPITIT is not dyn-compatible; a boxed future keeps
    /// the trait object-safe without needing a separate `DynEmbedder` wrapper (unlike
    /// `LlmProvider`/`DynLlmProvider`).
    #[allow(clippy::type_complexity)]
    fn embed<'a>(
        &'a self,
        texts: &'a [&'a str],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + 'a>>;

    /// Get the dimension of the embedding vectors produced by this embedder.
    fn dimension(&self) -> usize;

    /// Get the name/identifier of this embedder.
    fn name(&self) -> &str;
}

// Implement Embedder for Arc<T> where T: Embedder.
// Cannot use arc_impl! here because the embed signature carries a lifetime
// that the macro cannot express; hand-write the delegation instead.
impl<T: Embedder + ?Sized> Embedder for std::sync::Arc<T> {
    fn embed<'a>(
        &'a self,
        texts: &'a [&'a str],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + 'a>>
    {
        (**self).embed(texts)
    }

    fn dimension(&self) -> usize {
        (**self).dimension()
    }

    fn name(&self) -> &str {
        (**self).name()
    }
}

impl<T: Embedder + ?Sized> Embedder for Box<T> {
    fn embed<'a>(
        &'a self,
        texts: &'a [&'a str],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + 'a>>
    {
        (**self).embed(texts)
    }

    fn dimension(&self) -> usize {
        (**self).dimension()
    }

    fn name(&self) -> &str {
        (**self).name()
    }
}

/// Resolve a fastembed model name to an `EmbeddingModel`, falling back to the
/// default when the name is unrecognised. Returns `(model, fell_back)`.
///
/// Pure function: no I/O, no model download, no side effects. The fallback
/// preserves backward compatibility with config values like
/// "BAAI/bge-small-en-v1.5" that pre-date the Xenova naming convention used by
/// fastembed.
#[cfg(feature = "embedding-local")]
fn resolve_fastembed_model(name: &str) -> (fastembed::EmbeddingModel, bool) {
    match name.parse::<fastembed::EmbeddingModel>() {
        Ok(m) => (m, false),
        Err(_) => (fastembed::EmbeddingModel::BGESmallENV15, true),
    }
}

/// Create an embedder from configuration.
pub fn from_config(config: &EmbedderConfig) -> Result<Box<dyn Embedder + Send + Sync>> {
    match config {
        #[cfg(feature = "embedding-local")]
        EmbedderConfig::FastEmbed { model } => {
            let (model_type, fell_back) = resolve_fastembed_model(model);
            if fell_back {
                tracing::warn!(
                    "Unrecognised fastembed model '{}', falling back to default",
                    model
                );
            }
            Ok(Box::new(FastEmbedder::with_model(model_type)?))
        }
        #[cfg(not(feature = "embedding-local"))]
        EmbedderConfig::FastEmbed { .. } => Err(crate::Error::config(
            "FastEmbed embedder requires the 'embedding-local' feature flag. Configure an external embedder instead: `veclayer setup ollama --apply`",
        )),
        #[cfg(feature = "llm")]
        EmbedderConfig::Ollama {
            model,
            base_url,
            dimension,
        } => Ok(Box::new(OllamaEmbedder::new(model, base_url, *dimension)?)),
        #[cfg(not(feature = "llm"))]
        EmbedderConfig::Ollama { .. } => Err(crate::Error::config(
            "Ollama embedder requires the 'llm' feature flag. Build with default features or `--features llm`",
        )),
    }
}

#[cfg(all(test, feature = "embedding-local"))]
mod tests {
    use super::resolve_fastembed_model;
    use fastembed::EmbeddingModel;

    #[test]
    fn known_model_name_resolves_without_fallback() {
        // fastembed's FromStr matches the model_code; Display emits that same
        // code, so a Display round-trip yields a name parse() accepts.
        let name = EmbeddingModel::BGESmallENV15.to_string();
        let (model, fell_back) = resolve_fastembed_model(&name);
        assert!(!fell_back, "known name must not trigger fallback");
        assert!(
            matches!(model, EmbeddingModel::BGESmallENV15),
            "must resolve to the correct variant"
        );
    }

    #[test]
    fn unrecognised_model_name_falls_back_to_default() {
        // fastembed's model_code for BGESmallENV15 is "Xenova/bge-small-en-v1.5",
        // so the BAAI-prefixed legacy name misses FromStr and must fall back.
        let (model, fell_back) = resolve_fastembed_model("BAAI/bge-small-en-v1.5");
        assert!(fell_back, "unrecognised name must trigger fallback");
        assert!(
            matches!(model, EmbeddingModel::BGESmallENV15),
            "fallback must be BGESmallENV15"
        );
    }

    #[test]
    fn empty_model_name_falls_back_to_default() {
        let (model, fell_back) = resolve_fastembed_model("");
        assert!(fell_back);
        assert!(matches!(model, EmbeddingModel::BGESmallENV15));
    }
}

// Blanket-impl tests for Arc<T> and Box<T>: feature-gate-free because the
// blankets themselves are unconditional and MockEmbedder is plain #[cfg(test)].
#[cfg(test)]
mod blanket_tests {
    use crate::test_helpers::MockEmbedder;
    use crate::Embedder;

    #[test]
    fn arc_embedder_forwards_dimension_and_name() {
        let arc: std::sync::Arc<MockEmbedder> = std::sync::Arc::new(MockEmbedder::new(7));
        assert_eq!(Embedder::dimension(&arc), 7);
        assert_eq!(Embedder::name(&arc), "mock");
    }

    #[tokio::test]
    async fn arc_embedder_embed_forwards_to_inner() {
        let arc = std::sync::Arc::new(MockEmbedder::new(4));
        let result = arc.embed(&["a", "b"]).await.expect("embed should succeed");
        assert_eq!(result, vec![vec![1.0_f32; 4]; 2]);
    }

    #[test]
    fn box_embedder_forwards_dimension_and_name() {
        let boxed: Box<MockEmbedder> = Box::new(MockEmbedder::new(9));
        assert_eq!(Embedder::dimension(&boxed), 9);
        assert_eq!(Embedder::name(&boxed), "mock");
    }

    #[tokio::test]
    async fn box_embedder_embed_forwards_to_inner() {
        let boxed: Box<MockEmbedder> = Box::new(MockEmbedder::new(5));
        let result = boxed
            .embed(&["x", "y", "z"])
            .await
            .expect("embed should succeed");
        assert_eq!(result, vec![vec![1.0_f32; 5]; 3]);
    }
}
