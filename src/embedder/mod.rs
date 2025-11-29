mod fastembed_impl;

pub use fastembed_impl::FastEmbedder;

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

/// Box type for dynamic dispatch of embedders
pub type BoxedEmbedder = Box<dyn Embedder>;

// Implement Embedder for Arc<T> where T: Embedder
impl<T: Embedder> Embedder for std::sync::Arc<T> {
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
