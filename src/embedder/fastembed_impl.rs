use std::sync::{Arc, OnceLock};

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use super::Embedder;
use crate::{Error, Result};

/// Returns the fastembed model cache directory inside VecLayer's cache path.
fn fastembed_cache_dir() -> std::path::PathBuf {
    crate::default_cache_dir().join("fastembed")
}

/// FastEmbed-based embedder using local ONNX models.
/// Runs entirely on CPU, no external API required.
///
/// The underlying ONNX session is initialized lazily on first real embed call.
/// This keeps MCP stdio startup cheap for short-lived sessions that only need
/// initialization metadata and never actually run semantic operations.
pub struct FastEmbedder {
    model: OnceLock<std::result::Result<Arc<TextEmbedding>, String>>,
    model_type: EmbeddingModel,
    cache_dir: std::path::PathBuf,
    dimension: usize,
    model_name: String,
}

impl FastEmbedder {
    /// Create a new FastEmbedder with the default model (BGE-small-en-v1.5)
    pub fn new() -> Result<Self> {
        Self::with_model(EmbeddingModel::BGESmallENV15)
    }

    /// Create a FastEmbedder with a specific model
    pub fn with_model(model_type: EmbeddingModel) -> Result<Self> {
        let model_name = format!("{:?}", model_type);
        let dimension = Self::get_dimension(&model_type);

        Ok(Self {
            model: OnceLock::new(),
            model_type,
            cache_dir: fastembed_cache_dir(),
            dimension,
            model_name,
        })
    }

    fn get_or_init_model(&self) -> Result<Arc<TextEmbedding>> {
        let init = self.model.get_or_init(|| {
            tracing::debug!("Initializing FastEmbed model {}", self.model_name);
            let options =
                InitOptions::new(self.model_type.clone()).with_cache_dir(self.cache_dir.clone());
            TextEmbedding::try_new(options)
                .map(Arc::new)
                .map_err(|e| format!("Failed to initialize FastEmbed: {e}"))
        });

        match init {
            Ok(model) => Ok(Arc::clone(model)),
            Err(msg) => Err(Error::embedding(msg.clone())),
        }
    }

    fn get_dimension(model: &EmbeddingModel) -> usize {
        match model {
            EmbeddingModel::BGESmallENV15 => 384,
            EmbeddingModel::BGEBaseENV15 => 768,
            EmbeddingModel::BGELargeENV15 => 1024,
            EmbeddingModel::AllMiniLML6V2 => 384,
            EmbeddingModel::AllMiniLML12V2 => 384,
            _ => 384, // Default fallback
        }
    }
}

impl Default for FastEmbedder {
    fn default() -> Self {
        Self::new().expect("Failed to create default FastEmbedder")
    }
}

impl Embedder for FastEmbedder {
    fn embed<'a>(
        &'a self,
        texts: &'a [&'a str],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + 'a>>
    {
        let owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        Box::pin(async move {
            if owned.is_empty() {
                return Ok(vec![]);
            }
            let model = self.get_or_init_model()?;
            tokio::task::spawn_blocking(move || {
                model
                    .embed(owned, None)
                    .map_err(|e| Error::embedding(format!("Embedding failed: {}", e)))
            })
            .await
            .map_err(|e| Error::embedding(format!("Embedding task panicked: {}", e)))?
        })
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        &self.model_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires ONNX model file (download via fastembed)"]
    fn test_fastembed_creation() {
        let embedder = FastEmbedder::new();
        assert!(embedder.is_ok());
    }

    #[tokio::test]
    #[ignore = "requires ONNX model file (download via fastembed)"]
    async fn test_fastembed_embed() {
        let embedder = FastEmbedder::new().unwrap();
        let texts = vec!["Hello world", "This is a test"];
        let embeddings = embedder.embed(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), embedder.dimension());
        assert_eq!(embeddings[1].len(), embedder.dimension());
    }

    #[tokio::test]
    #[ignore = "requires ONNX model file (download via fastembed)"]
    async fn test_fastembed_empty() {
        let embedder = FastEmbedder::new().unwrap();
        let texts: Vec<&str> = vec![];
        let embeddings = embedder.embed(&texts).await.unwrap();
        assert!(embeddings.is_empty());
    }

    #[test]
    fn test_get_dimension_known_models() {
        assert_eq!(
            FastEmbedder::get_dimension(&EmbeddingModel::BGESmallENV15),
            384
        );
        assert_eq!(
            FastEmbedder::get_dimension(&EmbeddingModel::BGEBaseENV15),
            768
        );
        assert_eq!(
            FastEmbedder::get_dimension(&EmbeddingModel::BGELargeENV15),
            1024
        );
        assert_eq!(
            FastEmbedder::get_dimension(&EmbeddingModel::AllMiniLML6V2),
            384
        );
    }
}
