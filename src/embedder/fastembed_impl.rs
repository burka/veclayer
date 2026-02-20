use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use super::Embedder;
use crate::{Error, Result};

/// FastEmbed-based embedder using local ONNX models.
/// Runs entirely on CPU, no external API required.
pub struct FastEmbedder {
    model: TextEmbedding,
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

        let model = TextEmbedding::try_new(InitOptions::new(model_type))
            .map_err(|e| Error::embedding(format!("Failed to initialize FastEmbed: {}", e)))?;

        Ok(Self {
            model,
            dimension,
            model_name,
        })
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
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

        self.model
            .embed(texts, None)
            .map_err(|e| Error::embedding(format!("Embedding failed: {}", e)))
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

    #[test]
    #[ignore = "requires ONNX model file (download via fastembed)"]
    fn test_fastembed_embed() {
        let embedder = FastEmbedder::new().unwrap();
        let texts = vec!["Hello world", "This is a test"];
        let embeddings = embedder.embed(&texts).unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), embedder.dimension());
        assert_eq!(embeddings[1].len(), embedder.dimension());
    }

    #[test]
    #[ignore = "requires ONNX model file (download via fastembed)"]
    fn test_fastembed_empty() {
        let embedder = FastEmbedder::new().unwrap();
        let texts: Vec<&str> = vec![];
        let embeddings = embedder.embed(&texts).unwrap();
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
