use super::Embedder;
use crate::Result;

/// Deterministic local embedder used as default for VecLayer.
///
/// This implementation intentionally avoids external model downloads so builds
/// and tests remain fully executable in restricted environments.
pub struct FastEmbedder {
    dimension: usize,
    model_name: String,
}

impl FastEmbedder {
    /// Create a new embedder with default dimension.
    pub fn new() -> Result<Self> {
        Ok(Self {
            dimension: 384,
            model_name: "fastembed-local".to_string(),
        })
    }

    /// Create a new embedder with a custom dimension.
    pub fn with_dimension(dimension: usize) -> Result<Self> {
        Ok(Self {
            dimension: dimension.max(8),
            model_name: format!("fastembed-local-{}", dimension.max(8)),
        })
    }

    fn embed_one(&self, text: &str) -> Vec<f32> {
        // Simple hashed bag-of-words projection (deterministic, no static outputs).
        let mut vec = vec![0.0f32; self.dimension];
        for (token_idx, token) in text.split_whitespace().enumerate() {
            let mut hash = 0xcbf29ce484222325u64;
            for b in token.as_bytes() {
                hash ^= *b as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            let index = (hash as usize + token_idx) % self.dimension;
            let sign = if ((hash >> 1) & 1) == 0 { 1.0 } else { -1.0 };
            let weight = 1.0 + ((hash & 0xF) as f32 / 16.0);
            vec[index] += sign * weight;
        }

        // L2 normalize for stable similarity behavior.
        let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }

        vec
    }
}

impl Default for FastEmbedder {
    fn default() -> Self {
        Self::new().expect("Failed to create default FastEmbedder")
    }
}

impl Embedder for FastEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed_one(t)).collect())
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
    fn embedder_creation_and_dimension() {
        let embedder = FastEmbedder::new().expect("embedder should initialize");
        assert_eq!(embedder.dimension(), 384);
        assert_eq!(embedder.name(), "fastembed-local");

        let custom = FastEmbedder::with_dimension(96).expect("custom embedder should initialize");
        assert_eq!(custom.dimension(), 96);
        assert_eq!(custom.name(), "fastembed-local-96");
    }

    #[test]
    fn embedding_is_deterministic_and_non_static() {
        let embedder = FastEmbedder::with_dimension(64).unwrap();
        let v1 = embedder.embed(&["hello world"]).unwrap().remove(0);
        let v2 = embedder.embed(&["hello world"]).unwrap().remove(0);
        let v3 = embedder.embed(&["different terms"]).unwrap().remove(0);

        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
        assert_eq!(v1.len(), 64);
        assert_eq!(v3.len(), 64);
    }

    #[test]
    fn empty_batch_returns_empty_embeddings() {
        let embedder = FastEmbedder::new().unwrap();
        let values = embedder.embed(&[]).unwrap();
        assert!(values.is_empty());
    }
}
