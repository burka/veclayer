//! Entry: the portable, serializable form of a [`HierarchicalChunk`].
//!
//! An [`Entry`] is the canonical subset of chunk fields that can be persisted
//! and round-tripped without the embedding vector. [`StoredBlob`] wraps an
//! `Entry` with zero or more cached [`EmbeddingCache`] vectors, serialized via
//! postcard for compact binary storage in the blob store.

use serde::{Deserialize, Serialize};

use crate::chunk::{default_visibility, ChunkLevel, ChunkRelation, EntryType, HierarchicalChunk};

fn default_impression_strength() -> f32 {
    1.0
}

/// A canonical entry — the core unit persisted to storage.
///
/// Fields are the portable subset of `HierarchicalChunk` that can be round-tripped
/// through the blob store independently of the embedding or index details.
///
/// # Examples
///
/// ```
/// use veclayer::{HierarchicalChunk, ChunkLevel};
/// use veclayer::entry::Entry;
///
/// let chunk = HierarchicalChunk::new(
///     "Architecture decision: use LanceDB".to_string(),
///     ChunkLevel::H1,
///     None,
///     "root".to_string(),
///     "decisions.md".to_string(),
/// );
///
/// let entry = Entry::from_chunk(&chunk);
/// assert_eq!(entry.content, chunk.content);
/// assert_eq!(entry.content_id(), chunk.id);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Entry {
    pub content: String,
    pub entry_type: EntryType,
    pub source: String,
    pub created_at: i64,
    #[serde(default)]
    pub perspectives: Vec<String>,
    #[serde(default)]
    pub relations: Vec<ChunkRelation>,
    #[serde(default)]
    pub summarizes: Vec<String>,
    #[serde(default)]
    pub heading: Option<String>,
    #[serde(default)]
    pub parent_id: Option<String>,
    #[serde(default)]
    pub impression_hint: Option<String>,
    #[serde(default = "default_impression_strength")]
    pub impression_strength: f32,
    #[serde(default)]
    pub expires_at: Option<i64>,
    #[serde(default = "default_visibility")]
    pub visibility: String,
    pub level: ChunkLevel,
    #[serde(default)]
    pub path: String,
}

impl Entry {
    /// Returns a content-addressed ID derived from the entry's content.
    ///
    /// # Design: heading is intentionally excluded from the hash
    ///
    /// The heading is *not* included in the hash so that IDs remain stable when a
    /// heading is renamed.  Two entries with identical `content` but different
    /// headings therefore share a `content_id`, but their git filenames diverge
    /// because `entry_filename` derives the slug from the heading.
    ///
    /// Changing this contract would invalidate every persisted entry — do not
    /// alter the hash function without a migration plan.
    pub fn content_id(&self) -> String {
        crate::chunk::content_hash(&self.content)
    }

    /// Extracts canonical entry fields from an existing chunk.
    /// Preserves the chunk's `created_at` timestamp.
    pub fn from_chunk(chunk: &HierarchicalChunk) -> Self {
        Self {
            content: chunk.content.clone(),
            entry_type: chunk.entry_type,
            source: chunk.source_file.clone(),
            created_at: chunk.access_profile.created_at,
            perspectives: chunk.perspectives.clone(),
            relations: chunk.relations.clone(),
            summarizes: chunk.summarizes.clone(),
            heading: chunk.heading.clone(),
            parent_id: chunk.parent_id.clone(),
            impression_hint: chunk.impression_hint.clone(),
            impression_strength: chunk.impression_strength,
            expires_at: chunk.expires_at,
            visibility: chunk.visibility.clone(),
            level: chunk.level,
            path: chunk.path.clone(),
        }
    }
}

/// A cached embedding produced by a specific model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingCache {
    pub model: String,
    pub dimensions: u16,
    pub vector: Vec<f32>,
}

/// The full blob persisted to the object store: entry + zero or more embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredBlob {
    pub entry: Entry,
    #[serde(default)]
    pub embeddings: Vec<EmbeddingCache>,
}

impl StoredBlob {
    /// Content-addressed hash — based on the entry's content ID only.
    ///
    /// Uses `content_hash(content)` (SHA-256) as the key, then blake3-hashes
    /// that to produce the storage path. This ensures the same logical content
    /// always maps to the same blob, regardless of timestamps or embeddings.
    pub fn blob_hash(&self) -> blake3::Hash {
        let content_id = self.entry.content_id();
        blake3::hash(content_id.as_bytes())
    }

    /// Create a `StoredBlob` from a `HierarchicalChunk` and its embedding model name.
    ///
    /// Extracts the canonical `Entry` from the chunk and caches the chunk's
    /// embedding (if present) with model provenance.
    pub fn from_chunk_and_embedding(chunk: &HierarchicalChunk, model_name: &str) -> Self {
        let entry = Entry::from_chunk(chunk);
        let embeddings = match chunk.embedding.as_ref() {
            Some(vec) => {
                debug_assert!(
                    vec.len() <= u16::MAX as usize,
                    "embedding dimension {} exceeds u16::MAX (65535) — silent truncation would occur",
                    vec.len()
                );
                vec![EmbeddingCache {
                    model: model_name.to_string(),
                    dimensions: vec.len() as u16,
                    vector: vec.clone(),
                }]
            }
            None => vec![],
        };
        Self { entry, embeddings }
    }

    /// Serialise to postcard bytes.
    pub fn to_bytes(&self) -> crate::Result<Vec<u8>> {
        postcard::to_allocvec(self).map_err(|e| crate::Error::store(e.to_string()))
    }

    /// Deserialise from postcard bytes.
    pub fn from_bytes(bytes: &[u8]) -> crate::Result<Self> {
        postcard::from_bytes(bytes).map_err(|e| crate::Error::store(e.to_string()))
    }

    /// Return the embedding vector for the given model, if present.
    pub fn embedding_for_model(&self, model: &str) -> Option<&[f32]> {
        self.embeddings
            .iter()
            .find(|e| e.model == model)
            .map(|e| e.vector.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{ChunkLevel, ChunkRelation, EntryType};

    fn sample_entry() -> Entry {
        Entry {
            content: "Test content".to_string(),
            entry_type: EntryType::Raw,
            source: "test.md".to_string(),
            created_at: 1_700_000_000,
            perspectives: vec!["decisions".to_string()],
            relations: vec![ChunkRelation::new("related_to", "other-id")],
            summarizes: vec!["sum-id".to_string()],
            heading: Some("Test Heading".to_string()),
            parent_id: Some("parent-id".to_string()),
            impression_hint: Some("confident".to_string()),
            impression_strength: 1.0,
            expires_at: Some(1_800_000_000),
            visibility: "normal".to_string(),
            level: ChunkLevel::H1,
            path: "root".to_string(),
        }
    }

    fn sample_blob() -> StoredBlob {
        StoredBlob {
            entry: sample_entry(),
            embeddings: vec![EmbeddingCache {
                model: "test-model".to_string(),
                dimensions: 3,
                vector: vec![0.1, 0.2, 0.3],
            }],
        }
    }

    #[test]
    fn test_round_trip() {
        let blob = sample_blob();
        let bytes = blob.to_bytes().unwrap();
        let restored = StoredBlob::from_bytes(&bytes).unwrap();
        assert_eq!(blob.entry, restored.entry);
        assert_eq!(blob.embeddings, restored.embeddings);
    }

    #[test]
    fn test_content_id_deterministic() {
        let entry = sample_entry();
        assert_eq!(entry.content_id(), entry.content_id());

        let other = Entry {
            content: "Different content".to_string(),
            ..sample_entry()
        };
        assert_ne!(entry.content_id(), other.content_id());
    }

    #[test]
    fn test_blob_hash_deterministic() {
        let blob = sample_blob();
        let hash1 = blob.blob_hash();
        let hash2 = blob.blob_hash();
        assert_eq!(hash1, hash2);

        let other = StoredBlob {
            entry: Entry {
                content: "Different content".to_string(),
                ..sample_entry()
            },
            embeddings: vec![EmbeddingCache {
                model: "test-model".to_string(),
                dimensions: 3,
                vector: vec![0.1, 0.2, 0.3],
            }],
        };
        let hash3 = other.blob_hash();
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_embedding_for_model() {
        let blob = StoredBlob {
            entry: sample_entry(),
            embeddings: vec![
                EmbeddingCache {
                    model: "model-a".to_string(),
                    dimensions: 2,
                    vector: vec![1.0, 2.0],
                },
                EmbeddingCache {
                    model: "model-b".to_string(),
                    dimensions: 3,
                    vector: vec![3.0, 4.0, 5.0],
                },
            ],
        };

        assert_eq!(
            blob.embedding_for_model("model-a").unwrap(),
            &[1.0_f32, 2.0]
        );
        assert_eq!(
            blob.embedding_for_model("model-b").unwrap(),
            &[3.0_f32, 4.0, 5.0]
        );
        assert!(blob.embedding_for_model("model-c").is_none());
    }

    #[test]
    fn test_from_chunk_and_embedding() {
        let chunk = crate::chunk::HierarchicalChunk::new(
            "Test content".to_string(),
            crate::chunk::ChunkLevel::H1,
            None,
            "root".to_string(),
            "test.md".to_string(),
        )
        .with_entry_type(crate::chunk::EntryType::Meta)
        .with_perspectives(vec!["decisions".to_string()])
        .with_embedding(vec![0.1, 0.2, 0.3]);

        let blob = StoredBlob::from_chunk_and_embedding(&chunk, "test-model");

        assert_eq!(blob.entry.content, "Test content");
        assert_eq!(blob.entry.entry_type, crate::chunk::EntryType::Meta);
        assert_eq!(blob.entry.perspectives, vec!["decisions".to_string()]);
        assert_eq!(blob.embeddings.len(), 1);
        assert_eq!(blob.embeddings[0].model, "test-model");
        assert_eq!(blob.embeddings[0].dimensions, 3);
        assert_eq!(blob.embeddings[0].vector, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_from_chunk_and_embedding_no_embedding() {
        let chunk = crate::chunk::HierarchicalChunk::new(
            "No embedding".to_string(),
            crate::chunk::ChunkLevel::CONTENT,
            None,
            String::new(),
            "test.md".to_string(),
        );

        let blob = StoredBlob::from_chunk_and_embedding(&chunk, "test-model");
        assert!(blob.embeddings.is_empty());
    }

    // --- EmbeddingCache serde ---

    #[test]
    fn test_embedding_cache_serde_roundtrip() {
        let cache = EmbeddingCache {
            model: "text-embedding-3-small".to_string(),
            dimensions: 4,
            vector: vec![0.1, 0.2, 0.3, 0.4],
        };
        let json = serde_json::to_string(&cache).unwrap();
        let restored: EmbeddingCache = serde_json::from_str(&json).unwrap();
        assert_eq!(cache, restored);
    }

    #[test]
    fn test_embedding_cache_empty_vector_roundtrip() {
        let cache = EmbeddingCache {
            model: "none".to_string(),
            dimensions: 0,
            vector: vec![],
        };
        let json = serde_json::to_string(&cache).unwrap();
        let restored: EmbeddingCache = serde_json::from_str(&json).unwrap();
        assert_eq!(cache, restored);
    }

    // --- Entry default deserialization (serde defaults) ---

    #[test]
    fn test_entry_deserializes_with_minimal_fields() {
        // Only required fields; serde defaults should fill in the rest.
        // ChunkLevel(u8) serializes as a plain number.
        let json = r#"{
            "content": "minimal",
            "entry_type": "raw",
            "source": "src.md",
            "created_at": 1700000000,
            "level": 7
        }"#;
        let entry: Entry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.content, "minimal");
        assert!(entry.perspectives.is_empty());
        assert!(entry.relations.is_empty());
        assert!(entry.heading.is_none());
        assert!(entry.parent_id.is_none());
        assert_eq!(entry.impression_strength, 1.0); // default_impression_strength
        assert_eq!(entry.visibility, "normal"); // default_visibility
        assert!(entry.path.is_empty());
    }

    #[test]
    fn test_entry_default_visibility_is_normal() {
        // Construct via from_chunk with no explicit visibility set
        let chunk = crate::chunk::HierarchicalChunk::new(
            "visibility test".to_string(),
            ChunkLevel::CONTENT,
            None,
            String::new(),
            "test.md".to_string(),
        );
        let entry = Entry::from_chunk(&chunk);
        assert_eq!(entry.visibility, "normal");
    }

    #[test]
    fn test_entry_default_impression_strength_is_one() {
        let chunk = crate::chunk::HierarchicalChunk::new(
            "strength test".to_string(),
            ChunkLevel::CONTENT,
            None,
            String::new(),
            "test.md".to_string(),
        );
        let entry = Entry::from_chunk(&chunk);
        assert!((entry.impression_strength - 1.0).abs() < f32::EPSILON);
    }

    // --- from_chunk preservation ---

    #[test]
    fn test_from_chunk_preserves_created_at_timestamp() {
        let mut chunk = crate::chunk::HierarchicalChunk::new(
            "timestamped".to_string(),
            ChunkLevel::H1,
            None,
            String::new(),
            "test.md".to_string(),
        );
        let fixed_ts = 1_234_567_890_i64;
        chunk.access_profile.created_at = fixed_ts;
        let entry = Entry::from_chunk(&chunk);
        assert_eq!(entry.created_at, fixed_ts);
    }

    #[test]
    fn test_from_chunk_preserves_all_metadata_fields() {
        let chunk = crate::chunk::HierarchicalChunk::new(
            "full metadata".to_string(),
            ChunkLevel::H2,
            Some("parent-123".to_string()),
            "root/child".to_string(),
            "source.md".to_string(),
        )
        .with_entry_type(crate::chunk::EntryType::Summary)
        .with_perspectives(vec!["decisions".to_string(), "learnings".to_string()]);

        let entry = Entry::from_chunk(&chunk);
        assert_eq!(entry.entry_type, crate::chunk::EntryType::Summary);
        assert_eq!(entry.perspectives, vec!["decisions", "learnings"]);
        assert_eq!(entry.parent_id, Some("parent-123".to_string()));
        assert_eq!(entry.path, "root/child");
        assert_eq!(entry.source, "source.md");
        assert_eq!(entry.level, ChunkLevel::H2);
    }

    #[test]
    fn test_from_chunk_content_id_matches_chunk_id() {
        let chunk = crate::chunk::HierarchicalChunk::new(
            "content id check".to_string(),
            ChunkLevel::CONTENT,
            None,
            String::new(),
            "test.md".to_string(),
        );
        let entry = Entry::from_chunk(&chunk);
        assert_eq!(entry.content_id(), chunk.id);
    }

    // --- postcard roundtrip ---

    #[test]
    fn test_stored_blob_postcard_roundtrip_no_embeddings() {
        let blob = StoredBlob {
            entry: sample_entry(),
            embeddings: vec![],
        };
        let bytes = blob.to_bytes().unwrap();
        let restored = StoredBlob::from_bytes(&bytes).unwrap();
        assert_eq!(blob.entry, restored.entry);
        assert!(restored.embeddings.is_empty());
    }

    #[test]
    fn test_stored_blob_postcard_roundtrip_multiple_embeddings() {
        let blob = StoredBlob {
            entry: sample_entry(),
            embeddings: vec![
                EmbeddingCache {
                    model: "model-x".to_string(),
                    dimensions: 2,
                    vector: vec![1.0, 2.0],
                },
                EmbeddingCache {
                    model: "model-y".to_string(),
                    dimensions: 3,
                    vector: vec![3.0, 4.0, 5.0],
                },
            ],
        };
        let bytes = blob.to_bytes().unwrap();
        let restored = StoredBlob::from_bytes(&bytes).unwrap();
        assert_eq!(restored.embeddings.len(), 2);
        assert_eq!(restored.embeddings[0].model, "model-x");
        assert_eq!(restored.embeddings[1].model, "model-y");
    }

    #[test]
    fn test_stored_blob_from_bytes_rejects_garbage() {
        let result = StoredBlob::from_bytes(b"this is not postcard");
        assert!(result.is_err());
    }

    // --- blob hash independence ---

    #[test]
    fn test_blob_hash_independent_of_embeddings() {
        // Same content with different embeddings must produce the same blob hash.
        let blob_a = StoredBlob {
            entry: sample_entry(),
            embeddings: vec![EmbeddingCache {
                model: "model-a".to_string(),
                dimensions: 2,
                vector: vec![1.0, 2.0],
            }],
        };
        let blob_b = StoredBlob {
            entry: sample_entry(),
            embeddings: vec![EmbeddingCache {
                model: "model-b".to_string(),
                dimensions: 3,
                vector: vec![9.0, 8.0, 7.0],
            }],
        };
        assert_eq!(blob_a.blob_hash(), blob_b.blob_hash());
    }

    #[test]
    fn test_blob_hash_independent_of_timestamps() {
        // Two entries with the same content but different created_at values
        // map to the same content_id (hash only uses content), so same blob hash.
        let entry_early = Entry {
            created_at: 100,
            ..sample_entry()
        };
        let entry_late = Entry {
            created_at: 9_999_999,
            ..sample_entry()
        };
        let blob_early = StoredBlob {
            entry: entry_early,
            embeddings: vec![],
        };
        let blob_late = StoredBlob {
            entry: entry_late,
            embeddings: vec![],
        };
        assert_eq!(blob_early.blob_hash(), blob_late.blob_hash());
    }

    /// Round-trip test: Entry::from_chunk → HierarchicalChunk::from_entry.
    ///
    /// Documents known information loss: start_offset, end_offset,
    /// cluster_memberships, and access_profile are reset (zero / default).
    #[test]
    fn test_entry_chunk_round_trip() {
        use crate::chunk::HierarchicalChunk;

        let original = HierarchicalChunk::new(
            "Round-trip content".to_string(),
            crate::chunk::ChunkLevel::H2,
            Some("parent-id".to_string()),
            "root/child".to_string(),
            "source.md".to_string(),
        )
        .with_entry_type(crate::chunk::EntryType::Meta)
        .with_perspectives(vec!["knowledge".to_string()])
        .with_embedding(vec![1.0, 2.0, 3.0]);

        let entry = Entry::from_chunk(&original);
        let restored = HierarchicalChunk::from_entry(&entry, vec![1.0, 2.0, 3.0]);

        // Preserved fields
        assert_eq!(restored.content, original.content);
        assert_eq!(restored.id, original.id);
        assert_eq!(restored.level, original.level);
        assert_eq!(restored.parent_id, original.parent_id);
        assert_eq!(restored.path, original.path);
        assert_eq!(restored.source_file, original.source_file);
        assert_eq!(restored.entry_type, original.entry_type);
        assert_eq!(restored.perspectives, original.perspectives);
        assert_eq!(restored.visibility, original.visibility);

        // Known information loss
        assert_eq!(restored.start_offset, 0);
        assert_eq!(restored.end_offset, 0);
        assert!(restored.cluster_memberships.is_empty());
    }

    // --- dimension guard ---

    /// Happy path: a 384-dim vector (typical for text-embedding models) must
    /// round-trip without truncation — `dimensions` must equal the vec length.
    #[test]
    fn test_dimensions_normal_range_round_trips() {
        let embedding: Vec<f32> = vec![0.0_f32; 384];
        let chunk = crate::chunk::HierarchicalChunk::new(
            "dim test".to_string(),
            crate::chunk::ChunkLevel::CONTENT,
            None,
            String::new(),
            "test.md".to_string(),
        )
        .with_embedding(embedding.clone());

        let blob = StoredBlob::from_chunk_and_embedding(&chunk, "model");
        assert_eq!(blob.embeddings[0].dimensions as usize, embedding.len());
    }

    /// Edge: a vector exceeding u16::MAX dimensions trips the `debug_assert`.
    /// Gated on `debug_assertions` because the assert is a no-op under
    /// `--release` (and `cargo test --release` therefore skips this test).
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "embedding dimension")]
    fn test_dimensions_exceeding_u16_max_panics_in_debug() {
        // Allocate a 65536-element (256 KiB) vector — acceptable in a test.
        let oversized: Vec<f32> = vec![0.0_f32; (u16::MAX as usize) + 1];
        let chunk = crate::chunk::HierarchicalChunk::new(
            "oversize dim test".to_string(),
            crate::chunk::ChunkLevel::CONTENT,
            None,
            String::new(),
            "test.md".to_string(),
        )
        .with_embedding(oversized);

        // In debug builds this must panic; in release it would silently truncate.
        StoredBlob::from_chunk_and_embedding(&chunk, "model");
    }
}
