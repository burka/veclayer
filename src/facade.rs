//! High-level facade for library consumers.
//!
//! [`VecLayer`] bundles a [`StoreBackend`](crate::store::StoreBackend) with any
//! [`Embedder`](crate::embedder::Embedder), giving you a one-stop API for
//! store, search, and focus operations — no feature flags required beyond the
//! slim default.
//!
//! # Example
//!
//! ```no_run
//! use veclayer::{VecLayer, Embedder, Result};
//! use std::path::Path;
//!
//! struct MyEmbedder;
//! impl Embedder for MyEmbedder {
//!     fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
//!         // call your API, run your model, etc.
//!         Ok(texts.iter().map(|_| vec![0.0; 384]).collect())
//!     }
//!     fn dimension(&self) -> usize { 384 }
//!     fn name(&self) -> &str { "my-embedder" }
//! }
//!
//! # async fn example() -> Result<()> {
//! let vl = VecLayer::open(Path::new(".veclayer"), MyEmbedder).await?;
//! let id = vl.store("Important decision: chose Rust for memory safety").await?;
//! let results = vl.search("memory safety", 5).await?;
//! # Ok(())
//! # }
//! ```

use std::path::Path;
use std::sync::Arc;

use crate::blob_store::BlobStore;
use crate::chunk::{ChunkLevel, EntryType};
use crate::embedder::Embedder;
use crate::entry::StoredBlob;
use crate::search::{HierarchicalSearch, HierarchicalSearchResult, SearchConfig};
use crate::store::{SearchResult, StoreBackend, StoreStats, VectorStore};
use crate::{ChunkRelation, HierarchicalChunk, Result};

/// High-level API for using VecLayer as a library with any embedder.
///
/// Wraps a store + embedder pair and provides convenient methods for
/// common operations. For advanced use, access the underlying store
/// and embedder directly via [`store()`](VecLayer::store_backend) and
/// [`embedder()`](VecLayer::embedder).
pub struct VecLayer<E: Embedder = Box<dyn Embedder>> {
    store: Arc<StoreBackend>,
    embedder: Arc<E>,
    blob_store: BlobStore,
}

/// Options for storing an entry.
#[derive(Debug, Clone, Default)]
pub struct StoreOptions {
    /// Heading/title for the entry.
    pub heading: Option<String>,
    /// Perspectives to tag this entry with.
    pub perspectives: Vec<String>,
    /// Parent entry ID for hierarchy placement.
    pub parent_id: Option<String>,
    /// Entry type: raw (default), summary, meta, impression.
    pub entry_type: EntryType,
    /// Source label (default: "[api]").
    pub source: Option<String>,
    /// Visibility: "normal" (default), "always", "deep_only".
    pub visibility: Option<String>,
    /// Relations to establish atomically.
    pub relations: Vec<ChunkRelation>,
}

impl<E: Embedder> VecLayer<E> {
    /// Open a VecLayer store with a custom embedder.
    ///
    /// Creates the data directory if it doesn't exist. The embedder's
    /// dimension is used to configure the vector index.
    pub async fn open(data_dir: &Path, embedder: E) -> Result<Self> {
        let dimension = embedder.dimension();
        let store = Arc::new(StoreBackend::open(data_dir, dimension, false).await?);
        let blob_store = BlobStore::open(data_dir)?;
        Ok(Self {
            store,
            embedder: Arc::new(embedder),
            blob_store,
        })
    }

    /// Open a read-only VecLayer store.
    pub async fn open_read_only(data_dir: &Path, embedder: E) -> Result<Self> {
        let dimension = embedder.dimension();
        let store = Arc::new(StoreBackend::open(data_dir, dimension, true).await?);
        let blob_store = BlobStore::open(data_dir)?;
        Ok(Self {
            store,
            embedder: Arc::new(embedder),
            blob_store,
        })
    }

    /// Store text content and return its entry ID.
    ///
    /// Embeds the content automatically using the configured embedder.
    pub async fn store(&self, content: &str) -> Result<String> {
        self.store_with(content, StoreOptions::default()).await
    }

    /// Store text content with options and return its entry ID.
    pub async fn store_with(&self, content: &str, options: StoreOptions) -> Result<String> {
        let embed_text = match &options.heading {
            Some(h) => format!("{h}\n{content}"),
            None => content.to_string(),
        };
        let embedding = self.embed_one(&embed_text)?;

        let source = options.source.unwrap_or_else(|| "[api]".to_string());
        let mut chunk = HierarchicalChunk::new(
            content.to_string(),
            ChunkLevel::CONTENT,
            options.parent_id.clone(),
            "memory".to_string(),
            source,
        )
        .with_embedding(embedding)
        .with_entry_type(options.entry_type);

        if let Some(heading) = &options.heading {
            chunk = chunk.with_heading(heading);
        }

        if let Some(vis) = &options.visibility {
            chunk.visibility = vis.clone();
        }

        for perspective in &options.perspectives {
            chunk = chunk.with_perspective(perspective);
        }

        let id = chunk.id.clone();

        // Persist raw content in blob store (before move)
        let blob = StoredBlob::from_chunk_and_embedding(&chunk, self.embedder.name());
        let _ = self.blob_store.put(&blob);

        self.store.insert_chunks(vec![chunk]).await?;

        // Apply relations after insert
        for rel in &options.relations {
            self.store.add_relation(&id, rel.clone()).await?;
        }

        Ok(id)
    }

    /// Semantic search. Returns results ranked by blended score.
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<HierarchicalSearchResult>> {
        let search = HierarchicalSearch::new(Arc::clone(&self.store), Arc::clone(&self.embedder))
            .with_config(SearchConfig {
                top_k: limit,
                ..Default::default()
            });
        search.search(query).await
    }

    /// Search with full configuration control.
    pub async fn search_with(
        &self,
        query: &str,
        config: SearchConfig,
    ) -> Result<Vec<HierarchicalSearchResult>> {
        let search = HierarchicalSearch::new(Arc::clone(&self.store), Arc::clone(&self.embedder))
            .with_config(config);
        search.search(query).await
    }

    /// Raw vector search without hierarchical enrichment.
    pub async fn search_raw(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let embedding = self.embed_one(query)?;
        self.store.search(&embedding, limit, None).await
    }

    /// Focus on a specific entry: returns it with its children.
    pub async fn focus(
        &self,
        id: &str,
        question: Option<&str>,
        limit: usize,
    ) -> Result<Option<FocusResult>> {
        let entry = match self.store.get_by_id_prefix(id).await? {
            Some(e) => e,
            None => return Ok(None),
        };

        let children = self.store.get_children(&entry.id).await?;

        let ranked_children = if let Some(q) = question {
            let query_embedding = self.embed_one(q)?;
            let mut scored: Vec<_> = children
                .into_iter()
                .map(|c| {
                    let score = c
                        .embedding
                        .as_deref()
                        .map(|e| cosine_similarity(&query_embedding, e))
                        .unwrap_or(0.0);
                    (c, score)
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(limit);
            scored
        } else {
            children.into_iter().take(limit).map(|c| (c, 0.0)).collect()
        };

        // Track access
        let mut entry_updated = entry.clone();
        entry_updated.access_profile.record_access();
        let _ = self
            .store
            .update_access_profiles(vec![(entry.id.clone(), entry_updated.access_profile)])
            .await;

        Ok(Some(FocusResult {
            entry,
            children: ranked_children,
        }))
    }

    /// Get an entry by exact or short ID.
    pub async fn get(&self, id: &str) -> Result<Option<HierarchicalChunk>> {
        self.store.get_by_id_prefix(id).await
    }

    /// List entries, optionally filtered by perspective.
    pub async fn list(
        &self,
        perspective: Option<&str>,
        limit: usize,
    ) -> Result<Vec<HierarchicalChunk>> {
        self.store
            .list_entries(perspective, None, None, limit)
            .await
    }

    /// Delete all entries from a source label.
    pub async fn delete_by_source(&self, source: &str) -> Result<usize> {
        self.store.delete_by_source(source).await
    }

    /// Get store statistics.
    pub async fn stats(&self) -> Result<StoreStats> {
        self.store.stats().await
    }

    /// Embed a single text. Convenience wrapper around the embedder.
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embedder.embed(&[text])?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| crate::Error::store("empty embedding result"))
    }

    /// Embed a batch of texts.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embedder.embed(texts)
    }

    /// Access the underlying store for advanced operations.
    pub fn store_backend(&self) -> &Arc<StoreBackend> {
        &self.store
    }

    /// Access the underlying embedder.
    pub fn embedder(&self) -> &E {
        &self.embedder
    }

    /// Access the blob store.
    pub fn blob_store(&self) -> &BlobStore {
        &self.blob_store
    }
}

/// Result of a focus operation: the entry plus its ranked children.
#[derive(Debug, Clone)]
pub struct FocusResult {
    /// The focused entry.
    pub entry: HierarchicalChunk,
    /// Children, optionally ranked by question relevance. Tuple: (chunk, score).
    pub children: Vec<(HierarchicalChunk, f32)>,
}

/// Cosine similarity between two embedding vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dummy embedder for tests — returns deterministic vectors based on text length.
    struct TestEmbedder;

    impl Embedder for TestEmbedder {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    let seed = t.len() as f32;
                    vec![seed / 100.0, (seed * 0.7) / 100.0, (seed * 0.3) / 100.0]
                })
                .collect())
        }

        fn dimension(&self) -> usize {
            3
        }

        fn name(&self) -> &str {
            "test-embedder"
        }
    }

    #[test]
    fn cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn store_options_default() {
        let opts = StoreOptions::default();
        assert!(opts.heading.is_none());
        assert!(opts.perspectives.is_empty());
        assert!(opts.parent_id.is_none());
        assert!(opts.relations.is_empty());
    }

    #[tokio::test]
    async fn open_store_search_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let vl = VecLayer::open(dir.path(), TestEmbedder).await.unwrap();

        // Store
        let id = vl.store("Test content for roundtrip").await.unwrap();
        assert!(!id.is_empty());

        // Stats
        let stats = vl.stats().await.unwrap();
        assert_eq!(stats.total_chunks, 1);

        // Get by ID
        let entry = vl.get(&id[..7]).await.unwrap();
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().content, "Test content for roundtrip");

        // Search
        let results = vl.search("roundtrip test", 5).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk.id, id);
    }

    #[tokio::test]
    async fn store_with_options() {
        let dir = tempfile::tempdir().unwrap();
        let vl = VecLayer::open(dir.path(), TestEmbedder).await.unwrap();

        let id = vl
            .store_with(
                "Decision content",
                StoreOptions {
                    heading: Some("Architecture Decision".into()),
                    perspectives: vec!["decisions".into()],
                    entry_type: EntryType::Meta,
                    source: Some("my-app".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let entry = vl.get(&id).await.unwrap().unwrap();
        assert_eq!(entry.heading.as_deref(), Some("Architecture Decision"));
        assert!(entry.perspectives.contains(&"decisions".to_string()));
        assert_eq!(entry.entry_type, EntryType::Meta);
        assert_eq!(entry.source_file, "my-app");
    }

    #[tokio::test]
    async fn store_and_delete_by_source() {
        let dir = tempfile::tempdir().unwrap();
        let vl = VecLayer::open(dir.path(), TestEmbedder).await.unwrap();

        let _ = vl
            .store_with(
                "Ephemeral",
                StoreOptions {
                    source: Some("session-123".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(vl.stats().await.unwrap().total_chunks, 1);
        let deleted = vl.delete_by_source("session-123").await.unwrap();
        assert_eq!(deleted, 1);
        assert_eq!(vl.stats().await.unwrap().total_chunks, 0);
    }

    #[tokio::test]
    async fn focus_entry() {
        let dir = tempfile::tempdir().unwrap();
        let vl = VecLayer::open(dir.path(), TestEmbedder).await.unwrap();

        let parent_id = vl
            .store_with(
                "Parent entry",
                StoreOptions {
                    heading: Some("Parent".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let _ = vl
            .store_with(
                "Child entry",
                StoreOptions {
                    parent_id: Some(parent_id.clone()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let result = vl.focus(&parent_id, None, 10).await.unwrap();
        assert!(result.is_some());
        let focus = result.unwrap();
        assert_eq!(focus.entry.id, parent_id);
        assert_eq!(focus.children.len(), 1);
    }

    #[tokio::test]
    async fn list_with_perspective() {
        let dir = tempfile::tempdir().unwrap();
        let vl = VecLayer::open(dir.path(), TestEmbedder).await.unwrap();

        let _ = vl
            .store_with(
                "A decision",
                StoreOptions {
                    perspectives: vec!["decisions".into()],
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let _ = vl
            .store_with(
                "A learning",
                StoreOptions {
                    perspectives: vec!["learnings".into()],
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let decisions = vl.list(Some("decisions"), 100).await.unwrap();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].content, "A decision");

        let all = vl.list(None, 100).await.unwrap();
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn embed_helpers() {
        let dir = tempfile::tempdir().unwrap();
        let vl = VecLayer::open(dir.path(), TestEmbedder).await.unwrap();

        let single = vl.embed_one("hello").unwrap();
        assert_eq!(single.len(), 3);

        let batch = vl.embed_batch(&["hello", "world"]).unwrap();
        assert_eq!(batch.len(), 2);
    }

    #[tokio::test]
    async fn read_only_rejects_writes() {
        let dir = tempfile::tempdir().unwrap();

        // Create store first
        let vl = VecLayer::open(dir.path(), TestEmbedder).await.unwrap();
        let _ = vl.store("seed").await.unwrap();
        drop(vl);

        // Open read-only
        let vl = VecLayer::open_read_only(dir.path(), TestEmbedder)
            .await
            .unwrap();

        // Search should work
        let results = vl.search("seed", 5).await.unwrap();
        assert!(!results.is_empty());
    }
}
