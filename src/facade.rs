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
//! use std::path::Path;
//! use std::pin::Pin;
//!
//! use veclayer::{Embedder, Result, VecLayer};
//!
//! struct MyEmbedder;
//! impl Embedder for MyEmbedder {
//!     fn embed<'a>(
//!         &'a self,
//!         texts: &'a [&'a str],
//!     ) -> Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + 'a>> {
//!         // call your API, run your model, etc.
//!         let result: Vec<Vec<f32>> = texts.iter().map(|_| vec![0.0; 384]).collect();
//!         Box::pin(async move { Ok(result) })
//!     }
//!     fn dimension(&self) -> usize {
//!         384
//!     }
//!     fn name(&self) -> &str {
//!         "my-embedder"
//!     }
//! }
//!
//! # async fn example() -> Result<()> {
//! let vl = VecLayer::open(Path::new(".veclayer"), MyEmbedder).await?;
//! let id = vl.store("Important decision: chose Rust for memory safety").await?;
//! let results = vl.search("memory safety", 5).await?;
//! # Ok(())
//! # }
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::blob_store::BlobStore;
use crate::chunk::{ChunkLevel, EntryType};
use crate::embedder::Embedder;
use crate::entry::StoredBlob;
use crate::search::{self, HierarchicalSearch, HierarchicalSearchResult, SearchConfig};
use crate::store::{SearchResult, StoreBackend, StoreStats, VectorStore};
use crate::{ChunkRelation, HierarchicalChunk, Result};

/// High-level API for using VecLayer as a library with any embedder.
///
/// Wraps a store + embedder pair and provides convenient methods for
/// common operations. For advanced use, access the underlying store
/// and embedder directly via [`store()`](VecLayer::store_backend) and
/// [`embedder()`](VecLayer::embedder).
pub struct VecLayer<E: Embedder = Box<dyn Embedder>> {
    data_dir: PathBuf,
    store: Arc<StoreBackend>,
    embedder: Arc<E>,
    blob_store: BlobStore,
    #[cfg(feature = "llm")]
    llm: std::sync::RwLock<Option<Arc<dyn crate::llm::DynLlmProvider>>>,
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
    /// Visibility level. Use constants from [`chunk::visibility`](crate::chunk::visibility):
    /// `NORMAL` (default), `ALWAYS`, `DEEP_ONLY`, `SEASONAL`.
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
        Self::open_inner(data_dir, embedder, false).await
    }

    /// Open a read-only VecLayer store.
    pub async fn open_read_only(data_dir: &Path, embedder: E) -> Result<Self> {
        Self::open_inner(data_dir, embedder, true).await
    }

    async fn open_inner(data_dir: &Path, embedder: E, read_only: bool) -> Result<Self> {
        let dimension = embedder.dimension();
        let store = Arc::new(StoreBackend::open(data_dir, dimension, read_only).await?);
        let blob_store = BlobStore::open(data_dir)?;
        Ok(Self::from_parts(
            data_dir.to_path_buf(),
            store,
            embedder,
            blob_store,
        ))
    }

    /// Open a VecLayer store backed by SQLite.
    ///
    /// Same as [`open`](Self::open) but forces the SQLite backend regardless of
    /// which backends are compiled. Useful when you want a lightweight store
    /// without LanceDB's transitive dependencies.
    #[cfg(feature = "store-sqlite")]
    pub async fn open_sqlite(data_dir: &Path, embedder: E) -> Result<Self> {
        Self::open_sqlite_inner(data_dir, embedder, false).await
    }

    /// Open a read-only VecLayer store backed by SQLite.
    #[cfg(feature = "store-sqlite")]
    pub async fn open_sqlite_read_only(data_dir: &Path, embedder: E) -> Result<Self> {
        Self::open_sqlite_inner(data_dir, embedder, true).await
    }

    #[cfg(feature = "store-sqlite")]
    async fn open_sqlite_inner(data_dir: &Path, embedder: E, read_only: bool) -> Result<Self> {
        let dimension = embedder.dimension();
        let store = Arc::new(StoreBackend::open_sqlite(data_dir, dimension, read_only).await?);
        let blob_store = BlobStore::open(data_dir)?;
        Ok(Self::from_parts(
            data_dir.to_path_buf(),
            store,
            embedder,
            blob_store,
        ))
    }

    fn from_parts(
        data_dir: PathBuf,
        store: Arc<StoreBackend>,
        embedder: E,
        blob_store: BlobStore,
    ) -> Self {
        Self {
            data_dir,
            store,
            embedder: Arc::new(embedder),
            blob_store,
            #[cfg(feature = "llm")]
            llm: std::sync::RwLock::new(None),
        }
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
        let embedding = self.embed_one(&embed_text).await?;

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

        // Record initial access so newly stored entries are "hot" (visible to
        // identity/think cycles that use get_hot_chunks).
        chunk.access_profile.record_access();

        let id = chunk.id.clone();

        // Persist raw content in blob store (before move)
        let blob = StoredBlob::from_chunk_and_embedding(&chunk, self.embedder.name());
        self.blob_store.put(&blob)?;

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
        let embedding = self.embed_one(query).await?;
        let results = self.store.search(&embedding, limit, None, &[]).await?;

        // Track access for returned entries
        let ids: Vec<_> = results
            .iter()
            .map(|r| {
                let mut ap = r.chunk.access_profile.clone();
                ap.record_access();
                (r.chunk.id.clone(), ap)
            })
            .collect();
        if let Err(e) = self.store.update_access_profiles(ids).await {
            tracing::warn!("recall: failed to update access profiles: {}", e);
        }

        Ok(results)
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
            let query_embedding = self.embed_one(q).await?;
            let mut scored: Vec<_> = children
                .into_iter()
                .map(|c| {
                    let score = c
                        .embedding
                        .as_deref()
                        .map(|e| search::cosine_similarity(&query_embedding, e))
                        .unwrap_or(0.0);
                    (c, score)
                })
                .collect();
            crate::chunk::sort_f32_desc(&mut scored, |r| r.1);
            scored.truncate(limit);
            scored
        } else {
            children.into_iter().take(limit).map(|c| (c, 0.0)).collect()
        };

        // Track access
        let mut entry_updated = entry.clone();
        entry_updated.access_profile.record_access();
        if let Err(e) = self
            .store
            .update_access_profiles(vec![(entry.id.clone(), entry_updated.access_profile)])
            .await
        {
            tracing::warn!("focus: failed to update access profiles: {}", e);
        }

        Ok(Some(FocusResult {
            entry,
            children: ranked_children,
        }))
    }

    /// Get an entry by exact or short ID.
    pub async fn get(&self, id: &str) -> Result<Option<HierarchicalChunk>> {
        self.store.get_by_id_prefix(id).await
    }

    /// List entries, optionally filtered by perspectives. Empty slice = no filter.
    pub async fn list(
        &self,
        perspectives: &[&str],
        limit: usize,
    ) -> Result<Vec<HierarchicalChunk>> {
        self.store
            .list_entries(perspectives, None, None, limit)
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
    pub async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embedder.embed(&[text]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| crate::Error::store("empty embedding result"))
    }

    /// Embed a batch of texts.
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embedder.embed(texts).await
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

    /// Access the data directory path.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Set the LLM provider for think cycles.
    ///
    /// The provider is stored as a type-erased trait object and used by
    /// [`think()`](Self::think). Can be called multiple times to swap providers.
    #[cfg(feature = "llm")]
    pub fn configure_llm(&self, llm: impl crate::llm::LlmProvider + 'static) {
        *self.llm.write().unwrap_or_else(|p| p.into_inner()) =
            Some(Arc::new(crate::llm::DynLlmProviderWrapper(llm)));
    }

    /// Run a think cycle using the configured LLM provider.
    ///
    /// Returns an error if no LLM has been configured via [`configure_llm()`](Self::configure_llm).
    /// Use [`think_with()`](Self::think_with) for one-off calls with a specific provider.
    #[cfg(feature = "llm")]
    pub async fn think(&self) -> Result<crate::think::ThinkResult> {
        self.think_project(None).await
    }

    /// Run a think cycle scoped to a specific project.
    ///
    /// When `project` is `Some`, only entries tagged with that project are
    /// considered. When `None`, all entries are included.
    #[cfg(feature = "llm")]
    pub async fn think_project(&self, project: Option<&str>) -> Result<crate::think::ThinkResult> {
        let llm = {
            let guard = self.llm.read().unwrap_or_else(|p| p.into_inner());
            Arc::clone(guard.as_ref().ok_or_else(|| {
                crate::Error::llm("no LLM configured — call configure_llm() first")
            })?)
        }; // guard dropped here, before await
        crate::think::execute_dyn(
            self.store.as_ref(),
            self.embedder.as_ref(),
            llm.as_ref(),
            &self.data_dir,
            Some(&self.blob_store),
            project,
        )
        .await
    }

    /// Run a think cycle with a specific LLM provider (one-off override).
    ///
    /// Does not require or modify the configured LLM slot.
    #[cfg(feature = "llm")]
    pub async fn think_with(
        &self,
        llm: &dyn crate::llm::DynLlmProvider,
    ) -> Result<crate::think::ThinkResult> {
        self.think_with_project(llm, None).await
    }

    /// Run a think cycle with a specific LLM provider, scoped to a project.
    #[cfg(feature = "llm")]
    pub async fn think_with_project(
        &self,
        llm: &dyn crate::llm::DynLlmProvider,
        project: Option<&str>,
    ) -> Result<crate::think::ThinkResult> {
        crate::think::execute_dyn(
            self.store.as_ref(),
            self.embedder.as_ref(),
            llm,
            &self.data_dir,
            Some(&self.blob_store),
            project,
        )
        .await
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Dummy embedder for tests — returns deterministic vectors based on text length.
    struct TestEmbedder;

    impl Embedder for TestEmbedder {
        fn embed<'a>(
            &'a self,
            texts: &'a [&'a str],
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + 'a>>
        {
            let result: Vec<Vec<f32>> = texts
                .iter()
                .map(|t| {
                    let seed = t.len() as f32;
                    vec![seed / 100.0, (seed * 0.7) / 100.0, (seed * 0.3) / 100.0]
                })
                .collect();
            Box::pin(async move { Ok(result) })
        }

        fn dimension(&self) -> usize {
            3
        }

        fn name(&self) -> &str {
            "test-embedder"
        }
    }

    #[test]
    fn store_options_default() {
        let opts = StoreOptions::default();
        assert!(opts.heading.is_none());
        assert!(opts.perspectives.is_empty());
        assert!(opts.parent_id.is_none());
        assert!(opts.relations.is_empty());
    }

    /// Generate identical test suites for every compiled storage backend.
    ///
    /// Adding a new backend? Just add a `#[cfg] mod` block inside this macro
    /// with `open` and `open_ro` helper fns — the tests are written once.
    macro_rules! all_backend_tests {
        (
            $(fn $test_name:ident($vl:ident $(, $open_ro:ident)?) $body:block)+
        ) => {
            // ── Lance ──────────────────────────────────────────────
            #[cfg(feature = "store-lance")]
            mod lance {
                use super::*;

                async fn open(p: &std::path::Path) -> Result<VecLayer<TestEmbedder>> {
                    VecLayer::open(p, TestEmbedder).await
                }
                async fn open_ro(p: &std::path::Path) -> Result<VecLayer<TestEmbedder>> {
                    VecLayer::open_read_only(p, TestEmbedder).await
                }

                $( all_backend_tests!(@one $test_name, $vl $(, $open_ro)?, $body); )+
            }

            // ── SQLite ─────────────────────────────────────────────
            #[cfg(feature = "store-sqlite")]
            mod sqlite {
                use super::*;

                async fn open(p: &std::path::Path) -> Result<VecLayer<TestEmbedder>> {
                    VecLayer::open_sqlite(p, TestEmbedder).await
                }
                async fn open_ro(p: &std::path::Path) -> Result<VecLayer<TestEmbedder>> {
                    VecLayer::open_sqlite_read_only(p, TestEmbedder).await
                }

                $( all_backend_tests!(@one $test_name, $vl $(, $open_ro)?, $body); )+
            }

            // ─────────────────────────────────────────────────────────
            // NEW BACKEND? Copy one of the blocks above, swap the
            // feature gate and the open/open_ro bodies. Done — every
            // test defined below runs against your backend automatically.
            // ─────────────────────────────────────────────────────────
        };

        // Internal: emit a single test fn. The two-binding variant passes
        // the tempdir as `$dir` so the body can reopen with `open_ro($dir.path())`.
        (@one $test_name:ident, $vl:ident, $dir:ident, $body:block) => {
            #[tokio::test]
            async fn $test_name() {
                let $dir = tempfile::tempdir().unwrap();
                let $vl = open($dir.path()).await.unwrap();
                $body
            }
        };
        (@one $test_name:ident, $vl:ident, $body:block) => {
            #[tokio::test]
            async fn $test_name() {
                let _dir = tempfile::tempdir().unwrap();
                let $vl = open(_dir.path()).await.unwrap();
                $body
            }
        };
    }

    // Tests written once, run against every compiled backend.
    all_backend_tests! {
        // ── get / focus on missing id ─────────────────────────────────────────

        fn get_missing_id_returns_none(vl) {
            // "deadbeef" is a valid id prefix that does not exist in an empty store.
            let result = vl.get("deadbeef").await.unwrap();
            assert!(result.is_none(), "get on unknown id should return None");
        }

        fn focus_missing_id_returns_none(vl) {
            let result = vl.focus("deadbeef", None, 10).await.unwrap();
            assert!(result.is_none(), "focus on unknown id should return None");
        }

        fn focus_missing_id_with_question_returns_none(vl) {
            let result = vl.focus("deadbeef", Some("what happened?"), 10).await.unwrap();
            assert!(result.is_none(), "focus with question on unknown id should return None");
        }

        // ── search edge cases ─────────────────────────────────────────────────

        fn search_with_limit_zero_returns_empty(vl) {
            let _ = vl.store("some content here").await.unwrap();
            let results = vl.search("some content", 0).await.unwrap();
            // limit=0 → top_k=0; search may return empty or the store may clamp — either way, no panic.
            // In practice the store should respect top_k=0 and return nothing.
            assert_eq!(results.len(), 0, "search with limit=0 should return no results");
        }

        fn search_on_empty_store_returns_empty(vl) {
            let results = vl.search("query with nothing stored", 5).await.unwrap();
            assert!(results.is_empty(), "search on empty store should return empty vec");
        }

        fn search_with_on_empty_store_returns_empty(vl) {
            let config = SearchConfig {
                top_k: 10,
                ..Default::default()
            };
            let results = vl.search_with("anything", config).await.unwrap();
            assert!(results.is_empty(), "search_with on empty store should return empty vec");
        }

        fn search_with_zero_top_k_returns_empty(vl) {
            let _ = vl.store("stored content").await.unwrap();
            let config = SearchConfig {
                top_k: 0,
                ..Default::default()
            };
            let results = vl.search_with("stored content", config).await.unwrap();
            assert_eq!(results.len(), 0, "search_with top_k=0 should return no results");
        }

        // ── stats on empty store ───────────────────────────────────────────────

        fn stats_on_empty_store(vl) {
            let stats = vl.stats().await.unwrap();
            assert_eq!(stats.total_chunks, 0, "fresh store should have 0 chunks");
            assert!(
                stats.chunks_by_level.is_empty(),
                "fresh store should have no per-level chunk buckets"
            );
        }

        // ── list edge cases ───────────────────────────────────────────────────

        fn list_on_empty_store_returns_empty(vl) {
            let results = vl.list(&[], 100).await.unwrap();
            assert!(results.is_empty(), "list on empty store should return empty vec");
        }

        fn list_after_stores_returns_all(vl) {
            let _ = vl.store("first entry").await.unwrap();
            let _ = vl.store("second entry").await.unwrap();
            let _ = vl.store("third entry").await.unwrap();

            let results = vl.list(&[], 100).await.unwrap();
            assert_eq!(results.len(), 3, "list should return all 3 entries");
        }

        fn list_limit_is_respected(vl) {
            let _ = vl.store("alpha").await.unwrap();
            let _ = vl.store("beta").await.unwrap();
            let _ = vl.store("gamma").await.unwrap();

            let results = vl.list(&[], 2).await.unwrap();
            assert_eq!(results.len(), 2, "list should return exactly the requested limit");
        }

        fn list_with_no_matching_perspective_returns_empty(vl) {
            let _ = vl
                .store_with(
                    "A knowledge entry",
                    StoreOptions {
                        perspectives: vec!["knowledge".into()],
                        ..Default::default()
                    },
                )
                .await
                .unwrap();

            let results = vl.list(&["decisions"], 100).await.unwrap();
            assert!(results.is_empty(), "list with non-matching perspective should be empty");
        }

        // ── embed edge cases ──────────────────────────────────────────────────

        fn embed_batch_empty_slice_returns_empty(vl) {
            let results = vl.embed_batch(&[]).await.unwrap();
            assert!(results.is_empty(), "embed_batch of empty slice should return empty vec");
        }

        fn embed_batch_single_text(vl) {
            let results = vl.embed_batch(&["hello world"]).await.unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].len(), 3, "TestEmbedder produces 3-dimensional vectors");
        }

        fn embed_batch_multiple_texts(vl) {
            let results = vl.embed_batch(&["a", "bb", "ccc"]).await.unwrap();
            assert_eq!(results.len(), 3);
            // Each vector has 3 dimensions from TestEmbedder.
            for v in &results {
                assert_eq!(v.len(), 3);
            }
            // Vectors are deterministic: different length texts → different first component.
            assert_ne!(results[0][0], results[1][0]);
        }

        // ── StoreOptions variants ─────────────────────────────────────────────

        fn store_with_visibility_option(vl) {
            let id = vl
                .store_with(
                    "Seasonal memory",
                    StoreOptions {
                        visibility: Some(crate::chunk::visibility::ALWAYS.to_string()),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();

            let entry = vl.get(&id).await.unwrap().unwrap();
            assert_eq!(entry.visibility, crate::chunk::visibility::ALWAYS);
        }

        fn store_with_all_entry_types(vl) {
            use crate::chunk::EntryType;

            for et in [EntryType::Raw, EntryType::Summary, EntryType::Meta, EntryType::Impression] {
                let id = vl
                    .store_with(
                        &format!("entry of type {:?}", et),
                        StoreOptions {
                            entry_type: et,
                            ..Default::default()
                        },
                    )
                    .await
                    .unwrap();
                let entry = vl.get(&id).await.unwrap().unwrap();
                assert_eq!(entry.entry_type, et);
            }
        }

        fn store_with_custom_source(vl) {
            let id = vl
                .store_with(
                    "content",
                    StoreOptions {
                        source: Some("custom-source".into()),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            let entry = vl.get(&id).await.unwrap().unwrap();
            assert_eq!(entry.source_file, "custom-source");
        }

        fn store_default_source_is_api(vl) {
            // When no source is given, StoreOptions::default() uses "[api]".
            let id = vl.store("default source content").await.unwrap();
            let entry = vl.get(&id).await.unwrap().unwrap();
            assert_eq!(entry.source_file, "[api]");
        }

        fn store_with_relation(vl) {
            let id_a = vl.store("entry A").await.unwrap();
            let id_b = vl
                .store_with(
                    "entry B superseded by A",
                    StoreOptions {
                        relations: vec![ChunkRelation::superseded_by(id_a.clone())],
                        ..Default::default()
                    },
                )
                .await
                .unwrap();

            // Both entries exist and B has a relation pointing to A stored on the chunk.
            let entry_b = vl.get(&id_b).await.unwrap().unwrap();
            assert_eq!(entry_b.id, id_b);
            assert!(
                entry_b.relations.iter().any(|r| r.target_id == id_a),
                "relation from B to A should be stored on the chunk"
            );
        }

        fn store_with_parent_id(vl) {
            let parent_id = vl.store("parent entry content").await.unwrap();
            let child_id = vl
                .store_with(
                    "child entry content",
                    StoreOptions {
                        parent_id: Some(parent_id.clone()),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();

            let child = vl.get(&child_id).await.unwrap().unwrap();
            assert_eq!(child.parent_id.as_deref(), Some(parent_id.as_str()));
        }

        fn store_with_multiple_perspectives(vl) {
            let id = vl
                .store_with(
                    "multi-tagged entry",
                    StoreOptions {
                        perspectives: vec!["decisions".into(), "knowledge".into()],
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            let entry = vl.get(&id).await.unwrap().unwrap();
            assert!(entry.perspectives.contains(&"decisions".to_string()));
            assert!(entry.perspectives.contains(&"knowledge".to_string()));
        }

        fn store_with_heading_affects_embedding_but_stored_content(vl) {
            // Heading is prepended for embedding but the stored content field is the raw content.
            let id = vl
                .store_with(
                    "the actual body",
                    StoreOptions {
                        heading: Some("The Heading".into()),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            let entry = vl.get(&id).await.unwrap().unwrap();
            assert_eq!(entry.content, "the actual body");
            assert_eq!(entry.heading.as_deref(), Some("The Heading"));
        }

        // ── accessor methods ──────────────────────────────────────────────────

        fn accessor_data_dir(vl, dir) {
            let reported = vl.data_dir();
            // data_dir() should match the tempdir path we opened with.
            assert_eq!(reported, dir.path());
        }

        fn accessor_store_backend_is_functional(vl) {
            // store_backend() must return a usable Arc; verify by calling stats() on it.
            let stats = vl.store_backend().stats().await.unwrap();
            assert_eq!(stats.total_chunks, 0, "fresh store reports 0 chunks via store_backend()");
        }

        fn accessor_embedder_name(vl) {
            assert_eq!(vl.embedder().name(), "test-embedder");
        }

        fn accessor_embedder_dimension(vl) {
            assert_eq!(vl.embedder().dimension(), 3);
        }

        fn accessor_blob_store_is_accessible(vl) {
            // blob_store() should be usable — verify by counting objects (0 in fresh store).
            let count = vl.blob_store().count().unwrap();
            // No stores performed yet in this test.
            assert_eq!(count, 0, "fresh blob store should have no objects");
        }

        fn accessor_blob_store_counts_stored_blobs(vl) {
            // After storing entries, the blob store should contain exactly that many blobs.
            let _ = vl.store("blob one").await.unwrap();
            let _ = vl.store("blob two").await.unwrap();
            let count = vl.blob_store().count().unwrap();
            assert_eq!(count, 2, "blob store should hold one object per stored entry");
        }

        // ── focus with question (ranked children path) ────────────────────────

        fn focus_with_question_ranks_children(vl) {
            let parent_id = vl
                .store_with(
                    "parent node",
                    StoreOptions {
                        heading: Some("Parent".into()),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();

            // Store two children — they get different embeddings due to different text lengths.
            let _ = vl
                .store_with(
                    "child about rust memory safety",
                    StoreOptions {
                        parent_id: Some(parent_id.clone()),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            let _ = vl
                .store_with(
                    "child about something else",
                    StoreOptions {
                        parent_id: Some(parent_id.clone()),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();

            // focus with a question exercises the cosine-scoring ranked_children branch.
            let result = vl.focus(&parent_id, Some("rust memory"), 10).await.unwrap();
            assert!(result.is_some());
            let focus = result.unwrap();
            assert_eq!(focus.entry.id, parent_id);
            assert_eq!(focus.children.len(), 2);
            // The ranked branch attaches a finite score to every child and
            // returns them sorted by descending relevance. Assert both: each
            // score is finite, and the ordering is non-increasing.
            for (_, score) in &focus.children {
                assert!(score.is_finite(), "every ranked child must have a finite score");
            }
            assert!(
                focus.children[0].1 >= focus.children[1].1,
                "ranked children must be ordered by descending score"
            );
        }

        fn focus_with_limit_truncates_children(vl) {
            let parent_id = vl.store("root").await.unwrap();
            for i in 0..5 {
                let _ = vl
                    .store_with(
                        &format!("child number {}", i),
                        StoreOptions {
                            parent_id: Some(parent_id.clone()),
                            ..Default::default()
                        },
                    )
                    .await
                    .unwrap();
            }

            // focus without a question, limit=3 → children truncated via .take(limit).
            let result = vl.focus(&parent_id, None, 3).await.unwrap();
            assert!(result.is_some());
            let focus = result.unwrap();
            assert!(focus.children.len() <= 3, "children should be limited to 3");
        }

        // ── search_raw ────────────────────────────────────────────────────────

        fn search_raw_on_empty_store_returns_empty(vl) {
            let results = vl.search_raw("query", 5).await.unwrap();
            assert!(results.is_empty(), "search_raw on empty store should return empty");
        }

        fn search_raw_returns_results_after_store(vl) {
            let _ = vl.store("raw search target content").await.unwrap();
            let results = vl.search_raw("raw search target", 5).await.unwrap();
            assert!(!results.is_empty(), "search_raw should find stored content");
        }

        // ── store_search_roundtrip ────────────────────────────────────────────
        fn store_search_roundtrip(vl) {
            let id = vl.store("Test content for roundtrip").await.unwrap();
            assert!(!id.is_empty());

            let stats = vl.stats().await.unwrap();
            assert_eq!(stats.total_chunks, 1);

            let entry = vl.get(&id[..7]).await.unwrap();
            assert!(entry.is_some());
            assert_eq!(entry.unwrap().content, "Test content for roundtrip");

            let results = vl.search("roundtrip test", 5).await.unwrap();
            assert!(!results.is_empty());
            assert_eq!(results[0].chunk.id, id);
        }

        fn store_with_options(vl) {
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

        fn store_and_delete_by_source(vl) {
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

        fn focus_entry(vl) {
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

        fn list_with_perspective(vl) {
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

            let decisions = vl.list(&["decisions"], 100).await.unwrap();
            assert_eq!(decisions.len(), 1);
            assert_eq!(decisions[0].content, "A decision");

            let all = vl.list(&[], 100).await.unwrap();
            assert_eq!(all.len(), 2);
        }

        fn embed_helpers(vl) {
            let single = vl.embed_one("hello").await.unwrap();
            assert_eq!(single.len(), 3);

            let batch = vl.embed_batch(&["hello", "world"]).await.unwrap();
            assert_eq!(batch.len(), 2);
        }

        fn read_only_allows_search(vl, dir) {
            let _ = vl.store("seed").await.unwrap();
            drop(vl);

            let vl = open_ro(dir.path()).await.unwrap();
            let results = vl.search("seed", 5).await.unwrap();
            assert!(!results.is_empty());
        }
    }
}
