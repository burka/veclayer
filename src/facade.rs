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
        Ok(Self {
            data_dir: data_dir.to_path_buf(),
            store,
            embedder: Arc::new(embedder),
            blob_store,
            #[cfg(feature = "llm")]
            llm: std::sync::RwLock::new(None),
        })
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
        Ok(Self {
            data_dir: data_dir.to_path_buf(),
            store,
            embedder: Arc::new(embedder),
            blob_store,
            #[cfg(feature = "llm")]
            llm: std::sync::RwLock::new(None),
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
        let embedding = self.embed_one(query)?;
        let results = self.store.search(&embedding, limit, None, &[]).await?;

        // Track access for returned entries
        let ids: Vec<_> = results.iter().map(|r| {
            let mut ap = r.chunk.access_profile.clone();
            ap.record_access();
            (r.chunk.id.clone(), ap)
        }).collect();
        let _ = self.store.update_access_profiles(ids).await;

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
            let query_embedding = self.embed_one(q)?;
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
        *self.llm.write().unwrap() = Some(Arc::new(crate::llm::DynLlmProviderWrapper(llm)));
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
            let guard = self.llm.read().unwrap();
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
            let single = vl.embed_one("hello").unwrap();
            assert_eq!(single.len(), 3);

            let batch = vl.embed_batch(&["hello", "world"]).unwrap();
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
