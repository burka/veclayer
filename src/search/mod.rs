//! Hierarchical search engine with blended salience scoring.
//!
//! [`HierarchicalSearch`] traverses the document hierarchy in two phases:
//! first locating top-level matches via vector similarity, then enriching each
//! result with parent context and matching children. [`SearchConfig`] controls
//! blending of vector score with recency and salience signals.

use crate::chunk::now_epoch_secs;
use crate::embedder::Embedder;
use crate::salience::{self, SalienceWeights};
use crate::store::{SearchResult, VectorStore};
use crate::{ChunkLevel, HierarchicalChunk, RecencyWindow, Result};

/// Over-fetch factor when temporal filters will reduce the result set.
pub const TEMPORAL_PREFETCH_FACTOR: usize = 3;

/// Result of a hierarchical search including the path through the hierarchy
#[derive(Debug, Clone)]
pub struct HierarchicalSearchResult {
    /// The matched chunk
    pub chunk: HierarchicalChunk,
    /// Similarity score
    pub score: f32,
    /// Path through the hierarchy (from root to this chunk)
    pub hierarchy_path: Vec<HierarchicalChunk>,
    /// Children of this chunk that also match
    pub relevant_children: Vec<SearchResult>,
}

/// Default blending factor when no recency window is active.
pub const DEFAULT_RECENCY_ALPHA: f32 = 0.15;
/// Blending factor when a recency window is explicitly requested.
pub const ACTIVE_RECENCY_ALPHA: f32 = 0.3;
/// Default salience weight within the relevancy portion.
/// 0.0 = pure recency, 1.0 = pure salience.
pub const DEFAULT_SALIENCE_WEIGHT: f32 = 0.3;

/// Configuration for hierarchical search.
///
/// # Examples
///
/// ```
/// use veclayer::search::SearchConfig;
///
/// // Default: top-5 results, shallow search, no recency boost.
/// let config = SearchConfig::default();
/// assert_eq!(config.top_k, 5);
/// assert!(!config.deep);
///
/// // Convenience builder for a query with a recency window.
/// let config = SearchConfig::for_query(10, false, Some("7d"));
/// assert_eq!(config.top_k, 10);
/// assert!(config.recency_window.is_some());
/// ```
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Number of top-level chunks to retrieve
    pub top_k: usize,
    /// Number of children to retrieve per parent
    pub children_k: usize,
    /// Maximum depth to search
    pub max_depth: usize,
    /// Minimum score threshold for search relevance (applied to pre-blend vector score, not blended score)
    pub min_score: f32,
    /// Deep search: include all visibilities (DeepOnly, expired, custom)
    pub deep: bool,
    /// Recency window for relevancy scoring. None = balanced weighting.
    pub recency_window: Option<RecencyWindow>,
    /// Blending factor: 0.0 = pure vector similarity, 1.0 = pure relevancy.
    /// Default: 0.15 (or 0.3 when recency window is active).
    pub recency_alpha: f32,
    /// Optional perspective filter. Only return entries in this perspective.
    pub perspective: Option<String>,
    /// Weight for salience within the relevancy signal [0.0, 1.0].
    /// 0.0 = ignore salience (pure recency), 1.0 = ignore recency (pure salience).
    /// Default: 0.3
    pub salience_weight: f32,
    /// Minimum salience threshold (entries below this are excluded from blending)
    pub min_salience: Option<f32>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            children_k: 3,
            max_depth: 3,
            min_score: 0.0,
            deep: false,
            recency_window: None,
            recency_alpha: DEFAULT_RECENCY_ALPHA,
            perspective: None,
            salience_weight: DEFAULT_SALIENCE_WEIGHT,
            min_salience: None,
        }
    }
}

impl SearchConfig {
    /// Derive the appropriate recency_alpha from the recency window.
    pub fn alpha_for_window(window: Option<RecencyWindow>) -> f32 {
        if window.is_some() {
            ACTIVE_RECENCY_ALPHA
        } else {
            DEFAULT_RECENCY_ALPHA
        }
    }

    /// Build a SearchConfig from common query parameters.
    /// Derives recency_window and recency_alpha from the raw recency string.
    pub fn for_query(top_k: usize, deep: bool, recency: Option<&str>) -> Self {
        let recency_window = recency.and_then(RecencyWindow::from_str_opt);
        Self {
            top_k,
            deep,
            recency_window,
            recency_alpha: Self::alpha_for_window(recency_window),
            ..Default::default()
        }
    }

    /// Add a perspective filter to this config.
    pub fn with_perspective(mut self, perspective: Option<String>) -> Self {
        self.perspective = perspective;
        self
    }

    /// Add a salience threshold filter to this config.
    pub fn with_min_salience(mut self, min_salience: Option<f32>) -> Self {
        self.min_salience = min_salience;
        self
    }

    /// Add a minimum score threshold to this config.
    pub fn with_min_score(mut self, min_score: Option<f32>) -> Self {
        self.min_score = min_score.unwrap_or(0.0);
        self
    }

    /// Blend vector similarity with recency and salience signals.
    ///
    /// Formula: `vector * (1 - alpha) + relevancy_signal * alpha`
    /// where `relevancy_signal = recency * (1 - sw) + salience * sw`
    ///
    /// Returns pure vector score when alpha is 0.
    /// Entries below min_salience are excluded from salience boosting but not filtered.
    pub fn blend_score(&self, vector_score: f32, chunk: &HierarchicalChunk) -> f32 {
        if self.recency_alpha > 0.0 {
            let recency = chunk.access_profile.relevancy_score(self.recency_window);

            let relevancy_signal = if self.salience_weight > 0.0 {
                let salience = salience::compute(chunk, &SalienceWeights::default());
                if let Some(threshold) = self.min_salience {
                    if salience.composite >= threshold {
                        recency * (1.0 - self.salience_weight)
                            + salience.composite * self.salience_weight
                    } else {
                        recency
                    }
                } else {
                    recency * (1.0 - self.salience_weight)
                        + salience.composite * self.salience_weight
                }
            } else {
                recency
            };

            vector_score * (1.0 - self.recency_alpha) + relevancy_signal * self.recency_alpha
        } else {
            vector_score
        }
    }
}

/// Hierarchical search engine that traverses the document structure
pub struct HierarchicalSearch<S: VectorStore, E: Embedder> {
    store: S,
    embedder: E,
    config: SearchConfig,
}

impl<S: VectorStore, E: Embedder> HierarchicalSearch<S, E> {
    pub fn new(store: S, embedder: E) -> Self {
        Self {
            store,
            embedder,
            config: SearchConfig::default(),
        }
    }

    pub fn with_config(mut self, config: SearchConfig) -> Self {
        self.config = config;
        self
    }

    /// Embed a query string
    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let embeddings = self.embedder.embed(&[query])?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| crate::Error::embedding("No embedding returned"))
    }

    /// Search for top-level matches (H1/H2)
    pub async fn search_top_level(&self, query: &str) -> Result<Vec<SearchResult>> {
        let query_embedding = self.embed_query(query)?;

        // Search H1 chunks first
        let mut results = self
            .store
            .search(&query_embedding, self.config.top_k, Some(ChunkLevel::H1))
            .await?;

        // If not enough H1 results, also search H2
        if results.len() < self.config.top_k {
            let h2_results = self
                .store
                .search(
                    &query_embedding,
                    self.config.top_k - results.len(),
                    Some(ChunkLevel::H2),
                )
                .await?;
            results.extend(h2_results);
        }

        Ok(results)
    }

    /// Perform a full hierarchical search
    /// 1. Find top-level matches
    /// 2. Filter by visibility (unless deep mode)
    /// 3. For each match, search its children
    /// 4. Record access and compute combined scores (vector + relevancy)
    pub async fn search(&self, query: &str) -> Result<Vec<HierarchicalSearchResult>> {
        let query_embedding = self.embed_query(query)?;
        let now = now_epoch_secs();

        // Fetch more than top_k so we still have enough after visibility filtering
        let fetch_k = if self.config.deep {
            self.config.top_k
        } else {
            self.config.top_k * 2
        };

        // Step 1: Find top-level matches (optionally filtered by perspective)
        let top_results = if let Some(ref perspective) = self.config.perspective {
            self.store
                .search_by_perspective(&query_embedding, fetch_k, perspective)
                .await?
        } else {
            self.store.search(&query_embedding, fetch_k, None).await?
        };

        let mut hierarchical_results = Vec::new();
        let mut access_updates = Vec::new();

        for result in top_results {
            if result.score < self.config.min_score {
                continue;
            }

            // Step 2: Visibility filter (skip unless deep mode)
            if !self.config.deep && !result.chunk.is_visible_standard() {
                continue;
            }

            // Stop once we have enough results
            if hierarchical_results.len() >= self.config.top_k {
                break;
            }

            // Build the hierarchy path (from root to this chunk)
            let hierarchy_path = self.build_hierarchy_path(&result.chunk).await?;

            // Get relevant children
            let relevant_children = self
                .search_children(&query_embedding, &result.chunk)
                .await?;

            // Record access and compute combined score
            let mut chunk = result.chunk;
            let vector_score = result.score;
            chunk.access_profile.record_access_at(now);
            let final_score = self.config.blend_score(vector_score, &chunk);

            access_updates.push((chunk.id.clone(), chunk.access_profile.clone()));

            hierarchical_results.push(HierarchicalSearchResult {
                chunk,
                score: final_score,
                hierarchy_path,
                relevant_children,
            });
        }

        // Persist access tracking (best effort — don't fail search)
        if !access_updates.is_empty() {
            let _ = self.store.update_access_profiles(access_updates).await;
        }

        Ok(hierarchical_results)
    }

    /// Search for entries similar to a given entry ID.
    /// Uses the entry's embedding as the query vector instead of text.
    pub async fn search_by_embedding(
        &self,
        entry_id: &str,
        limit: usize,
    ) -> Result<Vec<HierarchicalSearchResult>> {
        let target = self
            .store
            .get_by_id_prefix(entry_id)
            .await?
            .ok_or_else(|| {
                crate::Error::not_found(format!(
                    "Entry '{}' not found for similarity search",
                    entry_id
                ))
            })?;

        let target_full_id = target.id.clone();

        let query_embedding = target.embedding.ok_or_else(|| {
            crate::Error::search(format!(
                "Entry '{}' exists but has no embedding for similarity search",
                entry_id
            ))
        })?;

        let now = now_epoch_secs();

        // +1 to account for the source entry itself appearing in ANN results (excluded below)
        let fetch_k = limit + 1;

        let top_results = if let Some(ref perspective) = self.config.perspective {
            self.store
                .search_by_perspective(&query_embedding, fetch_k, perspective)
                .await?
        } else {
            self.store.search(&query_embedding, fetch_k, None).await?
        };

        let mut hierarchical_results = Vec::new();
        let mut access_updates = Vec::new();

        for result in top_results {
            if result.chunk.id == target_full_id {
                continue;
            }

            if result.score < self.config.min_score {
                continue;
            }

            if !self.config.deep && !result.chunk.is_visible_standard() {
                continue;
            }

            if hierarchical_results.len() >= limit {
                break;
            }

            let hierarchy_path = self.build_hierarchy_path(&result.chunk).await?;
            let relevant_children = self
                .search_children(&query_embedding, &result.chunk)
                .await?;

            let mut chunk = result.chunk;
            let vector_score = result.score;
            chunk.access_profile.record_access_at(now);
            let final_score = self.config.blend_score(vector_score, &chunk);

            access_updates.push((chunk.id.clone(), chunk.access_profile.clone()));

            hierarchical_results.push(HierarchicalSearchResult {
                chunk,
                score: final_score,
                hierarchy_path,
                relevant_children,
            });
        }

        if !access_updates.is_empty() {
            let _ = self.store.update_access_profiles(access_updates).await;
        }

        Ok(hierarchical_results)
    }

    /// Search within a specific subtree (starting from a parent chunk)
    pub async fn search_subtree(
        &self,
        query: &str,
        parent_id: &str,
    ) -> Result<Vec<HierarchicalSearchResult>> {
        let query_embedding = self.embed_query(query)?;
        let now = now_epoch_secs();

        // Get all children of this parent
        let children = self.store.get_children(parent_id).await?;

        if children.is_empty() {
            return Ok(vec![]);
        }

        // Score each child against the query
        let mut scored_children: Vec<SearchResult> = Vec::new();
        for child in children {
            if let Some(ref embedding) = child.embedding {
                let score = cosine_similarity(&query_embedding, embedding);
                scored_children.push(SearchResult {
                    chunk: child,
                    score,
                });
            }
        }

        // Sort by score descending
        sort_by_score_desc(&mut scored_children);

        // Take top results, applying visibility filter
        let top_children: Vec<_> = scored_children
            .into_iter()
            .filter(|r| r.score >= self.config.min_score)
            .filter(|r| self.config.deep || r.chunk.is_visible_standard())
            .take(self.config.top_k)
            .collect();

        let mut results = Vec::new();
        let mut access_updates = Vec::new();

        for result in top_children {
            let hierarchy_path = self.build_hierarchy_path(&result.chunk).await?;
            let relevant_children = self
                .search_children(&query_embedding, &result.chunk)
                .await?;

            // Record access and compute combined score
            let mut chunk = result.chunk;
            let vector_score = result.score;
            chunk.access_profile.record_access_at(now);
            let final_score = self.config.blend_score(vector_score, &chunk);

            access_updates.push((chunk.id.clone(), chunk.access_profile.clone()));

            results.push(HierarchicalSearchResult {
                chunk,
                score: final_score,
                hierarchy_path,
                relevant_children,
            });
        }

        // Persist access tracking (best effort)
        if !access_updates.is_empty() {
            let _ = self.store.update_access_profiles(access_updates).await;
        }

        Ok(results)
    }

    /// Build the path from root to the given chunk
    async fn build_hierarchy_path(
        &self,
        chunk: &HierarchicalChunk,
    ) -> Result<Vec<HierarchicalChunk>> {
        let mut path = Vec::new();
        let mut current_id = chunk.parent_id.clone();

        while let Some(parent_id) = current_id {
            if let Some(parent) = self.store.get_by_id(&parent_id).await? {
                current_id = parent.parent_id.clone();
                path.push(parent);
            } else {
                break;
            }
        }

        path.reverse();
        Ok(path)
    }

    /// Search children of a chunk and return the most relevant ones
    async fn search_children(
        &self,
        query_embedding: &[f32],
        parent: &HierarchicalChunk,
    ) -> Result<Vec<SearchResult>> {
        let children = self.store.get_children(&parent.id).await?;

        if children.is_empty() {
            return Ok(vec![]);
        }

        // Score each child
        let mut scored: Vec<SearchResult> = children
            .into_iter()
            .filter_map(|child| {
                child.embedding.as_ref().map(|emb| {
                    let score = cosine_similarity(query_embedding, emb);
                    SearchResult {
                        chunk: child.clone(),
                        score,
                    }
                })
            })
            .collect();

        // Sort by score descending
        sort_by_score_desc(&mut scored);

        // Take top children_k
        Ok(scored.into_iter().take(self.config.children_k).collect())
    }

    /// Get store reference
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Get embedder reference
    pub fn embedder(&self) -> &E {
        &self.embedder
    }
}

/// Sort search results by score in descending order
fn sort_by_score_desc(results: &mut [SearchResult]) {
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// Mock vector store for testing
    struct MockStore {
        chunks: Mutex<HashMap<String, HierarchicalChunk>>,
        search_results: Mutex<Vec<SearchResult>>,
    }

    impl MockStore {
        fn new() -> Self {
            Self {
                chunks: Mutex::new(HashMap::new()),
                search_results: Mutex::new(Vec::new()),
            }
        }

        fn add_chunk(&self, chunk: HierarchicalChunk) {
            self.chunks.lock().unwrap().insert(chunk.id.clone(), chunk);
        }

        fn set_search_results(&self, results: Vec<SearchResult>) {
            *self.search_results.lock().unwrap() = results;
        }
    }

    impl VectorStore for MockStore {
        async fn insert_chunks(&self, _chunks: Vec<HierarchicalChunk>) -> Result<()> {
            Ok(())
        }

        async fn search(
            &self,
            _query_embedding: &[f32],
            limit: usize,
            level_filter: Option<ChunkLevel>,
        ) -> Result<Vec<SearchResult>> {
            let results = self.search_results.lock().unwrap();
            let filtered: Vec<_> = if let Some(level) = level_filter {
                results
                    .iter()
                    .filter(|r| r.chunk.level == level)
                    .take(limit)
                    .cloned()
                    .collect()
            } else {
                results.iter().take(limit).cloned().collect()
            };
            Ok(filtered)
        }

        async fn get_children(&self, parent_id: &str) -> Result<Vec<HierarchicalChunk>> {
            let chunks = self.chunks.lock().unwrap();
            let children: Vec<_> = chunks
                .values()
                .filter(|c| c.parent_id.as_deref() == Some(parent_id))
                .cloned()
                .collect();
            Ok(children)
        }

        async fn get_by_id(&self, id: &str) -> Result<Option<HierarchicalChunk>> {
            let chunks = self.chunks.lock().unwrap();
            Ok(chunks.get(id).cloned())
        }

        async fn get_by_id_prefix(&self, prefix: &str) -> Result<Option<HierarchicalChunk>> {
            let chunks = self.chunks.lock().unwrap();
            if let Some(chunk) = chunks.get(prefix) {
                return Ok(Some(chunk.clone()));
            }
            let matches: Vec<_> = chunks
                .iter()
                .filter(|(k, _)| k.starts_with(prefix))
                .collect();
            match matches.len() {
                0 => Ok(None),
                1 => Ok(Some(matches[0].1.clone())),
                _ => Err(crate::Error::config(format!(
                    "Ambiguous prefix '{}': {} matches",
                    prefix,
                    matches.len()
                ))),
            }
        }

        async fn get_by_source(&self, _source_file: &str) -> Result<Vec<HierarchicalChunk>> {
            Ok(vec![])
        }

        async fn delete_by_source(&self, _source_file: &str) -> Result<usize> {
            Ok(0)
        }

        async fn stats(&self) -> Result<crate::store::StoreStats> {
            Ok(crate::store::StoreStats::default())
        }

        async fn update_access_profiles(
            &self,
            _updates: Vec<(String, crate::AccessProfile)>,
        ) -> Result<()> {
            Ok(())
        }

        async fn update_visibility(&self, _chunk_id: &str, _visibility: &str) -> Result<()> {
            Ok(())
        }

        async fn add_relation(
            &self,
            _chunk_id: &str,
            _relation: crate::ChunkRelation,
        ) -> Result<()> {
            Ok(())
        }

        async fn search_by_perspective(
            &self,
            _query_embedding: &[f32],
            limit: usize,
            perspective: &str,
        ) -> Result<Vec<SearchResult>> {
            let results = self.search_results.lock().unwrap();
            let filtered: Vec<_> = results
                .iter()
                .filter(|r| r.chunk.perspectives.iter().any(|p| p == perspective))
                .take(limit)
                .cloned()
                .collect();
            Ok(filtered)
        }

        async fn get_hot_chunks(&self, _limit: usize) -> Result<Vec<HierarchicalChunk>> {
            Ok(vec![])
        }

        async fn get_stale_chunks(
            &self,
            _stale_seconds: i64,
            _limit: usize,
        ) -> Result<Vec<HierarchicalChunk>> {
            Ok(vec![])
        }

        async fn list_entries(
            &self,
            _perspective: Option<&str>,
            _since: Option<i64>,
            _until: Option<i64>,
            _limit: usize,
        ) -> Result<Vec<HierarchicalChunk>> {
            Ok(vec![])
        }

        async fn get_pending_embeddings(
            &self,
            _limit: usize,
        ) -> Result<Vec<HierarchicalChunk>> {
            Ok(vec![])
        }

        async fn batch_update_embeddings(
            &self,
            _updates: Vec<(String, Vec<f32>)>,
        ) -> Result<()> {
            Ok(())
        }

        async fn count_pending_embeddings(&self) -> Result<usize> {
            Ok(0)
        }
    }

    /// Mock embedder for testing
    struct MockEmbedder {
        dimension: usize,
    }

    impl MockEmbedder {
        fn new(dimension: usize) -> Self {
            Self { dimension }
        }
    }

    impl Embedder for MockEmbedder {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0; self.dimension]).collect())
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_zero_vectors() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_negative_values() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_sort_by_score_desc() {
        let chunk1 = HierarchicalChunk::new(
            "test1".to_string(),
            ChunkLevel::H1,
            None,
            "path1".to_string(),
            "test.md".to_string(),
        );
        let chunk2 = HierarchicalChunk::new(
            "test2".to_string(),
            ChunkLevel::H1,
            None,
            "path2".to_string(),
            "test.md".to_string(),
        );
        let chunk3 = HierarchicalChunk::new(
            "test3".to_string(),
            ChunkLevel::H1,
            None,
            "path3".to_string(),
            "test.md".to_string(),
        );

        let mut results = vec![
            SearchResult {
                chunk: chunk1,
                score: 0.5,
            },
            SearchResult {
                chunk: chunk2,
                score: 0.9,
            },
            SearchResult {
                chunk: chunk3,
                score: 0.3,
            },
        ];

        sort_by_score_desc(&mut results);

        assert_eq!(results[0].score, 0.9);
        assert_eq!(results[1].score, 0.5);
        assert_eq!(results[2].score, 0.3);
    }

    #[test]
    fn test_sort_by_score_desc_with_nan() {
        let chunk1 = HierarchicalChunk::new(
            "test1".to_string(),
            ChunkLevel::H1,
            None,
            "path1".to_string(),
            "test.md".to_string(),
        );
        let chunk2 = HierarchicalChunk::new(
            "test2".to_string(),
            ChunkLevel::H1,
            None,
            "path2".to_string(),
            "test.md".to_string(),
        );

        let mut results = vec![
            SearchResult {
                chunk: chunk1,
                score: f32::NAN,
            },
            SearchResult {
                chunk: chunk2,
                score: 0.5,
            },
        ];

        sort_by_score_desc(&mut results);
        // NaN should be handled without panic
        assert!(results.len() == 2);
    }

    #[test]
    fn test_search_config_default() {
        let config = SearchConfig::default();
        assert_eq!(config.top_k, 5);
        assert_eq!(config.children_k, 3);
        assert_eq!(config.max_depth, 3);
        assert_eq!(config.min_score, 0.0);
    }

    #[tokio::test]
    async fn test_hierarchical_search_new() {
        let store = MockStore::new();
        let embedder = MockEmbedder::new(3);
        let search = HierarchicalSearch::new(store, embedder);

        assert_eq!(search.config.top_k, 5);
        assert_eq!(search.embedder().dimension(), 3);
    }

    #[tokio::test]
    async fn test_hierarchical_search_with_config() {
        let store = MockStore::new();
        let embedder = MockEmbedder::new(3);
        let config = SearchConfig {
            top_k: 10,
            children_k: 5,
            max_depth: 2,
            min_score: 0.5,
            deep: false,
            ..Default::default()
        };

        let search = HierarchicalSearch::new(store, embedder).with_config(config);

        assert_eq!(search.config.top_k, 10);
        assert_eq!(search.config.children_k, 5);
        assert_eq!(search.config.max_depth, 2);
        assert_eq!(search.config.min_score, 0.5);
    }

    #[tokio::test]
    async fn test_embed_query() {
        let store = MockStore::new();
        let embedder = MockEmbedder::new(3);
        let search = HierarchicalSearch::new(store, embedder);

        let embedding = search.embed_query("test query").unwrap();
        assert_eq!(embedding.len(), 3);
        assert_eq!(embedding, vec![1.0, 1.0, 1.0]);
    }

    #[tokio::test]
    async fn test_build_hierarchy_path_no_parent() {
        let store = MockStore::new();
        let embedder = MockEmbedder::new(3);
        let search = HierarchicalSearch::new(store, embedder);

        let chunk = HierarchicalChunk::new(
            "test".to_string(),
            ChunkLevel::H1,
            None,
            "path".to_string(),
            "test.md".to_string(),
        );

        let path = search.build_hierarchy_path(&chunk).await.unwrap();
        assert_eq!(path.len(), 0);
    }

    #[tokio::test]
    async fn test_build_hierarchy_path_with_parents() {
        let store = MockStore::new();

        let parent1 = HierarchicalChunk::new(
            "parent1".to_string(),
            ChunkLevel::H1,
            None,
            "parent1".to_string(),
            "test.md".to_string(),
        );
        let parent1_id = parent1.id.clone();

        let parent2 = HierarchicalChunk::new(
            "parent2".to_string(),
            ChunkLevel::H2,
            Some(parent1_id.clone()),
            "parent1 > parent2".to_string(),
            "test.md".to_string(),
        );
        let parent2_id = parent2.id.clone();

        let child = HierarchicalChunk::new(
            "child".to_string(),
            ChunkLevel::H3,
            Some(parent2_id.clone()),
            "parent1 > parent2 > child".to_string(),
            "test.md".to_string(),
        );

        store.add_chunk(parent1.clone());
        store.add_chunk(parent2.clone());

        let embedder = MockEmbedder::new(3);
        let search = HierarchicalSearch::new(store, embedder);

        let path = search.build_hierarchy_path(&child).await.unwrap();
        assert_eq!(path.len(), 2);
        assert_eq!(path[0].id, parent1_id);
        assert_eq!(path[1].id, parent2_id);
    }

    #[tokio::test]
    async fn test_search_children_no_children() {
        let store = MockStore::new();
        let embedder = MockEmbedder::new(3);
        let search = HierarchicalSearch::new(store, embedder);

        let parent = HierarchicalChunk::new(
            "parent".to_string(),
            ChunkLevel::H1,
            None,
            "parent".to_string(),
            "test.md".to_string(),
        );

        let query_embedding = vec![1.0, 0.0, 0.0];
        let children = search
            .search_children(&query_embedding, &parent)
            .await
            .unwrap();

        assert_eq!(children.len(), 0);
    }

    #[tokio::test]
    async fn test_search_children_with_embeddings() {
        let store = MockStore::new();

        let parent = HierarchicalChunk::new(
            "parent".to_string(),
            ChunkLevel::H1,
            None,
            "parent".to_string(),
            "test.md".to_string(),
        );
        let parent_id = parent.id.clone();

        let child1 = HierarchicalChunk::new(
            "child1".to_string(),
            ChunkLevel::H2,
            Some(parent_id.clone()),
            "parent > child1".to_string(),
            "test.md".to_string(),
        )
        .with_embedding(vec![1.0, 0.0, 0.0]);

        let child2 = HierarchicalChunk::new(
            "child2".to_string(),
            ChunkLevel::H2,
            Some(parent_id.clone()),
            "parent > child2".to_string(),
            "test.md".to_string(),
        )
        .with_embedding(vec![0.5, 0.5, 0.0]);

        let child3 = HierarchicalChunk::new(
            "child3".to_string(),
            ChunkLevel::H2,
            Some(parent_id.clone()),
            "parent > child3".to_string(),
            "test.md".to_string(),
        )
        .with_embedding(vec![0.0, 1.0, 0.0]);

        store.add_chunk(parent.clone());
        store.add_chunk(child1);
        store.add_chunk(child2);
        store.add_chunk(child3);

        let embedder = MockEmbedder::new(3);
        let config = SearchConfig {
            children_k: 2,
            ..Default::default()
        };
        let search = HierarchicalSearch::new(store, embedder).with_config(config);

        let query_embedding = vec![1.0, 0.0, 0.0];
        let children = search
            .search_children(&query_embedding, &parent)
            .await
            .unwrap();

        // Should return top 2 children sorted by score
        assert_eq!(children.len(), 2);
        assert!(children[0].score >= children[1].score);
    }

    #[tokio::test]
    async fn test_search_children_without_embeddings() {
        let store = MockStore::new();

        let parent = HierarchicalChunk::new(
            "parent".to_string(),
            ChunkLevel::H1,
            None,
            "parent".to_string(),
            "test.md".to_string(),
        );
        let parent_id = parent.id.clone();

        // Child without embedding
        let child = HierarchicalChunk::new(
            "child".to_string(),
            ChunkLevel::H2,
            Some(parent_id.clone()),
            "parent > child".to_string(),
            "test.md".to_string(),
        );

        store.add_chunk(parent.clone());
        store.add_chunk(child);

        let embedder = MockEmbedder::new(3);
        let search = HierarchicalSearch::new(store, embedder);

        let query_embedding = vec![1.0, 0.0, 0.0];
        let children = search
            .search_children(&query_embedding, &parent)
            .await
            .unwrap();

        // Should not include children without embeddings
        assert_eq!(children.len(), 0);
    }

    #[tokio::test]
    async fn test_search_top_level_h1_only() {
        let store = MockStore::new();

        let h1_chunk = HierarchicalChunk::new(
            "h1 content".to_string(),
            ChunkLevel::H1,
            None,
            "h1".to_string(),
            "test.md".to_string(),
        )
        .with_embedding(vec![1.0, 0.0, 0.0]);

        let results = vec![SearchResult {
            chunk: h1_chunk,
            score: 0.95,
        }];

        store.set_search_results(results.clone());

        let embedder = MockEmbedder::new(3);
        let search = HierarchicalSearch::new(store, embedder);

        let top_results = search.search_top_level("test query").await.unwrap();

        assert_eq!(top_results.len(), 1);
        assert_eq!(top_results[0].chunk.level, ChunkLevel::H1);
    }

    #[tokio::test]
    async fn test_search_filters_by_min_score() {
        let store = MockStore::new();

        let chunk1 = HierarchicalChunk::new(
            "high score".to_string(),
            ChunkLevel::H1,
            None,
            "path1".to_string(),
            "test.md".to_string(),
        )
        .with_embedding(vec![1.0, 0.0, 0.0]);

        let chunk2 = HierarchicalChunk::new(
            "low score".to_string(),
            ChunkLevel::H1,
            None,
            "path2".to_string(),
            "test.md".to_string(),
        )
        .with_embedding(vec![0.0, 1.0, 0.0]);

        let results = vec![
            SearchResult {
                chunk: chunk1,
                score: 0.8,
            },
            SearchResult {
                chunk: chunk2,
                score: 0.3,
            },
        ];

        store.set_search_results(results);

        let embedder = MockEmbedder::new(3);
        let config = SearchConfig {
            min_score: 0.5,
            ..Default::default()
        };
        let search = HierarchicalSearch::new(store, embedder).with_config(config);

        let search_results = search.search("test query").await.unwrap();

        // Only chunk1 should pass the min_score filter
        // (min_score applies to the raw vector score, before relevancy blending)
        assert_eq!(search_results.len(), 1);
        // Score is blended: vector_score * (1 - alpha) + relevancy * alpha
        assert!(search_results[0].score > 0.5);
    }

    #[tokio::test]
    async fn test_search_subtree_empty() {
        let store = MockStore::new();
        let embedder = MockEmbedder::new(3);
        let search = HierarchicalSearch::new(store, embedder);

        let results = search
            .search_subtree("test query", "nonexistent")
            .await
            .unwrap();

        assert_eq!(results.len(), 0);
    }

    #[tokio::test]
    async fn test_search_subtree_with_children() {
        let store = MockStore::new();

        let parent = HierarchicalChunk::new(
            "parent".to_string(),
            ChunkLevel::H1,
            None,
            "parent".to_string(),
            "test.md".to_string(),
        );
        let parent_id = parent.id.clone();

        let child1 = HierarchicalChunk::new(
            "child1".to_string(),
            ChunkLevel::H2,
            Some(parent_id.clone()),
            "parent > child1".to_string(),
            "test.md".to_string(),
        )
        .with_embedding(vec![1.0, 0.0, 0.0]);

        let child2 = HierarchicalChunk::new(
            "child2".to_string(),
            ChunkLevel::H2,
            Some(parent_id.clone()),
            "parent > child2".to_string(),
            "test.md".to_string(),
        )
        .with_embedding(vec![0.0, 1.0, 0.0]);

        store.add_chunk(parent.clone());
        store.add_chunk(child1);
        store.add_chunk(child2);

        let embedder = MockEmbedder::new(3);
        let config = SearchConfig {
            top_k: 10,
            min_score: 0.0,
            ..Default::default()
        };
        let search = HierarchicalSearch::new(store, embedder).with_config(config);

        let results = search
            .search_subtree("test query", &parent_id)
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].score >= results[1].score);
    }

    #[tokio::test]
    async fn test_search_subtree_respects_min_score() {
        let store = MockStore::new();

        let parent = HierarchicalChunk::new(
            "parent".to_string(),
            ChunkLevel::H1,
            None,
            "parent".to_string(),
            "test.md".to_string(),
        );
        let parent_id = parent.id.clone();

        let child = HierarchicalChunk::new(
            "child".to_string(),
            ChunkLevel::H2,
            Some(parent_id.clone()),
            "parent > child".to_string(),
            "test.md".to_string(),
        )
        .with_embedding(vec![0.0, 1.0, 0.0]);

        store.add_chunk(parent.clone());
        store.add_chunk(child);

        let embedder = MockEmbedder::new(3);
        let config = SearchConfig {
            min_score: 0.9, // Very high threshold
            ..Default::default()
        };
        let search = HierarchicalSearch::new(store, embedder).with_config(config);

        let results = search
            .search_subtree("test query", &parent_id)
            .await
            .unwrap();

        // Child should be filtered out by high min_score
        assert_eq!(results.len(), 0);
    }

    // --- SearchConfig tests ---

    #[test]
    fn test_for_query_defaults() {
        let config = SearchConfig::for_query(10, true, None);
        assert_eq!(config.top_k, 10);
        assert!(config.deep);
        assert!(config.recency_window.is_none());
        assert_eq!(config.recency_alpha, DEFAULT_RECENCY_ALPHA);
        assert!(config.perspective.is_none());
        assert_eq!(config.salience_weight, DEFAULT_SALIENCE_WEIGHT);
        // Inherited defaults
        assert_eq!(config.children_k, 3);
        assert_eq!(config.max_depth, 3);
    }

    #[test]
    fn test_for_query_with_recency() {
        let config = SearchConfig::for_query(5, false, Some("24h"));
        assert_eq!(config.recency_window, Some(RecencyWindow::Day));
        assert_eq!(config.recency_alpha, ACTIVE_RECENCY_ALPHA);
    }

    #[test]
    fn test_for_query_with_invalid_recency() {
        let config = SearchConfig::for_query(5, false, Some("invalid"));
        assert!(config.recency_window.is_none());
        assert_eq!(config.recency_alpha, DEFAULT_RECENCY_ALPHA);
    }

    #[test]
    fn test_with_perspective() {
        let config =
            SearchConfig::for_query(5, false, None).with_perspective(Some("decisions".to_string()));
        assert_eq!(config.perspective.as_deref(), Some("decisions"));
    }

    #[test]
    fn test_with_perspective_none() {
        let config = SearchConfig::for_query(5, false, None).with_perspective(None);
        assert!(config.perspective.is_none());
    }

    /// Helper: create a minimal chunk for blend_score tests.
    fn blend_test_chunk() -> HierarchicalChunk {
        HierarchicalChunk::new(
            "blend test".to_string(),
            ChunkLevel::CONTENT,
            None,
            String::new(),
            "test.md".to_string(),
        )
    }

    #[test]
    fn test_blend_score_zero_alpha() {
        let config = SearchConfig {
            recency_alpha: 0.0,
            ..Default::default()
        };
        let chunk = blend_test_chunk();
        assert_eq!(config.blend_score(0.8, &chunk), 0.8);
    }

    #[test]
    fn test_blend_score_with_alpha() {
        let config = SearchConfig {
            recency_alpha: 0.5,
            recency_window: None,
            salience_weight: 0.0, // pure recency for predictable test
            ..Default::default()
        };
        let chunk = blend_test_chunk();
        // No accesses → relevancy=0.0 → blended = 0.8 * 0.5 + 0.0 * 0.5 = 0.4
        let score = config.blend_score(0.8, &chunk);
        assert!((score - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_blend_score_full_alpha() {
        let config = SearchConfig {
            recency_alpha: 1.0,
            recency_window: None,
            salience_weight: 0.0,
            ..Default::default()
        };
        let chunk = blend_test_chunk();
        // Full relevancy weight → 0.8 * 0.0 + relevancy(0) * 1.0 = 0.0
        let score = config.blend_score(0.8, &chunk);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_blend_score_with_accesses() {
        let config = SearchConfig {
            recency_alpha: 0.3,
            recency_window: None,
            ..Default::default()
        };
        let mut chunk = blend_test_chunk();
        chunk.access_profile.record_access();
        chunk.access_profile.record_access();
        // Has accesses → relevancy > 0 → blended > pure vector
        let score = config.blend_score(0.5, &chunk);
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_blend_score_with_salience() {
        let config = SearchConfig {
            recency_alpha: 0.5,
            salience_weight: 1.0, // pure salience
            ..Default::default()
        };
        let mut chunk = blend_test_chunk();
        chunk.perspectives = vec!["decisions".to_string(), "learnings".to_string()];
        chunk
            .relations
            .push(crate::ChunkRelation::superseded_by("newer"));

        let score = config.blend_score(0.6, &chunk);
        // Salience > 0 because of perspectives + relations
        assert!(score > 0.3); // vector portion (0.6 * 0.5 = 0.3) plus salience
    }

    #[test]
    fn test_with_min_salience() {
        let config = SearchConfig::for_query(5, false, None).with_min_salience(Some(0.5));
        assert_eq!(config.min_salience, Some(0.5));
    }

    #[test]
    fn test_with_min_salience_none() {
        let config = SearchConfig::for_query(5, false, None).with_min_salience(None);
        assert!(config.min_salience.is_none());
    }

    #[test]
    fn test_with_min_score() {
        let config = SearchConfig::for_query(5, false, None).with_min_score(Some(0.7));
        assert_eq!(config.min_score, 0.7);
    }

    #[test]
    fn test_with_min_score_none() {
        let config = SearchConfig::for_query(5, false, None).with_min_score(None);
        assert_eq!(config.min_score, 0.0);
    }

    #[test]
    fn test_blend_score_no_salience_boost_below_min() {
        let config = SearchConfig {
            recency_alpha: 0.5,
            min_salience: Some(0.5),
            salience_weight: 1.0,
            ..Default::default()
        };
        let mut chunk = blend_test_chunk();
        chunk.perspectives = vec!["decisions".to_string()];
        chunk.access_profile.record_access();
        chunk.access_profile.record_access();
        let vec_score = 0.7;
        let salience = salience::compute(&chunk, &salience::SalienceWeights::default());
        assert!(
            salience.composite < 0.5,
            "Sanity check: salience {} should be < 0.5",
            salience.composite
        );
        let score = config.blend_score(vec_score, &chunk);
        let expected = vec_score * 0.5 + chunk.access_profile.relevancy_score(None) * 0.5;
        assert!(
            (score - expected).abs() < 0.01,
            "Below min_salience should use recency only, not salience"
        );
    }

    #[test]
    fn test_blend_score_with_salience_boost_above_min() {
        let config = SearchConfig {
            recency_alpha: 0.5,
            min_salience: Some(0.3),
            salience_weight: 1.0,
            ..Default::default()
        };
        let mut chunk = blend_test_chunk();
        for _ in 0..4 {
            chunk.access_profile.record_access();
        }
        chunk.perspectives = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        chunk
            .relations
            .push(crate::ChunkRelation::superseded_by("newer"));
        let vec_score = 0.7;
        let salience = salience::compute(&chunk, &salience::SalienceWeights::default());
        assert!(
            salience.composite >= 0.3,
            "Sanity check: salience {} should be >= 0.3",
            salience.composite
        );
        let score = config.blend_score(vec_score, &chunk);
        let expected_relevancy =
            chunk.access_profile.relevancy_score(None) * 0.0 + salience.composite * 1.0;
        let expected = vec_score * 0.5 + expected_relevancy * 0.5;
        assert!(
            (score - expected).abs() < 0.01,
            "Above min_salience should use salience in blend"
        );
    }

    #[test]
    fn test_blend_score_no_min_salience_always_boosts() {
        let config = SearchConfig {
            recency_alpha: 0.5,
            min_salience: None,
            salience_weight: 1.0,
            ..Default::default()
        };
        let mut chunk = blend_test_chunk();
        chunk.perspectives = vec!["decisions".to_string()];
        chunk.access_profile.record_access();
        let vec_score = 0.7;
        let score = config.blend_score(vec_score, &chunk);
        let salience = salience::compute(&chunk, &salience::SalienceWeights::default());
        let expected_relevancy =
            chunk.access_profile.relevancy_score(None) * 0.0 + salience.composite * 1.0;
        let expected = vec_score * 0.5 + expected_relevancy * 0.5;
        assert!(
            (score - expected).abs() < 0.01,
            "No min_salience should always use salience"
        );
    }

    #[test]
    fn test_blend_score_min_salience_zero_allows_all() {
        let config = SearchConfig {
            recency_alpha: 0.0,
            min_salience: Some(0.0),
            ..Default::default()
        };
        let chunk = blend_test_chunk();
        let vec_score = 0.7;
        let score = config.blend_score(vec_score, &chunk);
        assert_eq!(score, vec_score, "Min_salience 0.0 should pass all entries");
    }

    #[test]
    fn test_blend_score_min_salience_zero_with_alpha() {
        let config = SearchConfig {
            recency_alpha: 0.5,
            min_salience: Some(0.0),
            salience_weight: 1.0,
            ..Default::default()
        };
        let chunk = blend_test_chunk();
        let vec_score = 0.7;
        let score = config.blend_score(vec_score, &chunk);
        let salience = salience::compute(&chunk, &salience::SalienceWeights::default());
        let expected = vec_score * 0.5
            + (chunk.access_profile.relevancy_score(None) * 0.0 + salience.composite) * 0.5;
        assert!(
            (score - expected).abs() < 0.01,
            "Min_salience 0.0 with alpha should still blend"
        );
    }

    #[test]
    fn test_alpha_for_window_none() {
        assert_eq!(SearchConfig::alpha_for_window(None), DEFAULT_RECENCY_ALPHA);
    }

    #[test]
    fn test_alpha_for_window_some() {
        assert_eq!(
            SearchConfig::alpha_for_window(Some(RecencyWindow::Day)),
            ACTIVE_RECENCY_ALPHA
        );
    }

    #[tokio::test]
    async fn test_search_with_perspective_filter() {
        // Create chunks: one with perspective, one without
        let mut chunk_with = HierarchicalChunk::new(
            "decisions content".to_string(),
            ChunkLevel::H1,
            None,
            "test".to_string(),
            "test.md".to_string(),
        );
        chunk_with.embedding = Some(vec![1.0, 0.0, 0.0]);
        chunk_with.perspectives = vec!["decisions".to_string()];

        let mut chunk_without = HierarchicalChunk::new(
            "general content".to_string(),
            ChunkLevel::H1,
            None,
            "test".to_string(),
            "test.md".to_string(),
        );
        chunk_without.embedding = Some(vec![0.0, 1.0, 0.0]);

        // Test WITHOUT perspective filter: gets all results
        {
            let store = MockStore::new();
            store.set_search_results(vec![
                SearchResult {
                    chunk: chunk_with.clone(),
                    score: 0.9,
                },
                SearchResult {
                    chunk: chunk_without.clone(),
                    score: 0.7,
                },
            ]);

            let config = SearchConfig::for_query(10, false, None);
            let search = HierarchicalSearch::new(store, MockEmbedder::new(3)).with_config(config);
            let results = search.search("test").await.unwrap();
            assert_eq!(results.len(), 2);
        }

        // Test WITH perspective filter: mock filters by perspective
        {
            let store = MockStore::new();
            store.set_search_results(vec![
                SearchResult {
                    chunk: chunk_with.clone(),
                    score: 0.9,
                },
                SearchResult {
                    chunk: chunk_without.clone(),
                    score: 0.7,
                },
            ]);

            let config = SearchConfig::for_query(10, false, None)
                .with_perspective(Some("decisions".to_string()));
            let search = HierarchicalSearch::new(store, MockEmbedder::new(3)).with_config(config);
            let results = search.search("test").await.unwrap();
            assert_eq!(results.len(), 1);
            assert!(results[0]
                .chunk
                .perspectives
                .contains(&"decisions".to_string()));
        }
    }

    // --- min_score filtering tests ---

    #[tokio::test]
    async fn test_search_filters_by_min_score_new() {
        let store = MockStore::new();

        let mut chunk1 = HierarchicalChunk::new(
            "high score".to_string(),
            ChunkLevel::H1,
            None,
            "path1".to_string(),
            "test.md".to_string(),
        );
        chunk1.embedding = Some(vec![1.0, 0.0, 0.0]);

        let mut chunk2 = HierarchicalChunk::new(
            "low score".to_string(),
            ChunkLevel::H1,
            None,
            "path2".to_string(),
            "test.md".to_string(),
        );
        chunk2.embedding = Some(vec![0.0, 1.0, 0.0]);

        let results = vec![
            SearchResult {
                chunk: chunk1,
                score: 0.8,
            },
            SearchResult {
                chunk: chunk2,
                score: 0.2,
            },
        ];

        store.set_search_results(results);

        let embedder = MockEmbedder::new(3);
        let config = SearchConfig {
            min_score: 0.5,
            ..Default::default()
        };
        let search = HierarchicalSearch::new(store, embedder).with_config(config);

        let search_results = search.search("test query").await.unwrap();

        // Only chunk1 should pass the min_score filter
        assert_eq!(
            search_results.len(),
            1,
            "Should filter out low-score entries"
        );
        assert_eq!(
            search_results[0].chunk.content, "high score",
            "Should retain high-score entry"
        );
    }

    #[tokio::test]
    async fn test_search_min_score_all_pass() {
        let store = MockStore::new();

        let chunks: Vec<SearchResult> = (0..5)
            .map(|i| {
                let mut chunk = HierarchicalChunk::new(
                    format!("content {}", i),
                    ChunkLevel::H1,
                    None,
                    format!("path{}", i),
                    "test.md".to_string(),
                );
                chunk.embedding = Some(vec![1.0; 3]);
                SearchResult {
                    chunk,
                    score: 0.6 + (i as f32 * 0.05),
                }
            })
            .collect();

        store.set_search_results(chunks);

        let embedder = MockEmbedder::new(3);
        let config = SearchConfig {
            min_score: 0.5,
            ..Default::default()
        };
        let search = HierarchicalSearch::new(store, embedder).with_config(config);

        let results = search.search("test").await.unwrap();
        assert_eq!(
            results.len(),
            5,
            "All entries should pass when min_score is low"
        );
    }

    #[tokio::test]
    async fn test_search_min_score_all_filtered() {
        let store = MockStore::new();

        let chunks: Vec<SearchResult> = (0..3)
            .map(|i| {
                let mut chunk = HierarchicalChunk::new(
                    format!("content {}", i),
                    ChunkLevel::H1,
                    None,
                    format!("path{}", i),
                    "test.md".to_string(),
                );
                chunk.embedding = Some(vec![1.0; 3]);
                SearchResult {
                    chunk,
                    score: 0.1 + (i as f32 * 0.05),
                }
            })
            .collect();

        store.set_search_results(chunks);

        let embedder = MockEmbedder::new(3);
        let config = SearchConfig {
            min_score: 0.9,
            top_k: 10,
            ..Default::default()
        };
        let search = HierarchicalSearch::new(store, embedder).with_config(config);

        let results = search.search("test").await.unwrap();
        assert_eq!(
            results.len(),
            0,
            "All entries should be filtered out by high min_score"
        );
    }

    #[tokio::test]
    async fn test_search_min_score_zero_allows_all() {
        let store = MockStore::new();

        let chunks: Vec<SearchResult> = (0..5)
            .map(|i| {
                let mut chunk = HierarchicalChunk::new(
                    format!("content {}", i),
                    ChunkLevel::H1,
                    None,
                    format!("path{}", i),
                    "test.md".to_string(),
                );
                chunk.embedding = Some(vec![1.0; 3]);
                SearchResult {
                    chunk,
                    score: i as f32 * 0.1,
                }
            })
            .collect();

        store.set_search_results(chunks);

        let embedder = MockEmbedder::new(3);
        let config = SearchConfig {
            min_score: 0.0,
            ..Default::default()
        };
        let search = HierarchicalSearch::new(store, embedder).with_config(config);

        let results = search.search("test").await.unwrap();
        assert_eq!(
            results.len(),
            5,
            "min_score=0.0 should allow all entries regardless of score"
        );
    }

    // --- Edge case tests ---

    #[test]
    fn test_blend_score_negative_vector_score() {
        let config = SearchConfig {
            recency_alpha: 0.0,
            min_salience: None,
            ..Default::default()
        };
        let chunk = blend_test_chunk();
        // Scores should be handled, even if negative (cosine similarity is always non-negative)
        let score = config.blend_score(-0.1, &chunk);
        assert_eq!(score, -0.1, "Negative scores should pass through unchanged");
    }

    #[test]
    fn test_blend_score_vector_score_greater_than_one() {
        let config = SearchConfig {
            recency_alpha: 0.0,
            min_salience: None,
            ..Default::default()
        };
        let chunk = blend_test_chunk();
        let score = config.blend_score(1.5, &chunk);
        assert_eq!(score, 1.5, "Scores > 1.0 should pass through unchanged");
    }

    #[test]
    fn test_blend_score_min_salience_negative_threshold() {
        let config = SearchConfig {
            recency_alpha: 0.0,
            min_salience: Some(-0.5),
            ..Default::default()
        };
        let chunk = blend_test_chunk();
        let vec_score = 0.7;
        let score = config.blend_score(vec_score, &chunk);
        let salience = salience::compute(&chunk, &salience::SalienceWeights::default());
        // Salience is always >= 0, so -0.5 threshold means everything passes
        assert!(
            salience.composite >= -0.5,
            "Salience should be >= negative threshold"
        );
        assert_eq!(
            score, vec_score,
            "Negative threshold should allow all entries"
        );
    }

    #[test]
    fn test_blend_score_min_salience_above_one() {
        let config = SearchConfig {
            recency_alpha: 0.0,
            min_salience: Some(1.5),
            ..Default::default()
        };
        let chunk = blend_test_chunk();
        let vec_score = 0.7;
        let score = config.blend_score(vec_score, &chunk);
        let salience = salience::compute(&chunk, &salience::SalienceWeights::default());
        // Salience max is 1.0, so threshold 1.5 means nothing passes
        assert!(salience.composite < 1.5, "Salience should be < 1.5");
        assert_eq!(
            score, vec_score,
            "Threshold > 1.0 should block all salience boosting"
        );
    }

    #[test]
    fn test_min_salience_boundary_exact_match() {
        let config = SearchConfig {
            recency_alpha: 0.0,
            min_salience: Some(0.5),
            ..Default::default()
        };
        let mut chunk = blend_test_chunk();
        // Create chunk with exact 0.5 salience (approximately)
        for _ in 0..4 {
            chunk.access_profile.record_access();
        }
        chunk.perspectives = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        chunk
            .relations
            .push(crate::ChunkRelation::superseded_by("newer"));

        let salience = salience::compute(&chunk, &salience::SalienceWeights::default());
        let vec_score = 0.7;
        let score = config.blend_score(vec_score, &chunk);
        // Exact or above threshold should pass (>=)
        assert!(salience.composite >= 0.5, "Should be at threshold");
        assert_eq!(
            score, vec_score,
            "Exact threshold should allow salience boost"
        );
    }

    #[tokio::test]
    async fn test_min_score_boundary_exact_match() {
        let store = MockStore::new();

        let mut chunk = HierarchicalChunk::new(
            "exact threshold".to_string(),
            ChunkLevel::H1,
            None,
            "path".to_string(),
            "test.md".to_string(),
        );
        chunk.embedding = Some(vec![1.0; 3]);

        store.set_search_results(vec![SearchResult { chunk, score: 0.5 }]);

        let embedder = MockEmbedder::new(3);
        let config = SearchConfig {
            min_score: 0.5,
            top_k: 10,
            ..Default::default()
        };
        let search = HierarchicalSearch::new(store, embedder).with_config(config);

        let results = search.search("test").await.unwrap();
        assert_eq!(
            results.len(),
            1,
            "Exact threshold score (0.5 >= 0.5) should pass"
        );
    }

    #[tokio::test]
    async fn test_min_score_boundary_below_excluded() {
        let store = MockStore::new();

        let mut chunk = HierarchicalChunk::new(
            "below threshold".to_string(),
            ChunkLevel::H1,
            None,
            "path".to_string(),
            "test.md".to_string(),
        );
        chunk.embedding = Some(vec![1.0; 3]);

        store.set_search_results(vec![SearchResult {
            chunk,
            score: 0.4999,
        }]);

        let embedder = MockEmbedder::new(3);
        let config = SearchConfig {
            min_score: 0.5,
            top_k: 10,
            ..Default::default()
        };
        let search = HierarchicalSearch::new(store, embedder).with_config(config);

        let results = search.search("test").await.unwrap();
        assert_eq!(
            results.len(),
            0,
            "Below threshold score (0.4999 < 0.5) should be excluded"
        );
    }

    #[test]
    fn test_blend_score_both_thresholds_combined() {
        let config = SearchConfig {
            recency_alpha: 0.5,
            min_salience: Some(0.3),
            salience_weight: 1.0,
            ..Default::default()
        };
        let mut chunk = blend_test_chunk();
        for _ in 0..4 {
            chunk.access_profile.record_access();
        }
        chunk.perspectives = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let vec_score = 0.7;
        let salience = salience::compute(&chunk, &salience::SalienceWeights::default());
        assert!(
            salience.composite >= 0.3,
            "Sanity: salience {} >= 0.3",
            salience.composite
        );

        let score = config.blend_score(vec_score, &chunk);
        let expected_relevancy =
            chunk.access_profile.relevancy_score(None) * 0.0 + salience.composite * 1.0;
        let expected = vec_score * 0.5 + expected_relevancy * 0.5;
        assert!(
            (score - expected).abs() < 0.01,
            "Combined thresholds should work correctly"
        );
    }

    // --- search_by_embedding tests ---

    #[tokio::test]
    async fn test_search_by_embedding_entry_not_found() {
        let store = MockStore::new();
        let embedder = MockEmbedder::new(3);
        let config = SearchConfig::default();
        let search = HierarchicalSearch::new(store, embedder).with_config(config);

        let result = search.search_by_embedding("nonexistent-id", 5).await;
        assert!(result.is_err(), "Should return error for nonexistent entry");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not found"),
            "Error should mention 'not found'"
        );
    }

    #[tokio::test]
    async fn test_search_by_embedding_no_embedding() {
        let store = MockStore::new();

        let mut chunk = HierarchicalChunk::new(
            "target without embedding".to_string(),
            ChunkLevel::H1,
            None,
            "target".to_string(),
            "test.md".to_string(),
        );
        chunk.embedding = None;

        store.add_chunk(chunk);
        let embedder = MockEmbedder::new(3);
        let config = SearchConfig::default();
        let search = HierarchicalSearch::new(store, embedder).with_config(config);

        let result = search.search_by_embedding("target", 5).await;
        assert!(
            result.is_err(),
            "Should return error for entry without embedding"
        );
        let err = result.unwrap_err().to_string();
        assert!(err.contains("search"), "Error should be of search type");
    }

    #[tokio::test]
    async fn test_search_by_embedding_excludes_target() {
        let store = MockStore::new();

        let mut target = HierarchicalChunk::new(
            "target entry".to_string(),
            ChunkLevel::H1,
            None,
            "target".to_string(),
            "test.md".to_string(),
        );
        target.embedding = Some(vec![1.0; 3]);
        target.id = "target-id".to_string();

        let mut other = HierarchicalChunk::new(
            "other entry".to_string(),
            ChunkLevel::H1,
            None,
            "other".to_string(),
            "test.md".to_string(),
        );
        other.embedding = Some(vec![1.0; 3]);
        other.id = "other-id".to_string();

        store.add_chunk(target);
        store.add_chunk(other.clone());

        store.set_search_results(vec![
            SearchResult {
                chunk: HierarchicalChunk {
                    id: "target-id".to_string(),
                    ..other.clone()
                },
                score: 0.9,
            },
            SearchResult {
                chunk: other,
                score: 0.8,
            },
        ]);

        let embedder = MockEmbedder::new(3);
        let config = SearchConfig::default();
        let search = HierarchicalSearch::new(store, embedder).with_config(config);

        let results = search.search_by_embedding("target-id", 5).await.unwrap();

        assert!(
            results.iter().all(|r| r.chunk.id != "target-id"),
            "Target entry should be excluded from results"
        );
        assert_eq!(results.len(), 1, "Should return only the non-target entry");
    }

    #[tokio::test]
    async fn test_search_by_embedding_respects_min_score() {
        let store = MockStore::new();

        let mut target = HierarchicalChunk::new(
            "target".to_string(),
            ChunkLevel::H1,
            None,
            "target".to_string(),
            "test.md".to_string(),
        );
        target.embedding = Some(vec![1.0; 3]);
        target.id = "target-id".to_string();
        store.add_chunk(target);

        let mut low = HierarchicalChunk::new(
            "low score".to_string(),
            ChunkLevel::H1,
            None,
            "low".to_string(),
            "test.md".to_string(),
        );
        low.embedding = Some(vec![1.0; 3]);
        low.id = "low".to_string();

        let mut high = HierarchicalChunk::new(
            "high score".to_string(),
            ChunkLevel::H1,
            None,
            "high".to_string(),
            "test.md".to_string(),
        );
        high.embedding = Some(vec![1.0; 3]);
        high.id = "high".to_string();

        let results_with_scores: Vec<SearchResult> = vec![
            SearchResult {
                chunk: low,
                score: 0.3,
            },
            SearchResult {
                chunk: high,
                score: 0.7,
            },
        ];

        store.set_search_results(results_with_scores);

        let embedder = MockEmbedder::new(3);
        let config = SearchConfig {
            min_score: 0.5,
            ..Default::default()
        };
        let search = HierarchicalSearch::new(store, embedder).with_config(config);

        let results = search.search_by_embedding("target-id", 5).await.unwrap();

        assert_eq!(results.len(), 1, "Should return only high-score entry");
        assert_eq!(
            results[0].chunk.id, "high",
            "Should return the entry that passes min_score"
        );
    }

    #[tokio::test]
    async fn test_search_by_embedding_respects_visibility_filter() {
        let store = MockStore::new();

        let mut target = HierarchicalChunk::new(
            "target".to_string(),
            ChunkLevel::H1,
            None,
            "target".to_string(),
            "test.md".to_string(),
        );
        target.embedding = Some(vec![1.0; 3]);
        target.id = "target-id".to_string();
        store.add_chunk(target);

        let mut normal = HierarchicalChunk::new(
            "normal visibility".to_string(),
            ChunkLevel::H1,
            None,
            "normal".to_string(),
            "test.md".to_string(),
        );
        normal.embedding = Some(vec![1.0; 3]);
        normal.visibility = "normal".to_string();

        let mut deep_only = HierarchicalChunk::new(
            "deep only".to_string(),
            ChunkLevel::H1,
            None,
            "deep-only".to_string(),
            "test.md".to_string(),
        );
        deep_only.embedding = Some(vec![1.0; 3]);
        deep_only.visibility = "deep_only".to_string();

        store.set_search_results(vec![
            SearchResult {
                chunk: normal,
                score: 0.8,
            },
            SearchResult {
                chunk: deep_only,
                score: 0.7,
            },
        ]);

        let embedder = MockEmbedder::new(3);
        let config = SearchConfig {
            deep: false,
            ..Default::default()
        };
        let search = HierarchicalSearch::new(store, embedder).with_config(config);

        let results = search.search_by_embedding("target-id", 5).await.unwrap();

        assert_eq!(results.len(), 1, "Should return only visible entry");
        assert_eq!(
            results[0].chunk.visibility, "normal",
            "Should return only 'normal' visibility entry"
        );
    }

    #[tokio::test]
    async fn test_search_by_embedding_deep_mode_includes_hidden() {
        let store = MockStore::new();

        let mut target = HierarchicalChunk::new(
            "target".to_string(),
            ChunkLevel::H1,
            None,
            "target".to_string(),
            "test.md".to_string(),
        );
        target.embedding = Some(vec![1.0; 3]);
        target.id = "target-id".to_string();
        store.add_chunk(target);

        let mut normal = HierarchicalChunk::new(
            "normal visibility".to_string(),
            ChunkLevel::H1,
            None,
            "normal".to_string(),
            "test.md".to_string(),
        );
        normal.embedding = Some(vec![1.0; 3]);
        normal.visibility = "normal".to_string();

        let mut deep_only = HierarchicalChunk::new(
            "deep only".to_string(),
            ChunkLevel::H1,
            None,
            "deep-only".to_string(),
            "test.md".to_string(),
        );
        deep_only.embedding = Some(vec![1.0; 3]);
        deep_only.visibility = "deep_only".to_string();

        store.set_search_results(vec![
            SearchResult {
                chunk: normal,
                score: 0.8,
            },
            SearchResult {
                chunk: deep_only,
                score: 0.7,
            },
        ]);

        let embedder = MockEmbedder::new(3);
        let config = SearchConfig {
            deep: true,
            ..Default::default()
        };
        let search = HierarchicalSearch::new(store, embedder).with_config(config);

        let results = search.search_by_embedding("target-id", 5).await.unwrap();

        assert_eq!(results.len(), 2, "Should return all entries in deep mode");
        assert!(
            results.iter().any(|r| r.chunk.visibility == "deep_only"),
            "Should include deep_only visibility entries"
        );
    }
}
