use crate::chunk::now_epoch_secs;
use crate::embedder::Embedder;
use crate::store::{SearchResult, VectorStore};
use crate::{ChunkLevel, HierarchicalChunk, RecencyWindow, Result};

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

/// Configuration for hierarchical search
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Number of top-level chunks to retrieve
    pub top_k: usize,
    /// Number of children to retrieve per parent
    pub children_k: usize,
    /// Maximum depth to search
    pub max_depth: usize,
    /// Minimum score threshold
    pub min_score: f32,
    /// Deep search: include all visibilities (DeepOnly, expired, custom)
    pub deep: bool,
    /// Recency window for relevancy scoring. None = balanced weighting.
    pub recency_window: Option<RecencyWindow>,
    /// Blending factor: 0.0 = pure vector similarity, 1.0 = pure relevancy.
    /// Default: 0.15
    pub recency_alpha: f32,
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
            recency_alpha: 0.15,
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
        let alpha = self.config.recency_alpha;
        let recency_window = self.config.recency_window;

        // Fetch more than top_k so we still have enough after visibility filtering
        let fetch_k = if self.config.deep {
            self.config.top_k
        } else {
            self.config.top_k * 2
        };

        // Step 1: Find top-level matches across all levels
        let top_results = self.store.search(&query_embedding, fetch_k, None).await?;

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

            let final_score = if alpha > 0.0 {
                let relevancy = chunk.access_profile.relevancy_score(recency_window);
                vector_score * (1.0 - alpha) + relevancy * alpha
            } else {
                vector_score
            };

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

    /// Search within a specific subtree (starting from a parent chunk)
    pub async fn search_subtree(
        &self,
        query: &str,
        parent_id: &str,
    ) -> Result<Vec<HierarchicalSearchResult>> {
        let query_embedding = self.embed_query(query)?;

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
        for result in top_children {
            let hierarchy_path = self.build_hierarchy_path(&result.chunk).await?;
            let relevant_children = self
                .search_children(&query_embedding, &result.chunk)
                .await?;

            results.push(HierarchicalSearchResult {
                chunk: result.chunk,
                score: result.score,
                hierarchy_path,
                relevant_children,
            });
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
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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
}
