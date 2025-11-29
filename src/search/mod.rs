use crate::embedder::Embedder;
use crate::store::{SearchResult, VectorStore};
use crate::{ChunkLevel, HierarchicalChunk, Result};

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
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            children_k: 3,
            max_depth: 3,
            min_score: 0.0,
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
    /// 2. For each match, search its children
    /// 3. Build a hierarchical result tree
    pub async fn search(&self, query: &str) -> Result<Vec<HierarchicalSearchResult>> {
        let query_embedding = self.embed_query(query)?;

        // Step 1: Find top-level matches across all levels
        let top_results = self
            .store
            .search(&query_embedding, self.config.top_k, None)
            .await?;

        let mut hierarchical_results = Vec::new();

        for result in top_results {
            if result.score < self.config.min_score {
                continue;
            }

            // Build the hierarchy path (from root to this chunk)
            let hierarchy_path = self.build_hierarchy_path(&result.chunk).await?;

            // Get relevant children
            let relevant_children = self.search_children(&query_embedding, &result.chunk).await?;

            hierarchical_results.push(HierarchicalSearchResult {
                chunk: result.chunk,
                score: result.score,
                hierarchy_path,
                relevant_children,
            });
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
                scored_children.push(SearchResult { chunk: child, score });
            }
        }

        // Sort by score descending
        scored_children.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Take top results
        let top_children: Vec<_> = scored_children
            .into_iter()
            .take(self.config.top_k)
            .filter(|r| r.score >= self.config.min_score)
            .collect();

        let mut results = Vec::new();
        for result in top_children {
            let hierarchy_path = self.build_hierarchy_path(&result.chunk).await?;
            let relevant_children = self.search_children(&query_embedding, &result.chunk).await?;

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
    async fn build_hierarchy_path(&self, chunk: &HierarchicalChunk) -> Result<Vec<HierarchicalChunk>> {
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
                    SearchResult { chunk: child.clone(), score }
                })
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

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

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.0001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.0001);
    }
}
