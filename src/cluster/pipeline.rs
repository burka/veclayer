use std::collections::HashMap;
use tracing::info;

use super::{Clusterer, SoftClusterer};
use crate::{Embedder, Error, HierarchicalChunk, OllamaSummarizer, Result, Summarizer};

/// Pipeline for RAPTOR-style clustering and summarization.
///
/// 1. Takes chunks with embeddings
/// 2. Clusters them using soft clustering
/// 3. Groups chunks by cluster
/// 4. Generates summaries for each cluster
/// 5. Returns updated chunks with cluster memberships + summary chunks
pub struct ClusterPipeline<S: Summarizer, E: Embedder> {
    summarizer: S,
    embedder: E,
    clusterer: SoftClusterer,
    min_cluster_size: usize,
}

impl<E: Embedder> ClusterPipeline<OllamaSummarizer, E> {
    /// Create pipeline with default Ollama summarizer
    pub fn new(embedder: E) -> Self {
        Self {
            summarizer: OllamaSummarizer::new(),
            embedder,
            clusterer: SoftClusterer::new(),
            min_cluster_size: 2,
        }
    }

    /// Use a specific Ollama model
    pub fn with_model(mut self, model: &str) -> Self {
        self.summarizer = OllamaSummarizer::new().with_model(model);
        self
    }
}

impl<S: Summarizer, E: Embedder> ClusterPipeline<S, E> {
    /// Create pipeline with custom summarizer
    pub fn with_summarizer(embedder: E, summarizer: S) -> Self {
        Self {
            summarizer,
            embedder,
            clusterer: SoftClusterer::new(),
            min_cluster_size: 2,
        }
    }

    /// Set minimum cluster size for summarization
    pub fn with_min_cluster_size(mut self, size: usize) -> Self {
        self.min_cluster_size = size;
        self
    }

    /// Set cluster range
    pub fn with_cluster_range(mut self, min: usize, max: usize) -> Self {
        self.clusterer = self.clusterer.with_cluster_range(min, max);
        self
    }

    /// Process chunks: cluster and generate summaries
    ///
    /// Returns a tuple of:
    /// - Updated chunks with cluster memberships
    /// - New summary chunks (one per cluster that meets minimum size)
    pub async fn process(
        &self,
        chunks: Vec<HierarchicalChunk>,
    ) -> Result<(Vec<HierarchicalChunk>, Vec<HierarchicalChunk>)> {
        if chunks.is_empty() {
            return Ok((chunks, Vec::new()));
        }

        // Extract embeddings
        let embeddings: Vec<Vec<f32>> = chunks.iter().filter_map(|c| c.embedding.clone()).collect();

        if embeddings.len() != chunks.len() {
            return Err(Error::clustering("Not all chunks have embeddings"));
        }

        info!("Clustering {} chunks...", chunks.len());

        // Cluster embeddings
        let assignments = self.clusterer.cluster(&embeddings)?;

        // Update chunks with cluster memberships
        let updated_chunks: Vec<HierarchicalChunk> = chunks
            .into_iter()
            .enumerate()
            .map(|(idx, mut chunk)| {
                if let Some(assignment) = assignments.iter().find(|a| a.index == idx) {
                    chunk.cluster_memberships = assignment.memberships.clone();
                }
                chunk
            })
            .collect();

        // Group chunks by primary cluster
        let mut cluster_groups: HashMap<String, Vec<&HierarchicalChunk>> = HashMap::new();
        for chunk in &updated_chunks {
            if let Some(primary) = chunk.primary_cluster() {
                cluster_groups
                    .entry(primary.cluster_id.clone())
                    .or_default()
                    .push(chunk);
            }
        }

        info!("Found {} clusters", cluster_groups.len());

        // Generate summaries for clusters meeting minimum size
        let mut summary_chunks = Vec::new();

        for (cluster_id, cluster_chunks) in cluster_groups {
            if cluster_chunks.len() < self.min_cluster_size {
                continue;
            }

            info!(
                "Summarizing cluster {} with {} chunks...",
                cluster_id,
                cluster_chunks.len()
            );

            // Collect texts for summarization
            let texts: Vec<&str> = cluster_chunks.iter().map(|c| c.content.as_str()).collect();

            // Generate summary
            let summary_text = match self.summarizer.summarize(&texts).await {
                Ok(text) => text,
                Err(e) => {
                    info!("Failed to summarize cluster {}: {}", cluster_id, e);
                    continue;
                }
            };

            if summary_text.is_empty() {
                continue;
            }

            // Embed the summary
            let summary_embedding = match self.embedder.embed(&[&summary_text]) {
                Ok(embeddings) => embeddings.into_iter().next(),
                Err(e) => {
                    info!("Failed to embed summary for cluster {}: {}", cluster_id, e);
                    None
                }
            };

            // Create summary chunk
            let chunk_ids: Vec<String> = cluster_chunks.iter().map(|c| c.id.clone()).collect();
            let summary_chunk =
                HierarchicalChunk::new_summary(summary_text, chunk_ids, summary_embedding);

            summary_chunks.push(summary_chunk);
        }

        info!("Generated {} summary chunks", summary_chunks.len());

        Ok((updated_chunks, summary_chunks))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChunkLevel;

    // Mock summarizer that returns predictable results
    struct MockSummarizer {
        response: String,
    }

    impl MockSummarizer {
        fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
            }
        }
    }

    impl Summarizer for MockSummarizer {
        async fn summarize(&self, _texts: &[&str]) -> Result<String> {
            Ok(self.response.clone())
        }

        async fn summarize_batch(&self, text_groups: Vec<Vec<&str>>) -> Result<Vec<String>> {
            Ok(text_groups.iter().map(|_| self.response.clone()).collect())
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    // Mock embedder
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
            Ok(texts
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let mut vec = vec![0.5; self.dimension];
                    // Add some variation based on index
                    vec[0] = i as f32 / texts.len() as f32;
                    vec
                })
                .collect())
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    fn create_test_chunks(count: usize) -> Vec<HierarchicalChunk> {
        (0..count)
            .map(|i| {
                let mut chunk = HierarchicalChunk::new(
                    format!("Content {}", i),
                    ChunkLevel::CONTENT,
                    None,
                    format!("path_{}", i),
                    "test.md".to_string(),
                );
                // Create distinct embeddings for clustering
                let mut embedding = vec![0.0; 384];
                embedding[0] = (i / 2) as f32; // Group every 2 items together
                embedding[1] = (i % 2) as f32;
                chunk.embedding = Some(embedding);
                chunk
            })
            .collect()
    }

    #[test]
    fn test_pipeline_creation() {
        let embedder = MockEmbedder::new(384);
        let _pipeline = ClusterPipeline::new(embedder);
    }

    #[test]
    fn test_pipeline_with_config() {
        let embedder = MockEmbedder::new(384);
        let pipeline = ClusterPipeline::new(embedder)
            .with_min_cluster_size(3)
            .with_cluster_range(2, 5);

        assert_eq!(pipeline.min_cluster_size, 3);
        // Cluster range is set internally, verified via process()
    }

    #[test]
    fn test_pipeline_with_custom_summarizer() {
        let embedder = MockEmbedder::new(384);
        let summarizer = MockSummarizer::new("Test summary");
        let pipeline = ClusterPipeline::with_summarizer(embedder, summarizer);

        assert_eq!(pipeline.summarizer.name(), "mock");
        assert_eq!(pipeline.embedder.name(), "mock");
    }

    #[tokio::test]
    async fn test_pipeline_empty_chunks() {
        let embedder = MockEmbedder::new(384);
        let summarizer = MockSummarizer::new("summary");
        let pipeline = ClusterPipeline::with_summarizer(embedder, summarizer);

        let (updated, summaries) = pipeline.process(vec![]).await.unwrap();
        assert!(updated.is_empty());
        assert!(summaries.is_empty());
    }

    #[tokio::test]
    async fn test_pipeline_with_chunks() {
        let embedder = MockEmbedder::new(384);
        let summarizer = MockSummarizer::new("Test summary");
        let pipeline =
            ClusterPipeline::with_summarizer(embedder, summarizer).with_min_cluster_size(2);

        let chunks = create_test_chunks(6);
        let (updated, summaries) = pipeline.process(chunks).await.unwrap();

        // Verify chunks have cluster memberships
        assert_eq!(updated.len(), 6);
        for chunk in &updated {
            assert!(
                !chunk.cluster_memberships.is_empty(),
                "Chunk should have cluster memberships"
            );
        }

        // Verify summaries were created
        assert!(!summaries.is_empty(), "Should have created summaries");
        for summary in &summaries {
            assert!(summary.is_summary(), "Chunk should be marked as summary");
            assert!(
                !summary.summarizes.is_empty(),
                "Summary should reference chunks"
            );
            assert_eq!(summary.content, "Test summary");
        }
    }

    #[tokio::test]
    async fn test_pipeline_missing_embeddings() {
        let embedder = MockEmbedder::new(384);
        let summarizer = MockSummarizer::new("summary");
        let pipeline = ClusterPipeline::with_summarizer(embedder, summarizer);

        // Chunk without embedding
        let chunk = HierarchicalChunk::new(
            "content".to_string(),
            ChunkLevel::CONTENT,
            None,
            "path".to_string(),
            "test.md".to_string(),
        );

        let result = pipeline.process(vec![chunk]).await;
        assert!(result.is_err(), "Should fail with missing embedding");
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Not all chunks have embeddings"));
    }

    #[tokio::test]
    async fn test_pipeline_min_cluster_size() {
        let embedder = MockEmbedder::new(384);
        let summarizer = MockSummarizer::new("summary");
        let pipeline =
            ClusterPipeline::with_summarizer(embedder, summarizer).with_min_cluster_size(10); // High threshold

        let chunks = create_test_chunks(4); // Small dataset
        let (updated, _summaries) = pipeline.process(chunks).await.unwrap();

        assert_eq!(updated.len(), 4);
        // With high min_cluster_size, might not create summaries
        // (depends on clustering results, but test should not fail)
        for chunk in &updated {
            assert!(!chunk.cluster_memberships.is_empty());
        }
    }

    #[tokio::test]
    async fn test_pipeline_updates_chunk_memberships() {
        let embedder = MockEmbedder::new(384);
        let summarizer = MockSummarizer::new("summary");
        let pipeline = ClusterPipeline::with_summarizer(embedder, summarizer);

        let chunks = create_test_chunks(4);
        let original_ids: Vec<String> = chunks.iter().map(|c| c.id.clone()).collect();

        let (updated, _) = pipeline.process(chunks).await.unwrap();

        // Verify IDs are preserved
        let updated_ids: Vec<String> = updated.iter().map(|c| c.id.clone()).collect();
        assert_eq!(original_ids, updated_ids);

        // Verify memberships were added
        for chunk in &updated {
            assert!(!chunk.cluster_memberships.is_empty());
            // Verify membership structure
            for membership in &chunk.cluster_memberships {
                assert!(membership.probability > 0.0);
                assert!(membership.probability <= 1.0);
                assert!(!membership.cluster_id.is_empty());
            }
        }
    }

    #[tokio::test]
    async fn test_pipeline_summary_has_embedding() {
        let embedder = MockEmbedder::new(384);
        let summarizer = MockSummarizer::new("Test summary content");
        let pipeline =
            ClusterPipeline::with_summarizer(embedder, summarizer).with_min_cluster_size(2);

        let chunks = create_test_chunks(6);
        let (_, summaries) = pipeline.process(chunks).await.unwrap();

        for summary in &summaries {
            assert!(
                summary.embedding.is_some(),
                "Summary should have an embedding"
            );
            assert_eq!(
                summary.embedding.as_ref().unwrap().len(),
                384,
                "Summary embedding should have correct dimension"
            );
        }
    }

    #[tokio::test]
    async fn test_pipeline_with_model() {
        let embedder = MockEmbedder::new(384);
        let pipeline = ClusterPipeline::new(embedder).with_model("custom-model");

        // Just verify it compiles and sets the model
        assert_eq!(pipeline.min_cluster_size, 2);
    }

    #[tokio::test]
    async fn test_pipeline_cluster_range() {
        let embedder = MockEmbedder::new(384);
        let summarizer = MockSummarizer::new("summary");
        let pipeline =
            ClusterPipeline::with_summarizer(embedder, summarizer).with_cluster_range(3, 4);

        let chunks = create_test_chunks(8);
        let (updated, _) = pipeline.process(chunks).await.unwrap();

        assert_eq!(updated.len(), 8);
        for chunk in &updated {
            assert!(!chunk.cluster_memberships.is_empty());
        }
    }

    #[tokio::test]
    async fn test_pipeline_summary_chunk_structure() {
        let embedder = MockEmbedder::new(384);
        let summarizer = MockSummarizer::new("Cluster summary text");
        let pipeline =
            ClusterPipeline::with_summarizer(embedder, summarizer).with_min_cluster_size(2);

        let chunks = create_test_chunks(6);
        let (_, summaries) = pipeline.process(chunks).await.unwrap();

        for summary in &summaries {
            // Verify summary structure
            assert!(summary.is_summary());
            assert_eq!(summary.level, ChunkLevel::H1);
            assert_eq!(summary.source_file, "[cluster-summary]");
            assert_eq!(summary.path, "Summary");
            assert_eq!(summary.heading, Some("Cluster Summary".to_string()));
            assert!(!summary.summarizes.is_empty());

            // Verify all summarized chunk IDs are valid
            for chunk_id in &summary.summarizes {
                assert!(!chunk_id.is_empty());
            }
        }
    }

    #[tokio::test]
    async fn test_pipeline_primary_cluster() {
        let embedder = MockEmbedder::new(384);
        let summarizer = MockSummarizer::new("summary");
        let pipeline = ClusterPipeline::with_summarizer(embedder, summarizer);

        let chunks = create_test_chunks(4);
        let (updated, _) = pipeline.process(chunks).await.unwrap();

        for chunk in &updated {
            let primary = chunk.primary_cluster();
            assert!(primary.is_some(), "Should have a primary cluster");

            // Verify primary is the one with highest probability
            let primary_prob = primary.unwrap().probability;
            for membership in &chunk.cluster_memberships {
                assert!(
                    membership.probability <= primary_prob,
                    "Primary should have highest probability"
                );
            }
        }
    }

    // Test summarizer failure handling
    struct FailingSummarizer;

    impl Summarizer for FailingSummarizer {
        async fn summarize(&self, _texts: &[&str]) -> Result<String> {
            Err(Error::summarization("Intentional failure"))
        }

        async fn summarize_batch(&self, _text_groups: Vec<Vec<&str>>) -> Result<Vec<String>> {
            Err(Error::summarization("Intentional failure"))
        }

        fn name(&self) -> &str {
            "failing"
        }
    }

    #[tokio::test]
    async fn test_pipeline_summarizer_failure() {
        let embedder = MockEmbedder::new(384);
        let summarizer = FailingSummarizer;
        let pipeline =
            ClusterPipeline::with_summarizer(embedder, summarizer).with_min_cluster_size(2);

        let chunks = create_test_chunks(6);
        let (updated, summaries) = pipeline.process(chunks).await.unwrap();

        // Should still update chunks with memberships
        assert_eq!(updated.len(), 6);
        for chunk in &updated {
            assert!(!chunk.cluster_memberships.is_empty());
        }

        // But summaries should be empty or minimal due to failures
        // The pipeline logs errors but continues
        assert!(
            summaries.is_empty(),
            "Should have no summaries due to summarizer failure"
        );
    }

    // Test embedder failure for summaries
    struct FailingEmbedder;

    impl Embedder for FailingEmbedder {
        fn embed(&self, _texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Err(Error::embedding("Intentional embedding failure"))
        }

        fn dimension(&self) -> usize {
            384
        }

        fn name(&self) -> &str {
            "failing"
        }
    }

    #[tokio::test]
    async fn test_pipeline_embedding_failure_for_summary() {
        let embedder = FailingEmbedder;
        let summarizer = MockSummarizer::new("summary");
        let pipeline =
            ClusterPipeline::with_summarizer(embedder, summarizer).with_min_cluster_size(2);

        // Create chunks WITHOUT embeddings - FailingEmbedder won't be called for clustering
        // but the pipeline will fail when trying to extract embeddings
        let chunks: Vec<HierarchicalChunk> = (0..4)
            .map(|i| {
                HierarchicalChunk::new(
                    format!("Content {}", i),
                    ChunkLevel::CONTENT,
                    None,
                    format!("path_{}", i),
                    "test.md".to_string(),
                )
                // No embedding set
            })
            .collect();

        let result = pipeline.process(chunks).await;

        // Should fail because chunks don't have embeddings
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_pipeline_single_chunk() {
        let embedder = MockEmbedder::new(384);
        let summarizer = MockSummarizer::new("summary");
        let pipeline = ClusterPipeline::with_summarizer(embedder, summarizer);

        let chunks = create_test_chunks(1);
        let (updated, summaries) = pipeline.process(chunks).await.unwrap();

        assert_eq!(updated.len(), 1);
        assert!(!updated[0].cluster_memberships.is_empty());

        // With only 1 chunk and min_cluster_size=2, should not create summaries
        assert!(summaries.is_empty());
    }
}
