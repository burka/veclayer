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
    use crate::embedder::FastEmbedder;

    // Note: Integration tests would require Ollama running
    // These are just basic structural tests

    #[test]
    fn test_pipeline_creation() {
        let embedder = FastEmbedder::new().unwrap();
        let _pipeline = ClusterPipeline::new(embedder);
    }
}
