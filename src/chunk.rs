use serde::{Deserialize, Serialize};

/// Soft cluster membership: probability of belonging to a cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterMembership {
    /// Cluster identifier
    pub cluster_id: String,
    /// Probability of belonging to this cluster (0.0 - 1.0)
    pub probability: f32,
}

impl ClusterMembership {
    pub fn new(cluster_id: impl Into<String>, probability: f32) -> Self {
        Self {
            cluster_id: cluster_id.into(),
            probability: probability.clamp(0.0, 1.0),
        }
    }
}

/// Represents the hierarchical level of a chunk in the document structure.
/// H1 = 1, H2 = 2, ..., H6 = 6, and content paragraphs under headings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ChunkLevel(pub u8);

impl ChunkLevel {
    pub const H1: Self = Self(1);
    pub const H2: Self = Self(2);
    pub const H3: Self = Self(3);
    pub const H4: Self = Self(4);
    pub const H5: Self = Self(5);
    pub const H6: Self = Self(6);
    pub const CONTENT: Self = Self(7);

    pub fn from_heading(level: u8) -> Self {
        Self(level.clamp(1, 6))
    }

    pub fn is_heading(&self) -> bool {
        self.0 <= 6
    }

    pub fn is_content(&self) -> bool {
        self.0 == 7
    }

    pub fn depth(&self) -> u8 {
        self.0
    }
}

impl std::fmt::Display for ChunkLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_heading() {
            write!(f, "H{}", self.0)
        } else {
            write!(f, "Content")
        }
    }
}

/// A hierarchical chunk representing a section of a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalChunk {
    /// Unique identifier for this chunk
    pub id: String,

    /// The text content of this chunk
    pub content: String,

    /// The embedding vector (populated after embedding)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,

    /// Hierarchical level (H1=1, H2=2, ..., Content=7)
    pub level: ChunkLevel,

    /// ID of the parent chunk (None for root-level chunks)
    pub parent_id: Option<String>,

    /// Full path in the hierarchy (e.g., "Chapter 1 > Section 1.1 > Details")
    pub path: String,

    /// Source file this chunk came from
    pub source_file: String,

    /// The heading text for this section (if this is a heading chunk)
    pub heading: Option<String>,

    /// Start position in the source document
    pub start_offset: usize,

    /// End position in the source document
    pub end_offset: usize,

    /// Soft cluster memberships (RAPTOR-style)
    /// A chunk can belong to multiple clusters with different probabilities
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cluster_memberships: Vec<ClusterMembership>,

    /// Whether this chunk is an AI-generated summary of a cluster
    #[serde(default)]
    pub is_summary: bool,

    /// If this is a summary chunk, IDs of chunks it summarizes
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub summarizes: Vec<String>,
}

impl HierarchicalChunk {
    /// Create a new chunk with a generated UUID
    pub fn new(
        content: String,
        level: ChunkLevel,
        parent_id: Option<String>,
        path: String,
        source_file: String,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content,
            embedding: None,
            level,
            parent_id,
            path,
            source_file,
            heading: None,
            start_offset: 0,
            end_offset: 0,
            cluster_memberships: Vec::new(),
            is_summary: false,
            summarizes: Vec::new(),
        }
    }

    /// Create a summary chunk for a cluster
    pub fn new_summary(
        content: String,
        summarized_chunk_ids: Vec<String>,
        embedding: Option<Vec<f32>>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content,
            embedding,
            level: ChunkLevel::H1, // Summaries are top-level abstractions
            parent_id: None,
            path: "Summary".to_string(),
            source_file: "[cluster-summary]".to_string(),
            heading: Some("Cluster Summary".to_string()),
            start_offset: 0,
            end_offset: 0,
            cluster_memberships: Vec::new(),
            is_summary: true,
            summarizes: summarized_chunk_ids,
        }
    }

    /// Set the heading for this chunk
    pub fn with_heading(mut self, heading: impl Into<String>) -> Self {
        self.heading = Some(heading.into());
        self
    }

    /// Set the document offsets
    pub fn with_offsets(mut self, start: usize, end: usize) -> Self {
        self.start_offset = start;
        self.end_offset = end;
        self
    }

    /// Set the embedding vector
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Add a cluster membership
    pub fn with_cluster_membership(
        mut self,
        cluster_id: impl Into<String>,
        probability: f32,
    ) -> Self {
        self.cluster_memberships
            .push(ClusterMembership::new(cluster_id, probability));
        self
    }

    /// Set cluster memberships
    pub fn with_cluster_memberships(mut self, memberships: Vec<ClusterMembership>) -> Self {
        self.cluster_memberships = memberships;
        self
    }

    /// Get primary cluster (highest probability)
    pub fn primary_cluster(&self) -> Option<&ClusterMembership> {
        self.cluster_memberships.iter().max_by(|a, b| {
            a.probability
                .partial_cmp(&b.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Check if this chunk has an embedding
    pub fn has_embedding(&self) -> bool {
        self.embedding.is_some()
    }

    /// Get the embedding dimension (returns 0 if no embedding)
    pub fn embedding_dim(&self) -> usize {
        self.embedding.as_ref().map(|e| e.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_level() {
        assert!(ChunkLevel::H1.is_heading());
        assert!(ChunkLevel::H6.is_heading());
        assert!(!ChunkLevel::CONTENT.is_heading());
        assert!(ChunkLevel::CONTENT.is_content());
    }

    #[test]
    fn test_chunk_creation() {
        let chunk = HierarchicalChunk::new(
            "Test content".to_string(),
            ChunkLevel::H1,
            None,
            "Test".to_string(),
            "test.md".to_string(),
        );

        assert!(!chunk.id.is_empty());
        assert_eq!(chunk.content, "Test content");
        assert_eq!(chunk.level, ChunkLevel::H1);
        assert!(chunk.parent_id.is_none());
    }

    #[test]
    fn test_chunk_level_from_heading() {
        assert_eq!(ChunkLevel::from_heading(1), ChunkLevel::H1);
        assert_eq!(ChunkLevel::from_heading(6), ChunkLevel::H6);
        // Test clamping
        assert_eq!(ChunkLevel::from_heading(0), ChunkLevel::H1);
        assert_eq!(ChunkLevel::from_heading(10), ChunkLevel::H6);
    }

    #[test]
    fn test_chunk_level_display() {
        assert_eq!(format!("{}", ChunkLevel::H1), "H1");
        assert_eq!(format!("{}", ChunkLevel::H6), "H6");
        assert_eq!(format!("{}", ChunkLevel::CONTENT), "Content");
    }

    #[test]
    fn test_chunk_level_depth() {
        assert_eq!(ChunkLevel::H1.depth(), 1);
        assert_eq!(ChunkLevel::CONTENT.depth(), 7);
    }

    #[test]
    fn test_cluster_membership_new() {
        let membership = ClusterMembership::new("cluster_1", 0.8);
        assert_eq!(membership.cluster_id, "cluster_1");
        assert!((membership.probability - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_cluster_membership_probability_clamping() {
        let low = ClusterMembership::new("c", -0.5);
        assert_eq!(low.probability, 0.0);

        let high = ClusterMembership::new("c", 1.5);
        assert_eq!(high.probability, 1.0);
    }

    #[test]
    fn test_chunk_with_heading() {
        let chunk = HierarchicalChunk::new(
            "content".to_string(),
            ChunkLevel::H1,
            None,
            "path".to_string(),
            "file.md".to_string(),
        )
        .with_heading("Test Heading");

        assert_eq!(chunk.heading, Some("Test Heading".to_string()));
    }

    #[test]
    fn test_chunk_with_offsets() {
        let chunk = HierarchicalChunk::new(
            "content".to_string(),
            ChunkLevel::CONTENT,
            None,
            "path".to_string(),
            "file.md".to_string(),
        )
        .with_offsets(10, 50);

        assert_eq!(chunk.start_offset, 10);
        assert_eq!(chunk.end_offset, 50);
    }

    #[test]
    fn test_chunk_with_embedding() {
        let embedding = vec![0.1, 0.2, 0.3];
        let chunk = HierarchicalChunk::new(
            "content".to_string(),
            ChunkLevel::CONTENT,
            None,
            "path".to_string(),
            "file.md".to_string(),
        )
        .with_embedding(embedding.clone());

        assert_eq!(chunk.embedding, Some(embedding));
        assert!(chunk.has_embedding());
        assert_eq!(chunk.embedding_dim(), 3);
    }

    #[test]
    fn test_chunk_no_embedding() {
        let chunk = HierarchicalChunk::new(
            "content".to_string(),
            ChunkLevel::CONTENT,
            None,
            "path".to_string(),
            "file.md".to_string(),
        );

        assert!(!chunk.has_embedding());
        assert_eq!(chunk.embedding_dim(), 0);
    }

    #[test]
    fn test_chunk_with_cluster_membership() {
        let chunk = HierarchicalChunk::new(
            "content".to_string(),
            ChunkLevel::CONTENT,
            None,
            "path".to_string(),
            "file.md".to_string(),
        )
        .with_cluster_membership("cluster_0", 0.9)
        .with_cluster_membership("cluster_1", 0.3);

        assert_eq!(chunk.cluster_memberships.len(), 2);
        assert_eq!(chunk.cluster_memberships[0].cluster_id, "cluster_0");
    }

    #[test]
    fn test_chunk_with_cluster_memberships() {
        let memberships = vec![
            ClusterMembership::new("c1", 0.8),
            ClusterMembership::new("c2", 0.6),
        ];

        let chunk = HierarchicalChunk::new(
            "content".to_string(),
            ChunkLevel::CONTENT,
            None,
            "path".to_string(),
            "file.md".to_string(),
        )
        .with_cluster_memberships(memberships);

        assert_eq!(chunk.cluster_memberships.len(), 2);
    }

    #[test]
    fn test_chunk_primary_cluster() {
        let chunk = HierarchicalChunk::new(
            "content".to_string(),
            ChunkLevel::CONTENT,
            None,
            "path".to_string(),
            "file.md".to_string(),
        )
        .with_cluster_membership("low", 0.2)
        .with_cluster_membership("high", 0.9)
        .with_cluster_membership("mid", 0.5);

        let primary = chunk.primary_cluster().unwrap();
        assert_eq!(primary.cluster_id, "high");
    }

    #[test]
    fn test_chunk_primary_cluster_empty() {
        let chunk = HierarchicalChunk::new(
            "content".to_string(),
            ChunkLevel::CONTENT,
            None,
            "path".to_string(),
            "file.md".to_string(),
        );

        assert!(chunk.primary_cluster().is_none());
    }

    #[test]
    fn test_new_summary() {
        let summary = HierarchicalChunk::new_summary(
            "Summary text".to_string(),
            vec!["chunk1".to_string(), "chunk2".to_string()],
            Some(vec![0.1, 0.2, 0.3]),
        );

        assert!(summary.is_summary);
        assert_eq!(summary.summarizes.len(), 2);
        assert_eq!(summary.source_file, "[cluster-summary]");
        assert_eq!(summary.level, ChunkLevel::H1);
        assert!(summary.embedding.is_some());
    }

    #[test]
    fn test_new_summary_without_embedding() {
        let summary =
            HierarchicalChunk::new_summary("Summary".to_string(), vec!["chunk1".to_string()], None);

        assert!(summary.is_summary);
        assert!(summary.embedding.is_none());
    }
}
