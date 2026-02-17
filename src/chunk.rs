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

// --- Identity & Memory Types ---

/// How a chunk should be treated in search and aging.
///
/// Data describes itself: an architecture decision is "Always" visible,
/// while an old chat log fades to "DeepOnly" over time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    /// Always visible, never degraded (core knowledge, architecture decisions)
    Always,
    /// Standard cascade, ages naturally with access patterns
    Normal,
    /// Only found during explicit deep search (old logs, discarded ideas)
    DeepOnly,
    /// Self-destructing after expires_at timestamp (temporary planning data)
    Expiring,
    /// Cyclically relevant, driven by access frequency (quarterly reports, tax season)
    Seasonal,
}

impl Default for Visibility {
    fn default() -> Self {
        Visibility::Normal
    }
}

impl std::fmt::Display for Visibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Visibility::Always => write!(f, "always"),
            Visibility::Normal => write!(f, "normal"),
            Visibility::DeepOnly => write!(f, "deep_only"),
            Visibility::Expiring => write!(f, "expiring"),
            Visibility::Seasonal => write!(f, "seasonal"),
        }
    }
}

impl std::str::FromStr for Visibility {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "always" => Ok(Visibility::Always),
            "normal" => Ok(Visibility::Normal),
            "deep_only" | "deep-only" | "deeponly" => Ok(Visibility::DeepOnly),
            "expiring" => Ok(Visibility::Expiring),
            "seasonal" => Ok(Visibility::Seasonal),
            _ => Err(format!("Unknown visibility: {}", s)),
        }
    }
}

/// Kind of relation between two chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationKind {
    /// This fact was replaced by newer information
    SupersededBy,
    /// Condensed into this summary node
    SummarizedBy,
    /// Loose thematic connection
    RelatedTo,
    /// Originated from this discussion/source
    DerivedFrom,
}

impl std::fmt::Display for RelationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RelationKind::SupersededBy => write!(f, "superseded_by"),
            RelationKind::SummarizedBy => write!(f, "summarized_by"),
            RelationKind::RelatedTo => write!(f, "related_to"),
            RelationKind::DerivedFrom => write!(f, "derived_from"),
        }
    }
}

/// A directed relation from one chunk to another.
///
/// Design constraint: relations are NOT the primary search path.
/// You find a chunk via vector search, THEN navigate relations.
/// Max 1-2 hops, no graph traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRelation {
    pub kind: RelationKind,
    pub target_id: String,
}

impl ChunkRelation {
    pub fn new(kind: RelationKind, target_id: impl Into<String>) -> Self {
        Self {
            kind,
            target_id: target_id.into(),
        }
    }

    pub fn superseded_by(target_id: impl Into<String>) -> Self {
        Self::new(RelationKind::SupersededBy, target_id)
    }

    pub fn summarized_by(target_id: impl Into<String>) -> Self {
        Self::new(RelationKind::SummarizedBy, target_id)
    }

    pub fn related_to(target_id: impl Into<String>) -> Self {
        Self::new(RelationKind::RelatedTo, target_id)
    }

    pub fn derived_from(target_id: impl Into<String>) -> Self {
        Self::new(RelationKind::DerivedFrom, target_id)
    }
}

/// Basic access tracking for memory aging.
///
/// This is the foundation for RRD-style bucketed access profiles.
/// For now: simple counters. Later: fixed-size time buckets (1min, 10min, 1h, 24h, 7d, 30d, total).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessProfile {
    /// When this chunk was created (Unix epoch seconds)
    pub created_at: i64,
    /// When this chunk was last accessed (Unix epoch seconds)
    pub last_accessed: i64,
    /// Total number of times this chunk has been accessed
    pub access_count: u32,
}

impl AccessProfile {
    pub fn new() -> Self {
        let now = now_epoch_secs();
        Self {
            created_at: now,
            last_accessed: now,
            access_count: 0,
        }
    }

    /// Record an access, updating last_accessed and incrementing count
    pub fn record_access(&mut self) {
        self.last_accessed = now_epoch_secs();
        self.access_count = self.access_count.saturating_add(1);
    }

    /// Seconds since last access
    pub fn seconds_since_access(&self) -> i64 {
        now_epoch_secs() - self.last_accessed
    }

    /// Seconds since creation
    pub fn age_seconds(&self) -> i64 {
        now_epoch_secs() - self.created_at
    }
}

impl Default for AccessProfile {
    fn default() -> Self {
        Self::new()
    }
}

fn now_epoch_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
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
///
/// This is the core data unit of VecLayer. Each chunk carries its content,
/// its position in the hierarchy, and metadata about how it should be treated
/// (visibility, relations, access patterns).
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

    // --- Identity & Memory fields ---
    /// How this chunk should be treated in search and aging
    #[serde(default)]
    pub visibility: Visibility,

    /// Relations to other chunks (SupersededBy, RelatedTo, etc.)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relations: Vec<ChunkRelation>,

    /// Access tracking for memory aging
    #[serde(default)]
    pub access_profile: AccessProfile,

    /// Optional expiry timestamp for Expiring visibility (Unix epoch seconds)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
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
            visibility: Visibility::default(),
            relations: Vec::new(),
            access_profile: AccessProfile::new(),
            expires_at: None,
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
            visibility: Visibility::default(),
            relations: Vec::new(),
            access_profile: AccessProfile::new(),
            expires_at: None,
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

    /// Set the visibility level
    pub fn with_visibility(mut self, visibility: Visibility) -> Self {
        self.visibility = visibility;
        self
    }

    /// Add a relation to another chunk
    pub fn with_relation(mut self, relation: ChunkRelation) -> Self {
        self.relations.push(relation);
        self
    }

    /// Set the expiry timestamp (for Expiring visibility)
    pub fn with_expires_at(mut self, expires_at: i64) -> Self {
        self.visibility = Visibility::Expiring;
        self.expires_at = Some(expires_at);
        self
    }

    /// Check if this chunk has expired
    pub fn is_expired(&self) -> bool {
        if self.visibility != Visibility::Expiring {
            return false;
        }
        match self.expires_at {
            Some(expires) => now_epoch_secs() >= expires,
            None => false,
        }
    }

    /// Check if this chunk is visible for a standard (non-deep) search
    pub fn is_visible_standard(&self) -> bool {
        match self.visibility {
            Visibility::Always | Visibility::Normal | Visibility::Seasonal => true,
            Visibility::DeepOnly => false,
            Visibility::Expiring => !self.is_expired(),
        }
    }

    /// Get relations of a specific kind
    pub fn relations_of_kind(&self, kind: RelationKind) -> Vec<&ChunkRelation> {
        self.relations.iter().filter(|r| r.kind == kind).collect()
    }

    /// Check if this chunk has been superseded
    pub fn is_superseded(&self) -> bool {
        self.relations
            .iter()
            .any(|r| r.kind == RelationKind::SupersededBy)
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

    // --- Visibility tests ---

    #[test]
    fn test_visibility_default() {
        assert_eq!(Visibility::default(), Visibility::Normal);
    }

    #[test]
    fn test_visibility_display() {
        assert_eq!(format!("{}", Visibility::Always), "always");
        assert_eq!(format!("{}", Visibility::Normal), "normal");
        assert_eq!(format!("{}", Visibility::DeepOnly), "deep_only");
        assert_eq!(format!("{}", Visibility::Expiring), "expiring");
        assert_eq!(format!("{}", Visibility::Seasonal), "seasonal");
    }

    #[test]
    fn test_visibility_from_str() {
        assert_eq!("always".parse::<Visibility>().unwrap(), Visibility::Always);
        assert_eq!("normal".parse::<Visibility>().unwrap(), Visibility::Normal);
        assert_eq!(
            "deep_only".parse::<Visibility>().unwrap(),
            Visibility::DeepOnly
        );
        assert_eq!(
            "deep-only".parse::<Visibility>().unwrap(),
            Visibility::DeepOnly
        );
        assert_eq!(
            "deeponly".parse::<Visibility>().unwrap(),
            Visibility::DeepOnly
        );
        assert_eq!(
            "expiring".parse::<Visibility>().unwrap(),
            Visibility::Expiring
        );
        assert_eq!(
            "seasonal".parse::<Visibility>().unwrap(),
            Visibility::Seasonal
        );
        assert!("unknown".parse::<Visibility>().is_err());
    }

    #[test]
    fn test_visibility_serde_roundtrip() {
        let json = serde_json::to_string(&Visibility::DeepOnly).unwrap();
        assert_eq!(json, "\"deep_only\"");
        let parsed: Visibility = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, Visibility::DeepOnly);
    }

    #[test]
    fn test_chunk_with_visibility() {
        let chunk = HierarchicalChunk::new(
            "core decision".to_string(),
            ChunkLevel::H1,
            None,
            "path".to_string(),
            "file.md".to_string(),
        )
        .with_visibility(Visibility::Always);

        assert_eq!(chunk.visibility, Visibility::Always);
    }

    #[test]
    fn test_chunk_default_visibility() {
        let chunk = HierarchicalChunk::new(
            "content".to_string(),
            ChunkLevel::CONTENT,
            None,
            "path".to_string(),
            "file.md".to_string(),
        );

        assert_eq!(chunk.visibility, Visibility::Normal);
    }

    #[test]
    fn test_is_visible_standard() {
        let always = HierarchicalChunk::new(
            "".to_string(),
            ChunkLevel::H1,
            None,
            "".to_string(),
            "".to_string(),
        )
        .with_visibility(Visibility::Always);
        assert!(always.is_visible_standard());

        let normal = HierarchicalChunk::new(
            "".to_string(),
            ChunkLevel::H1,
            None,
            "".to_string(),
            "".to_string(),
        )
        .with_visibility(Visibility::Normal);
        assert!(normal.is_visible_standard());

        let deep = HierarchicalChunk::new(
            "".to_string(),
            ChunkLevel::H1,
            None,
            "".to_string(),
            "".to_string(),
        )
        .with_visibility(Visibility::DeepOnly);
        assert!(!deep.is_visible_standard());
    }

    // --- Relation tests ---

    #[test]
    fn test_chunk_relation_constructors() {
        let r1 = ChunkRelation::superseded_by("new-id");
        assert_eq!(r1.kind, RelationKind::SupersededBy);
        assert_eq!(r1.target_id, "new-id");

        let r2 = ChunkRelation::related_to("other-id");
        assert_eq!(r2.kind, RelationKind::RelatedTo);

        let r3 = ChunkRelation::derived_from("source-id");
        assert_eq!(r3.kind, RelationKind::DerivedFrom);

        let r4 = ChunkRelation::summarized_by("summary-id");
        assert_eq!(r4.kind, RelationKind::SummarizedBy);
    }

    #[test]
    fn test_chunk_with_relation() {
        let chunk = HierarchicalChunk::new(
            "old fact".to_string(),
            ChunkLevel::CONTENT,
            None,
            "path".to_string(),
            "file.md".to_string(),
        )
        .with_relation(ChunkRelation::superseded_by("newer-fact"));

        assert_eq!(chunk.relations.len(), 1);
        assert!(chunk.is_superseded());
    }

    #[test]
    fn test_relations_of_kind() {
        let chunk = HierarchicalChunk::new(
            "content".to_string(),
            ChunkLevel::CONTENT,
            None,
            "path".to_string(),
            "file.md".to_string(),
        )
        .with_relation(ChunkRelation::related_to("a"))
        .with_relation(ChunkRelation::superseded_by("b"))
        .with_relation(ChunkRelation::related_to("c"));

        let related = chunk.relations_of_kind(RelationKind::RelatedTo);
        assert_eq!(related.len(), 2);

        let superseded = chunk.relations_of_kind(RelationKind::SupersededBy);
        assert_eq!(superseded.len(), 1);
    }

    #[test]
    fn test_relation_serde_roundtrip() {
        let relation = ChunkRelation::superseded_by("target-123");
        let json = serde_json::to_string(&relation).unwrap();
        let parsed: ChunkRelation = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.kind, RelationKind::SupersededBy);
        assert_eq!(parsed.target_id, "target-123");
    }

    // --- AccessProfile tests ---

    #[test]
    fn test_access_profile_new() {
        let profile = AccessProfile::new();
        assert_eq!(profile.access_count, 0);
        assert!(profile.created_at > 0);
        assert!(profile.last_accessed > 0);
    }

    #[test]
    fn test_access_profile_record_access() {
        let mut profile = AccessProfile::new();
        assert_eq!(profile.access_count, 0);

        profile.record_access();
        assert_eq!(profile.access_count, 1);

        profile.record_access();
        assert_eq!(profile.access_count, 2);
    }

    #[test]
    fn test_access_profile_default() {
        let profile = AccessProfile::default();
        assert_eq!(profile.access_count, 0);
        assert!(profile.created_at > 0);
    }

    // --- Expiring tests ---

    #[test]
    fn test_chunk_with_expires_at() {
        let past = now_epoch_secs() - 3600; // 1 hour ago
        let chunk = HierarchicalChunk::new(
            "temp".to_string(),
            ChunkLevel::CONTENT,
            None,
            "path".to_string(),
            "file.md".to_string(),
        )
        .with_expires_at(past);

        assert_eq!(chunk.visibility, Visibility::Expiring);
        assert!(chunk.is_expired());
        assert!(!chunk.is_visible_standard());
    }

    #[test]
    fn test_chunk_not_expired() {
        let future = now_epoch_secs() + 3600; // 1 hour from now
        let chunk = HierarchicalChunk::new(
            "temp".to_string(),
            ChunkLevel::CONTENT,
            None,
            "path".to_string(),
            "file.md".to_string(),
        )
        .with_expires_at(future);

        assert!(!chunk.is_expired());
        assert!(chunk.is_visible_standard());
    }

    #[test]
    fn test_non_expiring_never_expired() {
        let chunk = HierarchicalChunk::new(
            "normal".to_string(),
            ChunkLevel::CONTENT,
            None,
            "path".to_string(),
            "file.md".to_string(),
        );

        assert!(!chunk.is_expired());
    }
}
