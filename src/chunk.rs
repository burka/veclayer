//! Core domain types for VecLayer's hierarchical chunk model.
//!
//! This module defines [`HierarchicalChunk`] — the primary in-memory data unit —
//! along with [`EntryType`], [`ChunkLevel`], [`ChunkRelation`], and [`ClusterMembership`].
//! Visibility and relation constants live in sub-modules [`visibility`] and [`relation`].
//!
//! **Planned refactoring:** `ChunkRelation` and relation constants will be extracted
//! to the `relations` module before v0.2 to reduce this module's scope.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// Re-export from the dedicated access_profile module
pub use crate::access_profile::{now_epoch_secs, AccessProfile, RecencyWindow};

/// Generate a content-hash ID from content using SHA-256.
/// Returns the full hex-encoded hash.
pub fn content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Return the short form of a content-hash ID (first 7 hex chars, like git).
pub fn short_id(id: &str) -> &str {
    if id.len() >= 7 {
        &id[..7]
    } else {
        id
    }
}

/// The type of an entry. Replaces the old `is_summary` boolean.
///
/// - `raw` -- Original data, unmodified (ingested text, file content)
/// - `summary` -- Generated summary of child entries
/// - `meta` -- Reflection, assessment, evaluation
/// - `impression` -- Spontaneous observation, quick note
///
/// # Examples
///
/// ```
/// use veclayer::EntryType;
///
/// let t = EntryType::Raw;
/// assert_eq!(t, EntryType::default());
///
/// let s = EntryType::Summary;
/// assert_ne!(t, s);
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum EntryType {
    #[default]
    Raw,
    Summary,
    Meta,
    Impression,
}

impl std::fmt::Display for EntryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Raw => write!(f, "raw"),
            Self::Summary => write!(f, "summary"),
            Self::Meta => write!(f, "meta"),
            Self::Impression => write!(f, "impression"),
        }
    }
}

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

/// Well-known visibility values.
///
/// Visibility is an open String field -- these are conventions, not a closed set.
/// Custom values (e.g. "draft", "review", "archived") work without code changes.
/// The search filter decides which visibilities are included.
pub mod visibility {
    pub const ALWAYS: &str = "always";
    pub const NORMAL: &str = "normal";
    pub const DEEP_ONLY: &str = "deep_only";
    pub const EXPIRING: &str = "expiring";
    pub const SEASONAL: &str = "seasonal";
}

/// Default set of visibilities included in a standard (non-deep) search.
/// Custom visibilities are excluded by default -- opt-in via search config.
pub const STANDARD_VISIBLE: &[&str] = &[
    visibility::ALWAYS,
    visibility::NORMAL,
    visibility::SEASONAL,
    visibility::EXPIRING,
];

/// Well-known relation kinds.
///
/// Relation kind is an open String field -- these are conventions.
/// Custom kinds (e.g. "contradicts", "inspired_by", "blocks") work without code changes.
pub mod relation {
    pub const SUPERSEDED_BY: &str = "superseded_by";
    pub const SUMMARIZED_BY: &str = "summarized_by";
    pub const RELATED_TO: &str = "related_to";
    pub const DERIVED_FROM: &str = "derived_from";
    pub const SUPERSEDES: &str = "supersedes";
    pub const SUMMARIZES: &str = "summarizes";
    pub const VERSION_OF: &str = "version_of";

    /// All well-known relation kinds (for typo detection / validation).
    pub const KNOWN_KINDS: &[&str] = &[
        SUPERSEDED_BY,
        SUMMARIZED_BY,
        RELATED_TO,
        DERIVED_FROM,
        SUPERSEDES,
        SUMMARIZES,
        VERSION_OF,
    ];
}

/// A directed relation from one chunk to another.
///
/// Design constraint: relations are NOT the primary search path.
/// You find a chunk via vector search, THEN navigate relations.
/// Max 1-2 hops, no graph traversal.
///
/// `kind` is an open string. Use constants from `relation::` for well-known
/// types, or any custom string for domain-specific relations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkRelation {
    pub kind: String,
    pub target_id: String,
}

impl ChunkRelation {
    pub fn new(kind: impl Into<String>, target_id: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            target_id: target_id.into(),
        }
    }

    pub fn superseded_by(target_id: impl Into<String>) -> Self {
        Self::new(relation::SUPERSEDED_BY, target_id)
    }

    pub fn summarized_by(target_id: impl Into<String>) -> Self {
        Self::new(relation::SUMMARIZED_BY, target_id)
    }

    pub fn related_to(target_id: impl Into<String>) -> Self {
        Self::new(relation::RELATED_TO, target_id)
    }

    pub fn derived_from(target_id: impl Into<String>) -> Self {
        Self::new(relation::DERIVED_FROM, target_id)
    }

    pub fn supersedes(target_id: impl Into<String>) -> Self {
        Self::new(relation::SUPERSEDES, target_id)
    }

    pub fn summarizes(target_id: impl Into<String>) -> Self {
        Self::new(relation::SUMMARIZES, target_id)
    }

    pub fn version_of(target_id: impl Into<String>) -> Self {
        Self::new(relation::VERSION_OF, target_id)
    }
}

fn default_visibility() -> String {
    visibility::NORMAL.to_string()
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
///
/// # Examples
///
/// ```
/// use veclayer::{HierarchicalChunk, ChunkLevel};
///
/// let chunk = HierarchicalChunk::new(
///     "Rust is fast and safe.".to_string(),
///     ChunkLevel::H1,
///     None,
///     "Guide".to_string(),
///     "guide.md".to_string(),
/// );
/// assert_eq!(chunk.level, ChunkLevel::H1);
/// assert!(!chunk.has_embedding());
/// ```
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

    /// Entry type: raw, summary, meta, or impression.
    #[serde(default)]
    pub entry_type: EntryType,

    /// If this is a summary entry, IDs of entries it summarizes
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub summarizes: Vec<String>,

    // --- Identity & Memory fields ---
    /// How this chunk should be treated in search and aging.
    /// Open string -- use constants from `visibility::` or any custom value.
    /// Default: "normal"
    #[serde(default = "default_visibility")]
    pub visibility: String,

    /// Perspectives this entry belongs to (e.g. "decisions", "learnings").
    /// An entry can belong to multiple perspectives.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub perspectives: Vec<String>,

    /// Relations to other chunks. Each relation has an open `kind` string
    /// and a `target_id`. Use constants from `relation::` or custom kinds.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relations: Vec<ChunkRelation>,

    /// Access tracking for memory aging
    #[serde(default)]
    pub access_profile: AccessProfile,

    /// Optional expiry timestamp for Expiring visibility (Unix epoch seconds)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,

    /// Impression hint: qualitative label like "uncertain", "confident", "exploratory".
    /// Only meaningful when `entry_type == Impression`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub impression_hint: Option<String>,

    /// Impression strength: [0.0, 1.0] modulating salience weight.
    /// 1.0 = full weight (default), 0.0 = negligible.
    /// Only meaningful when `entry_type == Impression`.
    #[serde(default = "default_impression_strength")]
    pub impression_strength: f32,
}

fn default_impression_strength() -> f32 {
    1.0
}

impl HierarchicalChunk {
    /// Create a new chunk with a content-hash ID (SHA-256).
    /// Identical content always produces the same ID (idempotent).
    pub fn new(
        content: String,
        level: ChunkLevel,
        parent_id: Option<String>,
        path: String,
        source_file: String,
    ) -> Self {
        let id = content_hash(&content);
        Self {
            id,
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
            entry_type: EntryType::Raw,
            summarizes: Vec::new(),
            perspectives: Vec::new(),
            visibility: default_visibility(),
            relations: Vec::new(),
            access_profile: AccessProfile::new(),
            expires_at: None,
            impression_hint: None,
            impression_strength: 1.0,
        }
    }

    /// Create a summary entry for a cluster.
    /// ID is derived from the content hash (idempotent).
    pub fn new_summary(
        content: String,
        summarized_chunk_ids: Vec<String>,
        embedding: Option<Vec<f32>>,
    ) -> Self {
        let id = content_hash(&content);
        Self {
            id,
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
            entry_type: EntryType::Summary,
            summarizes: summarized_chunk_ids,
            perspectives: Vec::new(),
            visibility: default_visibility(),
            relations: Vec::new(),
            access_profile: AccessProfile::new(),
            expires_at: None,
            impression_hint: None,
            impression_strength: 1.0,
        }
    }

    /// Check if this entry is a summary.
    pub fn is_summary(&self) -> bool {
        self.entry_type == EntryType::Summary
    }

    /// Set the entry type
    pub fn with_entry_type(mut self, entry_type: EntryType) -> Self {
        self.entry_type = entry_type;
        self
    }

    /// Set perspectives this entry belongs to
    pub fn with_perspectives(mut self, perspectives: Vec<String>) -> Self {
        self.perspectives = perspectives;
        self
    }

    /// Add a single perspective
    pub fn with_perspective(mut self, perspective: impl Into<String>) -> Self {
        self.perspectives.push(perspective.into());
        self
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

    /// Set the visibility (any string -- use `visibility::` constants or custom values)
    pub fn with_visibility(mut self, visibility: impl Into<String>) -> Self {
        self.visibility = visibility.into();
        self
    }

    /// Add a relation to another chunk
    pub fn with_relation(mut self, relation: ChunkRelation) -> Self {
        self.relations.push(relation);
        self
    }

    /// Set the expiry timestamp. Sets visibility to "expiring" automatically.
    pub fn with_expires_at(mut self, expires_at: i64) -> Self {
        self.visibility = visibility::EXPIRING.to_string();
        self.expires_at = Some(expires_at);
        self
    }

    /// Check if this chunk has expired (has an expires_at in the past)
    pub fn is_expired(&self) -> bool {
        match self.expires_at {
            Some(expires) => now_epoch_secs() >= expires,
            None => false,
        }
    }

    /// Check if this chunk is visible for a standard (non-deep) search.
    /// Uses `STANDARD_VISIBLE` set. Expired chunks are excluded.
    /// Custom visibilities not in that set are excluded by default.
    pub fn is_visible_standard(&self) -> bool {
        if self.is_expired() {
            return false;
        }
        STANDARD_VISIBLE.contains(&self.visibility.as_str())
    }

    /// Get relations of a specific kind (string match)
    pub fn relations_of_kind(&self, kind: &str) -> Vec<&ChunkRelation> {
        self.relations.iter().filter(|r| r.kind == kind).collect()
    }

    /// Check if this chunk has been superseded
    pub fn is_superseded(&self) -> bool {
        self.relations
            .iter()
            .any(|r| r.kind == relation::SUPERSEDED_BY)
    }

    /// Get primary cluster (highest probability)
    pub fn primary_cluster(&self) -> Option<&ClusterMembership> {
        self.cluster_memberships.iter().max_by(|a, b| {
            a.probability
                .partial_cmp(&b.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Return a clone of this chunk with the embedding cleared.
    /// Used for JSONL export: embeddings are large and regenerated on import.
    pub fn without_embedding(mut self) -> Self {
        self.embedding = None;
        self
    }

    /// Check if this chunk has an embedding
    pub fn has_embedding(&self) -> bool {
        self.embedding.is_some()
    }

    /// Get the embedding dimension (returns 0 if no embedding)
    pub fn embedding_dim(&self) -> usize {
        self.embedding.as_ref().map(|e| e.len()).unwrap_or(0)
    }

    /// Materialize a `HierarchicalChunk` from an `Entry` and a pre-computed embedding.
    pub fn from_entry(entry: &crate::entry::Entry, embedding: Vec<f32>) -> Self {
        Self {
            id: entry.content_id(),
            content: entry.content.clone(),
            embedding: Some(embedding),
            level: entry.level,
            parent_id: entry.parent_id.clone(),
            path: entry.path.clone(),
            source_file: entry.source.clone(),
            heading: entry.heading.clone(),
            start_offset: 0,
            end_offset: 0,
            cluster_memberships: Vec::new(),
            entry_type: entry.entry_type,
            summarizes: entry.summarizes.clone(),
            visibility: entry.visibility.clone(),
            perspectives: entry.perspectives.clone(),
            relations: entry.relations.clone(),
            access_profile: AccessProfile::new(),
            expires_at: entry.expires_at,
            impression_hint: entry.impression_hint.clone(),
            impression_strength: entry.impression_strength,
        }
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

        assert_eq!(chunk.id, content_hash("Test content"));
        assert_eq!(chunk.content, "Test content");
        assert_eq!(chunk.level, ChunkLevel::H1);
        assert!(chunk.parent_id.is_none());
    }

    #[test]
    fn test_content_hash_deterministic() {
        let id1 = content_hash("hello world");
        let id2 = content_hash("hello world");
        assert_eq!(id1, id2);

        let id3 = content_hash("hello world!");
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_content_hash_is_sha256_hex() {
        let id = content_hash("test");
        assert_eq!(id.len(), 64);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_short_id() {
        let id = content_hash("test");
        assert_eq!(short_id(&id).len(), 7);
        assert_eq!(short_id(&id), &id[..7]);
    }

    #[test]
    fn test_idempotent_chunk_creation() {
        let chunk1 = HierarchicalChunk::new(
            "Same content".to_string(),
            ChunkLevel::H1,
            None,
            "path1".to_string(),
            "file1.md".to_string(),
        );
        let chunk2 = HierarchicalChunk::new(
            "Same content".to_string(),
            ChunkLevel::H2,
            Some("parent".to_string()),
            "path2".to_string(),
            "file2.md".to_string(),
        );
        assert_eq!(chunk1.id, chunk2.id);
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

        assert!(summary.is_summary());
        assert_eq!(summary.entry_type, EntryType::Summary);
        assert_eq!(summary.summarizes.len(), 2);
        assert_eq!(summary.source_file, "[cluster-summary]");
        assert_eq!(summary.level, ChunkLevel::H1);
        assert!(summary.embedding.is_some());
    }

    #[test]
    fn test_new_summary_without_embedding() {
        let summary =
            HierarchicalChunk::new_summary("Summary".to_string(), vec!["chunk1".to_string()], None);

        assert!(summary.is_summary());
        assert!(summary.embedding.is_none());
    }

    #[test]
    fn test_entry_type_default_is_raw() {
        let chunk = HierarchicalChunk::new(
            "content".to_string(),
            ChunkLevel::CONTENT,
            None,
            "path".to_string(),
            "file.md".to_string(),
        );
        assert_eq!(chunk.entry_type, EntryType::Raw);
        assert!(!chunk.is_summary());
    }

    #[test]
    fn test_entry_type_display() {
        assert_eq!(format!("{}", EntryType::Raw), "raw");
        assert_eq!(format!("{}", EntryType::Summary), "summary");
        assert_eq!(format!("{}", EntryType::Meta), "meta");
        assert_eq!(format!("{}", EntryType::Impression), "impression");
    }

    // --- Visibility tests ---

    #[test]
    fn test_visibility_default() {
        let chunk = HierarchicalChunk::new(
            "content".to_string(),
            ChunkLevel::CONTENT,
            None,
            "path".to_string(),
            "file.md".to_string(),
        );
        assert_eq!(chunk.visibility, "normal");
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
        .with_visibility(visibility::ALWAYS);

        assert_eq!(chunk.visibility, "always");
    }

    #[test]
    fn test_chunk_custom_visibility() {
        // Custom visibilities work without code changes
        let chunk = HierarchicalChunk::new(
            "draft doc".to_string(),
            ChunkLevel::H1,
            None,
            "path".to_string(),
            "file.md".to_string(),
        )
        .with_visibility("draft");

        assert_eq!(chunk.visibility, "draft");
        // Custom visibilities are NOT in standard search by default
        assert!(!chunk.is_visible_standard());
    }

    #[test]
    fn test_is_visible_standard() {
        let make = |vis: &str| {
            HierarchicalChunk::new(
                "".to_string(),
                ChunkLevel::H1,
                None,
                "".to_string(),
                "".to_string(),
            )
            .with_visibility(vis)
        };

        assert!(make(visibility::ALWAYS).is_visible_standard());
        assert!(make(visibility::NORMAL).is_visible_standard());
        assert!(make(visibility::SEASONAL).is_visible_standard());
        assert!(!make(visibility::DEEP_ONLY).is_visible_standard());
        assert!(!make("draft").is_visible_standard());
        assert!(!make("archived").is_visible_standard());
    }

    // --- Relation tests ---

    #[test]
    fn test_chunk_relation_constructors() {
        let r1 = ChunkRelation::superseded_by("new-id");
        assert_eq!(r1.kind, "superseded_by");
        assert_eq!(r1.target_id, "new-id");

        let r2 = ChunkRelation::related_to("other-id");
        assert_eq!(r2.kind, "related_to");

        let r3 = ChunkRelation::derived_from("source-id");
        assert_eq!(r3.kind, "derived_from");

        let r4 = ChunkRelation::summarized_by("summary-id");
        assert_eq!(r4.kind, "summarized_by");
    }

    #[test]
    fn test_custom_relation_kind() {
        // Custom relation kinds work without code changes
        let r = ChunkRelation::new("contradicts", "other-id");
        assert_eq!(r.kind, "contradicts");
        assert_eq!(r.target_id, "other-id");
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

        let related = chunk.relations_of_kind(relation::RELATED_TO);
        assert_eq!(related.len(), 2);

        let superseded = chunk.relations_of_kind(relation::SUPERSEDED_BY);
        assert_eq!(superseded.len(), 1);

        // Custom kinds work with relations_of_kind too
        let custom = chunk.relations_of_kind("contradicts");
        assert_eq!(custom.len(), 0);
    }

    #[test]
    fn test_relation_serde_roundtrip() {
        let relation = ChunkRelation::superseded_by("target-123");
        let json = serde_json::to_string(&relation).unwrap();
        let parsed: ChunkRelation = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.kind, "superseded_by");
        assert_eq!(parsed.target_id, "target-123");
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

        assert_eq!(chunk.visibility, "expiring");
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
