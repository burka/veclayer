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
}

// --- Time constants for RRD bucket boundaries ---

const SECS_PER_HOUR: i64 = 3_600;
const SECS_PER_DAY: i64 = 86_400;
const SECS_PER_WEEK: i64 = 604_800;
const SECS_PER_MONTH: i64 = 2_592_000; // 30 days
const SECS_PER_YEAR: i64 = 31_536_000; // 365 days

/// RRD-style access tracking with fixed time-window buckets.
///
/// Inspired by RRDtool: finer buckets roll into coarser ones on a schedule.
/// Each bucket tracks the number of accesses within its time window.
///
/// Buckets: hour | day | week | month | year | total
///
/// Layout (30 bytes, padded to 32):
///   created_at:  i64  (8 bytes) - when chunk was created
///   last_rolled: i64  (8 bytes) - when buckets were last maintained
///   hour:        u16  (2 bytes) - accesses in the current 1-hour window
///   day:         u16  (2 bytes) - accesses in the current 24-hour window
///   week:        u16  (2 bytes) - accesses in the current 7-day window
///   month:       u16  (2 bytes) - accesses in the current 30-day window
///   year:        u16  (2 bytes) - accesses in the current 365-day window
///   total:       u32  (4 bytes) - all-time access count
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccessProfile {
    /// When this chunk was created (Unix epoch seconds). Immutable after creation.
    pub created_at: i64,
    /// When the buckets were last rolled/maintained (Unix epoch seconds).
    pub last_rolled: i64,
    /// Accesses in the current 1-hour window.
    pub hour: u16,
    /// Accesses in the current 24-hour window (includes rolled hour values).
    pub day: u16,
    /// Accesses in the current 7-day window (includes rolled day values).
    pub week: u16,
    /// Accesses in the current 30-day window (includes rolled week values).
    pub month: u16,
    /// Accesses in the current 365-day window (includes rolled month values).
    pub year: u16,
    /// All-time total access count. Never resets.
    pub total: u32,
}

/// Time window for recency-weighted search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecencyWindow {
    Day,
    Week,
    Month,
}

impl RecencyWindow {
    /// Parse from string (e.g. "24h", "7d", "30d").
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "24h" | "day" => Some(Self::Day),
            "7d" | "week" => Some(Self::Week),
            "30d" | "month" => Some(Self::Month),
            _ => None,
        }
    }
}

/// Internal weight configuration for relevancy scoring.
struct RecencyWeights {
    w_hour: f32,
    w_day: f32,
    w_week: f32,
    w_month: f32,
    w_year: f32,
    w_total: f32,
    scale: f32,
}

impl RecencyWeights {
    /// Balanced weighting. Recent access counts more, but total still matters.
    fn balanced() -> Self {
        Self {
            w_hour: 8.0,
            w_day: 4.0,
            w_week: 2.0,
            w_month: 1.0,
            w_year: 0.3,
            w_total: 0.1,
            scale: 10.0,
        }
    }

    /// Strongly favor today's accesses (--recent 24h).
    fn day() -> Self {
        Self {
            w_hour: 10.0,
            w_day: 8.0,
            w_week: 0.5,
            w_month: 0.1,
            w_year: 0.0,
            w_total: 0.0,
            scale: 10.0,
        }
    }

    /// Favor this week's accesses (--recent 7d).
    fn week() -> Self {
        Self {
            w_hour: 6.0,
            w_day: 5.0,
            w_week: 4.0,
            w_month: 0.5,
            w_year: 0.1,
            w_total: 0.05,
            scale: 10.0,
        }
    }

    /// Broader window (--recent 30d).
    fn month() -> Self {
        Self {
            w_hour: 4.0,
            w_day: 3.0,
            w_week: 2.5,
            w_month: 2.0,
            w_year: 0.5,
            w_total: 0.1,
            scale: 10.0,
        }
    }
}

impl AccessProfile {
    pub fn new() -> Self {
        let now = now_epoch_secs();
        Self {
            created_at: now,
            last_rolled: now,
            hour: 0,
            day: 0,
            week: 0,
            month: 0,
            year: 0,
            total: 0,
        }
    }

    /// Create an AccessProfile with a specific creation time (for testing/migration).
    pub fn with_created_at(created_at: i64) -> Self {
        Self {
            created_at,
            last_rolled: created_at,
            hour: 0,
            day: 0,
            week: 0,
            month: 0,
            year: 0,
            total: 0,
        }
    }

    /// Maintain bucket invariants by rolling stale values from finer
    /// buckets into coarser ones.
    ///
    /// When multiple time boundaries are crossed (e.g. 2 days since last roll),
    /// all stale buckets cascade into the next non-stale level. For example,
    /// if 2 days have elapsed, both `hour` and `day` values move into `week`.
    ///
    /// Must be called before any read or write of bucket values.
    /// Idempotent: calling it multiple times with the same `now` is safe.
    pub fn roll_up(&mut self, now: i64) {
        let elapsed = now - self.last_rolled;
        if elapsed <= 0 {
            return;
        }

        if elapsed >= SECS_PER_YEAR {
            // Everything is stale. All counts already exist in total.
            self.year = 0;
            self.month = 0;
            self.week = 0;
            self.day = 0;
            self.hour = 0;
            self.last_rolled = now;
        } else if elapsed >= SECS_PER_MONTH {
            // hour + day + week + month all cascade into year
            self.year = self
                .year
                .saturating_add(self.month)
                .saturating_add(self.week)
                .saturating_add(self.day)
                .saturating_add(self.hour);
            self.month = 0;
            self.week = 0;
            self.day = 0;
            self.hour = 0;
            self.last_rolled = now;
        } else if elapsed >= SECS_PER_WEEK {
            // hour + day + week all cascade into month
            self.month = self
                .month
                .saturating_add(self.week)
                .saturating_add(self.day)
                .saturating_add(self.hour);
            self.week = 0;
            self.day = 0;
            self.hour = 0;
            self.last_rolled = now;
        } else if elapsed >= SECS_PER_DAY {
            // hour + day both cascade into week
            self.week = self.week.saturating_add(self.day).saturating_add(self.hour);
            self.day = 0;
            self.hour = 0;
            self.last_rolled = now;
        } else if elapsed >= SECS_PER_HOUR {
            // hour cascades into day
            self.day = self.day.saturating_add(self.hour);
            self.hour = 0;
            self.last_rolled = now;
        }
        // Less than 1 hour: nothing to roll. Don't update last_rolled
        // so partial-hour accesses accumulate correctly in the hour bucket.
    }

    /// Record a single access at the given time.
    pub fn record_access_at(&mut self, now: i64) {
        self.roll_up(now);
        self.hour = self.hour.saturating_add(1);
        self.total = self.total.saturating_add(1);
    }

    /// Record a single access using the current wall clock.
    pub fn record_access(&mut self) {
        self.record_access_at(now_epoch_secs());
    }

    /// Temporal relevancy score based on access buckets.
    ///
    /// Returns a value in [0.0, 1.0] representing how "temporally hot"
    /// this chunk is. Suitable for blending with vector similarity.
    pub fn relevancy_score(&self, recency_window: Option<RecencyWindow>) -> f32 {
        let weights = match recency_window {
            None => RecencyWeights::balanced(),
            Some(RecencyWindow::Day) => RecencyWeights::day(),
            Some(RecencyWindow::Week) => RecencyWeights::week(),
            Some(RecencyWindow::Month) => RecencyWeights::month(),
        };

        let raw = (self.hour as f32) * weights.w_hour
            + (self.day as f32) * weights.w_day
            + (self.week as f32) * weights.w_week
            + (self.month as f32) * weights.w_month
            + (self.year as f32) * weights.w_year
            + (self.total as f32) * weights.w_total;

        (raw / weights.scale).tanh()
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

pub(crate) fn now_epoch_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
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
    /// How this chunk should be treated in search and aging.
    /// Open string -- use constants from `visibility::` or any custom value.
    /// Default: "normal"
    #[serde(default = "default_visibility")]
    pub visibility: String,

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
            visibility: default_visibility(),
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
            visibility: default_visibility(),
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

    // --- AccessProfile RRD tests ---

    #[test]
    fn test_access_profile_new() {
        let profile = AccessProfile::new();
        assert_eq!(profile.total, 0);
        assert_eq!(profile.hour, 0);
        assert_eq!(profile.day, 0);
        assert_eq!(profile.week, 0);
        assert_eq!(profile.month, 0);
        assert_eq!(profile.year, 0);
        assert!(profile.created_at > 0);
        assert!(profile.last_rolled > 0);
    }

    #[test]
    fn test_access_profile_record_access() {
        let mut profile = AccessProfile::new();
        assert_eq!(profile.total, 0);

        profile.record_access();
        assert_eq!(profile.total, 1);
        assert_eq!(profile.hour, 1);

        profile.record_access();
        assert_eq!(profile.total, 2);
        assert_eq!(profile.hour, 2);
    }

    #[test]
    fn test_access_profile_default() {
        let profile = AccessProfile::default();
        assert_eq!(profile.total, 0);
        assert!(profile.created_at > 0);
    }

    #[test]
    fn test_roll_up_within_hour() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        // Record 3 accesses within the first hour
        profile.record_access_at(base + 10);
        profile.record_access_at(base + 20);
        profile.record_access_at(base + 30);
        assert_eq!(profile.hour, 3);
        assert_eq!(profile.day, 0);
        assert_eq!(profile.total, 3);
    }

    #[test]
    fn test_roll_up_hour_to_day() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        // 3 accesses in the first hour
        profile.record_access_at(base + 10);
        profile.record_access_at(base + 20);
        profile.record_access_at(base + 30);
        assert_eq!(profile.hour, 3);

        // 2 hours later: hour rolls into day
        profile.record_access_at(base + SECS_PER_HOUR + 100);
        assert_eq!(profile.hour, 1); // the new access
        assert_eq!(profile.day, 3); // rolled from hour
        assert_eq!(profile.total, 4);
    }

    #[test]
    fn test_roll_up_day_to_week() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.record_access_at(base + 10);
        profile.record_access_at(base + 20);

        // Jump forward 2 days
        profile.record_access_at(base + SECS_PER_DAY + 100);
        assert_eq!(profile.hour, 1);
        assert_eq!(profile.day, 0); // day was zeroed
        assert_eq!(profile.week, 2); // rolled from hour->day already, then day->week
        assert_eq!(profile.total, 3);
    }

    #[test]
    fn test_roll_up_week_to_month() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.record_access_at(base + 10);

        // Jump forward 8 days
        profile.record_access_at(base + SECS_PER_WEEK + 100);
        assert_eq!(profile.month, 1); // rolled from week
        assert_eq!(profile.week, 0);
        assert_eq!(profile.total, 2);
    }

    #[test]
    fn test_roll_up_month_to_year() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.record_access_at(base + 10);

        // Jump forward 31 days
        profile.record_access_at(base + SECS_PER_MONTH + 100);
        assert_eq!(profile.year, 1); // rolled from month
        assert_eq!(profile.month, 0);
        assert_eq!(profile.total, 2);
    }

    #[test]
    fn test_roll_up_beyond_year() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.record_access_at(base + 10);
        profile.record_access_at(base + 20);

        // Jump forward 400 days
        profile.record_access_at(base + SECS_PER_YEAR + 100);
        // All buckets zeroed (counts already in total), then new access in hour
        assert_eq!(profile.hour, 1);
        assert_eq!(profile.day, 0);
        assert_eq!(profile.week, 0);
        assert_eq!(profile.month, 0);
        assert_eq!(profile.year, 0);
        assert_eq!(profile.total, 3);
    }

    #[test]
    fn test_roll_up_idempotent() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.record_access_at(base + 10);

        let now = base + SECS_PER_HOUR + 100;
        profile.roll_up(now);
        let snapshot = profile.clone();

        // Rolling up again at the same time should be a no-op
        profile.roll_up(now);
        assert_eq!(profile, snapshot);
    }

    #[test]
    fn test_roll_up_clock_backwards() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.record_access_at(base + 10);

        // Roll up with a time in the past — should be a no-op
        profile.roll_up(base - 100);
        assert_eq!(profile.hour, 1);
        assert_eq!(profile.total, 1);
    }

    #[test]
    fn test_saturating_add_prevents_overflow() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.hour = u16::MAX;
        profile.total = u32::MAX;

        profile.record_access_at(base + 10);
        assert_eq!(profile.hour, u16::MAX); // saturated
        assert_eq!(profile.total, u32::MAX); // saturated
    }

    #[test]
    fn test_relevancy_score_no_accesses() {
        let profile = AccessProfile::with_created_at(1_000_000);
        let score = profile.relevancy_score(None);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_relevancy_score_with_accesses() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.record_access_at(base + 10);
        profile.record_access_at(base + 20);

        let score = profile.relevancy_score(None);
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_relevancy_score_recent_beats_old() {
        let base = 1_000_000;

        // Chunk A: 2 accesses in the current hour
        let mut recent = AccessProfile::with_created_at(base);
        recent.record_access_at(base + 10);
        recent.record_access_at(base + 20);

        // Chunk B: 2 accesses total, but all in the year bucket (old)
        let mut old = AccessProfile::with_created_at(base);
        old.year = 2;
        old.total = 2;

        assert!(recent.relevancy_score(None) > old.relevancy_score(None));
    }

    #[test]
    fn test_relevancy_score_with_recency_window() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.record_access_at(base + 10);

        let balanced = profile.relevancy_score(None);
        let day_weighted = profile.relevancy_score(Some(RecencyWindow::Day));

        // Day-weighted should give a higher score for hour accesses
        assert!(day_weighted >= balanced);
    }

    #[test]
    fn test_recency_window_from_str() {
        assert_eq!(RecencyWindow::from_str_opt("24h"), Some(RecencyWindow::Day));
        assert_eq!(RecencyWindow::from_str_opt("7d"), Some(RecencyWindow::Week));
        assert_eq!(
            RecencyWindow::from_str_opt("30d"),
            Some(RecencyWindow::Month)
        );
        assert_eq!(RecencyWindow::from_str_opt("day"), Some(RecencyWindow::Day));
        assert_eq!(RecencyWindow::from_str_opt("invalid"), None);
    }

    #[test]
    fn test_multi_step_cascade() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);

        // Accumulate accesses in the hour bucket
        for i in 0..5 {
            profile.record_access_at(base + i * 10);
        }
        assert_eq!(profile.hour, 5);
        assert_eq!(profile.total, 5);

        // 2 hours later: hour rolls to day
        profile.record_access_at(base + SECS_PER_HOUR * 2);
        assert_eq!(profile.day, 5);
        assert_eq!(profile.hour, 1);

        // 2 days later: day rolls to week
        profile.record_access_at(base + SECS_PER_HOUR * 2 + SECS_PER_DAY * 2);
        assert_eq!(profile.week, 6); // 5 from day + 1 from hour
        assert_eq!(profile.day, 0);
        assert_eq!(profile.hour, 1);
        assert_eq!(profile.total, 7);
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
