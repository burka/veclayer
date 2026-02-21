//! Input and output types for MCP tools.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ─── Input types ───────────────────────────────────────────────────────

/// Input for recall (semantic search)
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecallInput {
    /// The search query (optional — omit to browse without vector search)
    #[serde(default)]
    pub query: Option<String>,
    /// Number of results (default: 5)
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Include all visibilities (deep_only, expired, custom)
    #[serde(default)]
    pub deep: bool,
    /// Recency window for relevancy boosting: "24h", "7d", "30d"
    #[serde(default)]
    pub recency: Option<String>,
    /// Filter by perspective (e.g. "decisions", "learnings")
    #[serde(default)]
    pub perspective: Option<String>,
    /// Minimum salience threshold (soft filter: excludes from salience boosting but not from results)
    #[serde(default)]
    pub min_salience: Option<f32>,
    /// Minimum search score (hard filter applied to pre-blend vector score, not blended score)
    #[serde(default)]
    pub min_score: Option<f32>,
    /// Filter: only entries created after this ISO 8601 date or epoch seconds
    #[serde(default)]
    pub since: Option<String>,
    /// Filter: only entries created before this ISO 8601 date or epoch seconds
    #[serde(default)]
    pub until: Option<String>,
}

fn default_limit() -> usize {
    5
}

/// Input for focus (drill into a node)
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FocusInput {
    /// The chunk/node ID to focus on
    pub id: String,
    /// Optional question lens for semantic reranking of children
    pub question: Option<String>,
    /// Max children to return (default: 10)
    #[serde(default = "default_focus_limit")]
    pub limit: usize,
}

fn default_focus_limit() -> usize {
    10
}

/// A relation to establish when storing an entry.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StoreRelation {
    /// Relation kind (active voice): supersedes, summarizes, related_to, derived_from, version_of
    pub kind: String,
    /// Target entry ID
    pub target_id: String,
}

/// A single entry in a batch store operation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StoreItem {
    pub content: String,
    #[serde(default)]
    pub parent_id: Option<String>,
    #[serde(default)]
    pub heading: Option<String>,
    #[serde(default = "default_visibility")]
    pub visibility: String,
    #[serde(default)]
    pub perspectives: Vec<String>,
    #[serde(default)]
    pub source_file: Option<String>,
    #[serde(default)]
    pub entry_type: Option<String>,
    #[serde(default)]
    pub relations: Vec<StoreRelation>,
}

/// Input for store (write new memory)
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StoreInput {
    /// The text content to store (required in single mode; ignored when items is present)
    #[serde(default)]
    pub content: String,
    /// Optional parent chunk ID for hierarchy placement
    pub parent_id: Option<String>,
    /// Source label (default: "[agent]")
    #[serde(default = "default_agent_source")]
    pub source_file: String,
    /// Optional heading/title
    pub heading: Option<String>,
    /// Visibility: always, normal, deep_only (default: normal)
    #[serde(default = "default_visibility")]
    pub visibility: String,
    /// Perspectives to tag this entry with
    #[serde(default)]
    pub perspectives: Vec<String>,
    /// Relations to establish atomically when storing.
    /// Kinds: "supersedes", "summarizes", "related_to", "derived_from", "version_of"
    #[serde(default)]
    pub relations: Vec<StoreRelation>,
    /// Batch mode: array of entries to store. When present, top-level content/heading/etc are ignored.
    #[serde(default)]
    pub items: Vec<StoreItem>,
    /// Entry type: raw (default), summary, meta, impression
    #[serde(default)]
    pub entry_type: Option<String>,
}

fn default_agent_source() -> String {
    "[agent]".to_string()
}

fn default_visibility() -> String {
    "normal".to_string()
}

/// Input for think (curation hub)
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ThinkInput {
    /// Action to perform. Omit for reflection report.
    pub action: Option<String>,

    // ── reflect parameters (when action is None) ──
    #[serde(default = "default_reflect_limit")]
    pub hot_limit: Option<usize>,
    #[serde(default = "default_reflect_limit")]
    pub stale_limit: Option<usize>,

    // ── promote/demote parameters ──
    pub id: Option<String>,
    pub visibility: Option<String>,

    // ── relate parameters ──
    pub source_id: Option<String>,
    pub target_id: Option<String>,
    pub kind: Option<String>,

    // ── configure_aging parameters ──
    pub degrade_after_days: Option<u32>,
    pub degrade_to: Option<String>,
    pub degrade_from: Option<Vec<String>>,
}

fn default_reflect_limit() -> Option<usize> {
    Some(10)
}

/// Input for share (token generation)
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ShareInput {
    /// Tree/scope to share
    pub tree: String,
    /// Allowed operations (default: ["recall", "focus"])
    #[serde(default)]
    pub can: Vec<String>,
    /// Expiry hint (e.g. "90d")
    pub expires: Option<String>,
}

// ─── Response types ────────────────────────────────────────────────────

/// Access profile summary for API responses
#[derive(Debug, Serialize, JsonSchema)]
pub struct AccessProfileResponse {
    pub hour: u16,
    pub day: u16,
    pub week: u16,
    pub month: u16,
    pub year: u16,
    pub total: u32,
}

/// A simplified chunk for API responses
#[derive(Debug, Serialize, JsonSchema)]
pub struct ChunkResponse {
    pub id: String,
    pub content: String,
    pub level: String,
    pub entry_type: String,
    pub path: String,
    pub source_file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heading: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    pub visibility: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub perspectives: Vec<String>,
    pub access: AccessProfileResponse,
}

impl From<&crate::HierarchicalChunk> for ChunkResponse {
    fn from(chunk: &crate::HierarchicalChunk) -> Self {
        Self {
            id: chunk.id.clone(),
            content: chunk.content.clone(),
            level: chunk.level.to_string(),
            entry_type: chunk.entry_type.to_string(),
            path: chunk.path.clone(),
            source_file: chunk.source_file.clone(),
            heading: chunk.heading.clone(),
            parent_id: chunk.parent_id.clone(),
            visibility: chunk.visibility.clone(),
            perspectives: chunk.perspectives.clone(),
            access: AccessProfileResponse {
                hour: chunk.access_profile.hour,
                day: chunk.access_profile.day,
                week: chunk.access_profile.week,
                month: chunk.access_profile.month,
                year: chunk.access_profile.year,
                total: chunk.access_profile.total,
            },
        }
    }
}

/// Compute relevance tier from blended score.
pub fn relevance_tier(score: f32) -> &'static str {
    if score > 0.45 {
        "strong"
    } else if score > 0.30 {
        "moderate"
    } else if score > 0.15 {
        "weak"
    } else {
        "tangential"
    }
}

/// Search result for API responses
#[derive(Debug, Serialize, JsonSchema)]
pub struct SearchResultResponse {
    pub chunk: ChunkResponse,
    pub score: f32,
    /// Relevance tier: "strong" (>0.45), "moderate" (>0.30), "weak" (>0.15), "tangential" (<=0.15), "browse" (no query)
    pub relevance: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub hierarchy_path: Vec<ChunkResponse>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<ChunkResponse>,
}

/// Focus result: the node itself + its children
#[derive(Debug, Serialize, JsonSchema)]
pub struct FocusResponse {
    pub node: ChunkResponse,
    pub children: Vec<FocusChild>,
}

/// A child in a focus response, with optional relevance score
#[derive(Debug, Serialize, JsonSchema)]
pub struct FocusChild {
    #[serde(flatten)]
    pub chunk: ChunkResponse,
    pub relevance: Option<f32>,
}

/// API response wrapper
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ApiResponse<T> {
    Success(T),
    Error { error: String },
}
