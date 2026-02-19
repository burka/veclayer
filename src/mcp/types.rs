//! Input and output types for MCP tools.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ─── Input types ───────────────────────────────────────────────────────

/// Input for recall (semantic search)
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecallInput {
    /// The search query
    pub query: String,
    /// Number of results (default: 5)
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Include all visibilities (deep_only, expired, custom)
    #[serde(default)]
    pub deep: bool,
    /// Recency window for relevancy boosting: "24h", "7d", "30d"
    #[serde(default)]
    pub recency: Option<String>,
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

/// Input for store (write new memory)
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StoreInput {
    /// The text content to store
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
    pub heading: Option<String>,
    pub parent_id: Option<String>,
    pub visibility: String,
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

/// Search result for API responses
#[derive(Debug, Serialize, JsonSchema)]
pub struct SearchResultResponse {
    pub chunk: ChunkResponse,
    pub score: f32,
    pub hierarchy_path: Vec<ChunkResponse>,
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
