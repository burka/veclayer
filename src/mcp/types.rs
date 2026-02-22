//! Input and output types for MCP tools.

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};

/// Intermediate type for deserializing a value that may be a JSON array or a string.
#[derive(Deserialize)]
#[serde(untagged)]
enum StringOrVec {
    Vec(Vec<String>),
    Str(String),
}

/// Deserialize a `Vec<String>` that tolerates MCP clients sending a string instead of an array.
///
/// Accepts:
///   - `["a", "b"]`         → vec!["a", "b"]
///   - `"a, b"`             → vec!["a", "b"]  (comma-separated)
///   - `"a"`                → vec!["a"]
///   - `null` / missing     → vec![]
fn string_or_vec<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    match Option::<StringOrVec>::deserialize(deserializer)? {
        None => Ok(Vec::new()),
        Some(StringOrVec::Vec(v)) => Ok(v),
        Some(StringOrVec::Str(s)) => Ok(parse_string_as_vec(&s)),
    }
}

/// Same as [`string_or_vec`] but for `Option<Vec<String>>` fields.
fn string_or_option_vec<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    match Option::<StringOrVec>::deserialize(deserializer)? {
        None => Ok(None),
        Some(StringOrVec::Vec(v)) => Ok(Some(v)),
        Some(StringOrVec::Str(s)) => {
            let v = parse_string_as_vec(&s);
            if v.is_empty() {
                Ok(None)
            } else {
                Ok(Some(v))
            }
        }
    }
}

/// Parse a string as a `Vec<String>`, handling JSON-encoded arrays and comma-separated values.
fn parse_string_as_vec(s: &str) -> Vec<String> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    // Handle JSON array accidentally sent as string: "[\"a\", \"b\"]"
    if trimmed.starts_with('[') {
        if let Ok(parsed) = serde_json::from_str::<Vec<String>>(trimmed) {
            return parsed;
        }
    }
    // Comma-separated fallback
    trimmed
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

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
    /// Find entries similar to this entry ID (uses entry's embedding for search)
    #[serde(default)]
    pub similar_to: Option<String>,
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
    #[serde(default, deserialize_with = "string_or_vec")]
    pub perspectives: Vec<String>,
    #[serde(default)]
    pub source_file: Option<String>,
    #[serde(default)]
    pub entry_type: Option<String>,
    #[serde(default)]
    pub relations: Vec<StoreRelation>,
    /// Impression hint (only used when entry_type is "impression").
    #[serde(default)]
    pub impression_hint: Option<String>,
    /// Impression strength: 0.0–1.0 (only used when entry_type is "impression").
    #[serde(default)]
    pub impression_strength: Option<f32>,
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
    #[serde(default, deserialize_with = "string_or_vec")]
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
    /// Impression hint: qualitative label like "uncertain", "confident", "exploratory".
    /// Only used when entry_type is "impression".
    pub impression_hint: Option<String>,
    /// Impression strength: 0.0–1.0, modulates salience weight. Default: 1.0.
    /// Only used when entry_type is "impression".
    pub impression_strength: Option<f32>,
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
    #[serde(default, deserialize_with = "string_or_option_vec")]
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
    #[serde(default, deserialize_with = "string_or_vec")]
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── string_or_vec (Vec<String> fields) ──────────────────────────────

    #[test]
    fn store_perspectives_from_json_array() {
        let input: StoreInput =
            serde_json::from_value(json!({"content": "x", "perspectives": ["a", "b"]})).unwrap();
        assert_eq!(input.perspectives, vec!["a", "b"]);
    }

    #[test]
    fn store_perspectives_from_comma_string() {
        let input: StoreInput =
            serde_json::from_value(json!({"content": "x", "perspectives": "a, b, c"})).unwrap();
        assert_eq!(input.perspectives, vec!["a", "b", "c"]);
    }

    #[test]
    fn store_perspectives_from_single_string() {
        let input: StoreInput =
            serde_json::from_value(json!({"content": "x", "perspectives": "session"})).unwrap();
        assert_eq!(input.perspectives, vec!["session"]);
    }

    #[test]
    fn store_perspectives_from_stringified_json_array() {
        let input: StoreInput = serde_json::from_value(
            json!({"content": "x", "perspectives": r#"["session", "knowledge"]"#}),
        )
        .unwrap();
        assert_eq!(input.perspectives, vec!["session", "knowledge"]);
    }

    #[test]
    fn store_perspectives_missing_defaults_empty() {
        let input: StoreInput = serde_json::from_value(json!({"content": "x"})).unwrap();
        assert!(input.perspectives.is_empty());
    }

    #[test]
    fn store_perspectives_null_defaults_empty() {
        let input: StoreInput =
            serde_json::from_value(json!({"content": "x", "perspectives": null})).unwrap();
        assert!(input.perspectives.is_empty());
    }

    #[test]
    fn store_perspectives_empty_string_defaults_empty() {
        let input: StoreInput =
            serde_json::from_value(json!({"content": "x", "perspectives": ""})).unwrap();
        assert!(input.perspectives.is_empty());
    }

    #[test]
    fn store_item_perspectives_from_string() {
        let item: StoreItem =
            serde_json::from_value(json!({"content": "x", "perspectives": "a, b"})).unwrap();
        assert_eq!(item.perspectives, vec!["a", "b"]);
    }

    #[test]
    fn share_can_from_string() {
        let input: ShareInput =
            serde_json::from_value(json!({"tree": "/", "can": "recall, focus"})).unwrap();
        assert_eq!(input.can, vec!["recall", "focus"]);
    }

    // ── string_or_option_vec (Option<Vec<String>> fields) ───────────────

    #[test]
    fn think_degrade_from_json_array() {
        let input: ThinkInput = serde_json::from_value(
            json!({"action": "configure_aging", "degrade_from": ["normal"]}),
        )
        .unwrap();
        assert_eq!(input.degrade_from, Some(vec!["normal".to_string()]));
    }

    #[test]
    fn think_degrade_from_comma_string() {
        let input: ThinkInput = serde_json::from_value(
            json!({"action": "configure_aging", "degrade_from": "normal, deep_only"}),
        )
        .unwrap();
        assert_eq!(
            input.degrade_from,
            Some(vec!["normal".to_string(), "deep_only".to_string()])
        );
    }

    #[test]
    fn think_degrade_from_null_is_none() {
        let input: ThinkInput =
            serde_json::from_value(json!({"action": "configure_aging", "degrade_from": null}))
                .unwrap();
        assert_eq!(input.degrade_from, None);
    }

    #[test]
    fn think_degrade_from_missing_is_none() {
        let input: ThinkInput =
            serde_json::from_value(json!({"action": "configure_aging"})).unwrap();
        assert_eq!(input.degrade_from, None);
    }

    #[test]
    fn think_degrade_from_empty_string_is_none() {
        let input: ThinkInput =
            serde_json::from_value(json!({"action": "configure_aging", "degrade_from": ""}))
                .unwrap();
        assert_eq!(input.degrade_from, None);
    }

    // ── edge cases from review ──────────────────────────────────────────

    #[test]
    fn perspectives_filters_empty_segments_from_double_comma() {
        let input: StoreInput =
            serde_json::from_value(json!({"content": "x", "perspectives": "a,,b"})).unwrap();
        assert_eq!(input.perspectives, vec!["a", "b"]);
    }

    #[test]
    fn perspectives_malformed_json_array_falls_back_to_single_token() {
        let input: StoreInput =
            serde_json::from_value(json!({"content": "x", "perspectives": "[bad json"})).unwrap();
        assert_eq!(input.perspectives, vec!["[bad json"]);
    }
}
