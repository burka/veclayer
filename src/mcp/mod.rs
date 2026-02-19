use std::sync::Arc;

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

use crate::aging::{self, AgingConfig};
use crate::embedder::FastEmbedder;
use crate::search::{HierarchicalSearch, SearchConfig};
use crate::store::LanceStore;
use crate::{Config, Embedder, Result, VectorStore};

/// Instructions provided to agents on first connection.
/// Explains VecLayer's purpose, tools, and the reflection pattern.
const MCP_INSTRUCTIONS: &str = "\
VecLayer is a hierarchical vector database with memory — a persistent identity store for AI agents.

## Your Memory System

You have access to a structured, aging knowledge base. Unlike flat key-value memory, VecLayer \
organizes knowledge in trees (headings → subheadings → content) with visibility levels and \
access tracking. Knowledge that you use often stays prominent. Knowledge you ignore fades.

## Five Tools

### recall — What do I know about this?
Find relevant knowledge using semantic search. Results come with access profiles showing how \
often each piece was accessed. Use `deep: true` to include archived knowledge. Use `recency` \
to boost recently accessed memories.

### focus — Tell me more about this specific point.
Dive deeper into a specific memory node. Returns the node itself plus its children, optionally \
reranked by a question lens. Pass a `question` to surface the most relevant children for that \
angle — 'How was this decided?' yields different details than 'Who was involved?'

### store — I want to remember this.
Write new knowledge directly. Use this for observations, summaries, decisions, reflections — \
anything you want to persist. The server generates embeddings automatically. Use `parent_id` \
to place it in the hierarchy.

### think — Let me reflect and curate.
Your curation hub. Without an action, returns a reflection report: hot chunks, stale chunks, \
and suggested actions. With an action, executes curation:
- `action: 'promote'` — Make important knowledge always visible
- `action: 'demote'` — Archive outdated knowledge
- `action: 'relate'` — Connect knowledge (superseded_by, summarized_by, related_to, derived_from)
- `action: 'configure_aging'` — Set automatic degradation rules
- `action: 'apply_aging'` — Execute aging rules now

### share — Here, this is for you. (Preview)
Generates a scoped share-token payload describing what knowledge to share and with what \
permissions. This is a preview of the upcoming UCAN-based sharing system — tokens are not \
yet cryptographically signed.

## When to Think

Run `think` when you have time: start of session, end of session, between tasks. Review what's \
hot, what's stale, what needs promoting or archiving. Write summaries of related chunks using \
`store`. Mark outdated knowledge with `think(action='relate', kind='superseded_by')`.

## Summarization Pattern

1. `recall` a topic area
2. `focus` on interesting nodes to read children
3. Synthesize a summary
4. `store` it with `parent_id` to place in hierarchy
5. `think(action='relate', kind='summarized_by')` to link children to the summary

You are the curator of your own memory. Use these tools to build a knowledge base that reflects \
what matters to you.";

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    store: Arc<LanceStore>,
    embedder: Arc<FastEmbedder>,
    data_dir: std::path::PathBuf,
}

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
    /// Values: "promote", "demote", "relate", "configure_aging", "apply_aging"
    pub action: Option<String>,

    // ── reflect parameters (when action is None) ──
    /// Max hot chunks in report (default: 10)
    #[serde(default = "default_reflect_limit")]
    pub hot_limit: Option<usize>,
    /// Max stale chunks in report (default: 10)
    #[serde(default = "default_reflect_limit")]
    pub stale_limit: Option<usize>,

    // ── promote/demote parameters ──
    /// Chunk ID (for promote/demote)
    pub id: Option<String>,
    /// Target visibility (for promote: "always", for demote: "deep_only")
    pub visibility: Option<String>,

    // ── relate parameters ──
    /// Source chunk ID (for relate)
    pub source_id: Option<String>,
    /// Target chunk ID (for relate)
    pub target_id: Option<String>,
    /// Relation kind (for relate, default: "related_to")
    pub kind: Option<String>,

    // ── configure_aging parameters ──
    /// Days without access before degradation
    pub degrade_after_days: Option<u32>,
    /// Target visibility for degraded chunks
    pub degrade_to: Option<String>,
    /// Only degrade chunks with these visibilities
    pub degrade_from: Option<Vec<String>>,
}

fn default_reflect_limit() -> Option<usize> {
    Some(10)
}

/// Input for share (token generation)
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ShareInput {
    /// Tree/scope to share (e.g. "projects:veclayer")
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
    /// Semantic relevance to the focus question (if question was provided)
    pub relevance: Option<f32>,
}

/// API response wrapper
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ApiResponse<T> {
    Success(T),
    Error { error: String },
}

// ─── MCP Stdio transport ───────────────────────────────────────────────

/// Run the MCP server on stdio (for Claude Desktop integration)
pub async fn run_stdio(config: Config) -> Result<()> {
    use std::io::{BufRead, Write};

    info!("Starting MCP stdio server...");

    let embedder = FastEmbedder::new()?;
    let dimension = embedder.dimension();
    let store = LanceStore::open(&config.data_dir, dimension).await?;
    let store = Arc::new(store);
    let embedder = Arc::new(embedder);

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    let data_dir = config.data_dir.clone();

    for line in stdin.lock().lines() {
        let line = line.map_err(crate::Error::Io)?;
        if line.is_empty() {
            continue;
        }

        let response = handle_mcp_message(&line, &store, &embedder, &data_dir).await;
        writeln!(stdout, "{}", response).map_err(crate::Error::Io)?;
        stdout.flush().map_err(crate::Error::Io)?;
    }

    Ok(())
}

// ─── MCP message handler ───────────────────────────────────────────────

async fn handle_mcp_message(
    message: &str,
    store: &Arc<LanceStore>,
    embedder: &Arc<FastEmbedder>,
    data_dir: &std::path::Path,
) -> String {
    let parsed: serde_json::Value = match serde_json::from_str(message) {
        Ok(v) => v,
        Err(e) => {
            return serde_json::json!({
                "jsonrpc": "2.0",
                "error": { "code": -32700, "message": format!("Parse error: {}", e) }
            })
            .to_string();
        }
    };

    let id = parsed.get("id").cloned();
    let method = parsed.get("method").and_then(|m| m.as_str()).unwrap_or("");

    let result = match method {
        "initialize" => {
            serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "veclayer",
                    "version": env!("CARGO_PKG_VERSION")
                },
                "instructions": MCP_INSTRUCTIONS
            })
        }
        "tools/list" => {
            serde_json::json!({
                "tools": [
                    {
                        "name": "recall",
                        "description": "Find relevant knowledge using semantic vector search. Returns matching memories with access profiles, hierarchy paths, and children.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": { "type": "string", "description": "What to search for" },
                                "limit": { "type": "integer", "description": "Number of results (default: 5)", "default": 5 },
                                "deep": { "type": "boolean", "description": "Include archived/hidden memories", "default": false },
                                "recency": { "type": "string", "description": "Recency boost window: 24h, 7d, or 30d" }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "focus",
                        "description": "Dive deeper into a specific memory node. Returns the node itself plus its children, optionally reranked by a question lens using semantic similarity.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string", "description": "The chunk/node ID to focus on" },
                                "question": { "type": "string", "description": "Optional question lens — children are reranked by semantic relevance to this question" },
                                "limit": { "type": "integer", "description": "Max children to return (default: 10)", "default": 10 }
                            },
                            "required": ["id"]
                        }
                    },
                    {
                        "name": "store",
                        "description": "Persist a new memory (observation, summary, decision, reflection). Server generates embeddings automatically. Use parent_id to place in hierarchy.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "content": { "type": "string", "description": "The text content to remember" },
                                "parent_id": { "type": "string", "description": "Optional parent chunk ID for hierarchy placement" },
                                "source_file": { "type": "string", "description": "Source label (default: '[agent]')", "default": "[agent]" },
                                "heading": { "type": "string", "description": "Optional heading/title" },
                                "visibility": { "type": "string", "description": "Visibility: always, normal, deep_only (default: normal)", "default": "normal" }
                            },
                            "required": ["content"]
                        }
                    },
                    {
                        "name": "think",
                        "description": "Reflect and curate your memory. Without action: returns reflection report (hot/stale chunks, suggestions). With action: execute curation (promote, demote, relate, configure_aging, apply_aging).",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "action": { "type": "string", "description": "Curation action: promote, demote, relate, configure_aging, apply_aging. Omit for reflection report." },
                                "hot_limit": { "type": "integer", "description": "Max hot chunks in report (default: 10)", "default": 10 },
                                "stale_limit": { "type": "integer", "description": "Max stale chunks in report (default: 10)", "default": 10 },
                                "id": { "type": "string", "description": "Chunk ID (for promote/demote)" },
                                "visibility": { "type": "string", "description": "Target visibility (for promote/demote)" },
                                "source_id": { "type": "string", "description": "Source chunk ID (for relate)" },
                                "target_id": { "type": "string", "description": "Target chunk ID (for relate)" },
                                "kind": { "type": "string", "description": "Relation kind: superseded_by, summarized_by, related_to, derived_from (for relate)", "default": "related_to" },
                                "degrade_after_days": { "type": "integer", "description": "Days without access before degradation (for configure_aging)" },
                                "degrade_to": { "type": "string", "description": "Target visibility for degraded chunks (for configure_aging)" },
                                "degrade_from": { "type": "array", "items": { "type": "string" }, "description": "Only degrade chunks with these visibilities (for configure_aging)" }
                            }
                        }
                    },
                    {
                        "name": "share",
                        "description": "Generate a scoped share-token payload describing what to share and with what permissions. Preview: tokens are not yet cryptographically signed (UCAN planned).",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "tree": { "type": "string", "description": "Scope/tree to share (e.g. 'projects:veclayer')" },
                                "can": { "type": "array", "items": { "type": "string" }, "description": "Allowed operations (default: ['recall', 'focus'])" },
                                "expires": { "type": "string", "description": "Expiry hint (e.g. '90d')" }
                            },
                            "required": ["tree"]
                        }
                    }
                ]
            })
        }
        "tools/call" => {
            let params = parsed
                .get("params")
                .cloned()
                .unwrap_or(serde_json::json!({}));
            let tool_name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let arguments = params
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::json!({}));

            match tool_name {
                "recall" => {
                    let input: RecallInput = match serde_json::from_value(arguments) {
                        Ok(i) => i,
                        Err(e) => {
                            return format_mcp_error(id, -32602, &format!("Invalid params: {}", e));
                        }
                    };
                    match execute_recall(store, embedder, input).await {
                        Ok(results) => mcp_text_result(
                            &serde_json::to_string_pretty(&results).unwrap_or_default(),
                        ),
                        Err(e) => mcp_error_result(&e.to_string()),
                    }
                }
                "focus" => {
                    let input: FocusInput = match serde_json::from_value(arguments) {
                        Ok(i) => i,
                        Err(e) => {
                            return format_mcp_error(id, -32602, &format!("Invalid params: {}", e));
                        }
                    };
                    match execute_focus(store, embedder, input).await {
                        Ok(response) => mcp_text_result(
                            &serde_json::to_string_pretty(&response).unwrap_or_default(),
                        ),
                        Err(e) => mcp_error_result(&e.to_string()),
                    }
                }
                "store" => {
                    let input: StoreInput = match serde_json::from_value(arguments) {
                        Ok(i) => i,
                        Err(e) => {
                            return format_mcp_error(id, -32602, &format!("Invalid params: {}", e));
                        }
                    };
                    if input.content.is_empty() {
                        return format_mcp_error(id, -32602, "Missing required parameter: content");
                    }
                    match execute_store(store, embedder, input).await {
                        Ok(chunk_id) => {
                            mcp_text_result(&format!("Stored. ID: {}", chunk_id))
                        }
                        Err(e) => mcp_error_result(&e.to_string()),
                    }
                }
                "think" => {
                    let input: ThinkInput = match serde_json::from_value(arguments) {
                        Ok(i) => i,
                        Err(e) => {
                            return format_mcp_error(id, -32602, &format!("Invalid params: {}", e));
                        }
                    };
                    match execute_think(store, data_dir, input).await {
                        Ok(text) => mcp_text_result(&text),
                        Err(e) => mcp_error_result(&e.to_string()),
                    }
                }
                "share" => {
                    let input: ShareInput = match serde_json::from_value(arguments) {
                        Ok(i) => i,
                        Err(e) => {
                            return format_mcp_error(id, -32602, &format!("Invalid params: {}", e));
                        }
                    };
                    let token = build_share_token(input);
                    mcp_text_result(
                        &serde_json::to_string_pretty(&token).unwrap_or_default(),
                    )
                }
                _ => mcp_error_result(&format!("Unknown tool: {}. Available: recall, focus, store, think, share", tool_name)),
            }
        }
        "notifications/initialized" | "initialized" => {
            return String::new(); // No response for notifications
        }
        _ => {
            return format_mcp_error(id, -32601, &format!("Method not found: {}", method));
        }
    };

    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    })
    .to_string()
}

fn format_mcp_error(id: Option<serde_json::Value>, code: i32, message: &str) -> String {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message }
    })
    .to_string()
}

fn mcp_text_result(text: &str) -> serde_json::Value {
    serde_json::json!({
        "content": [{ "type": "text", "text": text }]
    })
}

fn mcp_error_result(text: &str) -> serde_json::Value {
    serde_json::json!({
        "content": [{ "type": "text", "text": format!("Error: {}", text) }],
        "isError": true
    })
}

// ─── Tool implementations ──────────────────────────────────────────────

async fn execute_recall(
    store: &Arc<LanceStore>,
    embedder: &Arc<FastEmbedder>,
    input: RecallInput,
) -> Result<Vec<SearchResultResponse>> {
    let recency_window = input
        .recency
        .as_deref()
        .and_then(crate::RecencyWindow::from_str_opt);

    let config = SearchConfig {
        top_k: input.limit,
        children_k: 3,
        max_depth: 3,
        min_score: 0.0,
        deep: input.deep,
        recency_window,
        recency_alpha: SearchConfig::alpha_for_window(recency_window),
    };

    let search =
        HierarchicalSearch::new(Arc::clone(store), Arc::clone(embedder)).with_config(config);

    let results = search.search(&input.query).await?;

    Ok(results
        .into_iter()
        .map(|r| SearchResultResponse {
            chunk: ChunkResponse::from(&r.chunk),
            score: r.score,
            hierarchy_path: r.hierarchy_path.iter().map(ChunkResponse::from).collect(),
            children: r
                .relevant_children
                .iter()
                .map(|c| ChunkResponse::from(&c.chunk))
                .collect(),
        })
        .collect())
}

async fn execute_focus(
    store: &Arc<LanceStore>,
    embedder: &Arc<FastEmbedder>,
    input: FocusInput,
) -> Result<FocusResponse> {
    // Get the node itself
    let node = store
        .get_by_id(&input.id)
        .await?
        .ok_or_else(|| crate::Error::not_found(format!("Chunk not found: {}", input.id)))?;

    // Get children
    let children = store.get_children(&input.id).await?;

    // Semantic reranking if question is provided
    let focus_children = if let Some(ref question) = input.question {
        // Embed the question
        let question_embedding = embedder.embed(&[question.as_str()])?;
        let question_vec = question_embedding
            .into_iter()
            .next()
            .ok_or_else(|| crate::Error::embedding("Failed to embed question"))?;

        // Score each child by cosine similarity to the question
        let mut scored: Vec<(crate::HierarchicalChunk, f32)> = children
            .into_iter()
            .map(|child| {
                let score = child
                    .embedding
                    .as_ref()
                    .map(|emb| crate::search::cosine_similarity(&question_vec, emb))
                    .unwrap_or(0.0);
                (child, score)
            })
            .collect();

        // Sort by relevance descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(input.limit);

        scored
            .into_iter()
            .map(|(chunk, score)| FocusChild {
                chunk: ChunkResponse::from(&chunk),
                relevance: Some(score),
            })
            .collect()
    } else {
        // No question: return children in natural order
        children
            .into_iter()
            .take(input.limit)
            .map(|chunk| FocusChild {
                chunk: ChunkResponse::from(&chunk),
                relevance: None,
            })
            .collect()
    };

    Ok(FocusResponse {
        node: ChunkResponse::from(&node),
        children: focus_children,
    })
}

async fn execute_store(
    store: &Arc<LanceStore>,
    embedder: &Arc<FastEmbedder>,
    input: StoreInput,
) -> Result<String> {
    let parent_id = input.parent_id.as_deref().filter(|s| !s.is_empty());

    // Determine the level
    let level = if let Some(pid) = parent_id {
        if let Ok(Some(parent)) = store.get_by_id(pid).await {
            crate::chunk::ChunkLevel(parent.level.0 + 1)
        } else {
            crate::chunk::ChunkLevel(7)
        }
    } else {
        crate::chunk::ChunkLevel(1)
    };

    // Build path
    let path = if let Some(pid) = parent_id {
        if let Ok(Some(parent)) = store.get_by_id(pid).await {
            format!("{}/agent", parent.path)
        } else {
            input.source_file.clone()
        }
    } else {
        input.source_file.clone()
    };

    // Generate embedding
    let embeddings = embedder.embed(&[input.content.as_str()])?;
    let embedding = embeddings
        .into_iter()
        .next()
        .ok_or_else(|| crate::Error::embedding("Failed to generate embedding"))?;

    let chunk_id = uuid::Uuid::new_v4().to_string();

    let chunk = crate::HierarchicalChunk {
        id: chunk_id.clone(),
        content: input.content,
        embedding: Some(embedding),
        level,
        parent_id: parent_id.map(String::from),
        path,
        source_file: input.source_file,
        heading: input.heading,
        start_offset: 0,
        end_offset: 0,
        cluster_memberships: vec![],
        is_summary: false,
        summarizes: vec![],
        visibility: input.visibility,
        relations: vec![],
        access_profile: crate::AccessProfile::new(),
        expires_at: None,
    };

    store.insert_chunks(vec![chunk]).await?;
    Ok(chunk_id)
}

async fn execute_think(
    store: &Arc<LanceStore>,
    data_dir: &std::path::Path,
    input: ThinkInput,
) -> Result<String> {
    match input.action.as_deref() {
        None => {
            // Reflection report
            let hot_limit = input.hot_limit.unwrap_or(10);
            let stale_limit = input.stale_limit.unwrap_or(10);
            execute_reflect(store, data_dir, hot_limit, stale_limit).await
        }
        Some("promote") => {
            let chunk_id = input
                .id
                .as_deref()
                .ok_or_else(|| crate::Error::config("think(promote) requires 'id'"))?;
            let vis = input.visibility.as_deref().unwrap_or("always");
            store.update_visibility(chunk_id, vis).await?;
            Ok(format!("Promoted `{}` to visibility '{}'", chunk_id, vis))
        }
        Some("demote") => {
            let chunk_id = input
                .id
                .as_deref()
                .ok_or_else(|| crate::Error::config("think(demote) requires 'id'"))?;
            let vis = input.visibility.as_deref().unwrap_or("deep_only");
            store.update_visibility(chunk_id, vis).await?;
            Ok(format!("Demoted `{}` to visibility '{}'", chunk_id, vis))
        }
        Some("relate") => {
            let source_id = input
                .source_id
                .as_deref()
                .ok_or_else(|| crate::Error::config("think(relate) requires 'source_id'"))?;
            let target_id = input
                .target_id
                .as_deref()
                .ok_or_else(|| crate::Error::config("think(relate) requires 'target_id'"))?;
            let kind = input.kind.as_deref().unwrap_or("related_to");

            let relation = crate::ChunkRelation::new(kind, target_id);
            store.add_relation(source_id, relation).await?;
            Ok(format!(
                "Added relation '{}' from `{}` to `{}`",
                kind, source_id, target_id
            ))
        }
        Some("configure_aging") => {
            let mut config = AgingConfig::load(data_dir);
            if let Some(days) = input.degrade_after_days {
                config.degrade_after_days = days;
            }
            if let Some(ref to) = input.degrade_to {
                config.degrade_to = to.clone();
            }
            if let Some(ref from) = input.degrade_from {
                config.degrade_from = from.clone();
            }
            config.save(data_dir)?;
            Ok(format!(
                "Aging configured: degrade {} → '{}' after {} days without access",
                config.degrade_from.join(", "),
                config.degrade_to,
                config.degrade_after_days
            ))
        }
        Some("apply_aging") => {
            let config = AgingConfig::load(data_dir);
            let result = aging::apply_aging(store.as_ref(), &config).await?;
            if result.degraded_count == 0 {
                Ok("No chunks needed aging. All knowledge is fresh.".to_string())
            } else {
                Ok(format!(
                    "Aged {} chunks (degraded to '{}'): {}",
                    result.degraded_count,
                    config.degrade_to,
                    result.degraded_ids.join(", ")
                ))
            }
        }
        Some(unknown) => Err(crate::Error::config(format!(
            "Unknown think action: '{}'. Available: promote, demote, relate, configure_aging, apply_aging",
            unknown
        ))),
    }
}

/// Build a reflection report: hot chunks, stale chunks, suggested actions.
async fn execute_reflect(
    store: &Arc<LanceStore>,
    data_dir: &std::path::Path,
    hot_limit: usize,
    stale_limit: usize,
) -> Result<String> {
    let aging_config = AgingConfig::load(data_dir);

    let hot = store.get_hot_chunks(hot_limit).await?;
    let stale = store
        .get_stale_chunks(aging_config.stale_seconds(), stale_limit)
        .await?;

    let mut report = String::new();

    // Hot chunks
    report.push_str("## Hot Chunks (most accessed)\n\n");
    if hot.is_empty() {
        report.push_str("No chunks have been accessed yet.\n\n");
    } else {
        for chunk in &hot {
            let heading = chunk.heading.as_deref().unwrap_or("(no heading)");
            report.push_str(&format!(
                "- **{}** (total: {}, hour: {}, day: {}, week: {}) [{}] `{}`\n",
                heading,
                chunk.access_profile.total,
                chunk.access_profile.hour,
                chunk.access_profile.day,
                chunk.access_profile.week,
                chunk.visibility,
                chunk.id
            ));
        }
        report.push('\n');
    }

    // Stale chunks
    report.push_str(&format!(
        "## Stale Chunks (no access in {} days)\n\n",
        aging_config.degrade_after_days
    ));
    if stale.is_empty() {
        report.push_str("No stale chunks found. Memory is well-maintained.\n\n");
    } else {
        for chunk in &stale {
            let heading = chunk.heading.as_deref().unwrap_or("(no heading)");
            report.push_str(&format!(
                "- **{}** [{}] `{}`\n",
                heading, chunk.visibility, chunk.id
            ));
        }
        report.push('\n');
    }

    // Summary stats
    let stats = store.stats().await?;
    report.push_str(&format!(
        "## Summary\n\n- Total chunks: {}\n- Source files: {}\n- Aging policy: degrade {} → '{}' after {} days\n",
        stats.total_chunks,
        stats.source_files.len(),
        aging_config.degrade_from.join("/"),
        aging_config.degrade_to,
        aging_config.degrade_after_days,
    ));

    // Suggested actions
    report.push_str("\n## Suggested Actions\n\n");
    let mut has_suggestions = false;

    if !stale.is_empty() {
        report.push_str(&format!(
            "- Run `think(action='apply_aging')` to degrade {} stale chunks automatically\n",
            stale.len()
        ));
        has_suggestions = true;
    }

    for chunk in &hot {
        if chunk.access_profile.total > 10 && chunk.visibility == "normal" {
            report.push_str(&format!(
                "- Consider `think(action='promote', id='{}')` — **{}** accessed {} times but still 'normal'\n",
                chunk.id,
                chunk.heading.as_deref().unwrap_or("(no heading)"),
                chunk.access_profile.total
            ));
            has_suggestions = true;
        }
    }

    if !has_suggestions {
        report.push_str("No urgent actions needed.\n");
    }

    Ok(report)
}

/// Build a share token payload (preview — not yet cryptographically signed).
fn build_share_token(input: ShareInput) -> serde_json::Value {
    let can = if input.can.is_empty() {
        vec!["recall".to_string(), "focus".to_string()]
    } else {
        input.can
    };

    serde_json::json!({
        "version": "veclayer-share-v1-preview",
        "tree": input.tree,
        "can": can,
        "expires": input.expires,
        "nonce": uuid::Uuid::new_v4().to_string(),
        "_note": "Preview token. UCAN signing not yet implemented."
    })
}

// ─── HTTP REST API ─────────────────────────────────────────────────────

/// Run the HTTP REST API server
pub async fn run_http(config: Config) -> Result<()> {
    let embedder = FastEmbedder::new()?;
    let dimension = embedder.dimension();
    let store = LanceStore::open(&config.data_dir, dimension).await?;

    let state = AppState {
        store: Arc::new(store),
        embedder: Arc::new(embedder),
        data_dir: config.data_dir.clone(),
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/api/recall", post(api_recall))
        .route("/api/focus", post(api_focus))
        .route("/api/store", post(api_store))
        .route("/api/think", post(api_think))
        .route("/api/share", post(api_share))
        .route("/api/stats", get(api_stats))
        .layer(cors)
        .with_state(state);

    let addr: std::net::SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .map_err(|e| crate::Error::config(format!("Invalid address: {}", e)))?;

    info!("VecLayer HTTP server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(crate::Error::Io)?;

    axum::serve(listener, app)
        .await
        .map_err(|e| crate::Error::InvalidOperation(format!("Server error: {}", e)))?;

    Ok(())
}

async fn api_recall(
    State(state): State<AppState>,
    Json(input): Json<RecallInput>,
) -> Json<ApiResponse<Vec<SearchResultResponse>>> {
    match execute_recall(&state.store, &state.embedder, input).await {
        Ok(results) => Json(ApiResponse::Success(results)),
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
    }
}

async fn api_focus(
    State(state): State<AppState>,
    Json(input): Json<FocusInput>,
) -> Json<ApiResponse<FocusResponse>> {
    match execute_focus(&state.store, &state.embedder, input).await {
        Ok(response) => Json(ApiResponse::Success(response)),
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
    }
}

async fn api_store(
    State(state): State<AppState>,
    Json(input): Json<StoreInput>,
) -> Json<ApiResponse<String>> {
    if input.content.is_empty() {
        return Json(ApiResponse::Error {
            error: "content is required".to_string(),
        });
    }
    match execute_store(&state.store, &state.embedder, input).await {
        Ok(chunk_id) => Json(ApiResponse::Success(format!("Stored. ID: {}", chunk_id))),
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
    }
}

async fn api_think(
    State(state): State<AppState>,
    Json(input): Json<ThinkInput>,
) -> Json<ApiResponse<String>> {
    match execute_think(&state.store, &state.data_dir, input).await {
        Ok(text) => Json(ApiResponse::Success(text)),
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
    }
}

async fn api_share(Json(input): Json<ShareInput>) -> Json<ApiResponse<serde_json::Value>> {
    Json(ApiResponse::Success(build_share_token(input)))
}

async fn api_stats(State(state): State<AppState>) -> Json<ApiResponse<serde_json::Value>> {
    match state.store.stats().await {
        Ok(stats) => Json(ApiResponse::Success(serde_json::json!({
            "total_chunks": stats.total_chunks,
            "chunks_by_level": stats.chunks_by_level,
            "source_files": stats.source_files,
        }))),
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn share_token_defaults_and_custom() {
        let token = build_share_token(ShareInput {
            tree: "projects:veclayer".to_string(),
            can: vec![],
            expires: None,
        });
        assert_eq!(token["tree"], "projects:veclayer");
        assert_eq!(token["can"], serde_json::json!(["recall", "focus"]));
        assert_eq!(token["version"], "veclayer-share-v1-preview");
        assert!(token["_note"].as_str().unwrap().contains("Preview"));
        assert!(token["nonce"].as_str().is_some_and(|s| !s.is_empty()));

        let token2 = build_share_token(ShareInput {
            tree: "people:florian".to_string(),
            can: vec!["recall".into(), "focus".into(), "store".into()],
            expires: Some("90d".to_string()),
        });
        assert_eq!(
            token2["can"],
            serde_json::json!(["recall", "focus", "store"])
        );
        assert_eq!(token2["expires"], "90d");
    }
}
