use std::sync::Arc;

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};
use tracing::info;
use uuid::Uuid;

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

## Core Tools

- **search** — Find relevant knowledge. Results include access profiles showing how often each \
  chunk has been accessed. Use `--deep` to include hidden/archived chunks. Use `--recent 24h/7d/30d` \
  to boost recently accessed knowledge.
- **get_chunk** / **get_children** — Navigate the hierarchy. Start broad, go deep when needed.
- **promote** / **demote** — Manage visibility. Promote important knowledge to 'always' (appears \
  in every search). Demote outdated knowledge to 'deep_only' (hidden from standard search).
- **relate** — Connect knowledge. Mark chunks as 'superseded_by', 'related_to', 'derived_from', \
  or any custom relation kind.
- **ingest_chunk** — Write new knowledge directly. Use this to store observations, summaries, \
  decisions, or any insight you want to remember. The server generates embeddings automatically.

## Reflection Pattern

When you have idle time (start/end of session, between tasks), run the **reflect** tool. It returns:
- **Hot chunks**: Most accessed knowledge this period — your current focus areas
- **Stale chunks**: Knowledge not accessed in a long time — candidates for archiving
- **Superseded chunks**: Outdated knowledge still visible — should be demoted
- **Suggested actions**: Specific recommendations (promote, demote, summarize)

### What to do with reflection data:
1. **Review hot chunks** — Are they still accurate? Do they need updating?
2. **Check stale chunks** — Demote or archive what's no longer relevant
3. **Summarize clusters** — If you see related chunks, write a summary using ingest_chunk
4. **Mark superseded content** — Use relate(kind='superseded_by') when you find newer versions
5. **Promote core knowledge** — Key decisions, preferences, and identity facts → visibility 'always'

## Aging Rules

Use **configure_aging** to set automatic degradation rules (e.g., 'degrade normal chunks to \
deep_only after 30 days without access'). Then **apply_aging** executes those rules. You control \
the policy — VecLayer just enforces it.

## Summarization Pattern

To create hierarchical summaries:
1. Search or get_children for a topic area
2. Read the chunks and synthesize a summary
3. Use ingest_chunk with parent_id to place it in the hierarchy
4. Use relate(kind='summarized_by') to link children to the summary

You are the curator of your own memory. Use these tools to build a knowledge base that reflects \
what matters to you, with important knowledge always accessible and outdated knowledge gracefully fading.";

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    store: Arc<LanceStore>,
    embedder: Arc<FastEmbedder>,
    data_dir: std::path::PathBuf,
}

/// Input for the search endpoint
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchInput {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub include_children: bool,
    /// Deep search: include all visibilities (deep_only, expired, custom)
    #[serde(default)]
    pub deep: bool,
    /// Recency window for relevancy scoring: "24h", "7d", "30d", or null
    #[serde(default)]
    pub recency: Option<String>,
}

fn default_limit() -> usize {
    5
}

/// Input for subtree search
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SubtreeSearchInput {
    pub query: String,
    pub parent_id: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

/// Input for getting chunk by ID
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetChunkInput {
    pub id: String,
}

/// Input for getting children
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetChildrenInput {
    pub parent_id: String,
}

/// Input for visibility changes (promote/demote)
#[derive(Debug, Deserialize, JsonSchema)]
pub struct VisibilityInput {
    /// New visibility value (e.g. "always", "normal", "deep_only")
    pub visibility: Option<String>,
}

/// Input for adding a relation
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RelateInput {
    /// The source chunk ID
    pub source_id: String,
    /// The target chunk ID
    pub target_id: String,
    /// Relation kind (default: "related_to")
    #[serde(default = "default_relation_kind")]
    pub kind: String,
}

fn default_relation_kind() -> String {
    "related_to".to_string()
}

/// Input for ingest_chunk
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IngestChunkInput {
    /// The text content to store
    pub content: String,
    /// Optional parent chunk ID for hierarchy placement
    pub parent_id: Option<String>,
    /// Source label (default: "[agent]")
    #[serde(default = "default_agent_source")]
    pub source_file: String,
    /// Optional heading/title
    pub heading: Option<String>,
    /// Visibility (default: "normal")
    #[serde(default = "default_visibility")]
    pub visibility: String,
}

fn default_agent_source() -> String {
    "[agent]".to_string()
}

fn default_visibility() -> String {
    "normal".to_string()
}

/// Input for reflect
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReflectInput {
    #[serde(default = "default_reflect_limit")]
    pub hot_limit: usize,
    #[serde(default = "default_reflect_limit")]
    pub stale_limit: usize,
}

fn default_reflect_limit() -> usize {
    10
}

/// Input for focus alias tool
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FocusInput {
    pub node_id: String,
    pub question: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

/// Input for share alias tool
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ShareInput {
    pub tree: String,
    #[serde(default)]
    pub can: Vec<String>,
    pub expires: Option<String>,
}

fn canonical_tool_name(tool_name: &str) -> &str {
    tool_name
}

fn build_share_token(input: ShareInput) -> serde_json::Value {
    let can = if input.can.is_empty() {
        vec!["recall".to_string(), "focus".to_string()]
    } else {
        input.can
    };

    serde_json::json!({
        "version": "veclayer-share-v1",
        "tree": input.tree,
        "can": can,
        "expires": input.expires,
        "nonce": Uuid::new_v4().to_string()
    })
}

/// Input for aging configuration
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AgingConfigInput {
    pub degrade_after_days: Option<u32>,
    pub degrade_to: Option<String>,
    pub degrade_from: Option<Vec<String>>,
}

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

/// API response wrapper
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ApiResponse<T> {
    Success(T),
    Error { error: String },
}

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
                        "description": "Recall relevant knowledge using vector search",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": { "type": "string", "description": "Memory query" },
                                "limit": { "type": "integer", "description": "Number of results", "default": 5 },
                                "deep": { "type": "boolean", "description": "Include deep-only and hidden memories", "default": false },
                                "recency": { "type": "string", "description": "Recency window: 24h, 7d, or 30d" }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "focus",
                        "description": "Go deeper from a node/chunk and optionally apply a question lens",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "node_id": { "type": "string", "description": "Chunk or node ID to focus into" },
                                "question": { "type": "string", "description": "Optional lens question for reranking" },
                                "limit": { "type": "integer", "description": "Number of children to return", "default": 5 }
                            },
                            "required": ["node_id"]
                        }
                    },
                    {
                        "name": "store",
                        "description": "Store a new memory chunk with optional hierarchy placement.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "content": { "type": "string", "description": "Memory content to persist" },
                                "parent_id": { "type": "string", "description": "Optional parent for placement" },
                                "source_file": { "type": "string", "description": "Source label", "default": "[agent]" },
                                "heading": { "type": "string", "description": "Optional heading/title" },
                                "visibility": { "type": "string", "description": "Visibility", "default": "normal" }
                            },
                            "required": ["content"]
                        }
                    },
                    {
                        "name": "think",
                        "description": "Run reflection and return curation suggestions.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "hot_limit": { "type": "integer", "description": "Max hot chunks", "default": 10 },
                                "stale_limit": { "type": "integer", "description": "Max stale chunks", "default": 10 }
                            }
                        }
                    },
                    {
                        "name": "share",
                        "description": "Generate a scoped share token payload.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "tree": { "type": "string", "description": "Shared tree scope" },
                                "can": { "type": "array", "items": { "type": "string" }, "description": "Allowed operations" },
                                "expires": { "type": "string", "description": "Optional expiry hint, e.g. 90d" }
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
            let canonical_tool = canonical_tool_name(tool_name);
            let arguments = params
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::json!({}));

            if matches!(
                tool_name,
                "search"
                    | "get_chunk"
                    | "get_children"
                    | "stats"
                    | "promote"
                    | "demote"
                    | "relate"
                    | "reflect"
                    | "ingest_chunk"
                    | "configure_aging"
                    | "apply_aging"
            ) {
                return format_mcp_error(
                    id,
                    -32601,
                    "Legacy tools are not supported. Use: recall, focus, store, think, share.",
                );
            }

            match canonical_tool {
                "recall" => {
                    let input: SearchInput = match serde_json::from_value(arguments) {
                        Ok(i) => i,
                        Err(e) => {
                            return format_mcp_error(id, -32602, &format!("Invalid params: {}", e));
                        }
                    };
                    match execute_search(store, embedder, input).await {
                        Ok(results) => serde_json::json!({
                            "content": [{
                                "type": "text",
                                "text": serde_json::to_string_pretty(&results).unwrap_or_default()
                            }]
                        }),
                        Err(e) => serde_json::json!({
                            "content": [{ "type": "text", "text": format!("Error: {}", e) }],
                            "isError": true
                        }),
                    }
                }
                "get_chunk" => {
                    let input: GetChunkInput = match serde_json::from_value(arguments) {
                        Ok(i) => i,
                        Err(e) => {
                            return format_mcp_error(id, -32602, &format!("Invalid params: {}", e));
                        }
                    };
                    match store.get_by_id(&input.id).await {
                        Ok(Some(chunk)) => serde_json::json!({
                            "content": [{
                                "type": "text",
                                "text": serde_json::to_string_pretty(&ChunkResponse::from(&chunk)).unwrap_or_default()
                            }]
                        }),
                        Ok(None) => serde_json::json!({
                            "content": [{ "type": "text", "text": "Chunk not found" }],
                            "isError": true
                        }),
                        Err(e) => serde_json::json!({
                            "content": [{ "type": "text", "text": format!("Error: {}", e) }],
                            "isError": true
                        }),
                    }
                }
                "get_children" => {
                    let input: GetChildrenInput = match serde_json::from_value(arguments) {
                        Ok(i) => i,
                        Err(e) => {
                            return format_mcp_error(id, -32602, &format!("Invalid params: {}", e));
                        }
                    };
                    match store.get_children(&input.parent_id).await {
                        Ok(children) => {
                            let responses: Vec<ChunkResponse> =
                                children.iter().map(ChunkResponse::from).collect();
                            serde_json::json!({
                                "content": [{
                                    "type": "text",
                                    "text": serde_json::to_string_pretty(&responses).unwrap_or_default()
                                }]
                            })
                        }
                        Err(e) => serde_json::json!({
                            "content": [{ "type": "text", "text": format!("Error: {}", e) }],
                            "isError": true
                        }),
                    }
                }
                "stats" => match store.stats().await {
                    Ok(stats) => {
                        let json = serde_json::json!({
                            "total_chunks": stats.total_chunks,
                            "chunks_by_level": stats.chunks_by_level,
                            "source_files": stats.source_files,
                        });
                        serde_json::json!({
                            "content": [{
                                "type": "text",
                                "text": serde_json::to_string_pretty(&json).unwrap_or_default()
                            }]
                        })
                    }
                    Err(e) => serde_json::json!({
                        "content": [{ "type": "text", "text": format!("Error: {}", e) }],
                        "isError": true
                    }),
                },
                "promote" | "demote" => {
                    let chunk_id = arguments.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    if chunk_id.is_empty() {
                        return format_mcp_error(id, -32602, "Missing required parameter: id");
                    }
                    let default_vis = if canonical_tool == "promote" {
                        "always"
                    } else {
                        "deep_only"
                    };
                    let vis = arguments
                        .get("visibility")
                        .and_then(|v| v.as_str())
                        .unwrap_or(default_vis);
                    let verb = if canonical_tool == "promote" {
                        "promoted"
                    } else {
                        "demoted"
                    };

                    match store.update_visibility(chunk_id, vis).await {
                        Ok(()) => serde_json::json!({
                            "content": [{
                                "type": "text",
                                "text": format!("Chunk {} {} to visibility '{}'", chunk_id, verb, vis)
                            }]
                        }),
                        Err(e) => serde_json::json!({
                            "content": [{ "type": "text", "text": format!("Error: {}", e) }],
                            "isError": true
                        }),
                    }
                }
                "relate" => {
                    let source_id = arguments
                        .get("source_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let target_id = arguments
                        .get("target_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if source_id.is_empty() || target_id.is_empty() {
                        return format_mcp_error(
                            id,
                            -32602,
                            "Missing required parameters: source_id and target_id",
                        );
                    }
                    let kind = arguments
                        .get("kind")
                        .and_then(|v| v.as_str())
                        .unwrap_or("related_to");

                    let relation = crate::ChunkRelation::new(kind, target_id);
                    match store.add_relation(source_id, relation).await {
                        Ok(()) => serde_json::json!({
                            "content": [{
                                "type": "text",
                                "text": format!("Added relation '{}' from {} to {}", kind, source_id, target_id)
                            }]
                        }),
                        Err(e) => serde_json::json!({
                            "content": [{ "type": "text", "text": format!("Error: {}", e) }],
                            "isError": true
                        }),
                    }
                }
                "think" => {
                    let hot_limit = arguments
                        .get("hot_limit")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(10) as usize;
                    let stale_limit = arguments
                        .get("stale_limit")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(10) as usize;

                    match execute_reflect(store, data_dir, hot_limit, stale_limit).await {
                        Ok(report) => serde_json::json!({
                            "content": [{
                                "type": "text",
                                "text": report
                            }]
                        }),
                        Err(e) => serde_json::json!({
                            "content": [{ "type": "text", "text": format!("Error: {}", e) }],
                            "isError": true
                        }),
                    }
                }
                "store" => {
                    let content = arguments
                        .get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if content.is_empty() {
                        return format_mcp_error(id, -32602, "Missing required parameter: content");
                    }

                    match execute_ingest_chunk(store, embedder, &arguments).await {
                        Ok(chunk_id) => serde_json::json!({
                            "content": [{
                                "type": "text",
                                "text": format!("Chunk ingested successfully. ID: {}", chunk_id)
                            }]
                        }),
                        Err(e) => serde_json::json!({
                            "content": [{ "type": "text", "text": format!("Error: {}", e) }],
                            "isError": true
                        }),
                    }
                }
                "focus" => {
                    let input: FocusInput = match serde_json::from_value(arguments) {
                        Ok(i) => i,
                        Err(e) => {
                            return format_mcp_error(id, -32602, &format!("Invalid params: {}", e));
                        }
                    };

                    match store.get_children(&input.node_id).await {
                        Ok(mut children) => {
                            if let Some(question) = input.question.as_ref() {
                                let lowered = question.to_lowercase();
                                children.sort_by_key(|c| {
                                    let hay = format!(
                                        "{} {} {}",
                                        c.heading.clone().unwrap_or_default(),
                                        c.path,
                                        c.content
                                    )
                                    .to_lowercase();
                                    if hay.contains(&lowered) {
                                        0
                                    } else {
                                        1
                                    }
                                });
                            }
                            let limit = input.limit.max(1);
                            let responses: Vec<ChunkResponse> = children
                                .into_iter()
                                .take(limit)
                                .map(|c| ChunkResponse::from(&c))
                                .collect();
                            serde_json::json!({
                                "content": [{
                                    "type": "text",
                                    "text": serde_json::to_string_pretty(&responses).unwrap_or_default()
                                }]
                            })
                        }
                        Err(e) => serde_json::json!({
                            "content": [{ "type": "text", "text": format!("Error: {}", e) }],
                            "isError": true
                        }),
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

                    serde_json::json!({
                        "content": [{
                            "type": "text",
                            "text": serde_json::to_string(&token).unwrap_or_default()
                        }]
                    })
                }
                "configure_aging" => {
                    let mut config = AgingConfig::load(data_dir);

                    if let Some(days) = arguments.get("degrade_after_days").and_then(|v| v.as_u64())
                    {
                        config.degrade_after_days = days as u32;
                    }
                    if let Some(to) = arguments.get("degrade_to").and_then(|v| v.as_str()) {
                        config.degrade_to = to.to_string();
                    }
                    if let Some(from) = arguments.get("degrade_from").and_then(|v| v.as_array()) {
                        config.degrade_from = from
                            .iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect();
                    }

                    match config.save(data_dir) {
                        Ok(()) => serde_json::json!({
                            "content": [{
                                "type": "text",
                                "text": format!(
                                    "Aging rules configured: degrade {} chunks to '{}' after {} days without access",
                                    config.degrade_from.join(", "),
                                    config.degrade_to,
                                    config.degrade_after_days
                                )
                            }]
                        }),
                        Err(e) => serde_json::json!({
                            "content": [{ "type": "text", "text": format!("Error: {}", e) }],
                            "isError": true
                        }),
                    }
                }
                "apply_aging" => {
                    let config = AgingConfig::load(data_dir);
                    match aging::apply_aging(store.as_ref(), &config).await {
                        Ok(result) => {
                            let text = if result.degraded_count == 0 {
                                "No chunks needed aging. All knowledge is fresh.".to_string()
                            } else {
                                format!(
                                    "Aged {} chunks (degraded to '{}'): {}",
                                    result.degraded_count,
                                    config.degrade_to,
                                    result.degraded_ids.join(", ")
                                )
                            };
                            serde_json::json!({
                                "content": [{ "type": "text", "text": text }]
                            })
                        }
                        Err(e) => serde_json::json!({
                            "content": [{ "type": "text", "text": format!("Error: {}", e) }],
                            "isError": true
                        }),
                    }
                }
                _ => serde_json::json!({
                    "content": [{ "type": "text", "text": format!("Unknown tool: {}", tool_name) }],
                    "isError": true
                }),
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

async fn execute_search(
    store: &Arc<LanceStore>,
    embedder: &Arc<FastEmbedder>,
    input: SearchInput,
) -> Result<Vec<SearchResultResponse>> {
    let recency_window = input
        .recency
        .as_deref()
        .and_then(crate::RecencyWindow::from_str_opt);

    let config = SearchConfig {
        top_k: input.limit,
        children_k: if input.include_children { 3 } else { 0 },
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

    // Superseded but still visible
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
            "- Run `apply_aging` to degrade {} stale chunks automatically\n",
            stale.len()
        ));
        has_suggestions = true;
    }

    for chunk in &hot {
        if chunk.access_profile.total > 10 && chunk.visibility == "normal" {
            report.push_str(&format!(
                "- Consider promoting **{}** (`{}`) — accessed {} times but still 'normal'\n",
                chunk.heading.as_deref().unwrap_or("(no heading)"),
                chunk.id,
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

/// Ingest a single chunk written by the agent.
async fn execute_ingest_chunk(
    store: &Arc<LanceStore>,
    embedder: &Arc<FastEmbedder>,
    arguments: &serde_json::Value,
) -> Result<String> {
    let content = arguments
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let parent_id = arguments
        .get("parent_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty());
    let source_file = arguments
        .get("source_file")
        .and_then(|v| v.as_str())
        .unwrap_or("[agent]");
    let heading = arguments
        .get("heading")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty());
    let visibility = arguments
        .get("visibility")
        .and_then(|v| v.as_str())
        .unwrap_or("normal");

    // Determine the level: if parent given, try to infer child level
    let level = if let Some(pid) = parent_id {
        if let Ok(Some(parent)) = store.get_by_id(pid).await {
            crate::chunk::ChunkLevel(parent.level.0 + 1)
        } else {
            crate::chunk::ChunkLevel(7) // Content level
        }
    } else {
        crate::chunk::ChunkLevel(1) // Top-level
    };

    // Build path
    let path = if let Some(pid) = parent_id {
        if let Ok(Some(parent)) = store.get_by_id(pid).await {
            format!("{}/agent", parent.path)
        } else {
            source_file.to_string()
        }
    } else {
        source_file.to_string()
    };

    // Generate embedding
    let embeddings = embedder.embed(&[content])?;
    let embedding = embeddings
        .into_iter()
        .next()
        .ok_or_else(|| crate::Error::embedding("Failed to generate embedding"))?;

    let chunk_id = uuid::Uuid::new_v4().to_string();

    let chunk = crate::HierarchicalChunk {
        id: chunk_id.clone(),
        content: content.to_string(),
        embedding: Some(embedding),
        level,
        parent_id: parent_id.map(String::from),
        path,
        source_file: source_file.to_string(),
        heading: heading.map(String::from),
        start_offset: 0,
        end_offset: 0,
        cluster_memberships: vec![],
        is_summary: false,
        summarizes: vec![],
        visibility: visibility.to_string(),
        relations: vec![],
        access_profile: crate::AccessProfile::new(),
        expires_at: None,
    };

    store.insert_chunks(vec![chunk]).await?;

    Ok(chunk_id)
}

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
        .route("/api/recall", post(api_search))
        .route("/api/focus", post(api_focus))
        .route("/api/store", post(api_ingest_chunk))
        .route("/api/think", get(api_reflect))
        .route("/api/share", post(api_share))
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

async fn api_focus(
    State(state): State<AppState>,
    Json(input): Json<FocusInput>,
) -> Json<ApiResponse<Vec<ChunkResponse>>> {
    match state.store.get_children(&input.node_id).await {
        Ok(mut children) => {
            if let Some(question) = input.question.as_ref() {
                let lowered = question.to_lowercase();
                children.sort_by_key(|c| {
                    let hay = format!(
                        "{} {} {}",
                        c.heading.clone().unwrap_or_default(),
                        c.path,
                        c.content
                    )
                    .to_lowercase();
                    if hay.contains(&lowered) {
                        0
                    } else {
                        1
                    }
                });
            }

            let rows = children
                .into_iter()
                .take(input.limit.max(1))
                .collect::<Vec<_>>();
            let responses: Vec<ChunkResponse> = rows.iter().map(ChunkResponse::from).collect();
            Json(ApiResponse::Success(responses))
        }
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
    }
}

async fn api_share(Json(input): Json<ShareInput>) -> Json<ApiResponse<serde_json::Value>> {
    Json(ApiResponse::Success(build_share_token(input)))
}

async fn api_search(
    State(state): State<AppState>,
    Json(input): Json<SearchInput>,
) -> Json<ApiResponse<Vec<SearchResultResponse>>> {
    match execute_search(&state.store, &state.embedder, input).await {
        Ok(results) => Json(ApiResponse::Success(results)),
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
    }
}

#[allow(dead_code)]
async fn api_get_chunk(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Json<ApiResponse<ChunkResponse>> {
    match state.store.get_by_id(&id).await {
        Ok(Some(chunk)) => Json(ApiResponse::Success(ChunkResponse::from(&chunk))),
        Ok(None) => Json(ApiResponse::Error {
            error: "Chunk not found".to_string(),
        }),
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
    }
}

#[allow(dead_code)]
async fn api_get_children(
    State(state): State<AppState>,
    axum::extract::Path(parent_id): axum::extract::Path<String>,
) -> Json<ApiResponse<Vec<ChunkResponse>>> {
    match state.store.get_children(&parent_id).await {
        Ok(children) => {
            let responses: Vec<ChunkResponse> = children.iter().map(ChunkResponse::from).collect();
            Json(ApiResponse::Success(responses))
        }
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
    }
}

#[allow(dead_code)]
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

#[allow(dead_code)]
async fn api_promote(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
    Json(input): Json<VisibilityInput>,
) -> Json<ApiResponse<String>> {
    let vis = input.visibility.as_deref().unwrap_or("always");
    match state.store.update_visibility(&id, vis).await {
        Ok(()) => Json(ApiResponse::Success(format!(
            "Chunk {} promoted to visibility '{}'",
            id, vis
        ))),
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
    }
}

#[allow(dead_code)]
async fn api_demote(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
    Json(input): Json<VisibilityInput>,
) -> Json<ApiResponse<String>> {
    let vis = input.visibility.as_deref().unwrap_or("deep_only");
    match state.store.update_visibility(&id, vis).await {
        Ok(()) => Json(ApiResponse::Success(format!(
            "Chunk {} demoted to visibility '{}'",
            id, vis
        ))),
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
    }
}

#[allow(dead_code)]
async fn api_relate(
    State(state): State<AppState>,
    Json(input): Json<RelateInput>,
) -> Json<ApiResponse<String>> {
    if input.source_id.is_empty() || input.target_id.is_empty() {
        return Json(ApiResponse::Error {
            error: "source_id and target_id are required".to_string(),
        });
    }
    let relation = crate::ChunkRelation::new(&input.kind, &input.target_id);
    match state.store.add_relation(&input.source_id, relation).await {
        Ok(()) => Json(ApiResponse::Success(format!(
            "Added relation '{}' from {} to {}",
            input.kind, input.source_id, input.target_id
        ))),
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
    }
}

async fn api_reflect(State(state): State<AppState>) -> Json<ApiResponse<String>> {
    match execute_reflect(&state.store, &state.data_dir, 10, 10).await {
        Ok(report) => Json(ApiResponse::Success(report)),
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
    }
}

async fn api_ingest_chunk(
    State(state): State<AppState>,
    Json(input): Json<IngestChunkInput>,
) -> Json<ApiResponse<String>> {
    if input.content.is_empty() {
        return Json(ApiResponse::Error {
            error: "content is required".to_string(),
        });
    }
    let args = serde_json::json!({
        "content": input.content,
        "parent_id": input.parent_id,
        "source_file": input.source_file,
        "heading": input.heading,
        "visibility": input.visibility,
    });
    match execute_ingest_chunk(&state.store, &state.embedder, &args).await {
        Ok(chunk_id) => Json(ApiResponse::Success(format!(
            "Chunk ingested. ID: {}",
            chunk_id
        ))),
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
    }
}

#[allow(dead_code)]
async fn api_get_aging_config(
    State(state): State<AppState>,
) -> Json<ApiResponse<serde_json::Value>> {
    let config = AgingConfig::load(&state.data_dir);
    Json(ApiResponse::Success(
        serde_json::to_value(&config).unwrap_or_default(),
    ))
}

#[allow(dead_code)]
async fn api_set_aging_config(
    State(state): State<AppState>,
    Json(input): Json<AgingConfigInput>,
) -> Json<ApiResponse<String>> {
    let mut config = AgingConfig::load(&state.data_dir);

    if let Some(days) = input.degrade_after_days {
        config.degrade_after_days = days;
    }
    if let Some(to) = input.degrade_to {
        config.degrade_to = to;
    }
    if let Some(from) = input.degrade_from {
        config.degrade_from = from;
    }

    match config.save(&state.data_dir) {
        Ok(()) => Json(ApiResponse::Success(format!(
            "Aging configured: degrade {} → '{}' after {} days",
            config.degrade_from.join(", "),
            config.degrade_to,
            config.degrade_after_days
        ))),
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
    }
}

#[allow(dead_code)]
async fn api_apply_aging(State(state): State<AppState>) -> Json<ApiResponse<serde_json::Value>> {
    let config = AgingConfig::load(&state.data_dir);
    match aging::apply_aging(state.store.as_ref(), &config).await {
        Ok(result) => Json(ApiResponse::Success(
            serde_json::to_value(&result).unwrap_or_default(),
        )),
        Err(e) => Json(ApiResponse::Error {
            error: e.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::{build_share_token, canonical_tool_name, ShareInput};

    #[test]
    fn canonical_tool_keeps_new_names() {
        assert_eq!(canonical_tool_name("recall"), "recall");
        assert_eq!(canonical_tool_name("focus"), "focus");
        assert_eq!(canonical_tool_name("store"), "store");
        assert_eq!(canonical_tool_name("think"), "think");
        assert_eq!(canonical_tool_name("share"), "share");
    }

    #[test]
    fn share_token_uses_defaults_and_honors_custom_values() {
        let default_token = build_share_token(ShareInput {
            tree: "projects:veclayer".to_string(),
            can: vec![],
            expires: None,
        });
        assert_eq!(default_token["tree"], "projects:veclayer");
        assert_eq!(default_token["can"], serde_json::json!(["recall", "focus"]));
        assert_eq!(default_token["version"], "veclayer-share-v1");
        assert!(default_token["nonce"]
            .as_str()
            .is_some_and(|s| !s.is_empty()));

        let custom_token = build_share_token(ShareInput {
            tree: "projects:veclayer:decisions".to_string(),
            can: vec![
                "recall".to_string(),
                "focus".to_string(),
                "store".to_string(),
            ],
            expires: Some("90d".to_string()),
        });
        assert_eq!(custom_token["tree"], "projects:veclayer:decisions");
        assert_eq!(
            custom_token["can"],
            serde_json::json!(["recall", "focus", "store"])
        );
        assert_eq!(custom_token["expires"], "90d");
    }
}
