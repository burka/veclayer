use std::sync::Arc;

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

use crate::embedder::FastEmbedder;
use crate::search::{HierarchicalSearch, SearchConfig};
use crate::store::LanceStore;
use crate::{Config, Embedder, Result, VectorStore};

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    store: Arc<LanceStore>,
    embedder: Arc<FastEmbedder>,
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

    for line in stdin.lock().lines() {
        let line = line.map_err(crate::Error::Io)?;
        if line.is_empty() {
            continue;
        }

        let response = handle_mcp_message(&line, &store, &embedder).await;
        writeln!(stdout, "{}", response).map_err(crate::Error::Io)?;
        stdout.flush().map_err(crate::Error::Io)?;
    }

    Ok(())
}

async fn handle_mcp_message(
    message: &str,
    store: &Arc<LanceStore>,
    embedder: &Arc<FastEmbedder>,
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
                }
            })
        }
        "tools/list" => {
            serde_json::json!({
                "tools": [
                    {
                        "name": "search",
                        "description": "Search documents using hierarchical vector search",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": { "type": "string", "description": "The search query" },
                                "limit": { "type": "integer", "description": "Number of results", "default": 5 },
                                "include_children": { "type": "boolean", "description": "Include children", "default": false },
                                "deep": { "type": "boolean", "description": "Deep search: include all visibilities (deep_only, expired, custom)", "default": false },
                                "recency": { "type": "string", "description": "Recency window for relevancy boosting: 24h, 7d, or 30d" }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "get_chunk",
                        "description": "Get a specific chunk by ID",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string", "description": "The chunk ID" }
                            },
                            "required": ["id"]
                        }
                    },
                    {
                        "name": "get_children",
                        "description": "Get all children of a chunk",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "parent_id": { "type": "string", "description": "Parent chunk ID" }
                            },
                            "required": ["parent_id"]
                        }
                    },
                    {
                        "name": "stats",
                        "description": "Get statistics about indexed documents",
                        "inputSchema": {
                            "type": "object",
                            "properties": {}
                        }
                    },
                    {
                        "name": "promote",
                        "description": "Promote a chunk's visibility (e.g. to 'always' so it appears in every search)",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string", "description": "The chunk ID" },
                                "visibility": { "type": "string", "description": "New visibility (always, normal, seasonal)", "default": "always" }
                            },
                            "required": ["id"]
                        }
                    },
                    {
                        "name": "demote",
                        "description": "Demote a chunk's visibility (e.g. to 'deep_only' to hide from standard search)",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string", "description": "The chunk ID" },
                                "visibility": { "type": "string", "description": "New visibility", "default": "deep_only" }
                            },
                            "required": ["id"]
                        }
                    },
                    {
                        "name": "relate",
                        "description": "Add a relation between two chunks (e.g. superseded_by, related_to, derived_from)",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "source_id": { "type": "string", "description": "The source chunk ID" },
                                "target_id": { "type": "string", "description": "The target chunk ID" },
                                "kind": { "type": "string", "description": "Relation kind (superseded_by, summarized_by, related_to, derived_from, or custom)", "default": "related_to" }
                            },
                            "required": ["source_id", "target_id"]
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
                "search" => {
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
                "promote" => {
                    let id = arguments.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let vis = arguments
                        .get("visibility")
                        .and_then(|v| v.as_str())
                        .unwrap_or("always");

                    match store.update_visibility(id, vis).await {
                        Ok(()) => serde_json::json!({
                            "content": [{
                                "type": "text",
                                "text": format!("Chunk {} promoted to visibility '{}'", id, vis)
                            }]
                        }),
                        Err(e) => serde_json::json!({
                            "content": [{ "type": "text", "text": format!("Error: {}", e) }],
                            "isError": true
                        }),
                    }
                }
                "demote" => {
                    let id = arguments.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let vis = arguments
                        .get("visibility")
                        .and_then(|v| v.as_str())
                        .unwrap_or("deep_only");

                    match store.update_visibility(id, vis).await {
                        Ok(()) => serde_json::json!({
                            "content": [{
                                "type": "text",
                                "text": format!("Chunk {} demoted to visibility '{}'", id, vis)
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
        recency_alpha: if recency_window.is_some() { 0.3 } else { 0.15 },
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

/// Run the HTTP REST API server
pub async fn run_http(config: Config) -> Result<()> {
    let embedder = FastEmbedder::new()?;
    let dimension = embedder.dimension();
    let store = LanceStore::open(&config.data_dir, dimension).await?;

    let state = AppState {
        store: Arc::new(store),
        embedder: Arc::new(embedder),
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/api/search", post(api_search))
        .route("/api/chunk/{id}", get(api_get_chunk))
        .route("/api/children/{parent_id}", get(api_get_children))
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
