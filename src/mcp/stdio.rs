//! MCP stdio transport for Claude Desktop integration.

use std::sync::Arc;

use tracing::{info, warn};

use crate::blob_store::BlobStore;
use crate::embedder::FastEmbedder;
use crate::store::StoreBackend;
use crate::{Config, Embedder, Result};

use super::tools;
use super::types::*;
use super::MCP_INSTRUCTIONS;

/// Run the MCP server on stdio.
pub async fn run_stdio(config: Config) -> Result<()> {
    use std::io::{BufRead, Write};

    info!("Starting MCP stdio server...");

    let embedder = FastEmbedder::new()?;
    let dimension = embedder.dimension();
    let store = StoreBackend::open(&config.data_dir, dimension, config.read_only).await?;
    let store = Arc::new(store);
    let embedder = Arc::new(embedder);
    let blob_store = BlobStore::open(&config.data_dir)?;
    let blob_store = Arc::new(blob_store);

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    let data_dir = config.data_dir.clone();
    let project = config.project.clone();
    let branch = config.branch.clone();

    for line in stdin.lock().lines() {
        let line = line.map_err(crate::Error::Io)?;
        if line.is_empty() {
            continue;
        }

        let response = handle_mcp_message(
            &line,
            &store,
            &embedder,
            &blob_store,
            &data_dir,
            project.as_deref(),
            branch.as_deref(),
        )
        .await;
        if !response.is_empty() {
            writeln!(stdout, "{}", response).map_err(crate::Error::Io)?;
            stdout.flush().map_err(crate::Error::Io)?;
        }
    }

    Ok(())
}

async fn handle_mcp_message(
    message: &str,
    store: &Arc<StoreBackend>,
    embedder: &Arc<FastEmbedder>,
    blob_store: &Arc<BlobStore>,
    data_dir: &std::path::Path,
    project: Option<&str>,
    branch: Option<&str>,
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
            let priming =
                match crate::identity::compute_identity(store.as_ref(), data_dir, project, branch)
                    .await
                {
                    Ok(snapshot) => {
                        let p = crate::identity::generate_priming(&snapshot);
                        super::build_priming_text(&p)
                    }
                    Err(e) => {
                        warn!("Identity priming failed, using static instructions: {}", e);
                        MCP_INSTRUCTIONS.to_string()
                    }
                };
            serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": {
                    "name": "veclayer",
                    "version": env!("CARGO_PKG_VERSION")
                },
                "instructions": priming
            })
        }
        "tools/list" => tool_list(project, branch),
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

            return handle_tool_call(
                id, tool_name, arguments, store, embedder, blob_store, data_dir, project, branch,
            )
            .await;
        }
        "notifications/initialized" | "initialized" => {
            return String::new();
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

#[allow(clippy::too_many_arguments)]
async fn handle_tool_call(
    id: Option<serde_json::Value>,
    tool_name: &str,
    arguments: serde_json::Value,
    store: &Arc<StoreBackend>,
    embedder: &Arc<FastEmbedder>,
    blob_store: &Arc<BlobStore>,
    data_dir: &std::path::Path,
    project: Option<&str>,
    branch: Option<&str>,
) -> String {
    let result = match tool_name {
        "recall" => {
            let input: RecallInput = match serde_json::from_value(arguments) {
                Ok(i) => i,
                Err(e) => {
                    return format_mcp_error(id, -32602, &format!("Invalid params: {}", e));
                }
            };
            let query = input.query.clone();
            match tools::execute_recall(store, embedder, input, project, branch).await {
                Ok(results) => {
                    let text = super::format::format_recall(query.as_deref(), &results);
                    mcp_text_result(&text)
                }
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
            match tools::execute_focus(store, embedder, input, project, branch).await {
                Ok(response) => {
                    let text = super::format::format_focus(&response);
                    mcp_text_result(&text)
                }
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
            if input.content.is_empty() && input.items.is_empty() {
                return format_mcp_error(
                    id,
                    -32602,
                    "Missing required parameter: content (or items for batch mode)",
                );
            }
            match tools::execute_store(store, embedder, blob_store, input, project, branch).await {
                Ok(result) => {
                    let text = if result.is_array() {
                        format!(
                            "Stored {} entries. IDs: {}",
                            result.as_array().map(|a| a.len()).unwrap_or(0),
                            serde_json::to_string_pretty(&result).unwrap_or_default()
                        )
                    } else {
                        format!("Stored. ID: {}", result.as_str().unwrap_or_default())
                    };
                    mcp_text_result(&text)
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
            match tools::execute_think(store, data_dir, blob_store, input, project, branch).await {
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
            let token = tools::build_share_token(input);
            mcp_text_result(&serde_json::to_string_pretty(&token).unwrap_or_default())
        }
        _ => mcp_error_result(&format!(
            "Unknown tool: {}. Available: recall, focus, store, think, share",
            tool_name
        )),
    };

    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    })
    .to_string()
}

fn tool_list(project: Option<&str>, branch: Option<&str>) -> serde_json::Value {
    let (store_desc, recall_desc) = if let Some(proj) = project {
        let branch_info = branch
            .map(|b| format!(" (branch: {})", b))
            .unwrap_or_default();
        (
            format!(
                "Persist a new memory. Default scope is the '{}' project{}. \
                 Use `scope: \"branch\"` for work-in-progress that only this branch should see. \
                 Use `scope: \"personal\"` for knowledge that follows you across all projects. \
                 Use `scope: \"project\"` for decisions and conventions visible to all branches.",
                proj, branch_info
            ),
            format!(
                "Find relevant knowledge within the '{}' project{}. \
                 Results include a relevance tier. Without a query, browse by perspective.",
                proj, branch_info
            ),
        )
    } else {
        (
            "Persist a new memory. Server generates embeddings automatically. Supports inline relations.".to_string(),
            "Find relevant knowledge using semantic vector search. Results include a relevance tier (strong/moderate/weak/tangential). Without a query, browse entries by perspective.".to_string(),
        )
    };

    serde_json::json!({
        "tools": [
            {
                "name": "recall",
                "description": recall_desc,
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "What to search for (optional — omit to browse)" },
                        "similar_to": { "type": "string", "description": "Search for entries similar to this entry ID (uses its embedding as query vector). When set, 'query' is ignored." },
                        "limit": { "type": "integer", "default": 5 },
                        "deep": { "type": "boolean", "default": false },
                        "recency": { "type": "string", "description": "Recency boost: 24h, 7d, 30d" },
                        "perspective": { "type": "string", "description": "Filter by perspective: intentions, people, temporal, knowledge, decisions, learnings, session" },
                        "since": { "type": "string", "description": "Filter: entries created after (ISO 8601 date or epoch seconds)" },
                        "until": { "type": "string", "description": "Filter: entries created before (ISO 8601 date or epoch seconds)" },
                        "ongoing": { "type": "boolean", "description": "Filter to open threads — unresolved items needing attention" }
                    }
                }
            },
            {
                "name": "focus",
                "description": "Dive deeper into a specific memory node. Returns node + children, optionally reranked by question.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "question": { "type": "string" },
                        "limit": { "type": "integer", "default": 10 }
                    },
                    "required": ["id"]
                }
            },
            {
                "name": "store",
                "description": store_desc,
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": { "type": "string" },
                        "parent_id": { "type": "string" },
                        "source_file": { "type": "string", "default": "[agent]" },
                        "heading": { "type": "string" },
                        "visibility": { "type": "string", "default": "normal" },
                        "perspectives": { "type": "array", "items": { "type": "string" }, "description": "Perspectives: intentions, people, temporal, knowledge, decisions, learnings, session" },
                        "entry_type": { "type": "string", "description": "Entry type: raw (default), summary, meta, impression" },
                        "scope": { "type": "string", "enum": ["project", "branch", "personal"], "default": "project", "description": "project = visible on all branches, branch = only this branch, personal = follows you everywhere" },
                        "relations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "kind": { "type": "string", "description": "supersedes, summarizes, related_to, derived_from, version_of" },
                                    "target_id": { "type": "string" }
                                },
                                "required": ["kind", "target_id"]
                            },
                            "description": "Relations to establish atomically when storing"
                        },
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": { "type": "string" },
                                    "parent_id": { "type": "string" },
                                    "heading": { "type": "string" },
                                    "visibility": { "type": "string", "default": "normal" },
                                    "perspectives": { "type": "array", "items": { "type": "string" } },
                                    "source_file": { "type": "string" },
                                    "entry_type": { "type": "string", "description": "raw (default), summary, meta, impression" },
                                    "scope": { "type": "string", "enum": ["project", "branch", "personal"], "default": "project", "description": "project = visible on all branches, branch = only this branch, personal = follows you everywhere" },
                                    "relations": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "kind": { "type": "string" },
                                                "target_id": { "type": "string" }
                                            },
                                            "required": ["kind", "target_id"]
                                        }
                                    }
                                },
                                "required": ["content"]
                            },
                            "description": "Batch mode: array of entries. When present, top-level fields are ignored."
                        }
                    }
                }
            },
            {
                "name": "think",
                "description": "Reflect and curate memory. Without action: reflection report. Actions: promote, demote, relate, configure_aging, apply_aging, salience, consolidate, discover (find similar-but-unlinked entries), perspectives (list all), status (store stats), history (entry relations).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action": { "type": "string" },
                        "hot_limit": { "type": "integer", "default": 10, "description": "Max hot entries to show (for review action, default 10)" },
                        "stale_limit": { "type": "integer", "default": 10, "description": "Max stale entries to show (for review action, default 5)" },
                        "id": { "type": "string", "description": "Target entry ID (for promote, demote, history, archive, link, unlink)" },
                        "visibility": { "type": "string", "description": "New visibility 0.0-1.0 (for promote/demote actions)" },
                        "source_id": { "type": "string", "description": "Source entry ID (for link/unlink actions)" },
                        "target_id": { "type": "string", "description": "Target entry ID (for link/unlink actions)" },
                        "kind": { "type": "string", "description": "Relation kind (for link/unlink actions)" },
                        "degrade_after_days": { "type": "integer", "description": "Days before degradation starts (for age-config action)" },
                        "degrade_to": { "type": "string", "description": "Target visibility after degradation (for age-config action)" },
                        "degrade_from": { "type": "array", "items": { "type": "string" }, "description": "Source visibility threshold for degradation (for age-config action)" }
                    }
                }
            },
            {
                "name": "share",
                "description": "[Experimental — not yet functional] Generate a scoped share-token payload (UCAN preview).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tree": { "type": "string" },
                        "can": { "type": "array", "items": { "type": "string" } },
                        "expires": { "type": "string" }
                    },
                    "required": ["tree"]
                }
            }
        ]
    })
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
