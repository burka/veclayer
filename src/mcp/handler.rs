//! Shared rmcp tool handler for the 5 VecLayer MCP tools.
//!
//! Provides a single [`McpHandler`] struct used by both the stdio and streamable
//! HTTP transports. All tool dispatch and formatting lives here; transports only
//! need to wire up the rmcp service machinery.

use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::Arc;

use rmcp::{
    handler::server::router::tool::ToolRouter, handler::server::wrapper::Parameters, model::*,
    service::RequestContext, tool, tool_handler, tool_router, ErrorData as McpError, RoleServer,
    ServerHandler,
};

use crate::auth::capability::Capability;
use crate::blob_store::BlobStore;
use crate::git::branch_config::PushMode;
use crate::store::StoreBackend;
use crate::Embedder;

use super::types::*;
use super::{format, tools};

/// Convert a domain error into a tool-level error result (not a protocol error).
fn tool_error(e: crate::Error) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::error(vec![Content::text(format!(
        "Error: {e}"
    ))]))
}

/// Shared MCP handler for all 5 VecLayer tools.
///
/// Created once per session. For HTTP each new connection gets a fresh handler
/// (with up-to-date identity priming and project-aware tool descriptions).
/// For stdio there is a single handler for the process lifetime.
#[derive(Clone)]
pub struct McpHandler {
    store: Arc<StoreBackend>,
    embedder: Arc<dyn Embedder + Send + Sync>,
    blob_store: Arc<BlobStore>,
    data_dir: PathBuf,
    project: Option<String>,
    branch: Option<String>,
    /// Instruction text returned in `get_info` (MCP `initialize` response).
    /// Computed from static instructions + identity priming at session creation.
    instructions: String,
    /// Authorization level for this session. Checked before executing each tool.
    capability: Capability,
    /// Git memory store for persisting entries to the memory branch, if configured.
    git_store: Option<Arc<crate::git::memory_store::MemoryStore>>,
    /// Push mode governing how entries are staged/pushed to git.
    push_mode: PushMode,
    tool_router: ToolRouter<Self>,
}

impl McpHandler {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        store: Arc<StoreBackend>,
        embedder: Arc<dyn Embedder + Send + Sync>,
        blob_store: Arc<BlobStore>,
        data_dir: PathBuf,
        project: Option<String>,
        branch: Option<String>,
        instructions: String,
        capability: Capability,
        git_store: Option<Arc<crate::git::memory_store::MemoryStore>>,
        push_mode: PushMode,
    ) -> Self {
        let mut tool_router = Self::tool_router();

        // Inject project/branch context into tool descriptions so agents know
        // which scope they're operating in. The #[tool] macro produces static
        // descriptions; we patch them per-instance (i.e. per session).
        if let Some(proj) = &project {
            let branch_info = branch
                .as_ref()
                .map(|b| format!(" (branch: {b})"))
                .unwrap_or_default();

            if let Some(route) = tool_router.map.get_mut("recall") {
                route.attr.description = Some(Cow::Owned(format!(
                    "Find relevant knowledge within the '{proj}' project{branch_info}. \
                     Results include a relevance tier (strong/moderate/weak/tangential). \
                     Without a query, browse by perspective."
                )));
            }

            // Store tool description varies by push mode
            let store_desc = match push_mode {
                PushMode::Always => format!(
                    "Persist a new memory in the '{proj}' project{branch_info}. \
                     Entries with scope 'project' are automatically shared via git. \
                     Supports relations, entry_type, and batch mode via items array."
                ),
                PushMode::Review | PushMode::PullRequest => format!(
                    "Persist a new memory in the '{proj}' project{branch_info}. \
                     Entries with scope 'project' are staged for team sharing — \
                     the user reviews and pushes them. Ask the user when ready to push. \
                     Supports relations, entry_type, and batch mode via items array."
                ),
                PushMode::Manual => format!(
                    "Persist a new memory in the '{proj}' project{branch_info}. \
                     Entries are local by default. The user promotes entries to \
                     shared git memory manually. \
                     Supports relations, entry_type, and batch mode via items array."
                ),
                PushMode::Off => format!(
                    "Persist a new memory in the '{proj}' project{branch_info}. \
                     Use `scope: \"branch\"` for WIP visible only on this branch. \
                     Supports relations, entry_type, and batch mode via items array."
                ),
            };
            if let Some(route) = tool_router.map.get_mut("store") {
                route.attr.description = Some(Cow::Owned(store_desc));
            }
        }

        Self {
            store,
            embedder,
            blob_store,
            data_dir,
            project,
            branch,
            instructions,
            capability,
            git_store,
            push_mode,
            tool_router,
        }
    }
}

#[tool_router]
impl McpHandler {
    #[tool(
        description = "Find relevant knowledge using semantic search. Results include a relevance tier (strong/moderate/weak/tangential). Without a query, browse by perspective."
    )]
    async fn recall(
        &self,
        Parameters(input): Parameters<RecallInput>,
    ) -> Result<CallToolResult, McpError> {
        if !self.capability.permits(Capability::Read) {
            return Ok(CallToolResult::error(vec![Content::text(
                "Insufficient permission: need read",
            )]));
        }
        let query = input.query.clone();
        match tools::execute_recall(
            &self.store,
            &self.embedder,
            input,
            self.project.as_deref(),
            self.branch.as_deref(),
        )
        .await
        {
            Ok(results) => {
                let text = format::format_recall(query.as_deref(), &results);
                Ok(CallToolResult::success(vec![Content::text(text)]))
            }
            Err(e) => tool_error(e),
        }
    }

    #[tool(
        description = "Dive deeper into a specific memory node. Returns node + children, optionally reranked by question."
    )]
    async fn focus(
        &self,
        Parameters(input): Parameters<FocusInput>,
    ) -> Result<CallToolResult, McpError> {
        if !self.capability.permits(Capability::Read) {
            return Ok(CallToolResult::error(vec![Content::text(
                "Insufficient permission: need read",
            )]));
        }
        match tools::execute_focus(
            &self.store,
            &self.embedder,
            input,
            self.project.as_deref(),
            self.branch.as_deref(),
        )
        .await
        {
            Ok(response) => {
                let text = format::format_focus(&response);
                Ok(CallToolResult::success(vec![Content::text(text)]))
            }
            Err(e) => tool_error(e),
        }
    }

    #[tool(
        description = "Persist a new memory. Supports relations, entry_type, and batch mode via items array."
    )]
    async fn store(
        &self,
        Parameters(input): Parameters<StoreInput>,
    ) -> Result<CallToolResult, McpError> {
        if !self.capability.permits(Capability::Write) {
            return Ok(CallToolResult::error(vec![Content::text(
                "Insufficient permission: need write",
            )]));
        }
        if input.content.is_empty() && input.items.is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(
                "Missing required parameter: content (or items for batch mode)",
            )]));
        }

        // Only pass git_store when the scope warrants staging and push mode allows it.
        let effective_git_store = if self.push_mode.auto_stages() {
            self.git_store.as_deref()
        } else {
            None
        };

        match tools::execute_store(
            &self.store,
            &self.embedder,
            &self.blob_store,
            input,
            self.project.as_deref(),
            self.branch.as_deref(),
            effective_git_store,
            self.push_mode,
        )
        .await
        {
            Ok(result) => {
                let text = result.as_str().unwrap_or_default().to_string();
                Ok(CallToolResult::success(vec![Content::text(text)]))
            }
            Err(e) => tool_error(e),
        }
    }

    #[tool(
        description = "Reflect and curate memory. Without action: reflection report. Actions: promote, demote, relate, configure_aging, apply_aging, salience, consolidate, discover, perspectives, status, history, sync."
    )]
    async fn think(
        &self,
        Parameters(input): Parameters<ThinkInput>,
    ) -> Result<CallToolResult, McpError> {
        if !self.capability.permits(Capability::Write) {
            return Ok(CallToolResult::error(vec![Content::text(
                "Insufficient permission: need write",
            )]));
        }
        match tools::execute_think(
            &self.store,
            &self.data_dir,
            &self.blob_store,
            input,
            self.project.as_deref(),
            self.branch.as_deref(),
            self.git_store.as_deref(),
            Some(self.push_mode),
        )
        .await
        {
            Ok(text) => Ok(CallToolResult::success(vec![Content::text(text)])),
            Err(e) => tool_error(e),
        }
    }

    #[tool(description = "[Experimental] Generate a scoped share-token payload (UCAN preview).")]
    async fn share(
        &self,
        Parameters(input): Parameters<ShareInput>,
    ) -> Result<CallToolResult, McpError> {
        if !self.capability.permits(Capability::Write) {
            return Ok(CallToolResult::error(vec![Content::text(
                "Insufficient permission: need write",
            )]));
        }
        let token = tools::build_share_token(input);
        let text = serde_json::to_string_pretty(&token).unwrap_or_default();
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }
}

#[tool_handler]
impl ServerHandler for McpHandler {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
        )
        .with_server_info(Implementation::new("veclayer", env!("CARGO_PKG_VERSION")))
        .with_instructions(self.instructions.clone())
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        Ok(ListResourcesResult {
            meta: None,
            resources: super::resources::static_resources(),
            next_cursor: None,
        })
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, McpError> {
        Ok(ListResourceTemplatesResult {
            meta: None,
            resource_templates: super::resources::templates(),
            next_cursor: None,
        })
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        super::resources::read(
            &request.uri,
            &self.store,
            &self.data_dir,
            self.project.as_deref(),
            self.branch.as_deref(),
        )
        .await
    }
}
