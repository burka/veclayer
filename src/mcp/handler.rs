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

use super::tools::{impl_tool_context, ToolContext};
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
    embedder_config: crate::config::EmbedderConfig,
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
    /// Build a `ToolContext` from the handler's fields.
    fn tool_context(&self) -> ToolContext {
        impl_tool_context!(self)
    }

    /// Build a `McpHandler` from an [`AppState`](super::http::AppState) plus
    /// per-session parameters.
    ///
    /// Preferred over [`new`](Self::new) in the HTTP server, where the state
    /// already carries all store/embedder/project fields.
    #[cfg(feature = "http")]
    pub fn from_state(
        state: &super::http::AppState,
        instructions: String,
        capability: Capability,
    ) -> Self {
        Self::new(
            Arc::clone(&state.store),
            Arc::clone(&state.embedder),
            state.embedder_config.clone(),
            Arc::clone(&state.blob_store),
            state.data_dir.clone(),
            state.project.clone(),
            state.branch.clone(),
            instructions,
            capability,
            state.git_store.clone(),
            state.push_mode,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        store: Arc<StoreBackend>,
        embedder: Arc<dyn Embedder + Send + Sync>,
        embedder_config: crate::config::EmbedderConfig,
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
            embedder_config,
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
        let ctx = self.tool_context();
        match tools::execute_recall(&ctx, input, Some(self.push_mode)).await {
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
        let ctx = self.tool_context();
        match tools::execute_focus(&ctx, input).await {
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

        // Only include git_store when the scope warrants staging and push mode allows it.
        let mut ctx = self.tool_context();
        if !self.push_mode.auto_stages() {
            ctx.git_store = None;
        }
        match tools::execute_store(&ctx, input).await {
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
        let ctx = self.tool_context();
        match tools::execute_think(&ctx, input, Some(self.push_mode)).await {
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
            &self.embedder_config,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use rmcp::handler::server::wrapper::Parameters;

    use super::*;
    use crate::auth::capability::Capability;
    use crate::blob_store::BlobStore;
    use crate::git::branch_config::PushMode;
    use crate::store::StoreBackend;
    use crate::{Embedder, Result};

    // ── Stub embedder ─────────────────────────────────────────────────────────

    struct StubEmbedder;

    impl Embedder for StubEmbedder {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.0f32; 384]).collect())
        }

        fn dimension(&self) -> usize {
            384
        }

        fn name(&self) -> &str {
            "stub"
        }
    }

    /// Build a minimal `McpHandler` backed by a temp-dir store.
    async fn make_handler(
        data_dir: &std::path::Path,
        capability: Capability,
        project: Option<String>,
        branch: Option<String>,
    ) -> McpHandler {
        let store = Arc::new(StoreBackend::open(data_dir, 384, false).await.unwrap());
        let embedder: Arc<dyn Embedder + Send + Sync> = Arc::new(StubEmbedder);
        let blob_store = Arc::new(BlobStore::open(data_dir).unwrap());

        McpHandler::new(
            store,
            embedder,
            crate::config::EmbedderConfig::default(),
            blob_store,
            data_dir.to_path_buf(),
            project,
            branch,
            "test instructions".to_string(),
            capability,
            None,
            PushMode::Off,
        )
    }

    // ── tool_error format ─────────────────────────────────────────────────────

    /// `tool_error` produces a tool-level error result (not a protocol error)
    /// with `is_error = true` and content prefixed with "Error:".
    #[test]
    fn tool_error_produces_error_result() {
        let err = crate::Error::not_found("test entry".to_string());
        let result = tool_error(err).expect("tool_error must not return McpError");

        assert_eq!(
            result.is_error,
            Some(true),
            "is_error flag must be set for tool-level errors"
        );
        assert!(!result.content.is_empty(), "error result must have content");

        let text = result.content[0]
            .as_text()
            .expect("first content item must be text")
            .text
            .clone();
        assert!(
            text.starts_with("Error:"),
            "error message must start with 'Error:' — got: {text}"
        );
    }

    /// `tool_error` preserves the original error message in the content.
    #[test]
    fn tool_error_includes_error_message() {
        let err = crate::Error::InvalidOperation("something went wrong".to_string());
        let result = tool_error(err).unwrap();

        let text = result.content[0].as_text().unwrap().text.clone();
        assert!(
            text.contains("something went wrong"),
            "error message must appear in content — got: {text}"
        );
    }

    // ── tool_context field propagation ────────────────────────────────────────

    /// `tool_context()` forwards every field from the handler without mutation.
    #[tokio::test]
    async fn tool_context_propagates_all_fields() {
        let dir = tempfile::TempDir::new().unwrap();
        let expected_project = Some("myproject".to_string());
        let expected_branch = Some("feature-x".to_string());

        let handler = make_handler(
            dir.path(),
            Capability::Admin,
            expected_project.clone(),
            expected_branch.clone(),
        )
        .await;

        let ctx = handler.tool_context();

        assert_eq!(ctx.project, expected_project, "project must be forwarded");
        assert_eq!(ctx.branch, expected_branch, "branch must be forwarded");
        assert_eq!(
            ctx.data_dir,
            PathBuf::from(dir.path()),
            "data_dir must be forwarded"
        );
        assert_eq!(ctx.push_mode, PushMode::Off, "push_mode must be forwarded");
        assert!(
            ctx.git_store.is_none(),
            "git_store must be None when not configured"
        );
    }

    /// `tool_context()` with no project/branch propagates both as `None`.
    #[tokio::test]
    async fn tool_context_none_project_and_branch() {
        let dir = tempfile::TempDir::new().unwrap();
        let handler = make_handler(dir.path(), Capability::Write, None, None).await;

        let ctx = handler.tool_context();
        assert!(ctx.project.is_none());
        assert!(ctx.branch.is_none());
    }

    // ── Capability enforcement ────────────────────────────────────────────────

    /// A `Read`-capability handler must reject `store` (a write operation) with
    /// a tool-level error, not a protocol error.
    #[tokio::test]
    async fn read_capability_blocks_store() {
        let dir = tempfile::TempDir::new().unwrap();
        let handler = make_handler(dir.path(), Capability::Read, None, None).await;

        let input = super::super::types::StoreInput {
            content: "some content".to_string(),
            parent_id: None,
            source_file: "[agent]".to_string(),
            heading: None,
            visibility: "normal".to_string(),
            perspectives: vec![],
            relations: vec![],
            items: vec![],
            entry_type: None,
            impression_hint: None,
            impression_strength: None,
            scope: "project".to_string(),
        };

        let result = handler
            .store(Parameters(input))
            .await
            .expect("store must return Ok (tool-level error), not a protocol error");

        assert_eq!(
            result.is_error,
            Some(true),
            "read-only handler must return an error result for store"
        );
        let text = result.content[0].as_text().unwrap().text.clone();
        assert!(
            text.contains("permission") || text.contains("Insufficient"),
            "error must mention permission — got: {text}"
        );
    }

    /// A `Read`-capability handler must reject `think` (a write operation).
    #[tokio::test]
    async fn read_capability_blocks_think() {
        let dir = tempfile::TempDir::new().unwrap();
        let handler = make_handler(dir.path(), Capability::Read, None, None).await;

        let input = super::super::types::ThinkInput {
            action: Some("promote".to_string()),
            hot_limit: None,
            stale_limit: None,
            id: Some("abc123".to_string()),
            visibility: Some("always".to_string()),
            source_id: None,
            target_id: None,
            kind: None,
            degrade_after_days: None,
            degrade_to: None,
            degrade_from: None,
            direction: None,
        };

        let result = handler
            .think(Parameters(input))
            .await
            .expect("think must not return a protocol error");

        assert_eq!(
            result.is_error,
            Some(true),
            "read-only handler must block think (write operation)"
        );
    }

    /// A `Write`-capability handler must allow `recall` (a read operation).
    #[tokio::test]
    async fn write_capability_permits_recall() {
        let dir = tempfile::TempDir::new().unwrap();
        let handler = make_handler(dir.path(), Capability::Write, None, None).await;

        let input = super::super::types::RecallInput {
            query: None,
            limit: 5,
            deep: false,
            recency: None,
            perspectives: None,
            similar_to: None,
            min_salience: None,
            min_score: None,
            since: None,
            until: None,
            ongoing: None,
        };

        let result = handler
            .recall(Parameters(input))
            .await
            .expect("recall must not return a protocol error");

        // On an empty store with no query the result must succeed.
        assert_ne!(
            result.is_error,
            Some(true),
            "write-capability handler must not block read operations"
        );
    }
}
