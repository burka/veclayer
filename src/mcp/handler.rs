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
    // Read by `#[tool_handler]`-generated `call_tool`/`list_tools`; rustc's
    // dead-code pass doesn't trace the macro expansion.
    #[allow(dead_code)]
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
                PushMode::Review => format!(
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
        let requested_limit = input.limit;
        let ctx = self.tool_context();
        match tools::execute_recall(&ctx, input, Some(self.push_mode)).await {
            Ok(results) => {
                let text = format::format_recall(query.as_deref(), &results, requested_limit);
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
        description = "Reflect and curate memory. Without action: reflection report. Actions: promote, demote, relate, configure_aging, apply_aging, salience, consolidate, prepare, discover, perspectives, status, history, sync."
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

    // `share` is intentionally NOT a tool here: UCAN capability-token signing is
    // unimplemented, so advertising it would mislead agents into retrying a tool
    // that always errors. The HTTP layer's `api_share` returns not-implemented
    // for parity; when UCAN signing lands, add `#[tool]` here. The
    // `share_not_advertised_in_tool_list` test guards against re-advertising it
    // prematurely.
}

// Use the per-instance `self.tool_router` (with project/branch context patched
// into descriptions) rather than the macro default `Self::tool_router()`, which
// would rebuild a fresh router from the static `#[tool]` attributes and discard
// our per-session description patches. (rmcp 1.7 changed the default to call
// `Self::tool_router()`; 1.1 read the instance field implicitly.)
#[tool_handler(router = self.tool_router)]
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
        fn embed<'a>(
            &'a self,
            texts: &'a [&'a str],
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + 'a>>
        {
            let result: Vec<Vec<f32>> = texts.iter().map(|_| vec![0.0f32; 384]).collect();
            Box::pin(async move { Ok(result) })
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

    // ── share not advertised ─────────────────────────────────────────────────

    /// `share` must NOT appear in the MCP tool list until UCAN signing is
    /// implemented.  Advertising a tool that always returns an error misleads
    /// agents into attempting it repeatedly.
    #[test]
    fn share_not_advertised_in_tool_list() {
        let router = McpHandler::tool_router();
        assert!(
            router.get("share").is_none(),
            "`share` must not be registered in the tool router until UCAN signing lands"
        );
    }

    // ── recall output shaping ────────────────────────────────────────────────

    /// `recall` with no query on an empty store returns a success result whose
    /// text body is the canonical "No entries found." message.
    #[tokio::test]
    async fn recall_empty_store_returns_no_entries_found() {
        let dir = tempfile::TempDir::new().unwrap();
        let handler = make_handler(dir.path(), Capability::Read, None, None).await;

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

        assert_ne!(
            result.is_error,
            Some(true),
            "empty-store recall must succeed"
        );
        let text = result.content[0].as_text().unwrap().text.clone();
        assert_eq!(
            text, "No entries found.",
            "empty browse must produce canonical message — got: {text}"
        );
    }

    /// `recall` with a query that matches nothing returns "No results for …".
    #[tokio::test]
    async fn recall_query_no_results_returns_no_results_message() {
        let dir = tempfile::TempDir::new().unwrap();
        let handler = make_handler(dir.path(), Capability::Read, None, None).await;

        let input = super::super::types::RecallInput {
            query: Some("xyzzy-missing".to_string()),
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

        assert_ne!(
            result.is_error,
            Some(true),
            "no-results recall must succeed"
        );
        let text = result.content[0].as_text().unwrap().text.clone();
        assert!(
            text.contains("No results for") && text.contains("xyzzy-missing"),
            "no-results recall must mention the query — got: {text}"
        );
    }

    /// `recall` with data produces structured markdown: numbered headings,
    /// metadata blockquotes, and a result footer.
    #[tokio::test]
    async fn recall_with_data_produces_structured_markdown() {
        let dir = tempfile::TempDir::new().unwrap();
        // Pre-populate via store tool, then recall without a query (browse mode)
        let handler = make_handler(dir.path(), Capability::Write, None, None).await;

        let store_input = super::super::types::StoreInput {
            content: "Architecture decision: use Rust".to_string(),
            parent_id: None,
            source_file: "[agent]".to_string(),
            heading: Some("Rust decision".to_string()),
            visibility: "normal".to_string(),
            perspectives: vec!["decisions".to_string()],
            relations: vec![],
            items: vec![],
            entry_type: None,
            impression_hint: None,
            impression_strength: None,
            scope: "project".to_string(),
        };
        let store_result = handler
            .store(Parameters(store_input))
            .await
            .expect("store must not return a protocol error");
        assert_ne!(
            store_result.is_error,
            Some(true),
            "store must succeed before recall test"
        );

        let recall_input = super::super::types::RecallInput {
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
            .recall(Parameters(recall_input))
            .await
            .expect("recall must not return a protocol error");

        assert_ne!(result.is_error, Some(true), "recall with data must succeed");
        let text = result.content[0].as_text().unwrap().text.clone();

        // Numbered header
        assert!(
            text.contains("### 1."),
            "recall output must have numbered entry header — got: {text}"
        );
        // Metadata blockquote line starts with ">"
        assert!(
            text.contains("> `"),
            "recall output must have metadata blockquote — got: {text}"
        );
        // Footer
        assert!(
            text.contains("result(s)"),
            "recall output must have result footer — got: {text}"
        );
        // Content present
        assert!(
            text.contains("Architecture decision"),
            "recall output must include entry content — got: {text}"
        );
    }

    // ── focus output shaping ─────────────────────────────────────────────────

    /// `focus` on a non-existent ID returns a tool-level error mentioning the ID.
    #[tokio::test]
    async fn focus_missing_id_returns_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let handler = make_handler(dir.path(), Capability::Read, None, None).await;

        let input = super::super::types::FocusInput {
            id: "nonexistent-id-xyz".to_string(),
            question: None,
            limit: 10,
        };

        let result = handler
            .focus(Parameters(input))
            .await
            .expect("focus must return Ok (tool-level error), not a protocol error");

        assert_eq!(
            result.is_error,
            Some(true),
            "focus on missing ID must return is_error=true"
        );
        let text = result.content[0].as_text().unwrap().text.clone();
        assert!(
            text.contains("Error:"),
            "focus error must be prefixed with 'Error:' — got: {text}"
        );
    }

    /// `focus` on an existing entry returns structured markdown: heading,
    /// metadata blockquote, content, and a children section.
    #[tokio::test]
    async fn focus_existing_entry_produces_structured_markdown() {
        let dir = tempfile::TempDir::new().unwrap();
        let handler = make_handler(dir.path(), Capability::Write, None, None).await;

        // Store an entry to focus on
        let store_input = super::super::types::StoreInput {
            content: "Focus target content".to_string(),
            parent_id: None,
            source_file: "[agent]".to_string(),
            heading: Some("Focus Heading".to_string()),
            visibility: "normal".to_string(),
            perspectives: vec![],
            relations: vec![],
            items: vec![],
            entry_type: None,
            impression_hint: None,
            impression_strength: None,
            scope: "project".to_string(),
        };
        handler
            .store(Parameters(store_input))
            .await
            .expect("store must succeed");

        // Browse to find the ID
        let recall_input = super::super::types::RecallInput {
            query: None,
            limit: 1,
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
        let recall_result = handler
            .recall(Parameters(recall_input))
            .await
            .expect("recall must succeed");
        let recall_text = recall_result.content[0].as_text().unwrap().text.clone();

        // Extract short ID from the backtick in metadata: "`<id>`"
        let short_id = recall_text
            .lines()
            .find(|l| l.trim_start().starts_with("> `") && l.contains("·"))
            .and_then(|l| {
                let start = l.find('`')? + 1;
                let end = l[start..].find('`')? + start;
                Some(l[start..end].to_string())
            })
            .expect("recall must contain a metadata blockquote with an ID");

        let focus_input = super::super::types::FocusInput {
            id: short_id.clone(),
            question: None,
            limit: 10,
        };
        let result = handler
            .focus(Parameters(focus_input))
            .await
            .expect("focus must not return a protocol error");

        assert_ne!(
            result.is_error,
            Some(true),
            "focus on existing entry must succeed"
        );
        let text = result.content[0].as_text().unwrap().text.clone();

        // Heading
        assert!(
            text.contains("## Focus Heading"),
            "focus output must include the entry heading — got: {text}"
        );
        // Metadata blockquote
        assert!(
            text.contains("> `"),
            "focus output must have metadata blockquote — got: {text}"
        );
        // Content
        assert!(
            text.contains("Focus target content"),
            "focus output must include the entry content — got: {text}"
        );
        // Empty children message (leaf node)
        assert!(
            text.contains("no children"),
            "focus on leaf entry must show no-children message — got: {text}"
        );
    }

    // ── think(reflect) output shaping ────────────────────────────────────────

    /// `think` with no action on an empty store returns the reflection report
    /// with the expected section headings (Hot Chunks, Stale Chunks, Summary,
    /// Suggested Actions) and a "No urgent actions" message.
    #[tokio::test]
    async fn think_reflect_empty_store_produces_full_report() {
        let dir = tempfile::TempDir::new().unwrap();
        // Initialize perspectives so think_perspectives used internally works
        crate::perspective::init(dir.path()).unwrap();
        let handler = make_handler(dir.path(), Capability::Write, None, None).await;

        let input = super::super::types::ThinkInput {
            action: None,
            hot_limit: None,
            stale_limit: None,
            id: None,
            visibility: None,
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

        assert_ne!(
            result.is_error,
            Some(true),
            "reflect on empty store must succeed"
        );
        let text = result.content[0].as_text().unwrap().text.clone();

        assert!(
            text.contains("## Hot Chunks"),
            "reflect report must have Hot Chunks section — got: {text}"
        );
        assert!(
            text.contains("## Stale Chunks"),
            "reflect report must have Stale Chunks section — got: {text}"
        );
        assert!(
            text.contains("## Summary"),
            "reflect report must have Summary section — got: {text}"
        );
        assert!(
            text.contains("## Suggested Actions"),
            "reflect report must have Suggested Actions section — got: {text}"
        );
        assert!(
            text.contains("No urgent actions"),
            "reflect on empty store must say no urgent actions — got: {text}"
        );
    }

    /// `think` reflect on a read-only handler is blocked (write permission required).
    #[tokio::test]
    async fn think_reflect_blocked_without_write_permission() {
        let dir = tempfile::TempDir::new().unwrap();
        let handler = make_handler(dir.path(), Capability::Read, None, None).await;

        let input = super::super::types::ThinkInput {
            action: None,
            hot_limit: None,
            stale_limit: None,
            id: None,
            visibility: None,
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
            "reflect without write permission must return is_error=true"
        );
        let text = result.content[0].as_text().unwrap().text.clone();
        assert!(
            text.contains("permission") || text.contains("Insufficient"),
            "error must mention permission — got: {text}"
        );
    }

    // ── think description completeness ───────────────────────────────────────

    /// Every action in `THINK_ACTIONS` must appear in the live `#[tool]` description.
    ///
    /// This test reads the description directly from the rmcp router — the same
    /// value that the MCP protocol sends to agents — so it catches the class of
    /// drift where the `#[tool(description = "...")]` attribute is edited without
    /// updating `THINK_ACTIONS`, or vice versa. It cannot be defeated by editing a
    /// separate copy of the string.
    ///
    /// If this test fails, update the `#[tool(description = "...")]` attribute on
    /// `McpHandler::think` to list every entry in `tools::THINK_ACTIONS`.
    #[test]
    fn think_description_covers_all_actions() {
        let router = McpHandler::tool_router();
        let tool = router
            .get("think")
            .expect("think tool must be registered in the router");
        let desc = tool
            .description
            .as_deref()
            .expect("think tool must have a description");
        for action in tools::THINK_ACTIONS {
            assert!(
                desc.contains(*action),
                "think #[tool] description is missing action: {action}\n\
                 Update the #[tool(description = \"...\")] attribute on McpHandler::think."
            );
        }
    }
}
