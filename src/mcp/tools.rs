//! Tool implementation functions for the 5 MCP tools.

use std::sync::Arc;

const THINK_ACTIONS: &[&str] = &[
    "promote",
    "demote",
    "relate",
    "configure_aging",
    "apply_aging",
    "salience",
    "consolidate",
    "prepare",
    "discover",
    "perspectives",
    "status",
    "history",
    "sync",
];

use crate::aging::{self, AgingConfig};
use crate::search::{HierarchicalSearch, SearchConfig, TEMPORAL_PREFETCH_FACTOR};
use crate::store::StoreBackend;
use crate::{Embedder, Result, VectorStore};

/// Maximum number of results any single tool call may request. Caller-supplied
/// limits are clamped to this bound so one request cannot drive unbounded
/// allocation or an integer overflow in downstream fetch-size arithmetic.
const MAX_RESULT_LIMIT: usize = 1000;

/// Clamp a caller-supplied result limit to [`MAX_RESULT_LIMIT`].
fn clamp_result_limit(requested: usize) -> usize {
    requested.min(MAX_RESULT_LIMIT)
}

/// Store fetch size for recall: when a time filter is active the limit is
/// over-fetched by [`TEMPORAL_PREFETCH_FACTOR`] (saturating, so a large limit
/// cannot overflow), otherwise it is used as-is.
fn temporal_fetch_limit(limit: usize, since: Option<i64>, until: Option<i64>) -> usize {
    if since.is_some() || until.is_some() {
        limit.saturating_mul(TEMPORAL_PREFETCH_FACTOR)
    } else {
        limit
    }
}

/// Resolve the perspective list for a `store` request given the requested
/// `scope`, appending the `project:`/`branch:` facets implied by the scope.
///
/// An unrecognized scope is rejected rather than silently treated as
/// `personal` — a typo must not widen an entry's visibility past the project
/// or branch the caller intended.
fn resolve_scope_perspectives(
    scope: &str,
    base: &[String],
    project: Option<&str>,
    branch: Option<&str>,
) -> Result<Vec<String>> {
    let mut perspectives = base.to_vec();
    match scope {
        "project" => {
            if let Some(proj) = project {
                perspectives.push(format!("project:{proj}"));
            }
        }
        "branch" => {
            if let Some(proj) = project {
                perspectives.push(format!("project:{proj}"));
            }
            if let Some(br) = branch {
                match project {
                    Some(proj) => perspectives.push(format!("branch:{proj}@{br}")),
                    None => perspectives.push(format!("branch:{br}")),
                }
            }
        }
        "personal" => {}
        other => {
            return Err(crate::Error::config(format!(
                "unknown scope '{other}', expected one of: project, branch, personal"
            )));
        }
    }
    Ok(perspectives)
}

use super::types::*;

/// Shared execution context passed to all tool functions.
///
/// Groups the stable dependencies that every tool needs, eliminating repeated
/// 7-9 parameter lists on each public `execute_*` function.
///
/// Holds `Arc` references so the context is cheap to clone and can be built
/// once per handler call without lifetime constraints.
#[derive(Clone)]
pub struct ToolContext {
    pub store: Arc<StoreBackend>,
    pub embedder: Arc<dyn Embedder + Send + Sync>,
    pub blob_store: Arc<crate::blob_store::BlobStore>,
    pub data_dir: std::path::PathBuf,
    pub project: Option<String>,
    pub branch: Option<String>,
    pub git_store: Option<Arc<crate::git::memory_store::MemoryStore>>,
    pub push_mode: crate::git::branch_config::PushMode,
}

impl ToolContext {
    /// Construct a `ToolContext` from its constituent parts.
    ///
    /// Both [`McpHandler`](super::handler::McpHandler) and
    /// [`AppState`](super::http::AppState) delegate to this to avoid duplicating
    /// field-by-field construction.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        store: Arc<StoreBackend>,
        embedder: Arc<dyn Embedder + Send + Sync>,
        blob_store: Arc<crate::blob_store::BlobStore>,
        data_dir: std::path::PathBuf,
        project: Option<String>,
        branch: Option<String>,
        git_store: Option<Arc<crate::git::memory_store::MemoryStore>>,
        push_mode: crate::git::branch_config::PushMode,
    ) -> Self {
        Self {
            store,
            embedder,
            blob_store,
            data_dir,
            project,
            branch,
            git_store,
            push_mode,
        }
    }
}

/// Macro to generate a `tool_context()` method that calls `ToolContext::from_parts`
/// with Arc-cloned fields. Eliminates copy-paste between McpHandler and AppState.
macro_rules! impl_tool_context {
    ($self:expr) => {
        ToolContext::from_parts(
            Arc::clone(&$self.store),
            Arc::clone(&$self.embedder),
            Arc::clone(&$self.blob_store),
            $self.data_dir.clone(),
            $self.project.clone(),
            $self.branch.clone(),
            $self.git_store.clone(),
            $self.push_mode,
        )
    };
}
pub(crate) use impl_tool_context;

/// Helper: check if a chunk passes project filter.
/// Returns true if:
/// - No project is set (no filtering)
/// - Chunk has the project perspective `project:<name>`
/// - Chunk has no project perspective (personal/unscoped)
pub(super) fn passes_scope_filter(
    chunk: &crate::HierarchicalChunk,
    project: Option<&str>,
    branch: Option<&str>,
) -> bool {
    let Some(proj_name) = project else {
        return true;
    };

    let project_tag = format!("project:{}", proj_name);
    let has_project = chunk.perspectives.contains(&project_tag);
    let has_any_project = chunk.perspectives.iter().any(|p| p.starts_with("project:"));
    let has_any_branch = chunk.perspectives.iter().any(|p| p.starts_with("branch:"));

    if !has_any_project && !has_any_branch {
        return true;
    }

    if !has_project {
        return false;
    }

    if has_any_branch {
        if let Some(br) = branch {
            let branch_tag = format!("branch:{}@{}", proj_name, br);
            return chunk.perspectives.contains(&branch_tag);
        }
        return false;
    }

    true
}

/// Apply temporal, scope, and ongoing filters to a search result.
///
/// Returns `true` when the result passes all active filters and should be
/// included in the response.
fn apply_recall_filters(
    result: &crate::search::HierarchicalSearchResult,
    project: Option<&str>,
    branch: Option<&str>,
    since_epoch: Option<i64>,
    until_epoch: Option<i64>,
    open_thread_ids: &Option<std::collections::HashSet<String>>,
) -> bool {
    passes_scope_filter(&result.chunk, project, branch) && {
        let created = result.chunk.access_profile.created_at;
        since_epoch.is_none_or(|s| created >= s)
            && until_epoch.is_none_or(|u| created <= u)
            && crate::identity::passes_ongoing_filter(open_thread_ids, &result.chunk.id)
    }
}

/// Helper: map HierarchicalSearchResult to SearchResultResponse
fn map_search_results(
    results: Vec<crate::search::HierarchicalSearchResult>,
) -> Vec<SearchResultResponse> {
    results
        .into_iter()
        .map(|r| SearchResultResponse {
            chunk: ChunkResponse::from(&r.chunk),
            score: r.score,
            relevance: relevance_tier(r.score).to_string(),
            hierarchy_path: r.hierarchy_path.iter().map(ChunkResponse::from).collect(),
            children: r
                .relevant_children
                .iter()
                .map(|c| ChunkResponse::from(&c.chunk))
                .collect(),
        })
        .collect()
}

/// Internal struct carrying all fields needed to store a single entry.
struct StoreSingleInput {
    content: String,
    parent_id: Option<String>,
    source_file: String,
    heading: Option<String>,
    visibility: String,
    perspectives: Vec<String>,
    entry_type: Option<String>,
    relations: Vec<StoreRelation>,
    impression_hint: Option<String>,
    impression_strength: Option<f32>,
    scope: String,
}

/// Write the entry and its embedding to the git memory branch.
///
/// Returns a status message describing what happened. Git failures are non-fatal —
/// they return an error description but don't block the main store operation.
fn commit_to_git(
    entry: &crate::entry::Entry,
    embedding: Option<&[f32]>,
    embedder_name: &str,
    git_store: Option<&crate::git::memory_store::MemoryStore>,
    push_mode: crate::git::branch_config::PushMode,
) -> Option<String> {
    let store = git_store?;

    if let Err(e) = store.store_entry(entry) {
        let msg = format!("⚠ Git staging failed: {e}");
        tracing::warn!("{msg}");
        return Some(msg);
    }

    if let Some(emb) = embedding {
        if let Err(e) = store.store_embedding(entry, embedder_name, emb) {
            tracing::warn!("⚠ Failed to cache embedding in git: {e}");
        }
    }

    if push_mode.auto_pushes() {
        Some("Shared via git.".to_string())
    } else {
        Some("Staged for sharing (pending review). Ask the user if ready to push.".to_string())
    }
}

/// Validate that impression_strength is in the [0.0, 1.0] range.
fn validate_impression_strength(value: Option<f32>) -> Result<()> {
    if let Some(v) = value {
        if !(0.0..=1.0).contains(&v) {
            return Err(crate::Error::parse(format!(
                "impression_strength must be between 0.0 and 1.0, got {v}"
            )));
        }
    }
    Ok(())
}

/// Store a single entry and return (chunk_id, git_status).
/// Pass `Some(embedding)` for immediate embedding, or `None` for deferred (pending).
async fn store_single_entry(
    ctx: &ToolContext,
    input: StoreSingleInput,
    embedding: Option<Vec<f32>>,
) -> Result<(String, Option<String>)> {
    let store = &ctx.store;
    let embedder = &ctx.embedder;
    let blob_store = &ctx.blob_store;
    let project = ctx.project.as_deref();
    let branch = ctx.branch.as_deref();
    let git_store = ctx.git_store.as_deref();
    let push_mode = ctx.push_mode;
    let parent_id = input.parent_id.as_deref().filter(|s| !s.is_empty());

    let (level, path) = if let Some(pid) = parent_id {
        if let Ok(Some(parent)) = store.get_by_id_prefix(pid).await {
            (
                crate::chunk::ChunkLevel(parent.level.0 + 1),
                format!("{}/agent", parent.path),
            )
        } else {
            (crate::chunk::ChunkLevel(7), input.source_file.clone())
        }
    } else {
        (crate::chunk::ChunkLevel(1), input.source_file.clone())
    };

    let entry_type = match input.entry_type.as_deref() {
        None => crate::chunk::EntryType::default(),
        Some(s) => s.parse().map_err(|e: String| crate::Error::config(e))?,
    };

    let perspectives =
        resolve_scope_perspectives(&input.scope, &input.perspectives, project, branch)?;

    let mut chunk = crate::HierarchicalChunk::new(
        input.content,
        level,
        parent_id.map(String::from),
        path,
        input.source_file,
    )
    .with_entry_type(entry_type)
    .with_perspectives(perspectives)
    .with_visibility(input.visibility);

    if let Some(emb) = embedding {
        chunk = chunk.with_embedding(emb);
    }
    if let Some(ref h) = input.heading {
        chunk = chunk.with_heading(h);
    }
    chunk.impression_hint = input.impression_hint;
    chunk.impression_strength = input.impression_strength.unwrap_or(1.0);

    let chunk_id = chunk.id.clone();

    // Persist to blob store
    let blob = crate::entry::StoredBlob::from_chunk_and_embedding(&chunk, embedder.name());
    blob_store.put(&blob)?;

    store.insert_chunks(vec![chunk.clone()]).await?;

    // Write to git memory branch only for project-scoped entries.
    let git_status = if input.scope == "project" {
        let entry = crate::entry::Entry::from_chunk(&chunk);
        commit_to_git(
            &entry,
            chunk.embedding.as_deref(),
            embedder.name(),
            git_store,
            push_mode,
        )
    } else {
        None
    };

    // Process relations via shared module (resolves IDs, writes inverses, auto-demotes)
    let raw_relations: Vec<crate::relations::RawRelation> = input
        .relations
        .iter()
        .map(|r| crate::relations::RawRelation {
            kind: r.kind.clone(),
            target_id: r.target_id.clone(),
        })
        .collect();
    crate::relations::process_relations(store, &chunk_id, raw_relations).await?;

    Ok((chunk_id, git_status))
}

pub async fn execute_recall(
    ctx: &ToolContext,
    input: RecallInput,
    push_mode: Option<crate::git::branch_config::PushMode>,
) -> Result<Vec<SearchResultResponse>> {
    let store = &ctx.store;
    let embedder = &ctx.embedder;
    let project = ctx.project.as_deref();
    let branch = ctx.branch.as_deref();
    let git_store = ctx.git_store.as_deref();
    let limit = clamp_result_limit(input.limit);
    // PushMode::Always implies bidirectional continuous sync — pull before
    // recall to serve the freshest data. Non-blocking: recall proceeds with
    // local data if the pull fails or conflicts.
    if let (Some(git), Some(pm)) = (git_store, push_mode) {
        if pm.auto_pushes() {
            match git.pull() {
                Ok(crate::git::SyncResult::Success) => {
                    tracing::debug!("Pre-recall pull: fetched new entries from remote");
                }
                Ok(crate::git::SyncResult::NothingToSync) => {}
                Ok(crate::git::SyncResult::Conflicts(files)) => {
                    tracing::warn!(
                        "Pre-recall pull: rebase conflict in {}; continuing with local data",
                        files.join(", ")
                    );
                }
                Err(e) => {
                    tracing::warn!("Pre-recall pull failed (continuing with local data): {e}");
                }
            }
        }
    }

    let since_epoch = input
        .since
        .as_deref()
        .and_then(crate::resolve::parse_temporal);
    let until_epoch = input
        .until
        .as_deref()
        .and_then(crate::resolve::parse_temporal);

    let open_thread_ids =
        crate::identity::resolve_ongoing_filter(store.as_ref(), input.ongoing == Some(true))
            .await?;

    if let Some(ref target_id) = input.similar_to {
        let fetch_limit = temporal_fetch_limit(limit, since_epoch, until_epoch);
        let config =
            SearchConfig::try_for_query(fetch_limit, input.deep, input.recency.as_deref())?
                .with_perspectives(input.perspectives.clone().unwrap_or_default())
                .with_min_salience(input.min_salience)
                .with_min_score(input.min_score);
        let search =
            HierarchicalSearch::new(Arc::clone(store), Arc::clone(embedder)).with_config(config);
        let results = search.search_by_embedding(target_id, fetch_limit).await?;

        let filtered: Vec<_> = results
            .into_iter()
            .filter(|r| {
                apply_recall_filters(
                    r,
                    project,
                    branch,
                    since_epoch,
                    until_epoch,
                    &open_thread_ids,
                )
            })
            .take(limit)
            .collect();

        return Ok(map_search_results(filtered));
    }

    match input.query {
        Some(ref query) if !query.is_empty() => {
            // Semantic search path with keyword fallback
            let fetch_limit = temporal_fetch_limit(limit, since_epoch, until_epoch);
            let config =
                SearchConfig::try_for_query(fetch_limit, input.deep, input.recency.as_deref())?
                    .with_perspectives(input.perspectives.clone().unwrap_or_default())
                    .with_min_salience(input.min_salience)
                    .with_min_score(input.min_score);
            let search = HierarchicalSearch::new(Arc::clone(store), Arc::clone(embedder))
                .with_config(config);
            let results = match search.search(query).await {
                Ok(r) => r,
                Err(e) if e.is_embedding() => {
                    tracing::warn!("Embedding unavailable, falling back to keyword search: {e}");
                    search.search_text_fallback(query).await?
                }
                Err(e) => return Err(e),
            };

            let filtered: Vec<_> = results
                .into_iter()
                .filter(|r| {
                    apply_recall_filters(
                        r,
                        project,
                        branch,
                        since_epoch,
                        until_epoch,
                        &open_thread_ids,
                    )
                })
                .take(limit)
                .collect();

            Ok(map_search_results(filtered))
        }
        _ => {
            // Browse mode: list entries without vector search
            let needs_client_filter = open_thread_ids.is_some() || project.is_some();
            let perspectives_refs: Vec<&str> = input
                .perspectives
                .as_deref()
                .unwrap_or_default()
                .iter()
                .map(String::as_str)
                .collect();
            let entries = store
                .list_entries(
                    &perspectives_refs,
                    since_epoch,
                    until_epoch,
                    if needs_client_filter {
                        // Over-fetch when client-side filtering is active (ongoing, project)
                        // because list_entries doesn't support these filters natively.
                        10_000
                    } else {
                        limit
                    },
                )
                .await?;

            Ok(entries
                .iter()
                .filter(|chunk| {
                    passes_scope_filter(chunk, project, branch)
                        && crate::identity::passes_ongoing_filter(&open_thread_ids, &chunk.id)
                })
                .take(limit)
                .map(|chunk| SearchResultResponse {
                    chunk: ChunkResponse::from(chunk),
                    score: 1.0,
                    relevance: "browse".to_string(),
                    hierarchy_path: vec![],
                    children: vec![],
                })
                .collect())
        }
    }
}

pub async fn execute_focus(ctx: &ToolContext, input: FocusInput) -> Result<FocusResponse> {
    let store = &ctx.store;
    let embedder = &ctx.embedder;
    let project = ctx.project.as_deref();
    let branch = ctx.branch.as_deref();
    let limit = clamp_result_limit(input.limit);
    let node = store
        .get_by_id_prefix(&input.id)
        .await?
        .ok_or_else(|| crate::Error::not_found(format!("Chunk not found: {}", input.id)))?;

    let children = store.get_children(&node.id).await?;

    let focus_children = if let Some(ref question) = input.question {
        let question_embedding = embedder.embed(&[question.as_str()])?;
        let question_vec = question_embedding
            .into_iter()
            .next()
            .ok_or_else(|| crate::Error::embedding("Failed to embed question"))?;

        let mut scored: Vec<(crate::HierarchicalChunk, f32)> = children
            .into_iter()
            .filter(|child| passes_scope_filter(child, project, branch))
            .map(|child| {
                let score = child
                    .embedding
                    .as_ref()
                    .map(|emb| crate::search::cosine_similarity(&question_vec, emb))
                    .unwrap_or(0.0);
                (child, score)
            })
            .collect();

        crate::chunk::sort_f32_desc(&mut scored, |r| r.1);
        scored.truncate(limit);

        scored
            .into_iter()
            .map(|(chunk, score)| FocusChild {
                chunk: ChunkResponse::from(&chunk),
                relevance: Some(score),
            })
            .collect()
    } else {
        children
            .into_iter()
            .filter(|child| passes_scope_filter(child, project, branch))
            .take(limit)
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

pub async fn execute_store(ctx: &ToolContext, input: StoreInput) -> Result<serde_json::Value> {
    // Validate impression_strength on the top-level input and on all batch items.
    validate_impression_strength(input.impression_strength)?;
    for item in &input.items {
        validate_impression_strength(item.impression_strength)?;
    }

    if !input.items.is_empty() {
        let mut ids = Vec::new();
        let mut long_entries = 0usize;
        let mut git_statuses: Vec<String> = Vec::new();
        for item in input.items {
            let content_len = item.content.len();
            let (id, git_status) = store_single_entry(
                ctx,
                StoreSingleInput {
                    content: item.content,
                    parent_id: item.parent_id,
                    source_file: item.source_file.unwrap_or_else(|| "[agent]".to_string()),
                    heading: item.heading,
                    visibility: item.visibility,
                    perspectives: item.perspectives,
                    entry_type: item.entry_type,
                    relations: item.relations,
                    impression_hint: item.impression_hint,
                    impression_strength: item.impression_strength,
                    scope: item.scope,
                },
                None, // deferred — background worker will embed
            )
            .await?;
            ids.push(crate::short_id(&id).to_string());
            if let Some(status) = git_status {
                git_statuses.push(status);
            }
            if content_len > 2000 {
                long_entries += 1;
            }
        }
        let mut msg = format!(
            "Stored {} entries. IDs: {}. Embeddings are being computed in the background \
             — entries become searchable as they complete.",
            ids.len(),
            ids.join(", ")
        );
        // Deduplicate git statuses (typically all identical for the same push mode)
        git_statuses.dedup();
        if git_statuses.len() == 1 {
            msg.push_str(&format!(" {}", git_statuses[0]));
        } else if !git_statuses.is_empty() {
            // Mixed results — summarize
            let failures = git_statuses.iter().filter(|s| s.contains("failed")).count();
            if failures > 0 {
                msg.push_str(&format!(" Git staging: {failures} failed."));
            }
        }
        if long_entries > 0 {
            msg.push_str(&format!(
                "\n\nNote: {} entr{} exceeded 2000 chars. Long content embeds less precisely — \
                 consider splitting into smaller entries under a shared parent_id.",
                long_entries,
                if long_entries == 1 { "y" } else { "ies" }
            ));
        }
        Ok(serde_json::json!(msg))
    } else {
        let content_len = input.content.len();
        let (id, git_status) = store_single_entry(
            ctx,
            StoreSingleInput {
                content: input.content,
                parent_id: input.parent_id,
                source_file: input.source_file,
                heading: input.heading,
                visibility: input.visibility,
                perspectives: input.perspectives,
                entry_type: input.entry_type,
                relations: input.relations,
                impression_hint: input.impression_hint,
                impression_strength: input.impression_strength,
                scope: input.scope,
            },
            None, // deferred — background worker will embed
        )
        .await?;
        let mut msg = format!(
            "Stored. ID: {}. Embedding is being computed in the background \
             — entry becomes searchable as it completes.",
            crate::short_id(&id)
        );
        if let Some(status) = git_status {
            msg.push_str(&format!(" {status}"));
        }
        if content_len > 2000 {
            msg.push_str(
                "\n\nNote: Content exceeded 2000 chars. Long entries embed less precisely — \
                 consider splitting into smaller entries under a shared parent_id for better recall.",
            );
        }
        Ok(serde_json::json!(msg))
    }
}

pub async fn execute_think(
    ctx: &ToolContext,
    input: ThinkInput,
    push_mode: Option<crate::git::branch_config::PushMode>,
) -> Result<String> {
    let store = &ctx.store;
    let data_dir = ctx.data_dir.as_path();
    let blob_store = &ctx.blob_store;
    let project = ctx.project.as_deref();
    let branch = ctx.branch.as_deref();
    let git_store = ctx.git_store.as_deref();
    match input.action.as_deref() {
        None => think_reflect(store, data_dir, &input, project, branch).await,
        Some("promote") => think_promote(store, &input).await,
        Some("demote") => think_demote(store, &input).await,
        Some("relate") => think_relate(store, &input).await,
        Some("configure_aging") => think_configure_aging(data_dir, &input),
        Some("apply_aging") => think_apply_aging(store, data_dir).await,
        Some("consolidate") => think_consolidate(store, data_dir, blob_store, project).await,
        Some("prepare") => think_prepare(store, data_dir).await,
        Some("salience") => think_salience(store, &input).await,
        Some("perspectives") => think_perspectives(data_dir),
        Some("status") => think_status(store, data_dir, git_store, push_mode).await,
        Some("history") => think_history(store, &input).await,
        Some("discover") => think_discover(store, &input).await,
        Some("sync") => think_sync(git_store, &input).await,
        Some(unknown) => Err(crate::Error::config(format!(
            "Unknown think action: '{}'. Available: {}",
            unknown,
            THINK_ACTIONS.join(", ")
        ))),
    }
}

async fn think_reflect(
    store: &Arc<StoreBackend>,
    data_dir: &std::path::Path,
    input: &ThinkInput,
    project: Option<&str>,
    branch: Option<&str>,
) -> Result<String> {
    let hot_limit = clamp_result_limit(input.hot_limit.unwrap_or(10));
    let stale_limit = clamp_result_limit(input.stale_limit.unwrap_or(10));
    execute_reflect(store, data_dir, hot_limit, stale_limit, project, branch).await
}

async fn think_promote(store: &Arc<StoreBackend>, input: &ThinkInput) -> Result<String> {
    let raw_id = input
        .id
        .as_deref()
        .ok_or_else(|| crate::Error::config("think(promote) requires 'id'"))?;
    let chunk_id = crate::resolve::resolve_id(store, raw_id).await?;
    let vis = input.visibility.as_deref().unwrap_or("always");
    store.update_visibility(&chunk_id, vis).await?;
    Ok(format!("Promoted `{}` to visibility '{}'", chunk_id, vis))
}

async fn think_demote(store: &Arc<StoreBackend>, input: &ThinkInput) -> Result<String> {
    let raw_id = input
        .id
        .as_deref()
        .ok_or_else(|| crate::Error::config("think(demote) requires 'id'"))?;
    let chunk_id = crate::resolve::resolve_id(store, raw_id).await?;
    let vis = input.visibility.as_deref().unwrap_or("deep_only");
    store.update_visibility(&chunk_id, vis).await?;
    Ok(format!("Demoted `{}` to visibility '{}'", chunk_id, vis))
}

async fn think_relate(store: &Arc<StoreBackend>, input: &ThinkInput) -> Result<String> {
    let raw_source = input
        .source_id
        .as_deref()
        .ok_or_else(|| crate::Error::config("think(relate) requires 'source_id'"))?;
    let raw_target = input
        .target_id
        .as_deref()
        .ok_or_else(|| crate::Error::config("think(relate) requires 'target_id'"))?;
    let kind = input.kind.as_deref().unwrap_or("related_to");
    let source_id = crate::resolve::resolve_id(store, raw_source).await?;
    let target_id = crate::resolve::resolve_id(store, raw_target).await?;
    let relation = crate::ChunkRelation::new(kind, &target_id);
    store.add_relation(&source_id, relation).await?;
    // Bidirectional: related_to gets a backward link (mirrors CLI think_relate)
    if kind == "related_to" {
        let backward = crate::ChunkRelation::new("related_to", &source_id);
        store.add_relation(&target_id, backward).await?;
    }
    Ok(format!(
        "Added relation '{}' from `{}` to `{}`",
        kind, source_id, target_id
    ))
}

fn think_configure_aging(data_dir: &std::path::Path, input: &ThinkInput) -> Result<String> {
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

async fn think_apply_aging(
    store: &Arc<StoreBackend>,
    data_dir: &std::path::Path,
) -> Result<String> {
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

#[cfg(feature = "llm")]
async fn think_consolidate(
    store: &Arc<StoreBackend>,
    data_dir: &std::path::Path,
    blob_store: &Arc<crate::blob_store::BlobStore>,
    project: Option<&str>,
) -> Result<String> {
    let config = crate::Config::new().with_data_dir(data_dir);
    let llm = crate::llm::LlmBackend::from_config(&config.llm);
    let embedder = crate::embedder::from_config(&config.embedder)
        .map_err(|e| crate::Error::llm(format!("Failed to init embedder: {}", e)))?;

    let result = match crate::think::execute(
        store.as_ref(),
        embedder.as_ref(),
        &llm,
        data_dir,
        Some(blob_store.as_ref()),
        project,
    )
    .await
    {
        Ok(result) => result,
        Err(crate::Error::Llm(_)) => {
            // LLM unreachable — fall back to nudge.
            // No need to re-check emptiness: execute() returns Ok(empty) before
            // reaching the LLM call when the store is empty.
            return Ok(format_consolidate_nudge(
                &config.llm.provider,
                &config.llm.model,
            ));
        }
        Err(e) => return Err(e),
    };

    if result.entries_created.is_empty() {
        return Ok("Nothing to consolidate. Memory is well-organized.".to_string());
    }

    let mut report = format!(
        "## Think Cycle Complete\n\n- Narrative: {}\n- Consolidations: {}\n- Learnings: {}\n\n",
        if result.narrative_id.is_some() {
            "yes"
        } else {
            "no"
        },
        result.consolidations_added,
        result.learnings_added,
    );
    report.push_str("### Entries Created\n\n");
    for entry in &result.entries_created {
        report.push_str(&format!(
            "- **{}** ({}) `{}`\n",
            entry.content_preview, entry.entry_type, entry.id
        ));
    }
    Ok(report)
}

#[cfg(not(feature = "llm"))]
async fn think_consolidate(
    _store: &Arc<StoreBackend>,
    _data_dir: &std::path::Path,
    _blob_store: &Arc<crate::blob_store::BlobStore>,
    _project: Option<&str>,
) -> Result<String> {
    // Without LLM feature, nudge the caller to do it themselves
    Ok(format_consolidate_nudge("none", "none"))
}

/// Format a short nudge telling MCP callers how to self-consolidate.
fn format_consolidate_nudge(provider: &str, model: &str) -> String {
    format!(
        "Think: LLM unavailable ({}/{} not reachable).\n\n\
         You can consolidate memory yourself:\n\
         1. Call think(action=\"prepare\") to get the reflection data\n\
         2. Reason about consolidations and learnings\n\
         3. Call store() for each result with entry_type \"meta\" or \"summary\"\n\n\
         This keeps your context clean — only fetch the data when ready to act on it.",
        provider, model
    )
}

/// Gather reflection data for a caller to reason about consolidation.
async fn think_prepare(store: &Arc<StoreBackend>, data_dir: &std::path::Path) -> Result<String> {
    let prep = crate::think::prepare(store.as_ref(), data_dir).await?;

    let Some(prep) = prep else {
        return Ok("Nothing to prepare. Memory is empty.".to_string());
    };

    let mut report = String::from("## Think Preparation\n\n");

    report.push_str("### Task\n\n");
    report.push_str(prep.system_prompt);
    report.push_str("\n\n");

    report.push_str("### Memory State\n\n");
    report.push_str(&prep.user_prompt);
    report.push_str("\n\n");

    report.push_str("### How to Apply\n\n");
    report.push_str("Reason about the memory state above, then call store() for each result:\n\n");
    report.push_str(
        "- **Narrative:** store(text=\"...\", heading=\"[think:narrative]\", entry_type=\"meta\")\n",
    );
    report.push_str(
        "- **Consolidation:** store(text=\"...\", heading=\"[think:consolidation]\", \
         entry_type=\"summary\", relations=[{kind: \"summarizes\", target_id: \"<id>\"}])\n",
    );
    report.push_str(
        "- **Learning:** store(text=\"...\", heading=\"[think:learning]\", \
         entry_type=\"meta\", perspectives=[\"learnings\"])\n",
    );

    Ok(report)
}

async fn think_salience(store: &Arc<StoreBackend>, input: &ThinkInput) -> Result<String> {
    let limit = clamp_result_limit(input.hot_limit.unwrap_or(10));
    let hot = store.get_hot_chunks(limit.saturating_mul(2)).await?;
    if hot.is_empty() {
        return Ok("No entries to analyze.".to_string());
    }
    let weights = crate::salience::SalienceWeights::default();
    let top = crate::salience::top_salient(&hot, &weights, limit);
    let mut report = String::from("## Salience Report\n\n");
    for (idx, score) in &top {
        let chunk = &hot[*idx];
        let heading = chunk.heading.as_deref().unwrap_or("(no heading)");
        report.push_str(&format!(
            "- **{}** [composite={:.3}, inter={:.2}, persp={:.2}, rev={:.2}] `{}`\n",
            heading,
            score.composite,
            score.interaction,
            score.perspective,
            score.revision,
            chunk.id
        ));
    }
    Ok(report)
}

fn think_perspectives(data_dir: &std::path::Path) -> Result<String> {
    let perspectives = crate::perspective::load(data_dir)?;
    if perspectives.is_empty() {
        return Ok("No perspectives defined.".to_string());
    }
    let mut report = String::from("## Perspectives\n\n");
    for p in &perspectives {
        let tag = if p.builtin { " [builtin]" } else { "" };
        report.push_str(&format!("- **{}** — {}{}\n", p.id, p.hint, tag));
    }
    report.push_str(&format!("\n{} perspective(s) total.", perspectives.len()));
    Ok(report)
}

async fn think_status(
    store: &Arc<StoreBackend>,
    data_dir: &std::path::Path,
    git_store: Option<&crate::git::memory_store::MemoryStore>,
    push_mode: Option<crate::git::branch_config::PushMode>,
) -> Result<String> {
    let stats = store.stats().await?;
    let aging_config = AgingConfig::load(data_dir);
    let mut status = super::format::format_store_status(&stats, &aging_config, None);

    if let Some(git) = git_store {
        status.push_str("\n### Git Memory\n\n");
        match git.unpushed_commit_count() {
            Ok(0) => status.push_str("- **Pending commits:** 0 (in sync with remote)\n"),
            Ok(n) => status.push_str(&format!("- **Pending commits:** {n} (not yet pushed)\n")),
            Err(_) => status.push_str("- **Pending commits:** unknown (no remote configured)\n"),
        }
        if let Some(pm) = push_mode {
            status.push_str(&format!("- **Push mode:** {pm}\n"));
        }
    }

    Ok(status)
}

async fn think_history(store: &Arc<StoreBackend>, input: &ThinkInput) -> Result<String> {
    let raw_id = input
        .id
        .as_deref()
        .ok_or_else(|| crate::Error::config("think(history) requires 'id'"))?;
    let chunk_id = crate::resolve::resolve_id(store, raw_id).await?;
    let chunk = store
        .get_by_id(&chunk_id)
        .await?
        .ok_or_else(|| crate::Error::not_found(format!("Entry '{}' not found", chunk_id)))?;

    let heading = chunk.heading.as_deref().unwrap_or("(no heading)");
    let mut report = format!("## Entry History: {} `{}`\n\n", heading, chunk.id);
    report.push_str(&format!("- **Type:** {}\n", chunk.entry_type));
    report.push_str(&format!("- **Visibility:** {}\n", chunk.visibility));
    report.push_str(&format!("- **Source:** {}\n", chunk.source_file));
    if !chunk.perspectives.is_empty() {
        report.push_str(&format!(
            "- **Perspectives:** {}\n",
            chunk.perspectives.join(", ")
        ));
    }

    if chunk.relations.is_empty() {
        report.push_str("\nNo relations.\n");
    } else {
        report.push_str(&format!("\n### Relations ({})\n\n", chunk.relations.len()));
        for rel in &chunk.relations {
            report.push_str(&format!(
                "- {} → `{}`\n",
                rel.kind,
                crate::chunk::short_id(&rel.target_id)
            ));
        }
    }

    report.push_str(&format!(
        "\n### Content\n\n{}\n",
        if chunk.content.len() > 500 {
            let end = chunk.content.floor_char_boundary(500);
            format!("{}...", &chunk.content[..end])
        } else {
            chunk.content.clone()
        }
    ));

    Ok(report)
}

#[cfg(feature = "llm")]
async fn think_discover(store: &Arc<StoreBackend>, input: &ThinkInput) -> Result<String> {
    let limit = clamp_result_limit(input.hot_limit.unwrap_or(10));
    crate::think::discover_unlinked_pairs(store, limit).await
}

#[cfg(not(feature = "llm"))]
async fn think_discover(_store: &Arc<StoreBackend>, _input: &ThinkInput) -> Result<String> {
    Err(crate::Error::config(
        "think(discover) requires the 'llm' feature",
    ))
}

async fn think_sync(
    git_store: Option<&crate::git::memory_store::MemoryStore>,
    input: &ThinkInput,
) -> Result<String> {
    use crate::git::{PushResult, SyncResult};

    let git = git_store.ok_or_else(|| {
        crate::Error::config(
            "Git memory storage is not configured. Run `veclayer init --share` first.",
        )
    })?;

    let direction = input.direction.as_deref().unwrap_or("both");
    let mut report = String::from("## Sync Report\n\n");

    match direction {
        "push" => match git.push() {
            Ok(PushResult::Success) => report.push_str("**Pushed** to remote successfully.\n"),
            Ok(PushResult::NothingToPush) => {
                report.push_str("Nothing to push — already up to date.\n")
            }
            Ok(PushResult::Rejected) => {
                report.push_str("Push **rejected** — remote has diverged.\n");
                report.push_str("Try `think(action='sync')` to pull first, then push.\n");
            }
            Err(e) => report.push_str(&format!("Push failed: {e}\n")),
        },
        "pull" => match git.pull() {
            Ok(SyncResult::Success) => report.push_str("**Pulled** new entries from remote.\n"),
            Ok(SyncResult::NothingToSync) => report.push_str("Already up to date with remote.\n"),
            Ok(SyncResult::Conflicts(files)) => {
                report.push_str("**Conflict detected** during pull. Rebase aborted.\n\n");
                report.push_str("Conflicting files:\n");
                for f in &files {
                    report.push_str(&format!("- `{f}`\n"));
                }
                report.push_str("\nResolve manually or use `think(action='sync')` after fixing.\n");
            }
            Err(e) => report.push_str(&format!("Pull failed: {e}\n")),
        },
        _ => match git.sync() {
            Ok(SyncResult::Success) => {
                report.push_str("**Synced** — pulled and pushed successfully.\n")
            }
            Ok(SyncResult::NothingToSync) => report.push_str("Already in sync with remote.\n"),
            Ok(SyncResult::Conflicts(files)) => {
                report.push_str("**Conflict detected** during sync.\n\n");
                report.push_str("Conflicting files:\n");
                for f in &files {
                    report.push_str(&format!("- `{f}`\n"));
                }
            }
            Err(e) => report.push_str(&format!("Sync failed: {e}\n")),
        },
    }

    match git.unpushed_commit_count() {
        Ok(0) => {}
        Ok(n) => report.push_str(&format!("\n{n} commit(s) still pending push.\n")),
        Err(_) => {}
    }

    Ok(report)
}

async fn execute_reflect(
    store: &Arc<StoreBackend>,
    data_dir: &std::path::Path,
    hot_limit: usize,
    stale_limit: usize,
    project: Option<&str>,
    branch: Option<&str>,
) -> Result<String> {
    let aging_config = AgingConfig::load(data_dir);

    // Over-fetch when project filter is active, then filter client-side
    let fetch_limit = if project.is_some() { 10_000 } else { hot_limit };
    let hot: Vec<_> = store
        .get_hot_chunks(fetch_limit)
        .await?
        .into_iter()
        .filter(|c| passes_scope_filter(c, project, branch))
        .take(hot_limit)
        .collect();
    let stale_fetch = if project.is_some() {
        10_000
    } else {
        stale_limit
    };
    let stale: Vec<_> = store
        .get_stale_chunks(aging_config.stale_seconds(), stale_fetch)
        .await?
        .into_iter()
        .filter(|c| passes_scope_filter(c, project, branch))
        .take(stale_limit)
        .collect();

    let mut report = String::new();

    let weights = crate::salience::SalienceWeights::default();

    report.push_str("## Hot Chunks (most accessed)\n\n");
    if hot.is_empty() {
        report.push_str("No chunks have been accessed yet.\n\n");
    } else {
        for chunk in &hot {
            let heading = chunk.heading.as_deref().unwrap_or("(no heading)");
            let salience = crate::salience::compute(chunk, &weights);
            report.push_str(&format!(
                "- **{}** (total: {}, salience: {:.2}) [{}] `{}`\n",
                heading, chunk.access_profile.total, salience.composite, chunk.visibility, chunk.id
            ));
        }
        report.push('\n');
    }

    report.push_str(&format!(
        "## Stale Chunks (no access in {} days)\n\n",
        aging_config.degrade_after_days
    ));
    if stale.is_empty() {
        report.push_str("No stale chunks found. Memory is well-maintained.\n\n");
    } else {
        for chunk in &stale {
            let heading = chunk.heading.as_deref().unwrap_or("(no heading)");
            let salience = crate::salience::compute(chunk, &weights);
            report.push_str(&format!(
                "- **{}** [vis={}, salience={:.2}] `{}`\n",
                heading, chunk.visibility, salience.composite, chunk.id
            ));
        }
        report.push('\n');
    }

    let stats = store.stats().await?;
    report.push_str(&format!(
        "## Summary\n\n- Total chunks: {}\n- Source files: {}\n- Aging policy: degrade {} → '{}' after {} days\n",
        stats.total_chunks,
        stats.source_files.len(),
        aging_config.degrade_from.join("/"),
        aging_config.degrade_to,
        aging_config.degrade_after_days,
    ));

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

pub fn build_share_token(input: ShareInput) -> serde_json::Value {
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
        "nonce": crate::chunk::content_hash(&format!("nonce-{}", crate::chunk::now_epoch_secs())),
        "_note": "Preview token. UCAN signing not yet implemented."
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // resolve_id and parse_temporal tests are in resolve::tests.
    // These tests cover tool-specific logic that remains in this module.

    #[cfg(feature = "embedding-local")]
    use crate::embedder::FastEmbedder;
    use crate::test_helpers::make_test_chunk;

    async fn make_test_store_with_dir() -> (
        Arc<StoreBackend>,
        Arc<crate::blob_store::BlobStore>,
        tempfile::TempDir,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let store = StoreBackend::open(dir.path(), 384, false).await.unwrap();
        let blob_store = crate::blob_store::BlobStore::open(dir.path()).unwrap();
        (Arc::new(store), Arc::new(blob_store), dir)
    }

    /// Build a minimal `ToolContext` for tests.
    ///
    /// All optional fields default to `None`; override them in the returned struct if needed.
    fn test_ctx(
        store: &Arc<StoreBackend>,
        embedder: &Arc<dyn crate::Embedder + Send + Sync>,
        blob_store: &Arc<crate::blob_store::BlobStore>,
        data_dir: &std::path::Path,
    ) -> ToolContext {
        ToolContext {
            store: Arc::clone(store),
            embedder: Arc::clone(embedder),
            blob_store: Arc::clone(blob_store),
            data_dir: data_dir.to_path_buf(),
            project: None,
            branch: None,
            git_store: None,
            push_mode: crate::git::branch_config::PushMode::Off,
        }
    }

    /// Build a `ToolContext` for recall/focus tests (blob_store and data_dir unused, so
    /// a temporary directory is created and returned as a guard alongside the context).
    ///
    /// The caller must hold the returned `TempDir` for the duration of the test; dropping
    /// it early will delete the directory while the context still references it.
    fn test_ctx_recall(
        store: &Arc<StoreBackend>,
        embedder: &Arc<dyn crate::Embedder + Send + Sync>,
    ) -> (ToolContext, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let blob_store = Arc::new(crate::blob_store::BlobStore::open(tmp.path()).unwrap());
        let ctx = ToolContext {
            store: Arc::clone(store),
            embedder: Arc::clone(embedder),
            blob_store,
            data_dir: tmp.path().to_path_buf(),
            project: None,
            branch: None,
            git_store: None,
            push_mode: crate::git::branch_config::PushMode::Off,
        };
        (ctx, tmp)
    }

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

    #[test]
    fn test_resolve_scope_perspectives_appends_facets() {
        let base = vec!["decisions".to_string()];
        let project = resolve_scope_perspectives("project", &base, Some("veclayer"), None).unwrap();
        assert_eq!(project, vec!["decisions", "project:veclayer"]);

        let branch =
            resolve_scope_perspectives("branch", &base, Some("veclayer"), Some("main")).unwrap();
        assert_eq!(
            branch,
            vec!["decisions", "project:veclayer", "branch:veclayer@main"]
        );

        let personal = resolve_scope_perspectives("personal", &base, Some("veclayer"), None);
        assert_eq!(personal.unwrap(), base);
    }

    #[test]
    fn test_resolve_scope_perspectives_rejects_unknown_scope() {
        // A typo must not silently fall through to the widest (personal) scope.
        let result = resolve_scope_perspectives("prject", &[], Some("veclayer"), None);
        assert!(
            result.is_err(),
            "an unknown scope must be rejected, not treated as personal"
        );
    }

    #[test]
    fn test_clamp_result_limit_bounds_huge_values() {
        assert_eq!(clamp_result_limit(usize::MAX), MAX_RESULT_LIMIT);
        assert_eq!(clamp_result_limit(MAX_RESULT_LIMIT + 1), MAX_RESULT_LIMIT);
        assert_eq!(clamp_result_limit(10), 10);
        assert_eq!(clamp_result_limit(0), 0);
    }

    #[test]
    fn test_temporal_fetch_limit_saturates_and_passes_through() {
        // No time filter: the limit is used as-is.
        assert_eq!(temporal_fetch_limit(50, None, None), 50);
        // Time filter active: over-fetched by the prefetch factor.
        assert_eq!(
            temporal_fetch_limit(50, Some(0), None),
            50 * TEMPORAL_PREFETCH_FACTOR
        );
        // A huge limit saturates instead of overflowing.
        assert_eq!(temporal_fetch_limit(usize::MAX, Some(0), None), usize::MAX);
        // A clamped limit can never overflow the downstream prefetch multiply.
        assert_eq!(
            clamp_result_limit(usize::MAX).checked_mul(TEMPORAL_PREFETCH_FACTOR),
            Some(MAX_RESULT_LIMIT * TEMPORAL_PREFETCH_FACTOR),
            "clamped limit must not overflow downstream arithmetic"
        );
    }

    #[test]
    fn test_relevance_tier() {
        assert_eq!(relevance_tier(0.5), "strong");
        assert_eq!(relevance_tier(0.46), "strong");
        assert_eq!(relevance_tier(0.45), "moderate");
        assert_eq!(relevance_tier(0.35), "moderate");
        assert_eq!(relevance_tier(0.30), "weak");
        assert_eq!(relevance_tier(0.20), "weak");
        assert_eq!(relevance_tier(0.15), "tangential");
        assert_eq!(relevance_tier(0.0), "tangential");
    }

    #[tokio::test]
    async fn test_think_perspectives_action() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;
        // Initialize perspectives so there's something to list
        crate::perspective::init(dir.path()).unwrap();

        let input = ThinkInput {
            action: Some("perspectives".to_string()),
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
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await
        .unwrap();
        assert!(result.contains("Perspectives"));
        // Should contain the built-in perspectives
        assert!(result.contains("decisions"));
        assert!(result.contains("knowledge"));
    }

    #[tokio::test]
    async fn test_think_status_action() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;
        // Insert a test chunk so stats are non-zero
        store
            .insert_chunks(vec![make_test_chunk("abc123", "test content")])
            .await
            .unwrap();

        let input = ThinkInput {
            action: Some("status".to_string()),
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
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await
        .unwrap();
        assert!(result.contains("Store Status"));
        assert!(result.contains("Total entries"));
        assert!(result.contains("1")); // 1 entry
    }

    #[tokio::test]
    async fn test_think_history_action() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;
        let mut chunk = make_test_chunk("abcdef1234567890", "historical content");
        chunk
            .relations
            .push(crate::ChunkRelation::new("supersedes", "older_entry"));
        store.insert_chunks(vec![chunk]).await.unwrap();

        let input = ThinkInput {
            action: Some("history".to_string()),
            hot_limit: None,
            stale_limit: None,
            id: Some("abcdef1".to_string()), // short ID
            visibility: None,
            source_id: None,
            target_id: None,
            kind: None,
            degrade_after_days: None,
            degrade_to: None,
            degrade_from: None,
            direction: None,
        };
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await
        .unwrap();
        assert!(result.contains("Entry History"));
        assert!(result.contains("Relations"));
        assert!(result.contains("supersedes"));
        assert!(result.contains("historical content"));
    }

    #[tokio::test]
    async fn test_think_history_requires_id() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;

        let input = ThinkInput {
            action: Some("history".to_string()),
            hot_limit: None,
            stale_limit: None,
            id: None, // Missing required ID
            visibility: None,
            source_id: None,
            target_id: None,
            kind: None,
            degrade_after_days: None,
            degrade_to: None,
            degrade_from: None,
            direction: None,
        };
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("requires 'id'"));
    }

    #[tokio::test]
    async fn test_think_status_empty_store() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;

        let input = ThinkInput {
            action: Some("status".to_string()),
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
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await
        .unwrap();
        assert!(result.contains("Total entries"));
        assert!(result.contains("0"));
    }

    #[tokio::test]
    async fn test_think_unknown_action() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;

        let input = ThinkInput {
            action: Some("nonexistent".to_string()),
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
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unknown think action"));
        assert!(err.contains("perspectives"));
        assert!(err.contains("status"));
        assert!(err.contains("history"));
        assert!(err.contains("sync"));
    }

    #[tokio::test]
    async fn test_think_sync_without_git_store_returns_error() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;

        let input = ThinkInput {
            action: Some("sync".to_string()),
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
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Git memory storage is not configured"));
    }

    #[tokio::test]
    async fn test_think_status_without_git_includes_store_status() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;
        store
            .insert_chunks(vec![make_test_chunk("abc123", "test content")])
            .await
            .unwrap();

        let input = ThinkInput {
            action: Some("status".to_string()),
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
        // With no git_store, status should still return store info without git section
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await
        .unwrap();
        assert!(result.contains("Store Status"));
        assert!(result.contains("Total entries"));
        // Git section should NOT appear when git_store is None
        assert!(!result.contains("Git Memory"));
    }

    #[test]
    fn test_store_single_input_structure() {
        let input = StoreSingleInput {
            content: "test content".to_string(),
            parent_id: None,
            source_file: "[test]".to_string(),
            heading: Some("Test Heading".to_string()),
            visibility: "normal".to_string(),
            perspectives: vec!["decisions".to_string(), "learnings".to_string()],
            entry_type: Some("raw".to_string()),
            relations: vec![
                StoreRelation {
                    kind: "supersedes".to_string(),
                    target_id: "old-id".to_string(),
                },
                StoreRelation {
                    kind: "related_to".to_string(),
                    target_id: "related-id".to_string(),
                },
            ],
            impression_hint: None,
            impression_strength: None,
            scope: "project".to_string(),
        };

        assert_eq!(input.content, "test content");
        assert_eq!(input.heading, Some("Test Heading".to_string()));
        assert_eq!(input.perspectives.len(), 2);
        assert_eq!(input.perspectives[0], "decisions");
        assert_eq!(input.relations.len(), 2);
        assert_eq!(input.relations[0].kind, "supersedes");
        assert_eq!(input.relations[0].target_id, "old-id");
    }

    #[test]
    fn test_store_item_structure() {
        let item = StoreItem {
            content: "item content".to_string(),
            parent_id: Some("parent-id".to_string()),
            heading: None,
            visibility: "deep_only".to_string(),
            perspectives: vec!["intentions".to_string()],
            source_file: Some("[file]".to_string()),
            entry_type: Some("meta".to_string()),
            relations: vec![StoreRelation {
                kind: "summarizes".to_string(),
                target_id: "target-id".to_string(),
            }],
            impression_hint: None,
            impression_strength: None,
            scope: "project".to_string(),
        };

        assert_eq!(item.content, "item content");
        assert_eq!(item.parent_id, Some("parent-id".to_string()));
        assert_eq!(item.visibility, "deep_only");
        assert_eq!(item.relations.len(), 1);
        assert_eq!(item.relations[0].kind, "summarizes");
    }

    #[test]
    fn test_store_input_structure() {
        let input = StoreInput {
            content: "main content".to_string(),
            parent_id: None,
            source_file: "[agent]".to_string(),
            heading: Some("Main Heading".to_string()),
            visibility: "normal".to_string(),
            perspectives: vec!["knowledge".to_string()],
            relations: vec![StoreRelation {
                kind: "derived_from".to_string(),
                target_id: "source-id".to_string(),
            }],
            entry_type: None,
            items: vec![],
            impression_hint: None,
            impression_strength: None,
            scope: "project".to_string(),
        };

        assert_eq!(input.content, "main content");
        assert_eq!(input.relations.len(), 1);
        assert_eq!(input.relations[0].kind, "derived_from");
        assert!(input.items.is_empty());
    }

    #[test]
    fn test_store_input_batch_mode() {
        let input = StoreInput {
            content: String::new(),
            heading: None,
            parent_id: None,
            source_file: "[agent]".to_string(),
            visibility: "normal".to_string(),
            perspectives: vec![],
            relations: vec![],
            entry_type: None,
            items: vec![
                StoreItem {
                    content: "item 1".to_string(),
                    parent_id: None,
                    heading: None,
                    visibility: "normal".to_string(),
                    perspectives: vec![],
                    source_file: None,
                    entry_type: None,
                    relations: vec![],
                    impression_hint: None,
                    impression_strength: None,
                    scope: "project".to_string(),
                },
                StoreItem {
                    content: "item 2".to_string(),
                    parent_id: Some("parent".to_string()),
                    heading: Some("Item 2".to_string()),
                    visibility: "deep_only".to_string(),
                    perspectives: vec!["decisions".to_string()],
                    source_file: Some("[file]".to_string()),
                    entry_type: Some("impression".to_string()),
                    relations: vec![StoreRelation {
                        kind: "related_to".to_string(),
                        target_id: "other".to_string(),
                    }],
                    impression_hint: None,
                    impression_strength: None,
                    scope: "project".to_string(),
                },
            ],
            impression_hint: None,
            impression_strength: None,
            scope: "project".to_string(),
        };

        assert!(input.content.is_empty());
        assert_eq!(input.items.len(), 2);
        assert_eq!(input.items[0].content, "item 1");
        assert_eq!(input.items[1].heading, Some("Item 2".to_string()));
        assert_eq!(input.items[1].relations.len(), 1);
    }

    #[test]
    fn test_store_relation_kinds() {
        let relations = [
            StoreRelation {
                kind: "supersedes".to_string(),
                target_id: "id1".to_string(),
            },
            StoreRelation {
                kind: "summarizes".to_string(),
                target_id: "id2".to_string(),
            },
            StoreRelation {
                kind: "related_to".to_string(),
                target_id: "id3".to_string(),
            },
            StoreRelation {
                kind: "derived_from".to_string(),
                target_id: "id4".to_string(),
            },
            StoreRelation {
                kind: "version_of".to_string(),
                target_id: "id5".to_string(),
            },
        ];

        assert_eq!(relations.len(), 5);
        assert_eq!(relations[0].kind, "supersedes");
        assert_eq!(relations[1].kind, "summarizes");
        assert_eq!(relations[2].kind, "related_to");
        assert_eq!(relations[3].kind, "derived_from");
        assert_eq!(relations[4].kind, "version_of");
    }

    #[cfg(feature = "embedding-local")]
    #[tokio::test]
    async fn test_recall_ongoing_filter_with_query() {
        let (store, _blob_store, _dir) = make_test_store_with_dir().await;
        let embedder: Arc<dyn Embedder + Send + Sync> = Arc::new(FastEmbedder::new().unwrap());

        // Insert a plain chunk with a real embedding so semantic search can find it
        let plain_content = "plain entry about architecture decisions";
        let plain_embeddings = embedder.embed(&[plain_content]).unwrap();
        let mut plain = make_test_chunk(
            "ccccccc3333333333333333333333333333333333333333333333333333333333",
            plain_content,
        );
        plain.embedding = Some(plain_embeddings.into_iter().next().unwrap());
        store.insert_chunks(vec![plain]).await.unwrap();

        // Insert a chunk that qualifies as an open thread (superseded, still "normal" visibility)
        let open_content = "unresolved entry about design decisions";
        let open_embeddings = embedder.embed(&[open_content]).unwrap();
        let mut open = make_test_chunk(
            "ddddddd4444444444444444444444444444444444444444444444444444444444",
            open_content,
        );
        open.embedding = Some(open_embeddings.into_iter().next().unwrap());
        open.relations
            .push(crate::ChunkRelation::superseded_by("newer-id"));
        store.insert_chunks(vec![open]).await.unwrap();

        // With ongoing: true and a query — only the open-thread entry should be returned
        let input_ongoing = RecallInput {
            query: Some("entry".to_string()),
            limit: 10,
            deep: false,
            recency: None,
            perspectives: None,
            similar_to: None,
            min_salience: None,
            min_score: None,
            since: None,
            until: None,
            ongoing: Some(true),
        };
        let (ctx_ongoing, _tmp_ongoing) = test_ctx_recall(&store, &embedder);
        let ongoing_results = execute_recall(&ctx_ongoing, input_ongoing, None)
            .await
            .unwrap();
        assert_eq!(
            ongoing_results.len(),
            1,
            "ongoing filter with query should return only open-thread entry, got: {:?}",
            ongoing_results
                .iter()
                .map(|r| &r.chunk.id)
                .collect::<Vec<_>>()
        );
        assert!(
            ongoing_results[0].chunk.id.starts_with("ddddddd"),
            "the open-thread entry should be the one returned"
        );

        // With ongoing: None and a query — both entries should be returned
        let input_all = RecallInput {
            query: Some("entry".to_string()),
            limit: 10,
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
        let (ctx_all, _tmp_all) = test_ctx_recall(&store, &embedder);
        let all_results = execute_recall(&ctx_all, input_all, None).await.unwrap();
        assert_eq!(
            all_results.len(),
            2,
            "no ongoing filter with query should return both entries"
        );
    }

    #[cfg(feature = "embedding-local")]
    #[tokio::test]
    async fn test_recall_ongoing_filter_browse_mode() {
        let (store, _blob_store, _dir) = make_test_store_with_dir().await;
        let embedder: Arc<dyn Embedder + Send + Sync> = Arc::new(FastEmbedder::new().unwrap());

        // Insert a plain chunk (no open thread criteria)
        let plain = make_test_chunk(
            "aaaaaaa1111111111111111111111111111111111111111111111111111111111",
            "plain entry",
        );
        store.insert_chunks(vec![plain]).await.unwrap();

        // Insert a chunk that qualifies as an open thread (superseded but still "normal")
        let mut open = make_test_chunk(
            "bbbbbbb2222222222222222222222222222222222222222222222222222222222",
            "unresolved entry",
        );
        open.relations
            .push(crate::ChunkRelation::superseded_by("newer-id"));
        store.insert_chunks(vec![open]).await.unwrap();

        // Without ongoing filter: both entries returned
        let input_all = RecallInput {
            query: None,
            limit: 10,
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
        let (ctx_all, _tmp_all) = test_ctx_recall(&store, &embedder);
        let all_results = execute_recall(&ctx_all, input_all, None).await.unwrap();
        assert_eq!(
            all_results.len(),
            2,
            "should return both entries without ongoing filter"
        );

        // With ongoing: true — only the open thread entry
        let input_ongoing = RecallInput {
            query: None,
            limit: 10,
            deep: false,
            recency: None,
            perspectives: None,
            similar_to: None,
            min_salience: None,
            min_score: None,
            since: None,
            until: None,
            ongoing: Some(true),
        };
        let (ctx_ongoing, _tmp_ongoing) = test_ctx_recall(&store, &embedder);
        let ongoing_results = execute_recall(&ctx_ongoing, input_ongoing, None)
            .await
            .unwrap();
        assert_eq!(
            ongoing_results.len(),
            1,
            "should return only open thread entries"
        );
        assert!(
            ongoing_results[0].chunk.id.starts_with("bbbbbbb"),
            "the open thread entry should be returned"
        );

        // With ongoing: false — behaves the same as no filter
        let input_not_ongoing = RecallInput {
            query: None,
            limit: 10,
            deep: false,
            recency: None,
            perspectives: None,
            similar_to: None,
            min_salience: None,
            min_score: None,
            since: None,
            until: None,
            ongoing: Some(false),
        };
        let (ctx_not_ongoing, _tmp_not_ongoing) = test_ctx_recall(&store, &embedder);
        let not_ongoing_results = execute_recall(&ctx_not_ongoing, input_not_ongoing, None)
            .await
            .unwrap();
        assert_eq!(
            not_ongoing_results.len(),
            2,
            "ongoing: false should not filter"
        );
    }

    // ── discover tests ──────────────────────────────────────────────────

    /// Build a chunk with a real embedding using the FastEmbedder.
    #[cfg(feature = "embedding-local")]
    async fn make_embedded_chunk(
        embedder: &Arc<dyn Embedder + Send + Sync>,
        id: &str,
        content: &str,
    ) -> crate::HierarchicalChunk {
        let embeddings = embedder.embed(&[content]).unwrap();
        let mut chunk = make_test_chunk(id, content);
        chunk.embedding = Some(embeddings.into_iter().next().unwrap());
        chunk
    }

    #[cfg(feature = "llm")]
    #[tokio::test]
    async fn test_discover_empty_store() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;

        let input = ThinkInput {
            action: Some("discover".to_string()),
            hot_limit: Some(10),
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
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await
        .unwrap();
        assert!(result.contains("Nothing to discover") || result.contains("No entries"));
    }

    #[cfg(all(feature = "embedding-local", feature = "llm"))]
    #[tokio::test]
    async fn test_discover_finds_unlinked_similar_pair() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;
        let embedder: Arc<dyn Embedder + Send + Sync> = Arc::new(FastEmbedder::new().unwrap());

        // Two semantically similar entries with no relation
        let chunk_a = make_embedded_chunk(
            &embedder,
            "aaa111aaa111aaa111aaa111aaa111aaa111aaa111aaa111aaa111aaa111aaaa",
            "Rust memory safety: ownership and borrowing prevent data races",
        )
        .await;
        let chunk_b = make_embedded_chunk(
            &embedder,
            "bbb222bbb222bbb222bbb222bbb222bbb222bbb222bbb222bbb222bbb222bbbb",
            "Rust ownership system eliminates memory bugs at compile time",
        )
        .await;

        store.insert_chunks(vec![chunk_a, chunk_b]).await.unwrap();

        let input = ThinkInput {
            action: Some("discover".to_string()),
            hot_limit: Some(10),
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
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await
        .unwrap();

        // Should produce a discover report with at least one pair
        assert!(
            result.contains("Discover") || result.contains("No unlinked"),
            "expected discover report, got: {}",
            &result[..result.len().min(200)]
        );
    }

    #[cfg(all(feature = "embedding-local", feature = "llm"))]
    #[tokio::test]
    async fn test_discover_skips_already_linked_pair() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;
        let embedder: Arc<dyn Embedder + Send + Sync> = Arc::new(FastEmbedder::new().unwrap());

        let mut chunk_a = make_embedded_chunk(
            &embedder,
            "ccc333ccc333ccc333ccc333ccc333ccc333ccc333ccc333ccc333ccc333cccc",
            "Database indexing speeds up query performance significantly",
        )
        .await;
        let chunk_b = make_embedded_chunk(
            &embedder,
            "ddd444ddd444ddd444ddd444ddd444ddd444ddd444ddd444ddd444ddd444dddd",
            "Adding an index to the database table improves query speed",
        )
        .await;

        // Explicitly link the pair before inserting
        chunk_a
            .relations
            .push(crate::ChunkRelation::new("related_to", &chunk_b.id));

        store.insert_chunks(vec![chunk_a, chunk_b]).await.unwrap();

        let input = ThinkInput {
            action: Some("discover".to_string()),
            hot_limit: Some(10),
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
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await
        .unwrap();

        // The linked pair should NOT appear (either no pairs found or different pairs only)
        // We verify the IDs of the linked pair do not appear together as a discovered pair
        let short_a = crate::chunk::short_id(
            "ccc333ccc333ccc333ccc333ccc333ccc333ccc333ccc333ccc333ccc333cccc",
        );
        let short_b = crate::chunk::short_id(
            "ddd444ddd444ddd444ddd444ddd444ddd444ddd444ddd444ddd444ddd444dddd",
        );

        // If both IDs appear in the same "Discovery N" block, the filter failed.
        // We check by finding the discovery sections and verifying no single section
        // contains both IDs.
        for section in result.split("### Discovery") {
            let has_a = section.contains(short_a);
            let has_b = section.contains(short_b);
            assert!(
                !(has_a && has_b),
                "linked pair should not appear as a discovery: section = {}",
                &section[..section.len().min(300)]
            );
        }
    }

    // ── Mock embedder (no external feature deps) ─────────────────────────

    struct MockEmbedder {
        dim: usize,
    }

    impl MockEmbedder {
        fn new() -> Self {
            Self { dim: 384 }
        }
    }

    impl crate::Embedder for MockEmbedder {
        fn embed(&self, texts: &[&str]) -> crate::Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let mut v = vec![0.1f32; self.dim];
                    v[0] = (i + 1) as f32 / 100.0;
                    v
                })
                .collect())
        }

        fn dimension(&self) -> usize {
            self.dim
        }

        fn name(&self) -> &str {
            "mock-embedder"
        }
    }

    // ── passes_scope_filter ──────────────────────────────────────────────

    #[test]
    fn scope_filter_no_project_passes_everything() {
        let chunk = make_test_chunk("id1", "content");
        assert!(passes_scope_filter(&chunk, None, None));
    }

    #[test]
    fn scope_filter_unscoped_chunk_passes_any_project() {
        let chunk = make_test_chunk("id1", "content");
        assert!(passes_scope_filter(&chunk, Some("myproject"), None));
    }

    #[test]
    fn scope_filter_matching_project_passes() {
        let mut chunk = make_test_chunk("id1", "content");
        chunk.perspectives = vec!["project:myproject".to_string()];
        assert!(passes_scope_filter(&chunk, Some("myproject"), None));
    }

    #[test]
    fn scope_filter_wrong_project_fails() {
        let mut chunk = make_test_chunk("id1", "content");
        chunk.perspectives = vec!["project:other".to_string()];
        assert!(!passes_scope_filter(&chunk, Some("myproject"), None));
    }

    #[test]
    fn scope_filter_branch_requires_exact_branch_match() {
        let mut chunk = make_test_chunk("id1", "content");
        chunk.perspectives = vec![
            "project:myproject".to_string(),
            "branch:myproject@main".to_string(),
        ];
        assert!(passes_scope_filter(&chunk, Some("myproject"), Some("main")));
        assert!(!passes_scope_filter(
            &chunk,
            Some("myproject"),
            Some("feature")
        ));
    }

    #[test]
    fn scope_filter_branch_chunk_without_branch_arg_fails() {
        let mut chunk = make_test_chunk("id1", "content");
        chunk.perspectives = vec![
            "project:myproject".to_string(),
            "branch:myproject@main".to_string(),
        ];
        assert!(!passes_scope_filter(&chunk, Some("myproject"), None));
    }

    #[test]
    fn scope_filter_non_project_perspectives_pass_project_filter() {
        let mut chunk = make_test_chunk("id1", "content");
        chunk.perspectives = vec!["decisions".to_string()];
        assert!(passes_scope_filter(&chunk, Some("myproject"), None));
    }

    // ── execute_store ────────────────────────────────────────────────────

    #[tokio::test]
    async fn execute_store_single_entry_returns_id() {
        let (store, blob_store, _dir) = make_test_store_with_dir().await;
        let embedder: Arc<dyn crate::Embedder + Send + Sync> = Arc::new(MockEmbedder::new());

        let input = StoreInput {
            content: "Test entry content".to_string(),
            parent_id: None,
            source_file: "[agent]".to_string(),
            heading: Some("Test".to_string()),
            visibility: "normal".to_string(),
            perspectives: vec![],
            relations: vec![],
            entry_type: None,
            items: vec![],
            impression_hint: None,
            impression_strength: None,
            scope: "project".to_string(),
        };
        let result = execute_store(
            &test_ctx(&store, &embedder, &blob_store, _dir.path()),
            input,
        )
        .await
        .unwrap();
        let msg = result.as_str().unwrap();
        assert!(msg.contains("Stored."), "got: {msg}");
        assert!(msg.contains("ID:"), "got: {msg}");
    }

    #[tokio::test]
    async fn execute_store_project_scope_adds_project_perspective() {
        let (store, blob_store, _dir) = make_test_store_with_dir().await;
        let embedder: Arc<dyn crate::Embedder + Send + Sync> = Arc::new(MockEmbedder::new());

        let input = StoreInput {
            content: "Project-scoped entry".to_string(),
            parent_id: None,
            source_file: "[agent]".to_string(),
            heading: None,
            visibility: "normal".to_string(),
            perspectives: vec![],
            relations: vec![],
            entry_type: None,
            items: vec![],
            impression_hint: None,
            impression_strength: None,
            scope: "project".to_string(),
        };
        execute_store(
            &ToolContext {
                project: Some("myproj".to_string()),
                ..test_ctx(&store, &embedder, &blob_store, _dir.path())
            },
            input,
        )
        .await
        .unwrap();

        let entries = store.list_entries(&[], None, None, 10).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert!(
            entries[0]
                .perspectives
                .contains(&"project:myproj".to_string()),
            "perspectives: {:?}",
            entries[0].perspectives
        );
    }

    #[tokio::test]
    async fn execute_store_personal_scope_no_project_perspective() {
        let (store, blob_store, _dir) = make_test_store_with_dir().await;
        let embedder: Arc<dyn crate::Embedder + Send + Sync> = Arc::new(MockEmbedder::new());

        let input = StoreInput {
            content: "Personal entry".to_string(),
            parent_id: None,
            source_file: "[agent]".to_string(),
            heading: None,
            visibility: "normal".to_string(),
            perspectives: vec![],
            relations: vec![],
            entry_type: None,
            items: vec![],
            impression_hint: None,
            impression_strength: None,
            scope: "personal".to_string(),
        };
        execute_store(
            &ToolContext {
                project: Some("myproj".to_string()),
                ..test_ctx(&store, &embedder, &blob_store, _dir.path())
            },
            input,
        )
        .await
        .unwrap();

        let entries = store.list_entries(&[], None, None, 10).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert!(
            !entries[0]
                .perspectives
                .iter()
                .any(|p| p.starts_with("project:")),
            "personal scope should not add project perspective: {:?}",
            entries[0].perspectives
        );
    }

    #[tokio::test]
    async fn execute_store_batch_mode_stores_multiple_entries() {
        let (store, blob_store, _dir) = make_test_store_with_dir().await;
        let embedder: Arc<dyn crate::Embedder + Send + Sync> = Arc::new(MockEmbedder::new());

        let items = vec![
            StoreItem {
                content: "Batch item one".to_string(),
                parent_id: None,
                heading: None,
                visibility: "normal".to_string(),
                perspectives: vec![],
                source_file: None,
                entry_type: None,
                relations: vec![],
                impression_hint: None,
                impression_strength: None,
                scope: "project".to_string(),
            },
            StoreItem {
                content: "Batch item two".to_string(),
                parent_id: None,
                heading: Some("Item Two".to_string()),
                visibility: "normal".to_string(),
                perspectives: vec!["decisions".to_string()],
                source_file: Some("[file]".to_string()),
                entry_type: Some("meta".to_string()),
                relations: vec![],
                impression_hint: None,
                impression_strength: None,
                scope: "project".to_string(),
            },
        ];
        let input = StoreInput {
            content: String::new(),
            parent_id: None,
            source_file: "[agent]".to_string(),
            heading: None,
            visibility: "normal".to_string(),
            perspectives: vec![],
            relations: vec![],
            entry_type: None,
            items,
            impression_hint: None,
            impression_strength: None,
            scope: "project".to_string(),
        };
        let result = execute_store(
            &test_ctx(&store, &embedder, &blob_store, _dir.path()),
            input,
        )
        .await
        .unwrap();
        let msg = result.as_str().unwrap();
        assert!(msg.contains("Stored 2 entries"), "got: {msg}");

        let entries = store.list_entries(&[], None, None, 10).await.unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[tokio::test]
    async fn execute_store_unknown_entry_type_returns_error() {
        let (store, blob_store, _dir) = make_test_store_with_dir().await;
        let embedder: Arc<dyn crate::Embedder + Send + Sync> = Arc::new(MockEmbedder::new());

        let input = StoreInput {
            content: "Test content".to_string(),
            parent_id: None,
            source_file: "[agent]".to_string(),
            heading: None,
            visibility: "normal".to_string(),
            perspectives: vec![],
            relations: vec![],
            entry_type: Some("unknown_type".to_string()),
            items: vec![],
            impression_hint: None,
            impression_strength: None,
            scope: "project".to_string(),
        };
        let result = execute_store(
            &test_ctx(&store, &embedder, &blob_store, _dir.path()),
            input,
        )
        .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unknown entry_type"));
    }

    #[tokio::test]
    async fn execute_store_long_content_includes_warning() {
        let (store, blob_store, _dir) = make_test_store_with_dir().await;
        let embedder: Arc<dyn crate::Embedder + Send + Sync> = Arc::new(MockEmbedder::new());

        let long_content = "x".repeat(2001);
        let input = StoreInput {
            content: long_content,
            parent_id: None,
            source_file: "[agent]".to_string(),
            heading: None,
            visibility: "normal".to_string(),
            perspectives: vec![],
            relations: vec![],
            entry_type: None,
            items: vec![],
            impression_hint: None,
            impression_strength: None,
            scope: "project".to_string(),
        };
        let result = execute_store(
            &test_ctx(&store, &embedder, &blob_store, _dir.path()),
            input,
        )
        .await
        .unwrap();
        let msg = result.as_str().unwrap();
        assert!(msg.contains("2000 chars"), "got: {msg}");
    }

    #[tokio::test]
    async fn execute_store_branch_scope_adds_branch_perspective() {
        let (store, blob_store, _dir) = make_test_store_with_dir().await;
        let embedder: Arc<dyn crate::Embedder + Send + Sync> = Arc::new(MockEmbedder::new());

        let input = StoreInput {
            content: "Branch-scoped entry".to_string(),
            parent_id: None,
            source_file: "[agent]".to_string(),
            heading: None,
            visibility: "normal".to_string(),
            perspectives: vec![],
            relations: vec![],
            entry_type: None,
            items: vec![],
            impression_hint: None,
            impression_strength: None,
            scope: "branch".to_string(),
        };
        execute_store(
            &ToolContext {
                project: Some("myproj".to_string()),
                branch: Some("main".to_string()),
                ..test_ctx(&store, &embedder, &blob_store, _dir.path())
            },
            input,
        )
        .await
        .unwrap();

        let entries = store.list_entries(&[], None, None, 10).await.unwrap();
        assert_eq!(entries.len(), 1);
        let persp = &entries[0].perspectives;
        assert!(
            persp.contains(&"branch:myproj@main".to_string()),
            "perspectives: {persp:?}"
        );
        assert!(
            persp.contains(&"project:myproj".to_string()),
            "perspectives: {persp:?}"
        );
    }

    // ── execute_focus ────────────────────────────────────────────────────

    #[tokio::test]
    async fn execute_focus_returns_node_with_children() {
        let (store, _blob_store, _dir) = make_test_store_with_dir().await;
        let embedder: Arc<dyn crate::Embedder + Send + Sync> = Arc::new(MockEmbedder::new());

        // All IDs must be lowercase hex for get_by_id_prefix to work
        let parent_id = "abcd1234deadbeef1234567890abcdef12345678";
        let parent = make_test_chunk(parent_id, "Parent content");

        let mut child =
            make_test_chunk("1234567890abcdef1234567890abcdef12345678", "Child content");
        child.parent_id = Some(parent_id.to_string());

        store.insert_chunks(vec![parent, child]).await.unwrap();

        // Use full ID for exact match
        let input = FocusInput {
            id: parent_id.to_string(),
            question: None,
            limit: 10,
        };
        let (ctx, _tmp) = test_ctx_recall(&store, &embedder);
        let response = execute_focus(&ctx, input).await.unwrap();

        assert_eq!(response.node.content, "Parent content");
        assert_eq!(response.children.len(), 1);
        assert_eq!(response.children[0].chunk.content, "Child content");
        assert!(response.children[0].relevance.is_none());
    }

    #[tokio::test]
    async fn execute_focus_not_found_returns_error() {
        let (store, _blob_store, _dir) = make_test_store_with_dir().await;
        let embedder: Arc<dyn crate::Embedder + Send + Sync> = Arc::new(MockEmbedder::new());

        let input = FocusInput {
            id: "0000000000000000000000000000000000000000".to_string(),
            question: None,
            limit: 10,
        };
        let (ctx, _tmp) = test_ctx_recall(&store, &embedder);
        let result = execute_focus(&ctx, input).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn execute_focus_with_question_returns_scored_children() {
        let (store, _blob_store, _dir) = make_test_store_with_dir().await;
        let embedder: Arc<dyn crate::Embedder + Send + Sync> = Arc::new(MockEmbedder::new());

        let parent_id = "beef5678deadbeef1234567890abcdef12345678";
        let mut parent = make_test_chunk(parent_id, "Parent node");
        parent.embedding = Some(vec![0.1f32; 384]);

        let child_id_a = "cafe5678deadbeef1234567890abcdef12345678";
        let mut child_a = make_test_chunk(child_id_a, "Architecture decisions");
        child_a.parent_id = Some(parent_id.to_string());
        child_a.embedding = Some(vec![0.2f32; 384]);

        let child_id_b = "fade5678deadbeef1234567890abcdef12345678";
        let mut child_b = make_test_chunk(child_id_b, "Implementation notes");
        child_b.parent_id = Some(parent_id.to_string());
        child_b.embedding = Some(vec![0.3f32; 384]);

        store
            .insert_chunks(vec![parent, child_a, child_b])
            .await
            .unwrap();

        let input = FocusInput {
            id: parent_id.to_string(),
            question: Some("architecture".to_string()),
            limit: 10,
        };
        let (ctx, _tmp) = test_ctx_recall(&store, &embedder);
        let response = execute_focus(&ctx, input).await.unwrap();

        assert_eq!(response.children.len(), 2);
        for child in &response.children {
            assert!(child.relevance.is_some(), "child missing relevance score");
        }
    }

    #[tokio::test]
    async fn execute_focus_respects_limit() {
        let (store, _blob_store, _dir) = make_test_store_with_dir().await;
        let embedder: Arc<dyn crate::Embedder + Send + Sync> = Arc::new(MockEmbedder::new());

        let parent_id = "dead9999deadbeef1234567890abcdef12345678";
        let parent = make_test_chunk(parent_id, "Parent");

        let children: Vec<_> = (0..5)
            .map(|i| {
                let id = format!("0000000{i}0000deadbeef1234567890abcdef1234{:04x}", i);
                let mut c = make_test_chunk(&id, &format!("Child {i}"));
                c.parent_id = Some(parent_id.to_string());
                c
            })
            .collect();

        let mut all = vec![parent];
        all.extend(children);
        store.insert_chunks(all).await.unwrap();

        let input = FocusInput {
            id: parent_id.to_string(),
            question: None,
            limit: 2,
        };
        let (ctx, _tmp) = test_ctx_recall(&store, &embedder);
        let response = execute_focus(&ctx, input).await.unwrap();

        assert!(
            response.children.len() <= 2,
            "expected at most 2 children, got {}",
            response.children.len()
        );
    }

    // ── execute_think: promote/demote/relate ─────────────────────────────

    #[tokio::test]
    async fn execute_think_promote_changes_visibility() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;
        // All-hex ID so get_by_id_prefix accepts the 8-char prefix
        let chunk_id = "9a3b1234deadbeef1234567890abcdef12345678";
        store
            .insert_chunks(vec![make_test_chunk(chunk_id, "content")])
            .await
            .unwrap();

        let input = ThinkInput {
            action: Some("promote".to_string()),
            hot_limit: None,
            stale_limit: None,
            id: Some(chunk_id[..8].to_string()),
            visibility: Some("always".to_string()),
            source_id: None,
            target_id: None,
            kind: None,
            degrade_after_days: None,
            degrade_to: None,
            degrade_from: None,
            direction: None,
        };
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await
        .unwrap();
        assert!(result.contains("Promoted"), "got: {result}");
        assert!(result.contains("always"), "got: {result}");

        let updated = store.get_by_id(chunk_id).await.unwrap().unwrap();
        assert_eq!(updated.visibility, "always");
    }

    #[tokio::test]
    async fn execute_think_promote_requires_id() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;
        let input = ThinkInput {
            action: Some("promote".to_string()),
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
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("requires 'id'"));
    }

    #[tokio::test]
    async fn execute_think_demote_defaults_to_deep_only() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;
        // All-hex ID so get_by_id_prefix accepts the 8-char prefix
        let chunk_id = "7f8e1234deadbeef1234567890abcdef12345678";
        let mut chunk = make_test_chunk(chunk_id, "content");
        chunk.visibility = "always".to_string();
        store.insert_chunks(vec![chunk]).await.unwrap();

        let input = ThinkInput {
            action: Some("demote".to_string()),
            hot_limit: None,
            stale_limit: None,
            id: Some(chunk_id[..8].to_string()),
            visibility: None,
            source_id: None,
            target_id: None,
            kind: None,
            degrade_after_days: None,
            degrade_to: None,
            degrade_from: None,
            direction: None,
        };
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await
        .unwrap();
        assert!(result.contains("Demoted"), "got: {result}");
        assert!(result.contains("deep_only"), "got: {result}");

        let updated = store.get_by_id(chunk_id).await.unwrap().unwrap();
        assert_eq!(updated.visibility, "deep_only");
    }

    #[tokio::test]
    async fn execute_think_demote_requires_id() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;
        let input = ThinkInput {
            action: Some("demote".to_string()),
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
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("requires 'id'"));
    }

    #[tokio::test]
    async fn execute_think_relate_adds_relation() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;
        // All-hex IDs so get_by_id_prefix accepts 8-char prefixes
        let src_id = "5c0e1234deadbeef1234567890abcdef12345678";
        let tgt_id = "6d1f2345deadbeef1234567890abcdef12345678";
        store
            .insert_chunks(vec![
                make_test_chunk(src_id, "source content"),
                make_test_chunk(tgt_id, "target content"),
            ])
            .await
            .unwrap();

        let input = ThinkInput {
            action: Some("relate".to_string()),
            hot_limit: None,
            stale_limit: None,
            id: None,
            visibility: None,
            source_id: Some(src_id[..8].to_string()),
            target_id: Some(tgt_id[..8].to_string()),
            kind: Some("related_to".to_string()),
            degrade_after_days: None,
            degrade_to: None,
            degrade_from: None,
            direction: None,
        };
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await
        .unwrap();
        assert!(result.contains("Added relation"), "got: {result}");
        assert!(result.contains("related_to"), "got: {result}");
    }

    #[tokio::test]
    async fn execute_think_relate_requires_source_id() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;
        let input = ThinkInput {
            action: Some("relate".to_string()),
            hot_limit: None,
            stale_limit: None,
            id: None,
            visibility: None,
            source_id: None,
            target_id: Some("target".to_string()),
            kind: Some("related_to".to_string()),
            degrade_after_days: None,
            degrade_to: None,
            degrade_from: None,
            direction: None,
        };
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("requires 'source_id'"));
    }

    #[tokio::test]
    async fn execute_think_relate_requires_target_id() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;
        let input = ThinkInput {
            action: Some("relate".to_string()),
            hot_limit: None,
            stale_limit: None,
            id: None,
            visibility: None,
            source_id: Some("source".to_string()),
            target_id: None,
            kind: Some("related_to".to_string()),
            degrade_after_days: None,
            degrade_to: None,
            degrade_from: None,
            direction: None,
        };
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("requires 'target_id'"));
    }

    // ── execute_think: configure_aging / apply_aging ──────────────────────

    #[tokio::test]
    async fn execute_think_configure_aging_updates_config() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;

        let input = ThinkInput {
            action: Some("configure_aging".to_string()),
            hot_limit: None,
            stale_limit: None,
            id: None,
            visibility: None,
            source_id: None,
            target_id: None,
            kind: None,
            degrade_after_days: Some(30),
            degrade_to: Some("expired".to_string()),
            degrade_from: Some(vec!["normal".to_string(), "deep_only".to_string()]),
            direction: None,
        };
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await
        .unwrap();
        assert!(result.contains("Aging configured"), "got: {result}");
        assert!(result.contains("30 days"), "got: {result}");
        assert!(result.contains("expired"), "got: {result}");
    }

    #[tokio::test]
    async fn execute_think_apply_aging_on_empty_store() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;

        let input = ThinkInput {
            action: Some("apply_aging".to_string()),
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
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await
        .unwrap();
        assert!(
            result.contains("No chunks needed aging") || result.contains("Aged"),
            "got: {result}"
        );
    }

    // ── execute_think: salience ───────────────────────────────────────────

    #[tokio::test]
    async fn execute_think_salience_empty_store_no_entries() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;

        let input = ThinkInput {
            action: Some("salience".to_string()),
            hot_limit: Some(5),
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
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await
        .unwrap();
        assert!(result.contains("No entries to analyze"), "got: {result}");
    }

    #[tokio::test]
    async fn execute_think_salience_with_entries_shows_report() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;
        // get_hot_chunks filters on access_total > 0 — set total to make it visible
        let mut chunk = make_test_chunk(
            "aa1234b5deadbeef1234567890abcdef12345678",
            "important content",
        );
        chunk.access_profile.total = 5;
        store.insert_chunks(vec![chunk]).await.unwrap();

        let input = ThinkInput {
            action: Some("salience".to_string()),
            hot_limit: Some(5),
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
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await
        .unwrap();
        assert!(result.contains("Salience Report"), "got: {result}");
    }

    // ── execute_recall: browse mode + project filter ─────────────────────

    #[tokio::test]
    async fn execute_recall_browse_mode_returns_all_entries() {
        let (store, _blob_store, _dir) = make_test_store_with_dir().await;
        let embedder: Arc<dyn crate::Embedder + Send + Sync> = Arc::new(MockEmbedder::new());

        store
            .insert_chunks(vec![
                make_test_chunk("entry1deadbeef1234567890abcdef123456789a", "Entry one"),
                make_test_chunk("entry2deadbeef1234567890abcdef123456789b", "Entry two"),
            ])
            .await
            .unwrap();

        let input = RecallInput {
            query: None,
            limit: 10,
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
        let (ctx, _tmp) = test_ctx_recall(&store, &embedder);
        let results = execute_recall(&ctx, input, None).await.unwrap();

        assert_eq!(results.len(), 2);
        for r in &results {
            assert_eq!(r.relevance, "browse");
            assert_eq!(r.score, 1.0);
        }
    }

    #[tokio::test]
    async fn execute_recall_browse_with_project_filter_excludes_other_projects() {
        let (store, _blob_store, _dir) = make_test_store_with_dir().await;
        let embedder: Arc<dyn crate::Embedder + Send + Sync> = Arc::new(MockEmbedder::new());

        let mut chunk_a = make_test_chunk(
            "projaaaa1deadbeef1234567890abcdef12345678",
            "Project A content",
        );
        chunk_a.perspectives = vec!["project:proj-a".to_string()];

        let mut chunk_b = make_test_chunk(
            "projbbbb1deadbeef1234567890abcdef12345678",
            "Project B content",
        );
        chunk_b.perspectives = vec!["project:proj-b".to_string()];

        let chunk_c = make_test_chunk(
            "unscoped1deadbeef1234567890abcdef12345678",
            "Unscoped content",
        );

        store
            .insert_chunks(vec![chunk_a, chunk_b, chunk_c])
            .await
            .unwrap();

        let input = RecallInput {
            query: None,
            limit: 10,
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
        let (base_ctx, _tmp) = test_ctx_recall(&store, &embedder);
        let ctx = ToolContext {
            project: Some("proj-a".to_string()),
            ..base_ctx
        };
        let results = execute_recall(&ctx, input, None).await.unwrap();

        // proj-a + unscoped → 2 results
        assert_eq!(
            results.len(),
            2,
            "expected proj-a + unscoped, got: {:?}",
            results.iter().map(|r| &r.chunk.id).collect::<Vec<_>>()
        );
        for r in &results {
            assert!(
                !r.chunk.perspectives.contains(&"project:proj-b".to_string()),
                "proj-b entry should be filtered out"
            );
        }
    }

    // ── execute_think: reflect (None action) ────────────────────────────

    #[tokio::test]
    async fn execute_think_reflect_empty_store() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;

        let input = ThinkInput {
            action: None,
            hot_limit: Some(5),
            stale_limit: Some(5),
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
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await
        .unwrap();
        assert!(result.contains("Hot Chunks"), "got: {result}");
        assert!(result.contains("Stale Chunks"), "got: {result}");
        assert!(result.contains("Suggested Actions"), "got: {result}");
    }

    #[tokio::test]
    async fn execute_think_reflect_with_entries_shows_count() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;
        store
            .insert_chunks(vec![make_test_chunk(
                "reflect1deadbeef1234567890abcdef12345678",
                "content to reflect on",
            )])
            .await
            .unwrap();

        let input = ThinkInput {
            action: None,
            hot_limit: Some(5),
            stale_limit: Some(5),
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
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await
        .unwrap();
        assert!(result.contains("Total chunks: 1"), "got: {result}");
    }

    #[test]
    fn test_validate_impression_strength_valid() {
        assert!(validate_impression_strength(None).is_ok());
        assert!(validate_impression_strength(Some(0.0)).is_ok());
        assert!(validate_impression_strength(Some(0.5)).is_ok());
        assert!(validate_impression_strength(Some(1.0)).is_ok());
    }

    #[test]
    fn test_validate_impression_strength_invalid() {
        let expected_msg = "impression_strength must be between 0.0 and 1.0";
        for bad in [1.1_f32, -0.1, 2.5] {
            let err = validate_impression_strength(Some(bad)).unwrap_err();
            assert!(
                err.to_string().contains(expected_msg),
                "expected message to contain '{expected_msg}', got: {err}"
            );
        }
    }

    // ── execute_think: prepare action ────────────────────────────────────

    #[tokio::test]
    async fn execute_think_prepare_empty_store() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;
        let input = ThinkInput {
            action: Some("prepare".to_string()),
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
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await
        .unwrap();
        assert!(
            result.contains("Nothing to prepare"),
            "expected empty-store message, got: {result}"
        );
    }

    #[tokio::test]
    async fn execute_think_prepare_with_entries_returns_preparation() {
        let (store, blob_store, dir) = make_test_store_with_dir().await;
        let mut chunk = make_test_chunk(
            "aabbccdd11223344aabbccdd11223344aabbccdd11223344aabbccdd11223344",
            "Architecture decision about async",
        );
        chunk.access_profile.record_access();
        store.insert_chunks(vec![chunk]).await.unwrap();

        let input = ThinkInput {
            action: Some("prepare".to_string()),
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
        let result = execute_think(
            &test_ctx(
                &store,
                &(Arc::new(MockEmbedder::new()) as Arc<dyn crate::Embedder + Send + Sync>),
                &blob_store,
                dir.path(),
            ),
            input,
            None,
        )
        .await
        .unwrap();
        assert!(
            result.contains("## Think Preparation"),
            "expected preparation header, got: {result}"
        );
        assert!(
            result.contains("### Task"),
            "expected Task section, got: {result}"
        );
        assert!(
            result.contains("### Memory State"),
            "expected Memory State section, got: {result}"
        );
        assert!(
            result.contains("### How to Apply"),
            "expected How to Apply section, got: {result}"
        );
        assert!(
            result.contains("store()"),
            "expected store() instruction, got: {result}"
        );
    }
}
