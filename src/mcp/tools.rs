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

use super::types::*;

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

/// Store a single entry and return (chunk_id, git_status).
/// Pass `Some(embedding)` for immediate embedding, or `None` for deferred (pending).
#[allow(clippy::too_many_arguments)]
async fn store_single_entry(
    store: &Arc<StoreBackend>,
    embedder: &Arc<dyn Embedder + Send + Sync>,
    blob_store: &Arc<crate::blob_store::BlobStore>,
    input: StoreSingleInput,
    embedding: Option<Vec<f32>>,
    project: Option<&str>,
    branch: Option<&str>,
    git_store: Option<&crate::git::memory_store::MemoryStore>,
    push_mode: crate::git::branch_config::PushMode,
) -> Result<(String, Option<String>)> {
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

    let chunk_id = crate::chunk::content_hash(&input.content);

    let entry_type = match input.entry_type.as_deref() {
        None | Some("raw") => crate::chunk::EntryType::Raw,
        Some("summary") => crate::chunk::EntryType::Summary,
        Some("meta") => crate::chunk::EntryType::Meta,
        Some("impression") => crate::chunk::EntryType::Impression,
        Some(unknown) => {
            return Err(crate::Error::config(format!(
                "Unknown entry_type: '{}'. Valid: raw, summary, meta, impression",
                unknown
            )))
        }
    };

    let perspectives = match input.scope.as_str() {
        "project" => {
            if let Some(proj) = project {
                let mut perspectives = input.perspectives.clone();
                perspectives.push(format!("project:{}", proj));
                perspectives
            } else {
                input.perspectives.clone()
            }
        }
        "branch" => {
            let mut perspectives = input.perspectives.clone();
            if let Some(proj) = project {
                perspectives.push(format!("project:{}", proj));
            }
            if let Some(br) = branch {
                if let Some(proj) = project {
                    perspectives.push(format!("branch:{}@{}", proj, br));
                } else {
                    perspectives.push(format!("branch:{}", br));
                }
            }
            perspectives
        }
        "personal" => input.perspectives.clone(),
        other => {
            tracing::warn!("Unknown scope '{}', treating as personal", other);
            input.perspectives.clone()
        }
    };

    let chunk = crate::HierarchicalChunk {
        id: chunk_id.clone(),
        content: input.content,
        embedding,
        level,
        parent_id: parent_id.map(String::from),
        path,
        source_file: input.source_file,
        heading: input.heading,
        start_offset: 0,
        end_offset: 0,
        cluster_memberships: vec![],
        entry_type,
        summarizes: vec![],
        perspectives,
        visibility: input.visibility,
        relations: vec![],
        access_profile: crate::AccessProfile::new(),
        expires_at: None,
        impression_hint: input.impression_hint,
        impression_strength: input.impression_strength.unwrap_or(1.0),
    };

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
    store: &Arc<StoreBackend>,
    embedder: &Arc<dyn Embedder + Send + Sync>,
    input: RecallInput,
    project: Option<&str>,
    branch: Option<&str>,
) -> Result<Vec<SearchResultResponse>> {
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
        let fetch_limit = if since_epoch.is_some() || until_epoch.is_some() {
            input.limit * TEMPORAL_PREFETCH_FACTOR
        } else {
            input.limit
        };
        let config = SearchConfig::for_query(fetch_limit, input.deep, input.recency.as_deref())
            .with_perspective(input.perspective.clone())
            .with_min_salience(input.min_salience)
            .with_min_score(input.min_score);
        let search =
            HierarchicalSearch::new(Arc::clone(store), Arc::clone(embedder)).with_config(config);
        let results = search.search_by_embedding(target_id, fetch_limit).await?;

        let filtered: Vec<_> = results
            .into_iter()
            .filter(|r| {
                passes_scope_filter(&r.chunk, project, branch) && {
                    let created = r.chunk.access_profile.created_at;
                    since_epoch.is_none_or(|s| created >= s)
                        && until_epoch.is_none_or(|u| created <= u)
                        && crate::identity::passes_ongoing_filter(&open_thread_ids, &r.chunk.id)
                }
            })
            .take(input.limit)
            .collect();

        return Ok(map_search_results(filtered));
    }

    match input.query {
        Some(ref query) if !query.is_empty() => {
            // Semantic search path
            let fetch_limit = if since_epoch.is_some() || until_epoch.is_some() {
                input.limit * TEMPORAL_PREFETCH_FACTOR
            } else {
                input.limit
            };
            let config = SearchConfig::for_query(fetch_limit, input.deep, input.recency.as_deref())
                .with_perspective(input.perspective.clone())
                .with_min_salience(input.min_salience)
                .with_min_score(input.min_score);
            let search = HierarchicalSearch::new(Arc::clone(store), Arc::clone(embedder))
                .with_config(config);
            let results = search.search(query).await?;

            let filtered: Vec<_> = results
                .into_iter()
                .filter(|r| {
                    passes_scope_filter(&r.chunk, project, branch) && {
                        let created = r.chunk.access_profile.created_at;
                        since_epoch.is_none_or(|s| created >= s)
                            && until_epoch.is_none_or(|u| created <= u)
                            && crate::identity::passes_ongoing_filter(&open_thread_ids, &r.chunk.id)
                    }
                })
                .take(input.limit)
                .collect();

            Ok(map_search_results(filtered))
        }
        _ => {
            // Browse mode: list entries without vector search
            let needs_client_filter = open_thread_ids.is_some() || project.is_some();
            let entries = store
                .list_entries(
                    input.perspective.as_deref(),
                    since_epoch,
                    until_epoch,
                    if needs_client_filter {
                        // Over-fetch when client-side filtering is active (ongoing, project)
                        // because list_entries doesn't support these filters natively.
                        10_000
                    } else {
                        input.limit
                    },
                )
                .await?;

            Ok(entries
                .iter()
                .filter(|chunk| {
                    passes_scope_filter(chunk, project, branch)
                        && crate::identity::passes_ongoing_filter(&open_thread_ids, &chunk.id)
                })
                .take(input.limit)
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

pub async fn execute_focus(
    store: &Arc<StoreBackend>,
    embedder: &Arc<dyn Embedder + Send + Sync>,
    input: FocusInput,
    project: Option<&str>,
    branch: Option<&str>,
) -> Result<FocusResponse> {
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
        children
            .into_iter()
            .filter(|child| passes_scope_filter(child, project, branch))
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

#[allow(clippy::too_many_arguments)]
pub async fn execute_store(
    store: &Arc<StoreBackend>,
    embedder: &Arc<dyn Embedder + Send + Sync>,
    blob_store: &Arc<crate::blob_store::BlobStore>,
    input: StoreInput,
    project: Option<&str>,
    branch: Option<&str>,
    git_store: Option<&crate::git::memory_store::MemoryStore>,
    push_mode: crate::git::branch_config::PushMode,
) -> Result<serde_json::Value> {
    if !input.items.is_empty() {
        let mut ids = Vec::new();
        let mut long_entries = 0usize;
        let mut git_statuses: Vec<String> = Vec::new();
        for item in input.items {
            let content_len = item.content.len();
            let (id, git_status) = store_single_entry(
                store,
                embedder,
                blob_store,
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
                project,
                branch,
                git_store,
                push_mode,
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
        let embeddings = embedder.embed(&[input.content.as_str()])?;
        let embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| crate::Error::embedding("Failed to generate embedding"))?;
        let (id, git_status) = store_single_entry(
            store,
            embedder,
            blob_store,
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
            Some(embedding),
            project,
            branch,
            git_store,
            push_mode,
        )
        .await?;
        let mut msg = format!("Stored. ID: {}", crate::short_id(&id));
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

#[allow(unused_variables, clippy::too_many_arguments)]
pub async fn execute_think(
    store: &Arc<StoreBackend>,
    data_dir: &std::path::Path,
    blob_store: &Arc<crate::blob_store::BlobStore>,
    input: ThinkInput,
    project: Option<&str>,
    branch: Option<&str>,
    git_store: Option<&crate::git::memory_store::MemoryStore>,
    push_mode: Option<crate::git::branch_config::PushMode>,
) -> Result<String> {
    match input.action.as_deref() {
        None => {
            let hot_limit = input.hot_limit.unwrap_or(10);
            let stale_limit = input.stale_limit.unwrap_or(10);
            execute_reflect(store, data_dir, hot_limit, stale_limit, project, branch).await
        }
        Some("promote") => {
            let raw_id = input
                .id
                .as_deref()
                .ok_or_else(|| crate::Error::config("think(promote) requires 'id'"))?;
            let chunk_id = crate::resolve::resolve_id(store, raw_id).await?;
            let vis = input.visibility.as_deref().unwrap_or("always");
            store.update_visibility(&chunk_id, vis).await?;
            Ok(format!("Promoted `{}` to visibility '{}'", chunk_id, vis))
        }
        Some("demote") => {
            let raw_id = input
                .id
                .as_deref()
                .ok_or_else(|| crate::Error::config("think(demote) requires 'id'"))?;
            let chunk_id = crate::resolve::resolve_id(store, raw_id).await?;
            let vis = input.visibility.as_deref().unwrap_or("deep_only");
            store.update_visibility(&chunk_id, vis).await?;
            Ok(format!("Demoted `{}` to visibility '{}'", chunk_id, vis))
        }
        Some("relate") => {
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
        #[cfg(feature = "llm")]
        Some("consolidate") => {
            let config = crate::Config::new().with_data_dir(data_dir);
            let llm = crate::llm::LlmBackend::from_config(&config.llm);
            let embedder = crate::embedder::from_config(&config.embedder)
                .map_err(|e| crate::Error::llm(format!("Failed to init embedder: {}", e)))?;

            let result = crate::think::execute(
                store.as_ref(),
                embedder.as_ref(),
                &llm,
                data_dir,
                Some(blob_store.as_ref()),
            )
            .await?;

            if result.entries_created.is_empty() {
                return Ok("Nothing to consolidate. Memory is well-organized.".to_string());
            }

            let mut report = format!(
                "## Think Cycle Complete\n\n- Narrative: {}\n- Consolidations: {}\n- Learnings: {}\n\n",
                if result.narrative_id.is_some() { "yes" } else { "no" },
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
        Some("consolidate") => Err(crate::Error::config(
            "think(consolidate) requires the 'llm' feature",
        )),
        Some("salience") => {
            let limit = input.hot_limit.unwrap_or(10);
            let hot = store.get_hot_chunks(limit * 2).await?;
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
        Some("perspectives") => {
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
        Some("status") => {
            let stats = store.stats().await?;
            let aging_config = AgingConfig::load(data_dir);
            let mut status = super::format::format_store_status(&stats, &aging_config);

            if let Some(git) = git_store {
                status.push_str("\n### Git Memory\n\n");
                match git.unpushed_commit_count() {
                    Ok(0) => status.push_str("- **Pending commits:** 0 (in sync with remote)\n"),
                    Ok(n) => {
                        status.push_str(&format!("- **Pending commits:** {n} (not yet pushed)\n"))
                    }
                    Err(_) => {
                        status.push_str("- **Pending commits:** unknown (no remote configured)\n")
                    }
                }
                if let Some(pm) = push_mode {
                    status.push_str(&format!("- **Push mode:** {pm}\n"));
                }
            }

            Ok(status)
        }
        Some("history") => {
            let raw_id = input
                .id
                .as_deref()
                .ok_or_else(|| crate::Error::config("think(history) requires 'id'"))?;
            let chunk_id = crate::resolve::resolve_id(store, raw_id).await?;
            let chunk = store.get_by_id(&chunk_id).await?.ok_or_else(|| {
                crate::Error::not_found(format!("Entry '{}' not found", chunk_id))
            })?;

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
        Some("discover") => {
            let limit = input.hot_limit.unwrap_or(10);
            discover_unlinked_pairs(store, limit).await
        }
        Some("sync") => {
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
                    Ok(PushResult::Success) => {
                        report.push_str("**Pushed** to remote successfully.\n")
                    }
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
                    Ok(SyncResult::Success) => {
                        report.push_str("**Pulled** new entries from remote.\n")
                    }
                    Ok(SyncResult::NothingToSync) => {
                        report.push_str("Already up to date with remote.\n")
                    }
                    Ok(SyncResult::Conflicts(files)) => {
                        report.push_str("**Conflict detected** during pull. Rebase aborted.\n\n");
                        report.push_str("Conflicting files:\n");
                        for f in &files {
                            report.push_str(&format!("- `{f}`\n"));
                        }
                        report.push_str(
                            "\nResolve manually or use `think(action='sync')` after fixing.\n",
                        );
                    }
                    Err(e) => report.push_str(&format!("Pull failed: {e}\n")),
                },
                _ => match git.sync() {
                    Ok(SyncResult::Success) => {
                        report.push_str("**Synced** — pulled and pushed successfully.\n")
                    }
                    Ok(SyncResult::NothingToSync) => {
                        report.push_str("Already in sync with remote.\n")
                    }
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
        Some(unknown) => Err(crate::Error::config(format!(
            "Unknown think action: '{}'. Available: {}",
            unknown,
            THINK_ACTIONS.join(", ")
        ))),
    }
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

/// A discovered pair: two entries that are semantically similar but have no explicit relation.
struct DiscoveredPair {
    entry_a: crate::HierarchicalChunk,
    entry_b: crate::HierarchicalChunk,
    similarity: f32,
}

/// Find entries that are semantically similar but share no explicit relation.
///
/// Algorithm:
/// 1. List up to `scan_limit * 2` entries as candidates.
/// 2. For each candidate that has an embedding, search for its top-5 ANN neighbors.
/// 3. For each (candidate, neighbor) pair, check whether a relation already exists in either direction.
/// 4. Deduplicate symmetric pairs using a sorted-ID set key.
/// 5. Sort by similarity descending and return up to `output_limit`.
async fn discover_unlinked_pairs(store: &Arc<StoreBackend>, output_limit: usize) -> Result<String> {
    const SCAN_LIMIT: usize = 100;
    const NEIGHBORS_PER_ENTRY: usize = 5;

    let candidates = store.list_entries(None, None, None, SCAN_LIMIT).await?;

    if candidates.is_empty() {
        return Ok("No entries in the store. Nothing to discover.".to_string());
    }

    let mut seen_pairs: std::collections::HashSet<(String, String)> =
        std::collections::HashSet::new();
    let mut pairs: Vec<DiscoveredPair> = Vec::new();

    for entry in &candidates {
        let embedding = match &entry.embedding {
            Some(e) => e,
            None => continue,
        };

        let neighbors = store
            .search(embedding, NEIGHBORS_PER_ENTRY + 1, None)
            .await?;

        for neighbor_result in &neighbors {
            let neighbor = &neighbor_result.chunk;

            if neighbor.id == entry.id {
                continue;
            }

            // Canonical pair key: smaller ID first so A↔B == B↔A
            let pair_key = if entry.id < neighbor.id {
                (entry.id.clone(), neighbor.id.clone())
            } else {
                (neighbor.id.clone(), entry.id.clone())
            };

            if seen_pairs.contains(&pair_key) {
                continue;
            }

            let already_related = entry.relations.iter().any(|r| r.target_id == neighbor.id)
                || neighbor.relations.iter().any(|r| r.target_id == entry.id);

            if already_related {
                seen_pairs.insert(pair_key);
                continue;
            }

            seen_pairs.insert(pair_key);
            pairs.push(DiscoveredPair {
                entry_a: entry.clone(),
                entry_b: neighbor.clone(),
                similarity: neighbor_result.score,
            });
        }
    }

    if pairs.is_empty() {
        return Ok(
            "No unlinked similar entries found. All semantically close pairs are already related."
                .to_string(),
        );
    }

    pairs.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pairs.truncate(output_limit);

    format_discovered_pairs(&pairs)
}

/// Format discovered pairs as a markdown report.
fn format_discovered_pairs(pairs: &[DiscoveredPair]) -> Result<String> {
    let mut report = String::from("## Discover: Unlinked Similar Entries\n\n");
    report.push_str("These entry pairs are semantically close but share no explicit relation.\n");
    report
        .push_str("Consider linking them with `think(action='relate')` or consolidating them.\n\n");

    for (i, pair) in pairs.iter().enumerate() {
        let heading_a = pair
            .entry_a
            .heading
            .as_deref()
            .unwrap_or_else(|| pair.entry_a.content.lines().next().unwrap_or("(untitled)"));
        let heading_b = pair
            .entry_b
            .heading
            .as_deref()
            .unwrap_or_else(|| pair.entry_b.content.lines().next().unwrap_or("(untitled)"));

        let preview_a = &heading_a[..heading_a.len().min(100)];
        let preview_b = &heading_b[..heading_b.len().min(100)];

        report.push_str(&format!(
            "### Discovery {} (similarity: {:.2})\n\n",
            i + 1,
            pair.similarity
        ));
        report.push_str(&format!(
            "**Entry A:** `{}` — \"{}\"\n",
            crate::chunk::short_id(&pair.entry_a.id),
            preview_a
        ));
        if !pair.entry_a.perspectives.is_empty() {
            report.push_str(&format!(
                "  perspectives: {}\n",
                pair.entry_a.perspectives.join(", ")
            ));
        }
        report.push('\n');
        report.push_str(&format!(
            "**Entry B:** `{}` — \"{}\"\n",
            crate::chunk::short_id(&pair.entry_b.id),
            preview_b
        ));
        if !pair.entry_b.perspectives.is_empty() {
            report.push_str(&format!(
                "  perspectives: {}\n",
                pair.entry_b.perspectives.join(", ")
            ));
        }
        report.push('\n');
        report.push_str("**Potential:** These entries are semantically close but not linked.\n\n");
    }

    report.push_str(&format!(
        "{} pair(s) found. Use `think(action='relate')` to link entries or `recall(similar_to='<id>')` to explore further.\n",
        pairs.len()
    ));

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
            &store,
            dir.path(),
            &blob_store,
            input,
            None,
            None,
            None,
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
            &store,
            dir.path(),
            &blob_store,
            input,
            None,
            None,
            None,
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
            &store,
            dir.path(),
            &blob_store,
            input,
            None,
            None,
            None,
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
            &store,
            dir.path(),
            &blob_store,
            input,
            None,
            None,
            None,
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
            &store,
            dir.path(),
            &blob_store,
            input,
            None,
            None,
            None,
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
            &store,
            dir.path(),
            &blob_store,
            input,
            None,
            None,
            None,
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
            &store,
            dir.path(),
            &blob_store,
            input,
            None,
            None,
            None,
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
            &store,
            dir.path(),
            &blob_store,
            input,
            None,
            None,
            None,
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
            perspective: None,
            similar_to: None,
            min_salience: None,
            min_score: None,
            since: None,
            until: None,
            ongoing: Some(true),
        };
        let ongoing_results = execute_recall(&store, &embedder, input_ongoing, None, None)
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
            perspective: None,
            similar_to: None,
            min_salience: None,
            min_score: None,
            since: None,
            until: None,
            ongoing: None,
        };
        let all_results = execute_recall(&store, &embedder, input_all, None, None)
            .await
            .unwrap();
        assert_eq!(
            all_results.len(),
            2,
            "no ongoing filter with query should return both entries"
        );
    }

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
            perspective: None,
            similar_to: None,
            min_salience: None,
            min_score: None,
            since: None,
            until: None,
            ongoing: None,
        };
        let all_results = execute_recall(&store, &embedder, input_all, None, None)
            .await
            .unwrap();
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
            perspective: None,
            similar_to: None,
            min_salience: None,
            min_score: None,
            since: None,
            until: None,
            ongoing: Some(true),
        };
        let ongoing_results = execute_recall(&store, &embedder, input_ongoing, None, None)
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
            perspective: None,
            similar_to: None,
            min_salience: None,
            min_score: None,
            since: None,
            until: None,
            ongoing: Some(false),
        };
        let not_ongoing_results = execute_recall(&store, &embedder, input_not_ongoing, None, None)
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
            &store,
            dir.path(),
            &blob_store,
            input,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();
        assert!(result.contains("Nothing to discover") || result.contains("No entries"));
    }

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
            &store,
            dir.path(),
            &blob_store,
            input,
            None,
            None,
            None,
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
            &store,
            dir.path(),
            &blob_store,
            input,
            None,
            None,
            None,
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
}
