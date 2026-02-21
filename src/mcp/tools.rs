//! Tool implementation functions for the 5 MCP tools.

use std::sync::Arc;

// Over-fetch when temporal filters are active, then filter client-side
const TEMPORAL_PREFETCH_FACTOR: usize = 3;

const THINK_ACTIONS: &[&str] = &[
    "promote",
    "demote",
    "relate",
    "configure_aging",
    "apply_aging",
    "salience",
    "consolidate",
    "perspectives",
    "status",
    "history",
];

use crate::aging::{self, AgingConfig};
use crate::embedder::FastEmbedder;
use crate::search::{HierarchicalSearch, SearchConfig};
use crate::store::LanceStore;
use crate::{Embedder, Result, VectorStore};

use super::types::*;

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
}


/// Store a single entry and return its chunk ID.
async fn store_single_entry(
    store: &Arc<LanceStore>,
    embedder: &Arc<FastEmbedder>,
    input: StoreSingleInput,
) -> Result<String> {
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

    let embeddings = embedder.embed(&[input.content.as_str()])?;
    let embedding = embeddings
        .into_iter()
        .next()
        .ok_or_else(|| crate::Error::embedding("Failed to generate embedding"))?;

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
        entry_type,
        summarizes: vec![],
        perspectives: input.perspectives,
        visibility: input.visibility,
        relations: vec![],
        access_profile: crate::AccessProfile::new(),
        expires_at: None,
        impression_hint: input.impression_hint,
        impression_strength: input.impression_strength.unwrap_or(1.0),
    };

    store.insert_chunks(vec![chunk]).await?;

    // Process relations atomically after insert
    for rel in &input.relations {
        let target = crate::resolve::resolve_id(store, &rel.target_id).await?;
        match rel.kind.as_str() {
            "supersedes" | "version_of" => {
                let inverse = crate::ChunkRelation::new("superseded_by", &chunk_id);
                store.add_relation(&target, inverse).await?;
            }
            "summarizes" => {
                let inverse = crate::ChunkRelation::new("summarized_by", &chunk_id);
                store.add_relation(&target, inverse).await?;
            }
            "related_to" => {
                let forward = crate::ChunkRelation::new("related_to", &target);
                store.add_relation(&chunk_id, forward).await?;
                let backward = crate::ChunkRelation::new("related_to", &chunk_id);
                store.add_relation(&target, backward).await?;
            }
            "derived_from" => {
                let forward = crate::ChunkRelation::new("derived_from", &target);
                store.add_relation(&chunk_id, forward).await?;
            }
            other => {
                let forward = crate::ChunkRelation::new(other, &target);
                store.add_relation(&chunk_id, forward).await?;
            }
        }
    }

    Ok(chunk_id)
}

pub async fn execute_recall(
    store: &Arc<LanceStore>,
    embedder: &Arc<FastEmbedder>,
    input: RecallInput,
) -> Result<Vec<SearchResultResponse>> {
    let since_epoch = input.since.as_deref().and_then(parse_temporal);
    let until_epoch = input.until.as_deref().and_then(parse_temporal);

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
                    let created = r.chunk.access_profile.created_at;
                    since_epoch.is_none_or(|s| created >= s)
                        && until_epoch.is_none_or(|u| created <= u)
                })
                .take(input.limit)
                .collect();

            Ok(filtered
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
                .collect())
        }
        _ => {
            // Browse mode: list entries without vector search
            let entries = store
                .list_entries(
                    input.perspective.as_deref(),
                    since_epoch,
                    until_epoch,
                    input.limit,
                )
                .await?;

            Ok(entries
                .iter()
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
    store: &Arc<LanceStore>,
    embedder: &Arc<FastEmbedder>,
    input: FocusInput,
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

pub async fn execute_store(
    store: &Arc<LanceStore>,
    embedder: &Arc<FastEmbedder>,
    input: StoreInput,
) -> Result<serde_json::Value> {
    if !input.items.is_empty() {
        // Batch mode
        let mut ids = Vec::new();
        for item in input.items {
            let id = store_single_entry(
                store,
                embedder,
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
                },
            )
            .await?;
            ids.push(id);
        }
        Ok(serde_json::json!(ids))
    } else {
        // Single mode
        let id = store_single_entry(
            store,
            embedder,
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
            },
        )
        .await?;
        Ok(serde_json::json!(id))
    }
}

pub async fn execute_think(
    store: &Arc<LanceStore>,
    data_dir: &std::path::Path,
    input: ThinkInput,
) -> Result<String> {
    match input.action.as_deref() {
        None => {
            let hot_limit = input.hot_limit.unwrap_or(10);
            let stale_limit = input.stale_limit.unwrap_or(10);
            execute_reflect(store, data_dir, hot_limit, stale_limit).await
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
            let embedder = crate::embedder::FastEmbedder::new()
                .map_err(|e| crate::Error::llm(format!("Failed to init embedder: {}", e)))?;

            let result =
                crate::think::execute(store.as_ref(), &embedder, &llm, data_dir).await?;

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
        Some("consolidate") => {
            Err(crate::Error::config("think(consolidate) requires the 'llm' feature"))
        }
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
                    heading, score.composite, score.interaction, score.perspective, score.revision, chunk.id
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
                report.push_str(&format!(
                    "- **{}** — {}{}\n",
                    p.id, p.hint, tag
                ));
            }
            report.push_str(&format!("\n{} perspective(s) total.", perspectives.len()));
            Ok(report)
        }
        Some("status") => {
            let stats = store.stats().await?;
            let mut report = String::from("## Store Status\n\n");
            report.push_str(&format!("- **Total entries:** {}\n", stats.total_chunks));
            report.push_str(&format!("- **Source files:** {}\n", stats.source_files.len()));

            if !stats.chunks_by_level.is_empty() {
                report.push_str("\n### Entries by level\n\n");
                for level in 1..=7 {
                    if let Some(count) = stats.chunks_by_level.get(&level) {
                        let level_name = if level <= 6 {
                            format!("H{}", level)
                        } else {
                            "Content".to_string()
                        };
                        report.push_str(&format!("- {}: {}\n", level_name, count));
                    }
                }
            }

            if !stats.source_files.is_empty() {
                report.push_str("\n### Source files\n\n");
                for file in &stats.source_files {
                    report.push_str(&format!("- {}\n", file));
                }
            }

            let aging_config = AgingConfig::load(data_dir);
            report.push_str(&format!(
                "\n### Aging policy\n\n- Degrade {} → '{}' after {} days\n",
                aging_config.degrade_from.join("/"),
                aging_config.degrade_to,
                aging_config.degrade_after_days,
            ));

            Ok(report)
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
            let mut report = format!(
                "## Entry History: {} `{}`\n\n",
                heading, chunk.id
            );
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
        Some(unknown) => Err(crate::Error::config(format!(
            "Unknown think action: '{}'. Available: {}",
            unknown,
            THINK_ACTIONS.join(", ")
        ))),
    }
}

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

/// Parse a temporal string — delegates to `resolve::parse_temporal`.
fn parse_temporal(s: &str) -> Option<i64> {
    crate::resolve::parse_temporal(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    // resolve_id and parse_temporal tests are in resolve::tests.
    // These tests cover tool-specific logic that remains in this module.

    use crate::test_helpers::make_test_chunk;

    async fn make_test_store_with_dir() -> (Arc<LanceStore>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = LanceStore::open(dir.path(), 384).await.unwrap();
        (Arc::new(store), dir)
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
        let (store, dir) = make_test_store_with_dir().await;
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
        };
        let result = execute_think(&store, dir.path(), input).await.unwrap();
        assert!(result.contains("Perspectives"));
        // Should contain the built-in perspectives
        assert!(result.contains("decisions"));
        assert!(result.contains("knowledge"));
    }

    #[tokio::test]
    async fn test_think_status_action() {
        let (store, dir) = make_test_store_with_dir().await;
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
        };
        let result = execute_think(&store, dir.path(), input).await.unwrap();
        assert!(result.contains("Store Status"));
        assert!(result.contains("Total entries"));
        assert!(result.contains("1")); // 1 entry
    }

    #[tokio::test]
    async fn test_think_history_action() {
        let (store, dir) = make_test_store_with_dir().await;
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
        };
        let result = execute_think(&store, dir.path(), input).await.unwrap();
        assert!(result.contains("Entry History"));
        assert!(result.contains("Relations"));
        assert!(result.contains("supersedes"));
        assert!(result.contains("historical content"));
    }

    #[tokio::test]
    async fn test_think_history_requires_id() {
        let (store, dir) = make_test_store_with_dir().await;

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
        };
        let result = execute_think(&store, dir.path(), input).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("requires 'id'"));
    }

    #[tokio::test]
    async fn test_think_status_empty_store() {
        let (store, dir) = make_test_store_with_dir().await;

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
        };
        let result = execute_think(&store, dir.path(), input).await.unwrap();
        assert!(result.contains("Total entries"));
        assert!(result.contains("0"));
    }

    #[tokio::test]
    async fn test_think_unknown_action() {
        let (store, dir) = make_test_store_with_dir().await;

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
        };
        let result = execute_think(&store, dir.path(), input).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unknown think action"));
        assert!(err.contains("perspectives"));
        assert!(err.contains("status"));
        assert!(err.contains("history"));
    }
}
