//! Tool implementation functions for the 5 MCP tools.

use std::sync::Arc;

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
}

/// Store a single entry and return its chunk ID.
async fn store_single_entry(
    store: &Arc<LanceStore>,
    embedder: &Arc<FastEmbedder>,
    input: StoreSingleInput,
) -> Result<String> {
    let parent_id = input.parent_id.as_deref().filter(|s| !s.is_empty());

    let level = if let Some(pid) = parent_id {
        if let Ok(Some(parent)) = store.get_by_id(pid).await {
            crate::chunk::ChunkLevel(parent.level.0 + 1)
        } else {
            crate::chunk::ChunkLevel(7)
        }
    } else {
        crate::chunk::ChunkLevel(1)
    };

    let path = if let Some(pid) = parent_id {
        if let Ok(Some(parent)) = store.get_by_id(pid).await {
            format!("{}/agent", parent.path)
        } else {
            input.source_file.clone()
        }
    } else {
        input.source_file.clone()
    };

    let embeddings = embedder.embed(&[input.content.as_str()])?;
    let embedding = embeddings
        .into_iter()
        .next()
        .ok_or_else(|| crate::Error::embedding("Failed to generate embedding"))?;

    let chunk_id = crate::chunk::content_hash(&input.content);

    let entry_type = match input.entry_type.as_deref() {
        Some("summary") => crate::chunk::EntryType::Summary,
        Some("meta") => crate::chunk::EntryType::Meta,
        Some("impression") => crate::chunk::EntryType::Impression,
        _ => crate::chunk::EntryType::Raw,
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
    };

    store.insert_chunks(vec![chunk]).await?;

    // Process relations atomically after insert
    for rel in &input.relations {
        match rel.kind.as_str() {
            "supersedes" | "version_of" => {
                let inverse = crate::ChunkRelation::new("superseded_by", &chunk_id);
                store.add_relation(&rel.target_id, inverse).await?;
            }
            "summarizes" => {
                let inverse = crate::ChunkRelation::new("summarized_by", &chunk_id);
                store.add_relation(&rel.target_id, inverse).await?;
            }
            "related_to" => {
                let forward = crate::ChunkRelation::new("related_to", &rel.target_id);
                store.add_relation(&chunk_id, forward).await?;
                let backward = crate::ChunkRelation::new("related_to", &chunk_id);
                store.add_relation(&rel.target_id, backward).await?;
            }
            "derived_from" => {
                let forward = crate::ChunkRelation::new("derived_from", &rel.target_id);
                store.add_relation(&chunk_id, forward).await?;
            }
            other => {
                let forward = crate::ChunkRelation::new(other, &rel.target_id);
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
                input.limit * 3
            } else {
                input.limit
            };
            let config = SearchConfig::for_query(fetch_limit, input.deep, input.recency.as_deref())
                .with_perspective(input.perspective);
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
        .get_by_id(&input.id)
        .await?
        .ok_or_else(|| crate::Error::not_found(format!("Chunk not found: {}", input.id)))?;

    let children = store.get_children(&input.id).await?;

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
        Some(unknown) => Err(crate::Error::config(format!(
            "Unknown think action: '{}'. Available: promote, demote, relate, configure_aging, apply_aging, salience, consolidate",
            unknown
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

/// Parse a temporal string (ISO 8601 date "2026-02-20" or epoch seconds "1740000000") to epoch seconds.
fn parse_temporal(s: &str) -> Option<i64> {
    // Try epoch seconds first
    if let Ok(epoch) = s.parse::<i64>() {
        return Some(epoch);
    }
    // Try ISO 8601 date (YYYY-MM-DD)
    if s.len() == 10 && s.as_bytes()[4] == b'-' && s.as_bytes()[7] == b'-' {
        let year: i32 = s[0..4].parse().ok()?;
        let month: u32 = s[5..7].parse().ok()?;
        let day: u32 = s[8..10].parse().ok()?;
        let days = days_since_epoch(year, month, day)?;
        return Some(days * 86400);
    }
    None
}

/// Convert a calendar date to days since Unix epoch (1970-01-01).
/// Uses the algorithm from http://howardhinnant.github.io/date_algorithms.html
fn days_since_epoch(year: i32, month: u32, day: u32) -> Option<i64> {
    let y = if month <= 2 { year - 1 } else { year } as i64;
    let m = if month <= 2 { month + 9 } else { month - 3 } as i64;
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * m + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146097 + doe - 719468)
}

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

    #[test]
    fn test_parse_temporal_epoch() {
        assert_eq!(parse_temporal("1740000000"), Some(1740000000));
        assert_eq!(parse_temporal("0"), Some(0));
    }

    #[test]
    fn test_parse_temporal_iso_date() {
        // 1970-01-01 = epoch 0
        assert_eq!(parse_temporal("1970-01-01"), Some(0));
        // 2026-02-20 should produce a reasonable epoch
        let result = parse_temporal("2026-02-20");
        assert!(result.is_some());
        let epoch = result.unwrap();
        // Should be around 2026 (> 2025-01-01 = ~1735689600)
        assert!(epoch > 1_735_689_600);
    }

    #[test]
    fn test_parse_temporal_invalid() {
        assert_eq!(parse_temporal("not-a-date"), None);
        // Malformed date formats (not YYYY-MM-DD and not a valid integer) return None
        assert_eq!(parse_temporal("2026/02/20"), None);
        assert_eq!(parse_temporal("Feb 20 2026"), None);
        assert_eq!(parse_temporal(""), None);
        // "20260220" is a valid integer epoch, not invalid
        assert!(parse_temporal("20260220").is_some());
    }
}
