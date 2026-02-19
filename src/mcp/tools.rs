//! Tool implementation functions for the 5 MCP tools.

use std::sync::Arc;

use crate::aging::{self, AgingConfig};
use crate::embedder::FastEmbedder;
use crate::search::{HierarchicalSearch, SearchConfig};
use crate::store::LanceStore;
use crate::{Embedder, Result, VectorStore};

use super::types::*;

pub async fn execute_recall(
    store: &Arc<LanceStore>,
    embedder: &Arc<FastEmbedder>,
    input: RecallInput,
) -> Result<Vec<SearchResultResponse>> {
    let config = SearchConfig::for_query(input.limit, input.deep, input.recency.as_deref());

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
        entry_type: crate::chunk::EntryType::Raw,
        summarizes: vec![],
        visibility: input.visibility,
        relations: vec![],
        access_profile: crate::AccessProfile::new(),
        expires_at: None,
    };

    store.insert_chunks(vec![chunk]).await?;
    Ok(chunk_id)
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
        Some(unknown) => Err(crate::Error::config(format!(
            "Unknown think action: '{}'. Available: promote, demote, relate, configure_aging, apply_aging",
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
}
