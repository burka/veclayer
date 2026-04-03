//! Shared relation processing for CLI and MCP paths.
//!
//! Both the CLI `add` command and MCP `store` tool insert chunks with
//! `relations: vec![]` and then call [`process_relations`] post-insert.
//! This ensures consistent semantics: IDs are resolved, inverse links
//! are written on targets, and auto-demotion happens in one place.

use std::sync::Arc;

use tracing::info;

use crate::chunk::relation;
use crate::store::StoreBackend;
use crate::{ChunkRelation, Result, VectorStore};

/// An unprocessed relation parsed from CLI flags or MCP input.
#[derive(Debug, Clone)]
pub struct RawRelation {
    pub kind: String,
    pub target_id: String,
}

impl RawRelation {
    /// Build a list of `RawRelation` from typed CLI option fields.
    pub fn from_typed_options(
        supersedes: &[String],
        summarizes: &[String],
        related_to: &[String],
        derived_from: &[String],
        version_of: &[String],
    ) -> Vec<Self> {
        let pairs: &[(&str, &[String])] = &[
            ("supersedes", supersedes),
            ("summarizes", summarizes),
            ("related_to", related_to),
            ("derived_from", derived_from),
            ("version_of", version_of),
        ];
        pairs
            .iter()
            .flat_map(|(kind, targets)| {
                targets.iter().map(move |t| Self {
                    kind: kind.to_string(),
                    target_id: t.clone(),
                })
            })
            .collect()
    }

    /// Parse custom relation specs in `KIND:ID` format.
    pub fn parse_custom(specs: &[String]) -> crate::Result<Vec<Self>> {
        specs
            .iter()
            .map(|spec| {
                let (kind, target_id) = spec.split_once(':').ok_or_else(|| {
                    crate::Error::parse(format!(
                        "Invalid --rel format '{}': expected KIND:ID",
                        spec
                    ))
                })?;
                if kind.is_empty() || target_id.is_empty() {
                    return Err(crate::Error::parse(format!(
                        "Invalid --rel format '{}': expected KIND:ID",
                        spec
                    )));
                }
                validate_relation_kind(kind)?;
                Ok(Self {
                    kind: kind.to_string(),
                    target_id: target_id.to_string(),
                })
            })
            .collect()
    }
}

/// Validate a relation kind string.
///
/// - Known kinds pass without error.
/// - Unknown kinds that are within Levenshtein distance ≤ 2 of a known kind
///   produce an error with a "did you mean?" suggestion.
/// - Truly custom kinds (distance > 2 from all known kinds) pass silently.
pub fn validate_relation_kind(kind: &str) -> Result<()> {
    if relation::KNOWN_KINDS.contains(&kind) {
        return Ok(());
    }

    for &known in relation::KNOWN_KINDS {
        let distance = strsim::levenshtein(kind, known);
        if distance <= 2 {
            return Err(crate::Error::InvalidOperation(format!(
                "Unknown relation kind '{}' — did you mean '{}'?",
                kind, known
            )));
        }
    }

    Ok(())
}

/// Process relations after a chunk has been inserted.
///
/// Resolves short IDs, writes inverse links on targets, and auto-demotes
/// targets for `supersedes` / `version_of`.
pub async fn process_relations(
    store: &Arc<StoreBackend>,
    source_id: &str,
    relations: Vec<RawRelation>,
) -> Result<()> {
    for raw in relations {
        let target = crate::resolve::resolve_id(store, &raw.target_id).await?;

        match raw.kind.as_str() {
            "supersedes" | "version_of" => {
                let inverse = ChunkRelation::superseded_by(source_id);
                store.add_relation(&target, inverse).await?;
                info!("Auto-demoting superseded entry: {}", target);
                store
                    .update_visibility(&target, crate::visibility::EXPIRING)
                    .await?;
            }
            "summarizes" => {
                let inverse = ChunkRelation::summarized_by(source_id);
                store.add_relation(&target, inverse).await?;
            }
            "related_to" => {
                let forward = ChunkRelation::related_to(&target);
                store.add_relation(source_id, forward).await?;
                let backward = ChunkRelation::related_to(source_id);
                store.add_relation(&target, backward).await?;
            }
            "derived_from" => {
                let forward = ChunkRelation::derived_from(&target);
                store.add_relation(source_id, forward).await?;
            }
            other => {
                let forward = ChunkRelation::new(other, &target);
                store.add_relation(source_id, forward).await?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{make_test_chunk, test_store_with_chunks};

    /// Check if a chunk has a relation with the given kind and target_id.
    fn assert_chunk_has_relation(
        chunk: &crate::HierarchicalChunk,
        kind: &str,
        target_id: &str,
        label: &str,
    ) {
        assert!(
            chunk
                .relations
                .iter()
                .any(|r| r.kind == kind && r.target_id == target_id),
            "{label}: expected relation {kind}→{target_id}",
        );
    }

    /// Check that a chunk has no relations.
    fn assert_chunk_no_relations(chunk: &crate::HierarchicalChunk, label: &str) {
        assert!(
            chunk.relations.is_empty(),
            "{label}: expected no relations, got {:?}",
            chunk.relations
        );
    }

    // --- validate_relation_kind ---

    #[test]
    fn test_validate_known_kind_passes() {
        for kind in relation::KNOWN_KINDS {
            assert!(validate_relation_kind(kind).is_ok(), "failed for {kind}");
        }
    }

    #[test]
    fn test_validate_custom_kind_passes() {
        // Far enough from any known kind → accepted as custom
        assert!(validate_relation_kind("contradicts").is_ok());
        assert!(validate_relation_kind("inspired_by").is_ok());
        assert!(validate_relation_kind("blocks").is_ok());
    }

    /// Check that an invalid kind within Levenshtein distance ≤ 2 of a known kind
    /// produces an error with a "did you mean" suggestion for that kind.
    fn assert_did_you_mean_typo(bad_kind: &str, suggestion: &str) {
        let err = validate_relation_kind(bad_kind).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("did you mean"), "got: {msg}");
        assert!(msg.contains(suggestion), "got: {msg}");
    }

    #[test]
    fn test_validate_typo_suggests() {
        assert_did_you_mean_typo("supsersedes", "supersedes");
    }

    #[test]
    fn test_validate_typo_related() {
        assert_did_you_mean_typo("relatd_to", "related_to");
    }

    // --- process_relations ---

    #[tokio::test]
    async fn test_process_supersedes_demotes_and_inverses() {
        let (store, _data_dir, _dir) = test_store_with_chunks(vec![
            make_test_chunk("aaaa000000000000", "old content"),
            make_test_chunk("bbbb000000000000", "new content"),
        ])
        .await;

        let relations = vec![RawRelation {
            kind: "supersedes".to_string(),
            target_id: "aaaa0000".to_string(), // short hex prefix
        }];
        process_relations(&store, "bbbb000000000000", relations)
            .await
            .unwrap();

        let target_chunk = store.get_by_id("aaaa000000000000").await.unwrap().unwrap();
        assert_eq!(target_chunk.visibility, crate::visibility::EXPIRING);
        assert!(
            target_chunk
                .relations
                .iter()
                .any(|r| r.kind == "superseded_by" && r.target_id == "bbbb000000000000"),
            "expected superseded_by inverse on target, got: {:?}",
            target_chunk.relations
        );
    }

    #[tokio::test]
    async fn test_process_related_to_bidirectional() {
        let (store, _data_dir, _dir) = test_store_with_chunks(vec![
            make_test_chunk("cccc000000000000", "alpha"),
            make_test_chunk("dddd000000000000", "beta"),
        ])
        .await;

        let relations = vec![RawRelation {
            kind: "related_to".to_string(),
            target_id: "dddd0000".to_string(),
        }];
        process_relations(&store, "cccc000000000000", relations)
            .await
            .unwrap();

        let a_chunk = store.get_by_id("cccc000000000000").await.unwrap().unwrap();
        let b_chunk = store.get_by_id("dddd000000000000").await.unwrap().unwrap();

        assert_chunk_has_relation(&a_chunk, "related_to", "dddd000000000000", "source");
        assert_chunk_has_relation(&b_chunk, "related_to", "cccc000000000000", "target");
    }

    #[tokio::test]
    async fn test_process_derived_from_forward_only() {
        let (store, _data_dir, _dir) = test_store_with_chunks(vec![
            make_test_chunk("eeee000000000000", "original"),
            make_test_chunk("ffff000000000000", "derived"),
        ])
        .await;

        let relations = vec![RawRelation {
            kind: "derived_from".to_string(),
            target_id: "eeee0000".to_string(),
        }];
        process_relations(&store, "ffff000000000000", relations)
            .await
            .unwrap();

        let source_chunk = store.get_by_id("ffff000000000000").await.unwrap().unwrap();
        let target_chunk = store.get_by_id("eeee000000000000").await.unwrap().unwrap();
        assert_chunk_has_relation(&source_chunk, "derived_from", "eeee000000000000", "source");
        assert_chunk_no_relations(&target_chunk, "target");
    }

    #[tokio::test]
    async fn test_process_custom_kind_forward_only() {
        let (store, _data_dir, _dir) = test_store_with_chunks(vec![
            make_test_chunk("1111000000000000", "source"),
            make_test_chunk("2222000000000000", "target"),
        ])
        .await;

        let relations = vec![RawRelation {
            kind: "contradicts".to_string(),
            target_id: "22220000".to_string(),
        }];
        process_relations(&store, "1111000000000000", relations)
            .await
            .unwrap();

        let source_chunk = store.get_by_id("1111000000000000").await.unwrap().unwrap();
        let target_chunk = store.get_by_id("2222000000000000").await.unwrap().unwrap();
        assert_chunk_has_relation(&source_chunk, "contradicts", "2222000000000000", "source");
        assert_chunk_no_relations(&target_chunk, "target");
    }
}
