//! Identity: emergent self-model from memory structure.
//!
//! Identity is computed purely from stored data — no LLM required.
//! It produces:
//! - **Centroids**: Weighted embedding averages per perspective
//! - **Open threads**: Unresolved contradictions and pending decisions
//! - **Priming blob**: Startup briefing for agents connecting to this memory

use crate::salience::{self, SalienceWeights};
use crate::{HierarchicalChunk, VectorStore};

/// A computed identity snapshot.
#[derive(Debug, Clone)]
pub struct IdentitySnapshot {
    /// Weighted centroid per perspective (perspective_id → centroid vector).
    pub centroids: Vec<PerspectiveCentroid>,
    /// Top entries by salience — the "core identity" knowledge.
    pub core_entries: Vec<CoreEntry>,
    /// Open threads: unresolved relations, pending decisions.
    pub open_threads: Vec<OpenThread>,
    /// Recent learnings (entries in "learnings" perspective with high salience).
    pub recent_learnings: Vec<CoreEntry>,
    /// Emergent clusters discovered via k-means on embeddings.
    pub emergent_clusters: Vec<EmergentCluster>,
}

impl IdentitySnapshot {
    /// Returns true when the snapshot contains no meaningful content.
    pub fn is_empty(&self) -> bool {
        self.core_entries.is_empty()
            && self.open_threads.is_empty()
            && self.recent_learnings.is_empty()
            && self.centroids.is_empty()
            && self.emergent_clusters.is_empty()
    }
}

/// An emergent cluster discovered from embedding similarity.
#[derive(Debug, Clone)]
pub struct EmergentCluster {
    pub cluster_id: String,
    /// Representative entry (highest membership probability).
    pub representative: CoreEntry,
    /// How many entries belong to this cluster.
    pub member_count: usize,
    /// Dominant perspectives across members.
    pub dominant_perspectives: Vec<String>,
}

/// Weighted centroid for a single perspective.
#[derive(Debug, Clone)]
pub struct PerspectiveCentroid {
    pub perspective: String,
    pub centroid: Vec<f32>,
    pub entry_count: usize,
    pub avg_salience: f32,
}

/// A high-salience entry that forms part of the core identity.
#[derive(Debug, Clone)]
pub struct CoreEntry {
    pub id: String,
    pub heading: Option<String>,
    pub content_preview: String,
    pub salience: f32,
    pub perspectives: Vec<String>,
}

/// An unresolved thread: a chunk that has been superseded, contradicted,
/// or has relations that suggest ongoing deliberation.
#[derive(Debug, Clone)]
pub struct OpenThread {
    pub id: String,
    pub heading: Option<String>,
    pub reason: String,
    pub related_ids: Vec<String>,
}

/// Compute a full identity snapshot from the store.
pub async fn compute_identity<S: VectorStore>(
    store: &S,
    data_dir: &std::path::Path,
) -> crate::Result<IdentitySnapshot> {
    let weights = SalienceWeights::default();

    // Fetch the most important entries (use a generous limit)
    let hot = store.get_hot_chunks(500).await?;

    // Compute centroids per perspective
    let perspectives = crate::perspective::load(data_dir)?;
    let centroids = compute_centroids(&hot, &perspectives, &weights);

    // Find core entries (top salient)
    let top = salience::top_salient(&hot, &weights, 15);
    let core_entries: Vec<CoreEntry> = top
        .iter()
        .map(|(idx, score)| {
            let chunk = &hot[*idx];
            CoreEntry {
                id: chunk.id.clone(),
                heading: chunk.heading.clone(),
                content_preview: truncate(&chunk.content, 200),
                salience: score.composite,
                perspectives: chunk.perspectives.clone(),
            }
        })
        .collect();

    // Find open threads
    let open_threads = find_open_threads(&hot);

    // Recent learnings: entries in "learnings" perspective with high salience
    let recent_learnings = hot
        .iter()
        .filter(|c| c.perspectives.iter().any(|p| p == "learnings"))
        .map(|c| {
            let score = salience::compute(c, &weights);
            CoreEntry {
                id: c.id.clone(),
                heading: c.heading.clone(),
                content_preview: truncate(&c.content, 200),
                salience: score.composite,
                perspectives: c.perspectives.clone(),
            }
        })
        .take(10)
        .collect();

    // Discover emergent clusters from embeddings (requires llm feature for SoftClusterer)
    #[cfg(feature = "llm")]
    let emergent_clusters = discover_clusters(&hot, &weights);
    #[cfg(not(feature = "llm"))]
    let emergent_clusters = Vec::new();

    Ok(IdentitySnapshot {
        centroids,
        core_entries,
        open_threads,
        recent_learnings,
        emergent_clusters,
    })
}

/// Compute salience-weighted centroids per perspective.
fn compute_centroids(
    chunks: &[HierarchicalChunk],
    perspectives: &[crate::perspective::Perspective],
    weights: &SalienceWeights,
) -> Vec<PerspectiveCentroid> {
    perspectives
        .iter()
        .filter_map(|p| {
            let members: Vec<_> = chunks
                .iter()
                .filter(|c| c.perspectives.iter().any(|cp| cp == &p.id) && c.embedding.is_some())
                .collect();

            if members.is_empty() {
                return None;
            }

            let dim = members[0].embedding.as_ref().unwrap().len();
            let mut centroid = vec![0.0f32; dim];
            let mut total_weight = 0.0f32;
            let mut total_salience = 0.0f32;

            for chunk in &members {
                let score = salience::compute(chunk, weights);
                let w = score.composite.max(0.01); // minimum weight to avoid zero-division
                total_weight += w;
                total_salience += score.composite;

                if let Some(ref emb) = chunk.embedding {
                    if emb.len() != dim {
                        continue; // skip mismatched embeddings
                    }
                    for (i, val) in emb.iter().enumerate() {
                        centroid[i] += val * w;
                    }
                }
            }

            if total_weight > 0.0 {
                for val in &mut centroid {
                    *val /= total_weight;
                }
            }

            Some(PerspectiveCentroid {
                perspective: p.id.clone(),
                centroid,
                entry_count: members.len(),
                avg_salience: total_salience / members.len() as f32,
            })
        })
        .collect()
}

/// Build an `EmergentCluster` from a group of member indices and their probabilities.
#[cfg(feature = "llm")]
fn build_cluster_info(
    cluster_id: String,
    members: Vec<(usize, f32)>,
    embedded: &[&HierarchicalChunk],
    weights: &SalienceWeights,
) -> EmergentCluster {
    use std::collections::HashMap;

    // Safe: members is non-empty (filtered to >= 2 before calling)
    let (rep_idx, _) = members
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(&members[0]);
    let rep = embedded[*rep_idx];
    let score = salience::compute(rep, weights);

    let mut persp_counts: HashMap<&str, usize> = HashMap::new();
    for (idx, _) in &members {
        for p in &embedded[*idx].perspectives {
            *persp_counts.entry(p.as_str()).or_default() += 1;
        }
    }
    let mut dominant: Vec<_> = persp_counts.into_iter().collect();
    dominant.sort_by(|a, b| b.1.cmp(&a.1));

    EmergentCluster {
        cluster_id,
        representative: CoreEntry {
            id: rep.id.clone(),
            heading: rep.heading.clone(),
            content_preview: truncate(&rep.content, 200),
            salience: score.composite,
            perspectives: rep.perspectives.clone(),
        },
        member_count: members.len(),
        dominant_perspectives: dominant
            .into_iter()
            .take(3)
            .map(|(p, _)| p.to_string())
            .collect(),
    }
}

/// Discover emergent clusters by running k-means on hot chunk embeddings.
#[cfg(feature = "llm")]
fn discover_clusters(
    chunks: &[HierarchicalChunk],
    weights: &SalienceWeights,
) -> Vec<EmergentCluster> {
    use crate::cluster::{Clusterer, SoftClusterer};
    use std::collections::HashMap;

    // Safe: filtered to Some above
    let embedded: Vec<_> = chunks.iter().filter(|c| c.embedding.is_some()).collect();
    if embedded.len() < 4 {
        return Vec::new();
    }

    let embeddings: Vec<Vec<f32>> = embedded
        .iter()
        .map(|c| c.embedding.as_ref().unwrap().clone())
        .collect();

    let clusterer = SoftClusterer::new().with_cluster_range(2, 8);
    let assignments = match clusterer.cluster(&embeddings) {
        Ok(a) => a,
        Err(e) => {
            tracing::warn!("Clustering failed: {e}");
            return Vec::new();
        }
    };

    // Group by primary (highest probability) cluster
    let mut groups: HashMap<String, Vec<(usize, f32)>> = HashMap::new();
    for assignment in &assignments {
        if let Some(best) = assignment.memberships.iter().max_by(|a, b| {
            a.probability
                .partial_cmp(&b.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            groups
                .entry(best.cluster_id.clone())
                .or_default()
                .push((assignment.index, best.probability));
        }
    }

    let mut clusters: Vec<EmergentCluster> = groups
        .into_iter()
        .filter(|(_, members)| members.len() >= 2)
        .map(|(cluster_id, members)| build_cluster_info(cluster_id, members, &embedded, weights))
        .collect();

    clusters.sort_by(|a, b| b.member_count.cmp(&a.member_count));
    clusters
}

/// Find open threads from the store: entries with unresolved relations.
///
/// Scans all entries (not just hot ones) so that unresolved items are surfaced
/// regardless of access count.
pub async fn open_threads_from_store<S: VectorStore>(store: &S) -> crate::Result<Vec<OpenThread>> {
    let all = store.list_entries(None, None, None, usize::MAX).await?;
    Ok(find_open_threads(&all))
}

/// Resolve open-thread IDs from the store when `ongoing` filtering is requested.
///
/// Returns `Some(HashSet)` when `ongoing` is true, `None` otherwise.
/// Callers use `passes_ongoing_filter` to filter entries.
///
/// NOTE: This scans the entire store (`list_entries` with `usize::MAX`).
/// Acceptable for current store sizes but should be optimized if stores
/// grow beyond ~10k entries (e.g. with a dedicated open-thread index).
pub async fn resolve_ongoing_filter<S: VectorStore>(
    store: &S,
    ongoing: bool,
) -> crate::Result<Option<std::collections::HashSet<String>>> {
    if ongoing {
        let threads = open_threads_from_store(store).await?;
        Ok(Some(threads.into_iter().map(|t| t.id).collect()))
    } else {
        Ok(None)
    }
}

/// Check whether an ID passes the ongoing filter.
///
/// Returns `true` if there is no filter or if the ID is in the filter set.
pub fn passes_ongoing_filter(filter: &Option<std::collections::HashSet<String>>, id: &str) -> bool {
    filter.as_ref().is_none_or(|ids| ids.contains(id))
}

/// Find open threads: entries with unresolved relations.
///
/// A chunk can match multiple criteria. Reasons are merged rather than
/// discarded so no context is lost.
pub(crate) fn find_open_threads(chunks: &[HierarchicalChunk]) -> Vec<OpenThread> {
    use std::collections::HashMap;

    let mut by_id: HashMap<String, OpenThread> = HashMap::new();

    for chunk in chunks {
        let mut reasons = Vec::new();
        let mut related = Vec::new();

        // Entries that have been superseded but are still "normal" visibility
        if chunk.is_superseded() && chunk.visibility == "normal" {
            reasons.push("Superseded but still visible — review or archive".to_string());
            for r in chunk.relations_of_kind(crate::relation::SUPERSEDED_BY) {
                related.push(r.target_id.clone());
            }
        }

        // Entries with many relations suggest active deliberation
        if chunk.relations.len() >= 3 && chunk.visibility == "normal" {
            reasons.push(format!(
                "High relation count ({}) — active deliberation or needs consolidation",
                chunk.relations.len()
            ));
            for r in &chunk.relations {
                if !related.contains(&r.target_id) {
                    related.push(r.target_id.clone());
                }
            }
        }

        if !reasons.is_empty() {
            let entry = by_id.entry(chunk.id.clone()).or_insert_with(|| OpenThread {
                id: chunk.id.clone(),
                heading: chunk.heading.clone(),
                reason: String::new(),
                related_ids: Vec::new(),
            });
            entry.reason = reasons.join("; ");
            entry.related_ids = related;
        }
    }

    let mut threads: Vec<OpenThread> = by_id.into_values().collect();
    threads.sort_by(|a, b| a.id.cmp(&b.id));
    threads
}

/// Generate a priming text for agent startup.
///
/// This is the "who am I, what's on my mind" briefing.
pub fn generate_priming(snapshot: &IdentitySnapshot) -> String {
    if snapshot.is_empty() {
        return String::new();
    }

    let mut priming = String::new();

    priming.push_str("# Identity Briefing\n\n");

    // Core knowledge
    if !snapshot.core_entries.is_empty() {
        priming.push_str("## Core Knowledge\n\n");
        priming.push_str("The most important things in memory:\n\n");
        for entry in &snapshot.core_entries {
            let heading = entry.heading.as_deref().unwrap_or("(untitled)");
            let persp = if entry.perspectives.is_empty() {
                String::new()
            } else {
                format!(" [{}]", entry.perspectives.join(", "))
            };
            priming.push_str(&format!(
                "- **{}**{} (salience: {:.2}): {}\n",
                heading, persp, entry.salience, entry.content_preview
            ));
        }
        priming.push('\n');
    }

    // Open threads
    if !snapshot.open_threads.is_empty() {
        priming.push_str("## Open Threads\n\n");
        priming.push_str("Unresolved items that may need attention:\n\n");
        for thread in &snapshot.open_threads {
            let heading = thread.heading.as_deref().unwrap_or("(untitled)");
            priming.push_str(&format!("- **{}**: {}\n", heading, thread.reason));
        }
        priming.push('\n');
    }

    // Recent learnings
    if !snapshot.recent_learnings.is_empty() {
        priming.push_str("## Recent Learnings\n\n");
        for learning in &snapshot.recent_learnings {
            let heading = learning.heading.as_deref().unwrap_or("(untitled)");
            priming.push_str(&format!(
                "- **{}**: {}\n",
                heading, learning.content_preview
            ));
        }
        priming.push('\n');
    }

    // Perspective coverage
    if !snapshot.centroids.is_empty() {
        priming.push_str("## Perspective Coverage\n\n");
        for c in &snapshot.centroids {
            priming.push_str(&format!(
                "- **{}**: {} entries, avg salience {:.2}\n",
                c.perspective, c.entry_count, c.avg_salience
            ));
        }
        priming.push('\n');
    }

    // Emergent clusters
    if !snapshot.emergent_clusters.is_empty() {
        priming.push_str("## Emergent Clusters\n\n");
        priming.push_str("Thematic groupings discovered from embedding similarity:\n\n");
        for cluster in &snapshot.emergent_clusters {
            let persp = if cluster.dominant_perspectives.is_empty() {
                String::new()
            } else {
                format!(" ({})", cluster.dominant_perspectives.join(", "))
            };
            let heading = cluster
                .representative
                .heading
                .as_deref()
                .unwrap_or("(untitled)");
            priming.push_str(&format!(
                "- **{}**: {} members{} — representative: {}\n",
                cluster.cluster_id, cluster.member_count, persp, heading
            ));
        }
        priming.push('\n');
    }

    priming
}

/// Truncate a string to a max length, appending "..." if truncated.
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.replace('\n', " ")
    } else {
        format!("{}...", s[..max].replace('\n', " "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChunkLevel, ChunkRelation};

    fn test_chunk(content: &str) -> HierarchicalChunk {
        HierarchicalChunk::new(
            content.to_string(),
            ChunkLevel::CONTENT,
            None,
            String::new(),
            "test.md".to_string(),
        )
    }

    #[test]
    fn test_truncate_short() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_long() {
        let long = "a".repeat(300);
        let result = truncate(&long, 200);
        assert!(result.ends_with("..."));
        assert_eq!(result.len(), 203); // 200 + "..."
    }

    #[test]
    fn test_truncate_newlines() {
        assert_eq!(truncate("hello\nworld", 20), "hello world");
    }

    #[test]
    fn test_find_open_threads_superseded() {
        let chunk =
            test_chunk("old decision").with_relation(ChunkRelation::superseded_by("newer-id"));
        let threads = find_open_threads(&[chunk]);
        assert_eq!(threads.len(), 1);
        assert!(threads[0].reason.contains("Superseded"));
    }

    #[test]
    fn test_find_open_threads_high_relations() {
        let mut chunk = test_chunk("contested point");
        chunk.relations.push(ChunkRelation::related_to("a"));
        chunk.relations.push(ChunkRelation::related_to("b"));
        chunk.relations.push(ChunkRelation::related_to("c"));
        let threads = find_open_threads(&[chunk]);
        assert_eq!(threads.len(), 1);
        assert!(threads[0].reason.contains("High relation count"));
    }

    #[test]
    fn test_find_open_threads_merged() {
        // Chunk that matches both criteria: reasons should be merged
        let mut chunk = test_chunk("both criteria");
        chunk.relations.push(ChunkRelation::superseded_by("newer"));
        chunk.relations.push(ChunkRelation::related_to("a"));
        chunk.relations.push(ChunkRelation::related_to("b"));
        let threads = find_open_threads(&[chunk]);
        assert_eq!(threads.len(), 1);
        // Both reasons should be present (merged with ";")
        assert!(threads[0].reason.contains("Superseded"));
        assert!(threads[0].reason.contains("High relation count"));
    }

    #[test]
    fn test_find_open_threads_archived_ignored() {
        let mut chunk = test_chunk("archived superseded");
        chunk.visibility = "deep_only".to_string();
        chunk.relations.push(ChunkRelation::superseded_by("newer"));
        let threads = find_open_threads(&[chunk]);
        // Already archived, not an open thread
        assert_eq!(threads.len(), 0);
    }

    #[test]
    fn test_compute_centroids_empty() {
        let perspectives = crate::perspective::defaults();
        let weights = SalienceWeights::default();
        let centroids = compute_centroids(&[], &perspectives, &weights);
        assert!(centroids.is_empty());
    }

    #[test]
    fn test_compute_centroids_with_data() {
        let mut chunk = test_chunk("decisions content");
        chunk.embedding = Some(vec![1.0, 0.0, 0.0]);
        chunk.perspectives = vec!["decisions".to_string()];

        let perspectives = crate::perspective::defaults();
        let weights = SalienceWeights::default();
        let centroids = compute_centroids(&[chunk], &perspectives, &weights);

        // Only "decisions" should have a centroid
        assert_eq!(centroids.len(), 1);
        assert_eq!(centroids[0].perspective, "decisions");
        assert_eq!(centroids[0].entry_count, 1);
        assert_eq!(centroids[0].centroid.len(), 3);
    }

    #[test]
    fn test_compute_centroids_weighted() {
        let mut c1 = test_chunk("decisions content 1");
        c1.embedding = Some(vec![1.0, 0.0, 0.0]);
        c1.perspectives = vec!["decisions".to_string()];
        // c1 has no accesses, low salience

        let mut c2 = test_chunk("decisions content 2");
        c2.embedding = Some(vec![0.0, 1.0, 0.0]);
        c2.perspectives = vec!["decisions".to_string()];
        c2.access_profile.record_access();
        c2.access_profile.record_access();
        c2.access_profile.record_access();
        // c2 has accesses, higher salience → should pull centroid toward [0,1,0]

        let perspectives = crate::perspective::defaults();
        let weights = SalienceWeights::default();
        let centroids = compute_centroids(&[c1, c2], &perspectives, &weights);

        assert_eq!(centroids.len(), 1);
        let c = &centroids[0];
        assert_eq!(c.entry_count, 2);
        // c2 has higher weight → centroid[1] > centroid[0]
        assert!(c.centroid[1] > c.centroid[0]);
    }

    #[test]
    fn test_generate_priming_empty() {
        let snapshot = IdentitySnapshot {
            centroids: vec![],
            core_entries: vec![],
            open_threads: vec![],
            recent_learnings: vec![],
            emergent_clusters: vec![],
        };
        let priming = generate_priming(&snapshot);
        assert!(priming.is_empty());
    }

    #[test]
    fn test_generate_priming_with_data() {
        let snapshot = IdentitySnapshot {
            centroids: vec![PerspectiveCentroid {
                perspective: "decisions".to_string(),
                centroid: vec![0.5, 0.5, 0.0],
                entry_count: 3,
                avg_salience: 0.42,
            }],
            core_entries: vec![CoreEntry {
                id: "abc123".to_string(),
                heading: Some("Backend Decision".to_string()),
                content_preview: "We chose Rust for the backend...".to_string(),
                salience: 0.85,
                perspectives: vec!["decisions".to_string()],
            }],
            open_threads: vec![OpenThread {
                id: "def456".to_string(),
                heading: Some("Database Choice".to_string()),
                reason: "Superseded but still visible".to_string(),
                related_ids: vec!["newer-id".to_string()],
            }],
            recent_learnings: vec![CoreEntry {
                id: "ghi789".to_string(),
                heading: Some("TLS Issues".to_string()),
                content_preview: "TLS cert validation fails in sandbox".to_string(),
                salience: 0.3,
                perspectives: vec!["learnings".to_string()],
            }],
            emergent_clusters: vec![],
        };
        let priming = generate_priming(&snapshot);
        assert!(priming.contains("Core Knowledge"));
        assert!(priming.contains("Backend Decision"));
        assert!(priming.contains("Open Threads"));
        assert!(priming.contains("Database Choice"));
        assert!(priming.contains("Recent Learnings"));
        assert!(priming.contains("TLS Issues"));
        assert!(priming.contains("Perspective Coverage"));
        assert!(priming.contains("decisions"));
    }

    #[test]
    fn test_core_entry_from_chunk() {
        let mut chunk = test_chunk("important knowledge about Rust");
        chunk.heading = Some("Rust Guide".to_string());
        chunk.perspectives = vec!["knowledge".to_string()];
        chunk.access_profile.record_access();

        let weights = SalienceWeights::default();
        let score = salience::compute(&chunk, &weights);

        let entry = CoreEntry {
            id: chunk.id.clone(),
            heading: chunk.heading.clone(),
            content_preview: truncate(&chunk.content, 200),
            salience: score.composite,
            perspectives: chunk.perspectives.clone(),
        };

        assert_eq!(entry.heading.as_deref(), Some("Rust Guide"));
        assert!(entry.salience > 0.0);
        assert_eq!(entry.perspectives, vec!["knowledge"]);
    }
}
