//! Identity: emergent self-model from memory structure.
//!
//! Identity is computed purely from stored data — no LLM required.
//! It produces:
//! - **Centroids**: Weighted embedding averages per perspective
//! - **Open threads**: Unresolved contradictions and pending decisions
//! - **Priming blob**: Startup briefing for agents connecting to this memory

use crate::salience::{self, SalienceWeights};
use crate::util::preview;
use crate::{HierarchicalChunk, VectorStore};

/// Summary of activity on another branch.
#[derive(Debug, Clone)]
pub struct BranchActivity {
    /// Branch name (e.g., "feat/new-ui")
    pub branch: String,
    /// Number of entries scoped to this branch
    pub entry_count: usize,
    /// Most recent entry's heading or content preview
    pub latest_heading: Option<String>,
}

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
    /// Activity on other branches (for cross-branch awareness).
    pub other_branches: Vec<BranchActivity>,
}

impl IdentitySnapshot {
    /// Returns true when the snapshot contains no meaningful content.
    pub fn is_empty(&self) -> bool {
        self.core_entries.is_empty()
            && self.open_threads.is_empty()
            && self.recent_learnings.is_empty()
            && self.centroids.is_empty()
            && self.emergent_clusters.is_empty()
            && self.other_branches.is_empty()
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
    project: Option<&str>,
    branch: Option<&str>,
) -> crate::Result<IdentitySnapshot> {
    let weights = SalienceWeights::default();

    // Fetch the most important entries (use a generous limit)
    let mut hot = store.get_hot_chunks(500).await?;

    // Filter by project if specified
    if let Some(proj_name) = project {
        let project_tag = format!("project:{}", proj_name);
        hot.retain(|chunk| {
            let is_personal = !chunk.perspectives.iter().any(|p| p.starts_with("project:"))
                && !chunk.perspectives.iter().any(|p| p.starts_with("branch:"));
            let is_project = chunk.perspectives.contains(&project_tag);
            let current_branch_tag = branch.map(|b| format!("branch:{}@{}", proj_name, b));
            let is_current_branch = current_branch_tag
                .as_ref()
                .map(|tag| chunk.perspectives.contains(tag))
                .unwrap_or(false);
            let is_branch_scoped = chunk.perspectives.iter().any(|p| p.starts_with("branch:"));

            is_personal || (is_project && !is_branch_scoped) || is_current_branch
        });
    }

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
                content_preview: preview(&chunk.content, 200),
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
                content_preview: preview(&c.content, 200),
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

    // Detect other branches' activity
    let other_branches = if let Some(proj_name) = project {
        let branch_prefix = format!("branch:{}@", proj_name);
        let current_branch_tag = branch.map(|b| format!("branch:{}@{}", proj_name, b));

        // Get all entries and find unique branch tags
        let all_entries = store
            .list_entries(None, None, None, 1000)
            .await
            .unwrap_or_default();

        let mut branch_map: std::collections::HashMap<String, Vec<&HierarchicalChunk>> =
            std::collections::HashMap::new();

        for entry in &all_entries {
            for p in &entry.perspectives {
                if let Some(branch_name) = p.strip_prefix(&branch_prefix) {
                    // Skip current branch
                    if let Some(ref current) = current_branch_tag {
                        if p == current {
                            continue;
                        }
                    }
                    branch_map
                        .entry(branch_name.to_string())
                        .or_default()
                        .push(entry);
                }
            }
        }

        branch_map
            .into_iter()
            .map(|(branch, entries)| {
                let latest = entries.iter().max_by_key(|e| e.access_profile.created_at);
                BranchActivity {
                    branch,
                    entry_count: entries.len(),
                    latest_heading: latest.and_then(|e| {
                        e.heading
                            .clone()
                            .or_else(|| Some(e.content.chars().take(80).collect::<String>()))
                    }),
                }
            })
            .collect()
    } else {
        vec![]
    };

    Ok(IdentitySnapshot {
        centroids,
        core_entries,
        open_threads,
        recent_learnings,
        emergent_clusters,
        other_branches,
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
            content_preview: preview(&rep.content, 200),
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

    // Other branches
    if !snapshot.other_branches.is_empty() {
        priming.push_str("## Other Branches\n\n");
        for branch in &snapshot.other_branches {
            let heading = branch.latest_heading.as_deref().unwrap_or("(no heading)");
            priming.push_str(&format!(
                "- **{}**: {} entries — latest: {}\n",
                branch.branch, branch.entry_count, heading
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

/// Generate a compact briefing (~500 tokens) with only the most actionable context.
///
/// Includes: top 5 core entries, open threads, and recent learnings.
/// Omits: perspective coverage, emergent clusters, other branches.
pub fn generate_brief_priming(snapshot: &IdentitySnapshot) -> String {
    if snapshot.is_empty() {
        return String::new();
    }

    let mut priming = String::new();

    priming.push_str("# Memory Briefing (compact)\n\n");

    // Top 5 core entries only
    if !snapshot.core_entries.is_empty() {
        priming.push_str("## Key Knowledge\n\n");
        for entry in snapshot.core_entries.iter().take(5) {
            let heading = entry.heading.as_deref().unwrap_or("(untitled)");
            let persp = if entry.perspectives.is_empty() {
                String::new()
            } else {
                format!(" [{}]", entry.perspectives.join(", "))
            };
            priming.push_str(&format!(
                "- **{}**{}: {}\n",
                heading, persp, entry.content_preview
            ));
        }
        priming.push('\n');
    }

    // Open threads — always show, these are actionable
    if !snapshot.open_threads.is_empty() {
        priming.push_str("## Open Threads\n\n");
        for thread in &snapshot.open_threads {
            let heading = thread.heading.as_deref().unwrap_or("(untitled)");
            priming.push_str(&format!("- **{}**: {}\n", heading, thread.reason));
        }
        priming.push('\n');
    }

    // Recent learnings — compact, high signal
    if !snapshot.recent_learnings.is_empty() {
        priming.push_str("## Recent Learnings\n\n");
        for learning in snapshot.recent_learnings.iter().take(3) {
            let heading = learning.heading.as_deref().unwrap_or("(untitled)");
            priming.push_str(&format!(
                "- **{}**: {}\n",
                heading, learning.content_preview
            ));
        }
        priming.push('\n');
    }

    priming.push_str(
        "Use recall/store/think/focus via MCP tools. \
         Run `veclayer status` for usage help.\n",
    );

    priming
}

#[cfg(all(test, feature = "store-lance"))]
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
            other_branches: vec![],
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
            other_branches: vec![],
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
            content_preview: preview(&chunk.content, 200),
            salience: score.composite,
            perspectives: chunk.perspectives.clone(),
        };

        assert_eq!(entry.heading.as_deref(), Some("Rust Guide"));
        assert!(entry.salience > 0.0);
        assert_eq!(entry.perspectives, vec!["knowledge"]);
    }

    // ── open_threads_from_store ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_open_threads_from_store_empty_store() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = crate::store::StoreBackend::open_metadata(dir.path(), false)
            .await
            .unwrap();
        let threads = open_threads_from_store(&store).await.unwrap();
        assert!(threads.is_empty());
    }

    #[tokio::test]
    async fn test_open_threads_from_store_with_superseded_entry() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = crate::store::StoreBackend::open_metadata(dir.path(), false)
            .await
            .unwrap();

        let mut chunk = test_chunk("this decision is now obsolete");
        chunk
            .relations
            .push(ChunkRelation::superseded_by("newer-decision-id"));
        store.insert_chunks(vec![chunk]).await.unwrap();

        let threads = open_threads_from_store(&store).await.unwrap();
        assert_eq!(threads.len(), 1);
        assert!(threads[0].reason.contains("Superseded"));
    }

    // ── resolve_ongoing_filter ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_resolve_ongoing_filter_false_returns_none() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = crate::store::StoreBackend::open_metadata(dir.path(), false)
            .await
            .unwrap();
        let filter = resolve_ongoing_filter(&store, false).await.unwrap();
        assert!(filter.is_none());
    }

    #[tokio::test]
    async fn test_resolve_ongoing_filter_true_empty_store() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = crate::store::StoreBackend::open_metadata(dir.path(), false)
            .await
            .unwrap();
        let filter = resolve_ongoing_filter(&store, true).await.unwrap();
        assert!(filter.is_some());
        assert!(filter.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_resolve_ongoing_filter_true_with_open_thread() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = crate::store::StoreBackend::open_metadata(dir.path(), false)
            .await
            .unwrap();

        let mut chunk = test_chunk("pending decision");
        let chunk_id = chunk.id.clone();
        chunk.relations.push(ChunkRelation::superseded_by("newer"));
        store.insert_chunks(vec![chunk]).await.unwrap();

        let filter = resolve_ongoing_filter(&store, true).await.unwrap();
        let ids = filter.unwrap();
        assert!(ids.contains(&chunk_id));
    }

    // ── compute_identity ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_compute_identity_empty_store() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = crate::store::StoreBackend::open_metadata(dir.path(), false)
            .await
            .unwrap();
        let snapshot = compute_identity(&store, dir.path(), None, None)
            .await
            .unwrap();
        assert!(snapshot.is_empty());
    }

    #[tokio::test]
    async fn test_compute_identity_with_entries() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = crate::store::StoreBackend::open(dir.path(), 3, false)
            .await
            .unwrap();

        let mut chunk = test_chunk("Rust is great for systems programming");
        chunk.embedding = Some(vec![1.0, 0.0, 0.0]);
        chunk.perspectives = vec!["knowledge".to_string()];
        chunk.access_profile.record_access();
        store.insert_chunks(vec![chunk]).await.unwrap();

        let snapshot = compute_identity(&store, dir.path(), None, None)
            .await
            .unwrap();
        assert!(!snapshot.is_empty());
        assert!(!snapshot.core_entries.is_empty());
    }

    #[tokio::test]
    async fn test_compute_identity_recent_learnings_filtered() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = crate::store::StoreBackend::open(dir.path(), 3, false)
            .await
            .unwrap();

        let mut chunk = test_chunk("async Rust is complex but worth it");
        chunk.embedding = Some(vec![0.5, 0.5, 0.0]);
        chunk.perspectives = vec!["learnings".to_string()];
        chunk.access_profile.record_access();
        store.insert_chunks(vec![chunk]).await.unwrap();

        let snapshot = compute_identity(&store, dir.path(), None, None)
            .await
            .unwrap();
        assert!(!snapshot.recent_learnings.is_empty());
        assert!(snapshot.recent_learnings[0]
            .perspectives
            .contains(&"learnings".to_string()));
    }

    #[tokio::test]
    async fn test_compute_identity_project_filter_includes_personal_and_project() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = crate::store::StoreBackend::open(dir.path(), 3, false)
            .await
            .unwrap();

        let mut project_chunk = test_chunk("project-specific knowledge");
        project_chunk.embedding = Some(vec![0.1, 0.9, 0.0]);
        project_chunk.perspectives = vec!["project:my-project".to_string()];
        project_chunk.access_profile.record_access();

        let mut personal_chunk = test_chunk("personal general knowledge");
        personal_chunk.embedding = Some(vec![0.9, 0.1, 0.0]);
        personal_chunk.access_profile.record_access();

        store
            .insert_chunks(vec![project_chunk.clone(), personal_chunk.clone()])
            .await
            .unwrap();

        let snapshot = compute_identity(&store, dir.path(), Some("my-project"), None)
            .await
            .unwrap();

        let ids: Vec<_> = snapshot.core_entries.iter().map(|e| &e.id).collect();
        assert!(ids.contains(&&project_chunk.id) || ids.contains(&&personal_chunk.id));
    }

    #[tokio::test]
    async fn test_compute_identity_other_branches_detected() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = crate::store::StoreBackend::open(dir.path(), 3, false)
            .await
            .unwrap();

        let mut branch_chunk = test_chunk("feature branch work");
        branch_chunk.embedding = Some(vec![0.5, 0.5, 0.0]);
        branch_chunk.perspectives = vec!["branch:my-project@feat/other-feature".to_string()];
        store.insert_chunks(vec![branch_chunk]).await.unwrap();

        let snapshot = compute_identity(
            &store,
            dir.path(),
            Some("my-project"),
            Some("feat/main-branch"),
        )
        .await
        .unwrap();

        assert!(!snapshot.other_branches.is_empty());
        assert_eq!(snapshot.other_branches[0].entry_count, 1);
    }

    #[tokio::test]
    async fn test_compute_identity_same_branch_not_in_other_branches() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = crate::store::StoreBackend::open(dir.path(), 3, false)
            .await
            .unwrap();

        let mut chunk = test_chunk("current branch work");
        chunk.embedding = Some(vec![0.5, 0.5, 0.0]);
        chunk.perspectives = vec!["branch:my-project@main".to_string()];
        store.insert_chunks(vec![chunk]).await.unwrap();

        let snapshot = compute_identity(&store, dir.path(), Some("my-project"), Some("main"))
            .await
            .unwrap();

        assert!(snapshot.other_branches.is_empty());
    }

    // ── generate_priming edge cases ───────────────────────────────────────────

    #[test]
    fn test_generate_priming_with_emergent_clusters() {
        let snapshot = IdentitySnapshot {
            centroids: vec![],
            core_entries: vec![CoreEntry {
                id: "e1".to_string(),
                heading: Some("Core".to_string()),
                content_preview: "content".to_string(),
                salience: 0.5,
                perspectives: vec![],
            }],
            open_threads: vec![],
            recent_learnings: vec![],
            emergent_clusters: vec![EmergentCluster {
                cluster_id: "cluster-0".to_string(),
                representative: CoreEntry {
                    id: "e1".to_string(),
                    heading: Some("Rep Entry".to_string()),
                    content_preview: "rep content".to_string(),
                    salience: 0.6,
                    perspectives: vec!["decisions".to_string()],
                },
                member_count: 3,
                dominant_perspectives: vec!["decisions".to_string(), "knowledge".to_string()],
            }],
            other_branches: vec![],
        };
        let priming = generate_priming(&snapshot);
        assert!(priming.contains("Emergent Clusters"));
        assert!(priming.contains("cluster-0"));
        assert!(priming.contains("3 members"));
    }

    #[test]
    fn test_generate_priming_cluster_empty_dominant_perspectives() {
        let snapshot = IdentitySnapshot {
            centroids: vec![],
            core_entries: vec![CoreEntry {
                id: "e1".to_string(),
                heading: None,
                content_preview: "something".to_string(),
                salience: 0.5,
                perspectives: vec![],
            }],
            open_threads: vec![],
            recent_learnings: vec![],
            emergent_clusters: vec![EmergentCluster {
                cluster_id: "cluster-1".to_string(),
                representative: CoreEntry {
                    id: "e1".to_string(),
                    heading: None,
                    content_preview: "rep".to_string(),
                    salience: 0.5,
                    perspectives: vec![],
                },
                member_count: 2,
                dominant_perspectives: vec![],
            }],
            other_branches: vec![],
        };
        let priming = generate_priming(&snapshot);
        assert!(priming.contains("cluster-1"));
        // No parentheses for empty dominant_perspectives
        assert!(!priming.contains("()"));
    }

    #[test]
    fn test_generate_priming_other_branches_no_heading() {
        let snapshot = IdentitySnapshot {
            centroids: vec![],
            core_entries: vec![CoreEntry {
                id: "e1".to_string(),
                heading: Some("Root".to_string()),
                content_preview: "root content".to_string(),
                salience: 0.5,
                perspectives: vec![],
            }],
            open_threads: vec![],
            recent_learnings: vec![],
            emergent_clusters: vec![],
            other_branches: vec![BranchActivity {
                branch: "feat/old".to_string(),
                entry_count: 1,
                latest_heading: None,
            }],
        };
        let priming = generate_priming(&snapshot);
        assert!(priming.contains("(no heading)"));
        assert!(priming.contains("feat/old"));
    }

    #[test]
    fn test_generate_priming_core_entry_no_heading() {
        let snapshot = IdentitySnapshot {
            centroids: vec![],
            core_entries: vec![CoreEntry {
                id: "e1".to_string(),
                heading: None,
                content_preview: "something important".to_string(),
                salience: 0.7,
                perspectives: vec!["decisions".to_string()],
            }],
            open_threads: vec![],
            recent_learnings: vec![],
            emergent_clusters: vec![],
            other_branches: vec![],
        };
        let priming = generate_priming(&snapshot);
        assert!(priming.contains("(untitled)"));
        assert!(priming.contains("[decisions]"));
    }

    #[test]
    fn test_generate_priming_open_thread_no_heading() {
        let snapshot = IdentitySnapshot {
            centroids: vec![],
            core_entries: vec![CoreEntry {
                id: "e1".to_string(),
                heading: Some("Item".to_string()),
                content_preview: "content".to_string(),
                salience: 0.5,
                perspectives: vec![],
            }],
            open_threads: vec![OpenThread {
                id: "t1".to_string(),
                heading: None,
                reason: "pending decision".to_string(),
                related_ids: vec![],
            }],
            recent_learnings: vec![],
            emergent_clusters: vec![],
            other_branches: vec![],
        };
        let priming = generate_priming(&snapshot);
        assert!(priming.contains("Open Threads"));
        assert!(priming.contains("(untitled)"));
    }

    // ── find_open_threads edge cases ──────────────────────────────────────────

    #[test]
    fn test_find_open_threads_exactly_two_relations_not_triggered() {
        let mut chunk = test_chunk("two relations not enough");
        chunk.relations.push(ChunkRelation::related_to("a"));
        chunk.relations.push(ChunkRelation::related_to("b"));
        let threads = find_open_threads(&[chunk]);
        // Threshold is >= 3, so 2 relations should not create a thread
        assert!(threads.is_empty());
    }

    #[test]
    fn test_find_open_threads_deduplicates_related_ids() {
        let mut chunk = test_chunk("relation dedup check");
        // superseded_by adds the target; high-relations also adds all targets
        chunk
            .relations
            .push(ChunkRelation::superseded_by("shared-target"));
        chunk.relations.push(ChunkRelation::related_to("b"));
        chunk.relations.push(ChunkRelation::related_to("c"));
        // total == 3 AND superseded → both reasons apply
        let threads = find_open_threads(&[chunk]);
        assert_eq!(threads.len(), 1);
        let count = threads[0]
            .related_ids
            .iter()
            .filter(|id| *id == "shared-target")
            .count();
        assert_eq!(count, 1, "shared-target should appear exactly once");
    }

    // ── compute_centroids: skip mismatched embedding dimensions ───────────────

    #[test]
    fn test_compute_centroids_skips_mismatched_dimensions() {
        let mut c1 = test_chunk("correct dim");
        c1.embedding = Some(vec![1.0, 0.0, 0.0]);
        c1.perspectives = vec!["decisions".to_string()];

        let mut c2 = test_chunk("wrong dim");
        c2.embedding = Some(vec![0.5, 0.5]); // 2D instead of 3D
        c2.perspectives = vec!["decisions".to_string()];

        let perspectives = crate::perspective::defaults();
        let weights = SalienceWeights::default();
        let centroids = compute_centroids(&[c1, c2], &perspectives, &weights);

        // Both are counted in entry_count; c2's embedding is skipped
        assert_eq!(centroids.len(), 1);
        assert_eq!(centroids[0].entry_count, 2);
    }
}
