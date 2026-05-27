//! Think: the sleep cycle orchestrator.
//!
//! The think cycle is: reflect → LLM → add → compact.
//! VecLayer gathers context (mechanical), the LLM generates consolidations
//! and learnings, VecLayer writes them back and cleans up.
//!
//! Without an LLM, everything else works. Think is the only module that
//! requires an LLM — and it's optional.

use std::path::Path;
use std::sync::Arc;

use crate::chunk::{ChunkRelation, EntryType, HierarchicalChunk};
use crate::identity::{self, IdentitySnapshot};
use crate::llm::{DynLlmProvider, LlmProvider, Message};
use crate::store::StoreBackend;
use crate::util::preview;
use crate::{Embedder, Result, VectorStore};

/// Result of a think cycle.
#[derive(Debug, Default)]
pub struct ThinkResult {
    /// ID of the narrative entry (if generated).
    pub narrative_id: Option<String>,
    /// Number of consolidation summaries created.
    pub consolidations_added: usize,
    /// Number of meta-learnings extracted.
    pub learnings_added: usize,
    /// All entries created during this cycle.
    pub entries_created: Vec<ThinkEntry>,
}

impl ThinkResult {
    /// Returns true if the think cycle produced any new entries.
    pub fn is_empty(&self) -> bool {
        self.entries_created.is_empty()
    }

    /// Total number of entries created (narrative + consolidations + learnings).
    pub fn total(&self) -> usize {
        self.entries_created.len()
    }
}

/// An entry created by the think cycle.
#[derive(Debug)]
pub struct ThinkEntry {
    pub id: String,
    pub entry_type: EntryType,
    pub content_preview: String,
    pub perspectives: Vec<String>,
}

// --- LLM response structures ---

/// What the LLM produces (parsed from JSON).
#[derive(Debug, serde::Deserialize)]
struct ThinkPlan {
    #[serde(default)]
    narrative: Option<String>,
    #[serde(default)]
    consolidations: Vec<Consolidation>,
    #[serde(default)]
    learnings: Vec<Learning>,
}

#[derive(Debug, serde::Deserialize)]
struct Consolidation {
    content: String,
    entry_ids: Vec<String>,
    #[serde(default)]
    perspectives: Vec<String>,
}

#[derive(Debug, serde::Deserialize)]
struct Learning {
    content: String,
    #[serde(default = "default_learnings_perspective")]
    perspectives: Vec<String>,
}

fn default_learnings_perspective() -> Vec<String> {
    vec!["learnings".to_string()]
}

const UNTITLED: &str = "(untitled)";

// --- System prompt ---

pub(crate) const THINK_SYSTEM_PROMPT: &str = r#"You are reflecting on a knowledge base to consolidate and distill learnings.

You will receive an identity briefing showing the current state of memory. Based on this:

1. Write a brief narrative (2-3 sentences, first person) capturing the essence of what this memory contains and what matters most.
2. Identify groups of related core entries that should be consolidated into higher-level summaries.
3. Extract meta-learnings: patterns, recurring themes, or insights that emerge from the memory as a whole.

Respond ONLY with valid JSON (no markdown fences, no commentary):
{
  "narrative": "I am... My core focus is...",
  "consolidations": [
    {
      "content": "Summary synthesizing multiple entries...",
      "entry_ids": ["full-64-char-hash-1", "full-64-char-hash-2"],
      "perspectives": ["knowledge"]
    }
  ],
  "learnings": [
    {
      "content": "Pattern observed: ...",
      "perspectives": ["learnings"]
    }
  ]
}

Rules:
- Use full 64-character entry IDs from the briefing, not short IDs
- Narrative should be 2-3 sentences in first person
- Only consolidate entries that genuinely belong together (same project or topic)
- Learnings should be genuine meta-observations, not repetitions of existing entries
- If nothing needs consolidation, return empty arrays
- Keep each consolidation to 1-3 concise sentences
- perspectives must use existing perspective IDs from the briefing
- Look for contradictions: entries that conflict with each other. Note them in learnings
- Look for progress patterns: tasks started, completed, or stalled across entries"#;

// --- Preparation (LLM-free) ---

/// Data needed for a caller to perform the think cycle itself.
///
/// Returned by [`prepare()`] when the caller wants to reason about
/// consolidation without an LLM — e.g. when Ollama is unreachable and
/// the MCP caller (Claude) can do the reasoning.
#[derive(Debug)]
pub struct ThinkPreparation {
    /// The system prompt that would have been sent to the LLM.
    pub system_prompt: &'static str,
    /// The user prompt (priming + entry ID reference).
    pub user_prompt: String,
    /// Entry IDs with their headings, for reference in consolidations.
    pub entry_ids: Vec<(String, String)>,
}

/// Gather reflection data without calling an LLM.
///
/// Returns `None` if the store is empty (nothing to think about).
/// The caller can use the returned data to reason about consolidations
/// and then call `store()` for each result.
pub async fn prepare(
    store: &impl VectorStore,
    data_dir: &Path,
) -> Result<Option<ThinkPreparation>> {
    let snapshot = identity::compute_identity(store, data_dir, None, None).await?;

    if snapshot.core_entries.is_empty() {
        return Ok(None);
    }

    let priming = identity::generate_priming(&snapshot);
    let prompt = build_prompt(&priming, &snapshot);

    let entry_ids: Vec<(String, String)> = snapshot
        .core_entries
        .iter()
        .map(|e| {
            let heading = e.heading.clone().unwrap_or_else(|| UNTITLED.to_string());
            (e.id.clone(), heading)
        })
        .collect();

    Ok(Some(ThinkPreparation {
        system_prompt: THINK_SYSTEM_PROMPT,
        user_prompt: prompt,
        entry_ids,
    }))
}

// --- Main entry point ---

/// Execute one think cycle: reflect → LLM → add → compact.
///
/// When `project` is `Some`, only entries tagged with that project are
/// considered for consolidation. When `None`, all entries are included.
pub async fn execute<L: LlmProvider>(
    store: &impl VectorStore,
    embedder: &dyn Embedder,
    llm: &L,
    data_dir: &Path,
    blob_store: Option<&crate::blob_store::BlobStore>,
    project: Option<&str>,
) -> Result<ThinkResult> {
    // 1. Reflect: compute identity snapshot
    let snapshot = identity::compute_identity(store, data_dir, project, None).await?;
    let priming = identity::generate_priming(&snapshot);

    if snapshot.core_entries.is_empty() {
        return Ok(ThinkResult::default());
    }

    // 2. Build prompt with full entry IDs for reference
    let prompt = build_prompt(&priming, &snapshot);

    // 3. Call LLM
    let response = llm
        .complete(&[Message::system(THINK_SYSTEM_PROMPT), Message::user(prompt)])
        .await?;

    // 4. Parse and write back
    write_think_results(store, embedder, &response, data_dir, blob_store).await
}

/// Execute one think cycle with a type-erased LLM provider.
///
/// Same as [`execute`] but accepts `&dyn DynLlmProvider` for use from
/// the facade where the LLM is stored as a trait object.
pub async fn execute_dyn(
    store: &impl VectorStore,
    embedder: &dyn Embedder,
    llm: &dyn DynLlmProvider,
    data_dir: &Path,
    blob_store: Option<&crate::blob_store::BlobStore>,
    project: Option<&str>,
) -> Result<ThinkResult> {
    // 1. Reflect: compute identity snapshot
    let snapshot = identity::compute_identity(store, data_dir, project, None).await?;
    let priming = identity::generate_priming(&snapshot);

    if snapshot.core_entries.is_empty() {
        return Ok(ThinkResult::default());
    }

    // 2. Build prompt with full entry IDs for reference
    let prompt = build_prompt(&priming, &snapshot);

    // 3. Call LLM (via type-erased DynLlmProvider)
    let response = llm
        .complete(&[Message::system(THINK_SYSTEM_PROMPT), Message::user(prompt)])
        .await?;

    // 4. Parse and write back
    write_think_results(store, embedder, &response, data_dir, blob_store).await
}

// --- Helpers ---

/// Parse LLM response and write entries (narrative, consolidations, learnings) to the store.
async fn write_think_results(
    store: &impl VectorStore,
    embedder: &dyn Embedder,
    response: &str,
    data_dir: &Path,
    blob_store: Option<&crate::blob_store::BlobStore>,
) -> Result<ThinkResult> {
    let plan = parse_response(response)?;

    let mut entries_created = Vec::new();
    let mut consolidations_added = 0;
    let mut learnings_added = 0;
    let mut narrative_id = None;

    // Narrative → Meta entry
    if let Some(ref narrative_text) = plan.narrative {
        if !narrative_text.trim().is_empty() {
            let id = write_entry(
                store,
                embedder,
                narrative_text,
                EntryType::Meta,
                vec![],
                vec![],
                "[think:narrative]",
                blob_store,
            )
            .await?;
            entries_created.push(ThinkEntry {
                id: id.clone(),
                entry_type: EntryType::Meta,
                content_preview: preview(narrative_text, 100),
                perspectives: vec![],
            });
            narrative_id = Some(id);
        }
    }

    // Consolidations → Summary entries with summarized_by relations
    for consolidation in &plan.consolidations {
        if consolidation.content.trim().is_empty() || consolidation.entry_ids.is_empty() {
            continue;
        }

        let valid_ids = validate_entry_ids(store, &consolidation.entry_ids).await;
        if valid_ids.is_empty() {
            continue;
        }

        let relations: Vec<ChunkRelation> =
            valid_ids.iter().map(ChunkRelation::summarized_by).collect();

        let id = write_entry(
            store,
            embedder,
            &consolidation.content,
            EntryType::Summary,
            relations,
            consolidation.perspectives.clone(),
            "[think:consolidation]",
            blob_store,
        )
        .await?;

        entries_created.push(ThinkEntry {
            id,
            entry_type: EntryType::Summary,
            content_preview: preview(&consolidation.content, 100),
            perspectives: consolidation.perspectives.clone(),
        });
        consolidations_added += 1;

        // Demote summarized source entries to deep_only so the summary
        // replaces them in standard search, reducing redundancy.
        for source_id in &valid_ids {
            let _ = store
                .update_visibility(source_id, crate::chunk::visibility::DEEP_ONLY)
                .await;
        }
    }

    // Learnings → Meta entries in learnings perspective
    for learning in &plan.learnings {
        if learning.content.trim().is_empty() {
            continue;
        }

        let id = write_entry(
            store,
            embedder,
            &learning.content,
            EntryType::Meta,
            vec![],
            learning.perspectives.clone(),
            "[think:learning]",
            blob_store,
        )
        .await?;

        entries_created.push(ThinkEntry {
            id,
            entry_type: EntryType::Meta,
            content_preview: preview(&learning.content, 100),
            perspectives: learning.perspectives.clone(),
        });
        learnings_added += 1;
    }

    // Compact: apply aging
    let aging_config = crate::aging::AgingConfig::load(data_dir);
    let _ = crate::aging::apply_aging(store, &aging_config).await;

    Ok(ThinkResult {
        narrative_id,
        consolidations_added,
        learnings_added,
        entries_created,
    })
}

/// Build the user prompt from priming + entry ID reference.
fn build_prompt(priming: &str, snapshot: &IdentitySnapshot) -> String {
    let mut prompt = priming.to_string();

    // Add full entry IDs so the LLM can reference them in consolidations
    prompt.push_str("\n## Entry ID Reference\n\n");
    prompt.push_str("Use these full IDs when referencing entries in consolidations:\n\n");
    for entry in &snapshot.core_entries {
        let heading = entry.heading.as_deref().unwrap_or(UNTITLED);
        prompt.push_str(&format!("- `{}` — {}\n", entry.id, heading));
    }

    prompt
}

/// Write a single entry to the store with embedding.
#[allow(clippy::too_many_arguments)]
async fn write_entry(
    store: &impl VectorStore,
    embedder: &dyn Embedder,
    content: &str,
    entry_type: EntryType,
    relations: Vec<ChunkRelation>,
    perspectives: Vec<String>,
    source: &str,
    blob_store: Option<&crate::blob_store::BlobStore>,
) -> Result<String> {
    let embeddings = embedder.embed(&[content]).await?;
    let embedding = embeddings
        .into_iter()
        .next()
        .ok_or_else(|| crate::Error::embedding("Failed to generate embedding for think entry"))?;

    let mut chunk = HierarchicalChunk::new(
        content.to_string(),
        crate::chunk::ChunkLevel::CONTENT,
        None,
        String::new(),
        source.to_string(),
    )
    .with_entry_type(entry_type)
    .with_perspectives(perspectives);

    chunk.embedding = Some(embedding);
    chunk.relations = relations;

    // Persist to blob store when available
    if let Some(bs) = blob_store {
        let blob = crate::entry::StoredBlob::from_chunk_and_embedding(&chunk, embedder.name());
        bs.put(&blob)?;
    }

    let id = chunk.id.clone();
    store.insert_chunks(vec![chunk]).await?;
    Ok(id)
}

/// Validate that entry IDs actually exist in the store.
async fn validate_entry_ids(store: &impl VectorStore, ids: &[String]) -> Vec<String> {
    let mut valid = Vec::new();
    for id in ids {
        if store.get_by_id(id).await.ok().flatten().is_some() {
            valid.push(id.clone());
        }
    }
    valid
}

// --- Discover ---

/// A discovered pair: two entries that are semantically similar but have no explicit relation.
struct DiscoveredPair {
    entry_a: HierarchicalChunk,
    entry_b: HierarchicalChunk,
    similarity: f32,
}

/// Find entries that are semantically similar but share no explicit relation, returning
/// a formatted markdown report.
///
/// Algorithm:
/// 1. List up to `scan_limit * 2` entries as candidates.
/// 2. For each candidate that has an embedding, search for its top-5 ANN neighbors.
/// 3. For each (candidate, neighbor) pair, check whether a relation already exists in either direction.
/// 4. Deduplicate symmetric pairs using a sorted-ID set key.
/// 5. Sort by similarity descending and return up to `output_limit`.
pub async fn discover_unlinked_pairs(
    store: &Arc<StoreBackend>,
    output_limit: usize,
) -> Result<String> {
    const SCAN_LIMIT: usize = 100;
    const NEIGHBORS_PER_ENTRY: usize = 5;

    let candidates = store.list_entries(&[], None, None, SCAN_LIMIT).await?;

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
            .search(embedding, NEIGHBORS_PER_ENTRY + 1, None, &[])
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

    crate::chunk::sort_f32_desc(&mut pairs, |p| p.similarity);
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

        let preview_a = preview(heading_a, 100);
        let preview_b = preview(heading_b, 100);

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

/// Parse LLM response as JSON ThinkPlan.
fn parse_response(response: &str) -> Result<ThinkPlan> {
    let json_str = extract_json(response);
    serde_json::from_str(&json_str).map_err(|e| {
        crate::Error::parse(format!(
            "Failed to parse think response as JSON: {}. Response: {}",
            e,
            preview(response, 300)
        ))
    })
}

/// Extract JSON from a response that might be wrapped in markdown fences.
fn extract_json(s: &str) -> String {
    let trimmed = s.trim();

    // ```json ... ```
    if let Some(start) = trimmed.find("```json") {
        let after = &trimmed[start + 7..];
        if let Some(end) = after.find("```") {
            return after[..end].trim().to_string();
        }
    }

    // ``` ... ```
    if let Some(start) = trimmed.find("```") {
        let after = &trimmed[start + 3..];
        if let Some(end) = after.find("```") {
            return after[..end].trim().to_string();
        }
    }

    trimmed.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_bare() {
        let input = r#"{"narrative": "test"}"#;
        assert_eq!(extract_json(input), input);
    }

    #[test]
    fn test_extract_json_fenced() {
        let input = "```json\n{\"narrative\": \"test\"}\n```";
        assert_eq!(extract_json(input), r#"{"narrative": "test"}"#);
    }

    #[test]
    fn test_extract_json_bare_fenced() {
        let input = "```\n{\"narrative\": \"test\"}\n```";
        assert_eq!(extract_json(input), r#"{"narrative": "test"}"#);
    }

    #[test]
    fn test_extract_json_with_surrounding_text() {
        let input = "Here is the JSON:\n```json\n{\"narrative\": \"test\"}\n```\nDone.";
        assert_eq!(extract_json(input), r#"{"narrative": "test"}"#);
    }

    #[test]
    fn test_parse_response_valid() {
        let json = r#"{
            "narrative": "I am a knowledge base focused on Rust development.",
            "consolidations": [
                {
                    "content": "Summary of backend decisions",
                    "entry_ids": ["abc123"],
                    "perspectives": ["decisions"]
                }
            ],
            "learnings": [
                {
                    "content": "Pattern: prefer simple solutions",
                    "perspectives": ["learnings"]
                }
            ]
        }"#;
        let plan = parse_response(json).unwrap();
        assert_eq!(
            plan.narrative.unwrap(),
            "I am a knowledge base focused on Rust development."
        );
        assert_eq!(plan.consolidations.len(), 1);
        assert_eq!(plan.consolidations[0].entry_ids, vec!["abc123"]);
        assert_eq!(plan.learnings.len(), 1);
    }

    #[test]
    fn test_parse_response_minimal() {
        let json = r#"{"narrative": null, "consolidations": [], "learnings": []}"#;
        let plan = parse_response(json).unwrap();
        assert!(plan.narrative.is_none());
        assert!(plan.consolidations.is_empty());
        assert!(plan.learnings.is_empty());
    }

    #[test]
    fn test_parse_response_empty_object() {
        let json = r#"{}"#;
        let plan = parse_response(json).unwrap();
        assert!(plan.narrative.is_none());
        assert!(plan.consolidations.is_empty());
        assert!(plan.learnings.is_empty());
    }

    #[test]
    fn test_parse_response_learning_default_perspective() {
        let json = r#"{"learnings": [{"content": "something"}]}"#;
        let plan = parse_response(json).unwrap();
        assert_eq!(plan.learnings[0].perspectives, vec!["learnings"]);
    }

    #[test]
    fn test_build_prompt_includes_ids() {
        let snapshot = IdentitySnapshot {
            centroids: vec![],
            core_entries: vec![crate::identity::CoreEntry {
                id: "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890".to_string(),
                heading: Some("Test Entry".to_string()),
                content_preview: "test content".to_string(),
                salience: 0.5,
                perspectives: vec![],
            }],
            open_threads: vec![],
            recent_learnings: vec![],
            emergent_clusters: vec![],
            other_branches: vec![],
        };
        let prompt = build_prompt("# Briefing\n", &snapshot);
        assert!(prompt.contains("Entry ID Reference"));
        assert!(prompt.contains("abcdef1234567890"));
        assert!(prompt.contains("Test Entry"));
    }

    // ── execute() integration tests with mock LLM ─────────────────────────────

    use crate::embedder::Embedder;
    use crate::llm::{LlmProvider, Message};

    struct MockLlm {
        response: String,
    }

    impl LlmProvider for MockLlm {
        fn name(&self) -> &str {
            "mock-llm"
        }
        async fn complete(&self, _messages: &[Message]) -> crate::Result<String> {
            Ok(self.response.clone())
        }
    }

    struct FailingLlm;
    impl LlmProvider for FailingLlm {
        fn name(&self) -> &str {
            "failing-llm"
        }
        async fn complete(&self, _messages: &[Message]) -> crate::Result<String> {
            Err(crate::Error::llm("simulated LLM failure"))
        }
    }

    struct FixedEmbedder {
        dim: usize,
    }

    impl Embedder for FixedEmbedder {
        fn embed<'a>(
            &'a self,
            texts: &'a [&'a str],
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = crate::Result<Vec<Vec<f32>>>> + Send + 'a>,
        > {
            let result: Vec<Vec<f32>> = texts.iter().map(|_| vec![0.1f32; self.dim]).collect();
            Box::pin(async move { Ok(result) })
        }
        fn dimension(&self) -> usize {
            self.dim
        }
        fn name(&self) -> &str {
            "fixed-test"
        }
    }

    // Use 384-dim to match make_test_chunk which creates 384-dim embeddings.
    fn make_embedder() -> FixedEmbedder {
        FixedEmbedder { dim: 384 }
    }

    async fn open_test_store(dir: &std::path::Path) -> crate::store::StoreBackend {
        crate::store::StoreBackend::open(dir, 384, false)
            .await
            .expect("open store")
    }

    /// Create a test chunk that appears "hot" (access_total > 0 for get_hot_chunks).
    fn make_hot_test_chunk(id: &str, content: &str) -> crate::HierarchicalChunk {
        let mut chunk = crate::test_helpers::make_test_chunk(id, content);
        chunk.access_profile.record_access();
        chunk
    }

    #[tokio::test]
    async fn test_execute_empty_store_returns_empty_result() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = open_test_store(dir.path()).await;
        let embedder = make_embedder();
        let llm = MockLlm {
            response: r#"{"narrative": "I am.", "consolidations": [], "learnings": []}"#.to_owned(),
        };

        let result = execute(&store, &embedder, &llm, dir.path(), None, None)
            .await
            .expect("execute should succeed on empty store");

        // Empty store → no entries to think about
        assert!(result.narrative_id.is_none());
        assert_eq!(result.consolidations_added, 0);
        assert_eq!(result.learnings_added, 0);
        assert!(result.entries_created.is_empty());
    }

    #[tokio::test]
    async fn test_execute_with_entries_creates_narrative_and_learning() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = open_test_store(dir.path()).await;
        let embedder = make_embedder();

        let chunk = make_hot_test_chunk(
            "aabbccdd11223344aabbccdd11223344aabbccdd11223344aabbccdd11223344",
            "Important architectural decision about async Rust",
        );
        store.insert_chunks(vec![chunk]).await.unwrap();

        let llm = MockLlm {
            response: r#"{
                "narrative": "I am a Rust knowledge base.",
                "consolidations": [],
                "learnings": [{"content": "async Rust is great", "perspectives": ["learnings"]}]
            }"#
            .to_owned(),
        };

        let result = execute(&store, &embedder, &llm, dir.path(), None, None)
            .await
            .expect("execute");

        assert!(result.narrative_id.is_some(), "narrative should be created");
        assert_eq!(result.learnings_added, 1);
        assert_eq!(result.entries_created.len(), 2); // narrative + learning
    }

    #[tokio::test]
    async fn test_execute_consolidation_with_valid_ids() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = open_test_store(dir.path()).await;
        let embedder = make_embedder();

        let entry_id = "aabbccdd11223344aabbccdd11223344aabbccdd11223344aabbccdd11223344";
        let chunk = make_hot_test_chunk(entry_id, "Decision: use async Rust for concurrency");
        store.insert_chunks(vec![chunk]).await.unwrap();

        let llm = MockLlm {
            response: format!(
                r#"{{
                "narrative": null,
                "consolidations": [{{
                    "content": "Rust async decisions summary",
                    "entry_ids": ["{entry_id}"],
                    "perspectives": ["decisions"]
                }}],
                "learnings": []
            }}"#
            ),
        };

        let result = execute(&store, &embedder, &llm, dir.path(), None, None)
            .await
            .expect("execute");

        assert_eq!(result.consolidations_added, 1);
        let summary_entry = result
            .entries_created
            .iter()
            .find(|e| e.entry_type == crate::chunk::EntryType::Summary);
        assert!(summary_entry.is_some());
        assert!(summary_entry
            .unwrap()
            .perspectives
            .contains(&"decisions".to_string()));
    }

    #[tokio::test]
    async fn test_execute_consolidation_with_invalid_ids_skipped() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = open_test_store(dir.path()).await;
        let embedder = make_embedder();

        // Seed one real entry so snapshot is non-empty
        let chunk = make_hot_test_chunk(
            "aabbccdd11223344aabbccdd11223344aabbccdd11223344aabbccdd11223344",
            "Real entry content here",
        );
        store.insert_chunks(vec![chunk]).await.unwrap();

        let llm = MockLlm {
            response: r#"{
                "narrative": null,
                "consolidations": [{
                    "content": "Summary pointing at ghost",
                    "entry_ids": ["0000000000000000000000000000000000000000000000000000000000000000"],
                    "perspectives": []
                }],
                "learnings": []
            }"#
            .to_owned(),
        };

        let result = execute(&store, &embedder, &llm, dir.path(), None, None)
            .await
            .expect("execute");

        assert_eq!(result.consolidations_added, 0);
    }

    #[tokio::test]
    async fn test_execute_skips_empty_narrative() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = open_test_store(dir.path()).await;
        let embedder = make_embedder();

        let chunk = make_hot_test_chunk(
            "aabbccdd11223344aabbccdd11223344aabbccdd11223344aabbccdd11223344",
            "Some stored knowledge",
        );
        store.insert_chunks(vec![chunk]).await.unwrap();

        let llm = MockLlm {
            response: r#"{"narrative": "   ", "consolidations": [], "learnings": []}"#.to_owned(),
        };

        let result = execute(&store, &embedder, &llm, dir.path(), None, None)
            .await
            .expect("execute");

        assert!(result.narrative_id.is_none());
        assert!(result.entries_created.is_empty());
    }

    #[tokio::test]
    async fn test_execute_skips_empty_learning_content() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = open_test_store(dir.path()).await;
        let embedder = make_embedder();

        let chunk = make_hot_test_chunk(
            "aabbccdd11223344aabbccdd11223344aabbccdd11223344aabbccdd11223344",
            "Base content for snapshot",
        );
        store.insert_chunks(vec![chunk]).await.unwrap();

        let llm = MockLlm {
            response:
                r#"{"narrative": null, "consolidations": [], "learnings": [{"content": "  "}]}"#
                    .to_owned(),
        };

        let result = execute(&store, &embedder, &llm, dir.path(), None, None)
            .await
            .expect("execute");

        assert_eq!(result.learnings_added, 0);
    }

    #[tokio::test]
    async fn test_execute_skips_empty_consolidation_content() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = open_test_store(dir.path()).await;
        let embedder = make_embedder();

        let entry_id = "aabbccdd11223344aabbccdd11223344aabbccdd11223344aabbccdd11223344";
        let chunk = make_hot_test_chunk(entry_id, "Content to consolidate");
        store.insert_chunks(vec![chunk]).await.unwrap();

        let llm = MockLlm {
            response: format!(
                r#"{{"narrative": null, "consolidations": [{{"content": "  ", "entry_ids": ["{entry_id}"], "perspectives": []}}], "learnings": []}}"#
            ),
        };

        let result = execute(&store, &embedder, &llm, dir.path(), None, None)
            .await
            .expect("execute");

        assert_eq!(result.consolidations_added, 0);
    }

    #[tokio::test]
    async fn test_execute_llm_failure_returns_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = open_test_store(dir.path()).await;
        let embedder = make_embedder();

        let chunk = make_hot_test_chunk(
            "aabbccdd11223344aabbccdd11223344aabbccdd11223344aabbccdd11223344",
            "Content that triggers LLM call",
        );
        store.insert_chunks(vec![chunk]).await.unwrap();

        let result = execute(&store, &embedder, &FailingLlm, dir.path(), None, None).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("simulated LLM failure"));
    }

    #[tokio::test]
    async fn test_execute_invalid_llm_json_returns_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = open_test_store(dir.path()).await;
        let embedder = make_embedder();

        let chunk = make_hot_test_chunk(
            "aabbccdd11223344aabbccdd11223344aabbccdd11223344aabbccdd11223344",
            "Content that triggers LLM call",
        );
        store.insert_chunks(vec![chunk]).await.unwrap();

        let llm = MockLlm {
            response: "not json at all".to_owned(),
        };

        let result = execute(&store, &embedder, &llm, dir.path(), None, None).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Failed to parse think response"));
    }

    #[tokio::test]
    async fn test_validate_entry_ids_returns_only_existing() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = open_test_store(dir.path()).await;

        let real_id = "aabbccdd11223344aabbccdd11223344aabbccdd11223344aabbccdd11223344";
        let fake_id = "0000000000000000000000000000000000000000000000000000000000000001";

        let chunk = crate::test_helpers::make_test_chunk(real_id, "Existing entry");
        store.insert_chunks(vec![chunk]).await.unwrap();

        let ids = vec![real_id.to_owned(), fake_id.to_owned()];
        let valid = validate_entry_ids(&store, &ids).await;

        assert_eq!(valid, vec![real_id.to_owned()]);
    }

    #[tokio::test]
    async fn test_validate_entry_ids_all_missing() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = open_test_store(dir.path()).await;

        let ids = vec![
            "0000000000000000000000000000000000000000000000000000000000000001".to_owned(),
            "0000000000000000000000000000000000000000000000000000000000000002".to_owned(),
        ];

        let valid = validate_entry_ids(&store, &ids).await;
        assert!(valid.is_empty());
    }

    #[tokio::test]
    async fn test_validate_entry_ids_empty_input() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = open_test_store(dir.path()).await;
        let valid = validate_entry_ids(&store, &[]).await;
        assert!(valid.is_empty());
    }

    // ── build_prompt with untitled entry ─────────────────────────────────────

    // ── prepare() tests ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_prepare_empty_store_returns_none() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = open_test_store(dir.path()).await;
        let result = prepare(&store, dir.path()).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_prepare_with_entries_returns_data() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = open_test_store(dir.path()).await;

        let chunk = make_hot_test_chunk(
            "aabbccdd11223344aabbccdd11223344aabbccdd11223344aabbccdd11223344",
            "Architecture decision about async",
        );
        store.insert_chunks(vec![chunk]).await.unwrap();

        let result = prepare(&store, dir.path()).await.unwrap();
        assert!(result.is_some());

        let prep = result.unwrap();
        assert_eq!(prep.system_prompt, THINK_SYSTEM_PROMPT);
        assert!(!prep.user_prompt.is_empty());
        assert!(!prep.entry_ids.is_empty());
        assert_eq!(
            prep.entry_ids[0].0,
            "aabbccdd11223344aabbccdd11223344aabbccdd11223344aabbccdd11223344"
        );
    }

    #[tokio::test]
    async fn test_prepare_entry_ids_include_headings() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = open_test_store(dir.path()).await;

        let mut chunk = make_hot_test_chunk(
            "aabbccdd11223344aabbccdd11223344aabbccdd11223344aabbccdd11223344",
            "Content text here",
        );
        chunk.heading = Some("My Heading".to_string());
        store.insert_chunks(vec![chunk]).await.unwrap();

        let prep = prepare(&store, dir.path()).await.unwrap().unwrap();
        assert_eq!(prep.entry_ids[0].1, "My Heading");
    }

    #[tokio::test]
    async fn test_prepare_untitled_entry_gets_placeholder() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = open_test_store(dir.path()).await;

        let mut chunk = make_hot_test_chunk(
            "aabbccdd11223344aabbccdd11223344aabbccdd11223344aabbccdd11223344",
            "Content without heading",
        );
        chunk.heading = None;
        store.insert_chunks(vec![chunk]).await.unwrap();

        let prep = prepare(&store, dir.path()).await.unwrap().unwrap();
        assert_eq!(prep.entry_ids[0].1, "(untitled)");
    }

    // ── build_prompt tests ──────────────────────────────────────────────

    #[test]
    fn test_build_prompt_untitled_entry() {
        let snapshot = IdentitySnapshot {
            centroids: vec![],
            core_entries: vec![crate::identity::CoreEntry {
                id: "aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff66660000111122223333".to_string(),
                heading: None,
                content_preview: "mystery content".to_string(),
                salience: 0.3,
                perspectives: vec![],
            }],
            open_threads: vec![],
            recent_learnings: vec![],
            emergent_clusters: vec![],
            other_branches: vec![],
        };
        let prompt = build_prompt("Prefix\n", &snapshot);
        assert!(prompt.contains("(untitled)"));
        assert!(prompt.contains("aaaa1111bbbb2222"));
    }
}
