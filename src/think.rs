//! Think: the sleep cycle orchestrator.
//!
//! The think cycle is: reflect → LLM → add → compact.
//! VecLayer gathers context (mechanical), the LLM generates consolidations
//! and learnings, VecLayer writes them back and cleans up.
//!
//! Without an LLM, everything else works. Think is the only module that
//! requires an LLM — and it's optional.

use std::path::Path;

use crate::chunk::{ChunkRelation, EntryType, HierarchicalChunk};
use crate::embedder::FastEmbedder;
use crate::identity::{self, IdentitySnapshot};
use crate::llm::{LlmProvider, Message};
use crate::store::LanceStore;
use crate::{Embedder, Result, VectorStore};

/// Result of a think cycle.
#[derive(Debug)]
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

// --- System prompt ---

const THINK_SYSTEM_PROMPT: &str = r#"You are reflecting on a knowledge base to consolidate and distill learnings.

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
- Only consolidate entries that genuinely belong together
- Learnings should be genuine meta-observations, not repetitions of existing entries
- If nothing needs consolidation, return empty arrays
- Keep each consolidation to 1-3 concise sentences
- perspectives must use existing perspective IDs from the briefing"#;

// --- Main entry point ---

/// Execute one think cycle: reflect → LLM → add → compact.
pub async fn execute<L: LlmProvider>(
    store: &LanceStore,
    embedder: &FastEmbedder,
    llm: &L,
    data_dir: &Path,
) -> Result<ThinkResult> {
    // 1. Reflect: compute identity snapshot
    let snapshot = identity::compute_identity(store, data_dir).await?;
    let priming = identity::generate_priming(&snapshot);

    // Nothing to think about if memory is empty
    if snapshot.core_entries.is_empty() {
        return Ok(ThinkResult {
            narrative_id: None,
            consolidations_added: 0,
            learnings_added: 0,
            entries_created: vec![],
        });
    }

    // 2. Build prompt with full entry IDs for reference
    let prompt = build_prompt(&priming, &snapshot);

    // 3. Call LLM
    let response = llm
        .complete(&[Message::system(THINK_SYSTEM_PROMPT), Message::user(prompt)])
        .await?;

    // 4. Parse response
    let plan = parse_response(&response)?;

    // 5. Write back: create entries
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
            )
            .await?;
            entries_created.push(ThinkEntry {
                id: id.clone(),
                entry_type: EntryType::Meta,
                content_preview: truncate(narrative_text, 100),
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

        // Validate that referenced entries exist
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
        )
        .await?;

        entries_created.push(ThinkEntry {
            id,
            entry_type: EntryType::Summary,
            content_preview: truncate(&consolidation.content, 100),
            perspectives: consolidation.perspectives.clone(),
        });
        consolidations_added += 1;
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
        )
        .await?;

        entries_created.push(ThinkEntry {
            id,
            entry_type: EntryType::Meta,
            content_preview: truncate(&learning.content, 100),
            perspectives: learning.perspectives.clone(),
        });
        learnings_added += 1;
    }

    // 6. Compact: apply aging
    let aging_config = crate::aging::AgingConfig::load(data_dir);
    let _ = crate::aging::apply_aging(store, &aging_config).await;

    Ok(ThinkResult {
        narrative_id,
        consolidations_added,
        learnings_added,
        entries_created,
    })
}

// --- Helpers ---

/// Build the user prompt from priming + entry ID reference.
fn build_prompt(priming: &str, snapshot: &IdentitySnapshot) -> String {
    let mut prompt = priming.to_string();

    // Add full entry IDs so the LLM can reference them in consolidations
    prompt.push_str("\n## Entry ID Reference\n\n");
    prompt.push_str("Use these full IDs when referencing entries in consolidations:\n\n");
    for entry in &snapshot.core_entries {
        let heading = entry.heading.as_deref().unwrap_or("(untitled)");
        prompt.push_str(&format!("- `{}` — {}\n", entry.id, heading));
    }

    prompt
}

/// Write a single entry to the store with embedding.
async fn write_entry(
    store: &LanceStore,
    embedder: &FastEmbedder,
    content: &str,
    entry_type: EntryType,
    relations: Vec<ChunkRelation>,
    perspectives: Vec<String>,
    source: &str,
) -> Result<String> {
    let embeddings = embedder.embed(&[content])?;
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

    let id = chunk.id.clone();
    store.insert_chunks(vec![chunk]).await?;
    Ok(id)
}

/// Validate that entry IDs actually exist in the store.
async fn validate_entry_ids(store: &LanceStore, ids: &[String]) -> Vec<String> {
    let mut valid = Vec::new();
    for id in ids {
        if store.get_by_id(id).await.ok().flatten().is_some() {
            valid.push(id.clone());
        }
    }
    valid
}

/// Parse LLM response as JSON ThinkPlan.
fn parse_response(response: &str) -> Result<ThinkPlan> {
    let json_str = extract_json(response);
    serde_json::from_str(&json_str).map_err(|e| {
        crate::Error::llm(format!(
            "Failed to parse think response as JSON: {}. Response: {}",
            e,
            truncate(response, 300)
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

fn truncate(s: &str, max: usize) -> String {
    let clean = s.replace('\n', " ");
    if clean.len() <= max {
        clean
    } else {
        format!("{}...", &clean[..max])
    }
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
        };
        let prompt = build_prompt("# Briefing\n", &snapshot);
        assert!(prompt.contains("Entry ID Reference"));
        assert!(prompt.contains("abcdef1234567890"));
        assert!(prompt.contains("Test Entry"));
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 100), "short");
        assert_eq!(truncate("hello\nworld", 100), "hello world");

        let long = "a".repeat(200);
        let result = truncate(&long, 100);
        assert!(result.ends_with("..."));
        assert_eq!(result.len(), 103);
    }
}
