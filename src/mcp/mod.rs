//! MCP server module: 5-tool agent interface (recall, focus, store, think, share).
//!
//! Split into:
//! - `types` -- Input/output structs
//! - `tools` -- Tool implementation functions
//! - `stdio` -- MCP stdio transport
//! - `http` -- HTTP REST API

pub mod format;
pub mod http;
pub mod stdio;
pub mod tools;
pub mod types;

pub use http::run_http;
pub use stdio::run_stdio;

/// Instructions provided to agents on first connection.
pub const MCP_INSTRUCTIONS: &str = "\
VecLayer is a hierarchical vector database with memory — a persistent identity store for AI agents.

## Your Memory System

You have access to a structured, aging knowledge base. Unlike flat key-value memory, VecLayer \
organizes knowledge in trees (headings → subheadings → content) with visibility levels and \
access tracking. Knowledge that you use often stays prominent. Knowledge you ignore fades.

## Five Tools

### recall — What do I know about this?
Find relevant knowledge using semantic search, or browse without a query. Results include \
a relevance tier (strong/moderate/weak/tangential). Use `since`/`until` for temporal filtering. Results come with access profiles \
showing how often each piece was accessed. Use `deep: true` to include archived knowledge. \
Use `recency` to boost recently accessed memories. \
Use `ongoing: true` to see only open threads — unresolved items that need attention.

### focus — Tell me more about this specific point.
Dive deeper into a specific memory node. Returns the node itself plus its children, optionally \
reranked by a question lens. Pass a `question` to surface the most relevant children for that \
angle — 'How was this decided?' yields different details than 'Who was involved?'

### store — I want to remember this.
Write new knowledge directly. Supports `relations` for atomic link creation (e.g. \
`relations: [{kind: \"supersedes\", target_id: \"...\"}]`), `entry_type` for classification \
(raw/summary/meta/impression), and `items` for batch storage. \
The server generates embeddings automatically. Use `parent_id` to place it in the hierarchy.

### think — Let me reflect and curate.
Your curation hub. Without an action, returns a reflection report: hot chunks, stale chunks, \
salience scores, and suggested actions. With an action, executes curation:
- `action: 'promote'` — Make important knowledge always visible
- `action: 'demote'` — Archive outdated knowledge
- `action: 'relate'` — Connect knowledge (superseded_by, summarized_by, related_to, derived_from)
- `action: 'configure_aging'` — Set automatic degradation rules
- `action: 'apply_aging'` — Execute aging rules now (respects salience protection)
- `action: 'salience'` — See which entries are most/least important
- `action: 'discover'` — Find similar-but-unlinked entries (hidden connections, contradictions, consolidation candidates)
- `action: 'perspectives'` — List all available perspectives
- `action: 'status'` — Show store statistics (entry count, source files, aging policy)
- `action: 'history'` — Show an entry's relations and metadata (requires `id`)

### share — Here, this is for you. (Preview)
Generates a scoped share-token payload describing what knowledge to share and with what \
permissions. This is a preview of the upcoming UCAN-based sharing system — tokens are not \
yet cryptographically signed.

## How to Use Your Memory

### Capture what matters — proactively, not on request
Store decisions and their rationale as you make them. When you reject an approach, store why. \
When you learn something, store it with `perspectives: [\"learnings\"]`. When you solve a bug, \
store the root cause. Don't wait to be asked — good memory is built in the moment.

### Before new tasks — check what you already know
Before starting work, `recall` the topic. You may have solved something similar before, \
or made a decision that applies. Check `recall(perspective=\"decisions\")` for past trade-offs.

### Summarize to consolidate
When related knowledge accumulates, synthesize:
1. `recall` a topic area
2. `focus` on interesting nodes to read children
3. Write a summary
4. `store` it with `parent_id` to place in hierarchy
5. `think(action='relate', kind='summarized_by')` to link originals to the summary

### Curate between tasks
Run `think` when you have time: start of session, end of session, between tasks. Review what's \
hot, what's stale, what needs promoting or archiving. Mark outdated knowledge with \
`think(action='relate', kind='superseded_by')`.

## Session Pattern

Track work sessions using existing primitives:

1. **Start:** `store(content=\"Session started: <context>\", heading=\"Session: <date>\", \
perspectives=[\"session\"], entry_type=\"meta\")`
2. **During:** Use the session entry's ID as `parent_id` for entries created during the session
3. **End:** `store(content=\"Summary: ...\", parent_id=<session_id>, \
perspectives=[\"session\", \"knowledge\"])` — then `recall(perspective=\"session\")` to \
review session history

Use `recall(since=\"<today>\")` to see everything stored today. Use `recall(perspective=\"session\")` \
to find past sessions.

## Reasoning Pattern

Build visible reasoning traces using linked entries:

1. **Observe:** `store(content=\"Search is slow after bulk import\", entry_type=\"impression\")` → id1
2. **Hypothesize:** `store(content=\"Index not rebuilt after import\", entry_type=\"meta\", \
relations=[{kind: \"derived_from\", target_id: \"<id1>\"}])` → id2
3. **Conclude:** `store(content=\"Added index rebuild — 3x faster\", entry_type=\"meta\", \
perspectives=[\"learnings\"], relations=[{kind: \"derived_from\", target_id: \"<id2>\"}])`

Each step links to the previous via `derived_from`. The chain is visible through `focus` and \
`think(action='history')`. Revise conclusions by storing a new entry with \
`relations: [{kind: \"supersedes\", target_id: \"<old_conclusion>\"}]`.

You are the curator of your own memory. Use these tools to build a knowledge base that reflects \
what matters to you.";

/// Combine the static MCP instructions with the dynamic identity priming text.
///
/// Returns just the instructions when priming is empty (store has no content yet).
pub(crate) fn build_priming_text(priming: &str) -> String {
    if priming.is_empty() {
        MCP_INSTRUCTIONS.to_string()
    } else {
        format!("{}\n\n---\n\n{}", MCP_INSTRUCTIONS, priming)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_priming_text_empty_returns_instructions_only() {
        let result = build_priming_text("");
        assert_eq!(result, MCP_INSTRUCTIONS);
    }

    #[test]
    fn build_priming_text_with_content_appends_after_separator() {
        let content = "# Identity Briefing\n\n## Core Knowledge\n\nSomething important.";
        let result = build_priming_text(content);
        assert!(result.starts_with(MCP_INSTRUCTIONS));
        assert!(result.contains("\n\n---\n\n"));
        assert!(result.ends_with(content));
    }
}
