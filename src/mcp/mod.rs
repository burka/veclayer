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
a relevance tier. Use `since`/`until` for temporal filtering. Results come with access profiles \
showing how often each piece was accessed. Use `deep: true` to include archived knowledge. \
Use `recency` to boost recently accessed memories.

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

### share — Here, this is for you. (Preview)
Generates a scoped share-token payload describing what knowledge to share and with what \
permissions. This is a preview of the upcoming UCAN-based sharing system — tokens are not \
yet cryptographically signed.

## When to Think

Run `think` when you have time: start of session, end of session, between tasks. Review what's \
hot, what's stale, what needs promoting or archiving. Write summaries of related chunks using \
`store`. Mark outdated knowledge with `think(action='relate', kind='superseded_by')`.

## Summarization Pattern

1. `recall` a topic area
2. `focus` on interesting nodes to read children
3. Synthesize a summary
4. `store` it with `parent_id` to place in hierarchy
5. `think(action='relate', kind='summarized_by')` to link children to the summary

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

You are the curator of your own memory. Use these tools to build a knowledge base that reflects \
what matters to you.";
