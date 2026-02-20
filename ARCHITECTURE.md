# VecLayer Architecture

VecLayer is a hierarchical vector database with memory functions. Its goal: cross-session identity for AI agents through structured, aging, self-describing knowledge.

## Principles

- **Summaries are the hierarchy** — not metadata, but condensed content with its own embedding.
- **Data describes itself** — visibility, relations, and access profiles are fields on every entry, not external metadata.
- **Find, then navigate** — vector search finds; relations enable 1–2 hops. No graph traversal.
- **Pragmatic, not perfect** — established patterns (RRD, hierarchical indexing, event sourcing) over reinvention.

## Data Model

### Entry (HierarchicalChunk)

The central data object. Every entry carries:

```rust
HierarchicalChunk {
    // Identity
    id: String,               // SHA-256 of content (idempotent: same content = same ID)
    content: String,
    embedding: Option<Vec<f32>>,

    // Hierarchy
    level: ChunkLevel,        // H1–H6 or Content (7)
    parent_id: Option<String>,
    path: String,             // "Chapter 1 > Section 1.1 > Details"
    source_file: String,
    heading: Option<String>,
    start_offset: usize,
    end_offset: usize,

    // Clustering (RAPTOR-style soft membership)
    cluster_memberships: Vec<ClusterMembership>,  // { cluster_id, probability }

    // Type and summarization
    entry_type: EntryType,    // raw | summary | meta | impression
    summarizes: Vec<String>,  // IDs of entries this summarizes

    // Identity and memory (open types — extensible without code changes)
    visibility: String,               // "normal" default; open string
    perspectives: Vec<String>,        // facet labels, e.g. ["decisions", "learnings"]
    relations: Vec<ChunkRelation>,    // { kind: String, target_id: String }
    access_profile: AccessProfile,    // RRD-style time-bucket tracking
    expires_at: Option<i64>,          // Unix epoch; requires visibility = "expiring"
}
```

### Entry Types

| Type | Description |
|------|-------------|
| `raw` | Original data, unmodified (ingested text, file content) |
| `summary` | Generated summary of child entries |
| `meta` | Reflection, assessment, evaluation |
| `impression` | Spontaneous observation, quick note |

### Visibility

Open string field — not an enum. Well-known values are constants; custom values work without code changes.

| Value | Description |
|-------|-------------|
| `"always"` | Never degraded; always visible in standard search |
| `"normal"` | Standard visibility; subject to aging degradation |
| `"deep_only"` | Excluded from standard search; included in deep search |
| `"expiring"` | Excluded once `expires_at` is in the past |
| `"seasonal"` | Included in standard search (e.g. recurring reports) |
| *(custom)* | Any other string; excluded from standard search by default |

Standard search includes: `always`, `normal`, `seasonal`, and non-expired `expiring`.
Deep search (`--deep`) includes everything.

### Relations

Open string `kind` field. Directed, lightweight links. Relations are NOT the primary search path — you find an entry via vector search, then optionally navigate 1–2 hops.

| Kind | Semantics |
|------|-----------|
| `"superseded_by"` | This fact has been replaced by a newer entry |
| `"summarized_by"` | This entry has been condensed into the target |
| `"related_to"` | Loose thematic connection |
| `"derived_from"` | Originated from this discussion or source |
| *(custom)* | Any domain-specific relation kind |

## Perspectives

Seven built-in perspectives ship with VecLayer. Custom perspectives can be added at runtime via the `p` command or MCP. Perspectives are stored in `perspectives.json` in the data directory.

| ID | Hint |
|----|------|
| `intentions` | Goals, motivations, plans |
| `people` | Persons, relationships, roles, organizations |
| `temporal` | Timelines, milestones, chronology |
| `knowledge` | Durable expertise, definitions, concepts, references |
| `decisions` | Decisions, trade-offs, alternatives |
| `learnings` | Insights, mistakes, lessons learned, aha-moments |
| `session` | Work sessions, context summaries, handoffs |

An entry can belong to multiple perspectives. Search can be filtered by perspective, and identity computes per-perspective centroids from the active set.

## Search Strategy

### Standard Search

1. Embed the query string via `Embedder::embed`.
2. Nearest-neighbor search over all stored entries (optionally filtered by perspective via `search_by_perspective`).
3. Visibility filter: exclude entries where `is_visible_standard()` returns false (deep_only, expired, unknown custom values).
4. For each result: build the hierarchy path (root → match) and retrieve relevant children scored by cosine similarity.
5. Compute final score via `blend_score`.
6. Persist updated access profiles (best-effort, does not fail the search).

### Blend Score Formula

```
final_score = vector_score * (1 - alpha) + relevancy_signal * alpha

relevancy_signal = recency * (1 - sw) + salience * sw
```

Where:
- `alpha` (`recency_alpha`) — blend factor. Default `0.15`; raised to `0.30` when a recency window is explicitly requested.
- `sw` (`salience_weight`) — weight of salience within the relevancy signal. Default `0.30`.
- `recency` — `AccessProfile::relevancy_score(recency_window)`, a tanh-saturated weighted sum of time-bucket counts.
- `salience` — composite salience score (see Salience section).

Setting `alpha = 0.0` yields pure vector similarity.

### Deep Search

Identical to standard search but the visibility filter is skipped. All entries, including `deep_only`, archived, and unknown custom visibilities, are candidates.

### Browse Mode (no query)

`VectorStore::list_entries` returns entries filtered by optional perspective and time range, sorted newest-first. No embedding or nearest-neighbor involved.

## Salience

Salience measures significance, not frequency. It is a composite score in `[0.0, 1.0]`:

```
composite = interaction * 0.50
          + perspective * 0.25
          + revision    * 0.25
```

Where:
- **Interaction density** (weight 0.50) — `AccessProfile::relevancy_score(None)`, i.e. the recency-weighted access signal.
- **Perspective spread** (weight 0.25) — `perspectives.len() / 8.0`, capped at 1.0. Entries referenced from more perspectives are more salient.
- **Revision activity** (weight 0.25) — `tanh(relations.len() / 5.0)`. Saturates around 5 relations to avoid unbounded growth.

Salience protects entries from aging degradation: entries with `composite >= salience_protection` (default `0.15`) are skipped by the aging pass even when stale.

## Memory Aging

Access tracking uses RRD-style fixed buckets per entry. Finer buckets roll into coarser ones as time passes. Memory per entry is constant regardless of access count.

**AccessProfile buckets:** `hour`, `day`, `week`, `month`, `year`, `total` (plus `created_at` and `last_rolled`).

**Visibility degradation** is applied by `apply_aging` (triggered at the end of the think cycle or manually via `compact`):

- Entries not accessed within `degrade_after_days` (default 30) whose visibility is in `degrade_from` (default `["normal"]`) are demoted to `degrade_to` (default `"deep_only"`).
- Entries with salience above the `salience_protection` threshold are exempt.
- The aging configuration is stored in `aging_config.json` in the data directory and is agent-configurable.

The effective cascade is: `normal` → `deep_only` → (manual) `archived` / deletion.

## Identity

Identity is computed mechanically from stored data — no LLM required. `compute_identity` produces an `IdentitySnapshot`:

- **Centroids** — salience-weighted embedding averages per perspective. Higher-salience entries pull the centroid more strongly. Used to understand where each perspective "points" in semantic space.
- **Core entries** — top-15 entries by composite salience. Form the "core knowledge" of the agent.
- **Open threads** — entries that are superseded but still `normal` visibility, or entries with 3+ relations suggesting active deliberation.
- **Recent learnings** — entries in the `learnings` perspective, up to 10.

On MCP connect, `generate_priming` assembles these into a startup briefing: core knowledge, open threads, recent learnings, and perspective coverage statistics.

## Think Cycle (optional, requires LLM)

The think cycle is the only module that requires an LLM. All other functionality is purely mechanical.

**Phases:** reflect → LLM → add → compact.

1. **Reflect** — `compute_identity` gathers the current memory state and `generate_priming` formats it as a human-readable briefing.
2. **LLM** — the briefing plus a full entry-ID reference is sent to the configured `LlmProvider`. The model returns JSON with: a `narrative` (2–3 sentence first-person summary), `consolidations` (summaries over existing entries), and `learnings` (meta-observations).
3. **Add** — the returned narrative becomes a `meta` entry; consolidations become `summary` entries with `summarized_by` relations; learnings become `meta` entries in the `learnings` perspective. All are embedded and written to the store.
4. **Compact** — `apply_aging` runs to degrade stale entries.

## VecLayer vs. LLM Boundary

| VecLayer does (mechanically) | LLM does |
|------------------------------|----------|
| Embed text, store vectors | Decide what to store and when |
| Find similar entries via ANN | Interpret search results |
| Compute salience, age entries | Decide what is worth keeping |
| Build identity snapshot | Reflect on identity |
| Detect open threads | Resolve open threads |
| Compute perspective centroids | Understand what they mean |
| Write think-cycle entries | Generate the content for them |

## System Components (Traits)

- **`DocumentParser`** — `MarkdownParser` (pulldown-cmark, heading hierarchy extraction). Extensible for PDF, HTML, code (tree-sitter).
- **`Embedder`** — `FastEmbedder` (ONNX runtime, CPU, `BAAI/bge-small-en-v1.5`, 384 dimensions). Extensible for Ollama (GPU) and OpenAI-compatible APIs. Interface: `embed(&[&str]) -> Vec<Vec<f32>>`.
- **`VectorStore`** — `LanceStore` (serverless, file-based, no external service). Extensible for Turso/Limbo and PostgreSQL + pgvector. Interface covers insert, ANN search, perspective search, children, access profile updates, visibility updates, and stale-chunk queries.
- **`Summarizer`** — `OllamaSummarizer` (local LLM via REST). Behind the `llm` feature flag.
- **`LlmProvider`** — OpenAI-compatible interface used by the think cycle. Implementations: `OllamaProvider`, `OpenAiProvider`. Behind the `llm` feature flag.

## Configuration

Resolution order: ENV > TOML file > defaults. TOML file is discovered from `$VECLAYER_CONFIG`, then `<data_dir>/veclayer.toml`, then `./veclayer.toml`.

| Variable | Default | Description |
|----------|---------|-------------|
| `VECLAYER_DATA_DIR` | `./veclayer-data` | Data directory (LanceDB files, config JSON) |
| `VECLAYER_CONFIG` | *(auto-discovered)* | Explicit path to `veclayer.toml` |
| `VECLAYER_EMBEDDER` | `fastembed` | Embedder type (`fastembed`, `ollama`) |
| `VECLAYER_FASTEMBED_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model name |
| `VECLAYER_OLLAMA_MODEL` | `llama3.2` | Ollama embedding model |
| `VECLAYER_OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `VECLAYER_LLM_PROVIDER` | `ollama` | LLM provider (`ollama`, `openai`) |
| `VECLAYER_LLM_MODEL` | `llama3.2` | LLM model for think cycle |
| `VECLAYER_LLM_BASE_URL` | `http://localhost:11434` | LLM base URL |
| `VECLAYER_LLM_API_KEY` | *(none)* | API key for OpenAI-compatible providers |
| `VECLAYER_HOST` | `127.0.0.1` | Server bind host |
| `VECLAYER_PORT` | `8080` | Server port |
| `VECLAYER_SEARCH_TOP_K` | `5` | Top-level results per search |
| `VECLAYER_SEARCH_CHILDREN_K` | `3` | Children per result |
| `VECLAYER_READ_ONLY` | `false` | Disallow writes |

## Deployment

Single binary plus a data directory. No external services required for basic usage. The MCP server runs over stdio (Claude Desktop compatible) or HTTP (axum). The `llm` feature flag gates all LLM-dependent code; builds without it have no LLM dependency.
