# Graph-Aware Retrieval

Status: proposal
Phase: enhancement (post-5.5)
Source: Analysis of vector DB vs Graph RAG trade-offs
  (machinelearningmastery.com/vector-databases-vs-graph-rag-for-agent-memory-when-to-use-which)

## Problem

VecLayer stores relations (`related_to`, `superseded_by`, `derived_from`, etc.)
but does not traverse them during search. The relation graph exists as data but
is invisible to retrieval. This means multi-hop connections — the core strength
of Graph RAG systems — are lost despite the data being present.

## Proposal 1: Depth Traversal on Recall

Add an optional `--depth N` parameter to `recall` (CLI) and the `recall` MCP tool.

**Behavior:**

1. Standard vector search returns top-K results (unchanged)
2. For each result, follow outgoing relations up to N hops
3. Connected entries appear in results with decaying score:
   `connected_score = original_score * decay^hop_distance`
   (e.g. decay = 0.6 → hop 1 = 60%, hop 2 = 36%)
4. Deduplicate: if an entry appears via vector search AND traversal, keep the
   higher score
5. Default depth = 0 (current behavior, no traversal)

**Why this works for VecLayer:**

- No separate graph store needed — relations already exist on entries
- Opt-in via parameter — zero cost when not used
- Leverages existing `ChunkRelation` data that `think discover` and
  `think relate` already produce
- Makes the relation graph functionally useful, not just decorative

**Example:**

```
recall "authentication design" --depth 2
```

Finds auth entry via embedding → follows `related_to` → crypto entry →
follows `derived_from` → UCAN entry. All three appear in results.

**Open questions:**

- Should relation kind affect decay? (`superseded_by` might warrant lower
  decay than `related_to` since it's a stronger signal)
- Should `summarized_by` relations be traversed in reverse? (from detail
  to summary, not just summary to detail)
- Maximum practical depth — likely 2-3, beyond that noise increases

## Proposal 2: Connectivity Term in Salience

Add incoming relation count and neighbor quality as a signal in the salience
composite score.

**Current salience formula:**

```
composite = interaction * 0.50 + perspective * 0.25 + revision * 0.25
```

The `revision` component counts outgoing relations (`tanh(relations.len() / 5.0)`).
This misses **incoming** relations — entries that others point to.

**Proposed addition:**

```
connectivity = tanh(incoming_count / 4.0) * avg_neighbor_salience
```

Fold this into the existing `revision` component or add as a fourth term
with rebalanced weights:

```
composite = interaction * 0.45
          + perspective * 0.20
          + revision    * 0.15
          + connectivity * 0.20
```

**Why:**

- An entry referenced by many high-salience entries is structurally important
  (PageRank intuition applied to memory entries)
- Currently, a well-connected hub entry and an isolated entry with the same
  access pattern get identical salience — that's wrong
- This makes `think relate` and `think discover` more valuable: creating
  relations now improves retrieval quality, not just documentation

**Implementation concern:**

Computing incoming relations requires a reverse index or a scan. Options:
- Maintain a `HashMap<id, Vec<source_id>>` reverse index in memory (simple,
  fits in RAM for realistic store sizes)
- Compute lazily during `think` cycles and cache on the entry
- Store as a denormalized field updated on relation creation

**Recommendation:** Start with lazy computation during `think` cycles. The
`think` command already scans all entries for aging and discovery — adding
incoming relation counting is marginal cost. Cache the result as a field
on the entry to avoid repeated computation.

## What We Explicitly Do NOT Propose

- **Separate graph database** — Relations on entries in LanceDB can serve
  graph queries without doubling infrastructure complexity
- **Knowledge graph construction** — VecLayer entries are self-describing;
  no schema or ontology needed
- **Query routing heuristics** — Language-dependent pattern matching for
  deciding "vector vs graph" is fragile and arbitrary. Better to let depth
  be an explicit parameter the caller controls.

---

*— Claude Code - Opus 4.6*
