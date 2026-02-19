# VecLayer Roadmap (Vision Alignment)

## Current State Summary

VecLayer has a strong prototype foundation:
- Hierarchical chunking + summaries with real BGE embeddings (fastembed)
- Vector search with recency boost + visibility filters
- RRD-style access tracking (hour/day/week/month/year buckets)
- Agent-driven 5-tool MCP interface: `recall`, `focus`, `store`, `think`, `share`
- `think` as curation hub (reflect + promote/demote/relate/aging)
- `focus` with semantic reranking via embedding cosine similarity
- Aging rules: agent-configurable, apply on demand
- MCP stdio + HTTP REST API

## Alignment Scorecard

| Domain | Vision Target | Current Status | Fit |
|---|---|---|---|
| Tool model | `recall`, `focus`, `store`, `think`, `share` | Fully implemented as 5 MCP tools | **High** |
| Access profile | Constant-size fixed buckets | hour/day/week/month/year/total RRD | **High** |
| Aging / degradation | Agent-configurable visibility rules | configure_aging + apply_aging via think | **High** |
| Reflection loop (`think`) | Reflect + curate + consolidate | reflect + promote/demote/relate/aging | **Medium-High** |
| Agent priming at connect | Identity narrative + open threads + meta-learnings | Static MCP instructions, no per-agent generation | **Low** |
| Data primitives | Immutable leaves + mutable nodes | Single mutable chunk type with hierarchy fields | **Low** |
| Multi-tree memory | Projects/people/time trees reusing same leaves | Single hierarchy from source headings | **Low** |
| Supersession semantics | First-class temporal truth chains | Supported as relation kind, not policy-driven in recall | **Medium** |
| Salience engine | Computed from density/revisions/spread/proximity | Access-based relevance + visibility, no full salience | **Medium-Low** |
| Sharing via UCAN | Delegatable, attenuated, expiring capability tokens | Preview token payload only, no crypto | **Low** |
| Multi-agent isolation | Isolated/shared leaves/shared trees | Not implemented | **None** |
| Storage backends | Turso/SQLite + Postgres/pgvector | LanceDB only | **Medium-Low** |

## Priority Phases

### Phase 1 — Agent Experience (done)
- [x] 5-tool MCP surface: recall, focus, store, think, share
- [x] think as curation hub with action parameter
- [x] focus with semantic reranking (embedding-based, not string match)
- [x] MCP instructions with reflection pattern + summarization pattern
- [x] Agent-configurable aging rules
- [x] HTTP REST API matching 5-tool surface

### Phase 2 — Dynamic Agent Priming
1. Per-agent startup briefing generation (identity narrative, open threads, meta-learnings)
2. Generate briefing from highest-level summaries + hot chunks + unresolved relations
3. Structured "open threads" output from reflection data

### Phase 3 — Data Model Evolution
1. Leaf/Node behavioral semantics (immutable evidence vs. mutable summaries)
2. Versioning + supersession chains as first-class truth resolution
3. Explicit summary-node consolidation bookkeeping

### Phase 4 — Multi-Tree Cognition
1. Tree dimension model (projects, people, timeline, knowledge)
2. Cross-tree referencing of same leaves
3. Faceted recall returning multi-tree perspectives

### Phase 5 — Salience Engine
1. Compute salience from: interaction density, revision events, tree spread, child generation, contradiction proximity
2. Blend salience with semantic + recency scoring
3. Expose salience explanation metadata in recall/focus responses

### Phase 6 — Trust and Collaboration
1. UCAN-based share with attenuation and expiry
2. Multi-agent isolation modes
3. Auditable capability checks

### Phase 7 — Backend and Ops
1. Turso/SQLite backend for embedded default
2. PostgreSQL + pgvector for production
3. Scheduled aging/consolidation jobs

## Success Criteria

A release is vision-aligned when:
- An agent can start a session with personalized identity priming
- Recall is faceted across multiple trees
- Focus can descend from summary nodes to evidence leaves with revision history
- Think can consolidate, detect contradictions, and evolve summaries
- Share uses delegatable cryptographic capabilities
