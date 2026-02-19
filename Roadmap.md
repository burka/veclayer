# VecLayer Roadmap (Vision Alignment)

## Current State Summary

VecLayer already has a strong prototype foundation:
- hierarchical chunking + summaries
- vector search + recency boost
- visibility and relations
- MCP integration + ingest/query/serve CLI

This aligns with the core vision direction, but not yet with the full "agent-first memory operating system" described in the target concept.

## Alignment Scorecard

| Domain | Vision Target | Current Status | Fit |
|---|---|---|---|
| Agent priming at connect | Identity narrative + open threads + meta-learnings | Static MCP instructions only | Low |
| Tool model | `recall`, `focus`, `store`, `think`, `share` | `search`, `get_chunk`, `get_children`, `promote`, `demote`, `relate`, `ingest_chunk`, `reflect` | Medium |
| Data primitives | Immutable leaves + mutable nodes | Single mutable chunk type with hierarchy fields | Low |
| Multi-tree memory | Projects/people/time/knowledge trees reusing same leaves | Single hierarchy from source headings | Low |
| Supersession semantics | First-class temporal truth chains | Supported relation kind, not yet policy-driven in recall | Medium |
| Salience engine | computed from density/revisions/spread/proximity | access-based relevance + visibility, no full salience engine | Medium-Low |
| Access profile | Constant-size fixed buckets | Implemented fixed buckets (hour/day/week/month/year/total) | Medium |
| Reflection loop (`think`) | Contradiction detection + consolidation + priming updates | `reflect` gives heuristics; no autonomous consolidation workflow | Medium-Low |
| Sharing via UCAN | Delegatable, attenuated, expiring capability tokens | not implemented | Low |
| Multi-agent isolation modes | isolated/shared leaves/shared trees | not implemented | Low |
| Storage strategy | Turso/SQLite embedded + Postgres/pgvector server | LanceDB implemented; Turso/pgvector planned | Medium-Low |

## Priority Phases

### Phase 1 — Agent Experience Parity
1. Introduce MCP aliases and semantics for `recall`, `focus`, `store`, `think`, `share`.
2. Add dynamic connect-time briefing generation per agent.
3. Add structured "open threads" and "meta-learnings" outputs from reflection.
4. Keep README/Architecture/Tasks synchronized with the current vision language and API naming.

### Phase 2 — Data Model Evolution (Leaf/Node semantics)
1. Clarify behavioral semantics first (recall/focus/store/think/share) before hard splitting storage objects.
2. Support mutable leaves where corrections are needed, tracked via versioning/supersession metadata.
3. Add explicit summary-node behavior (`SummarizedBy`) and consolidation bookkeeping with backward-compatible reads.

### Phase 3 — Multi-Tree Cognition
1. Add tree dimension model (`projects`, `people`, `timeline`, `knowledge`).
2. Support cross-tree referencing of the same leaf.
3. Add faceted recall returning multi-tree perspectives.

### Phase 4 — Meaning/Salience Engine
1. Compute salience from interaction density, revision events, tree spread, child generation, contradiction proximity, and explicit emphasis.
2. Blend salience with semantic and recency scoring.
3. Expose salience explanation metadata in recall/focus responses.

### Phase 5 — Trust and Collaboration
1. Implement UCAN-based `share` with attenuation and expiry.
2. Add multi-agent isolation modes.
3. Add auditable capability checks in server layer.

### Phase 6 — Backend and Ops Completion
1. Add Turso/SQLite backend for embedded default path.
2. Add PostgreSQL + pgvector backend for server deployments.
3. Add scheduled aging/consolidation jobs and operational metrics.

## Success Criteria

A release is vision-aligned when:
- An agent can start a session with personalized identity priming.
- Recall is faceted across multiple trees.
- Focus can descend from summary nodes to evidence leaves with clear revision history.
- Think can consolidate, detect contradictions, and evolve summaries.
- Share uses delegatable cryptographic capabilities.
