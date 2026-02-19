# VecLayer Tasks

## Open

- [ ] Implement per-agent startup briefing generation (identity narrative, open threads, meta-learnings)
- [ ] Implement leaf/node semantics with version history and supersession links
- [ ] Implement multi-tree model and faceted recall across trees
- [ ] Implement contradiction detection in think (compare overlapping chunks, flag conflicts)
- [ ] Implement computed salience engine and integrate into ranking
- [ ] Implement UCAN-based share with attenuation, expiry, and offline verifiability
- [ ] Add multi-agent isolation modes (isolated DB, shared leaves/separate trees, shared trees)
- [ ] Implement Turso/SQLite backend and trait-backed selection
- [ ] Implement PostgreSQL + pgvector backend for production
- [ ] Add scheduled jobs for aging/salience/consolidation
- [ ] Add integration tests for end-to-end recall→focus→store→think flows
- [ ] Add security tests for capability enforcement
- [ ] Add regression tests for relation semantics (superseded_by, summarized_by, related_to, derived_from)

## Completed

- [x] 5-tool MCP surface: recall, focus, store, think, share (replacing legacy search/get_chunk/promote/demote/etc.)
- [x] think as curation hub: reflect (default) + promote/demote/relate/configure_aging/apply_aging via action parameter
- [x] focus with semantic reranking: children sorted by cosine similarity to question embedding
- [x] share as honest preview (token payload without UCAN crypto, clearly labeled)
- [x] MCP instructions: comprehensive guide to tools, reflection pattern, summarization pattern
- [x] HTTP REST API: /api/recall, /api/focus, /api/store, /api/think, /api/share, /api/stats
- [x] Agent-configurable aging rules (AgingConfig JSON, get_hot_chunks, get_stale_chunks)
- [x] RRD-style access tracking (hour/day/week/month/year/total with roll-up)
- [x] Trait-oriented architecture (DocumentParser, Embedder, VectorStore, Summarizer)
- [x] Hierarchical parsing/search with BGE embeddings (fastembed)
- [x] Visibility, relation metadata, and recency-aware search
