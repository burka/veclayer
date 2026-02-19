# VecLayer Tasks

## Open

- [ ] Finalize MCP tool contract for `recall`, `focus`, `store`, `think`, `share` (legacy tool surface removed).
  - [x] Added MCP aliases (`recall`, `focus`, `store`, `think`, `share`) mapped to existing handlers, including a scoped share-token response.
- [ ] Implement per-agent startup briefing generation (identity narrative, open threads, meta-learnings).
- [ ] Implement leaf/node semantics where leaves may be corrected/updated with explicit version history and supersession links.
- [ ] Implement multi-tree model and faceted recall across trees.
- [ ] Implement contradiction workflow: detect conflicts and maintain `superseded_by` chains as first-class truth resolution.
- [ ] Implement computed salience engine and integrate into ranking.
- [ ] Expand reflect into `think` pipeline with consolidation actions and trace logging.
- [ ] Implement `share` with UCAN capabilities, attenuation, expiry, and offline verifiability.
- [ ] Add multi-agent isolation modes (isolated DB, shared leaves/separate trees, shared trees).
- [ ] Implement Turso/SQLite backend and trait-backed selection.
- [ ] Implement PostgreSQL + pgvector backend for production mode.
- [ ] Add scheduled jobs for aging/salience/consolidation.
- [ ] Add integration tests for end-to-end recall→focus→store→think flows.
- [ ] Add security tests for capability enforcement and unauthorized scope access.
- [ ] Add regression tests for relation semantics (`superseded_by`, `summarized_by`, `related_to`, `derived_from`).

## Completed

- [x] Updated README/Architecture to reflect the explicit vision narrative and unified 5-tool mental model.
- [x] Added deterministic local embedder implementation to remove external ONNX download dependency in core build/tests.
- [x] Enabled full local test execution by vendoring required protobuf includes for dependency codegen.
- [x] Added first-class CLI commands for `recall`, `store`, `think`, and `share`.
- [x] Added REST endpoints for `/api/recall`, `/api/focus`, `/api/store`, `/api/think`, `/api/share`.
- [x] Established trait-oriented architecture (`DocumentParser`, `Embedder`, `VectorStore`, `Summarizer`).
- [x] Implemented hierarchical parsing/search and basic MCP server endpoints.
- [x] Implemented visibility, relation metadata, and recency-aware access profiles.
