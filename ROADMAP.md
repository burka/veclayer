# VecLayer Roadmap

The roadmap is tracked as [GitHub Issues](https://github.com/burka/veclayer/issues?q=label%3Aphase).

## Completed

- **Phase 1:** Core (Entry model, SHA-256 IDs, CLI, config)
- **Phase 2:** Perspectives (7 defaults, relations, faceted search)
- **Phase 3:** Memory Aging + Salience (composite scoring, aging protection)
- **Phase 4:** Identity + Reflect (embedding centroids, priming on connect)
- **Phase 5:** Think Cycle (LLM-powered consolidation, narrative generation)
- **Phase 5.5:** Tool Ergonomics (inline relations, batch store, browse mode, temporal filters, relevance tiers, discover unlinked pairs [#57](https://github.com/burka/veclayer/issues/57))

## Next

- **Phase 6:** Identity + Auth + Sync ([#8](https://github.com/burka/veclayer/issues/8))
  - **6a:** Identity — Ed25519 keypair, DID, `secrecy` crate, local keystore
  - **6b:** Server Auth — UCAN tokens, Bearer middleware, capability scoping
  - **6c:** Sync + Multi-Embedding — S3 backend (LanceDB), BYOE, embedding queue, user embedding column, Lance-versioned sync
  - **6d:** Platform Integration — Hosted server (Fly.io, scale-to-zero), remote MCP for claude.ai/chatgpt.com, WebAuthn, device linking
- **Phase 7:** Polish — aliases, multi-format parsing, alternative backends ([#9](https://github.com/burka/veclayer/issues/9))

## Success Criteria

A release is vision-aligned when:
1. An agent starts with personalized identity priming
2. Search is faceted across multiple perspectives
3. Focus can descend from summary entries to raw entries with revision history
4. Think can consolidate, discover unlinked connections, evolve summaries
5. Share uses delegatable cryptographic capabilities
6. VecLayer core contains no LLM — only embeddings, structure, computation
