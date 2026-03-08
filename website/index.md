# VecLayer

**Long-term memory for AI agents. Hierarchical, perspectival, aging knowledge.**

---

VecLayer organizes knowledge as a hierarchy: summaries over summaries, at arbitrary depth,
from different perspectives on the same raw data. A search starts with the overview and
drills down on demand — like human remembering.

## Core Concepts

- **One primitive: Entry** — Everything is an Entry. Four types: `raw`, `summary`, `meta`, `impression`.
  ID = `sha256(content)`. Identical content = identical ID = idempotent.
- **Seven perspectives** — `intentions`, `people`, `temporal`, `knowledge`, `decisions`, `learnings`, `session`.
- **Memory aging** — RRD-inspired access tracking. Important stays present, unused fades.
- **Salience** — Measures significance, not frequency.
- **Identity** — Emerges from salience-weighted embedding centroids per perspective.

## Five Tools

| Tool | Purpose |
|------|---------|
| **recall** | Semantic search — *What do I know about this?* |
| **focus** | Drill deeper — *Tell me more about this specific point.* |
| **store** | Write knowledge — *I want to remember this.* |
| **think** | Curate and reflect — *Let me consolidate.* |
| **share** | Scoped access tokens — *Here, this is for you.* |

## Quick Start

```bash
# Build from source
cargo build --release

# Initialize a store
veclayer init

# Add knowledge
veclayer add ./docs
veclayer add --perspective decisions "We chose Rust for performance"

# Search
veclayer search "architecture decisions"

# Start server
veclayer serve
```

## API Endpoints

```
GET  /health       → "OK"
GET  /api/stats    → Store statistics
POST /api/recall   → { query, perspective?, top_k?, deep? }
POST /api/focus    → { id, question? }
POST /api/store    → { content, heading?, perspectives?, parent_id? }
POST /api/think    → { action?, target_id?, kind? }
POST /api/share    → { scope, permissions }
```

## MCP Integration

VecLayer runs as an MCP server for Claude Desktop and other MCP-compatible agents:

```bash
# HTTP transport (default)
veclayer serve

# Stdio transport (Claude Desktop)
veclayer serve --mcp-stdio
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Rust |
| Storage | LanceDB |
| Embeddings | fastembed (CPU, ONNX) |
| Parsing | pulldown-cmark |
| Server | axum |

## Links

- [GitHub](https://github.com/burka/veclayer)
- [Architecture](https://github.com/burka/veclayer/blob/main/ARCHITECTURE.md)

---

*MIT License — Florian Burka, developed in dialogue with Claude*
