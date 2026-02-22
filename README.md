# VecLayer

**Long-term memory for AI agents. Hierarchical, perspectival, aging knowledge.**

> Status: In Development — prototype, APIs may change
> Author: Florian Burka, developed in dialogue with Claude

## What is VecLayer?

VecLayer organizes knowledge as a hierarchy: summaries over summaries, at arbitrary depth, from different perspectives on the same raw data. A search starts with the overview and drills down on demand — like human remembering.

Instead of flat chunk lists or key-value stores, VecLayer provides structured, aging, self-describing memory. From the statistical shape of all memories — embedding clusters weighted by salience — an identity emerges organically.

### The Core Thesis

Summaries are not a feature alongside others — they *are* the memory itself. The hierarchy that makes RAG better (overview before detail, navigation instead of flat lists) is the same structure from which identity emerges. And personality is not shaped by what you do often, but by what moved you. That is why salience measures significance, not frequency.

## Core Concepts

- **One primitive: Entry** — Everything is an Entry. Four types: `raw`, `summary`, `meta`, `impression`. ID = `sha256(content)`, first 7 hex chars in CLI (like git). Identical content = identical ID = idempotent.
- **Seven perspectives** — `intentions`, `people`, `temporal`, `knowledge`, `decisions`, `learnings`, `session`. Each perspective has hints for LLMs. Extensible with custom perspectives.
- **Memory aging** — RRD-inspired access tracking with fixed time windows. Important stays present, unused fades. Configurable degradation rules.
- **Salience** — Measures significance, not frequency. Composite of interaction density (0.5), perspective spread (0.25), and revision activity (0.25). High-salience entries survive aging.
- **Identity** — Emerges from salience-weighted embedding centroids per perspective. On connect, the agent receives a priming: core knowledge, open threads, recent learnings. The moment an agent wakes up and knows itself.
- **Sleep cycle** — Optional LLM-powered consolidation: reflect → think → add → compact. The only module requiring an LLM.

For technical details see [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick Start

```bash
# Initialize a new VecLayer store
veclayer init

# Store knowledge
veclayer store ./docs                        # files/directories
veclayer store "Core decision: Rust"         # single text
veclayer store --perspective decisions "We chose Turso over Postgres"

# Recall
veclayer recall "architecture decisions"
veclayer recall --perspective decisions "backend"

# Drill down
veclayer focus abc1234

# Start server (MCP/HTTP)
veclayer serve
```

## CLI Overview

| Command | Description |
|---------|-------------|
| `store` | Store knowledge (text, file, directory) |
| `recall` | Semantic search with perspective filter |
| `focus` | Drill into an entry, show children |
| `reflect` | Identity snapshot, salience ranking, archive candidates |
| `think` | Curate: promote, demote, relate, aging, LLM consolidation |
| `serve` | Start MCP/HTTP server |
| `status` | Store statistics |
| `perspective` | Manage perspectives (list, add, remove) |
| `history` | Show version/relation history of an entry |
| `archive` | Demote entries to deep_only visibility |
| `export` | Export entries to JSONL |
| `import` | Import entries from JSONL |

Aliases: `add` = `store`, `search`/`s` = `recall`, `f` = `focus`, `id` = `reflect`

## Building from Source

### Prerequisites

- **Rust** toolchain (stable, edition 2021+)
- **protoc** (Protocol Buffers compiler) — required by LanceDB
  - Debian/Ubuntu: `apt-get install protobuf-compiler`
  - macOS: `brew install protobuf`
- **Internet access** during first build — `ort-sys` downloads ONNX Runtime (~19 MB)

### Build

```bash
cargo build              # debug build
cargo build --release    # optimized build
```

### First Run

On first use, VecLayer downloads the embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB via HuggingFace) to a local cache (`.fastembed_cache/` relative to the working directory). This requires internet access.

```bash
veclayer init
veclayer store "test"   # triggers model download on first run
```

### Troubleshooting

**`Failed to initialize FastEmbed: Failed to retrieve onnx/model.onnx`**
The embedding model couldn't be downloaded. Common causes:
- No internet access or corporate TLS proxy intercepting HTTPS
- Fix: manually download the model files from `Xenova/bge-small-en-v1.5` on HuggingFace and place them in `.fastembed_cache/models--Xenova--bge-small-en-v1.5/snapshots/<commit_hash>/`

**`Could not find protoc`**
Install the Protocol Buffers compiler (see prerequisites above).

**`Failed to connect to Ollama`**
The think cycle and cluster summarization require a running Ollama instance. These features are optional — `store`, `recall`, `focus`, and all non-LLM commands work without it.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Rust |
| Storage | LanceDB (prototype), Turso/SQLite (planned) |
| Embeddings | fastembed (CPU, ONNX) — trait-based, swappable |
| Parsing | pulldown-cmark (Markdown), extensible |
| Server | axum (MCP + HTTP) |
| CLI | clap v4 |
| Config | TOML + ENV overrides (12-Factor) |

## Status

Phases 1–5.5 complete: core model, perspectives, aging/salience, identity, think cycle, tool ergonomics. Next up: Phase 6 (UCAN sharing). See [Issues](https://github.com/burka/veclayer/issues?q=label%3Aphase) for the full roadmap.

## Design Decisions: What VecLayer Does NOT Do

Explicitly rejected approaches — documented and reasoned, not forgotten.

| Rejected | Instead | Why |
|----------|---------|-----|
| JSON annotations on entries | Content carries the semantics | No schema drift from optional fields |
| Paths as sole structure | Perspectives | Same entry, different views |
| Tags | Perspectives with hints | Tags are flat and unexplained |
| Separate vector spaces for emotions | Salience as composite score | One space, different weightings |
| S3 backends | Local files + Turso/pgvector | Simplicity, latency, offline capability |
| ACLs | UCAN | Decentralized, delegatable, offline-verifiable |
| Bearer tokens | UCAN with DID | Cryptographic, attenuatable |
| Static tool descriptions | Dynamic priming | Personalized per agent and session |
| Leaf/node separation | Everything is an Entry | One primitive, four types |
| "Trees" as concept | Perspectives | Trees are rigid, perspectives are views |
| Graph database | Relations on entries | The graph reveals itself in visualization |
| Metadata fields for emotions | Perspectives + content | The perspective *is* the semantics |
| Tool call hooks for auto-capture | Behavioral hints in priming | Intelligence stays with the agent |

## License

MIT
