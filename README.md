# VecLayer

**Multiple layers of vector representations for smarter retrieval.**

Hierarchical vector indexing with semantic clustering.

> **Status:** Experimental - APIs may change

## What is this?

VecLayer builds **layered vector representations** of your documents. Instead of naive fixed-size chunking, it preserves document structure (headings, sections, hierarchy) and creates semantic clusters across documents with LLM-generated summaries.

The result: queries can match at the right level of abstraction - a specific code snippet, a conceptual section, or a high-level topic spanning multiple files.

**Key ideas:**
- Heading-aware parsing preserves document hierarchy (H1 → H2 → H3 → content)
- Soft clustering groups semantically related chunks across documents
- LLM summaries (via Ollama) create searchable abstractions of clusters
- Hierarchical retrieval: find a chunk, optionally expand to its context

## Current State

- [x] Markdown parsing with heading hierarchy
- [x] FastEmbed embeddings (ONNX, runs on CPU)
- [x] LanceDB backend (serverless, no setup)
- [x] RAPTOR-style soft clustering + summaries
- [x] MCP server for AI assistant integration
- [ ] PostgreSQL / pgvector backend
- [ ] Multi-format parsing (PDF, images, OCR)
- [ ] Code-aware parsing (tree-sitter)

## How It Works

```
Ingest:
  Document → Parser → Hierarchy Builder → Embeddings → Clustering → Store
                         │                               │
                    parent/child              soft assignments +
                    relationships             LLM summaries

Query:
  Query → Embed → Vector Search → Hierarchy Traversal → Context Assembly
                       │                   │
                 chunks + summaries    expand to siblings/parents
```

**Hierarchy levels:** Each heading creates a chunk that contains its children. A search hit on a paragraph can surface its parent section for context.

**Soft clustering:** Unlike hard clustering, each chunk can belong to multiple clusters with different probabilities. A chunk about "Rust memory safety" might be 60% in a "Rust" cluster and 40% in a "memory management" cluster.

**Summaries:** For each cluster, an LLM generates a summary that becomes searchable itself - enabling discovery of cross-document themes.

## vs Flat Chunking

| Aspect | Flat Chunking | VecLayer |
|--------|---------------|----------|
| Structure | Fixed-size windows | Heading-aware hierarchy |
| Context | Lost between chunks | Preserved via parent links |
| Cross-doc links | None | Cluster summaries |
| Retrieval | Just top-k | Top-k + hierarchy expansion |
| Abstraction levels | One | Multiple (content → section → summary) |

## Installation

```bash
cargo install --path .
```

Requirements:
- Rust 1.75+
- Ollama (optional, for summarization) - `ollama pull llama3.2`

## Quick Start

```bash
# Ingest a documentation folder
veclayer ingest ./docs

# Query
veclayer query "how does authentication work"

# Query with hierarchy path
veclayer query -p "error handling"

# Check what's indexed
veclayer stats
veclayer sources
```

## CLI Reference

### `veclayer ingest <PATH>`

Ingest documents into the vector store.

```bash
veclayer ingest ./docs                    # Recursive + summarization (default)
veclayer ingest ./docs --no-summarize     # Skip clustering/summaries (faster)
veclayer ingest ./docs --no-recursive     # Single directory only
veclayer ingest ./docs --model tinyllama  # Use different Ollama model
```

### `veclayer query <QUERY>`

Search the vector store.

```bash
veclayer query "memory safety"            # Basic search
veclayer query "memory safety" -k 10      # Top 10 results
veclayer query "memory safety" -p         # Show hierarchy path
veclayer query "auth" --subtree chunk_id  # Search within subtree
```

### `veclayer serve`

Start the MCP (Model Context Protocol) server for AI assistant integration.

```bash
veclayer serve
```

### `veclayer stats` / `veclayer sources`

```bash
veclayer stats    # Show index statistics
veclayer sources  # List indexed files
```

### Global Options

```bash
-d, --data-dir <PATH>   # Storage location (default: ./veclayer-data)
-v, --verbose           # Enable verbose output
```

Environment variables:
- `VECLAYER_DATA_DIR` - Default data directory
- `VECLAYER_OLLAMA_MODEL` - Default summarization model

## Roadmap

### PostgreSQL / pgvector Backend

Production deployments need a real database. Planned architecture:

```
Document → Parser → Hierarchy Builder → Embeddings → PostgreSQL/pgvector
Query → Traversal Engine → Context Assembly → Response
```

### Multi-Format Parsing

Evaluating parsers for document processing pipeline:

| Parser | Focus | Notes |
|--------|-------|-------|
| [Docling](https://github.com/DS4SD/docling) | Multi-format | IBM, handles PDF/DOCX/images |
| [Apache Tika](https://tika.apache.org/) | Multi-format | Battle-tested, broad support |
| [MinerU](https://github.com/opendatalab/MinerU) | PDF/Images | OCR focus |
| [tree-sitter](https://tree-sitter.github.io/) | Code | Syntax-aware code chunking |

## Development

```bash
# Run tests
cargo test

# Run with Ollama integration tests
cargo test --test ollama_integration -- --ignored

# Check coverage
cargo tarpaulin --out Html
```

## License

MIT

