# VecLayer Architecture

VecLayer is a hierarchical vector indexing system for documents. It preserves document structure during embedding and enables navigation from overview to detail.

## Core Concepts

### Hierarchical Chunking

Documents are split into chunks based on their natural structure (headings, sections). Each chunk maintains a parent-child relationship.

#### Chunk Levels

- H1: Top-level chapters
- H2: Sections within chapters
- H3-H6: Subsections
- Content: Paragraphs and text blocks

#### Parent-Child Relationships

Each chunk stores a reference to its parent chunk, enabling tree traversal during search.

### Embedding Strategy

Every chunk is embedded independently, but search can leverage the hierarchy.

## System Components

### DocumentParser Trait

The DocumentParser trait provides an abstraction for parsing different document formats.

#### MarkdownParser

Uses pulldown-cmark to parse Markdown files. Extracts headings and content blocks.

#### Future Parsers

- PdfParser: For PDF documents
- HtmlParser: For web pages
- CodeParser: Using tree-sitter for source code

### VectorStore Trait

Provides an abstraction over vector databases.

#### LanceDB Implementation

LanceDB is the default backend. It's a serverless vector database that stores data locally.

##### Advantages

- No server required
- Fast local queries
- Easy deployment

##### Schema

The LanceDB schema includes: id, content, embedding, level, parent_id, path, source_file

### Embedder Trait

Abstracts the embedding model.

#### FastEmbed

Uses ONNX runtime for local CPU embedding. No external API calls required.

#### Future Embedders

- Ollama: GPU-accelerated local embedding
- OpenAI: Cloud-based embedding API

## Search Strategy

### Top-Down Search

1. Embed the query
2. Search H1/H2 level chunks first
3. Find the best matching sections
4. Drill down into matching sections
5. Return hierarchical results

### Subtree Search

When you know which section to search in, you can search within a specific subtree.

## Deployment

### Single Binary

VecLayer compiles to a single binary with all dependencies included.

### Data Directory

All data is stored in a single directory (./veclayer-data/). Copy this directory for backup or deployment.

### Environment Variables

- VECLAYER_DATA_DIR: Data directory path
- VECLAYER_PORT: Server port
- VECLAYER_HOST: Server host
