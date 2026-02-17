# VecLayer Architektur

VecLayer ist eine hierarchische Vektor-Datenbank mit Gedächtnis-Funktionen. Das Ziel: Session-übergreifende Identität für KI-Agenten durch strukturiertes, alterndes, selbstbeschreibendes Wissen.

## Kernprinzipien

1. **Zusammenfassungen sind die Hierarchie** – Nicht Metadaten, sondern verdichteter Inhalt mit eigenem Embedding
2. **Daten beschreiben sich selbst** – Visibility, Relationen, Access-Profile sind Teil jedes Chunks
3. **Finden, dann navigieren** – Vektor-Suche findet, Relations ermöglichen 1-2 Hops, keine Graph-Traversierung
4. **Pragmatisch, nicht perfekt** – Bewährte Patterns (RRD, Event Sourcing, hierarchisches Indexing) statt Neuerfindung

## Datenmodell

### HierarchicalChunk

Das zentrale Datenobjekt. Jeder Chunk trägt:

```rust
HierarchicalChunk {
    // Identität
    id: String,
    content: String,
    embedding: Option<Vec<f32>>,

    // Hierarchie
    level: ChunkLevel,        // H1-H6, Content
    parent_id: Option<String>,
    path: String,             // "Chapter 1 > Section 1.1 > Details"

    // Herkunft
    source_file: String,
    heading: Option<String>,

    // Clustering
    cluster_memberships: Vec<ClusterMembership>,
    is_summary: bool,
    summarizes: Vec<String>,

    // Identity & Memory (open types -- extensible without code changes)
    visibility: String,               // "always", "normal", "deep_only", or any custom value
    relations: Vec<ChunkRelation>,    // kind is String: "superseded_by", "contradicts", ...
    access_profile: AccessProfile,    // created_at, last_accessed, access_count
    expires_at: Option<i64>,          // Für Expiring-Chunks
}
```

### Visibility

**Offener String**, nicht Enum. Bekannte Werte als Konstanten, beliebig erweiterbar:

| Wert | Beschreibung | Beispiel |
|------|-------------|---------|
| `"always"` | Immer sichtbar, nie degradiert | Architekturentscheidungen, Kernwissen |
| `"normal"` | Standard-Kaskade | Projektdiskussionen, aktuelle Arbeit |
| `"deep_only"` | Nur bei expliziter tiefer Suche | Alte Chat-Logs, verworfene Ideen |
| `"expiring"` | Selbstzerstörend nach Zeitstempel | Temporäre Planungsdaten |
| `"seasonal"` | Zyklisch relevant | Quartalsberichte, wiederkehrende Aufgaben |
| `"draft"` | *Eigener Wert* | Entwürfe in Arbeit |
| *beliebig* | *Eigener Wert* | Projektspezifische Kategorien |

Standard-Suche: `always` + `normal` + `seasonal` + nicht-abgelaufene `expiring` (konfigurierbar).
Deep-Suche (`--deep`): Alles. Unbekannte Visibilities: nur bei tiefer Suche.

### Relations

**Offener String** als `kind`. Gerichtete, schlanke Verbindungen:

| Kind | Semantik | Nutzen |
|------|----------|--------|
| `"superseded_by"` | Fakt wurde durch neuere Info ersetzt | Wissensevolution |
| `"summarized_by"` | Verdichtet in diesem Knoten | Navigierbare Verdichtung |
| `"related_to"` | Lose thematische Verbindung | Kontext-Entdeckung |
| `"derived_from"` | Entstand aus dieser Diskussion/Quelle | Herkunft nachvollziehen |
| `"contradicts"` | *Eigener Typ* | Widerspruchs-Erkennung |
| *beliebig* | *Eigener Typ* | Domainspezifische Relationen |

### AccessProfile

Basis für Memory Aging:

```rust
AccessProfile {
    created_at: i64,      // Wann erstellt (Unix epoch)
    last_accessed: i64,   // Wann zuletzt abgerufen
    access_count: u32,    // Wie oft abgerufen
}
```

Später erweiterbar zu RRD-Style Buckets (feste Zeitfenster, konstante Größe).

## System-Komponenten

### DocumentParser Trait

Abstraktion für verschiedene Dokumentformate.

- **MarkdownParser** (implementiert): pulldown-cmark, Heading-Hierarchie
- **Geplant:** PDF, HTML, Code (tree-sitter)

### Embedder Trait

Abstraktion über Embedding-Modelle.

- **FastEmbedder** (implementiert): ONNX Runtime, CPU, BAAI/bge-small-en-v1.5 (384 Dim.)
- **Geplant:** Ollama (GPU), OpenAI API

### VectorStore Trait

Abstraktion über Vektor-Datenbanken.

- **LanceStore** (implementiert): Serverless, file-basiert, kein Setup
- **Geplant:** Turso/Limbo (SQLite, Pure Rust), PostgreSQL + pgvector

Schema umfasst alle Chunk-Felder inkl. Visibility, Relations, AccessProfile.

### Summarizer Trait

- **OllamaSummarizer** (implementiert): Lokales LLM via REST API

### Clusterer Trait

- **SoftClusterer** (implementiert): K-Means mit Soft Assignments
- **ClusterPipeline**: Clustering → Summarization → Embedding → Store

### HierarchicalSearch

Orchestriert die Suche:

1. Query einbetten
2. Top-k-Matches finden (über alle Ebenen oder gefiltert)
3. Für jeden Match: Hierarchie-Pfad aufbauen (root → match)
4. Für jeden Match: Relevante Kinder suchen
5. Ergebnis: Chunk + Score + Hierarchie-Pfad + Kinder

### MCP Server

Model Context Protocol für KI-Assistenten-Integration:

- HTTP (axum, Standard) oder Stdio (für Claude Desktop)
- Endpoints: /search, /get-chunk, /get-children, /subtree-search, /stats

## Suchstrategie

### Standard-Suche

1. Query einbetten
2. Nearest-Neighbor über alle Chunks
3. Post-Filter: nur `is_visible_standard()` (schließt DeepOnly + Expired aus)
4. Hierarchie-Kontext aufbauen
5. Access-Profile der Treffer aktualisieren

### Deep-Suche (`--deep`)

Wie Standard, aber ohne Visibility-Filter. Findet auch DeepOnly-Chunks.

### Subtree-Suche

Suche eingeschränkt auf Kinder eines bestimmten Parent-Chunks.

## Deployment

Single Binary + Datenverzeichnis (`./veclayer-data/`). Keine externen Services für den Einstieg.

## Konfiguration

12-Factor via Environment-Variablen:

| Variable | Default | Beschreibung |
|----------|---------|-------------|
| `VECLAYER_DATA_DIR` | `./veclayer-data` | Datenverzeichnis |
| `VECLAYER_EMBEDDER` | `fastembed` | Embedder (fastembed, ollama) |
| `VECLAYER_OLLAMA_MODEL` | `llama3.2` | LLM für Summarization |
| `VECLAYER_OLLAMA_URL` | `http://localhost:11434` | Ollama-Endpoint |
| `VECLAYER_PORT` | `8080` | Server-Port |
| `VECLAYER_HOST` | `127.0.0.1` | Server-Host |
| `VECLAYER_SEARCH_TOP_K` | `5` | Top-Level-Ergebnisse |
| `VECLAYER_SEARCH_CHILDREN_K` | `3` | Kinder pro Ergebnis |
