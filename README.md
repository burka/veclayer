# VecLayer

**Hierarchische Vektor-Datenbank, die Wissen in Bäumen organisiert statt in flachen Chunk-Listen.**

> **Status:** Experimental – Prototyp, APIs können sich ändern
> **Autor:** Florian Schmidt, entwickelt im Dialog mit Claude

## Was ist VecLayer?

VecLayer ist eine hierarchische Vektor-Datenbank, die Wissen in Bäumen organisiert. Höhere Knoten im Baum enthalten automatisch generierte Zusammenfassungen der darunterliegenden Inhalte. Dadurch kann eine Suche zuerst den Überblick liefern und bei Bedarf in die Tiefe gehen – wie ein Mensch, der sich erst an das Thema erinnert und dann an die Details.

Zusätzlich wird VecLayer ein optionales **Memory-Aging-System** bekommen, das Zugriffsmuster über feste Zeitfenster trackt (inspiriert von RRDtool) und Daten kontrolliert altern lässt – ohne sie zu verlieren. Daten beschreiben selbst, wie sie altern sollen. Ein Agent oder Nutzer kann zur Suchzeit entscheiden, welches Relevanzprofil gilt.

VecLayer ist als Rust-Bibliothek mit MCP-Server-Interface konzipiert. Es soll sowohl als Embedded-Lösung (Single Binary + Datenverzeichnis) als auch mit PostgreSQL als Backing Store nutzbar sein.

**Kernideen:**
- Heading-aware Parsing bewahrt die Dokumenthierarchie (H1 → H2 → H3 → Content)
- Zusammenfassungen *sind* die Hierarchie – nicht Metadaten, sondern verdichteter Inhalt mit eigenem Embedding
- Soft Clustering gruppiert semantisch verwandte Chunks über Dokumentgrenzen hinweg
- Überlappende Bäume: Daten können in mehreren Dimensionen gleichzeitig organisiert sein (Thema, Zeit, Projekt)
- Memory Aging: Zeitbasierte Relevanz statt "alles gleich laut" (geplant)
- Self-Describing Data: Daten konfigurieren selbst, wie sie behandelt werden (geplant)

## Aktueller Stand

### Implementiert (v0.1 – Prototyp)
- [x] Markdown-Parsing mit Heading-Hierarchie (pulldown-cmark)
- [x] FastEmbed-Embeddings (ONNX, CPU-only)
- [x] LanceDB-Backend (serverless, kein Setup)
- [x] RAPTOR-Style Soft Clustering + LLM-Summaries (via Ollama)
- [x] Hierarchische Suche mit Kontext-Aufbau
- [x] MCP-Server für KI-Assistenten-Integration (HTTP + stdio)
- [x] CLI: ingest, query, serve, stats, sources
- [x] Trait-basierte Architektur (DocumentParser, Embedder, VectorStore, Summarizer)
- [x] 12-Factor-Konfiguration via Environment-Variablen

### Geplant
- [ ] Memory Aging (RRD-Style Access Tracking)
- [ ] Self-Describing Data (Visibility: Always, Normal, DeepOnly, Expiring, Seasonal)
- [ ] Überlappende Bäume (Multi-Dimension: Thema × Zeit × Projekt)
- [ ] Relationen (SupersededBy, SummarizedBy, RelatedTo, DerivedFrom)
- [ ] Turso/Limbo als Embedded-Backend (SQLite-kompatibel, Pure Rust)
- [ ] PostgreSQL + pgvector Backend (Production)
- [ ] Multi-Format-Parsing (PDF, HTML, Code via tree-sitter)
- [ ] Erweiterte CLI (promote, demote, relate, --deep, --recent)

## Wie es funktioniert

### Ingest-Pipeline

```
Dokument → Parser → Hierarchy Builder → Embeddings → Clustering → Store
                       │                                │
                  parent/child                soft assignments +
                  relationships               LLM-Zusammenfassungen
```

### Suche: Top-Down-Kaskade

```
Suche: "Was war die Architekturentscheidung bei VecLayer?"

1. Suche in Wurzelknoten (Zusammenfassungen aller Themen)
   → Match: "Softwarearchitektur" (Score 0.8)

2. Suche in Kindern von "Softwarearchitektur"
   → Match: "VecLayer" (Score 0.9)

3. Suche in Kindern von "VecLayer"
   → Match: "Architektur-Evolution" (Score 0.95)

4. Liefere Zusammenfassung + Option auf Detail-Chunks
```

Der Agent bekommt bei jeder Ebene genug Kontext, um zu entscheiden: "Reicht mir das, oder gehe ich tiefer?"

### Hierarchie-Ebenen

Jede Heading-Ebene erzeugt einen Chunk, der seine Kinder enthält. Ein Suchtreffer auf einen Absatz kann die übergeordnete Sektion für Kontext liefern.

### Soft Clustering

Anders als hartes Clustering kann jeder Chunk mit unterschiedlicher Wahrscheinlichkeit zu mehreren Clustern gehören. Ein Chunk über "Rust Memory Safety" könnte zu 60% im "Rust"-Cluster und zu 40% im "Memory Management"-Cluster sein.

### Zusammenfassungen

Für jeden Cluster generiert ein LLM eine Zusammenfassung, die selbst durchsuchbar wird – das ermöglicht die Entdeckung dokumentübergreifender Themen.

## Motivation: Das Problem mit flachem RAG

Klassische RAG-Systeme zerlegen Dokumente in Chunks gleicher Größe, betten sie ein und suchen per Cosine Similarity. Dabei geht verloren:

- **Struktur:** Ein Kapitel hat Unterabschnitte, die haben Absätze. Flache Chunks wissen nichts voneinander.
- **Kontext:** Ein Chunk über "Authentifizierung" ohne Wissen, dass er zum Kapitel "Sicherheitsarchitektur" gehört, ist weniger nützlich.
- **Zeitliche Entwicklung:** Alle Chunks sind gleich "laut", egal ob sie von gestern oder vor zwei Jahren stammen.
- **Verdichtung:** Es gibt keine Zusammenfassungen. Man bekommt entweder alles oder nichts.

| Aspekt | Flat Chunking | VecLayer |
|--------|---------------|----------|
| Struktur | Fixed-size Windows | Heading-aware Hierarchie |
| Kontext | Verloren zwischen Chunks | Bewahrt via Parent-Links |
| Cross-Doc-Links | Keine | Cluster-Zusammenfassungen |
| Retrieval | Nur top-k | Top-k + Hierarchie-Expansion |
| Abstraktionsebenen | Eine | Mehrere (Content → Section → Summary) |
| Zeitliche Relevanz | Keine | Memory Aging (geplant) |
| Datenleben | Statisch | Self-Describing (geplant) |

## Installation

```bash
cargo install --path .
```

Voraussetzungen:
- Rust 1.75+
- Ollama (optional, für Summarization) – `ollama pull llama3.2`

## Quick Start

```bash
# Dokumentenordner indexieren
veclayer ingest ./docs

# Suchen
veclayer query "Wie funktioniert Authentifizierung"

# Suche mit Hierarchie-Pfad
veclayer query -p "Fehlerbehandlung"

# Status prüfen
veclayer stats
veclayer sources
```

## CLI-Referenz

### `veclayer ingest <PATH>`

Dokumente in den Vector Store aufnehmen.

```bash
veclayer ingest ./docs                    # Rekursiv + Summarization (Standard)
veclayer ingest ./docs --no-summarize     # Ohne Clustering/Summaries (schneller)
veclayer ingest ./docs --no-recursive     # Nur ein Verzeichnis
veclayer ingest ./docs --model tinyllama  # Anderes Ollama-Modell
```

### `veclayer query <QUERY>`

Den Vector Store durchsuchen.

```bash
veclayer query "memory safety"            # Standard-Suche
veclayer query "memory safety" -k 10      # Top 10 Ergebnisse
veclayer query "memory safety" -p         # Hierarchie-Pfad anzeigen
veclayer query "auth" --subtree chunk_id  # Innerhalb eines Teilbaums suchen
```

### `veclayer serve`

MCP-Server (Model Context Protocol) starten für KI-Assistenten-Integration.

```bash
veclayer serve                # HTTP-Server (Standard)
veclayer serve --mcp-stdio    # Stdio-Transport (für Claude Desktop)
veclayer serve --read-only    # Production-Modus
```

### `veclayer stats` / `veclayer sources`

```bash
veclayer stats    # Index-Statistiken anzeigen
veclayer sources  # Indexierte Dateien auflisten
```

### Globale Optionen

```bash
-d, --data-dir <PATH>   # Speicherort (Standard: ./veclayer-data)
-v, --verbose           # Ausführliche Ausgabe
```

### Konfiguration (12-Factor)

```env
VECLAYER_DATA_DIR=./veclayer-data
VECLAYER_EMBEDDER=fastembed        # oder ollama
VECLAYER_STORE=lancedb             # aktuell: lancedb, geplant: turso, postgres
VECLAYER_OLLAMA_MODEL=llama3.2
VECLAYER_OLLAMA_URL=http://localhost:11434
VECLAYER_PORT=8080
VECLAYER_HOST=127.0.0.1
VECLAYER_SEARCH_TOP_K=5
VECLAYER_SEARCH_CHILDREN_K=3
```

## Roadmap

### Phase 1: Kernverbesserungen (v0.2)

**Zusammenfassungen als Hierarchie**
- Zusammenfassungen werden vom Add-on zum Kernkonzept: Jeder Elternknoten IST eine Zusammenfassung seiner Kinder
- Batch-Generierung im Nachgang mit LLM-Unterstützung und Kontext
- Reflexionsfunktion: Die Datenbank stellt gezielt Fragen zu den Daten (Was ist wichtig? Welche Tags? Zusammenfassung bitte.)

**Self-Describing Data (Visibility)**
- `Always` – Immer sichtbar, nie degradiert (Architekturentscheidungen, Kernwissen)
- `Normal` – Standard-Kaskade, altert natürlich mit Zugriffsmuster
- `DeepOnly` – Nur bei expliziter tiefer Suche (alte Chat-Logs, verworfene Ideen)
- `Expiring` – Selbstzerstörend nach Datum (temporäre Planungsdaten)
- `Seasonal` – Zyklisch relevant, gesteuert über Zugriffshäufigkeit

**Relationen zwischen Daten**
- `SupersededBy(id)` – "Dieser Fakt wurde durch neuere Info ersetzt"
- `SummarizedBy(id)` – "Verdichtet in diesem Knoten"
- `RelatedTo(id)` – "Lose thematische Verbindung"
- `DerivedFrom(id)` – "Entstand aus dieser Diskussion"
- Bewusste Einschränkung: Maximal ein bis zwei Hops, keine Graph-Traversierung

### Phase 2: Memory Aging (v0.3)

**RRD-Style Access Tracking**
```
┌─────────────────────────────────────────────────────┐
│              AccessProfile (40 Bytes)                │
├──────────┬──────────┬────────┬───────┬──────────────┤
│  1min    │  10min   │  1h    │  24h  │  7d  30d ... │
│  (u16)   │  (u16)   │  (u16) │ (u16) │  total (u32) │
└──────────┴──────────┴────────┴───────┴──────────────┘
```

- Feste Buckets pro Chunk, konstante Größe, keine wachsenden Logs
- Periodische Aggregation von feineren in gröbere Buckets
- Relevanzprofil zur Suchzeit wählbar:
  - `--recent 7d` → letzte Woche gewichten
  - `--deep` → alles durchsuchen inkl. DeepOnly
  - Standard → `total` nutzen

**Erweiterte CLI-Kommandos**
```bash
veclayer promote <id> --visibility always
veclayer demote <id> --visibility deep-only
veclayer relate <id> --superseded-by <new-id>
veclayer query "aktuelle Planung" --recent 7d
veclayer query "alte Sprint-Notizen" --deep
```

### Phase 3: Überlappende Bäume (v0.4)

**Multi-Dimension-Bäume**

Ein Datensatz kann in mehreren Bäumen gleichzeitig existieren – gruppiert nach verschiedenen Dimensionen:

```
Nach Thema:                    Nach Zeit:
┌─ Softwarearchitektur         ┌─ 2026
│  ├─ Rust-Projekte            │  ├─ Q1
│  │  ├─ VecLayer              │  │  ├─ Februar
│  └─ Go-Projekte              │  ...

Nach Projekt:                  Nach Person:
┌─ VecLayer                    ┌─ Florian
│  ├─ Konzeptphase             │  ├─ Beruflich
│  └─ Prototype Fund           │  └─ Projekte
```

- Mehrere Parent-Pointer pro Chunk
- Automatische Einordnung über Metadaten und Embedding-Ähnlichkeit

### Phase 4: Backends & Parsing (v0.5+)

**Turso/Limbo als Embedded Backend**
- SQLite-kompatibel, Pure Rust, Vector Search eingebaut
- Ersetzt LanceDB als primäres Embedded-Backend
- Fallback auf plain SQLite falls nötig

**PostgreSQL + pgvector (Production)**
```
Dokument → Parser → Hierarchy Builder → Embeddings → PostgreSQL/pgvector
Query → Traversal Engine → Context Assembly → Response
```

**Multi-Format Parsing**

| Parser | Focus | Notes |
|--------|-------|-------|
| [Docling](https://github.com/DS4SD/docling) | Multi-format | IBM, PDF/DOCX/Images |
| [Apache Tika](https://tika.apache.org/) | Multi-format | Battle-tested, breite Unterstützung |
| [MinerU](https://github.com/opendatalab/MinerU) | PDF/Images | OCR-Fokus |
| [tree-sitter](https://tree-sitter.github.io/) | Code | Syntax-aware Code-Chunking |

### Reflexions-Pattern (Agent-Ebene)

Kein Core-Feature, sondern ein Agent-Pattern auf VecLayer – extern getriggert:

- Nicht gezielt suchen, sondern schauen was "hochkommt"
- Chunks mit hohen Access-Counts der letzten Woche reviewen
- Widersprüche erkennen und flaggen
- Ungelöste Dinge identifizieren
- Entscheiden: Zusammenfassen? Promoten? Archivieren?

Vergleichbar mit menschlicher Meditation: Frei sein von gezielter Erinnerung und beobachten, was Aufmerksamkeit erzeugt hat. Getriggert von außen (Timer, Agent-Steuerung), nicht automatisch.

## Technischer Stack

| Komponente | Technologie | Begründung |
|---|---|---|
| Sprache | Rust | Performance, Memory Safety, Single Binary |
| Embeddings | fastembed (CPU) | Trait-basiert, austauschbar |
| Summarization | Ollama (lokal) | Lokales LLM, kein Cloud-Dependency |
| Vektor-Suche | LanceDB (Prototyp) | File-basiert, einfach |
| Backing Store (geplant) | Turso/Limbo | SQLite-kompatibel, Pure Rust |
| Backing Store (geplant) | PostgreSQL + pgvector | Skalierbar, Backup, HA |
| Parsing | pulldown-cmark | Markdown → Heading-Hierarchie direkt nutzbar |
| MCP Server | axum | Direkte Integration in Claude, andere Agenten |
| CLI | clap | `ingest`, `query`, `serve`, `stats`, `sources` |

## Deployment

```
Minimal:    veclayer binary + veclayer-data/ → fertig
Production: veclayer binary → PostgreSQL + pgvector (geplant)
```

## Entwicklung

```bash
# Tests ausführen
cargo test

# Ollama-Integrationstests
cargo test --test ollama_integration -- --ignored

# Coverage prüfen
cargo tarpaulin --out Html
```

## Entstehungsgeschichte

Das Konzept entstand in mehreren Gesprächen zwischen Florian und Claude (November 2025 – Februar 2026):

1. **Ausgangspunkt:** Frustration mit flachem RAG – Dokumente für brieflotse.de verloren ihre Struktur beim Chunking.
2. **Erste Idee (Nov 2025):** Hierarchische Vektor-Indexierung – Heading-Levels als natürliche Baumstruktur. Rust-Prototyp mit LanceDB, MCP-Server für Claude.
3. **Naming & Domain (Nov 2025):** Aus Kandidaten wie Hive, Strata, Hierav wurde "VecLayer". Prototype-Fund-Bewerbung vorbereitet.
4. **Memory-Aging-Erweiterung (Feb 2026):** Aus der Frage "Wie könnte ein KI-Agent seine Erinnerungen managen?" entstand das RRD-inspirierte Access-Tracking, Self-Describing Data, und das Meditations-Pattern.
5. **Beziehungen & temporale Wahrheit (Feb 2026):** Statt Graph-DB schlanke Relationen (SupersededBy, SummarizedBy). Event-Sourcing-Analogie: History bleibt, Zusammenfassungen sind die Projektion.

## Lizenz

MIT

