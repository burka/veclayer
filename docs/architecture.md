# VecLayer Architektur

VecLayer ist eine hierarchische Vektor-Datenbank, die Wissen in Bäumen organisiert. Zusammenfassungen bilden die Hierarchie – nicht als Metadaten, sondern als verdichteter Inhalt mit eigenem Embedding.

## Kernkonzepte

### Hierarchisches Chunking

Dokumente werden anhand ihrer natürlichen Struktur (Überschriften, Abschnitte) in Chunks aufgeteilt. Jeder Chunk pflegt eine Eltern-Kind-Beziehung.

#### Chunk-Ebenen

- H1: Kapitel (oberste Ebene)
- H2: Abschnitte innerhalb von Kapiteln
- H3-H6: Unterabschnitte
- Content: Absätze und Textblöcke

#### Eltern-Kind-Beziehungen

Jeder Chunk speichert eine Referenz zu seinem Eltern-Chunk, was Baum-Traversierung bei der Suche ermöglicht.

### Zusammenfassungen als Hierarchie

Die zentrale Erkenntnis: Zusammenfassungen *sind* die Hierarchie. Jeder Knoten auf höherer Ebene enthält eine verdichtete Repräsentation seiner Kinder. Die Zusammenfassung ist nicht ein Label, sondern durchsuchbarer Inhalt mit eigenem Embedding.

### Embedding-Strategie

Jeder Chunk wird unabhängig eingebettet, aber die Suche nutzt die Hierarchie für kontextuelles Traversieren.

## System-Komponenten

### DocumentParser Trait

Abstraktion für verschiedene Dokumentformate.

#### MarkdownParser (implementiert)

Nutzt pulldown-cmark für Markdown-Parsing. Extrahiert Überschriften und Inhaltsblöcke mit Hierarchie.

#### Geplante Parser

- PdfParser: PDF-Dokumente
- HtmlParser: Webseiten
- CodeParser: Syntax-aware Chunking via tree-sitter

### VectorStore Trait

Abstraktion über Vektor-Datenbanken.

#### LanceDB (aktuell)

Serverless Vektor-Datenbank, lokal gespeichert.

- Kein Server erforderlich
- Schnelle lokale Queries
- Einfaches Deployment

Schema: id, content, embedding, level, parent_id, path, source_file, heading, cluster_memberships, is_summary, summarizes

#### Turso/Limbo (geplant)

SQLite-kompatibel, Pure Rust, Vector Search eingebaut. Ersetzt LanceDB als primäres Embedded-Backend.

#### PostgreSQL + pgvector (geplant)

Für Production-Deployments: skalierbar, Backup, HA.

### Embedder Trait

Abstraktion über Embedding-Modelle.

#### FastEmbed (implementiert)

ONNX Runtime für lokales CPU-Embedding. Standard: BAAI/bge-small-en-v1.5 (384 Dimensionen).

#### Geplante Embedder

- Ollama: GPU-beschleunigtes lokales Embedding
- OpenAI: Cloud-basierte Embedding-API

### Summarizer Trait

Abstraktion für Zusammenfassungs-Generierung.

#### OllamaSummarizer (implementiert)

Lokales LLM via Ollama REST API. Standard: llama3.2.

### Clusterer Trait

RAPTOR-Style Soft Clustering mit K-Means.

- Soft Assignments (ein Chunk kann zu mehreren Clustern gehören)
- Automatische Bestimmung der optimalen Cluster-Anzahl (Elbow-Methode)
- LLM-Zusammenfassungen pro Cluster

## Suchstrategie

### Top-Down-Kaskade

1. Query einbetten
2. In Wurzelknoten/Zusammenfassungen suchen
3. Beste Matches identifizieren
4. In Kinder der Matches abtauchen
5. Hierarchische Ergebnisse mit Kontext zurückgeben

### Subtree-Suche

Wenn bekannt ist, in welchem Abschnitt gesucht werden soll, kann direkt in einem Teilbaum gesucht werden.

## Geplante Konzepte

### Self-Describing Data (Visibility)

Daten konfigurieren selbst, wie sie behandelt werden:

- **Always** – Immer sichtbar, nie degradiert (Architekturentscheidungen)
- **Normal** – Standard-Kaskade, altert natürlich
- **DeepOnly** – Nur bei expliziter tiefer Suche
- **Expiring** – Selbstzerstörend nach Datum
- **Seasonal** – Zyklisch relevant, gesteuert über Zugriffshäufigkeit

### Memory Aging (RRD-Style)

Access-Tracking über feste Zeitfenster (inspiriert von RRDtool):

```
AccessProfile (40 Bytes):
[1min | 10min | 1h | 24h | 7d | 30d | total]
```

- Feste Buckets, konstanter Speicher
- Periodische Aggregation (fein → grob)
- Relevanzprofil zur Suchzeit wählbar

### Relationen

Schlanke Annotationen zwischen Chunks:

- `SupersededBy(id)` – Fakt wurde durch neuere Info ersetzt
- `SummarizedBy(id)` – Verdichtet in diesem Knoten
- `RelatedTo(id)` – Lose thematische Verbindung
- `DerivedFrom(id)` – Entstand aus dieser Diskussion

Bewusste Einschränkung: Maximal 1-2 Hops, keine Graph-Traversierung.

### Überlappende Bäume

Ein Datensatz kann in mehreren Bäumen gleichzeitig existieren (Thema, Zeit, Projekt, Person). Realisierung über mehrere Parent-Pointer pro Chunk.

## Deployment

### Single Binary

VecLayer kompiliert zu einer einzigen Binary mit allen Dependencies.

### Datenverzeichnis

Alle Daten in einem Verzeichnis (./veclayer-data/). Kopieren für Backup oder Deployment.

### Konfiguration

Environment-Variablen (12-Factor):

- `VECLAYER_DATA_DIR`: Datenverzeichnis
- `VECLAYER_EMBEDDER`: fastembed oder ollama
- `VECLAYER_OLLAMA_MODEL`: LLM-Modell für Summarization
- `VECLAYER_OLLAMA_URL`: Ollama-Endpoint
- `VECLAYER_PORT`: Server-Port
- `VECLAYER_HOST`: Server-Host
- `VECLAYER_SEARCH_TOP_K`: Anzahl Top-Level-Ergebnisse
- `VECLAYER_SEARCH_CHILDREN_K`: Anzahl Kinder pro Ergebnis
