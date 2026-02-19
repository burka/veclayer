# VecLayer

**Hierarchische Vektor-Datenbank mit Gedächtnis – persistente Identität für KI-Agenten.**

> **Status:** Experimental – Prototyp, APIs können sich ändern
> **Autor:** Florian Schmidt, entwickelt im Dialog mit Claude

## Was ist VecLayer?

VecLayer ist eine hierarchische Vektor-Datenbank, die Wissen in Bäumen organisiert statt in flachen Chunk-Listen. Höhere Knoten im Baum enthalten automatisch generierte Zusammenfassungen der darunterliegenden Inhalte. Dadurch kann eine Suche zuerst den Überblick liefern und bei Bedarf in die Tiefe gehen – wie ein Mensch, der sich erst an das Thema erinnert und dann an die Details.

Das zentrale Ziel: **Session-übergreifende Identität für KI-Agenten ermöglichen.** Nicht als flache Faktensammlung, sondern als strukturiertes, alterndes, selbstbeschreibendes Gedächtnis.

VecLayer kombiniert dafür drei Konzepte:

1. **Hierarchische Wissensorganisation** – Zusammenfassungen *sind* die Hierarchie, nicht ein Extra-Feature
2. **Self-Describing Data** – Daten konfigurieren selbst, wie wichtig sie sind und wie sie altern
3. **Memory Aging** – Zugriffsmuster bestimmen Relevanz, Wichtiges bleibt präsent, Unwichtiges verblasst

VecLayer ist als Rust-Bibliothek mit MCP-Server-Interface konzipiert. Single Binary + Datenverzeichnis, kein Setup.

## Warum Identität?

Aktuelle KI-Systeme haben ein Gedächtnis-Problem:

- Jede Session startet als Tabula rasa – oder mit ein paar flachen Fakten
- Es gibt kein Gefühl für "wichtig" vs. "Rauschen"
- Nichts altert. Veraltete Information steht gleichberechtigt neben aktueller
- Ein Agent kann seine Erinnerungen nicht aktiv pflegen oder reflektieren

Identität entsteht aus dem Zusammenspiel von:

| Dimension | VecLayer-Feature |
|-----------|-----------------|
| Akkumuliertes Wissen | Hierarchische Bäume mit Zusammenfassungen |
| Was ist wichtig? | Visibility (Always / Normal / DeepOnly) |
| Was ist aktuell? | Memory Aging (Access Tracking) |
| Wie hat sich mein Wissen entwickelt? | Relations (SupersededBy) |
| Aktive Selbstpflege | Reflexions-Pattern (extern getriggert) |

## Vision-Zielbild (abgleichbar)

VecLayer zielt auf ein agentenzentriertes Langzeitgedächtnis mit fünf Kernfähigkeiten:

1. **recall** – relevante Erinnerungen facettiert finden
2. **focus** – aus Zusammenfassung in Evidenz-Details absteigen
3. **store** – neue Erkenntnisse dauerhaft ablegen und verorten
4. **think** – Widersprüche/Konsolidierung aktiv kuratieren
5. **share** – gezielt und nachvollziehbar Berechtigungen teilen

Die aktuelle Implementierung ist auf diese API ausgerichtet; die verbleibenden Gaps sind in `Roadmap.md` und `Tasks.md` explizit nachverfolgbar.

## Aktueller Stand

### Implementiert (v0.1 – Prototyp)
- [x] Markdown-Parsing mit Heading-Hierarchie (pulldown-cmark)
- [x] Deterministische lokale Embeddings (offline, ohne Model-Download)
- [x] LanceDB-Backend (serverless, kein Setup)
- [x] RAPTOR-Style Soft Clustering + LLM-Summaries (via Ollama)
- [x] Hierarchische Suche mit Kontext-Aufbau
- [x] MCP-Server für KI-Assistenten-Integration (HTTP + stdio)
- [x] CLI: ingest, query, serve, stats, sources
- [x] Trait-basierte Architektur (DocumentParser, Embedder, VectorStore, Summarizer)
- [x] 12-Factor-Konfiguration via Environment-Variablen

### Datenmodell vorhanden (v0.2 – in Arbeit)
- [x] **Visibility** – Offene Strings mit bekannten Werten (always, normal, deep_only, expiring, seasonal) + beliebige eigene
- [x] **Relations** – Offene Strings (superseded_by, summarized_by, related_to, derived_from) + beliebige eigene
- [x] **AccessProfile** – RRD-Style Buckets (hour, day, week, month, year, total)
- [x] **Expiry** – Selbstzerstörende Daten mit Zeitstempel
- [x] Visibility-Filter in der Suche (`--deep` für alle, Standard filtert deep_only/expired)
- [x] MCP-Tools: recall, focus, store, think, share (stdio + HTTP API)
- [x] Access-Tracking bei Suchergebnissen (RRD-Buckets + Relevancy Scoring)
- [x] Ingest mit `--visibility` Flag
- [x] Recency-Suche mit `--recent 24h/7d/30d`

### Geplant
- [ ] Dynamisches Agent-Priming (Identitätsnarrativ, offene Threads, Meta-Learnings)
- [ ] Überlappende Bäume (Multi-Dimension: Thema × Zeit × Projekt)
- [ ] Turso/Limbo als Embedded-Backend (SQLite-kompatibel, Pure Rust)
- [ ] PostgreSQL + pgvector Backend (Production)
- [ ] Multi-Format-Parsing (PDF, HTML, Code via tree-sitter)
- [ ] Periodisches Auto-Aging (cron/scheduled)

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

### Self-Describing Data

Jeder Chunk trägt seine eigene Sichtbarkeit als offenen String. Bekannte Werte als Konstanten, aber erweiterbar ohne Code-Änderung:

```rust
// Bekannte Werte (visibility::ALWAYS, visibility::NORMAL, ...)
chunk.with_visibility("always")     // Kernwissen, nie degradiert
chunk.with_visibility("normal")     // Standard, altert natürlich
chunk.with_visibility("deep_only")  // Nur bei expliziter tiefer Suche
chunk.with_visibility("expiring")   // Selbstzerstörend nach Zeitstempel

// Eigene Werte -- funktionieren ohne Code-Änderung
chunk.with_visibility("draft")      // Eigene Kategorie
chunk.with_visibility("archived")   // Eigene Kategorie
```

Standard-Suche zeigt `always`, `normal`, `seasonal`, `expiring` (nicht abgelaufen). Alles andere nur bei tiefer Suche. Welche Werte in welchem Suchmodus sichtbar sind, ist konfigurierbar.

### Relationen

Schlanke, gerichtete Verbindungen. Auch hier: `kind` ist ein offener String.

```rust
// Bekannte Werte
ChunkRelation::superseded_by("newer-fact-id")
ChunkRelation::summarized_by("summary-id")
ChunkRelation::related_to("other-id")
ChunkRelation::derived_from("source-id")

// Eigene Relationen
ChunkRelation::new("contradicts", "other-id")
ChunkRelation::new("blocks", "issue-id")
ChunkRelation::new("inspired_by", "source-id")
```

Bewusste Einschränkung: Max 1-2 Hops, keine Graph-Traversierung.

### Soft Clustering

Jeder Chunk kann mit unterschiedlicher Wahrscheinlichkeit zu mehreren Clustern gehören. Für jeden Cluster generiert ein LLM eine Zusammenfassung, die selbst durchsuchbar wird.

## vs Flat Chunking / Flat Memory

| Aspekt | Flat Chunking | Flat Memory (Claude/GPT) | VecLayer |
|--------|---------------|-------------------------|----------|
| Struktur | Fixed-size Windows | Key-Value-Paare | Heading-aware Hierarchie |
| Kontext | Verloren | Kein Zusammenhang | Bewahrt via Parent-Links |
| Wichtigkeit | Alles gleich | Alles gleich | Visibility (Always → DeepOnly) |
| Zeitliche Relevanz | Keine | Keine | RRD Access Tracking + Aging |
| Wissensevolution | Keine | Überschreiben | SupersededBy-Relations |
| Selbstreflexion | Keine | Keine | Reflexions-Pattern |

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

```bash
veclayer ingest ./docs                    # Rekursiv + Summarization (Standard)
veclayer ingest ./docs --no-summarize     # Ohne Clustering/Summaries (schneller)
veclayer ingest ./docs --no-recursive     # Nur ein Verzeichnis
veclayer ingest ./docs --model tinyllama  # Anderes Ollama-Modell
veclayer ingest ./docs --visibility always  # Alle Chunks als "always" markieren
```

### `veclayer query <QUERY>`

```bash
veclayer query "memory safety"            # Standard-Suche
veclayer query "memory safety" -k 10      # Top 10 Ergebnisse
veclayer query "memory safety" -p         # Hierarchie-Pfad anzeigen
veclayer query "auth" --subtree chunk_id  # Innerhalb eines Teilbaums suchen
veclayer query "auth" --deep              # Alle Visibilities durchsuchen
veclayer query "auth" --recent 24h        # Favorisiert kürzlich zugegriffene Chunks
veclayer query "auth" --recent 7d         # Wochenfenster
veclayer query "auth" --recent 30d        # Monatsfenster
```

### `veclayer serve`

```bash
veclayer serve                # HTTP-Server (Standard)
veclayer serve --mcp-stdio    # Stdio-Transport (für Claude Desktop)
veclayer serve --read-only    # Production-Modus
```

### MCP Tools (via `--mcp-stdio` oder HTTP API)

| Tool | Beschreibung |
|------|-------------|
| `recall` | Semantische Erinnerungssuche mit Recency-Optionen |
| `focus` | Vertiefung ab Node/Chunk mit optionaler Frage-Linse |
| `store` | Speichert neue Erinnerung als Chunk inkl. optionaler Einordnung |
| `think` | Reflexionsbericht (hot/stale + Kuratierungshinweise) |
| `share` | Erzeugt Scope-/Capability-Token-Payload für Sharing-Workflows |

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
VECLAYER_EMBEDDER=fastembed-local        # oder ollama
VECLAYER_OLLAMA_MODEL=llama3.2
VECLAYER_OLLAMA_URL=http://localhost:11434
VECLAYER_PORT=8080
VECLAYER_HOST=127.0.0.1
VECLAYER_SEARCH_TOP_K=5
VECLAYER_SEARCH_CHILDREN_K=3
```

## Roadmap

### Phase 1: Identity-Grundlagen nutzbar machen (v0.2) ✓

Das Datenmodell ist vorhanden und in Suche, MCP und CLI nutzbar:

- ✓ **Visibility-Filter in der Suche** – Standard-Suche schließt DeepOnly und abgelaufene Chunks aus; `--deep` durchsucht alles
- ✓ **RRD-Style Access-Tracking** – Jeder Suchtreffer aktualisiert Buckets (hour/day/week/month/year/total) + Relevancy Scoring
- ✓ **MCP: promote/demote** – Visibility über MCP-Tools und HTTP API ändern
- ✓ **MCP: relate** – Relationen über MCP-Tools und HTTP API hinzufügen
- ✓ **Ingest mit Visibility** – `veclayer ingest --visibility always ./core-docs`
- ✓ **Recency-Suche** – `--recent 24h/7d/30d` für zeitgewichtete Relevanz

### Phase 2: Agent-Driven Memory Management (v0.3) ✓

Der Agent kuratiert seinen eigenen Speicher über MCP-Tools:

- ✓ **reflect** – Reflexionsbericht: Hot Chunks, stale Chunks, empfohlene Aktionen
- ✓ **ingest_chunk** – Agent schreibt Chunks direkt (Beobachtungen, Summaries, Entscheidungen)
- ✓ **configure_aging** – Agent setzt Degradierungsregeln (z.B. "normal → deep_only nach 30 Tagen")
- ✓ **apply_aging** – Aging-Regeln ausführen, stale Chunks automatisch degradieren
- ✓ **MCP Instructions** – Ausführliche Einführung beim Connect: Was VecLayer ist, Reflexions-Pattern, Summarization-Pattern
- Zusammenfassungen werden vom Agent erstellt (search → synthesize → ingest_chunk + relate)

### Phase 3: Erweitertes Memory Aging (v0.4)

Grundlagen implementiert (configure_aging + apply_aging). Geplant:

- Automatische periodische Aging-Ausführung (z.B. cron-getriggert)
- Agent-gesteuerte Reflexion als Routine: "Was war diese Woche relevant?"

### Phase 4: Überlappende Bäume (v0.5)

Ein Datensatz kann in mehreren Bäumen gleichzeitig existieren – gruppiert nach Thema, Zeit, Projekt, Person. Mehrere Parent-Pointer pro Chunk.

### Phase 5: Backends & Parsing (v0.6+)

- **Turso/Limbo** als Embedded-Backend (SQLite-kompatibel, Pure Rust)
- **PostgreSQL + pgvector** für Production
- **Multi-Format Parsing** (PDF, HTML, Code via tree-sitter)

### Reflexions-Pattern (Agent-Ebene) ✓

Implementiert als `reflect` MCP-Tool + Instructions. Der Agent wird beim Connect instruiert:

- `reflect` aufrufen, wenn Zeit ist (Session-Start, Idle-Time, Session-Ende)
- Hot Chunks reviewen: Sind sie noch aktuell? Brauchen sie ein Update?
- Stale Chunks degradieren: `apply_aging` oder manuell `demote`
- Summaries erstellen: Verwandte Chunks lesen → Zusammenfassung → `ingest_chunk`
- Kernwissen promoten: Wichtige Entscheidungen, Präferenzen → `promote` zu 'always'

## Technischer Stack

| Komponente | Technologie | Begründung |
|---|---|---|
| Sprache | Rust | Performance, Memory Safety, Single Binary |
| Embeddings | fastembed (CPU) | Trait-basiert, austauschbar |
| Summarization | Ollama (lokal) | Lokales LLM, kein Cloud-Dependency |
| Vektor-Suche | LanceDB (Prototyp) | File-basiert, einfach |
| Parsing | pulldown-cmark | Markdown → Heading-Hierarchie direkt nutzbar |
| MCP Server | axum | Direkte Integration in Claude, andere Agenten |
| CLI | clap | `ingest`, `query`, `serve`, `stats`, `sources` + MCP tools |

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
3. **Naming (Nov 2025):** Aus Kandidaten wie Hive, Strata, Hierav wurde "VecLayer".
4. **Memory-Aging-Erweiterung (Feb 2026):** Aus der Frage "Wie könnte ein KI-Agent seine Erinnerungen managen?" entstand das RRD-inspirierte Access-Tracking, Self-Describing Data, und das Meditations-Pattern. Die Erkenntnis: Zusammenfassungen *sind* die Hierarchie.
5. **Identitäts-Framing (Feb 2026):** VecLayer ist nicht "besseres RAG" – es ist ein **persistenter Identitätsspeicher für KI-Agenten**. Die Kombination aus Hierarchie, Visibility, Aging und Reflexion ermöglicht session-übergreifende Identität.

## Lizenz

MIT
