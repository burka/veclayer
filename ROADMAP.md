# VecLayer Roadmap

## Wo wir stehen

VecLayer hat einen funktionierenden Prototyp mit ~8100 Zeilen Rust-Code:
- Hierarchisches Chunking + Summaries mit BGE-Embeddings (fastembed)
- Vektorsuche mit Recency-Boost + Visibility-Filtern
- RRD-Style Access-Tracking (hour/day/week/month/year Buckets)
- 5-Tool MCP-Interface: recall, focus, store, think, share
- Agent-konfigurierbare Aging-Regeln
- HTTP REST API + MCP stdio

## Alignment-Scorecard: Konzept vs. Code

| Konzept-Element | Status | Anmerkung |
|---|---|---|
| Hierarchie (Zusammenfassungen ueber Zusammenfassungen) | **Vorhanden** | Heading-Hierarchie + RAPTOR-Summaries |
| Memory Aging (RRD-Tracking, Visibility-Degradierung) | **Vorhanden** | Konfigurierbar, on-demand ausfuehrbar |
| 5-Tool MCP (recall, focus, store, think, share) | **Vorhanden** | Vollstaendig implementiert |
| Trait-Architektur (austauschbare Backends) | **Vorhanden** | DocumentParser, Embedder, VectorStore, Summarizer |
| Content-Hash IDs (SHA-256) | **Fehlt** | Aktuell UUID v4 |
| Entry-Typ (raw/summary/meta/impression) | **Fehlt** | Nur is_summary Boolean |
| Perspektiven (6 Defaults + Custom) | **Fehlt** | Kernkonzept nicht implementiert |
| Salienz (Dichte, Widersprueche, Spread) | **Fehlt** | Nur Zugriffsfrequenz |
| Identity-View + Priming | **Fehlt** | Nur statische MCP-Instructions |
| CLI-Spec (init, add, search, focus) | **Abweichend** | Aktuell: ingest, query, serve |
| VecLayer denkt nicht (kein LLM in Core) | **Verletzt** | Summarizer/Clustering in Core-Lib |
| TOML-Config + ENV Overrides | **Teilweise** | Nur ENV, kein TOML |
| Relationen als separate Tabelle | **Teilweise** | JSON in Entry, nicht normalisiert |
| Schlaf-Zyklus (optionaler Orchestrator) | **Fehlt** | |

## Code-Qualitaet: SRP/DRY Befund

### Gut
- Saubere Trait-Trennung (DocumentParser, Embedder, VectorStore, Summarizer)
- arc_impl! Macro eliminiert echten Boilerplate
- Umfangreiche Tests (~200+ Unit + Integration)
- Builder-Pattern auf HierarchicalChunk
- Offene Strings fuer Visibility/Relations (erweiterbar ohne Code-Aenderung)

### Handlungsbedarf
- **mcp/mod.rs (1076 Zeilen)** -- God Module. Aufteilen in types, tools, stdio, http
- **SearchConfig-Konstruktion** dupliziert zwischen commands.rs und mcp/mod.rs
- **Score-Berechnung** (vector * (1-alpha) + relevancy * alpha) dupliziert in search/search_subtree
- **Store/Embedder-Initialisierung** an 5+ Stellen wiederholt
- **chunk.rs (1300 Zeilen)** -- AccessProfile in eigenes Modul extrahieren
- **main.rs** -- Formatierungslogik gehoert in commands oder display-Modul

---

## Phasen

### Phase 1 -- Core (Entry + Storage) **DONE**

Den Prototyp auf das Konzept-Datenmodell umbauen.

- [x] **Entry-Struct** mit Content-Hash ID (`sha256(content)`, 7 Hex-Chars CLI)
  - Typ-Feld: `raw`, `summary`, `meta`, `impression`
  - is_summary Boolean ersetzt durch EntryType Enum
- [x] **CLI aligned mit Spec:**
  - `veclayer init` -- Speicher initialisieren
  - `veclayer add` (single entry, file, directory) -- ersetzt `ingest`
  - `veclayer search` -- ersetzt `query`
  - `veclayer focus` -- als CLI-Kommando (nicht nur MCP)
  - `veclayer status` -- ersetzt `stats`
- [x] **Summarizer/Clustering aus Core extrahieren**
  - Feature-Flag `llm` (default on) fuer reqwest, linfa, ndarray
  - Core-Lib: kein LLM noetig wenn `--no-default-features`
- [x] **TOML-Config** im Data-Dir + ENV Overrides (ENV > Config > Defaults)
  - Discover: $VECLAYER_CONFIG > <data_dir>/veclayer.toml > ./veclayer.toml
- [x] **SRP-Refactoring:**
  - mcp/mod.rs (1076 Zeilen) aufgeteilt in types, tools, stdio, http
  - AccessProfile in eigenes Modul (access_profile.rs)
  - SearchConfig::for_query() + blend_score() dedupliziert
  - Formatierungslogik aus main.rs nach commands.rs extrahiert

### Phase 2 -- Perspektiven + Relationen **DONE**

Das Kernkonzept implementieren: verschiedene Sichten auf die gleichen Daten.

- [x] **Perspektiven-Modell:**
  - Perspektiven CRUD (create, list, remove) mit JSON-Persistenz
  - 6 Default-Perspektiven mit Hints
  - Validierung bei `add` (fail fast am CLI-Rand)
- [x] **Relationen normalisieren:**
  - SupersededBy, SummarizedBy, VersionOf, RelatedTo, DerivedFrom
  - `add --summarizes`, `--supersedes`, `--version-of`
- [x] **Facettierte Suche:**
  - `search --perspective decisions "Backend"`
  - VectorStore::search_by_perspective + LanceDB Impl
- [x] **CLI:**
  - `veclayer p` (Perspektiven verwalten)
  - `veclayer history` (Versionsgeschichte eines Entries)
  - `veclayer archive` (Gezieltes Archivieren)

### Phase 3 -- Memory Aging + Salienz **DONE**

Von Zugriffsfrequenz zu echtem Bedeutungs-Ranking.

- [x] **Salienz-Berechnung** aus bestehenden Signalen:
  - Interaktionsdichte (Access-Profile relevancy_score)
  - Wirkungsbreite (Perspektiven-Spread: perspectives.len / 8)
  - Revisions-Events (Relations-Count mit tanh-Saettigung)
  - Gewichteter Komposit-Score (0.5 interaction + 0.25 perspective + 0.25 revision)
- [x] **Ranking:** `semantic_similarity * (1-alpha) + (recency * (1-sw) + salience * sw) * alpha`
  - salience_weight in SearchConfig (default: 0.3)
  - Nahtlose Integration in blend_score()
- [x] **compact-Kommando:**
  - `veclayer compact rotate` — Access-Profile rollen + Aging ausfuehren
  - `veclayer compact salience` — Top-N Salienz-Report
  - `veclayer compact archive-candidates` — Archivierungs-Vorschlaege (low salience)
- [x] **Visibility-Erweiterung:**
  - Automatische Degradierung prueft Salienz: high-salience Entries ueberleben Aging
  - `salience_protection` Schwelle in AgingConfig (default: 0.15)
- [x] **MCP Integration:**
  - `think(action='salience')` — Salienz-Report via MCP
  - reflect-Report zeigt Salienz-Scores fuer Hot und Stale Chunks

### Phase 4 -- Identity + Reflect **DONE**

Aus dem Gedaechtnis eine Identitaet emergieren lassen.

- [x] **Identity-Cluster:**
  - Salienz-gewichtete Embedding-Centroids pro Perspektive
  - Berechnet ohne LLM: Vektor-Durchschnitt gewichtet mit Salienz-Score
- [x] **Open Threads:**
  - Superseded-but-visible: Entries die ersetzt wurden aber noch sichtbar sind
  - High-relation-count: Entries mit 3+ Relationen → aktive Deliberation
  - Deduplizierung bei mehrfachen Kriterien
- [x] **Reflect** (read-only Material-Aufbereitung):
  - Core Knowledge: Top-15 salienteste Entries
  - Open Threads: Ungeloeste Widersprueche
  - Recent Learnings: Entries aus "learnings"-Perspektive
  - Perspective Coverage: Centroids mit Entry-Counts und Avg-Salienz
- [x] **CLI:**
  - `veclayer id` — Kompakte Identitaets-Zusammenfassung
  - `veclayer reflect` — Umfassender Priming-Report
  - `veclayer` (kein Argument) — Orientierung: Stats, Core, Threads, Hints
- [x] **Priming beim Connect:**
  - MCP initialize liefert dynamisches Priming (statische Instruktionen + Identity Briefing)
  - Priming enthalt Core Knowledge, Open Threads, Recent Learnings, Perspective Coverage

### Phase 5 -- Think (optional, braucht LLM)

Der optionale Orchestrator.

- [ ] **LLMProvider Trait** (OpenAI, Anthropic, Ollama)
- [ ] **think-Kommando:**
  - reflect -> LLM -> add -> compact Zyklus
  - Narrativ-Generierung
  - Meta-Learning-Destillation
- [ ] **Schlaf-Zyklus:**
  - Automatische Reflexion + Konsolidierung
  - Konfigurierbar (Intervall, Tiefe)
- [ ] **CLI:**
  - `veclayer think`
  - `veclayer config llm.*`

### Phase 6 -- Server + Sharing

MCP-Server mit Identitaet und UCAN-basiertem Sharing.

- [ ] **MCP Server** mit personalisierten Tool-Descriptions
  - Descriptions abhaengig vom Agent-Kontext
- [ ] **REST API** (vorhanden, anpassen an neues Datenmodell)
- [ ] **Priming beim Connect:**
  - Narrativ, Threads, Learnings als Startup-Briefing
- [ ] **UCAN Auth:**
  - Keypair-Generierung, DID
  - Delegierbare, attenuierbare Capability Tokens
- [ ] **CLI:**
  - `veclayer serve`
  - `veclayer share` / `connect` / `revoke`

### Phase 7 -- Polish

- [ ] Alias-Support (`store`=`add`, `s`=`search`, `f`=`focus`, etc.)
- [ ] Kontextuelle Hilfen in jeder Ausgabe (naechste Aktionen)
- [ ] Summary-ueber-Summary Vorschlaege bei vielen Kindern
- [ ] Error Messages mit Hilfe
- [ ] Completion Scripts (bash, zsh, fish)
- [ ] Multi-Format Parsing (PDF, HTML, Code via tree-sitter)
- [ ] Alternative Backends (Turso/SQLite, PostgreSQL+pgvector)

---

## Was NICHT implementiert wird

Aus der Entstehungsgeschichte -- explizit verworfene Ansaetze:

- Keine JSON-Annotations an Entries
- Keine Pfade als einzige Struktur
- Keine Tags
- Keine separaten Vektorraeurme fuer Emotionen
- Keine S3-Backends
- Keine ACLs (nur UCAN)
- Keine Bearer-Tokens
- Keine statischen Tool-Descriptions
- Keine Leaf/Node-Trennung (alles ist ein Entry)
- Keine "Baeume" als Konzept (Perspektiven stattdessen)

## Erfolgskriterien

Ein Release ist vision-aligned wenn:
1. Ein Agent startet mit personalisiertem Identity-Priming
2. Suche ist facettiert ueber mehrere Perspektiven
3. Focus kann von Summary-Entries zu Raw-Entries mit Revisionsgeschichte absteigen
4. Think kann konsolidieren, Widersprueche erkennen, Summaries weiterentwickeln
5. Share nutzt delegierbare kryptographische Capabilities
6. VecLayer Core enthaelt kein LLM -- nur Embeddings, Struktur, Rechnung
