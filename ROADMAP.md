# VecLayer Roadmap

## Wo wir stehen

**Phasen 1-4 sind abgeschlossen.** Das gesamte mechanische Fundament steht -- alles was VecLayer ohne LLM leisten soll, funktioniert:

- Entry-Modell mit SHA-256 Content-Hash, vier Typen, typisierten Relationen
- Sechs Default-Perspektiven + Custom, facettierte Suche, Validierung
- Salienz-Komposit (Interaktion, Perspektiven-Spread, Revisions-Aktivitaet) mit Aging-Schutz
- Identitaet aus salienz-gewichteten Embedding-Centroids, Open Threads, Learnings
- Dynamisches Priming beim MCP-Connect
- LLM aus Core extrahiert (Feature-Flag), TOML + ENV Konfiguration
- 13 CLI-Kommandos, MCP stdio + HTTP

**Naechster Schritt:** Phase 5 (Think) -- der optionale Orchestrator, der Selbstreflexion ermoeglicht. Plus eine offene Konzeptfrage: echte Widerspruchserkennung vs. der aktuelle Proxy.

## Alignment-Scorecard: Konzept vs. Code

| Konzept-Element | Status | Anmerkung |
|---|---|---|
| Hierarchie (Zusammenfassungen ueber Zusammenfassungen) | **Vorhanden** | Heading-Hierarchie + RAPTOR-Summaries |
| Memory Aging (RRD-Tracking, Visibility-Degradierung) | **Vorhanden** | Konfigurierbar, Salienz-Schutz beim Aging |
| 5-Tool MCP (recall, focus, store, think, share) | **Vorhanden** | Vollstaendig implementiert |
| Trait-Architektur (austauschbare Backends) | **Vorhanden** | DocumentParser, Embedder, VectorStore, Summarizer |
| Content-Hash IDs (SHA-256) | **Vorhanden** | 64-Char Hex, 7-Char Short-ID wie git (Phase 1) |
| Entry-Typ (raw/summary/meta/impression) | **Vorhanden** | EntryType Enum mit vier Varianten (Phase 1) |
| Perspektiven (6 Defaults + Custom) | **Vorhanden** | CRUD, Hints, facettierte Suche, Validierung (Phase 2) |
| Salienz (Dichte, Spread, Revisionen) | **Vorhanden** | Komposit-Score, Ranking-Integration, Aging-Schutz (Phase 3) |
| Identity-View + Priming | **Vorhanden** | Centroids, Open Threads, Learnings, dynamisches MCP-Priming (Phase 4) |
| CLI-Spec (init, add, search, focus) | **Vorhanden** | 13 Kommandos, aligned mit Spec (Phase 1) |
| VecLayer denkt nicht (kein LLM in Core) | **Vorhanden** | Feature-Flag `llm`, Core kompiliert ohne LLM (Phase 1) |
| TOML-Config + ENV Overrides | **Vorhanden** | 12-Factor: ENV > TOML > Defaults, 3-Stufen-Discovery (Phase 1) |
| Relationen typisiert | **Vorhanden** | SupersededBy, SummarizedBy, VersionOf, RelatedTo, DerivedFrom (Phase 2) |
| Widerspruchserkennung (semantisch) | **Offen** | Revisions-Aktivitaet als Proxy; echte Erkennung noch zu konzipieren |
| Schlaf-Zyklus (optionaler Orchestrator) | **Fehlt** | Phase 5 |
| UCAN Auth + Sharing | **Fehlt** | Phase 6 |

## Code-Qualitaet: SRP/DRY Befund

### Gut
- Saubere Trait-Trennung (DocumentParser, Embedder, VectorStore, Summarizer)
- arc_impl! Macro eliminiert echten Boilerplate
- Umfangreiche Tests (~200+ Unit + Integration)
- Builder-Pattern auf HierarchicalChunk
- Offene Strings fuer Visibility/Relations (erweiterbar ohne Code-Aenderung)

### In Phase 1 behoben
- ~~mcp/mod.rs (1076 Zeilen)~~ → aufgeteilt in types.rs, tools.rs, stdio.rs, http.rs
- ~~SearchConfig-Konstruktion dupliziert~~ → SearchConfig::for_query() + blend_score() dedupliziert
- ~~chunk.rs (1300 Zeilen)~~ → AccessProfile in eigenes Modul (access_profile.rs)
- ~~main.rs Formatierungslogik~~ → nach commands.rs extrahiert

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

### Offene Konzeptfragen

Themen, die noch durchdacht werden muessen bevor sie in eine Phase muenden.

#### Widerspruchserkennung: Proxy vs. echte Semantik

**Status:** Revisions-Aktivitaet (Anzahl Relationen) dient aktuell als Proxy fuer Widersprueche in der Salienz-Berechnung. Das funktioniert: viele Relationen deuten auf Ueberarbeitungen, konkurrierende Versionen oder aktive Auseinandersetzung hin. Aber es ist kein echtes Erkennen von semantischen Spannungen.

**Die Frage:** Soll VecLayer Widersprueche zwischen Entries erkennen koennen -- also Paare von Entries identifizieren, die inhaltlich in Spannung zueinander stehen? Und wenn ja: mechanisch (ohne LLM) oder als Input fuer den Schlaf-Zyklus (mit LLM)?

**Moegliche Ansaetze (noch nicht bewertet):**
- **Embedding-Distanz + Relationstyp:** Entries die semantisch nah sind aber via SupersededBy oder VersionOf verbunden → wahrscheinlich Widerspruch/Revision
- **Negation-Heuristik:** Gleiche Embedding-Nachbarschaft, aber entgegengesetzte Aussagen (schwer ohne LLM)
- **LLM-gestuetzt im Schlaf-Zyklus:** reflect liefert Kandidaten-Paare, LLM bewertet ob echter Widerspruch
- **Manuell:** Agent oder Mensch markiert Widersprueche explizit via Relation

**Designprinzip:** VecLayer denkt nicht. Wenn Widerspruchserkennung ein LLM braucht, gehoert sie in den Schlaf-Zyklus (Phase 5), nicht in Core. Mechanische Heuristiken (Embedding-Distanz + Relationstyp) koennten als Kandidaten-Filter in Core leben.

---

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
