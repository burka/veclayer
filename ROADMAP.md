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

### Phase 1 -- Core (Entry + Storage)

Den Prototyp auf das Konzept-Datenmodell umbauen.

- [ ] **Entry-Struct** mit Content-Hash ID (`sha256(content)`, 7 Hex-Chars CLI)
  - Typ-Feld: `raw`, `summary`, `meta`, `impression`
  - Ersetze HierarchicalChunk und is_summary Boolean
  - Migrationslogik fuer bestehende Daten
- [ ] **Storage Trait** anpassen an Entry statt HierarchicalChunk
  - LanceDB-Implementierung aktualisieren
  - Relationen als separate Tabelle `relations(source_id, target_id, relation_type)`
- [ ] **Embedding Trait** bleibt, fastembed als Default
- [ ] **CLI aligned mit Spec:**
  - `veclayer init` -- Speicher initialisieren
  - `veclayer add` (single entry, file, directory) -- ersetzt `ingest`
  - `veclayer search` -- ersetzt `query`
  - `veclayer focus` -- als CLI-Kommando (nicht nur MCP)
  - `veclayer status` -- ersetzt `stats`
  - `veclayer help`
- [ ] **Summarizer/Clustering aus Core extrahieren**
  - In separates `veclayer-orchestrator` Crate oder Feature-Flag
  - Core-Lib: kein LLM, nur Embeddings + Struktur + Rechnung
- [ ] **TOML-Config** im Data-Dir + ENV Overrides (Reihenfolge: ENV > Config > Defaults)
- [ ] **SRP-Refactoring:**
  - mcp/mod.rs aufteilen (types, tools, stdio, http)
  - AccessProfile in eigenes Modul
  - SearchConfig-Konstruktion deduplizieren
  - Formatierungslogik aus main.rs extrahieren

### Phase 2 -- Perspektiven + Relationen

Das Kernkonzept implementieren: verschiedene Sichten auf die gleichen Daten.

- [ ] **Perspektiven-Modell:**
  - Perspektiven CRUD (create, list, remove)
  - 6 Default-Perspektiven mit Hints:
    - `intentions` -- "Absichten, Ziele, Vorhaben"
    - `people` -- "Personen, Beziehungen, Rollen"
    - `temporal` -- "Zeitverlauf, Entwicklungen, Chronologie"
    - `knowledge` -- "Dauerhaftes Fachwissen, Definitionen"
    - `decisions` -- "Entscheidungen, Abwaegungen, Trade-offs"
    - `learnings` -- "Erkenntnisse, Fehler, Lessons Learned"
- [ ] **Relationen normalisieren:**
  - SupersededBy, SummarizedBy, VersionOf, RelatedTo, DerivedFrom
  - `add --summarizes`, `--supersedes`, `--version-of`
- [ ] **Facettierte Suche:**
  - Ergebnisse gruppiert nach Perspektive
  - `search --perspective decisions "Backend"`
- [ ] **CLI:**
  - `veclayer p` (Perspektiven verwalten)
  - `veclayer history` (Versionsgeschichte eines Entries)
  - `veclayer archive` (Gezieltes Archivieren)

### Phase 3 -- Memory Aging + Salienz

Von Zugriffsfrequenz zu echtem Bedeutungs-Ranking.

- [ ] **Salienz-Berechnung** aus bestehenden Signalen:
  - Interaktionsdichte (Access-Profile)
  - Widersprueche (Entries die sich widersprechen)
  - Wirkungsbreite (In wie vielen Perspektiven referenziert)
  - Revisions-Events (Wie oft superseded/updated)
- [ ] **Ranking:** `semantic_similarity x recency x salience`
- [ ] **compact-Kommando:**
  - rotate: Access-Profile rollen
  - salience: Salienz neu berechnen
  - archive candidates: Vorschlaege fuer Archivierung
- [ ] **Visibility-Erweiterung:**
  - `always`, `normal`, `deep_only`, `expiring`, `seasonal` bleiben
  - Automatische Degradierung basiert auf Salienz, nicht nur Alter

### Phase 4 -- Identity + Reflect

Aus dem Gedaechtnis eine Identitaet emergieren lassen.

- [ ] **Identity-Cluster:**
  - Embedding-Centroids gewichtet nach Salienz
  - Berechenbar ohne LLM
- [ ] **Open Threads:**
  - Ungeloeste Widersprueche, offene Entscheidungen
  - Berechenbar aus Relationen + Salienz
- [ ] **Reflect** (read-only Material-Aufbereitung):
  - Hot Chunks, Stale Chunks, Identity-Snapshot
  - Input fuer LLM-Reflexion, aber selbst kein LLM
- [ ] **CLI:**
  - `veclayer id` -- Identitaets-Zusammenfassung
  - `veclayer reflect` -- Material aufbereiten
  - `veclayer` (kein Argument) -- Orientierung: "Wer bin ich, was beschaeftigt mich"
- [ ] **Priming beim Connect:**
  - Identity-Narrativ + offene Faeden + aktuelle Learnings
  - Generiert aus Cluster + Hot + Unresolved

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
