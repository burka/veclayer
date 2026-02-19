# VecLayer

**Langzeitgedaechtnis fuer KI-Agenten. Hierarchisch, perspektivisch, alterndes Wissen.**

> Status: In Development -- Prototyp, APIs koennen sich aendern
> Autor: Florian Schmidt, entwickelt im Dialog mit Claude

## Was ist VecLayer?

VecLayer organisiert Wissen als Hierarchie: Zusammenfassungen ueber Zusammenfassungen, in beliebiger Tiefe, aus verschiedenen Perspektiven auf die gleichen Rohdaten. Eine Suche beginnt beim Ueberblick und geht bei Bedarf in die Tiefe -- wie menschliches Erinnern.

Statt flacher Chunk-Listen oder Key-Value-Speicher bietet VecLayer strukturiertes, alterndes, selbstbeschreibendes Gedaechtnis. Aus der statistischen Form aller Erinnerungen -- Embedding-Cluster gewichtet nach Salienz -- waechst organisch eine Identitaet.

## Warum

Aktuelle KI-Systeme haben ein Gedaechtnis-Problem:

- Jede Session startet als Tabula rasa -- oder mit flachen Fakten
- Flaches RAG verliert Struktur: ein 200-seitiges Dokument wird zu gleichwertigen Fragmenten
- Es gibt kein Gefuehl fuer "wichtig" vs. "Rauschen"
- Nichts altert. Veraltete Information steht gleichberechtigt neben aktueller
- Ein Agent kann seine Erinnerungen nicht aktiv pflegen oder reflektieren

VecLayer loest das durch drei Konzepte:

1. **Hierarchische Wissensorganisation** -- Zusammenfassungen *sind* die Hierarchie
2. **Perspektiven** -- Sechs Default-Perspektiven strukturieren das Wissen; jederzeit erweiterbar
3. **Memory Aging + Salienz** -- Nicht die haeufigsten, sondern die bedeutsamsten Erinnerungen kommen nach oben

Identitaet entsteht aus dem Zusammenspiel: Beim Verbinden erhaelt der Agent ein Priming -- wer bin ich, was beschaeftigt mich, worauf sollte ich achten.

## Architektur-Ueberblick

### Ein Primitiv: Entry

Alles ist ein Entry. Keine Leaf/Node-Trennung. Ein Typ-Feld klassifiziert:

| Typ | Bedeutung |
|-----|-----------|
| `raw` | Originaldaten, unveraendert |
| `summary` | Zusammenfassung von Kindern |
| `meta` | Reflexion, Bewertung |
| `impression` | Spontane Beobachtung |

ID = `sha256(content)` -- erste 7 Hex-Chars in der CLI (wie git). Identischer Content = identische ID = idempotent.

### Perspektiven, nicht Baeume

Sechs Default-Perspektiven strukturieren das Wissen:

| Perspektive | Was sie erfasst |
|-------------|----------------|
| `intentions` | Absichten, Ziele, Vorhaben |
| `people` | Personen, Beziehungen, Rollen |
| `temporal` | Zeitverlauf, Entwicklungen |
| `knowledge` | Dauerhaftes Fachwissen |
| `decisions` | Entscheidungen, Abwaegungen |
| `learnings` | Erkenntnisse, Lessons Learned |

Jederzeit erweiterbar mit eigenen Perspektiven. Jede Perspektive hat Hints, die dem LLM zeigen, welches Material relevant ist.

### Memory Aging

RRD-inspiriertes Access-Tracking mit festen Zeitfenstern:

```
hour | day | week | month | year | total
```

Zugriffsmuster bestimmen Sichtbarkeit. Wichtiges bleibt praesent, Ungenutztes verblasst kontrolliert. Konfigurierbare Degradierungs-Regeln.

### Salienz

Salienz != Haeufigkeit. Gemessen an:
- Interaktionsdichte (wie oft zugegriffen)
- Widerspruechen (kollidiert mit anderem Wissen)
- Wirkungsbreite (in wie vielen Perspektiven relevant)

Ranking: `semantic_similarity x recency x salience`

### Identitaet

Aus Embedding-Clustern gewichtet nach Salienz waechst eine Identitaet. Beim Connect erhaelt der Agent ein Priming: Identity-Narrativ, offene Faeden, aktuelle Learnings.

## Was VecLayer macht vs. was das LLM macht

| VecLayer (speichert, strukturiert, rechnet) | LLM (sucht, vertieft, reflektiert) |
|---------------------------------------------|-------------------------------------|
| Entries speichern + Embeddings erzeugen | Zusammenfassungen generieren |
| Semantische Suche mit Perspektiven-Filter | Suchstrategie waehlen |
| Access-Profile tracken + Salienz berechnen | Material bewerten + reflektieren |
| Visibility degradieren (Aging) | Entscheiden was promoted/demoted wird |
| Hierarchie navigieren (Focus) | Synthese aus aufbereitetem Material |
| Identity-Cluster berechnen | Priming interpretieren |
| Relationen verwalten | Relationen vorschlagen |

VecLayer denkt nicht selbst. Es speichert, strukturiert und rechnet. Das LLM sucht, vertieft, reflektiert ueber aufbereitetes Material, generiert Zusammenfassungen und schreibt sie zurueck. Optional orchestriert ein Schlaf-Zyklus das automatisch.

## Quick Start

```bash
# Neuen VecLayer-Speicher initialisieren
veclayer init

# Wissen hinzufuegen
veclayer add ./docs                        # Dateien/Verzeichnisse
veclayer add "Kernentscheidung: Rust"      # Einzelner Text
veclayer add --perspective decisions "Wir nehmen Turso statt Postgres"

# Suchen
veclayer search "Architekturentscheidungen"
veclayer search --perspective decisions "Backend"

# In die Tiefe gehen
veclayer focus abc1234

# Server starten (MCP/HTTP)
veclayer serve
```

## CLI-Uebersicht

| Kommando | Beschreibung |
|----------|-------------|
| `init` | Neuen VecLayer-Speicher anlegen |
| `add` | Wissen hinzufuegen (Text, Datei, Verzeichnis) |
| `search` / `s` | Semantische Suche mit Perspektiven-Filter |
| `focus` / `f` | In einen Entry eintauchen, Kinder anzeigen |
| `status` | Speicher-Statistiken |
| `serve` | MCP/HTTP-Server starten |
| `compact` | Aging ausfuehren, Salienz berechnen |
| `id` | Identitaets-Zusammenfassung anzeigen |
| `reflect` | Read-only Material-Aufbereitung |
| `think` | LLM-gestuetzte Reflexion + Konsolidierung |
| `p` | Perspektiven verwalten (list, create, remove) |
| `config` | Konfiguration anzeigen/aendern |
| `help` | Kontextuelle Hilfe |

Aliase: `store` = `add`, `s` = `search`, `f` = `focus`

## Technischer Stack

| Komponente | Technologie |
|---|---|
| Sprache | Rust |
| Storage | LanceDB (Prototyp), Turso/SQLite (geplant) |
| Embeddings | fastembed (CPU, ONNX) -- Trait-basiert, austauschbar |
| Parsing | pulldown-cmark (Markdown), erweiterbar |
| Server | axum (MCP + HTTP) |
| CLI | clap v4 |
| Konfiguration | TOML + ENV Overrides (12-Factor) |

## Aktueller Stand

### Implementiert (Prototyp)
- [x] Markdown-Parsing mit Heading-Hierarchie
- [x] FastEmbed-Embeddings (ONNX, CPU-only)
- [x] LanceDB-Backend (serverless)
- [x] RAPTOR-Style Soft Clustering + LLM-Summaries (via Ollama)
- [x] Hierarchische Suche mit Visibility-Filter
- [x] RRD-Style Access-Tracking (hour/day/week/month/year/total)
- [x] Agent-konfigurierbare Aging-Regeln
- [x] 5-Tool MCP-Interface: recall, focus, store, think, share
- [x] HTTP REST API
- [x] Trait-basierte Architektur (DocumentParser, Embedder, VectorStore, Summarizer)
- [x] 12-Factor-Konfiguration via ENV

### Gaps zum Konzept (in Arbeit)
- [ ] Content-Hash IDs statt UUID
- [ ] Entry-Typ (raw/summary/meta/impression) statt is_summary bool
- [ ] Perspektiven (6 Defaults + Custom)
- [ ] Salienz-Berechnung (Dichte, Widersprueche, Spread)
- [ ] Identity-View + Priming beim Connect
- [ ] CLI aligned mit Spec (init, add, search, focus)
- [ ] Summarizer/Clustering aus Core raus (VecLayer denkt nicht)
- [ ] TOML-Config + ENV Overrides
- [ ] Schlaf-Zyklus (optionaler Orchestrator)

## Lizenz

MIT
