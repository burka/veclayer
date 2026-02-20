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

### Die Kernthese

Zusammenfassungen sind nicht ein Feature neben anderen -- sie *sind* das Gedaechtnis selbst. Die Hierarchie, die RAG besser macht (Ueberblick vor Detail, Navigation statt flacher Liste), ist dieselbe Struktur, aus der Identitaet emergiert. Centroids pro Perspektive, gewichtet nach Salienz -- das ist kein Profil, das jemand anlegt, sondern eine Beobachtung, die aus der statistischen Form aller Erinnerungen waechst.

Und die Persoenlichkeit wird nicht von dem geformt, was man oft tut, sondern von dem, was einen bewegt hat. Deshalb misst Salienz Bedeutsamkeit, nicht Haeufigkeit: Interaktionsdichte, Wirkungsbreite ueber Perspektiven, Revisionen und Widersprueche. Was den Agenten gepraegt hat, bleibt praesent -- auch wenn es selten abgerufen wird.

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
- **Interaktionsdichte** (Gewicht 0.5) -- Access-Profile ueber Zeitfenster
- **Wirkungsbreite** (Gewicht 0.25) -- in wie vielen Perspektiven relevant
- **Revisions-Aktivitaet** (Gewicht 0.25) -- Anzahl Relationen als Proxy fuer aktive Deliberation

> Hinweis: Echte Widerspruchserkennung (semantische Spannungen zwischen Entries) ist ein offenes Forschungsthema. Aktuell dient die Revisions-Aktivitaet als Proxy -- viele Relationen deuten auf Ueberarbeitungen, Widersprueche, oder aktive Auseinandersetzung hin.

Ranking: `semantic_similarity * (1-alpha) + (recency * (1-sw) + salience * sw) * alpha`

Entries mit hoher Salienz werden beim Aging geschuetzt -- sie ueberleben Visibility-Degradierung.

### Identitaet

Aus Embedding-Clustern gewichtet nach Salienz waechst eine Identitaet. Kein Profil, das jemand anlegt -- eine Beobachtung: Welche Themen tauchen immer wieder auf, wo liegen die Schwerpunkte, was hat den Agenten gepraegt? Beim Connect erhaelt der Agent ein Priming: Core Knowledge (die salientesten Erinnerungen), offene Faeden (ungeloeste Widersprueche, aktive Deliberation), aktuelle Learnings. Das ist der Moment, in dem ein Agent aufwacht und sich kennt.

### Schlaf-Zyklus (geplant)

Wie ein Mensch im Schlaf konsolidiert, soll ein Agent sein Gedaechtnis aktiv pflegen koennen. Der Zyklus: `reflect` bereitet Material auf (was braucht Aufmerksamkeit?), ein LLM generiert daraus Zusammenfassungen und Meta-Erkenntnisse, `add` schreibt sie zurueck, `compact` raeumt mechanisch auf. Das ist `think` -- der optionale Orchestrator, der alles ohne LLM Funktionierende um die Faehigkeit zur Selbstreflexion ergaenzt.

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

### Phase 1-4: Erledigt

Das gesamte mechanische Fundament steht -- alles was VecLayer ohne LLM leisten soll, funktioniert:

- [x] **Core (Phase 1):** Entry mit SHA-256 Content-Hash, EntryType-Enum (raw/summary/meta/impression), CLI aligned mit Spec, LLM aus Core extrahiert (Feature-Flag `llm`), TOML-Config + ENV Overrides, SRP-Refactoring
- [x] **Perspektiven (Phase 2):** 6 Default-Perspektiven + Custom, typisierte Relationen (SupersededBy, SummarizedBy, VersionOf, RelatedTo, DerivedFrom), facettierte Suche, `veclayer p` fuer Perspektiven-Management
- [x] **Aging + Salienz (Phase 3):** Salienz-Komposit aus Interaktionsdichte, Perspektiven-Spread und Revisions-Aktivitaet, Ranking-Formel mit Salienz-Gewichtung, compact-Kommando, Salienz-Schutz beim Aging
- [x] **Identity + Reflect (Phase 4):** Salienz-gewichtete Embedding-Centroids pro Perspektive, Open Threads (ungeloeste Widersprueche, aktive Deliberation), Reflect-Report, dynamisches Priming beim MCP-Connect

### Offen

- [ ] **Schlaf-Zyklus (Phase 5):** LLMProvider Trait, think-Kommando (reflect -> LLM -> add -> compact), Narrativ-Generierung, automatische Konsolidierung
- [ ] **Server + Sharing (Phase 6):** Personalisierte MCP-Tool-Descriptions, UCAN-basiertes Sharing, REST API an neues Datenmodell anpassen
- [ ] **Polish (Phase 7):** Alias-Support, Multi-Format Parsing (PDF, HTML, Code), alternative Backends (Turso, pgvector)
- [ ] **Widerspruchserkennung:** Aktuell misst Salienz Revisions-Aktivitaet (Anzahl Relationen) als Proxy fuer Widersprueche. Echte semantische Widerspruchserkennung -- Spannungen zwischen Entries erkennen, die sich inhaltlich widersprechen -- ist ein offenes Thema (siehe ROADMAP)

## Lizenz

MIT
