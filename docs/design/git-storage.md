# Git-Based Memory Storage & Distribution

**Status:** Draft
**Date:** 2026-03-05
**Authors:** @flob, Claude

## Motivation

VecLayer currently stores memory locally per user. There is no mechanism to share
knowledge across a team or organization without building a full distributed system
(UCAN, DID, HTTP sync — Phase 6+).

Git solves 80% of the distribution problem with zero new infrastructure. Every team
already has repositories, access control, and review workflows. This design uses git
as the storage and distribution layer for VecLayer memory.

## Goals

- Share memory across team members via git push/pull
- Share canonical knowledge across projects via a company repository
- Human-readable, diffable, mergeable entry format
- Configurable sync behavior (auto-push, PR-based review, manual)
- Customizable MCP tool prompts per team/company
- No new infrastructure — git is the only dependency

## Non-Goals

- Replace the UCAN/DID-based public sharing model (Phase 6)
- Real-time sync between agents
- Conflict-free concurrent writes (git merge is good enough)

## Design

### Entry Format

Each entry is a Markdown file with YAML frontmatter. The Markdown body **is** the
content. The heading structure determines the entry level. Defaults are omitted.

**Minimal entry:**

```markdown
---
id: bf81639
created: 2026-03-05T14:23:00Z
---
VecLayer uses LanceDB for storage and fastembed for CPU-based embeddings.
```

**Entry with perspectives and relations:**

```markdown
---
id: 2b0bd4c
created: 2026-02-21T10:15:00Z
perspectives: [decisions, knowledge]
visibility: always
relations:
  derived_from: 6c121fe
  supersedes: a3f8812
---
# Dual ID Scheme

Local id (sha256(content)) stays stable forever for identity/dedup.
Distributed cid (sha256(ciphertext)) is ephemeral per sync cycle.
```

**Impression with strength:**

```markdown
---
id: e4a9201
created: 2026-03-05T16:00:00Z
perspectives: [learnings]
parent: bf81639
impression: Search is slow after bulk import
impression_strength: 0.6
---
Noticed 3x latency increase when importing >500 entries at once.
Likely missing index rebuild.
```

**Entry with expiry:**

```markdown
---
id: c12ab34
created: 2026-03-04T09:00:00Z
perspectives: [session]
expires: 2026-04-04T09:00:00Z
---
# Sprint 12 Planning

Decided to prioritize git-based distribution over UCAN auth.
```

**Summary entry:**

```markdown
---
id: a1b2c3d
created: 2026-03-05T18:00:00Z
perspectives: [knowledge]
summarizes: [6c121fe, 4a2b3c1, e4a9201]
---
# Embedding Performance

Three separate investigations confirmed that bulk imports need explicit
index rebuilds. The 3x latency is consistent across store sizes >500 entries.
```

### Frontmatter Fields

| Field | Type | Default | Omit when |
|---|---|---|---|
| `id` | string (7-char short hash) | — | never (required) |
| `created` | ISO 8601 datetime | — | never (required) |
| `perspectives` | string list | `[]` | no perspectives |
| `visibility` | string | `"normal"` | normal visibility |
| `relations` | map of kind → id or id list | `{}` | no relations |
| `parent` | string (short id) | none | no parent |
| `expires` | ISO 8601 datetime | none | no expiry |
| `impression` | string | none | not an impression |
| `impression_strength` | float 0.0–1.0 | `1.0` | strength is 1.0 |
| `summarizes` | string list (short ids) | `[]` | not a summary |

**No `type` field.** Entry classification is inferred:
- Has `summarizes` → summary
- Has `impression` → impression
- Otherwise → regular entry

The `entry_type` enum (`raw`, `meta`, `summary`, `impression`) existed for internal
classification but has no behavioral impact except impression strength modulation in
salience scoring. Since `impression_strength` defaults to 1.0, applying it universally
(`composite *= impression_strength`) is a no-op for regular entries.

### Relations (Compact Format)

Relations use the kind as key, target id(s) as value:

```yaml
relations:
  derived_from: 6c121fe
  supersedes: a3f8812
```

Multiple targets of the same kind:

```yaml
relations:
  derived_from: [6c121fe, 4a2b3c1]
  supersedes: a3f8812
```

Well-known relation kinds: `supersedes`, `superseded_by`, `derived_from`,
`related_to`, `summarized_by`, `summarizes`, `version_of`.

### Heading & Level

The Markdown heading in the body determines the entry level:

- `# Heading` → H1
- `## Heading` → H2
- No heading → Content level

No `level` field in frontmatter. The Markdown **is** the structure.

### File Naming

Slug derived from heading (or content), suffixed with short id:

```
dual-id-scheme-2b0bd4c.md
search-slow-after-bulk-import-e4a9201.md
sprint-12-planning-c12ab34.md
veclayer-uses-lancedb-bf81639.md
```

Heading changes cause file renames. Git tracks renames well. The short-id suffix
ensures entries are always findable by ID regardless of slug.

### Directory Structure

First perspective determines directory. No perspectives → `_unsorted/`.

```
veclayer-memory/              # orphan branch or separate repo
├── config.toml
├── decisions/
│   ├── dual-id-scheme-2b0bd4c.md
│   └── auth-progressive-model-a8f3201.md
├── knowledge/
│   └── architecture-overview-4c2e91a.md
├── learnings/
│   └── search-slow-after-bulk-import-e4a9201.md
├── session/
│   └── sprint-12-planning-c12ab34.md
├── intentions/
│   └── git-distribution-before-ucan-f91a3bc.md
├── _unsorted/
│   └── veclayer-uses-lancedb-bf81639.md
├── .embeddings/               # force-pushed with lease, GC-able
│   ├── bf81639.bge-small-en-v1.5.bin
│   ├── 2b0bd4c.bge-small-en-v1.5.bin
│   └── e4a9201.bge-small-en-v1.5.bin
├── .index/                    # local only, in .gitignore
│   └── lance/                 # local vector index
└── .gitignore                 # ignores .index/
```

Multi-perspective entries (e.g. `[decisions, knowledge]`) live in the first
perspective's directory. The full list is in frontmatter.

### Storage Layers

Three distinct layers with different git strategies:

```
┌─────────────────────────────────────────────────────────────┐
│  Entries (Markdown)          git merge, append-only         │
│  ─ Human-readable, diffable, never force-pushed             │
├─────────────────────────────────────────────────────────────┤
│  Embedding Cache (.embeddings/)   force-push with lease     │
│  ─ Binary blobs grouped by age, GC cleans stale batches     │
├─────────────────────────────────────────────────────────────┤
│  Local Index (.index/)            local only, .gitignore    │
│  ─ LanceDB/HNSW index, rebuilt from entries + cache         │
└─────────────────────────────────────────────────────────────┘
```

### Embedding Cache

Embeddings are expensive to compute but trivial to store. Rather than making every
team member re-embed the same content, embedding vectors are cached in git as binary
blobs grouped by time window (weekly batches).

**File naming:** `<short-id>.<model-name>.bin` — content-addressed by entry ID and
embedding model. One file per entry per model.

```
.embeddings/
├── bf81639.bge-small-en-v1.5.bin    # 384 dims × 4 bytes = 1.5 KB
├── 2b0bd4c.bge-small-en-v1.5.bin
└── e4a9201.bge-small-en-v1.5.bin
```

**O(1) lookup:** Need the embedding for entry `bf81639`? Check if
`bf81639.<model>.bin` exists. No index, no batch scanning.

**Blob format:** Raw float32 vector (dimensions × 4 bytes). No header, no
framing — the model name in the filename tells you the dimension. A 384-dim
embedding is exactly 1,536 bytes. Simple, fast, memory-mappable.

**Lifecycle:**
- New entry → compute embedding → write `<id>.<model>.bin` → commit
- Entry unchanged → embedding file untouched (git sees no diff)
- Entry superseded → embedding file deleted in next cleanup commit
- Orphaned `.bin` files (no matching `.md`) are garbage

**Force-push with lease:**
- Periodic cleanup: delete `.bin` files with no matching entry, force-push with lease
- Safe: fails if remote diverged (someone else pushed), never loses work
- Git GC prunes unreferenced objects after `gc.reflogExpire` (default 90 days)
- Entries (Markdown files) are **never** force-pushed — only embedding cleanup

**Cold start (no cache):**
1. Clone/checkout the memory branch
2. Missing `.bin` files → compute embeddings locally
3. Commit + push new `.bin` files (normal push, or force-with-lease for cleanup)
4. Next team member pulls and gets the cache for free

**Model switch:**
- `config.toml` specifies the embedding model
- Model changes → all `*.<old-model>.bin` files are stale
- Next session computes `*.<new-model>.bin` for all entries
- Cleanup commit removes old model files, force-push with lease
- Old blobs get GC'd naturally

### What's Local-Only

These never leave the machine:

- **Vector index** (`.index/`) — rebuilt from entries + embedding cache
- **Access profiles** — per-user interaction tracking
- **Cluster memberships** — computed from embeddings
- **Salience scores** — derived from access + aging

## Configuration

### `config.toml`

Lives in the branch root. Versioned and shared with the team.

```toml
[memory]
version = "1"

# Embedding model for local index rebuilds
embedding_model = "bge-small-en-v1.5"

[memory.sync]
push_mode = "auto"            # auto | review | manual
branch = "veclayer-memory"

[memory.company]
remote = "git@github.com:acme/shared-memory.git"
branch = "veclayer-memory"
push_mode = "review"

[memory.prompts]
# Override MCP tool descriptions
recall = "Search the team's shared knowledge base..."
store = "Always tag decisions with project name..."

# Priming text injected into agent context
priming = """
You have access to the team's shared memory.
Check recall before starting new work.
Store decisions and rationale as you make them.
"""

[memory.prompts.perspectives]
# Custom perspective descriptions
decisions = "Architectural and design choices with rationale"
playbook = "Team-specific patterns and conventions"
onboarding = "Context for new team members"
```

### Push Modes

| Mode | Behavior | Use case |
|---|---|---|
| `auto` | Commit and push on every store | Session/project memory, no review needed |
| `review` | Commit locally, open PR for review | Company memory, canonical knowledge |
| `manual` | Commit locally, push only on instruction | Explicit control |

Defaults are recommendations. The agent can deviate per interaction when the user
instructs it:

> "Store as company pattern, but open a PR for review"
> "Push directly, this is uncontroversial"

### Prompt Configuration

The `[memory.prompts]` section allows teams to customize how agents interact with
memory without code changes. This includes:

- **Tool descriptions** — override the default `recall`, `store`, `focus`, `think`
  descriptions shown to the agent
- **Priming** — injected into the agent's context at session start
- **Perspective descriptions** — explain what each perspective means for this team

## Distribution Model

### Three Tiers

```
┌─────────────────────────────────────────────┐
│  Company Repo (shared-memory.git)           │
│  PR-reviewed, canonical knowledge           │
├─────────────────────────────────────────────┤
│  Project Branch (veclayer-memory)           │
│  Auto-pushed, team-level knowledge          │
├─────────────────────────────────────────────┤
│  Local Store (current VecLayer)             │
│  Personal memory, never pushed              │
└─────────────────────────────────────────────┘
```

- **Local** → project branch: auto or manual push
- **Project branch** → company repo: PR-based review
- Access control inherited from git permissions (push/clone rights)
- Append-only convention: entries are never deleted, only superseded

### Conflict Resolution

**Entries (Markdown):**
- Session-level UUIDs in short-id prevent write conflicts
- Content merges naturally (Markdown diffs are clean)
- Rebase preferred over merge for linear history
- Never force-pushed

**Embedding cache (binary blobs):**
- Force-push with lease — safe, fails if remote diverged
- On conflict: pull remote embeddings, merge locally, push again
- Worst case: discard remote cache, re-embed locally, push fresh
- Stale blobs are cleaned up by git GC automatically

### Sync Flow

On `store`:
1. Write entry as Markdown file to the branch working tree
2. `git add` + `git commit` (always)
3. If `push_mode == "auto"`: `git pull --rebase && git push`
4. If `push_mode == "review"`: accumulate commits, open PR on threshold
5. If `push_mode == "manual"`: do nothing until instructed

On session start (or `recall`):
1. `git pull --rebase` on the memory branch
2. Parse all Markdown files → build in-memory entry set
3. Load embedding cache from `.embeddings/` blobs
4. For entries with cached embeddings (matching model) → use cache
5. For new/changed entries → compute embeddings locally
6. Rebuild local vector index (`.index/`)
7. Push updated embedding cache blobs (force-with-lease)
8. Ready for search

## Migration Path

### From Current VecLayer Store

Export existing entries → write as Markdown files → commit to branch.
The entry format captures all fields except local-only data (embeddings,
access profiles, clusters). Re-embedding happens automatically on first load.

### To Phase 6 (UCAN/DID)

The git layer coexists with the future UCAN layer:
- Git for team/company distribution (private, authenticated via git)
- UCAN for public/cross-org sharing (open, authenticated via DID)
- Same entry format, different transport

## Open Questions

1. **Branch vs. repo** — Orphan branch in the project repo, or a separate repo?
   Orphan branch is simpler but mixes concerns.
4. **Perspective directories for company repo** — Should company repos enforce
   a fixed set of perspectives, or allow teams to add their own?
5. **Large repos** — At what entry count does clone/pull become painful?
   Should we support shallow clones or sparse checkout for the embedding blobs?
