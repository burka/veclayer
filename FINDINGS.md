# VecLayer Git Memory Feature — Test Findings

> Comprehensive testing from an agentic user perspective.
> Date: 2026-03-07 | Binary: `target/debug/veclayer` | All tests local-only.

## Executive Summary

**5 test agents** ran ~60 test scenarios covering init, store/recall, sync workflows,
error handling, and data portability. Results:

- **Bugs found: 26 distinct bugs** (8 high severity, 9 medium, 9 low)
- **Fixed in c38849f: 12 bugs** (B01, B02, B04, B06, B08, B12, B13, B16, B20, B23, B25, B26)
- **Fixed in 550de70: 8 bugs** (B05, B09, B10, B14, B16, B17, B21, B23/B24)
- **Already implemented: 2** (B07✓relations in frontmatter, B19✓config on branch)
- **Not a bug: 1** (B11 — focus --question works correctly with children)
- **Remaining: 3 bugs** (B03 external dep, B15 cosmetic, B22 needs investigation)
- **UX issues: 23** (usability problems that confuse agents)
- **Suggestions: 10** (improvements for production readiness)

### Critical Bugs (Must Fix)

| ID | Severity | Summary | Status |
|----|----------|---------|--------|
| B01 | HIGH | Panic on malformed `.veclayer/config.toml` (exit 101, not graceful error) | ✅ c38849f |
| B02 | HIGH | Stale worktree not auto-pruned → all git ops fail after cache cleanup | ✅ c38849f |
| B03 | HIGH | Concurrent stores fail 4/5 due to FastEmbed ONNX model contention | ⏳ External dep |
| B04 | HIGH | Silent global-store fallback when no local `.veclayer/` exists | ✅ c38849f |
| B05 | HIGH | Identical content from 2 clients causes unresolvable merge conflict | ✅ 550de70 |
| B06 | HIGH | `--dry-run` still executes `git pull` (side effect in read-only mode) | ✅ c38849f |
| B23 | HIGH | `entry_type` lost through git sync roundtrip (meta→raw) | ✅ 550de70 |
| B26 | HIGH | Import/merge silently targets global store (same root cause as B04) | ✅ c38849f |

---

## Bug Details

### B01 — Panic on malformed config.toml [HIGH]

**Location:** `src/config.rs:901`
**Trigger:** Any command when `.veclayer/config.toml` contains invalid TOML.
**Behavior:** Thread panic with exit code 101, Rust backtrace suggestion.
**Expected:** Graceful `Error:` message with exit code 1.
**Root cause:** `unwrap_or_else(|e| panic!(...))` instead of `Result::Err` propagation.

```
thread 'main' panicked at src/config.rs:901:21:
Invalid TOML in .veclayer/config.toml: TOML parse error at line 1...
```

### B02 — Stale worktree blocks all git operations [HIGH]

**Location:** `src/git/worktree.rs`
**Trigger:** `~/.cache/veclayer/` deleted (OS cleanup, manual rm, path hash change).
**Behavior:** `git worktree add` fails with "already used by worktree" because git's
internal `.git/worktrees/` still references the deleted path.
**Expected:** Auto-recovery via `git worktree prune` before `git worktree add`.
**Impact:** `store` degrades to LanceDB-only (silent data loss from git), `sync --migrate` fails hard.

```
Error: InvalidOperation("Failed to open memory branch: git worktree add failed (exit 128):
  fatal: 'veclayer-memory' is already used by worktree at '...prunable'")
```

**Fix:** Call `git worktree prune` before attempting `git worktree add`.

### B03 — Concurrent FastEmbed initialization fails [HIGH]

**Location:** FastEmbed ONNX model initialization
**Trigger:** Multiple `veclayer store` processes running simultaneously.
**Behavior:** 4 out of 5 parallel stores fail with `Embedding("Failed to initialize FastEmbed: Failed to retrieve onnx/model.onnx")`.
**Expected:** All 5 succeed (or queue/retry).
**Impact:** An agent running rapid-fire stores loses most entries silently.

### B04 — Silent fallback to global store [HIGH]

**Location:** `src/main.rs` data-dir resolution
**Trigger:** Running `store`, `recall`, `status`, `focus` in a directory without `.veclayer/`.
**Behavior:** Silently targets `~/.local/share/veclayer` (global store). No warning.
**Expected:** Either error ("no store found, run `veclayer init`") or explicit message
("using global store at ...").
**Impact:** Agent accidentally pollutes global store or reads stale cross-project data.

### B05 — Identical content causes unresolvable conflict [HIGH] ✅ Fixed in 550de70

**Location:** `src/git/sync.rs` pull_rebase path
**Trigger:** Two clients store identical content (same content-hash → same filename),
both push independently.
**Behavior:** Git reports add/add conflict on the same filename. `veclayer sync` reports
conflict but provides no resolution command. User is stuck.
**Expected:** Auto-resolve: files are byte-identical, keep either copy.
**Fix:** After detecting conflict, check if conflicting files are identical; if so, resolve
automatically. Or use a merge strategy that handles this.

### B06 — dry-run still performs git pull [MEDIUM→HIGH]

**Location:** `src/commands/sync.rs`
**Trigger:** `veclayer sync --dry-run`
**Behavior:** Executes `git pull` (modifies local state) before reporting dry-run results.
**Expected:** No side effects in dry-run mode.

### B07 — Relations not persisted in git frontmatter [MEDIUM] ✅ Already implemented

**Location:** `src/git/markdown.rs` render/parse
**Trigger:** Store entry with `--rel-supersedes`, `--rel-to`, `--rel-derived-from`.
**Behavior:** Relations stored in LanceDB only. Git markdown frontmatter contains no
relation fields (except `parent`). Rebuilding from git loses all relations.
**Expected:** Relations serialized in frontmatter like `parent` is.
**Impact:** Team members syncing from git branch get entries without relation graphs.

### B08 — .txt files silently ignored by store [MEDIUM]

**Location:** `src/commands/store_ops.rs` file processing
**Trigger:** `veclayer store file.txt`
**Behavior:** Returns `Added 0 entries (0 summaries) from 0 files` — no error, no warning.
**Expected:** Either process the file or give a clear error ("unsupported file type: .txt").
**Impact:** Agent thinks file was stored when it wasn't.

### B09 — Duplicate files on git branch for same entry ID [MEDIUM] ✅ Fixed in 550de70

**Location:** `src/git/memory_store.rs` store_entry
**Trigger:** Store entry inline, then store a file whose content produces the same hash.
**Behavior:** Two `.md` files with the same 7-char ID exist on the branch.
**Expected:** One file per content ID; second store overwrites or skips.

### B10 — Duplicate recall results [MEDIUM] ✅ Fixed in 550de70

**Location:** Vector search deduplication
**Trigger:** Entry stored from a file (creates LanceDB entry from file + possibly inline).
**Behavior:** Same entry ID appears twice in recall results with identical scores.
**Expected:** Deduplication before display.

### B11 — `focus --question` is a no-op [MEDIUM] ❌ Not a bug (works with children)

**Location:** `src/commands/` focus handler
**Trigger:** `veclayer focus <id> --question "What was the rationale?"`
**Behavior:** Output identical to `focus <id>` without `--question`.
**Expected:** Children reranked by question relevance, or flag hidden if unimplemented.

### B12 — Spurious WARN on every invocation [MEDIUM]

**Location:** Embedding model initialization
**Trigger:** Any command that loads the embedding model.
**Behavior:** `WARN Unrecognised fastembed model 'BAAI/bge-small-en-v1.5', falling back to default`
**Expected:** Model recognized (it works fine), no warning.

### B13 — Tracing logs go to stdout instead of stderr [MEDIUM]

**Location:** `src/main.rs` tracing_subscriber setup
**Trigger:** Any CLI command.
**Behavior:** WARN/INFO/DEBUG logs go to stdout, mixed with program output.
**Expected:** Logs to stderr (only MCP mode correctly uses stderr).
**Impact:** Breaks `veclayer export > file.jsonl` and any script parsing stdout.

### B14 — `veclayer config` doesn't show storage/push settings [MEDIUM] ✅ Fixed in 550de70

**Location:** `src/commands/` config display
**Trigger:** `veclayer config` after `init --share`.
**Behavior:** `storage = "git"` and `push = "review"` from `.veclayer/config.toml` not displayed.
**Expected:** Config output shows all resolved settings.

### B15 — `veclayer config` shows data_dir as "(default)" always [LOW]

**Location:** `src/commands/` config display
**Trigger:** `veclayer config` with project-local `.veclayer/`.
**Behavior:** Shows `data_dir: (default)` even when `.veclayer/` is actively used.

### B16 — Sync --pending shows commit count, not entry count [LOW] ✅ Fixed in 550de70

**Location:** `src/commands/sync.rs`
**Trigger:** `veclayer sync --pending`
**Behavior:** "7 commit(s) not yet pushed" but only 5 entries listed (extra commits are
embedding cache commits).
**Expected:** Show entry count, or explain the commit vs entry distinction.

### B17 — Sync --reject increases pending count [LOW] ✅ Fixed in 550de70

**Location:** `src/commands/sync.rs`
**Trigger:** `veclayer sync --reject <id>`
**Behavior:** Pending count goes from 7 to 8 (removal commit added).
**Expected:** Count decreases or stays same. Counterintuitive to users.

### B18 — Scope filter wrong error message [LOW] ✅ Fixed in c38849f

**Location:** `src/commands/sync.rs`
**Trigger:** `veclayer sync --scope nonexistent`
**Behavior:** "No scopes configured" (implies no config at all).
**Expected:** "Scope 'nonexistent' not found. Available scopes: project"

### B19 — Config.toml not written to git branch on init [LOW] ✅ Already implemented

**Location:** `src/commands/store_ops.rs` init_share
**Trigger:** `init --share` when veclayer-memory branch already exists (created manually).
**Behavior:** Branch has no `config.toml`; only `.veclayer/config.toml` in working tree.
**Expected:** Branch should get a config.toml for team members.

### B20 — Inconsistent sync exit codes [LOW]

**Location:** `src/commands/sync.rs`
**Trigger:** Sync subcommands in non-git directory.
**Behavior:** `sync` exits 0 ("No scopes"), `sync --push` exits 1, `sync --migrate` exits 1.
**Expected:** Consistent exit codes for equivalent situations.

### B21 — Misleading "already exists" on first init --share [LOW] ✅ Fixed in 550de70

**Location:** `src/commands/store_ops.rs`
**Trigger:** First `init --share` in a fresh project directory.
**Behavior:** Prints "VecLayer store already exists at ~/.local/share/veclayer" (refers to
global store, not the project store being created).
**Expected:** No mention of global store during project init.

### B22 — veclayer-memory branch not orphan in some scenarios [LOW]

**Location:** `src/git/worktree.rs`
**Trigger:** After sync operations in some scenarios.
**Behavior:** `git log veclayer-memory` shows project's main branch history mixed with
memory commits (branch may have inherited history from main).
**Expected:** Clean orphan branch with only veclayer memory commits.

---

## UX Issues

### UX01 — No indication of which store is targeted

When no `.veclayer/` exists, commands silently use the global store. An agent can't tell
if it's reading/writing project data or global data.

### UX02 — "Created memory branch" on every init --share

Even when the branch already exists, the message says "Created" instead of "Already exists".

### UX03 — Quiet flag doesn't suppress println! output

`-q` only suppresses tracing logs, not `println!` messages. Machine-friendly silence not achievable.

### UX04 — Impression fields not shown in focus output

`impression_hint` and `impression_strength` are stored but not displayed by `focus`.

### UX05 — Superseding entry doesn't show reverse relation

`focus` on the superseding entry doesn't show "supersedes → <id>". Relationship is one-sided in display.

### UX06 — Recall without query caps at 5 entries silently

`recall` (no query) shows only 5 entries with no "showing 5 of N" message. Agent assumes
only 5 entries exist.

### UX07 — Inconsistent behavior: recall --perspective vs recall --since

`--perspective` (no query) returns all matching entries exhaustively.
`--since` (no query) caps at 5. Inconsistent.

### UX08 — Empty heading accepted silently

`store "x" --heading ""` succeeds but the entry shows as blank in all listings.
Should warn or require a heading.

### UX09 — Long heading pollutes recall output

10K-char heading rendered in full in `recall` output. Should truncate in list views.

### UX10 — Status level vs type breakdown unclear

`Entries by level: Content 23` doesn't explain the Content/H1/H2 distinction or break
down by entry_type (raw/meta/impression).

### UX11 — Raw Rust error struct exposed to user

Permission denied shows `Io(Os { code: 13, kind: PermissionDenied, ... })` instead of
a human-readable message.

### UX12 — Empty ID produces invisible error text

`focus ""` → `Error: NotFound("Entry  not found")` — the empty string is invisible in
the message.

### UX13 — Stage/reject use inconsistent error formats

`--stage` gives `NotFound`, `--reject` wraps a git subprocess error string in
`InvalidOperation`. Different error types for similar situations.

### UX14 — No discoverable scope names

`sync --scope` doesn't offer `--list-scopes`. Users must inspect config.toml manually.

### UX15 — No conflict resolution command

When sync reports conflicts, users must manually navigate git worktrees. No `veclayer`
command to resolve or abort.

### UX16 — Migrate doesn't auto-pull before committing

`sync --migrate` → `sync --push` fails if remote is ahead. Migrate should pull first
or warn.

### UX17 — Sync --pending stages vs commits confusion

"7 commit(s)" includes embedding cache commits, but only 5 entries listed. Misleading.

### UX18 — Config doesn't show which config file was loaded

Users can't tell if `~/.config/veclayer/config.toml`, `$XDG_CONFIG_HOME`, or
`$VECLAYER_USER_CONFIG` was used.

### UX19 — No `init --repair` or `sync --reset` recovery commands

When worktrees break, users must use raw git commands. A recovery command would help.

### UX20 — `think relate` kind is named option, not positional

`think relate <id1> <id2> related_to` fails — kind must be `--kind related_to`.
User/agent intuition expects positional. Consider accepting both.

### UX21 — `history` doesn't follow relation chains

For multi-hop history (A supersedes B supersedes C), user must run `history` on each
entry individually. A `--follow` flag would improve usability.

### UX22 — `merge` error message doesn't show what to pass for `-p`

"use -p or --force" without showing how to find the project name.

### UX23 — `.fastembed_cache` per working directory (128MB each)

Cache created in each working directory. Multiple projects accumulate ~128MB each.
Should use a centralized cache location.

---

## Suggestions

| ID | Suggestion |
|----|------------|
| S01 | Relations (`superseded_by`, `related_to`, `derived_from`) should be persisted in git frontmatter, not just LanceDB |
| S02 | Implement or remove `focus --question` flag |
| S03 | Auto-resolve identical-content merge conflicts (files are byte-identical) |
| S04 | Add `sync --list-scopes` to discover available scope names |
| S05 | Add progress output for long operations (store directory, migrate 1000+ entries) |
| S06 | Document filename truncation limit (~50 chars) |
| S07 | Add `entry_type` to git markdown frontmatter so it survives sync roundtrip |
| S08 | Separate import skip counter: "N skipped (M duplicates, K parse errors)" |
| S09 | `sync --migrate` should auto-prune stale worktrees before adding new one |
| S10 | When no `.veclayer/` exists, warn which store is targeted before writing |

---

## Additional Bugs from Data Portability Testing

### B23 — entry_type lost through git sync roundtrip [HIGH] ✅ Fixed in 550de70

**Location:** `src/git/markdown.rs` frontmatter serialization
**Trigger:** Store entry with `--entry-type meta`, sync via git to another store.
**Behavior:** Git markdown frontmatter omits `entry_type` field. Synced entries default to `raw`.
**Expected:** `entry_type` preserved in frontmatter and restored on sync.
**Impact:** AI agents lose type annotations (meta/summary/impression) when sharing knowledge via git.

### B24 — Level changes from Content to H1 after git sync [LOW] ✅ Fixed in 550de70

**Location:** `src/git/markdown.rs` re-ingestion
**Trigger:** Inline-stored entry (Content level) synced via git.
**Behavior:** Markdown heading (`# Title`) causes re-ingestion at H1 level.
**Impact:** Search ranking may differ after sync roundtrip.

### B25 — Import skip summary says "already exist" for parse errors [MEDIUM]

**Location:** `src/commands/` import handler
**Trigger:** Import a JSONL file with malformed lines.
**Behavior:** "3 skipped (already exist)" when actually 3 lines failed to parse.
**Expected:** "3 skipped (3 parse errors)" or similar breakdown.

### B26 — Import/merge silently targets global store [HIGH]

**Location:** Data-dir resolution fallback
**Trigger:** `veclayer import file.jsonl` in directory without `.veclayer/`.
**Behavior:** Entries silently imported into `~/.local/share/veclayer` (global store).
**Expected:** Warning or error about missing local store.
**Note:** Same root cause as B04 (silent global-store fallback).

---

## What Works Well

Despite the bugs, the core design is solid:

1. **Init workflow is clear** — `init --share` creates branch + config with actionable next steps
2. **Sync lifecycle works** — A pushes, B syncs, entries flow correctly between repos
3. **Diverged repo handling** — Clear error message with resolution steps
4. **Content-addressing** — Entries get deterministic IDs from content hash
5. **Perspective-based organization** — Entries organized into directories by perspective
6. **Filename sanitization** — All special characters handled correctly
7. **Git branch structure** — Clean, human-readable markdown files with YAML frontmatter
8. **Export/import roundtrip** — JSONL format works, pipe support works
9. **Recall scoring** — Semantic search returns relevant results with meaningful scores
10. **Idempotent operations** — Double-init is safe, double-store is safe
11. **Rebuild-index** — Full recovery from blob store after index corruption
12. **Archive/promote/demote** — Visibility management works correctly
13. **Think discover** — Finds similar-but-unlinked entries effectively
14. **Merge with --force** — Cross-store data movement works
15. **Malformed JSONL handling** — Graceful per-line skip with warnings

---

## Test Coverage Analysis

### Bug Test Coverage (B01–B26)

| Bug | Test | File |
|-----|------|------|
| B01 | ✅ `test_discover_project_bad_toml_returns_none` | src/config.rs |
| B02 | ✅ `test_stale_worktree_is_detected_and_recreated` | tests/git_review_integration.rs |
| B03 | ❌ No test (external dep) | — |
| B04 | ❌ No test (code fix only) | — |
| B05 | ✅ `test_pull_rebase_auto_resolves_identical_content_conflict` | tests/git_review_integration.rs |
| B06 | ❌ No test (code fix only) | — |
| B07 | ✅ `test_relations_compact_format_*` | src/git/markdown.rs |
| B08 | ❌ No test (code fix only) | — |
| B09 | ✅ `test_store_same_content_different_heading_no_duplicate` | src/git/memory_store.rs |
| B10 | ✅ `test_search_deduplicates_by_chunk_id` | src/search/mod.rs |
| B11 | ❌ Not a bug | — |
| B12 | ❌ No test (config change) | — |
| B13 | ❌ No test (code fix only) | — |
| B14 | ❌ No test (code fix only) | — |
| B15 | ❌ Not fixed (cosmetic) | — |
| B16 | ❌ No test (message change) | — |
| B17 | ❌ No test (message change) | — |
| B18 | ❌ No test (code fix only) | — |
| B19 | ✅ Already implemented | — |
| B20 | ❌ No test (code fix only) | — |
| B21 | ❌ No test (code fix only) | — |
| B22 | ❌ Not fixed (needs investigation) | — |
| B23 | ✅ `test_roundtrip_meta_entry_type` + 3 more | src/git/markdown.rs |
| B24 | ✅ `test_roundtrip_content_level_preserved` + 2 more | src/git/markdown.rs |
| B25 | ✅ `test_build_skip_detail_*` (4 tests) | src/commands/data.rs |
| B26 | ❌ No test (same fix as B04) | — |

**Summary:** 13 bugs have regression tests. 12 bugs fixed without tests (mostly CLI output/messaging changes). 0 of 23 UX issues have tests.

### UX Issues — Not Fixed

None of the 23 UX issues (UX01–UX23) were fixed. They are usability improvements, not correctness bugs. Some overlap with bug fixes (UX01↔B04 warning, UX17↔B16 message).

### Remaining Unfixed

| ID | Severity | Summary | Reason |
|----|----------|---------|--------|
| B03 | HIGH | Concurrent FastEmbed ONNX model contention | External dependency (fastembed-rs) |
| B15 | LOW | Config shows data_dir as "(default)" always | Cosmetic; ResolvedConfig lacks project-local discovery info |
| B22 | LOW | veclayer-memory branch not orphan in some scenarios | Needs investigation |

---

## Embedding Architecture

### What Exists
- **Background worker** (`src/mcp/embed_worker.rs`): `tokio::spawn` task, polls 32 entries/batch, offloads to `spawn_blocking`
- **Batch-capable trait**: `embed(&[&str]) -> Vec<Vec<f32>>` accepts multiple texts
- **Two backends**: FastEmbed (ONNX, CPU) and Ollama (HTTP, behind `llm` feature)

### What's Missing
- **No ONNX thread tuning**: `intra_op_num_threads` / `inter_op_num_threads` not exposed by fastembed-rs
- **No Rayon / data parallelism**: No `par_iter`, no parallel batches
- **Single-text call sites**: `search.rs`, `sync.rs`, `think.rs` all call `embed(&[single_text])` sequentially
- **No GPU support**: fastembed-rs is CPU-only

### Suggested Improvements
1. Expose ONNX thread config via fastembed-rs or switch to ort directly
2. Batch sequential `embed(&[text])` calls into `embed(&[text1, text2, ...])` where possible
3. Centralize `.fastembed_cache` to `~/.cache/fastembed/` (currently 128MB per working directory)

---

## Recommended UX Improvements (Priority Order)

1. **UX11 — Human-readable errors**: Replace raw `Os { code: 13, kind: PermissionDenied }` with clean messages. Low effort, high impact.
2. **UX06/UX07 — Recall cap transparency**: Show "5 of N results" when capping. Agents think there are only 5 entries.
3. **UX12 — Empty ID guard**: `focus ""` → "Entry ID required" instead of "Entry  not found".
4. **UX14 — `sync --list-scopes`**: Agents can't discover scope names without reading config files.
5. **UX04 — Show impression fields in focus**: Display `impression_hint`/`strength` if stored.
6. **UX23 — Centralize `.fastembed_cache`**: 128MB per working directory is wasteful.

---

## Test Environment

- Binary: `/home/flob/work/veclayer/target/debug/veclayer`
- Git: local bare repos as remotes
- OS: Linux 6.17.0-14-generic
- 5 parallel test agents, ~60 test scenarios
- No network access required
