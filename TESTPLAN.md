# VecLayer Git Memory Feature — Test Plan

> Comprehensive CLI and MCP test plan for the git-based memory sharing feature.
> All tests use local git repos only (bare repos as "remotes", no network).

## 1. Init & Config

| # | Test Case | Steps | Expected |
|---|-----------|-------|----------|
| 1.1 | Happy path init --share | Fresh git repo → `veclayer init --share` | Orphan branch `veclayer-memory` created, `.veclayer/config.toml` written with `storage = "git"` |
| 1.2 | Idempotent init --share | Run `init --share` twice | Second run succeeds, no data loss, config not duplicated |
| 1.3 | Init outside git repo | `init --share` in plain directory | Clear error: "Not a git project" |
| 1.4 | Init in bare repo | `git init --bare` → `init --share` | Clear error |
| 1.5 | Init in repo with no commits | `git init` (no commit) → `init --share` | Succeeds, orphan branch created |
| 1.6 | Init with pre-existing veclayer-memory branch | Manually create branch → `init --share` | Existing content preserved, not overwritten |
| 1.7 | Config output shows git settings | `veclayer config` after init --share | Shows storage=git, push mode, data_dir, git remote |
| 1.8 | Custom data dir | `veclayer -d /custom/path init --share` | Respects custom dir |
| 1.9 | Status empty then after store | `status` → `store` → `status` | Count progresses correctly |
| 1.10 | Malformed config.toml | Write garbage to `.veclayer/config.toml` | Graceful error (not panic) |
| 1.11 | Verbose flag | `veclayer -v init --share` | Additional debug output to stderr |
| 1.12 | Quiet flag | `veclayer -q init --share` | Suppress all non-error output |

## 2. Store & Recall

| # | Test Case | Steps | Expected |
|---|-----------|-------|----------|
| 2.1 | Basic store/recall roundtrip | Store 3 entries → recall by keyword | Relevant results returned with scores |
| 2.2 | Store from .md file | `store path/to/file.md` | File content indexed |
| 2.3 | Store from .txt file | `store path/to/file.txt` | File content indexed (or clear error) |
| 2.4 | Store from directory | `store path/to/dir/` | All supported files indexed |
| 2.5 | Store with --heading | `store "content" --heading "Title"` | Heading preserved in recall/focus |
| 2.6 | Store with -P perspectives | `-P decisions,knowledge` | Perspectives stored, filterable |
| 2.7 | Store with --entry-type | `--entry-type meta/impression` | Type preserved in focus output |
| 2.8 | Store with --visibility | `--visibility deep_only/always` | Visibility affects recall filtering |
| 2.9 | Store with --parent-id | Parent-child hierarchy | Focus on parent shows children |
| 2.10 | Store with relations | `--rel-supersedes`, `--rel-to`, `--rel-derived-from` | Relations stored in both LanceDB and git frontmatter |
| 2.11 | Recall with --perspective | Filter by perspective | Only matching entries returned |
| 2.12 | Recall with --since/--until | Date filtering | Correct temporal filtering |
| 2.13 | Recall with --deep | Include deep_only entries | Shows archived/deep entries |
| 2.14 | Recall with --similar-to | Find similar entries | Semantically similar results |
| 2.15 | Recall with --ongoing | Open threads only | Only unresolved items |
| 2.16 | Focus with --question | Rerank children by question | Children reranked (not no-op) |
| 2.17 | Focus on nonexistent ID | `focus 0000000` | Clear error, exit 1 |
| 2.18 | Special chars in heading | Slashes, quotes, brackets, unicode, emoji | Stored and retrieved correctly |
| 2.19 | Empty content/heading | `store "" --heading ""` | Validation warning or clear behavior |
| 2.20 | Very long content (10K chars) | Store large inline text | Succeeds, heading truncated in filenames |
| 2.21 | Git branch state after store | `git ls-tree veclayer-memory` | Markdown files exist, organized by perspective |
| 2.22 | No duplicate files on git branch | Store same entry → check branch | One file per content ID |
| 2.23 | No duplicate recall results | Recall with various queries | Each entry appears at most once |

## 3. Sync Workflow

| # | Test Case | Steps | Expected |
|---|-----------|-------|----------|
| 3.1 | Full lifecycle: A pushes, B syncs | bare + 2 clients, A stores+pushes, B inits+syncs | B sees A's entries in recall |
| 3.2 | Bidirectional sync | A→push, B→push, A→sync, B→sync | Both see all entries |
| 3.3 | Sync --migrate with filters | `--perspective`, `--exclude-perspective`, `--since` | Only matching entries migrated |
| 3.4 | Sync --pending | After storing without push | Shows unpushed entry count (not commit count) |
| 3.5 | Sync --dry-run | `sync --dry-run` | Reports what would happen, NO side effects (no pull) |
| 3.6 | Sync --stage | Stage entry by ID | Entry added to git branch |
| 3.7 | Sync --reject | Remove entry by ID | Entry removed from git branch |
| 3.8 | Diverged repos | Both store without sync, one pushes first | Clear error with resolution steps; sync resolves |
| 3.9 | Identical content from both clients | Same text stored by A and B | Auto-resolved (no false conflict) |
| 3.10 | Bulk sync (50+ entries) | Store 50 entries, migrate, push, sync | Completes in reasonable time |
| 3.11 | Sync after branch deletion | Delete veclayer-memory, then sync | Recovers from remote |
| 3.12 | Sync scope filtering | `--scope <name>` | Only specified scope synced; clear error for unknown scope |
| 3.13 | Migrate then push without intermediate sync | `sync --migrate` → `sync --push` | Either auto-pulls or warns about stale state |

## 4. Error Handling & Edge Cases

| # | Test Case | Steps | Expected |
|---|-----------|-------|----------|
| 4.1 | Commands without init | store/recall/sync in plain directory | Clear indication of which store is targeted (not silent global fallback) |
| 4.2 | Corrupted config.toml | Write invalid TOML | Graceful error with fix guidance (not panic) |
| 4.3 | Readonly data directory | `chmod 555 .veclayer/` | Clear permission error with path |
| 4.4 | Invalid IDs | `focus ""`, `focus zzz`, `sync --stage 000` | Clear errors, exit 1 |
| 4.5 | Concurrent stores | 5 parallel `store` processes | All succeed or clear error per-process |
| 4.6 | Worktree cache deleted | `rm -rf ~/.cache/veclayer/` → store | Auto-recovers (prune stale worktrees) |
| 4.7 | Same content, different headings | Multiple entries with identical content | Distinct files on git branch, no ID collision |
| 4.8 | Extremely long heading (300+ chars) | Store with very long heading | Filename truncated, full heading preserved in file |
| 4.9 | Nonexistent parent-id | `--parent-id 0000000` | Clear error, exit 1 |
| 4.10 | Double init preserves data | `init --share` → store → `init --share` | Entries preserved |
| 4.11 | Consistent exit codes | All error paths | Exit 1 for errors, exit 0 for success (not 101) |

## 5. Data Portability & Integrity

| # | Test Case | Steps | Expected |
|---|-----------|-------|----------|
| 5.1 | Export/import roundtrip | Store entries → `export` → fresh store → `import` | All fields preserved |
| 5.2 | Pipe export to import | `export | import -` | Works via stdin |
| 5.3 | Import malformed JSONL | Feed invalid JSON lines | Partial import with clear per-line errors |
| 5.4 | Merge between stores | `merge /path/to/other/.veclayer` | Entries from other store available |
| 5.5 | Rebuild index | Delete lance files → `rebuild-index` | All entries recovered from blob store |
| 5.6 | Git branch vs LanceDB consistency | Compare entry counts | Counts match after sync --migrate |
| 5.7 | Archive and recall visibility | Archive entry → recall → recall --deep | Normal recall excludes, deep includes |
| 5.8 | Think promote/demote | `think promote <id>`, `think demote <id>` | Visibility changes correctly |
| 5.9 | Think relate | `think relate <id1> <id2> related_to` | Relation created |
| 5.10 | Sync preserves metadata | Push from A, sync to B | Perspectives, types, visibility all preserved |
| 5.11 | History command | `history <id>` | Shows version/relation chain |

## Running the Tests

```bash
# Build the binary
cargo build

# Run automated integration tests
cargo test --test git_review_integration

# Manual CLI tests use temp directories
export VL=./target/debug/veclayer
export TESTDIR=$(mktemp -d)
```
