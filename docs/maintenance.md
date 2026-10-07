# Store Maintenance — keeping the data dir clean & self-healing

VecLayer stores data in LanceDB, which keeps a **version manifest** for every
write (MVCC). Without pruning, those manifests — and the data fragments they
reference — accumulate forever. A high write rate (e.g. the `observe`
PostToolUse hook firing on every tool call) can grow the store to tens of GiB.

## How self-healing works

Pruning is built in and runs automatically. Two kinds of work happen:

- **Version pruning** drops old version manifests and the data files only they
  reference. It is cheap metadata work plus file deletes.
- **Fragment compaction** rewrites data fragments to merge small files and
  materialize deletions. On a large store it rewrites hundreds of MB and can
  take well over 10 seconds.

Who does what:

| Context | Prunes versions | Compacts fragments |
|---|---|---|
| Any write (`observe`, `store`, other one-shot CLI writes) | yes, bounded | **never** |
| Opening a store with >500 versions | yes, bounded, in the background | **never** |
| `serve` startup (MCP server) | yes | yes |
| `serve` daily background worker (every 24h) | yes | yes |
| `veclayer prune` / `veclayer reflect prune` | yes | yes |
| `scripts/veclayer-prune.sh` (cron / loop) | yes | yes (runs `reflect prune`) |

1. **Auto-prune on write.** After an insert or batch update releases its write
   lock, the store checks its version count. Once more than **3 old versions**
   accumulate (beyond the 3-version safety margin it always keeps), it prunes
   them. The pass is **bounded to 50 versions** so it finishes quickly, and a
   large backlog drains over successive writes. It never compacts fragments:
   writes run in short-lived processes like the `observe` hook, which is killed
   after a timeout, and a killed fragment rewrite leaks a staging file (see
   [Leaked upload files](#leaked-upload-files-uuidlancen)). Fragments written
   between compactions just stay small until a long-lived process merges them.

2. **Compaction in long-lived processes.** `serve` compacts on startup and
   then once every 24h, so a store used from an MCP session stays compacted and
   an idle-but-fragmented store self-cleans.

3. **One-time recovery on open.** If a store is opened with a wildly excessive
   version count (>500), it starts a bounded prune in the background.

## Manual prune

To reclaim space on demand, drain the whole backlog with progress output:

```bash
veclayer -d <store> reflect prune
# Compact: fragments + versions
#   pass 1: 50 versions pruned, 1.8 GB reclaimed (46174 fragments merged)
#   pass 2: 12 versions pruned, 30 MB reclaimed (0 fragments merged)
# Done after 2 pass(es): ...
```

This keeps the newest 3 versions and removes the rest, plus their orphaned
data files. It is safe to run any time and idempotent on a clean store.

Before compacting, prune also removes **leaked upload files** — see below.
This runs first so a nearly full disk regains the space compaction needs.

## Leaked upload files (`<uuid>.lance#N`)

LanceDB writes each data file through `object_store`, which stages it as
`data/<uuid>.lance#1` and renames it to `<uuid>.lance` when the write
completes. A writer killed mid-write leaves the staging file behind. The main
source used to be the `observe` hook: its post-write auto-compaction rewrote
the whole ~300 MB fragment, took longer than the hook's 10 s timeout, and
leaked a staging file on every tool call. Writes now only prune versions (see
above), so this no longer happens on the write path. `object_store` hides `#<digits>` names from directory listings,
so Lance's own version cleanup never sees them and they pile up — one store
reached 100+ GiB of them next to ~300 MB of live data.

Prune (and the daily `serve` compaction) deletes such a file only when its
name is exactly `<uuid>.lance#<digits>`, it lives in the table's `data/`
directory, it is older than 1 hour (so in-flight writes are never touched),
and its `<uuid>.lance` name appears in no manifest or transaction file. The
summary line `Leaked uploads: N removed, X reclaimed` reports the result.

## Scheduled / looped cleanup with disk guard

`scripts/veclayer-prune.sh` wraps the native prune with a disk-space guard for
unattended use:

```bash
# Prune the default global store (~/.local/share/veclayer)
scripts/veclayer-prune.sh

# Prune specific stores (e.g. a project-local one too)
scripts/veclayer-prune.sh ~/.local/share/veclayer ./.veclayer
```

It first runs `reflect prune`. If free disk is **still below 10 GiB**
afterward (e.g. the native prune lost a race for the write lock), it falls
back to deleting old version manifests directly, keeping the newest 100.
The fallback only ever touches `_versions/*.manifest*` snapshots — never the
live `data/` fragments — so current data is always preserved.

Tunables via environment: `MIN_FREE_GIB` (default 10), `KEEP_MANIFESTS`
(default 100), `VECLAYER_BIN` (override the binary).

### Run it on a loop

In Claude Code, keep the store clean continuously:

```
/loop 30m scripts/veclayer-prune.sh
```

Or schedule it with cron / systemd timer:

```cron
*/30 * * * * /path/to/veclayer/scripts/veclayer-prune.sh >> ~/.cache/veclayer/prune.log 2>&1
```

## Why the store once grew to 100+ GiB

Historically auto-compaction only triggered at 50 versions and then tried to
compact the entire backlog in **one unbounded pass**. When that pass ran inside
the short-lived `observe` hook process, the process exited before it finished,
so nothing was reclaimed — fragments and 5 MiB manifests piled up until the
disk filled. The bounded, early-triggering, non-blocking design above fixes
that for version manifests.

Later, a second leak appeared: the write path also compacted fragments, and a
fragment rewrite cannot be bounded the same way. In the `observe` hook it was
killed partway through, leaking a 200-300 MB staging file per tool call, about
100 GiB within minutes of active use. Fragment compaction now runs only in
`serve` and in explicit prunes, and prune sweeps any staging files still
left over (see above).
