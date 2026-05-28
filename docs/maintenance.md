# Store Maintenance — keeping the data dir clean & self-healing

VecLayer stores data in LanceDB, which keeps a **version manifest** for every
write (MVCC). Without pruning, those manifests — and the data fragments they
reference — accumulate forever. A high write rate (e.g. the `observe`
PostToolUse hook firing on every tool call) can grow the store to tens of GiB.

## How self-healing works

Pruning is built in and runs automatically. Three layers:

1. **Auto-compaction on write.** After an insert releases its write lock,
   the store checks its version count. Once more than **3 old versions**
   accumulate (beyond the 3-version safety margin it always keeps), it runs a
   compaction pass. The pass is **bounded to 50 versions** so it always
   finishes quickly — even inside a short-lived process like the `observe`
   hook — and a large backlog drains over successive writes instead of one
   long pass that could be killed mid-flight. This is non-blocking: it runs
   off the write path, so it never stalls your `store`/`recall`.

2. **Daily background compaction.** A long-running `serve` process compacts
   once every 24h, so even an idle-but-fragmented store self-cleans.

3. **One-time recovery on open.** If a store is opened with a wildly excessive
   version count (>500), it kicks off a bounded compaction in the background.

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
that: every pass completes, so the store can never run away again.
