#!/usr/bin/env bash
# veclayer-prune.sh — keep the veclayer data dir clean and self-healing.
#
# Strategy (least-destructive first):
#   1. Run veclayer's own `reflect prune` (compacts fragments, prunes old
#      LanceDB versions, keeps the newest 3). This is the normal, safe path.
#   2. Disk guard: if free space on the store's filesystem is still below
#      MIN_FREE_GIB, fall back to deleting old version manifests directly,
#      keeping the newest KEEP_MANIFESTS. Manual deletion only touches
#      `_versions/*.manifest*` (snapshots), never the live `data/` fragments.
#
# The native prune is bounded per pass (50 versions) but loops until drained,
# so a single invocation fully cleans the backlog. The manual fallback exists
# only for the emergency where the disk is so full the native prune cannot run.
#
# Designed to be run on a schedule or via Claude Code `/loop`. Idempotent:
# on a clean store it does almost nothing and exits 0.
#
# Usage:
#   scripts/veclayer-prune.sh [STORE_DIR ...]
#
# With no arguments it prunes the default global store. Pass one or more store
# directories to prune those instead (e.g. a project-local .veclayer).
#
# Environment:
#   VECLAYER_BIN     veclayer binary to use (default: target/release/veclayer
#                    if present, else `veclayer` from PATH)
#   MIN_FREE_GIB     disk-guard threshold in GiB (default: 10)
#   KEEP_MANIFESTS   manifests to keep in the manual fallback (default: 100)
set -euo pipefail

MIN_FREE_GIB="${MIN_FREE_GIB:-10}"
KEEP_MANIFESTS="${KEEP_MANIFESTS:-100}"

# Resolve the veclayer binary: prefer a freshly built release binary in this
# repo, fall back to whatever is on PATH.
resolve_bin() {
  if [ -n "${VECLAYER_BIN:-}" ]; then echo "$VECLAYER_BIN"; return; fi
  local here repo
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo="$(dirname "$here")"
  if [ -x "$repo/target/release/veclayer" ]; then
    echo "$repo/target/release/veclayer"
  else
    echo "veclayer"
  fi
}
BIN="$(resolve_bin)"

# Default global store, matching directories::ProjectDirs data_local_dir.
default_store() {
  echo "${XDG_DATA_HOME:-$HOME/.local/share}/veclayer"
}

# Free GiB on the filesystem holding $1 (integer, floored).
free_gib() {
  df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'
}

log() { printf '[veclayer-prune] %s\n' "$*"; }

# Manual fallback: delete old version manifests, keep the newest $KEEP_MANIFESTS.
# Only removes snapshot manifests, never live data fragments.
manual_prune() {
  local store="$1" versions="$1/chunks.lance/_versions"
  if [ ! -d "$versions" ]; then
    log "  no _versions dir at $versions — nothing to fall back to"
    return 0
  fi
  local before floor del=0
  before="$(find "$versions" -maxdepth 1 -name '*.manifest*' | wc -l | tr -d ' ')"
  log "  manual fallback: $before manifests, keeping newest $KEEP_MANIFESTS"
  # Distinct version numbers, newest first; the KEEP_MANIFESTS-th is the floor.
  floor="$(find "$versions" -maxdepth 1 -name '*.manifest*' -printf '%f\n' \
    | sed 's/\.manifest.*$//' | grep -E '^[0-9]+$' | sort -rn | uniq \
    | sed -n "${KEEP_MANIFESTS}p")"
  if ! printf '%s' "$floor" | grep -qE '^[0-9]+$'; then
    log "  fewer than $KEEP_MANIFESTS versions — nothing to delete"
    return 0
  fi
  while IFS= read -r f; do
    local v="${f%%.manifest*}"
    case "$v" in ''|*[!0-9]*) continue ;; esac
    if [ "$v" -lt "$floor" ]; then rm -f -- "$versions/$f" && del=$((del + 1)); fi
  done < <(find "$versions" -maxdepth 1 -name '*.manifest*' -printf '%f\n')
  log "  manual fallback removed $del old manifest(s)"
}

prune_store() {
  local store="$1"
  if [ ! -d "$store" ]; then
    log "store not found, skipping: $store"
    return 0
  fi
  local free
  free="$(free_gib "$store")"
  log "store: $store (${free:-?} GiB free)"

  # 1. Native, safe prune. Drains the whole backlog in bounded passes.
  if "$BIN" -d "$store" reflect prune; then
    log "  native prune ok"
  else
    log "  native prune failed (lock contention or OOM) — checking disk guard"
  fi

  # 2. Disk guard: only if still critically low after the native attempt.
  free="$(free_gib "$store")"
  if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE_GIB" ]; then
    log "  WARNING: ${free} GiB free < ${MIN_FREE_GIB} GiB threshold — manual fallback"
    manual_prune "$store"
  fi

  free="$(free_gib "$store")"
  log "  done: ${free:-?} GiB free"
}

main() {
  log "binary: $BIN ($("$BIN" --version 2>/dev/null || echo 'version unknown'))"
  if [ "$#" -eq 0 ]; then
    prune_store "$(default_store)"
  else
    for s in "$@"; do prune_store "$s"; done
  fi
  log "all stores processed"
}

main "$@"
