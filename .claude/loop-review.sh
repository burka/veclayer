#!/bin/bash
# Review loop — every 30m, review + fix + commit
cd /home/flob/work/veclayer

ITER=0
while true; do
  ITER=$((ITER + 1))
  echo "[+loop] iteration $ITER at $(date)"

  claude --agent reviewer --dangerously-skip-permissions \
    "Review the whole codebase. Fix any issues you find and make a commit. Apply /simplify where possible. Apply SRP, DRY, SOLID, and clean code principles throughout." \
    2>&1 | tail -30

  echo "[+loop] iteration $ITER done at $(date), sleeping 30m"
  sleep 1800
done
