#!/usr/bin/env bash
# PostToolUse hook — auto-capture tool observations into veclayer memory.
# Reads the Claude Code PostToolUse JSON payload from stdin and stores
# a compact observation. Silently skips noise tools (Read, Grep, etc.).
#
# Install: add to .claude/settings.json under hooks.PostToolUse
# See: .claude/settings.json.example

exec veclayer observe < /dev/stdin
