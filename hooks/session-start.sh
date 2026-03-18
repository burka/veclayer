#!/usr/bin/env bash
# SessionStart hook — inject recent memory context into the conversation.
# Outputs an identity briefing to stdout, which Claude Code injects
# as context at the start of the session.
#
# Install: add to .claude/settings.json under hooks.SessionStart
# See: .claude/settings.json.example

exec veclayer context
