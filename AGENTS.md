# VecLayer — Agent Conventions

## Identity & Attribution

Every agent interaction that produces a GitHub artifact (issue, comment, PR) **must** include attribution:

```
— <tool> - <model>
```

| Tool | When |
|------|------|
| `Claude Web - Opus 4.6` | Conversations in claude.ai web interface |
| `Claude Code - Opus 4.6` | Claude Code CLI sessions |
| `Claude Code - Sonnet 4.6` | Claude Code CLI with Sonnet model |
| `Claude Code - Haiku 4.5` | Claude Code CLI subagents using Haiku |
| `Opencode - Z.AI GLM-4.7` | Opencode CLI sessions |
| `Florian Burka` | Human author |

This allows us to trace which model and tool produced each decision, design, or code change.

## Roadmap

The roadmap is tracked as GitHub Issues:

```bash
# View the full roadmap
gh issue list --label phase

# What's next?
gh issue list --label next

# What's done?
gh issue list --label done

# Open design questions
gh issue list --label open-question

# Discuss a roadmap item
gh issue comment <number> --body "..."
```

### Labels

| Label | Color | Purpose |
|-------|-------|---------|
| `phase` | green | Roadmap phase (1-7) |
| `done` | grey-green | Phase completed |
| `next` | red-orange | Next for implementation |
| `future` | light blue | Planned for later |
| `enhancement` | teal | Enhancement to existing phase |
| `open-question` | purple | Design question needing resolution |

### Rules

- One issue per phase, with a checklist of deliverables
- Enhancements and open questions get their own issues, linked to the parent phase
- Close phase issues only when all checklist items are done and verified
- Use comments for design discussion — keep the issue body as the source of truth
- Update issue body when decisions are made (don't let decisions get buried in comments)

## Memory Capture

- Use VecLayer perspectives to track decisions, learnings, and ongoing work
  - `decisions`: Architectural choices, configuration decisions
  - `learnings`: What worked/didn't, insights gained
  - `intentions`: Goals, upcoming work, ongoing threads
- Priming: Use `veclayer id` and `veclayer reflect` on session start
- Store significant moments automatically during work

## MCP Servers

- **veclayer**: Local hierarchical vector database for long-term memory
- **sonote.ai**: Remote MCP server for analysis and document storage (OAuth authenticated)

## Coordination

When multiple agents work on the same codebase:
- Check `gh issue list --label next` before starting work
- Comment on the issue you're working on to signal intent
- Reference issue numbers in commit messages (`fixes #N`, `ref #N`)
- Do not close issues without explicit human approval
