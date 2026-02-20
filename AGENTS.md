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

## Agent Attribution

Commit messages should include attribution via `Co-Authored-By`:

```
<message body>

Co-Authored-By: <tool> - <model>
```

| Tool | Attribution |
|------|-------------|
| Claude Web - Opus 4.6 | `Co-Authored-By: Claude Web - Opus 4.6 <noreply@anthropic.com>` |
| Claude Code - Opus 4.6 | `Co-Authored-By: Claude Code - Opus 4.6 <noreply@anthropic.com>` |
| Claude Code - Sonnet 4.6 | `Co-Authored-By: Claude Code - Sonnet 4.6 <noreply@anthropic.com>` |
| Claude Code - Haiku 4.5 | `Co-Authored-By: Claude Code - Haiku 4.5 <noreply@anthropic.com>` |
| Opencode - Z.AI GLM-4.7 | `Co-Authored-By: Opencode - Z.AI GLM-4.7 <noreply@z.ai>` |
| Florian Burka | Human author (no attribution line) |

## Pull Request Workflow

```bash
# 1. Create feature branch
git checkout -b feature-name

# 2. Work and commit
git add <files>
git commit -m "<type>: <description>"
# (include Co-Authored-By for attribution)

# 3. Push to remote
git push -u origin feature-name

# 4. Create PR with detailed description
gh pr create --title "feat: ..." --body "<detailed description>"
```

**PR Body Template:**
```markdown
## Summary
Brief description of what this PR does.

## Changes
- Change 1: description
- Change 2: description

## Test Coverage
- What was tested
- Test results

## Example Usage
\`\`\`bash
# CLI example
veclayer <command> ...

# MCP example
{...}
\`\`\`

Fixes #<issue-number>

— <tool> - <model>
```

**Commit Message Format:** `<type>: <description>`
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code refactoring
- `test`: Tests only
- `chore`: Build/config changes
