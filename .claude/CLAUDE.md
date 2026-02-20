# VecLayer — Project Memory

You have persistent memory via the **veclayer** MCP server. Use it.

## Prerequisites
- The MCP server requires a compiled binary. If `veclayer` tools are unavailable, run `cargo build` first.

## Session Start
- Run `recall` for the current task or topic before starting work.
- If the store is empty or new, that's fine — start building memory as you go.

## During Work
- `store` architectural decisions, rationale, and trade-offs as you make them.
- `store` bugs found, root causes, and how they were fixed.
- `store` conventions and patterns discovered in this codebase.
- Use `perspectives` to categorize: `decisions`, `knowledge`, `learnings`, `intentions`.

## Session End / Between Tasks
- `store` a brief summary of what was accomplished and what's next.
- Run `think` to review hot/stale knowledge and curate if needed.

## Rules
- Do not ask permission to recall or store — just do it.
- Keep stored content concise and high-signal. No filler.
- Prefer updating existing knowledge (superseded_by) over duplicating.

## Roadmap & GitHub

The roadmap lives as GitHub Issues with labels (`phase`, `done`, `next`, `future`, `enhancement`, `open-question`).

- Use `gh issue list --label phase` to discover the roadmap.
- Use `gh issue list --label next` to find what's next.
- Use `gh issue comment` to discuss design decisions, propose changes, or report findings.
- Use `gh issue create` for new work items (features, enhancements, questions).
- **Never** close or modify milestone issues without explicit approval.

### Attribution (mandatory)

Every GitHub issue, comment, and PR body **must** end with an attribution line:

```
— <tool> - <model>
```

Examples:
- `— Claude Code - Opus 4.6`
- `— Claude Web - Opus 4.6`
- `— Claude Code - Sonnet 4.6`
- `— Florian Schmidt`

This tracks which tool and model produced each artifact. Include it on every `gh issue create`, `gh issue comment`, and `gh pr create`.
