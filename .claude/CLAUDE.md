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