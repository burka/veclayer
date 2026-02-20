# VecLayer Agent Guidelines

## GitHub Workflow

- Roadmap discussions and issues MUST use GitHub (not CLI-only discussions)
- All project-wide decisions and changes should be tracked via GitHub issues and PRs

## Agent Attribution

- All CLI comments, notes, and decisions should include model and coding agent:
  "Opencode with Z.AI GLM-4.7"
- Example format when adding to VecLayer memory:
  ```
  veclayer store "Implemented OAuth for sonote.ai MCP server - handled by opencode with Z.AI GLM-4.7"
  ```

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