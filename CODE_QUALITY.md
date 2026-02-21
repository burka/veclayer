# Code Quality

One-liner: Shorter is better. Optimize for the reader.

## Principles
- **SRP**: One module, one reason to change.
- **SSOT**: One canonical source for every piece of data.
- **KISS**: Simple solutions beat clever ones.
- **YAGNI**: Don't build it until you need it.
- **Fail fast, fail loud, fail helpful**: Surface errors immediately, never swallow them, make the message tell you what to fix.

## Quality Gate
0 errors, 0 warnings, 0 test failures. Must pass before merge:
- `cargo fmt --check`
- `cargo clippy --all-targets`
- `cargo build --tests`
- `cargo nextest run`

## Code Expectations
- Functions < 30 lines, cognitive complexity < 15
- Intention-revealing names, no magic numbers
- Prefer immutable data
- No dead code, no unused exports
- Error handling at system boundaries only

## Test Quality

A test only counts if it:
- **Asserts real outcomes** — not just "doesn't panic"
- **Fails on wrong results** — break the logic; if the test stays green, it's worthless

Every public function needs tests covering three dimensions:
- **Green path** — expected inputs produce expected outputs
- **Edge cases** — boundaries, empty inputs, zero, extremes
- **Error path** — invalid inputs, missing data, broken state

### Rust Test Patterns
- Use `#[test]` for sync, `#[tokio::test]` for async
- Name tests descriptively: `test_<function>_<scenario>` (e.g. `test_search_by_embedding_excludes_target`)
- Use the existing `MockStore`/`MockEmbedder` for search tests
- `#[ignore]` only for tests requiring external services (ONNX, Ollama)
- Never estimate coverage — measure with `cargo tarpaulin` or state "coverage unknown"

### What counts as tested
- Build compiles ≠ tested
- "Runs without panic" ≠ tested
- Assert on the actual result = tested
- Would fail if the logic broke = tested

## Naming
- Intention-revealing, domain-accurate
- No forbidden patterns: Smart*, Unified*, Better*, V2*, Helper*, Util*, Generic*
- Improve names in place — never copy and rename