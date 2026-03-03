# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-03-03

### Bug Fixes
- [`5cb97cb`](https://github.com/burka/veclayer/commit/5cb97cbbc5807e58711dd9cf2794b5d120149d8f) Cleanup PR #25 - remove dead code, fix doc typo, add with_min_score builder ([#25](https://github.com/burka/veclayer/issues/25))
- [`77c0e1b`](https://github.com/burka/veclayer/commit/77c0e1beeafcd4fd58d6566d7521981462af4bb2) Resolve short ID prefixes consistently across all MCP tool inputs
- [`6e242e3`](https://github.com/burka/veclayer/commit/6e242e34c7e2ecb9a0041e46e5809bea77dc9254) Include session in MCP tool perspective descriptions
- [`64ed783`](https://github.com/burka/veclayer/commit/64ed78388eeadadc3f146b7bb117c8b6eb5ed9c0) Rename helpers→resolve, fix MCP relate bidirectionality, update docs
- [`4aba45b`](https://github.com/burka/veclayer/commit/4aba45b92b09369c04f5329d5fa4537df27a5013) *(#53)* Return short IDs in MCP store response
- [`de0fea8`](https://github.com/burka/veclayer/commit/de0fea838dcf4de5e5cb353a2b791c60186a8fad) Resolve rebase conflicts and add missing struct fields
- [`03e0e3f`](https://github.com/burka/veclayer/commit/03e0e3f350a2a38e35069b504ff4dce475e92165) Handle 3-part reference format in parse_references
- [`2f48e4c`](https://github.com/burka/veclayer/commit/2f48e4ccfae08cd021774b1a92ca17b048b0a477) *(#62)* Accept string-or-array for Vec<String> MCP parameters
- [`bf91907`](https://github.com/burka/veclayer/commit/bf9190725ae9b5ee95d1e904ac8ed668be042142) CLI UX bugs - duplicate perspectives, German hints, comma-split, archive guard
- [`0e95aac`](https://github.com/burka/veclayer/commit/0e95aac63149542489c6a7940990dd8b7dc62b6a) Per-write lancedb lock for concurrent access
- [`bba29af`](https://github.com/burka/veclayer/commit/bba29afb17a46e1ece878c2e1bbfa98f9f19e78e) Stop project discovery walk-up at $HOME, add -P alias for --project
- [`158ec48`](https://github.com/burka/veclayer/commit/158ec487cea6622a07bd493b53144339d3c691e8) Pre-publish hardening - security, metadata, and UX improvements
- [`7f16e46`](https://github.com/burka/veclayer/commit/7f16e4686ca2fe99d4cdbaa3b2f4fac7b3a9f6ec) Protect schema migration with write lock, allow concurrent store access (#70) ([#70](https://github.com/burka/veclayer/issues/70))
- [`b37c3d4`](https://github.com/burka/veclayer/commit/b37c3d44031489c83d33e39efb8fabea04e357c7) Pre-publish hardening - HTTP limits, LLM timeouts, API errors, CLI help
- [`7e110d4`](https://github.com/burka/veclayer/commit/7e110d4f7f18eb21753af34b8b8c655ba9639f0d) Resolve CVEs by updating transitive deps, drop atomic-polyfill


### Documentation
- [`8a17404`](https://github.com/burka/veclayer/commit/8a174046dd2abdda08aeb12c2f769150d14960ce) Update README + ROADMAP to reflect Phase 1-4 completion
- [`0f2c0cd`](https://github.com/burka/veclayer/commit/0f2c0cdd238127992e9029965555720df9864926) Update "6 defaults" → "7 defaults" after session perspective
- [`00c55b7`](https://github.com/burka/veclayer/commit/00c55b71a87707ee4cb3b6a697943057e90c1de4) Sync README with current state, add design anti-patterns section
- [`a9182f0`](https://github.com/burka/veclayer/commit/a9182f04e06ea110cbc22aa5ad57efe896734067) Consolidate and translate all documentation to English
- [`bb36016`](https://github.com/burka/veclayer/commit/bb36016fafb2495e1ab2fd7aef80710f99372d69) Add CODE_QUALITY.md with testing standards, link from AGENTS.md
- [`b81fa90`](https://github.com/burka/veclayer/commit/b81fa90f86db1b9985aaffe9d17eebabf748f25b) *(#11)* Add Reasoning Pattern to MCP priming instructions
- [`3f09802`](https://github.com/burka/veclayer/commit/3f09802d097cfb4dc58e0644647362a3b32e56cd) *(#69)* Add MCP server setup and multi-project configuration
- [`70ad524`](https://github.com/burka/veclayer/commit/70ad524cb538f459bcfb1729a7a3a014debe16ee) Update agent attribution rules
- [`1f02fe2`](https://github.com/burka/veclayer/commit/1f02fe2a49fa79a8ad87363563ba8090c717b2ef) Public API cleanup, module docs, doc-tests, risk documentation


### Features
- [`293f9ed`](https://github.com/burka/veclayer/commit/293f9ed8000c857ff33804c6200b60f92de86646) Add min_score and min_salience threshold filters to search
- [`b9cd9d1`](https://github.com/burka/veclayer/commit/b9cd9d1606ef4ad764e24d5fa1014042a223c0e9) Short ID prefix resolution for focus/get operations
- [`fc27457`](https://github.com/burka/veclayer/commit/fc27457a0b59ae2a3d191fc8ea35474ed35abf79) Markdown formatting for MCP results and improved logging
- [`62c1dd2`](https://github.com/burka/veclayer/commit/62c1dd2c05793538b7c36e3363de45e6dd3174be) Unify CLI commands - reflect/think split, browse, temporal filters
- [`778291c`](https://github.com/burka/veclayer/commit/778291c7dbef5383d12b148b309fe229f7f3918a) Add perspectives, status, history actions to MCP think tool
- [`e9c3cbf`](https://github.com/burka/veclayer/commit/e9c3cbfa23ea9aece73a7667354838c2fd9c5889) *(#39)* Add impression_hint and impression_strength to memory entries
- [`997397a`](https://github.com/burka/veclayer/commit/997397a0dbf1f11617615e724a9a21768244916d) *(#38)* Wire k-means clustering into identity emergence
- [`a6aed42`](https://github.com/burka/veclayer/commit/a6aed42ff52e02e5e13396bdebc93fe7f22fe951) *(#40)* Add GET /api/identity HTTP endpoint
- [`0b21d18`](https://github.com/burka/veclayer/commit/0b21d187f91006cd4d29882e49e521ce3983bc1b) *(#43)* Add ANSI color formatting to CLI with owo-colors
- [`83cfbb9`](https://github.com/burka/veclayer/commit/83cfbb9cbcae1181b769e920ed5716040c4ec281) *(#42)* Show perspectives and visibility in CLI search/browse/focus
- [`99a4b15`](https://github.com/burka/veclayer/commit/99a4b1539f0d0e9e7ed099363eced332bb1ed569) *(#52)* Add GitHub Actions CI pipeline
- [`7864f78`](https://github.com/burka/veclayer/commit/7864f781f3d9d5e77ddcb3fc5f8e7197f96e30b0) *(#55)* Add file-based advisory lock for single-writer safety
- [`351116d`](https://github.com/burka/veclayer/commit/351116d24c4c27faa8aefb19b8a8872da84d23fe) Markdown formatting for MCP results and improved logging
- [`946f657`](https://github.com/burka/veclayer/commit/946f6574c4913d887889405da46145492107cdab) *(#18)* Add recall --similar-to for related entry discovery
- [`40dc595`](https://github.com/burka/veclayer/commit/40dc595132629777ba072c31166ab5a6de51d73d) Add --references CLI flag for universal relations
- [`05a57b0`](https://github.com/burka/veclayer/commit/05a57b0cf09d9802bb428d39aaa0d53408f650f6) Unified --rel-* CLI relation flags with shared processing module
- [`36bef1c`](https://github.com/burka/veclayer/commit/36bef1ce9cae5eaa224b254f66771b2251129745) *(#18)* Recall --similar-to, unified relations, CLI polish
- [`3ce787a`](https://github.com/burka/veclayer/commit/3ce787a603103628b0d817eb99d6787c823314aa) Auto-migrate LanceDB schema on startup (#54) (#61) ([#54](https://github.com/burka/veclayer/issues/54))([#61](https://github.com/burka/veclayer/issues/61))
- [`58f0d76`](https://github.com/burka/veclayer/commit/58f0d76b113345f47446d80f4a6f3db4ebeca4b4) *(#64)* Align CLI naming with MCP - recall/store as primary commands
- [`217b58e`](https://github.com/burka/veclayer/commit/217b58ea5cc9155c455f6f0ebccb72b706b378de) *(#19)* Recall --ongoing filter for open threads
- [`f1221e4`](https://github.com/burka/veclayer/commit/f1221e4b0d519d1b85a75ba3f8baba5bc4672601) *(#40)* Add GET /api/priming endpoint for HTTP transport
- [`7a035ff`](https://github.com/burka/veclayer/commit/7a035ff4194ce439dfb0b79ba3e839c4feee313d) *(#57)* Add think(action='discover') - find similar-but-unlinked entries
- [`94ad8c2`](https://github.com/burka/veclayer/commit/94ad8c28a19ae50b9c93d747a326fe9e61dcbf9f) Content-addressed blob store
- [`dfa0a4e`](https://github.com/burka/veclayer/commit/dfa0a4ecbc92e80a237111a162ab86c27f28e606) *(#56)* Define SyncBackend + NameResolver trait boundaries
- [`aa07a57`](https://github.com/burka/veclayer/commit/aa07a5788b045214898e55f315ebc5e52377b267) *(#68)* Add project-scoped memory isolation with --project flag
- [`09fd555`](https://github.com/burka/veclayer/commit/09fd555aa4251c2fac85646379df8eda15786524) Platform-aware default data directory
- [`9690fb8`](https://github.com/burka/veclayer/commit/9690fb8617d465816140b5117a0eac7bbab58cc0) Lock timeout (2s) and project discovery via .veclayer/config.toml
- [`9b6cd34`](https://github.com/burka/veclayer/commit/9b6cd34f6a60ac7efd16f0de0eea0abe4455d80f) Git auto-detect project/branch, scope: branch, cross-branch awareness
- [`7defe39`](https://github.com/burka/veclayer/commit/7defe39fd7305cdf14b243b7cf27956e0a2c95e7) User config with [[match]] overrides (path glob + git-remote regex)
- [`84d30a7`](https://github.com/burka/veclayer/commit/84d30a7bf453edb242380d24b00d0b979e3d9b6d) Veclayer merge <source> - project-aware blob merge between stores
- [`4551f8c`](https://github.com/burka/veclayer/commit/4551f8c4f5b18bc1c7584027acbfa27c640e041d) OllamaEmbedder, config-based embedder factory, ROADMAP Phase 6
- [`57d6641`](https://github.com/burka/veclayer/commit/57d664106c01f99d5b9eeb3008bbc0862946ff90) Guide agents on content size and batch limits via tool hints
- [`d8f59f0`](https://github.com/burka/veclayer/commit/d8f59f0a75b9dafdf19c25940393cb08034d7ff0) Async embedding pipeline - background worker, pending annotations, queue stats


### Maintenance
- [`4071ff5`](https://github.com/burka/veclayer/commit/4071ff5f7d6f635af1e78102bb82758b855b353e) Remove dead identity() function after reflect/id merge
- [`6336b95`](https://github.com/burka/veclayer/commit/6336b9555a0432b43f7936b049dd787c4b4c80b4) Apply cargo fmt and commit Cargo.lock for supports-colors dependency
- [`9e52e4b`](https://github.com/burka/veclayer/commit/9e52e4b4df19c61cc8bfa1cc5102a3ba10aadf64) Gitignore .gitmessage.txt (local config file)
- [`d227646`](https://github.com/burka/veclayer/commit/d227646fd82354e31f3885403a10b951deecaab9) Use veclayer mcp server for claude code
- [`d0f473a`](https://github.com/burka/veclayer/commit/d0f473acc0c32ba4054b8c188cb962001ff2710b) Gitignore *.rlib compiler artifacts
- [`c406ac9`](https://github.com/burka/veclayer/commit/c406ac9b47be1202ed05fc47b1c1757ff0e072ed) Add TODOs for deferred review findings


### Other
- [`ff6cac5`](https://github.com/burka/veclayer/commit/ff6cac51221b96bb1cf9423921673d952001636b) Initial VecLayer prototype - hierarchical vector indexing for documents
- [`1ee3e8d`](https://github.com/burka/veclayer/commit/1ee3e8dd3b5ad8874e58de7dbc23492fda3dcaad) Add RAPTOR-style clustering and summarization
- [`b6f347b`](https://github.com/burka/veclayer/commit/b6f347bb8f20e45d5a5483f37c88ff4761b3e538) Add comprehensive LanceStore test coverage and reduce boilerplate
- [`4daa358`](https://github.com/burka/veclayer/commit/4daa358e35ba4fec6402497b29f2bf5dded0d456) Improve code quality: DRY refactoring and test coverage
- [`a78d64a`](https://github.com/burka/veclayer/commit/a78d64a86754dda05831806ada70f19dfd149527) Refactor main.rs: extract command logic to library module
- [`c0dc626`](https://github.com/burka/veclayer/commit/c0dc6262d5f2e06aa083e630bc3a5799d6accb16) Add comprehensive unit tests for core modules
- [`bf4aecc`](https://github.com/burka/veclayer/commit/bf4aecc9052054f9236a2b9b3d81d2cb034fad5f) Add Ollama integration tests
- [`57e344b`](https://github.com/burka/veclayer/commit/57e344bcf7b9d0270370183e313402953eace70e) Add README with architecture and usage docs
- [`1de22b5`](https://github.com/burka/veclayer/commit/1de22b573a03b4d357284c9442c8f37401887ada) Integrate concept document into roadmap and documentation
- [`02c8833`](https://github.com/burka/veclayer/commit/02c883314a5e8122b5da04b48b5c058a031cfb6b) Add identity & memory data model, reframe project as agent identity store
- [`522963f`](https://github.com/burka/veclayer/commit/522963ff5d7ae987005928d785e0a7829a902e02) Replace Visibility/RelationKind enums with open strings
- [`454f0dc`](https://github.com/burka/veclayer/commit/454f0dce6f1113b01b5624c0ad305e67e2d473df) Wire visibility filtering into CLI, MCP, and search
- [`d8250f4`](https://github.com/burka/veclayer/commit/d8250f413285de561afc2269b231fa3b620bd216) Implement RRD-style access tracking with 6 time-window buckets
- [`4aff453`](https://github.com/burka/veclayer/commit/4aff453eafe678bb1c8a528cf7e549c2e54ebd99) SOLID/DRY/SRP audit: fix violations, add tests, complete HTTP API
- [`1d7d9eb`](https://github.com/burka/veclayer/commit/1d7d9ebbf50ddc7fd3560b0fe5a43f05a1dd58bf) Fix all remaining test failures: decouple stats/sources from embedder
- [`698762d`](https://github.com/burka/veclayer/commit/698762db38f0660d0f708a95064e02c5007ee93d) Implement Phase 2+3: agent-driven memory management + aging
- [`daee521`](https://github.com/burka/veclayer/commit/daee5212544b1630c2f66a11def15cd68c6584f5) Redesign MCP to 5-tool agent interface: recall, focus, store, think, share
- [`422f040`](https://github.com/burka/veclayer/commit/422f0408105df579445eb234ad759aab9fd2fcee) Rewrite README and ROADMAP: align with VecLayer vision
- [`ca35d76`](https://github.com/burka/veclayer/commit/ca35d7628d32694ccc2b9bf9c0e369fc215bfe81) Remove old Roadmap.md and Tasks.md, replaced by ROADMAP.md
- [`e8247d6`](https://github.com/burka/veclayer/commit/e8247d6e193a1e59dcbbe03be92b52cb6294075c) Replace UUID with SHA-256 content-hash IDs, add EntryType enum
- [`fbcda28`](https://github.com/burka/veclayer/commit/fbcda2841532690d2ae1315bc5d58b8f58078381) Rename CLI to match spec: add, search, focus, init, status
- [`cabdbdf`](https://github.com/burka/veclayer/commit/cabdbdff5cc195cea2654fc6946967763e26c899) Split mcp/mod.rs (1076 lines) into 4 modules
- [`ea7215e`](https://github.com/burka/veclayer/commit/ea7215edef632fb90c5a04e94353ef1a9b383a26) Phase 1: SRP/DRY cleanup, TOML config, LLM feature flag
- [`c367724`](https://github.com/burka/veclayer/commit/c367724083475807a0b35903d41d4a037def79ac) Phase 2: Perspectives, faceted search, relation flags
- [`521aaa0`](https://github.com/burka/veclayer/commit/521aaa0faf2d102dc74c4562e3d0fe26c62ce315) Quality audit fixes: SQL DRY helpers, test coverage, perspective validation
- [`0ea4636`](https://github.com/burka/veclayer/commit/0ea463635a93f91b9eaa5e19838c9dfdbf7babbb) Phase 3: Salience scoring, compact command, salience-aware aging
- [`6251e05`](https://github.com/burka/veclayer/commit/6251e050a4b072dd7a1c62be6c8f0e12f3ce8fab) Phase 4: Identity module, reflect/id commands, dynamic MCP priming
- [`dee15fc`](https://github.com/burka/veclayer/commit/dee15fcea8365e42115cee6e716c669583ca3aee) DRY/quality fixes: preview() helper, open thread merge, MCP error logging
- [`51b30db`](https://github.com/burka/veclayer/commit/51b30dbc7e29cf1864c77c9c70322ebaa3d18d0c) Open_store helper, proper error propagation, test cleanup
- [`c5fa02f`](https://github.com/burka/veclayer/commit/c5fa02fe04ef6558856d7b21349eda62a144ab13) Phase 5: Think/Sleep cycle - LLM-powered memory consolidation
- [`7629858`](https://github.com/burka/veclayer/commit/76298587468475bbaeb70b4190e3a453e9b22afa) MCP stdio: fix stdout pollution, add project memory config
- [`1ed0867`](https://github.com/burka/veclayer/commit/1ed08674df6332ab51b9f6a4ad6f64191c4ba423) Phase 5.5: Tool ergonomics - relations, batch store, relevance tiers, browse mode, temporal filters, session pattern
- [`1d5c8c3`](https://github.com/burka/veclayer/commit/1d5c8c36ce5e1b142e5c011b9f0c4ccb1661ce9e) Phase 5.5: Tool ergonomics - review fixes, tests, clippy cleanup
- [`7e4042c`](https://github.com/burka/veclayer/commit/7e4042c9de076f1da7c21073bde7b6e872c901a7) Add OAuth-enabled sonote.ai MCP server + AGENTS.md workflow guidelines
- [`8fa1b06`](https://github.com/burka/veclayer/commit/8fa1b06a12ce6859d8bf7f98e36d38a9cfd06921) Add roadmap conventions: GitHub Issues tracking, agent attribution
- [`349f9fa`](https://github.com/burka/veclayer/commit/349f9fac13e34c6cb966a7362ddd80661278fde1) Register "session" as 7th default perspective (fixes #12) ([#12](https://github.com/burka/veclayer/issues/12))
- [`8b37347`](https://github.com/burka/veclayer/commit/8b37347fc334a1b74467117c105260b8a2bb99cd) Fix author name: Schmidt → Burka
- [`ae17e9b`](https://github.com/burka/veclayer/commit/ae17e9b0ee80c9db0ddfb0368b5bd5918e390cbd) Export/import CLI, file locking, and read-only fix (#59) ([#59](https://github.com/burka/veclayer/issues/59))


### Refactoring
- [`caa7ac2`](https://github.com/burka/veclayer/commit/caa7ac27550fc0a3fa611d6b50b1a83233df4b7a) Extract resolve_id and parse_temporal to shared helpers module
- [`be4e949`](https://github.com/burka/veclayer/commit/be4e9499a5ea5413a2ab825724e2cce3d80661fb) Introduce StoreBackend enum, decouple consumers from LanceStore
- [`237c18e`](https://github.com/burka/veclayer/commit/237c18ed761732845edbada7e9c3aa6c79fdc1f2) Split commands.rs into focused submodules


### Testing
- [`23b1dca`](https://github.com/burka/veclayer/commit/23b1dca2184975ea6c3bf4877ea90a48f815503b) Add 15 tests for think subcommands; drop needless embedder init
- [`19e2cbe`](https://github.com/burka/veclayer/commit/19e2cbe85f840b9fe6f555b0b3a9a46b2f335702) Add MCP relation structure tests
- [`67e2eca`](https://github.com/burka/veclayer/commit/67e2ecabbb92376641ec008c9ba437abb28c99ef) Add 5 missing tests for lock coverage (concurrent access, #70) ([#70](https://github.com/burka/veclayer/issues/70))


