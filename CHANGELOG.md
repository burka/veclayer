# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-06-04

> First public release. The CLI, MCP stdio server, local/Ollama embeddings, git-backed
> memory, and both storage backends (LanceDB + SQLite) are stable. The HTTP server and
> authentication stack are **experimental / work-in-progress** - see the dedicated section
> at the end and the Known Limitations in the README.

### Security
- [`5584b71`](https://github.com/burka/veclayer/commit/5584b7147240dcaf80e89b17a52c97dec5947a1b) Fix SQL injection, harden error handling, improve UX
- [`8748ae0`](https://github.com/burka/veclayer/commit/8748ae02d437d57408a39d44b35cd7ff648bc9fb) *(security,store)* Zeroize LLM API key; fail loud on corrupt embedding blobs
- [`d64c2da`](https://github.com/burka/veclayer/commit/d64c2daf784076c481d8a45110d4cfb920888b03) *(security)* Harden secret handling, auth logging, and input bounds
- [`f200ddb`](https://github.com/burka/veclayer/commit/f200ddb8ee356e98725ef2fe33a495325055a95c) *(git)* Validate memory branch name against option injection
- [`b646df6`](https://github.com/burka/veclayer/commit/b646df65705586d0fbfe357ef595990970d9cbef) *(git)* Strip embedded credentials from remote URLs in normalize_remote ([#98](https://github.com/burka/veclayer/issues/98))
- [`f281438`](https://github.com/burka/veclayer/commit/f2814383b93039d364101e2d66daca0229b2bdbe) *(config)* Escape TOML string values in user-config serialization ([#114](https://github.com/burka/veclayer/issues/114))
- [`7574a30`](https://github.com/burka/veclayer/commit/7574a3052b270bfcf6103728b4ce2b05376a0a4e) *(http)* Harden outbound clients against SSRF redirects and OOM bodies
- [`c23f773`](https://github.com/burka/veclayer/commit/c23f773598ba4ae82991c3e77dfba4391c339fb8) Resolve RUSTSEC CVEs and drop the rsa crate tree
- [`91bff2d`](https://github.com/burka/veclayer/commit/91bff2d144a1d7871a38912dcbe158546f09c308) *(privacy)* Remove hardcoded home paths and untrack local settings

### Added
- [`ff7957c`](https://github.com/burka/veclayer/commit/ff7957cf9e48b4561cd6837aac4837151956d83c) *(git)* Implement workspace-safe git memory storage and review workflow
- [`9b24e53`](https://github.com/burka/veclayer/commit/9b24e53ab0d0a400ee4579853b8cb3ab93b5a905) *(git)* Sync indexes into LanceDB, think(sync/status), error handling
- [`1fe2e56`](https://github.com/burka/veclayer/commit/1fe2e56bfa5e163782bb7487258d17bf310ddb38) *(git)* Pre-recall pull, sync CLI polish, conflict guidance
- [`d7eebd9`](https://github.com/burka/veclayer/commit/d7eebd9ae5ba2ad19ceb5337f8d5ff9e34f34158) *(mcp)* Rmcp SDK migration, resources, priming, HTTP improvements
- [`a48704c`](https://github.com/burka/veclayer/commit/a48704cbf75b9ae45fa58c351f5ec67ecddf7e6c) *(facade)* Add think cycle API with DynLlmProvider type erasure
- [`e471445`](https://github.com/burka/veclayer/commit/e471445d7d301ec6de6c167e9c64dbf791ccc19f) Add VecLayer facade: high-level API with embedder injection
- [`a2ff645`](https://github.com/burka/veclayer/commit/a2ff645904737c6c28026b54c4fc2fdb0c0c94de) Add SQLite storage backend and simplify VectorStore trait
- [`df41b0a`](https://github.com/burka/veclayer/commit/df41b0a538ecc680df949e44469e6cadbe265ba3) *(search)* Add creation-time recency boost to blend_score
- [`0962e64`](https://github.com/burka/veclayer/commit/0962e6459f0f3624b4abf16385e19229f0256481) *(think)* Auto-demote summarized entries to deep_only after consolidation
- [`c722cb8`](https://github.com/burka/veclayer/commit/c722cb8b0d17732fb810cbc8d18283945bf20b69) *(think)* Add project parameter to execute/execute_dyn for scoped think
- [`57af98b`](https://github.com/burka/veclayer/commit/57af98b8c0d091c2faf7020757d21f017ad9d534) *(think)* Enhance system prompt - detect contradictions and progress
- [`799c123`](https://github.com/burka/veclayer/commit/799c123de41bbbe9170625d47247135c0fc62fa8) *(salience)* Boost score for entries that have been consolidated
- [`4cf12d3`](https://github.com/burka/veclayer/commit/4cf12d304cd5bdd7dd59efaab950c06a952caaad) Track access on search_raw + ThinkResult convenience methods
- [`bc4c30b`](https://github.com/burka/veclayer/commit/bc4c30b02168821f15b20ea19fb39944aee088dc) Replace single perspective filter with multi-perspective slice throughout the stack
- [`24f1d22`](https://github.com/burka/veclayer/commit/24f1d2271a37cb11d6fd722a0b4d03f8c3f2df82) Add `veclayer stale` command for memory freshness checks
- [`2f6eca6`](https://github.com/burka/veclayer/commit/2f6eca6b3ec31438cf2ace30a16fb678d5b8ccba) Add lazy FastEmbed init, Ollama setup command, and decouple CLI from embedding-local
- [`0acc279`](https://github.com/burka/veclayer/commit/0acc2797e545c3fd9c76db3f20da0411e129b51d) Add hook-based auto-capture and guided setup for Claude Code
- [`8e95750`](https://github.com/burka/veclayer/commit/8e95750f594c1b66d4d522f3bc872ac44e9e5fdc) Add --global flag to setup claude and apply to project + global
- [`8645329`](https://github.com/burka/veclayer/commit/86453296170dcda7ac6dd914828b9ee4037767d5) Add --brief flag to context command for compact session injection
- [`08b7b57`](https://github.com/burka/veclayer/commit/08b7b5744a5a43c55ab502cc01a6be170cd88e5f) Add graceful think fallback when LLM is unavailable
- [`fca41c5`](https://github.com/burka/veclayer/commit/fca41c5a3cbc90eb90da54d2c662044f6b07d2b5) Add comprehensive tests across all modules - 1189 tests, 85% coverage
- [`37e6d36`](https://github.com/burka/veclayer/commit/37e6d36b4b5790576d2a89b1a8a33570b028f6e3) Ollama auto-discovery, async embedding, CI matrix, changelog
- [`7cb0255`](https://github.com/burka/veclayer/commit/7cb025551640fb45a5e44260e65d363d73bc6a03) Make embedding-local opt-in, add keyword search fallback
- [`d66dec1`](https://github.com/burka/veclayer/commit/d66dec137c7675e99aa38ba9b7a9d4e642faf602) Improve UX for Ollama-first embedding setup
- [`6dbcdca`](https://github.com/burka/veclayer/commit/6dbcdca4e517136324567d4522d6e87c539b3e01) *(embedder)* Auto-detect OpenAI-compatible embed services for all providers
- [`186108b`](https://github.com/burka/veclayer/commit/186108b79272bf506ecfd3423ef18df099d32b26) *(mcp)* Add embedding provider health to veclayer://status resource
- [`f2c3718`](https://github.com/burka/veclayer/commit/f2c3718fb175a6b67379d54a87f2bfb2d4ba3011) *(mcp)* Surface relations, impressions, and result-cap hints
- [`2f04185`](https://github.com/burka/veclayer/commit/2f04185b332579f3fc3221bb5f99d18dbe0b8c4a) Add `veclayer setup openclaw` for OpenClaw MCP integration
- [`ab7c769`](https://github.com/burka/veclayer/commit/ab7c769f1658df1763e05d0a4a87bb36e27e3676) *(store)* Enforce sqlite read-only and embedding dimension
- [`905f4df`](https://github.com/burka/veclayer/commit/905f4df0a9ccc55dff828b3642d05b2cbd828bbd) *(lance)* Auto-compact LanceDB versions after writes
- [`100e404`](https://github.com/burka/veclayer/commit/100e4047bd5201f23dd40bfcc4f858b834216cbe) *(lance)* One-time aggressive prune on first open if >500 versions
- [`d205cc9`](https://github.com/burka/veclayer/commit/d205cc91cf1e1e1ffaf2bafdd98d7f1e7566141b) *(store)* Bounded, non-blocking auto-prune to stop version blowup
- [`ff1a8ad`](https://github.com/burka/veclayer/commit/ff1a8adfda01162ef65ba8d58938a39519f97d5c) *(mcp)* Daily background compactor in long-running MCP server
- [`c14acdb`](https://github.com/burka/veclayer/commit/c14acdb35bd8d3c1c63f826badeb809611779311) Add reflect prune CLI command for manual LanceDB version compaction
- [`887744d`](https://github.com/burka/veclayer/commit/887744d643bf2192469ddf994bf5d43f95b565c1) *(reflect)* Humanize prune byte counts (MB/GB, ~4 sig digits)
- [`21f058f`](https://github.com/burka/veclayer/commit/21f058f7faf154b1315fefee8ab9fa765402c888) *(ux)* Surface model download, server URL, and missing-embedder fast-fail

### Changed
- [`33eba31`](https://github.com/burka/veclayer/commit/33eba316c2cdddbbb97d96555d93e4cd9f0b47e6) Make library slim by default via feature flags
- [`8bcba92`](https://github.com/burka/veclayer/commit/8bcba925f7d4fa26c66222bf040e8eb559ccbe1b) Gate non-core deps behind config and parser features
- [`3e00ccc`](https://github.com/burka/veclayer/commit/3e00ccc8add2e00921ac37ca5815eb4f99f2ba93) Gate auth and http features behind feature flags ([#74](https://github.com/burka/veclayer/issues/74))
- [`0bfb1d3`](https://github.com/burka/veclayer/commit/0bfb1d36dc6cd2203fbecbc3700e2220e59f8422) Replace native-tls with rustls-tls to remove OpenSSL system dependency
- [`6f295b7`](https://github.com/burka/veclayer/commit/6f295b7394aa64230246403b5f1cffa6abe388c0) Use system cache dir for fastembed models instead of CWD
- [`26644c3`](https://github.com/burka/veclayer/commit/26644c3692cfbdd7c6b1c5306f35b843c8c73b6b) Simplify codebase: eliminate DRY/SRP/SOLID violations across 30 files
- [`0f4131e`](https://github.com/burka/veclayer/commit/0f4131e70f56beb837d045b4be0409d3f52c2931) Extract ToolContext to replace repeated parameters in MCP tool functions
- [`de5f95b`](https://github.com/burka/veclayer/commit/de5f95bffa89590c863b5c6b7beb41ec434bafa6) Replace PreCompact warning with PostCompact context re-injection
- [`c1b971c`](https://github.com/burka/veclayer/commit/c1b971c3e44a1022ea8fa2c5806c031d00f04620) Update rust-version, upgrade rand to 0.9, fix release profile
- [`75541a4`](https://github.com/burka/veclayer/commit/75541a4b437f94906f9499dda7771c386c150c01) *(features)* Split clustering from llm, fix bundles, local-first default
- [`005333d`](https://github.com/burka/veclayer/commit/005333de4e300a83be622921cdd98a1e1b188b29) *(mcp)* Extract ServerCore, decompose McpHandler god-object
- [`df37c94`](https://github.com/burka/veclayer/commit/df37c9474705b6499fa3a6081ceb711ffad58bd6) *(config)* Split 2826-line god-module into config/ submodule
- [`2c5de13`](https://github.com/burka/veclayer/commit/2c5de136c2b39cd045c9058f32cea292b617dd96) *(identity)* Couple embedding access to its guard; fix doc link
- [`565c5de`](https://github.com/burka/veclayer/commit/565c5de1fa2cb0cebfaca5128bec33af739ef2b5) *(sync)* Remove dead sync feature scaffolding
- [`6a9914d`](https://github.com/burka/veclayer/commit/6a9914d9d6147b8ea2f359af9eb90b07d0d369a4) *(git)* Drop the unimplemented pull-request push mode and dead error ([#103](https://github.com/burka/veclayer/issues/103))
- [`f659fff`](https://github.com/burka/veclayer/commit/f659fff952c4a21ade0f29efd25ade2ba163ae94) *(project)* Make resolve_project_data_dir source exclusion type-level ([#115](https://github.com/burka/veclayer/issues/115))
- [`330a3fe`](https://github.com/burka/veclayer/commit/330a3fe49a60820c858bf223d1838df983db1fd7) *(embedder)* Make Embedder::embed async to end the block_in_place panic
- [`60d8e1b`](https://github.com/burka/veclayer/commit/60d8e1b29a5f38058285b964311477fbecbbee06) *(llm)* Propagate client-build errors instead of panicking
- [`5b0d38b`](https://github.com/burka/veclayer/commit/5b0d38b7dae3435eedf11748b1f8c7ae83eeaa37) DRY consolidation, security hardening, and 3 robustness fixes
- [`62c6856`](https://github.com/burka/veclayer/commit/62c6856083ef05721e672e6443fb2a6e9e8eeb4a) DRY consolidation of duplicate constants and functions
- [`ccba0bf`](https://github.com/burka/veclayer/commit/ccba0bf4c5d4038e96a41de9c03429d2c68f21a7) *(store)* Drop the orphaned non-blocking FileLock::acquire
- [`154a5c9`](https://github.com/burka/veclayer/commit/154a5c93f473d8bcda72a91e0e32a11332cc10a7) Allowlist packaged files and add docs.rs metadata
- [`7f31e3e`](https://github.com/burka/veclayer/commit/7f31e3e13c127f4b077dfad79a25f956bd65fbb6) Update dependencies; adapt to rmcp 1.7 API/behavior changes
- [`4c7c72b`](https://github.com/burka/veclayer/commit/4c7c72b4e58ee3b8683c494e28e5e9099d535f3a) Update rusqlite 0.37→0.40 and toml 0.8→1.0
- [`319fbde`](https://github.com/burka/veclayer/commit/319fbdef4a49966025919b3a7db2c98b2a084cc8) Update 17 semver-compatible dependencies
- [`9e5413d`](https://github.com/burka/veclayer/commit/9e5413ddb0561631d3b8605d10f4133afd5ae2c0) Migrate serde_yml to maintained serde_norway (RUSTSEC-2025-0067/0068)
- [`fbb0abe`](https://github.com/burka/veclayer/commit/fbb0abef32b048f883a7dd560d0fbf78ee5c7bbe) Correct README CLI table, status, and feature-flag accuracy
- [`0488aff`](https://github.com/burka/veclayer/commit/0488affd73e9d46b8201243ca7ece135396aadd9) Add crates.io, docs.rs, CI, and license badges to README
- [`c69704e`](https://github.com/burka/veclayer/commit/c69704e504197450d89ef1337ae8afc81f29e602) *(maintenance)* Self-healing prune script with disk guard + docs
- [`678cc25`](https://github.com/burka/veclayer/commit/678cc25b54a21e9c7c98d921dbbbc61cae40c65f) *(entry)* Document blob_hash two-stage derivation and pin the contract ([#90](https://github.com/burka/veclayer/issues/90))
- [`0a516b5`](https://github.com/burka/veclayer/commit/0a516b57d67f3fee664f1829a230c02d80f2b7e3) Fix feature docs, secure Docker default, broaden CI

### Fixed
- [`dc64ee4`](https://github.com/burka/veclayer/commit/dc64ee45514785d089e670a4d3fb32a8ee7233e9) 12 bugs from git memory feature testing
- [`42d4ae3`](https://github.com/burka/veclayer/commit/42d4ae3045fcb1fac754fb06481de5710daa33b7) *(git)* Wave 2 - 8 more bugs from git memory feature testing
- [`2e370b7`](https://github.com/burka/veclayer/commit/2e370b74845d6ecb5e86eca934e3f074b0453698) *(git)* Review fixes - security, correctness, robustness, and test coverage
- [`f07472d`](https://github.com/burka/veclayer/commit/f07472de51fcc4ef02b23586bf23826400372c35) *(think)* Pass project parameter to think_consolidate
- [`61c1ef1`](https://github.com/burka/veclayer/commit/61c1ef175efac4a1a582cdf60d498039eecee4d0) *(search)* Stabilize blend_score tests after recency boost addition
- [`ba9d1e8`](https://github.com/burka/veclayer/commit/ba9d1e807c729527a6003da959cd9ccd8887ffd2) Fix LanceDB commit conflicts on concurrent store access
- [`588e054`](https://github.com/burka/veclayer/commit/588e054b9da164979d7b36376d81b658388ea5df) Fix code review findings: dup2 error handling and missing test coverage
- [`3571884`](https://github.com/burka/veclayer/commit/3571884e9f01132942b1da20823cdd5ae0befdb9) *(ci)* Pin Rust toolchain to 1.93.1 via rust-toolchain.toml
- [`1e2b6d8`](https://github.com/burka/veclayer/commit/1e2b6d805d392308b7a3129dfd38b8fc1c751fde) Detect LanceDB dimension on open and improve error messages
- [`be9d81d`](https://github.com/burka/veclayer/commit/be9d81de2ed352dc6c0d06c0a477a482b4f281a2) *(import)* Distinguish parse errors from import failures
- [`abda541`](https://github.com/burka/veclayer/commit/abda541b5cdf6d8e7f28070c18d91151030b309b) Bound scoring math to documented ranges
- [`af6b6d7`](https://github.com/burka/veclayer/commit/af6b6d7122a293f9606530ce3e23f54e1fbde5c1) *(mcp)* Clamp caller-supplied result limits
- [`2847cb6`](https://github.com/burka/veclayer/commit/2847cb6cd2af3fa21682535c847d780ea76ff1ab) *(config)* Actually fall back when LLM base_url is invalid
- [`1c04ecf`](https://github.com/burka/veclayer/commit/1c04ecf71e0aac2064b78bca2d80dc45ca18e415) *(mcp)* Reject unknown store scope instead of widening visibility
- [`200e987`](https://github.com/burka/veclayer/commit/200e987c4b4bd50c950922ae16d2bd51c154e7e8) *(git)* Preserve impression_strength without an impression hint
- [`85b3072`](https://github.com/burka/veclayer/commit/85b3072a296ff8f68d45ba4dba2842aa7db016cb) *(store)* Typed lock contention; reject out-of-range SQLite values
- [`a9c202f`](https://github.com/burka/veclayer/commit/a9c202fd044228245885b5eba3d7ddb052e4cbfd) *(store)* Align LanceDB search score and result ordering
- [`8549bef`](https://github.com/burka/veclayer/commit/8549bef62f555ab16cdceb40c137575d0d3c460e) *(git)* Close worktree TOCTOU with an advisory lock; serialize env tests
- [`c90a6f3`](https://github.com/burka/veclayer/commit/c90a6f311a567204166f411b8c1d725bd0b626ee) *(mcp)* Retry embed worker on error instead of idle-throttling
- [`4007483`](https://github.com/burka/veclayer/commit/400748360420dc9bd52d7b3d397339ac0fcbfae8) *(embedder)* Chunk large embed batches into sub-batches
- [`3d8fae6`](https://github.com/burka/veclayer/commit/3d8fae637f6ce8785641e099830869136d8e274c) *(sync)* Return error for unsupported remote git URLs
- [`86723b6`](https://github.com/burka/veclayer/commit/86723b6be5482f59feecd7e3fca3b9d869ce9f23) *(mcp)* Keep think tool description in sync with THINK_ACTIONS
- [`b58d71d`](https://github.com/burka/veclayer/commit/b58d71d9a13a5f2b8edb2b9f920edd75f644f55d) *(mcp)* Stop advertising the unimplemented share tool ([#105](https://github.com/burka/veclayer/issues/105))
- [`7ef84bd`](https://github.com/burka/veclayer/commit/7ef84bd39ce84e82ed1cc85e1c86c71a8ba2ddff) *(store)* Surface swallowed compaction errors that let the store grow unbounded ([#92](https://github.com/burka/veclayer/issues/92))
- [`933dadc`](https://github.com/burka/veclayer/commit/933dadcfa7db152c96e6ca3594ea7546cfd84a5f) *(cli)* Cap LANCE_IO_THREADS for reflect prune to enable recovery at scale ([#92](https://github.com/burka/veclayer/issues/92))
- [`369661c`](https://github.com/burka/veclayer/commit/369661c339bf7b5cdedda51362ec75d3de124ff6) *(store)* Surface open_table/list_versions errors in auto_compact
- [`c12cbbb`](https://github.com/burka/veclayer/commit/c12cbbbd125fc00161f21e45c13f62703a7a1fb4) *(lance)* Compact fragments + materialize deletions, not just version manifests
- [`f8ee78e`](https://github.com/burka/veclayer/commit/f8ee78eff80f7d828d138036b5aa0b2edf78ed7b) *(store)* Tolerate concurrent-compactor commit conflicts in prune
- [`8a7e19d`](https://github.com/burka/veclayer/commit/8a7e19d88cfd893cfe1dc56b9548202f00f2555c) *(embedder)* Derive embedding dimension from fastembed metadata
- [`f0b248c`](https://github.com/burka/veclayer/commit/f0b248c7f26ee44d455a8aba5017a96b0bba6953) *(search)* Clamp blend_score to [0,1] on the recency_alpha==0 path ([#112](https://github.com/burka/veclayer/issues/112))
- [`44e8add`](https://github.com/burka/veclayer/commit/44e8addb2127ffb2a95793c95cd90e91d446773c) *(salience)* Clamp impression_strength and bound aging batch size ([#113](https://github.com/burka/veclayer/issues/113))
- [`e1eb90c`](https://github.com/burka/veclayer/commit/e1eb90cf0d60297d065d7ea00a7a95643dca3261) *(mcp)* Reject unresolvable parent_id instead of silently re-rooting ([#106](https://github.com/burka/veclayer/issues/106))
- [`17fafca`](https://github.com/burka/veclayer/commit/17fafca25ac6983a618a77322bf247fc7ada9ec0) *(relations)* Create forward link for summarizes relation ([#109](https://github.com/burka/veclayer/issues/109))
- [`e5f661b`](https://github.com/burka/veclayer/commit/e5f661b4dbb579e22d9b815287a9b07784197a83) *(cluster)* Implement num_clusters() instead of returning 0 stub ([#110](https://github.com/burka/veclayer/issues/110))
- [`e4c2af4`](https://github.com/burka/veclayer/commit/e4c2af43c45ae7109f820aa85caeb70d0419337b) *(sync)* Return error exit code when no scopes are configured ([#104](https://github.com/burka/veclayer/issues/104))
- [`3d11f44`](https://github.com/burka/veclayer/commit/3d11f44a9a0036bca4ee16041c81579bcd1d3d5f) *(init)* Clearer --share progress and canonical command hint ([#107](https://github.com/burka/veclayer/issues/107))
- [`47b00dd`](https://github.com/burka/veclayer/commit/47b00ddf2c1c86aeb73f70cf599be3b20cf19def) *(mcp)* Make think(discover) unconditional and drop dead llm gate
- [`e5362fa`](https://github.com/burka/veclayer/commit/e5362fa0d279df10cf5936093a37ea16ede2a9a7) *(store)* Reject writes on read-only lancedb stores
- [`b0450ae`](https://github.com/burka/veclayer/commit/b0450ae1684bf627c0dc5bcd29c98128661dc34a) *(store)* Skip migration writes on read-only lancedb opens
- [`d8ec583`](https://github.com/burka/veclayer/commit/d8ec5832df73d2419355d6e5b99416e4104a75e5) *(embedder)* Robust OpenAI-compatible auto-detection
- [`e682e71`](https://github.com/burka/veclayer/commit/e682e715790d95d4f2a229360a0c44cdba3f2685) *(config)* Warn on invalid env-var values; fix tautological test
- [`ebc5ad7`](https://github.com/burka/veclayer/commit/ebc5ad776f6f9a0df69c7a29b273e4f85137005d) *(store)* Return empty on limit==0 in LanceDB search
- [`e7ea5ee`](https://github.com/burka/veclayer/commit/e7ea5ee3075c96bf492cb73494120c30731f89cf) *(sync)* List scopes when filter is empty but scopes exist
- [`9f0bb4b`](https://github.com/burka/veclayer/commit/9f0bb4bf7aaa632ea1b7437930f186d3497ac5fd) Surface think demotion failure, fail-loud sqlite stats level cast, guard embedding dim bound
- [`0d672be`](https://github.com/burka/veclayer/commit/0d672beb4e3e8095521da55a3cca7eac4e135f3d) *(chunk)* Saturate ChunkLevel child to prevent u8 overflow wrap
- [`33bed63`](https://github.com/burka/veclayer/commit/33bed63b28f94332d68eb174d6f61f6f2ba4c06e) *(config)* Tolerate non-UTF-8 cwd in resolve instead of panicking
- [`518b3f0`](https://github.com/burka/veclayer/commit/518b3f06a0cf17bca1eff30a867ed1c509d1a65d) *(cli)* Return a clean error on undeterminable cwd instead of panicking
- [`1001c93`](https://github.com/burka/veclayer/commit/1001c9384c137154846f750e477cd33fb8d4ecb7) *(setup)* Error instead of panic on non-object mcpServers/hooks config
- [`700afd9`](https://github.com/burka/veclayer/commit/700afd9d919e37fb87c49130a516f232e21b5059) *(entry)* Saturate oversized embedding dimensions instead of truncating
- [`7571799`](https://github.com/burka/veclayer/commit/7571799a9a72fbfc387aa1807f6dacded66897b5) *(memory)* Log aging and access-profile errors instead of discarding
- [`e41502f`](https://github.com/burka/veclayer/commit/e41502fdc8daa8ec1fda95f3c1efabddfca4aab9) *(store)* Error on missing chunk in update_visibility across backends
- [`c7131fe`](https://github.com/burka/veclayer/commit/c7131fe5c1e7dcaa116fb99f8fc7fc7aa716ad37) *(commands)* Skip empty re-embedding result in rebuild_index
- [`8d5fa94`](https://github.com/burka/veclayer/commit/8d5fa94c4016217bb13a9eec5cb52dc0332d7549) *(mcp)* Reject empty per-item content in batch store path
- [`31fcf36`](https://github.com/burka/veclayer/commit/31fcf369d5ed93df1562b455fb311cc78a3c3adf) *(relations)* Write forward supersedes/version_of link on source
- [`7fc6311`](https://github.com/burka/veclayer/commit/7fc63116e69ccbbfe0065abd250779b59bed4639) *(access)* Preserve accumulated count in roll_up year bucket
- [`fe169f4`](https://github.com/burka/veclayer/commit/fe169f4bd06e8039baec09beb5b7eb70ccbc648e) *(blob)* Treat a directory at the hash path as absent
- [`c482f60`](https://github.com/burka/veclayer/commit/c482f60284cab008f0cb4b31de4dde72993cdbbb) *(embedder)* Exclude chat models from embed-model auto-pick
- [`9061379`](https://github.com/burka/veclayer/commit/906137904a6db49325dc2160064af4dbfff0713f) *(store)* Align sqlite get_by_id_prefix empty/non-hex with LanceDB
- [`80c0680`](https://github.com/burka/veclayer/commit/80c06802ef85bd73c716885b1ccc7f30a2a16716) *(embedder)* Use char-array split in is_chat_model
- [`c61d30a`](https://github.com/burka/veclayer/commit/c61d30a65986a59ae61a910835bff174d9eaf60f) *(embedder)* Retry FastEmbed model init instead of caching failure
- [`2025dd3`](https://github.com/burka/veclayer/commit/2025dd3f12991070d214d09172dc2646832a1cf4) *(cli)* Print errors via Display, validate --output, name failed import path
- [`9b8079c`](https://github.com/burka/veclayer/commit/9b8079c57ccefb7b57cfe5c03688e0e10128920b) *(ux)* Polish init messaging, think-relate kind arg, and auth error leakage
- [`12ef1cd`](https://github.com/burka/veclayer/commit/12ef1cd63984d44fd49b75348a703ca8e07363b0) *(store)* Make batch_update_embeddings atomic (add-then-delete)

### Testing
- Expanded test coverage to ~1,940 assertions across all modules: storage backends,
  git sync/conflict resolution, MCP dispatch & workers, embedder/LLM providers, config
  discovery, salience/aging, and crypto - including hermetic fixtures and injected
  embedders to remove network/model-download flakiness.

### Experimental - HTTP & Auth (WIP)
> These ship in the published crate behind the `http`/`auth` features but are not yet
> considered stable. Interfaces and on-disk formats may still change.

- [`c992561`](https://github.com/burka/veclayer/commit/c992561bae8d85e6d90ffd6bed0a702ece1607a0) *(auth)* Ed25519 identity, OAuth 2.0, JWT tokens, HTTP auth
- [`9f6ce37`](https://github.com/burka/veclayer/commit/9f6ce37b27aef488cb780ca7e3a0fa6fb0210fb4) Harden OAuth consent CSRF, state length cap, and JWT iss/nbf claims
- [`31697fb`](https://github.com/burka/veclayer/commit/31697fb94f9dd8bcaf37ca064219b8eb35ee57ef) *(serve)* Wire --auth-required flag through to Config
- [`28b1a57`](https://github.com/burka/veclayer/commit/28b1a57580dd0b21da4923b13dd3e958a6a180c4) *(http)* Rate-limit unauthenticated oauth surface and warn on open bind
- [`fa8ce53`](https://github.com/burka/veclayer/commit/fa8ce5343b36770331dbb258fd00a7f2cffa7a08) *(auth)* Harden JWT, PKCE, and token-store file handling
- [`2b35de4`](https://github.com/burka/veclayer/commit/2b35de4f722364a831bc2bf0ce4606ef228b9bf4) *(auth)* Harden OAuth flows, JWT issuer enforcement, and open-mode capability (#95, #96, #97)
- [`5364349`](https://github.com/burka/veclayer/commit/5364349b18eaa638f487b36eb54ebac3dc3d7f2e) *(auth)* Harden oauth registration, device, and token endpoints
- [`7cb87f0`](https://github.com/burka/veclayer/commit/7cb87f071b53acf41099082c97d226c8f3cfc136) *(auth)* Harden OAuth device & consent flows against silent scope default and unbounded map growth
- [`a80a8eb`](https://github.com/burka/veclayer/commit/a80a8eb78849cf0b52b36650dd21535e344ffeaa) *(auth)* Harden OAuth token lifecycle and device flow
- [`895b106`](https://github.com/burka/veclayer/commit/895b10667e1923d103037e9a674924ad954cd551) *(auth)* Require redirect_uri on token exchange; block re-approval
- [`d91b595`](https://github.com/burka/veclayer/commit/d91b5953e0997e2a20f8c85dc85be2feb542575a) *(auth)* CSRF-protect the device authorization page
- [`5d7ec95`](https://github.com/burka/veclayer/commit/5d7ec9593c1611d7c223416fc504eb3e0e4cfcfd) *(auth)* Bind client_id at device-code token exchange
- [`bd2487b`](https://github.com/burka/veclayer/commit/bd2487bd711ae30612e471e01563d2e665e7018e) *(auth)* Enforce max state length in POST consent handler
- [`300b717`](https://github.com/burka/veclayer/commit/300b71777b4381e4537c0e2717266576cbf67d7b) *(auth)* Bound PKCE code_challenge length on the consent POST path
- [`35d5b52`](https://github.com/burka/veclayer/commit/35d5b5298623ab1fa483f1f4e1d4dceee3e19c5b) *(auth)* Propagate token-store persistence failures
- [`de96384`](https://github.com/burka/veclayer/commit/de96384c6819a3a15358e3782ee52d720fe9b064) *(auth)* Return generic error_description for invalid grant and mint failure
- [`84a3744`](https://github.com/burka/veclayer/commit/84a374486e3bf55c0bdb06c8ef203e044abe0ca3) *(auth)* Stop reflecting caller grant_type into unsupported_grant_type body
- [`2018a15`](https://github.com/burka/veclayer/commit/2018a15addad8e23fe372f206203a76266f9a1c8) *(security)* Fail loud on unsigned share and empty auth passphrase
- [`66902dc`](https://github.com/burka/veclayer/commit/66902dcb9e5692ce902aa70af1500c8899848a82) *(crypto)* Reject DID with trailing bytes in from_did
- [`f530b87`](https://github.com/burka/veclayer/commit/f530b8715cb3a56210b07f8f929938f7cb2cafe9) Harden keystore length validation, embedder count check, blob temp cleanup
- [`ad3dc5c`](https://github.com/burka/veclayer/commit/ad3dc5c28c82eccf7aaac49ea8774c179757420c) *(mcp)* Reject CORS look-alike loopback origins
- [`27e3f3d`](https://github.com/burka/veclayer/commit/27e3f3da37af908d3199475a0fa61c2c59f8425f) *(auth)* Explain deliberate slow_down throttle exemption
- [`6aa6a19`](https://github.com/burka/veclayer/commit/6aa6a19673e294628252306e356f99974ce5dc17) Bump 0.2.0, fix MSRV/docs.rs/CI, mark auth+http WIP
- [`d8068fe`](https://github.com/burka/veclayer/commit/d8068fe802f3c15884475682ab77af557b155884) *(cli)* Gate WIP http/auth surface, validate enums, race-free --quiet
- [`ae62ceb`](https://github.com/burka/veclayer/commit/ae62cebf8a03caa71a5c4fde44c2a92f74522fe3) *(auth)* Zeroize passphrases, hash OAuth codes, RFC7519 aud, argon2 p=4
- [`04b0029`](https://github.com/burka/veclayer/commit/04b0029d3636c74b460731e5639e86ff46a7234c) *(http)* Fail-closed open-bind, tighten CORS, fix PKCE, drop block_in_place

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


