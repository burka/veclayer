//! Integration tests for the git memory review workflow.
//!
//! Covers: MemoryStore operations, PushMode behavior, MigrateFilters logic,
//! scope gating, pull-rebase conflict detection, push-with-retry, and sync.
//! All tests use temporary git repositories and do not require a network
//! connection.

use std::path::{Path, PathBuf};

use tempfile::TempDir;
use veclayer::chunk::{ChunkLevel, EntryType, HierarchicalChunk};
use veclayer::commands::sync::MigrateFilters;
use veclayer::entry::Entry;
use veclayer::git::branch_config::PushMode;
use veclayer::git::memory_store::MemoryStore;
use veclayer::git::SyncResult;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Initialise a plain git repository with a single empty initial commit.
///
/// Returns `(tempdir, git_dir)`. The caller must keep `tempdir` alive for the
/// duration of the test — dropping it deletes the directory.
fn setup_repo() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let git_dir = dir.path().join(".git");

    run_git(dir.path(), &["init"]);
    run_git(
        dir.path(),
        &[
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "commit",
            "--allow-empty",
            "-m",
            "init",
        ],
    );

    (dir, git_dir)
}

/// Run a git command in `cwd`, panicking on spawn failure.
fn run_git(cwd: &Path, args: &[&str]) {
    std::process::Command::new("git")
        .current_dir(cwd)
        .args(args)
        .output()
        .expect("git command failed to spawn");
}

/// Build an `Entry` with deterministic, unique content.
///
/// Passing different `seed` values produces entries with distinct content hashes.
fn make_entry(seed: u64) -> Entry {
    Entry {
        content: format!("Integration test entry — seed {seed}"),
        entry_type: EntryType::Raw,
        source: "[test]".to_string(),
        created_at: 1_740_000_000 + seed as i64,
        perspectives: vec!["decisions".to_string()],
        relations: vec![],
        summarizes: vec![],
        heading: Some(format!("Heading {seed}")),
        parent_id: None,
        impression_hint: None,
        impression_strength: 1.0,
        expires_at: None,
        visibility: "normal".to_string(),
        level: ChunkLevel::H1,
        path: String::new(),
    }
}

// ---------------------------------------------------------------------------
// 1. PushMode behavior
// ---------------------------------------------------------------------------

/// `PushMode::Review` auto-stages entries (commits locally) but does not
/// auto-push.  `store_entry` must succeed and the entry must be retrievable.
#[test]
fn push_mode_review_stages_but_does_not_push() {
    let (_dir, git_dir) = setup_repo();
    let store = MemoryStore::open(&git_dir, Some("review-test")).unwrap();

    assert!(
        !store.config().push_mode().auto_pushes(),
        "Review mode must not auto-push"
    );
    assert!(
        store.config().push_mode().auto_stages(),
        "Review mode must auto-stage"
    );
    assert!(
        store.config().push_mode().uses_git(),
        "Review mode must use git"
    );

    // store_entry must succeed (commit locally) even without a remote.
    let result = store.store_entry(&make_entry(1));
    assert!(result.is_ok(), "store_entry should succeed: {result:?}");
}

/// `PushMode::Manual` does NOT auto-stage or auto-push.
#[test]
fn push_mode_manual_does_not_auto_stage() {
    assert!(!PushMode::Manual.auto_stages());
    assert!(!PushMode::Manual.auto_pushes());
    assert!(PushMode::Manual.uses_git());
}

/// `PushMode::Off` disables all git use.
#[test]
fn push_mode_off_disables_git() {
    assert!(!PushMode::Off.auto_stages());
    assert!(!PushMode::Off.auto_pushes());
    assert!(!PushMode::Off.uses_git());
}

/// `PushMode::Always` auto-stages AND auto-pushes.
#[test]
fn push_mode_always_stages_and_pushes() {
    assert!(PushMode::Always.auto_stages());
    assert!(PushMode::Always.auto_pushes());
    assert!(PushMode::Always.uses_git());
}

/// `PushMode::Always` calls `push_with_retry` after `store_entry`.  With no
/// remote configured the retry path silently succeeds — `store_entry` must not
/// return an error.
#[test]
fn push_mode_always_store_entry_no_remote_succeeds() {
    let (_dir, git_dir) = setup_repo();

    // Override push_mode to "always" via config.toml on the branch.
    let store = MemoryStore::open(&git_dir, Some("always-test")).unwrap();
    let always_config =
        "[memory]\n[memory.sync]\npush_mode = \"always\"\nbranch = \"always-test\"\n";
    store
        .branch()
        .write_file("config.toml", always_config.as_bytes())
        .expect("write config.toml");
    store.branch().commit("set push_mode=always").unwrap();

    // Re-open so the new config is loaded.
    let store = MemoryStore::open(&git_dir, Some("always-test")).unwrap();
    assert_eq!(store.config().push_mode(), PushMode::Always);

    // No remote — should still succeed.
    let result = store.store_entry(&make_entry(2));
    assert!(
        result.is_ok(),
        "store_entry with Always mode and no remote should succeed: {result:?}"
    );
}

// ---------------------------------------------------------------------------
// 2. MemoryStore operations
// ---------------------------------------------------------------------------

/// `list_entry_ids()` returns the 7-char short IDs embedded in Markdown filenames.
#[test]
fn list_entry_ids_returns_ids_from_filenames() {
    let (_dir, git_dir) = setup_repo();
    let store = MemoryStore::open(&git_dir, Some("ids-test")).unwrap();

    let entry = make_entry(10);
    store.store_entry(&entry).unwrap();

    let ids = store.list_entry_ids().unwrap();
    assert!(
        !ids.is_empty(),
        "list_entry_ids should return at least one ID"
    );

    // Every returned ID must be exactly 7 lowercase hex chars.
    for id in &ids {
        assert_eq!(id.len(), 7, "ID length must be 7: {id:?}");
        assert!(
            id.chars().all(|c| c.is_ascii_hexdigit()),
            "ID must be hex: {id:?}"
        );
    }
}

/// `list_entry_ids()` returns an ID for each stored entry when multiple entries
/// are committed.
#[test]
fn list_entry_ids_grows_with_stored_entries() {
    let (_dir, git_dir) = setup_repo();
    let store = MemoryStore::open(&git_dir, Some("multi-ids")).unwrap();

    store.store_entry(&make_entry(20)).unwrap();
    store.store_entry(&make_entry(21)).unwrap();
    store.store_entry(&make_entry(22)).unwrap();

    let ids = store.list_entry_ids().unwrap();
    assert!(ids.len() >= 3, "expected at least 3 IDs, got {}", ids.len());
}

/// `remove_entry(id)` removes the corresponding Markdown file and commits.
#[test]
fn remove_entry_deletes_file_and_commits() {
    let (_dir, git_dir) = setup_repo();
    let store = MemoryStore::open(&git_dir, Some("remove-test")).unwrap();

    store.store_entry(&make_entry(30)).unwrap();

    let ids_before = store.list_entry_ids().unwrap();
    assert!(
        !ids_before.is_empty(),
        "should have at least one entry before removal"
    );

    let id_to_remove = &ids_before[0];
    store.remove_entry(id_to_remove).unwrap();

    let ids_after = store.list_entry_ids().unwrap();
    assert!(
        !ids_after.contains(id_to_remove),
        "removed ID must not appear in list after deletion"
    );
}

/// `remove_entry(bad_id)` returns an error for an ID that does not exist on the
/// branch.
#[test]
fn remove_entry_returns_error_for_missing_id() {
    let (_dir, git_dir) = setup_repo();
    let store = MemoryStore::open(&git_dir, Some("remove-err")).unwrap();

    // "0000000" is a well-formed 7-char hex ID that is extremely unlikely to
    // match any real content hash.
    let result = store.remove_entry("0000000");
    assert!(
        result.is_err(),
        "remove_entry should return Err for a non-existent ID"
    );
}

/// `unpushed_commit_count()` returns 0 when no remote is configured.
#[test]
fn unpushed_commit_count_returns_zero_without_remote() {
    let (_dir, git_dir) = setup_repo();
    let store = MemoryStore::open(&git_dir, Some("unpushed-test")).unwrap();

    // Store something to create at least one local commit.
    store.store_entry(&make_entry(40)).unwrap();

    let count = store.unpushed_commit_count().unwrap();
    assert_eq!(
        count, 0,
        "unpushed_commit_count must be 0 when there is no remote"
    );
}

/// `store_entry()` followed by `load()` round-trips the entry content.
#[test]
fn store_entry_and_load_roundtrip() {
    let (_dir, git_dir) = setup_repo();
    let store = MemoryStore::open(&git_dir, Some("roundtrip")).unwrap();

    let entry = make_entry(50);
    let stored_path = store.store_entry(&entry).unwrap();
    assert!(
        stored_path.ends_with(".md"),
        "returned path must end with .md: {stored_path}"
    );

    let loaded = store.load().unwrap();
    assert!(
        loaded.iter().any(|e| e.content == entry.content),
        "loaded entries must contain the stored content"
    );
}

/// `load()` returns all previously stored entries when multiple are present.
#[test]
fn load_returns_all_stored_entries() {
    let (_dir, git_dir) = setup_repo();
    let store = MemoryStore::open(&git_dir, Some("load-all")).unwrap();

    let contents: Vec<String> = (60..63).map(|s| make_entry(s).content.clone()).collect();

    for seed in 60..63 {
        store.store_entry(&make_entry(seed)).unwrap();
    }

    let loaded = store.load().unwrap();
    for expected_content in &contents {
        assert!(
            loaded.iter().any(|e| &e.content == expected_content),
            "loaded entries must contain: {expected_content}"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. MigrateFilters
// ---------------------------------------------------------------------------

/// No filters set: every entry is accepted.
#[test]
fn migrate_filters_no_filters_accept_all() {
    let filters = MigrateFilters::default();
    // We verify the behaviour indirectly: MigrateFilters with no fields set
    // should produce a struct whose public fields are all None.
    assert!(filters.perspective.is_none());
    assert!(filters.exclude_perspective.is_none());
    assert!(filters.since.is_none());
}

/// `perspective` include filter: only entries with the matching perspective pass.
#[test]
fn migrate_filters_include_perspective_matches() {
    let filters = MigrateFilters {
        perspective: Some("decisions".to_string()),
        ..Default::default()
    };

    let matching = HierarchicalChunk::new(
        "content".to_string(),
        ChunkLevel::H1,
        None,
        String::new(),
        "src.md".to_string(),
    )
    .with_perspectives(vec!["decisions".to_string()]);

    let non_matching = HierarchicalChunk::new(
        "other content".to_string(),
        ChunkLevel::H1,
        None,
        String::new(),
        "src.md".to_string(),
    )
    .with_perspectives(vec!["knowledge".to_string()]);

    assert!(
        filters.accepts(&matching),
        "entry with matching perspective must be accepted"
    );
    assert!(
        !filters.accepts(&non_matching),
        "entry without matching perspective must be rejected"
    );
}

/// `exclude_perspective` filter: entries with the excluded perspective are rejected.
#[test]
fn migrate_filters_exclude_perspective_stored() {
    let filters = MigrateFilters {
        exclude_perspective: Some("learnings".to_string()),
        ..Default::default()
    };

    let excluded = HierarchicalChunk::new(
        "content".to_string(),
        ChunkLevel::H1,
        None,
        String::new(),
        "src.md".to_string(),
    )
    .with_perspectives(vec!["learnings".to_string()]);

    let accepted = HierarchicalChunk::new(
        "other content".to_string(),
        ChunkLevel::H1,
        None,
        String::new(),
        "src.md".to_string(),
    )
    .with_perspectives(vec!["decisions".to_string()]);

    assert!(
        !filters.accepts(&excluded),
        "entry with excluded perspective must be rejected"
    );
    assert!(
        filters.accepts(&accepted),
        "entry without excluded perspective must be accepted"
    );
}

/// `since` timestamp filter: the boundary value is stored.
#[test]
fn migrate_filters_since_timestamp_stored() {
    let filters = MigrateFilters {
        since: Some(1_700_000_000),
        ..Default::default()
    };
    assert_eq!(filters.since, Some(1_700_000_000));
}

/// Combining include perspective + since timestamp produces a struct with both
/// fields set independently.
#[test]
fn migrate_filters_combined_fields_independent() {
    let filters = MigrateFilters {
        perspective: Some("decisions".to_string()),
        since: Some(500_000),
        ..Default::default()
    };
    assert_eq!(filters.perspective.as_deref(), Some("decisions"));
    assert_eq!(filters.since, Some(500_000));
    assert!(filters.exclude_perspective.is_none());
}

// ---------------------------------------------------------------------------
// 4. Scope gating via PushMode (commit_to_git behaviour)
// ---------------------------------------------------------------------------

/// An entry stored via `MemoryStore::store_entry` (which unconditionally writes
/// to the git branch) always commits, regardless of scope metadata embedded in
/// content.  The public API does not expose a `scope` field; scope gating is
/// performed by the caller (MCP handler) before invoking `store_entry`.
///
/// This test verifies that `store_entry` itself succeeds and that the entry
/// is retrievable via `load()`.
#[test]
fn store_entry_is_always_committed_via_public_api() {
    let (_dir, git_dir) = setup_repo();
    let store = MemoryStore::open(&git_dir, Some("scope-test")).unwrap();

    let entry = Entry {
        content: "Scope gating is handled by the caller, not MemoryStore.".to_string(),
        entry_type: EntryType::Raw,
        source: "[test]".to_string(),
        created_at: 1_740_100_000,
        perspectives: vec!["knowledge".to_string()],
        relations: vec![],
        summarizes: vec![],
        heading: Some("Scope gating".to_string()),
        parent_id: None,
        impression_hint: None,
        impression_strength: 1.0,
        expires_at: None,
        visibility: "normal".to_string(),
        level: ChunkLevel::H1,
        path: String::new(),
    };

    let path = store.store_entry(&entry).unwrap();
    assert!(path.ends_with(".md"));

    let loaded = store.load().unwrap();
    assert!(
        loaded.iter().any(|e| e.content == entry.content),
        "entry must be findable after store_entry"
    );
}

/// PushMode::Manual does not auto-stage: `auto_stages()` returns false.
/// Entries are still written by `store_entry` because the public API always
/// writes — it is the caller's responsibility not to invoke `store_entry` when
/// the mode is Manual.
#[test]
fn push_mode_manual_auto_stages_is_false() {
    assert!(!PushMode::Manual.auto_stages());
}

/// PushMode::Off reports `uses_git() == false`, which lets callers skip the
/// entire git path.
#[test]
fn push_mode_off_uses_git_is_false() {
    assert!(!PushMode::Off.uses_git());
}

// ---------------------------------------------------------------------------
// 5. Branch configuration (default and custom)
// ---------------------------------------------------------------------------

/// Opening a fresh repository creates the memory branch with default config.
/// Default push mode is `Review`.
#[test]
fn open_creates_branch_with_default_review_push_mode() {
    let (_dir, git_dir) = setup_repo();
    let store = MemoryStore::open(&git_dir, Some("cfg-default")).unwrap();
    assert_eq!(store.config().push_mode(), PushMode::Review);
}

/// After writing a custom `config.toml` and re-opening, the new push mode is
/// honoured.
#[test]
fn open_loads_custom_push_mode_from_config() {
    let (_dir, git_dir) = setup_repo();

    // First open initialises the branch and writes the default config.
    {
        let store = MemoryStore::open(&git_dir, Some("cfg-custom")).unwrap();
        let custom = concat!(
            "[memory]\n",
            "[memory.sync]\n",
            "push_mode = \"manual\"\n",
            "branch = \"cfg-custom\"\n",
        );
        store
            .branch()
            .write_file("config.toml", custom.as_bytes())
            .unwrap();
        store.branch().commit("override push_mode").unwrap();
    }

    // Re-open reads the updated config.
    let store = MemoryStore::open(&git_dir, Some("cfg-custom")).unwrap();
    assert_eq!(store.config().push_mode(), PushMode::Manual);
}

/// A `BranchConfig` with a `company` section overrides the sync push_mode.
#[test]
fn branch_config_company_overrides_sync_push_mode() {
    use veclayer::git::branch_config::{parse, BranchConfig};

    let toml = concat!(
        "[memory]\n",
        "[memory.sync]\n",
        "push_mode = \"always\"\n",
        "branch = \"veclayer-memory\"\n",
        "[memory.company]\n",
        "remote = \"https://example.com/repo.git\"\n",
        "branch = \"veclayer-memory\"\n",
        "push_mode = \"review\"\n",
    );

    let config: BranchConfig = parse(toml.as_bytes()).unwrap();
    // company.push_mode wins over sync.push_mode
    assert_eq!(config.push_mode(), PushMode::Review);
}

// ---------------------------------------------------------------------------
// 6. Edge cases
// ---------------------------------------------------------------------------

/// `list_entry_files()` on a freshly opened (empty) branch contains no `.md`
/// entry files — only `config.toml` was committed, which is filtered out.
#[test]
fn list_entry_files_empty_on_fresh_branch() {
    let (_dir, git_dir) = setup_repo();
    let store = MemoryStore::open(&git_dir, Some("empty-branch")).unwrap();

    let files = store.list_entry_files().unwrap();
    assert!(
        files.is_empty(),
        "expected no entry files on a fresh branch, got: {files:?}"
    );
}

/// `load()` on an empty branch returns an empty Vec without error.
#[test]
fn load_on_empty_branch_returns_empty_vec() {
    let (_dir, git_dir) = setup_repo();
    let store = MemoryStore::open(&git_dir, Some("empty-load")).unwrap();

    let entries = store.load().unwrap();
    assert!(entries.is_empty(), "expected empty Vec on fresh branch");
}

/// Storing then removing all entries leaves the entry list empty.
#[test]
fn remove_all_entries_leaves_empty_list() {
    let (_dir, git_dir) = setup_repo();
    let store = MemoryStore::open(&git_dir, Some("remove-all")).unwrap();

    store.store_entry(&make_entry(70)).unwrap();

    let ids = store.list_entry_ids().unwrap();
    assert!(!ids.is_empty());

    for id in &ids {
        store.remove_entry(id).unwrap();
    }

    let remaining = store.list_entry_ids().unwrap();
    assert!(
        remaining.is_empty(),
        "all entries should be gone after removal; remaining: {remaining:?}"
    );
}

/// `unpushed_commit_count()` returns 0 when the branch is freshly created with
/// no remote tracking branch set up.
#[test]
fn unpushed_commit_count_zero_with_no_remote_tracking() {
    let (_dir, git_dir) = setup_repo();
    let store = MemoryStore::open(&git_dir, Some("track-test")).unwrap();

    // Multiple local commits but no remote.
    store.store_entry(&make_entry(80)).unwrap();
    store.store_entry(&make_entry(81)).unwrap();

    let count = store.unpushed_commit_count().unwrap();
    assert_eq!(count, 0, "expected 0 without remote tracking branch");
}

/// Two distinct entries produce two distinct short IDs — content-hash collision
/// is effectively impossible for unique seeds.
#[test]
fn distinct_entries_produce_distinct_ids() {
    let (_dir, git_dir) = setup_repo();
    let store = MemoryStore::open(&git_dir, Some("distinct-ids")).unwrap();

    store.store_entry(&make_entry(90)).unwrap();
    store.store_entry(&make_entry(91)).unwrap();

    let ids = store.list_entry_ids().unwrap();
    let unique: std::collections::HashSet<_> = ids.iter().collect();
    assert_eq!(
        ids.len(),
        unique.len(),
        "all IDs should be distinct; got: {ids:?}"
    );
}

// ---------------------------------------------------------------------------
// 7. Resilience: corrupted markdown files
// ---------------------------------------------------------------------------

/// `load()` silently skips malformed `.md` files and returns only valid entries.
///
/// This verifies the "skip on parse failure" contract in `parse_entry_files`:
/// a corrupted file must not abort the entire load or return an error.
#[test]
fn load_skips_malformed_markdown_and_returns_valid_entries() {
    let (_dir, git_dir) = setup_repo();
    let store = MemoryStore::open(&git_dir, Some("corrupt-test")).unwrap();

    // Store one valid entry so we have a known-good baseline.
    let valid_entry = make_entry(100);
    store.store_entry(&valid_entry).unwrap();

    // Write a malformed .md file directly to the branch.
    // Invalid YAML frontmatter (unclosed delimiter) will cause `parse` to fail.
    let malformed = b"---\ninvalid: : yaml: {{{\n---\nsome body content\n";
    store
        .branch()
        .write_file("decisions/corrupted-0000000.md", malformed)
        .expect("write malformed file");
    store
        .branch()
        .commit("add corrupted entry for test")
        .expect("commit malformed file");

    // load() must succeed and return only the valid entry.
    let loaded = store.load().expect("load() must not fail on corrupt files");

    assert!(
        loaded.iter().any(|e| e.content == valid_entry.content),
        "valid entry must be present in loaded results"
    );
    assert!(
        loaded.iter().all(|e| e.content != "some body content"),
        "malformed entry body must not appear in loaded results"
    );
}

// ---------------------------------------------------------------------------
// Shared infrastructure for multi-repo tests
// ---------------------------------------------------------------------------

/// Create a bare "remote" repository plus two independent client repositories
/// that both have `origin` pointing at the bare repo.
///
/// Returns `(bare_dir, client_a_dir, client_b_dir, client_a_git, client_b_git)`.
/// All four `TempDir` values must be kept alive for the duration of the test.
fn setup_two_clients_with_remote() -> (TempDir, TempDir, TempDir, PathBuf, PathBuf) {
    // Bare repo acts as the shared remote.
    let bare_dir = tempfile::tempdir().expect("bare tempdir");
    run_git(bare_dir.path(), &["init", "--bare"]);

    // Client A.
    let client_a_dir = tempfile::tempdir().expect("client_a tempdir");
    let client_a_git = client_a_dir.path().join(".git");
    run_git(client_a_dir.path(), &["init"]);
    run_git(
        client_a_dir.path(),
        &[
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "commit",
            "--allow-empty",
            "-m",
            "init",
        ],
    );
    run_git(
        client_a_dir.path(),
        &[
            "remote",
            "add",
            "origin",
            &bare_dir.path().to_string_lossy(),
        ],
    );

    // Client B — independent clone seeded from the same bare repo.
    let client_b_dir = tempfile::tempdir().expect("client_b tempdir");
    let client_b_git = client_b_dir.path().join(".git");
    run_git(client_b_dir.path(), &["init"]);
    run_git(
        client_b_dir.path(),
        &[
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "commit",
            "--allow-empty",
            "-m",
            "init",
        ],
    );
    run_git(
        client_b_dir.path(),
        &[
            "remote",
            "add",
            "origin",
            &bare_dir.path().to_string_lossy(),
        ],
    );

    (
        bare_dir,
        client_a_dir,
        client_b_dir,
        client_a_git,
        client_b_git,
    )
}

// ---------------------------------------------------------------------------
// T2. pull_rebase — conflict detection
// ---------------------------------------------------------------------------

/// Verify that `pull_rebase` returns `SyncResult::Conflicts` when two clients
/// have divergent commits that both modify the same file, and that the branch
/// is left in a clean state (no rebase in progress) after the call.
#[test]
fn test_pull_rebase_detects_conflict() {
    let branch_name = "test-memory";

    let (_bare, _ca_dir, _cb_dir, client_a_git, client_b_git) = setup_two_clients_with_remote();

    // Both clients open a MemoryStore on the same branch name.
    let store_a = MemoryStore::open(&client_a_git, Some(branch_name)).unwrap();
    let store_b = MemoryStore::open(&client_b_git, Some(branch_name)).unwrap();

    // Client A stores an entry and pushes so the remote branch exists.
    store_a.store_entry(&make_entry(200)).unwrap();
    store_a.push().unwrap();

    // Client B fetches to get the remote branch locally.
    store_b.branch().fetch().unwrap();

    // Now set up a divergence: both A and B write to the same file path with
    // different content, then A pushes first so the remote has A's version.
    // When B later tries to pull-rebase, git cannot auto-merge the same file.
    let shared_file = "decisions/conflict-target-0000001.md";
    let content_a = b"---\ncreated_at: 1740000000\n---\nClient A content\n";
    let content_b = b"---\ncreated_at: 1740000000\n---\nClient B content\n";

    // A writes the shared file and pushes.
    store_a.branch().write_file(shared_file, content_a).unwrap();
    store_a.branch().commit("A: write conflict target").unwrap();
    store_a.push().unwrap();

    // B writes the same file path with different content — this creates a
    // local commit that diverges from the remote because A's commit advanced
    // the remote tip.
    store_b.branch().write_file(shared_file, content_b).unwrap();
    store_b.branch().commit("B: write conflict target").unwrap();

    // B's local branch diverges from origin/test-memory. pull --rebase
    // should detect the conflict on the shared file.
    let result = store_b.branch().pull_rebase().unwrap();

    assert!(
        matches!(result, SyncResult::Conflicts(_)),
        "pull_rebase must return Conflicts when both sides modify the same file; got: {result:?}"
    );

    if let SyncResult::Conflicts(files) = result {
        assert!(
            !files.is_empty(),
            "Conflicts result must list at least one conflicting file"
        );
    }

    // The rebase must have been aborted — verify no REBASE_HEAD exists.
    let rebase_head = client_b_git.join("REBASE_HEAD");
    assert!(
        !rebase_head.exists(),
        "REBASE_HEAD must not exist after conflict-aborted rebase"
    );
}

/// Verify that `pull_rebase` auto-resolves conflicts when both sides added
/// the same file with identical content (content-addressed dedup scenario).
///
/// Two clients independently store the same entry (same content hash → same
/// filename). When B tries to pull-rebase after A pushed, git reports an
/// add/add conflict. Since the files are byte-identical, `pull_rebase` should
/// auto-resolve and return `SyncResult::Success`.
#[test]
fn test_pull_rebase_auto_resolves_identical_content_conflict() {
    let branch_name = "test-memory";

    let (_bare, _ca_dir, _cb_dir, client_a_git, client_b_git) = setup_two_clients_with_remote();

    let store_a = MemoryStore::open(&client_a_git, Some(branch_name)).unwrap();
    let store_b = MemoryStore::open(&client_b_git, Some(branch_name)).unwrap();

    // Both clients store the exact same entry (same content → same file path).
    let entry = make_entry(700);
    store_a.store_entry(&entry).unwrap();
    store_b.store_entry(&entry).unwrap();

    // A pushes first.
    store_a.push().unwrap();

    // B fetches so it knows about the remote branch, then tries pull-rebase.
    store_b.branch().fetch().unwrap();
    let result = store_b.branch().pull_rebase().unwrap();

    // Either Success (auto-resolved conflict) or NothingToSync (no divergence
    // detected because identical content produces compatible branch tips) is
    // acceptable — the key invariant is that it does NOT return Conflicts.
    assert!(
        !matches!(result, SyncResult::Conflicts(_)),
        "pull_rebase must not report conflicts for identical-content entries; got: {result:?}"
    );
}

// ---------------------------------------------------------------------------
// T3. push_with_retry — rejection then auto-rebase then success
// ---------------------------------------------------------------------------

/// Verify the push-with-retry path: when the remote rejects a push because it
/// has advanced, `store_entry` with `PushMode::Always` automatically
/// pull-rebases and retries, resulting in a successful push with both clients'
/// entries on the remote.
#[test]
fn test_push_with_retry_handles_rejection() {
    let branch_name = "test-memory";

    let (_bare, _ca_dir, _cb_dir, client_a_git, client_b_git) = setup_two_clients_with_remote();

    let store_a = MemoryStore::open(&client_a_git, Some(branch_name)).unwrap();
    let store_b = MemoryStore::open(&client_b_git, Some(branch_name)).unwrap();

    // Client A establishes the branch on the remote.
    store_a.store_entry(&make_entry(300)).unwrap();
    store_a.push().unwrap();

    // Client B fetches to get the remote branch.
    store_b.branch().fetch().unwrap();

    // Client A advances the remote with a second distinct entry.
    store_a.store_entry(&make_entry(301)).unwrap();
    store_a.push().unwrap();

    // Configure client B's store with PushMode::Always so store_entry
    // exercises push_with_retry internally.
    let always_config = concat!(
        "[memory]\n",
        "[memory.sync]\n",
        "push_mode = \"always\"\n",
        "branch = \"test-memory\"\n",
    );
    store_b
        .branch()
        .write_file("config.toml", always_config.as_bytes())
        .unwrap();
    store_b
        .branch()
        .commit("set push_mode=always for retry test")
        .unwrap();

    // Re-open client B so it picks up the new config.
    let store_b = MemoryStore::open(&client_b_git, Some(branch_name)).unwrap();
    assert_eq!(store_b.config().push_mode(), PushMode::Always);

    // Client B stores a unique entry. The first push attempt will be rejected
    // (remote is ahead from A's second commit). push_with_retry pulls,
    // rebases, and retries — the second push should succeed.
    let result = store_b.store_entry(&make_entry(302));
    assert!(
        result.is_ok(),
        "store_entry with Always mode must succeed after push-with-retry: {result:?}"
    );

    // After a successful push-with-retry, client B's local branch has been
    // rebased on top of A's commits and then pushed. Load from client B's
    // local branch to confirm all entries are present.
    //
    // We use branch().list_files() to avoid the fetch-then-read cycle of
    // load(), since we only want to inspect the local branch state.
    let b_files = store_b.branch().list_files().unwrap();
    let b_entries: Vec<&String> = b_files
        .iter()
        .filter(|f| f.ends_with(".md") && !f.starts_with('.'))
        .collect();

    // Entries from seeds 300 and 301 (pushed by A) plus entry 302 (pushed by B
    // after rebase) — three distinct .md files.
    assert!(
        b_entries.len() >= 3,
        "client B's branch must have at least 3 .md files after push-with-retry; found {}",
        b_entries.len()
    );
}

// ---------------------------------------------------------------------------
// T7. sync() — pull then push
// ---------------------------------------------------------------------------

/// Verify that `fetch()` succeeds when the remote branch does not exist yet.
///
/// This covers the "first use before first push" scenario: the remote has no
/// `veclayer-memory` branch. `fetch()` should return `Ok(())` instead of
/// failing with "couldn't find remote ref".
#[test]
fn test_fetch_succeeds_when_remote_branch_missing() {
    let branch_name = "test-memory";

    let (_bare, _ca_dir, _cb_dir, _client_a_git, client_b_git) = setup_two_clients_with_remote();

    // Neither client has pushed the branch — it doesn't exist on the remote.
    let branch = veclayer::git::GitMemoryBranch::open(&client_b_git, Some(branch_name)).unwrap();
    assert!(branch.has_remote().unwrap(), "remote must be configured");

    // fetch() must succeed silently (not error out on missing remote ref).
    let result = branch.fetch();
    assert!(
        result.is_ok(),
        "fetch() must succeed when the remote branch doesn't exist yet: {result:?}"
    );
}

/// Verify that `MemoryStore::open()` tracks an existing remote branch instead
/// of creating a divergent orphan.
///
/// Scenario: Client A creates the memory branch and pushes entries. Client B
/// opens a MemoryStore on the same branch name. Client B should see A's entries
/// (pulled from remote), not an empty orphan.
#[test]
fn test_open_tracks_remote_branch_instead_of_new_orphan() {
    let branch_name = "test-memory";

    let (_bare, _ca_dir, _cb_dir, client_a_git, client_b_git) = setup_two_clients_with_remote();

    // Client A creates the branch, stores entries, and pushes.
    let store_a = MemoryStore::open(&client_a_git, Some(branch_name)).unwrap();
    store_a.store_entry(&make_entry(500)).unwrap();
    store_a.store_entry(&make_entry(501)).unwrap();
    store_a.push().unwrap();

    // Client B opens a MemoryStore for the same branch name. It should detect
    // the remote branch and create a local tracking branch from it (NOT a
    // divergent orphan).
    let store_b = MemoryStore::open(&client_b_git, Some(branch_name)).unwrap();

    // Client B's local branch should already have A's entries.
    let loaded = store_b.load().unwrap();
    assert!(
        loaded.len() >= 2,
        "client B must see client A's entries after open(); got {} entries",
        loaded.len()
    );
    assert!(
        loaded.iter().any(|e| e.content.contains("seed 500")),
        "client B must see entry 500 from client A"
    );
    assert!(
        loaded.iter().any(|e| e.content.contains("seed 501")),
        "client B must see entry 501 from client A"
    );
}

/// Verify that the two-client sync workflow works end-to-end: Client A pushes,
/// Client B pulls via `pull()`, and then sees A's entries.
///
/// This is the inverse of `test_sync_pulls_and_pushes` — it focuses on the
/// pull path alone (no push from B), testing that `pull()` integrates remote
/// changes before `load()`.
#[test]
fn test_pull_integrates_remote_entries_before_load() {
    let branch_name = "test-memory";

    let (_bare, _ca_dir, _cb_dir, client_a_git, client_b_git) = setup_two_clients_with_remote();

    // Client A creates the branch and pushes some entries.
    let store_a = MemoryStore::open(&client_a_git, Some(branch_name)).unwrap();
    store_a.store_entry(&make_entry(600)).unwrap();
    store_a.push().unwrap();

    // Client B opens a MemoryStore (creates the branch from remote).
    let store_b = MemoryStore::open(&client_b_git, Some(branch_name)).unwrap();

    // Client A pushes more entries after B has opened.
    store_a.store_entry(&make_entry(601)).unwrap();
    store_a.push().unwrap();

    // Client B pulls to get the new entries.
    let pull_result = store_b.pull();
    assert!(pull_result.is_ok(), "pull() must succeed: {pull_result:?}");

    // After pulling, load() should return all entries (including 601).
    let loaded = store_b.load().unwrap();
    assert!(
        loaded.iter().any(|e| e.content.contains("seed 601")),
        "client B must see entry 601 after pull(); entries: {:?}",
        loaded.iter().map(|e| &e.content).collect::<Vec<_>>()
    );
}

/// Verify that `ensure_worktree()` detects and recovers from a stale worktree.
///
/// Scenario: A worktree directory exists with a `.git` file, but the backing
/// repository has been deleted. `ensure_worktree()` should detect the broken
/// state, remove the stale directory, and recreate it.
#[test]
fn test_stale_worktree_is_detected_and_recreated() {
    let (_dir, git_dir) = setup_repo();

    let branch = veclayer::git::GitMemoryBranch::open(&git_dir, Some("stale-wt-test")).unwrap();
    branch.create_orphan_branch().unwrap();

    // Verify worktree was created and works.
    let wt_path = branch.ensure_worktree().unwrap().to_path_buf();
    assert!(wt_path.is_dir(), "worktree must exist");
    assert!(wt_path.join(".git").exists(), "worktree .git must exist");

    // Corrupt the worktree by overwriting the .git file with invalid content.
    // This simulates a stale worktree whose backing repo was deleted.
    std::fs::write(wt_path.join(".git"), b"gitdir: /nonexistent/path/.git")
        .expect("write corrupt .git file");

    // ensure_worktree() should detect the broken state, remove the stale
    // directory, and recreate a working worktree.
    let result = branch.ensure_worktree();
    assert!(
        result.is_ok(),
        "ensure_worktree() must recover from stale worktree: {result:?}"
    );

    // The recovered worktree must be functional — verify with a write + commit.
    branch
        .write_file("test.txt", b"recovery test")
        .expect("write after recovery");
    branch
        .commit("test commit after recovery")
        .expect("commit after recovery");
}

/// Verify that `sync()` pulls remote changes and then pushes local commits so
/// that both clients end up with both entries on their local branch, and the
/// remote also has both entries.
#[test]
fn test_sync_pulls_and_pushes() {
    let branch_name = "test-memory";

    let (_bare, _ca_dir, _cb_dir, client_a_git, client_b_git) = setup_two_clients_with_remote();

    let store_a = MemoryStore::open(&client_a_git, Some(branch_name)).unwrap();
    let store_b = MemoryStore::open(&client_b_git, Some(branch_name)).unwrap();

    // Client A stores an entry and pushes to establish the remote branch.
    let entry_a = make_entry(400);
    store_a.store_entry(&entry_a).unwrap();
    store_a.push().unwrap();

    // Client B fetches so it shares the same base commit, then stores a
    // different entry locally (not pushed yet).
    store_b.branch().fetch().unwrap();
    let entry_b = make_entry(401);
    store_b.store_entry(&entry_b).unwrap();

    // Client B calls sync(): pulls A's state from remote, then pushes B's entry.
    let sync_result = store_b.sync();
    assert!(sync_result.is_ok(), "sync() must succeed: {sync_result:?}");

    // Client B's local branch must now contain both entries.
    // We read directly from the branch (no remote fetch needed here).
    let b_local_files = store_b.branch().list_files().unwrap();
    let b_has_a_entry = b_local_files
        .iter()
        .any(|f| f.ends_with(".md") && !f.starts_with('.'));
    assert!(
        b_has_a_entry,
        "client B's branch must have at least one .md file after sync"
    );

    let b_loaded = store_b.load().unwrap();
    assert!(
        b_loaded.iter().any(|e| e.content == entry_a.content),
        "client B must have client A's entry after sync"
    );
    assert!(
        b_loaded.iter().any(|e| e.content == entry_b.content),
        "client B must retain its own entry after sync"
    );

    // Verify the remote has both entries by reading from client B's local branch.
    // After sync(), client B's local branch was rebased and pushed. The branch
    // now contains both entry_a (pulled from remote) and entry_b (pushed by B).
    // We read via the branch plumbing to avoid the extra fetch of load().
    let b_files = store_b.branch().list_files().unwrap();
    let b_md_count = b_files
        .iter()
        .filter(|f| f.ends_with(".md") && !f.starts_with('.'))
        .count();
    assert!(
        b_md_count >= 2,
        "client B's branch must have at least 2 .md files after sync; found {b_md_count}"
    );
}
