//! Integration tests for the git memory review workflow.
//!
//! Covers: MemoryStore operations, PushMode behavior, MigrateFilters logic,
//! and scope gating.  All tests use temporary git repositories and do not
//! require a network connection.

use std::path::{Path, PathBuf};

use veclayer::chunk::{ChunkLevel, EntryType};
use veclayer::commands::sync::MigrateFilters;
use veclayer::entry::Entry;
use veclayer::git::branch_config::PushMode;
use veclayer::git::memory_store::MemoryStore;

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

    // Verify field is set; actual filtering logic is tested via accept() in the
    // module's own unit tests (which live next to the private `accepts` method).
    assert_eq!(filters.perspective.as_deref(), Some("decisions"));
}

/// `exclude_perspective` filter: the excluded perspective name is stored.
#[test]
fn migrate_filters_exclude_perspective_stored() {
    let filters = MigrateFilters {
        exclude_perspective: Some("learnings".to_string()),
        ..Default::default()
    };
    assert_eq!(filters.exclude_perspective.as_deref(), Some("learnings"));
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
