//! Orchestration layer for the git-based memory store.
//!
//! [`MemoryStore`] ties together git operations ([`GitMemoryBranch`]),
//! Markdown entry serialization ([`markdown`]), the embedding cache, and
//! branch configuration ([`branch_config`]) into a single high-level API.
//!
//! All sync operations are optional — when no remote is configured they
//! silently succeed. Push behaviour is governed by [`PushMode`] from the
//! branch config.

use std::path::Path;

use tracing::{debug, warn};

use crate::chunk::short_id;
use crate::entry::Entry;
use crate::git::branch_config::{self, BranchConfig};
use crate::git::markdown;
use crate::git::{
    is_file_not_found, run_git_with_gitdir, GitError, GitMemoryBranch, PushResult, SyncResult,
    REMOTE,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CONFIG_FILENAME: &str = "config.toml";

// ---------------------------------------------------------------------------
// MemoryStore
// ---------------------------------------------------------------------------

/// Orchestrates git-based memory storage.
///
/// Combines git operations, Markdown entry serialization, embedding cache,
/// and branch configuration into a high-level API.
pub struct MemoryStore {
    branch: GitMemoryBranch,
    config: BranchConfig,
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

impl MemoryStore {
    /// Open the git memory branch and load its configuration.
    ///
    /// If the branch does not exist locally but exists on the remote, fetches it
    /// and creates a local tracking branch. If it doesn't exist anywhere, an
    /// orphan branch is created with a default `config.toml`.
    pub fn open(git_dir: &Path, branch_name: Option<&str>) -> Result<Self, GitError> {
        let branch = GitMemoryBranch::open(git_dir, branch_name)?;

        if !branch.branch_exists()? {
            // Try to pick up the branch from the remote before creating a new orphan.
            if branch.has_remote().unwrap_or(false) {
                let _ = branch.fetch(); // best-effort; may fail if branch not on remote yet
                if try_create_from_remote(&branch)? {
                    let config = load_config(&branch)?;
                    return Ok(Self { branch, config });
                }
            }

            // No remote branch found — create a fresh orphan.
            branch.create_orphan_branch()?;
            write_default_config(&branch)?;
        } else {
            branch.create_orphan_branch()?; // idempotent — just ensures worktree
        }

        let config = load_config(&branch)?;
        Ok(Self { branch, config })
    }

    /// Returns a reference to the branch configuration.
    pub fn config(&self) -> &BranchConfig {
        &self.config
    }

    /// Returns a reference to the underlying git memory branch.
    pub fn branch(&self) -> &GitMemoryBranch {
        &self.branch
    }
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

impl MemoryStore {
    /// Load all entries from the memory branch.
    ///
    /// Fetches from remote first (if a remote is configured), then reads and
    /// parses every `.md` file. Files that fail to parse are logged via
    /// [`tracing::warn`] and skipped — they do not abort the load.
    pub fn load(&self) -> Result<Vec<Entry>, GitError> {
        self.fetch_if_remote()?;
        let files = self.list_entry_files()?;
        Ok(self.parse_entry_files(&files))
    }

    /// Load all entries together with their cached embeddings for `model`.
    ///
    /// Equivalent to [`load`] but also reads the embedding cache for each
    /// entry. An entry without a cached embedding gets `None` as its pair.
    #[allow(clippy::type_complexity)]
    pub fn load_with_embeddings(
        &self,
        model: &str,
    ) -> Result<Vec<(Entry, Option<Vec<f32>>)>, GitError> {
        self.fetch_if_remote()?;
        let files = self.list_entry_files()?;
        let entries = self.parse_entry_files(&files);

        let pairs = entries
            .into_iter()
            .map(|entry| {
                let id = entry.content_id();
                let short = short_id(&id).to_string();
                let embedding = self
                    .branch
                    .read_embedding(&short, model)
                    .unwrap_or_else(|e| {
                        warn!("failed to read embedding for {short}: {e}");
                        None
                    });
                (entry, embedding)
            })
            .collect();

        Ok(pairs)
    }
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

impl MemoryStore {
    /// Render `entry` to Markdown and write it to the memory branch.
    ///
    /// Commits with a short descriptive message. If push mode is [`Always`],
    /// pushes immediately — handling a rejection by pulling then retrying.
    ///
    /// Returns the relative file path that was written.
    ///
    /// [`Always`]: PushMode::Always
    pub fn store_entry(&self, entry: &Entry) -> Result<String, GitError> {
        let markdown_content = markdown::render(entry)?;
        let path = markdown::entry_path(entry);

        self.remove_stale_duplicate(&path, entry)?;
        self.branch.write_file(&path, markdown_content.as_bytes())?;

        let commit_message = commit_message_for_entry(entry);
        self.branch.commit(&commit_message)?;

        if self.config.push_mode().auto_pushes() {
            self.push_with_retry()?;
        }

        Ok(path)
    }

    /// Write an embedding vector to the cache and commit.
    ///
    /// The file is written to `.embeddings/<short-id>.<model>.bin`. The caller
    /// is responsible for pushing if needed.
    pub fn store_embedding(
        &self,
        entry: &Entry,
        model: &str,
        vector: &[f32],
    ) -> Result<(), GitError> {
        let id = entry.content_id();
        let short = short_id(&id);
        self.branch.write_embedding(short, model, vector)?;
        let message = format!("cache: embedding for {short}");
        self.branch.commit(&message)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Sync
// ---------------------------------------------------------------------------

impl MemoryStore {
    /// Pull-rebase from remote without pushing.
    ///
    /// Silently succeeds when no remote is configured.
    pub fn pull(&self) -> Result<SyncResult, GitError> {
        if !self.branch.has_remote()? {
            return Ok(SyncResult::NothingToSync);
        }
        self.branch.pull_rebase()
    }

    /// Pull-rebase from remote then push local commits.
    ///
    /// Both operations silently succeed when no remote is configured.
    pub fn sync(&self) -> Result<SyncResult, GitError> {
        if !self.branch.has_remote()? {
            return Ok(SyncResult::NothingToSync);
        }

        let pull_result = self.branch.pull_rebase()?;
        match self.branch.push()? {
            PushResult::Rejected => return Err(GitError::RemoteDiverged),
            PushResult::Success | PushResult::NothingToPush => {}
        }
        Ok(pull_result)
    }

    /// Remove orphaned embedding cache files and force-push the result.
    ///
    /// Deletes `.bin` files whose ID is not in `live_ids` for `model`. If any
    /// files were deleted, commits and force-pushes with lease. Returns the
    /// number of files removed.
    pub fn gc(&self, live_ids: &[&str], model: &str) -> Result<usize, GitError> {
        let deleted = self.branch.gc_embeddings(live_ids, model)?;

        if deleted > 0 {
            let message = format!("gc: remove {deleted} orphaned {model} embeddings");
            self.branch.commit(&message)?;
            if self.branch.has_remote()? {
                self.branch.push_force_with_lease()?;
            }
        }

        Ok(deleted)
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

impl MemoryStore {
    /// List all `.md` entry files on the branch.
    ///
    /// Excludes `config.toml`, dotfiles (`.gitignore`, `.embeddings/`, etc.),
    /// and anything that is not a `.md` file.
    pub fn list_entry_files(&self) -> Result<Vec<String>, GitError> {
        let all = self.branch.list_files()?;
        let filtered = all.into_iter().filter(|path| is_entry_file(path)).collect();
        Ok(filtered)
    }

    /// List all entry IDs currently on the memory branch.
    ///
    /// IDs are extracted from the filename suffix: `slug-<7-char-id>.md`.
    pub fn list_entry_ids(&self) -> Result<Vec<String>, GitError> {
        let files = self.list_entry_files()?;
        let ids = files
            .iter()
            .filter_map(|path| extract_id_from_path(path))
            .collect();
        Ok(ids)
    }

    /// Remove an entry from the memory branch by ID prefix.
    ///
    /// Searches all `.md` files for one whose filename contains the given `id`
    /// prefix, deletes the file via `git rm`, and commits the removal.
    ///
    /// Returns an error if no file matches the prefix or if more than one file
    /// matches (ambiguous prefix).
    pub fn remove_entry(&self, id: &str) -> Result<(), GitError> {
        let files = self.list_entry_files()?;

        let matches: Vec<&String> = files
            .iter()
            .filter(|path| path_matches_id(path, id))
            .collect();

        match matches.len() {
            0 => Err(GitError::CommandFailed {
                command: format!("remove-entry {id}"),
                stderr: format!("no entry found matching id '{id}'"),
                exit_code: 1,
            }),
            2.. => Err(GitError::CommandFailed {
                command: format!("remove-entry {id}"),
                stderr: format!(
                    "ambiguous id '{id}' matches {} entries; use a longer prefix",
                    matches.len()
                ),
                exit_code: 1,
            }),
            1 => {
                let path = matches[0].clone();
                self.branch.delete_file(&path)?;
                let message = format!("remove: {path}");
                self.branch.commit(&message)?;
                Ok(())
            }
        }
    }

    /// Delete any existing entry file that shares the same content ID but has a
    /// different path than `new_path`.
    ///
    /// When the heading changes but the content (and thus the content hash) is
    /// identical, `entry_path` produces a different filename for the same logical
    /// entry. This method finds such stale files and removes them with `git rm`
    /// so that `write_file` can then create the canonical new path without
    /// leaving a duplicate behind.
    fn remove_stale_duplicate(&self, new_path: &str, entry: &Entry) -> Result<(), GitError> {
        let id = entry.content_id();
        let short = short_id(&id);

        let existing_files = match self.list_entry_files() {
            Ok(files) => files,
            Err(_) => return Ok(()), // branch may be empty; nothing to clean up
        };

        for stale_path in existing_files
            .into_iter()
            .filter(|p| p != new_path && path_matches_id(p, short))
        {
            debug!("removing stale duplicate {stale_path} (same id as {new_path})");
            self.branch.delete_file(&stale_path)?;
        }

        Ok(())
    }

    /// Return the number of local commits not yet pushed to the remote.
    ///
    /// Uses `git log origin/<branch>..<branch> --oneline` to count ahead commits.
    /// Returns `0` if no remote tracking branch exists (nothing to push).
    pub fn unpushed_commit_count(&self) -> Result<usize, GitError> {
        if !self.branch.has_remote()? {
            return Ok(0);
        }

        let remote_ref = format!("{REMOTE}/{}", self.branch.branch);
        let local_ref = self.branch.branch.clone();
        let range = format!("{remote_ref}..{local_ref}");

        let output = run_git_with_gitdir(&self.branch.git_dir, &["log", "--oneline", &range])?;

        // If the remote tracking ref doesn't exist yet (first push), treat all
        // local commits as unpushed. git log exits non-zero in that case.
        if !output.status.success() {
            let log_all =
                run_git_with_gitdir(&self.branch.git_dir, &["log", "--oneline", &local_ref])?;
            if log_all.status.success() {
                let count = String::from_utf8_lossy(&log_all.stdout)
                    .lines()
                    .filter(|l| !l.is_empty())
                    .count();
                return Ok(count);
            }
            return Ok(0);
        }

        let count = String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter(|l| !l.is_empty())
            .count();

        Ok(count)
    }

    /// Push local commits to the remote.
    ///
    /// Returns [`PushResult`] describing the outcome. Silently succeeds when
    /// no remote is configured (returns `NothingToPush`).
    pub fn push(&self) -> Result<PushResult, GitError> {
        if !self.branch.has_remote()? {
            return Ok(PushResult::NothingToPush);
        }
        self.branch.push()
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Write and commit a default `config.toml` to the branch.
fn write_default_config(branch: &GitMemoryBranch) -> Result<(), GitError> {
    let config = branch_config::default_config();
    let toml = branch_config::render(&config)?;
    branch.write_file(CONFIG_FILENAME, toml.as_bytes())?;
    branch.commit("init: default config.toml")?;
    Ok(())
}

/// Load `config.toml` from the branch, falling back to defaults if absent.
fn load_config(branch: &GitMemoryBranch) -> Result<BranchConfig, GitError> {
    match branch.read_file(CONFIG_FILENAME) {
        Ok(bytes) => branch_config::parse(&bytes),
        Err(GitError::CommandFailed { ref stderr, .. }) if is_file_not_found(stderr) => {
            Ok(branch_config::default_config())
        }
        Err(e) => Err(e),
    }
}

/// Extract the 7-char short ID from a file path like `dir/slug-abcdef1.md`.
///
/// The ID is the last hyphen-separated token before `.md`.
fn extract_id_from_path(path: &str) -> Option<String> {
    let filename = path.rsplit('/').next()?;
    let stem = filename.strip_suffix(".md")?;
    let id = stem.rsplit('-').next()?;
    // Short IDs are exactly 7 lowercase hex chars
    if id.len() == 7 && id.chars().all(|c| c.is_ascii_hexdigit()) {
        Some(id.to_string())
    } else {
        None
    }
}

/// Return `true` if the filename at `path` contains the given `id` as a suffix
/// or prefix match of the short ID embedded in the filename.
///
/// An empty `id` never matches: `starts_with("")` is universally true, which
/// would otherwise make every entry "match" and surface a misleading
/// ambiguous-id error to the caller instead of a clean not-found.
fn path_matches_id(path: &str, id: &str) -> bool {
    if id.is_empty() {
        return false;
    }
    let filename = match path.rsplit('/').next() {
        Some(f) => f,
        None => return false,
    };
    let stem = match filename.strip_suffix(".md") {
        Some(s) => s,
        None => return false,
    };
    let short = match stem.rsplit('-').next() {
        Some(s) => s,
        None => return false,
    };
    short == id || short.starts_with(id)
}

/// Return `true` if `path` is a regular `.md` entry file (not a hidden path
/// and not `config.toml`).
fn is_entry_file(path: &str) -> bool {
    if !path.ends_with(".md") {
        return false;
    }
    // Exclude dotfiles and directories under hidden paths (e.g. `.embeddings/`)
    if path.starts_with('.') || path.contains("/.") {
        return false;
    }
    // config.toml is a TOML file, filtered out above by the .md check, but
    // keep this guard in case naming conventions change.
    if path == CONFIG_FILENAME {
        return false;
    }
    true
}

/// Parse a list of entry file paths into [`Entry`] values, skipping failures.
///
/// Parse errors are logged via [`tracing::warn`] but do not abort iteration.
impl MemoryStore {
    fn parse_entry_files(&self, files: &[String]) -> Vec<Entry> {
        let mut entries = Vec::with_capacity(files.len());
        for path in files {
            match self.branch.read_file(path) {
                Ok(bytes) => match markdown::parse(&bytes) {
                    Ok((entry, _body)) => entries.push(entry),
                    Err(e) => warn!("skipping {path}: {e}"),
                },
                Err(e) => warn!("failed to read {path}: {e}"),
            }
        }
        entries
    }
}

/// Build a commit message for a stored entry.
fn commit_message_for_entry(entry: &Entry) -> String {
    if let Some(heading) = &entry.heading {
        return format!("store: {heading}");
    }
    let char_count = entry.content.chars().count();
    let preview: String = entry.content.chars().take(60).collect();
    if char_count > 60 {
        format!("store: {preview}...")
    } else {
        format!("store: {preview}")
    }
}

/// Try to create a local branch from the remote tracking ref.
///
/// Returns `true` if the local branch was successfully created from
/// `origin/<branch>`, `false` if the remote ref doesn't exist.
fn try_create_from_remote(branch: &GitMemoryBranch) -> Result<bool, GitError> {
    let remote_ref = format!("{REMOTE}/{}", branch.branch);
    let output = run_git_with_gitdir(&branch.git_dir, &["rev-parse", "--verify", &remote_ref])?;

    if !output.status.success() {
        return Ok(false); // Remote ref doesn't exist
    }

    let commit = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if commit.is_empty() {
        return Ok(false);
    }

    // Create local branch pointing at the remote commit.
    let output = run_git_with_gitdir(&branch.git_dir, &["branch", &branch.branch, &commit])?;

    if !output.status.success() {
        return Ok(false);
    }

    // Set up the worktree so subsequent operations work.
    branch.ensure_worktree()?;
    Ok(true)
}

/// Fetch from the remote when one is configured; silently succeed otherwise.
impl MemoryStore {
    fn fetch_if_remote(&self) -> Result<(), GitError> {
        if self.branch.has_remote()? {
            self.branch.fetch()?;
        }
        Ok(())
    }

    /// Push, retrying after a pull-rebase if the remote rejects the push.
    fn push_with_retry(&self) -> Result<(), GitError> {
        if !self.branch.has_remote()? {
            return Ok(());
        }

        match self.branch.push()? {
            PushResult::Success | PushResult::NothingToPush => Ok(()),
            PushResult::Rejected => {
                self.branch.pull_rebase()?;
                match self.branch.push()? {
                    PushResult::Success | PushResult::NothingToPush => Ok(()),
                    PushResult::Rejected => Err(GitError::RemoteDiverged),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{ChunkLevel, EntryType};
    use crate::git::branch_config::PushMode;
    use crate::git::test_helpers::setup_test_repo;

    fn sample_entry() -> Entry {
        Entry {
            content: "An architectural decision about the storage layer.".to_string(),
            entry_type: EntryType::Raw,
            source: "[git]".to_string(),
            created_at: 1_740_000_000,
            perspectives: vec!["decisions".to_string()],
            relations: vec![],
            summarizes: vec![],
            heading: Some("Storage Decision".to_string()),
            parent_id: None,
            impression_hint: None,
            impression_strength: 1.0,
            expires_at: None,
            visibility: "normal".to_string(),
            level: ChunkLevel::H1,
            path: String::new(),
        }
    }

    /// Set up a MemoryStore with a stored sample entry, returning both.
    fn setup_with_entry(git_dir: &Path) -> (MemoryStore, Entry) {
        let store = MemoryStore::open(git_dir, Some("test-memory")).unwrap();
        let entry = sample_entry();
        store.store_entry(&entry).unwrap();
        (store, entry)
    }

    // -----------------------------------------------------------------------
    // test_open_creates_branch
    // -----------------------------------------------------------------------

    #[test]
    fn test_open_creates_branch() {
        let (_dir, git_dir) = setup_test_repo();
        let store = MemoryStore::open(&git_dir, Some("test-memory")).unwrap();

        assert!(
            store.branch().branch_exists().unwrap(),
            "branch should exist after open"
        );
    }

    // -----------------------------------------------------------------------
    // test_open_loads_config
    // -----------------------------------------------------------------------

    #[test]
    fn test_open_loads_config() {
        let (_dir, git_dir) = setup_test_repo();

        // Create the branch with a custom config.
        {
            let store = MemoryStore::open(&git_dir, Some("cfg-test")).unwrap();

            let custom_toml = concat!(
                "[memory]\n",
                "[memory.sync]\n",
                "push_mode = \"manual\"\n",
                "branch = \"cfg-test\"\n",
            );
            store
                .branch()
                .write_file(CONFIG_FILENAME, custom_toml.as_bytes())
                .unwrap();
            store.branch().commit("write custom config").unwrap();
        }

        // Re-open and verify the config was loaded.
        let store = MemoryStore::open(&git_dir, Some("cfg-test")).unwrap();
        assert_eq!(store.config().push_mode(), PushMode::Manual);
    }

    // -----------------------------------------------------------------------
    // test_store_and_load_entry
    // -----------------------------------------------------------------------

    #[test]
    fn test_store_and_load_entry() {
        let (_dir, git_dir) = setup_test_repo();
        let store = MemoryStore::open(&git_dir, Some("test-memory")).unwrap();

        let entry = sample_entry();
        let path = store.store_entry(&entry).unwrap();
        assert!(path.ends_with(".md"), "path should end with .md: {path}");

        let loaded = store.load().unwrap();
        assert!(
            loaded.iter().any(|e| e.content == entry.content),
            "stored entry should appear in loaded entries"
        );
    }

    // -----------------------------------------------------------------------
    // test_store_with_auto_push_no_remote
    // -----------------------------------------------------------------------

    #[test]
    fn test_store_with_auto_push_no_remote() {
        let (_dir, git_dir) = setup_test_repo();
        let store = MemoryStore::open(&git_dir, Some("test-memory")).unwrap();

        // Default push_mode is Review; no remote configured — must not fail.
        let entry = sample_entry();
        let result = store.store_entry(&entry);
        assert!(
            result.is_ok(),
            "store_entry should succeed without a remote: {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // test_list_entry_files
    // -----------------------------------------------------------------------

    #[test]
    fn test_list_entry_files() {
        let (_dir, git_dir) = setup_test_repo();
        let store = MemoryStore::open(&git_dir, Some("test-memory")).unwrap();

        // Store an entry so there is at least one .md file.
        store.store_entry(&sample_entry()).unwrap();

        let files = store.list_entry_files().unwrap();

        assert!(
            files.iter().all(|f| f.ends_with(".md")),
            "list should contain only .md files"
        );
        assert!(
            !files.contains(&CONFIG_FILENAME.to_string()),
            "config.toml must not appear in entry file list"
        );
        assert!(
            !files.iter().any(|f| f.starts_with('.') || f.contains("/.")),
            "hidden files must be excluded"
        );
        assert!(!files.is_empty(), "should have at least one entry file");
    }

    // -----------------------------------------------------------------------
    // test_load_with_embeddings
    // -----------------------------------------------------------------------

    #[test]
    fn test_load_with_embeddings() {
        let (_dir, git_dir) = setup_test_repo();
        let (store, entry) = setup_with_entry(&git_dir);

        let model = "test-model";
        let vector: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        store.store_embedding(&entry, model, &vector).unwrap();

        let pairs = store.load_with_embeddings(model).unwrap();
        assert!(!pairs.is_empty(), "should have at least one pair");

        let matching = pairs
            .iter()
            .find(|(e, _)| e.content == entry.content)
            .expect("stored entry should appear in load_with_embeddings");

        let embedding = matching.1.as_ref().expect("embedding should be cached");
        assert_eq!(embedding.len(), vector.len());
        for (got, expected) in embedding.iter().zip(vector.iter()) {
            assert!(
                (got - expected).abs() < f32::EPSILON,
                "embedding value mismatch: got {got}, expected {expected}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // test_gc_removes_orphaned_embeddings
    // -----------------------------------------------------------------------

    #[test]
    fn test_gc_removes_orphaned_embeddings() {
        let (_dir, git_dir) = setup_test_repo();
        let (store, entry) = setup_with_entry(&git_dir);

        let model = "gc-model";
        let vector: Vec<f32> = (0..4).map(|i| i as f32 * 0.5).collect();
        store.store_embedding(&entry, model, &vector).unwrap();

        // Compute the short ID so we can verify the embedding is gone afterwards.
        let id = entry.content_id();
        let short = crate::chunk::short_id(&id).to_string();

        // Confirm the embedding exists before GC.
        let before = store
            .branch()
            .read_embedding(&short, model)
            .expect("read_embedding should not error before gc");
        assert!(before.is_some(), "embedding should exist before gc");

        // Run GC with an empty live-ids list — every embedding is orphaned.
        let deleted = store.gc(&[], model).unwrap();
        assert_eq!(deleted, 1, "gc should report one deleted embedding");

        // The embedding file must be gone.
        let after = store
            .branch()
            .read_embedding(&short, model)
            .expect("read_embedding should not error after gc");
        assert!(after.is_none(), "embedding should be absent after gc");

        // A cleanup commit must exist on the branch.
        let log_output =
            run_git_with_gitdir(&git_dir, &["log", "--oneline", "--grep=gc:", "test-memory"])
                .expect("git log should succeed");
        let log = String::from_utf8_lossy(&log_output.stdout);
        assert!(
            !log.trim().is_empty(),
            "a gc: commit should appear in the branch log"
        );
    }

    // -----------------------------------------------------------------------
    // test_unpushed_commit_count_with_remote
    // -----------------------------------------------------------------------

    #[test]
    fn test_unpushed_commit_count_with_remote() {
        // Create a bare repository to act as the remote.
        let remote_dir = tempfile::tempdir().unwrap();
        std::process::Command::new("git")
            .args(["init", "--bare"])
            .current_dir(remote_dir.path())
            .output()
            .unwrap();

        let (local_dir, git_dir) = setup_test_repo();

        // Configure the bare repo as origin.
        std::process::Command::new("git")
            .args([
                "--git-dir",
                &git_dir.to_string_lossy(),
                "remote",
                "add",
                "origin",
                &remote_dir.path().to_string_lossy(),
            ])
            .output()
            .unwrap();

        let store = MemoryStore::open(&git_dir, Some("test-memory")).unwrap();

        // Store an entry to create a local commit.
        store.store_entry(&sample_entry()).unwrap();

        // Push and set up the remote tracking branch so git log origin/..local works.
        std::process::Command::new("git")
            .args([
                "--git-dir",
                &git_dir.to_string_lossy(),
                "push",
                "--set-upstream",
                "origin",
                "test-memory",
            ])
            .output()
            .unwrap();

        // After the push the local branch is in sync with the remote.
        let count_after_push = store
            .unpushed_commit_count()
            .expect("unpushed_commit_count should succeed");
        assert_eq!(
            count_after_push, 0,
            "no unpushed commits expected immediately after push"
        );

        // Add another local commit with distinct content so git sees a real change.
        let second_entry = Entry {
            content: "A second architectural decision, distinct from the first.".to_string(),
            heading: Some("Second Decision".to_string()),
            ..sample_entry()
        };
        store.store_entry(&second_entry).unwrap();

        // Now there should be exactly one commit ahead of the remote.
        let count_with_new = store
            .unpushed_commit_count()
            .expect("unpushed_commit_count should succeed");
        assert_eq!(
            count_with_new, 1,
            "one unpushed commit expected after storing a second entry"
        );

        // Keep temp dirs alive until end of test.
        drop(local_dir);
        drop(remote_dir);
    }

    // -----------------------------------------------------------------------
    // Remote test helpers
    // -----------------------------------------------------------------------

    /// Helper: run a git command via `--git-dir` with hermetic identity.
    fn git_hermetic_gitdir(git_dir: &Path, args: &[&str]) -> std::process::Output {
        std::process::Command::new("git")
            .arg("--git-dir")
            .arg(git_dir)
            .args(args)
            .env("GIT_AUTHOR_NAME", "Test")
            .env("GIT_AUTHOR_EMAIL", "test@example.com")
            .env("GIT_COMMITTER_NAME", "Test")
            .env("GIT_COMMITTER_EMAIL", "test@example.com")
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_ASKPASS", "echo")
            .env("LC_ALL", "C")
            .output()
            .expect("git command failed to spawn")
    }

    /// Create a bare repo as a local "remote" and add it as `origin` to the
    /// repo at `git_dir`.  Returns the `TempDir` for the bare repo (must be
    /// kept alive for the duration of the test).
    fn add_bare_remote(git_dir: &Path) -> tempfile::TempDir {
        let remote_dir = tempfile::tempdir().unwrap();
        std::process::Command::new("git")
            .args(["init", "--bare"])
            .current_dir(remote_dir.path())
            .output()
            .unwrap();

        git_hermetic_gitdir(
            git_dir,
            &[
                "remote",
                "add",
                "origin",
                &remote_dir.path().to_string_lossy(),
            ],
        );

        remote_dir
    }

    /// Push `branch` from `git_dir` to origin and set the upstream tracking ref.
    fn push_set_upstream(git_dir: &Path, branch: &str) {
        let out = git_hermetic_gitdir(git_dir, &["push", "--set-upstream", "origin", branch]);
        assert!(
            out.status.success(),
            "push --set-upstream failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    // -----------------------------------------------------------------------
    // try_create_from_remote
    // -----------------------------------------------------------------------

    /// When origin has the branch, `try_create_from_remote` must create a
    /// local branch and return `true`.
    #[test]
    fn test_try_create_from_remote_success() {
        // Set up "source" repo with the memory branch and push it to a bare remote.
        let (_src_dir, src_git_dir) = setup_test_repo();
        let _remote_dir = add_bare_remote(&src_git_dir);

        let src_store = MemoryStore::open(&src_git_dir, Some("test-memory")).unwrap();
        src_store.store_entry(&sample_entry()).unwrap();
        push_set_upstream(&src_git_dir, "test-memory");

        // Set up a fresh "destination" repo that also points at the bare remote.
        let (_dst_dir, dst_git_dir) = setup_test_repo();
        git_hermetic_gitdir(
            &dst_git_dir,
            &[
                "remote",
                "add",
                "origin",
                &_remote_dir.path().to_string_lossy(),
            ],
        );

        // Fetch so origin/test-memory exists in the destination.
        let branch = crate::git::GitMemoryBranch::open(&dst_git_dir, Some("test-memory")).unwrap();
        branch.fetch().unwrap();

        // The local branch must not exist yet.
        assert!(!branch.branch_exists().unwrap());

        // try_create_from_remote must return `true` and create the branch.
        let created = try_create_from_remote(&branch).unwrap();
        assert!(created, "expected try_create_from_remote to return true");
        assert!(
            branch.branch_exists().unwrap(),
            "local branch must exist after try_create_from_remote"
        );
    }

    /// When origin does not have the branch, `try_create_from_remote` must
    /// return `false` (not an error).
    #[test]
    fn test_try_create_from_remote_missing_remote_ref_returns_false() {
        let (_dir, git_dir) = setup_test_repo();
        // Add a real remote so `has_remote()` passes, but don't push any branch.
        let _remote_dir = add_bare_remote(&git_dir);

        let branch = crate::git::GitMemoryBranch::open(&git_dir, Some("test-memory")).unwrap();
        // No fetch performed — origin/test-memory does not exist.
        let created = try_create_from_remote(&branch).unwrap();
        assert!(!created, "must return false when remote ref does not exist");
    }

    // -----------------------------------------------------------------------
    // push / pull round-trip
    // -----------------------------------------------------------------------

    /// Store an entry in repo A, push it, then open the same branch in a
    /// second clone and verify the entry arrives via `pull`.
    #[test]
    fn test_push_pull_round_trip() {
        // Repo A — writer.
        let (_dir_a, git_dir_a) = setup_test_repo();
        let remote_dir = add_bare_remote(&git_dir_a);

        let store_a = MemoryStore::open(&git_dir_a, Some("test-memory")).unwrap();
        store_a.store_entry(&sample_entry()).unwrap();

        // Push from A.
        let push_result = store_a.push().unwrap();
        assert_eq!(
            push_result,
            crate::git::PushResult::Success,
            "first push must succeed"
        );
        // Set up the tracking ref so pull knows the remote branch.
        push_set_upstream(&git_dir_a, "test-memory");

        // Repo B — reader.
        let (_dir_b, git_dir_b) = setup_test_repo();
        git_hermetic_gitdir(
            &git_dir_b,
            &[
                "remote",
                "add",
                "origin",
                &remote_dir.path().to_string_lossy(),
            ],
        );

        // Open store B; open() will fetch and find origin/test-memory,
        // calling try_create_from_remote internally.
        let store_b = MemoryStore::open(&git_dir_b, Some("test-memory")).unwrap();

        // load() fetches and reads all entries.
        let entries_b = store_b.load().unwrap();
        assert!(
            entries_b
                .iter()
                .any(|e| e.content == sample_entry().content),
            "entry written in repo A must be visible after pull in repo B; got: {:?}",
            entries_b.iter().map(|e| &e.content).collect::<Vec<_>>()
        );

        drop(remote_dir);
        drop(_dir_a);
        drop(_dir_b);
    }

    // -----------------------------------------------------------------------
    // push — no-remote returns NothingToPush, not an error
    // -----------------------------------------------------------------------

    #[test]
    fn test_push_no_remote_returns_nothing_to_push() {
        let (_dir, git_dir) = setup_test_repo();
        let store = MemoryStore::open(&git_dir, Some("test-memory")).unwrap();
        let result = store.push().unwrap();
        assert_eq!(
            result,
            crate::git::PushResult::NothingToPush,
            "push without a remote must return NothingToPush"
        );
    }

    // -----------------------------------------------------------------------
    // pull — no-remote returns NothingToSync, not an error
    // -----------------------------------------------------------------------

    #[test]
    fn test_pull_no_remote_returns_nothing_to_sync() {
        let (_dir, git_dir) = setup_test_repo();
        let store = MemoryStore::open(&git_dir, Some("test-memory")).unwrap();
        match store.pull().unwrap() {
            crate::git::SyncResult::NothingToSync => {}
            other => panic!("expected NothingToSync, got: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // pull — already up-to-date after push returns NothingToSync
    // -----------------------------------------------------------------------

    #[test]
    fn test_pull_already_up_to_date_after_push() {
        let (_dir, git_dir) = setup_test_repo();
        let _remote_dir = add_bare_remote(&git_dir);

        let store = MemoryStore::open(&git_dir, Some("test-memory")).unwrap();
        store.store_entry(&sample_entry()).unwrap();
        push_set_upstream(&git_dir, "test-memory");

        // After pushing, there's nothing new on remote — pull must be NothingToSync.
        match store.pull().unwrap() {
            crate::git::SyncResult::NothingToSync => {}
            other => panic!("expected NothingToSync immediately after push, got: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // sync — no-remote returns NothingToSync
    // -----------------------------------------------------------------------

    #[test]
    fn test_sync_no_remote_returns_nothing_to_sync() {
        let (_dir, git_dir) = setup_test_repo();
        let store = MemoryStore::open(&git_dir, Some("test-memory")).unwrap();
        match store.sync().unwrap() {
            crate::git::SyncResult::NothingToSync => {}
            other => panic!("expected NothingToSync, got: {other:?}"),
        }
    }

    /// Full sync cycle: store, push (via sync), store more, sync again.
    /// After each sync the repo must be in sync with the remote.
    #[test]
    fn test_sync_push_pull_cycle() {
        let (_dir, git_dir) = setup_test_repo();
        let _remote_dir = add_bare_remote(&git_dir);

        let store = MemoryStore::open(&git_dir, Some("test-memory")).unwrap();
        store.store_entry(&sample_entry()).unwrap();
        // First push via raw git to set up the tracking ref; then sync.
        push_set_upstream(&git_dir, "test-memory");

        // Sync after the upstream is set — should succeed (nothing new on remote).
        let sync_result = store.sync().unwrap();
        // NothingToSync or Success both mean "in sync now".
        match sync_result {
            crate::git::SyncResult::NothingToSync | crate::git::SyncResult::Success => {}
            crate::git::SyncResult::Conflicts(files) => {
                panic!("unexpected conflicts after sync: {files:?}")
            }
        }

        // Store a second entry — now we have a local commit ahead of remote.
        let second = Entry {
            content: "Second entry for sync test.".to_string(),
            heading: Some("Sync Entry Two".to_string()),
            ..sample_entry()
        };
        store.store_entry(&second).unwrap();

        // sync() should push it and report success (not a conflict).
        let sync_result2 = store.sync().unwrap();
        match sync_result2 {
            crate::git::SyncResult::NothingToSync | crate::git::SyncResult::Success => {}
            crate::git::SyncResult::Conflicts(files) => {
                panic!("unexpected conflicts after second sync: {files:?}")
            }
        }
    }

    // -----------------------------------------------------------------------
    // remove_entry
    // -----------------------------------------------------------------------

    /// Store an entry, remove it by ID, verify it no longer appears in load.
    #[test]
    fn test_remove_entry_success() {
        let (_dir, git_dir) = setup_test_repo();
        let store = MemoryStore::open(&git_dir, Some("test-memory")).unwrap();

        let entry = sample_entry();
        store.store_entry(&entry).unwrap();

        let ids = store.list_entry_ids().unwrap();
        assert_eq!(ids.len(), 1, "expected exactly one entry id");
        let id = &ids[0];

        store.remove_entry(id).unwrap();

        let files_after = store.list_entry_files().unwrap();
        assert!(
            files_after.is_empty(),
            "no entry files should remain after removal; got: {files_after:?}"
        );

        let loaded = store.load().unwrap();
        assert!(
            loaded.is_empty(),
            "no entries should be visible after removal"
        );
    }

    /// The removal must persist after reopening the store (it's committed to git).
    #[test]
    fn test_remove_entry_persists_after_reopen() {
        let (_dir, git_dir) = setup_test_repo();

        // Write and remove in the first store instance.
        {
            let store = MemoryStore::open(&git_dir, Some("test-memory")).unwrap();
            store.store_entry(&sample_entry()).unwrap();
            let ids = store.list_entry_ids().unwrap();
            store.remove_entry(&ids[0]).unwrap();
        }

        // Reopen and confirm the entry is gone.
        let store2 = MemoryStore::open(&git_dir, Some("test-memory")).unwrap();
        let files = store2.list_entry_files().unwrap();
        assert!(
            files.is_empty(),
            "entry must not reappear after reopen; got: {files:?}"
        );
    }

    /// Removing a non-existent ID must return a descriptive error.
    #[test]
    fn test_remove_entry_not_found_returns_error() {
        let (_dir, git_dir) = setup_test_repo();
        let store = MemoryStore::open(&git_dir, Some("test-memory")).unwrap();

        let result = store.remove_entry("0000000");
        assert!(
            result.is_err(),
            "remove_entry with unknown id must return an error"
        );
        match result.unwrap_err() {
            GitError::CommandFailed { stderr, .. } => {
                assert!(
                    stderr.contains("no entry found"),
                    "error message should mention 'no entry found', got: {stderr:?}"
                );
            }
            other => panic!("expected CommandFailed, got: {other:?}"),
        }
    }

    /// Ambiguous prefix (matches >1 file) must return an error naming both matches.
    #[test]
    fn test_remove_entry_ambiguous_prefix_returns_error() {
        let (_dir, git_dir) = setup_test_repo();
        let store = MemoryStore::open(&git_dir, Some("test-memory")).unwrap();

        // Store two entries with different content so they get distinct IDs.
        let entry_a = Entry {
            content: "Distinct content for ambiguity test entry A.".to_string(),
            heading: Some("Ambiguous A".to_string()),
            ..sample_entry()
        };
        let entry_b = Entry {
            content: "Distinct content for ambiguity test entry B.".to_string(),
            heading: Some("Ambiguous B".to_string()),
            ..sample_entry()
        };
        store.store_entry(&entry_a).unwrap();
        store.store_entry(&entry_b).unwrap();

        // Both IDs are 7-char hex.  An empty prefix would match both but is not
        // a valid prefix to pass (each 7-char ID is unique; we test a real shared
        // prefix by constructing a partial-match scenario via direct file creation).
        //
        // Instead: write two files whose embedded IDs share the same first char,
        // so that passing that single character as the id is ambiguous.
        // We do this by planting two .md files via the branch directly.
        let branch = store.branch();
        branch
            .write_file("decisions/heading-aaa0001.md", b"# Heading A\ncontent A\n")
            .unwrap();
        branch
            .write_file("decisions/heading-aaa0002.md", b"# Heading B\ncontent B\n")
            .unwrap();
        branch.commit("add two entries with shared prefix").unwrap();

        // Remove with prefix "aaa" — matches both files.
        let result = store.remove_entry("aaa");
        assert!(result.is_err(), "ambiguous prefix must return an error");
        match result.unwrap_err() {
            GitError::CommandFailed { stderr, .. } => {
                assert!(
                    stderr.contains("ambiguous"),
                    "error message should mention 'ambiguous', got: {stderr:?}"
                );
            }
            other => panic!("expected CommandFailed, got: {other:?}"),
        }
    }

    /// Exact full 7-char ID that is also the beginning of another ID must NOT
    /// be treated as ambiguous — path_matches_id requires the short-ID to equal
    /// the query or start with it, so "aaa0001" must not match "aaa0002".
    #[test]
    fn test_remove_entry_exact_id_not_ambiguous() {
        let (_dir, git_dir) = setup_test_repo();
        let store = MemoryStore::open(&git_dir, Some("test-memory")).unwrap();

        // Plant two files whose IDs share a 6-char prefix but differ at position 7.
        let branch = store.branch();
        branch
            .write_file("decisions/entry-abc0001.md", b"# Entry 1\ncontent 1\n")
            .unwrap();
        branch
            .write_file("decisions/entry-abc0002.md", b"# Entry 2\ncontent 2\n")
            .unwrap();
        branch.commit("plant two entries").unwrap();

        // "abc0001" is an exact 7-char match; it must not match "abc0002".
        let result = store.remove_entry("abc0001");
        assert!(
            result.is_ok(),
            "exact 7-char id must not be treated as ambiguous: {:?}",
            result.unwrap_err()
        );

        // "abc0002" must still exist.
        let files = store.list_entry_files().unwrap();
        assert!(
            files.iter().any(|f| f.contains("abc0002")),
            "abc0002 must survive removal of abc0001; files: {files:?}"
        );
    }

    /// An empty id must never match (it would otherwise `starts_with("")`-match
    /// every entry): the caller gets a clean not-found and nothing is deleted.
    #[test]
    fn test_remove_entry_empty_id_matches_nothing() {
        let (_dir, git_dir) = setup_test_repo();
        let store = MemoryStore::open(&git_dir, Some("test-memory")).unwrap();

        let branch = store.branch();
        branch
            .write_file("decisions/entry-abc0001.md", b"# Entry 1\ncontent 1\n")
            .unwrap();
        branch.commit("plant one entry").unwrap();

        let err = store.remove_entry("").unwrap_err();
        match err {
            GitError::CommandFailed { stderr, .. } => assert!(
                stderr.contains("no entry found"),
                "empty id must be not-found, not ambiguous; got: {stderr:?}"
            ),
            other => panic!("expected CommandFailed not-found, got: {other:?}"),
        }

        let files = store.list_entry_files().unwrap();
        assert!(
            files.iter().any(|f| f.contains("abc0001")),
            "no entry may be deleted for an empty id; files: {files:?}"
        );
    }

    // -----------------------------------------------------------------------
    // test_store_same_content_different_heading_no_duplicate
    // -----------------------------------------------------------------------

    #[test]
    fn test_store_same_content_different_heading_no_duplicate() {
        let (_dir, git_dir) = setup_test_repo();
        let store = MemoryStore::open(&git_dir, Some("test-memory")).unwrap();

        // First store: heading "Alpha".
        let entry_alpha = Entry {
            content: "Shared body content for duplicate detection test.".to_string(),
            heading: Some("Alpha".to_string()),
            ..sample_entry()
        };
        store.store_entry(&entry_alpha).unwrap();

        let files_after_first = store.list_entry_files().unwrap();
        assert_eq!(
            files_after_first.len(),
            1,
            "expected exactly one file after first store, got: {files_after_first:?}"
        );

        // Second store: same content, different heading "Beta".
        // The content_id is identical, so only one file should remain.
        let entry_beta = Entry {
            content: "Shared body content for duplicate detection test.".to_string(),
            heading: Some("Beta".to_string()),
            ..sample_entry()
        };
        store.store_entry(&entry_beta).unwrap();

        let files_after_second = store.list_entry_files().unwrap();
        assert_eq!(
            files_after_second.len(),
            1,
            "expected exactly one file after second store, got: {files_after_second:?}"
        );

        // The surviving file should use the new heading slug.
        assert!(
            files_after_second[0].contains("beta"),
            "surviving file should use the new heading slug, got: {:?}",
            files_after_second[0]
        );
    }
}
