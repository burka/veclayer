//! Worktree lifecycle management for the memory branch.
//!
//! Provides a deterministic worktree path under `~/.cache/veclayer/` and
//! write operations (create, delete, commit) that operate via a dedicated
//! git worktree — the user's working directory is never touched.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::{check_output, run_git_in, run_git_with_gitdir, GitError, GitMemoryBranch};

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Compute the worktree path for a given git dir and branch.
///
/// Returns `~/.cache/veclayer/<project_hash>/<branch>/` where `project_hash`
/// is the first 12 hex characters of the SHA-256 of the canonical git dir path.
/// Falls back to `/tmp/veclayer/` if the cache directory cannot be determined.
pub fn worktree_path(git_dir: &Path, branch: &str) -> PathBuf {
    let canonical = git_dir
        .canonicalize()
        .unwrap_or_else(|_| git_dir.to_path_buf());

    let hash = project_hash(&canonical);

    let base = directories::BaseDirs::new()
        .map(|b| b.cache_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("/tmp"));

    base.join("veclayer").join(hash).join(branch)
}

/// Return the first 12 hex characters of the SHA-256 of the path string.
fn project_hash(path: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(path.to_string_lossy().as_bytes());
    format!("{:x}", hasher.finalize())[..12].to_string()
}

// ---------------------------------------------------------------------------
// GitMemoryBranch — worktree and write operations
// ---------------------------------------------------------------------------

impl GitMemoryBranch {
    /// Ensure the worktree for the memory branch exists and return its path.
    ///
    /// If the directory already exists and is a valid git worktree, returns
    /// immediately. Otherwise, adds a new worktree via `git worktree add`.
    ///
    /// NOTE: This method has a TOCTOU race: two concurrent callers could both pass the
    /// is_valid_worktree check and race on removal/creation. This is acceptable because:
    /// 1. MemoryStore is used single-threaded in practice (one MCP request at a time)
    /// 2. The failure mode is a git error (not silent data corruption)
    /// 3. If multi-process access becomes needed, add an advisory file lock here.
    pub fn ensure_worktree(&self) -> Result<&Path, GitError> {
        if is_valid_worktree(&self.worktree_path) {
            // Verify the worktree actually works (catches stale .git pointers
            // from deleted repos that shared the same cache hash).
            let probe = run_git_in(&self.worktree_path, &["rev-parse", "--git-dir"]);
            if probe.as_ref().is_ok_and(|o| o.status.success()) {
                return Ok(&self.worktree_path);
            }
            // Worktree is broken — fall through to recreate it.
        }

        // Remove a stale/incomplete directory that is not a valid worktree.
        if self.worktree_path.exists() {
            std::fs::remove_dir_all(&self.worktree_path).map_err(|e| {
                GitError::WorktreeError(format!("failed to remove stale worktree directory: {e}"))
            })?;
            // Clean up stale worktree metadata in .git/worktrees/
            let _ = run_git_with_gitdir(&self.git_dir, &["worktree", "prune"]);
        }

        let path_str = self.worktree_path.to_string_lossy();
        let output =
            run_git_with_gitdir(&self.git_dir, &["worktree", "add", &path_str, &self.branch])?;
        check_output(&output, "worktree add")?;

        Ok(&self.worktree_path)
    }

    /// Create an orphan memory branch if it does not already exist.
    ///
    /// Uses git plumbing only — no checkout is performed. After creating
    /// the branch, sets up the worktree via [`ensure_worktree`].
    pub fn create_orphan_branch(&self) -> Result<(), GitError> {
        if self.branch_exists()? {
            self.ensure_worktree()?;
            return Ok(());
        }

        let tree_hash = create_empty_tree(&self.git_dir)?;
        let commit_hash = create_initial_commit(&self.git_dir, &tree_hash)?;
        create_branch_ref(&self.git_dir, &self.branch, &commit_hash)?;

        self.ensure_worktree()?;
        Ok(())
    }

    /// Write `content` to `path` (relative to the worktree root).
    ///
    /// Creates parent directories as needed. Does not commit — call
    /// [`commit`] separately.
    pub fn write_file(&self, path: &str, content: &[u8]) -> Result<(), GitError> {
        let worktree = self.ensure_worktree()?;
        let target = validate_path(worktree, path)?;

        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                GitError::WorktreeError(format!("failed to create parent directories: {e}"))
            })?;
        }

        std::fs::write(&target, content)
            .map_err(|e| GitError::WorktreeError(format!("failed to write file '{path}': {e}")))?;

        Ok(())
    }

    /// Remove `path` (relative to the worktree root) from the filesystem and
    /// stage the deletion via `git rm`.
    pub fn delete_file(&self, path: &str) -> Result<(), GitError> {
        let worktree = self.ensure_worktree()?;
        validate_path(worktree, path)?;
        // Use `--` to prevent the path from being interpreted as a flag.
        let output = run_git_in(worktree, &["rm", "--", path])?;
        check_output(&output, "rm")?;
        Ok(())
    }

    /// Stage all changes and create a commit with `message`.
    ///
    /// If there is nothing to commit (clean tree), returns `Ok(())`.
    /// Configures a fallback commit author if none is set in the repository.
    pub fn commit(&self, message: &str) -> Result<(), GitError> {
        let worktree = self.ensure_worktree()?;

        configure_author_if_missing(worktree)?;

        let add_output = run_git_in(worktree, &["add", "-A"])?;
        check_output(&add_output, "add -A")?;

        let commit_output = run_git_in(worktree, &["commit", "-m", message])?;

        if commit_output.status.success() {
            return Ok(());
        }

        let stderr = String::from_utf8_lossy(&commit_output.stderr);
        let stdout = String::from_utf8_lossy(&commit_output.stdout);
        if stderr.contains("nothing to commit") || stdout.contains("nothing to commit") {
            return Ok(());
        }

        Err(GitError::CommandFailed {
            command: "commit".to_string(),
            stderr: stderr.trim().to_string(),
            exit_code: commit_output.status.code().unwrap_or(-1),
        })
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Validate that a relative path stays within the worktree root.
///
/// Rejects paths containing `..` components or absolute paths to prevent
/// path traversal attacks from untrusted data (e.g., perspective names).
fn validate_path(worktree: &Path, relative: &str) -> Result<PathBuf, GitError> {
    use std::path::Component;

    let has_traversal = Path::new(relative)
        .components()
        .any(|c| c == Component::ParentDir);

    if has_traversal || Path::new(relative).is_absolute() {
        return Err(GitError::WorktreeError(format!(
            "path '{relative}' escapes worktree root"
        )));
    }
    Ok(worktree.join(relative))
}

/// Return true if `path` is a directory containing a `.git` entry,
/// indicating it is a valid git worktree checkout.
fn is_valid_worktree(path: &Path) -> bool {
    path.is_dir() && path.join(".git").exists()
}

/// Return the well-known empty tree SHA-1 (a git canonical constant).
fn create_empty_tree(_git_dir: &Path) -> Result<String, GitError> {
    Ok("4b825dc642cb6eb9a060e54bf8d69288fbee4904".to_string())
}

/// Create an initial commit on top of `tree_hash` and return the commit hash.
fn create_initial_commit(git_dir: &Path, tree_hash: &str) -> Result<String, GitError> {
    let output = run_git_with_gitdir(
        git_dir,
        &["commit-tree", tree_hash, "-m", "Initialize memory branch"],
    )?;
    check_output(&output, "commit-tree")?;
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Point `refs/heads/<branch>` at `commit_hash`.
fn create_branch_ref(git_dir: &Path, branch: &str, commit_hash: &str) -> Result<(), GitError> {
    let ref_name = format!("refs/heads/{branch}");
    let output = run_git_with_gitdir(git_dir, &["update-ref", &ref_name, commit_hash])?;
    check_output(&output, "update-ref")?;
    Ok(())
}

/// Set `user.name` and `user.email` locally in the worktree if not already configured.
fn configure_author_if_missing(worktree: &Path) -> Result<(), GitError> {
    let name_output = run_git_in(worktree, &["config", "user.name"])?;
    if !name_output.status.success()
        || String::from_utf8_lossy(&name_output.stdout)
            .trim()
            .is_empty()
    {
        let out = run_git_in(worktree, &["config", "user.name", "VecLayer"])?;
        check_output(&out, "config user.name")?;
    }

    let email_output = run_git_in(worktree, &["config", "user.email"])?;
    if !email_output.status.success()
        || String::from_utf8_lossy(&email_output.stdout)
            .trim()
            .is_empty()
    {
        let out = run_git_in(worktree, &["config", "user.email", "noreply@veclayer.dev"])?;
        check_output(&out, "config user.email")?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_test_repo() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let git_dir = dir.path().join(".git");
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "--allow-empty", "-m", "init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        (dir, git_dir)
    }

    // -----------------------------------------------------------------------
    // validate_path
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_path_traversal_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let result = validate_path(dir.path(), "decisions/../../../etc/passwd");
        assert!(result.is_err(), "path with .. should be rejected");
    }

    #[test]
    fn test_validate_path_absolute_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let result = validate_path(dir.path(), "/etc/passwd");
        assert!(result.is_err(), "absolute path should be rejected");
    }

    #[test]
    fn test_validate_path_normal_relative_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let result = validate_path(dir.path(), "decisions/heading-abc1234.md");
        assert!(result.is_ok(), "normal relative path should be accepted");
    }

    #[test]
    fn test_validate_path_dash_prefix_accepted() {
        // validate_path does NOT guard against paths starting with '-' — that
        // guard is provided by the '--' separator in git commands. Document
        // that the function itself accepts such paths (the git call is safe).
        let dir = tempfile::tempdir().unwrap();
        let result = validate_path(dir.path(), "-not-a-flag.md");
        assert!(
            result.is_ok(),
            "validate_path allows dash-prefixed paths; git '--' separator protects the call"
        );
    }

    #[test]
    fn test_worktree_path_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        let git_dir = dir.path().join(".git");

        let path_a = worktree_path(&git_dir, "test-branch");
        let path_b = worktree_path(&git_dir, "test-branch");

        assert_eq!(path_a, path_b);
    }

    #[test]
    fn test_worktree_path_different_repos() {
        let dir_a = tempfile::tempdir().unwrap();
        let dir_b = tempfile::tempdir().unwrap();

        let git_dir_a = dir_a.path().join(".git");
        let git_dir_b = dir_b.path().join(".git");

        let path_a = worktree_path(&git_dir_a, "memory");
        let path_b = worktree_path(&git_dir_b, "memory");

        assert_ne!(path_a, path_b);
    }

    #[test]
    fn test_create_orphan_and_worktree() {
        let (_dir, git_dir) = setup_test_repo();

        let branch = crate::git::GitMemoryBranch::open(&git_dir, Some("test-memory")).unwrap();
        branch.create_orphan_branch().unwrap();

        assert!(
            branch.worktree_path.is_dir(),
            "worktree directory should exist"
        );
        assert!(
            branch.worktree_path.join(".git").exists(),
            "worktree should contain a .git file"
        );
    }

    #[test]
    fn test_write_and_commit() {
        let (_dir, git_dir) = setup_test_repo();

        let branch = crate::git::GitMemoryBranch::open(&git_dir, Some("write-test")).unwrap();
        branch.create_orphan_branch().unwrap();

        branch
            .write_file("notes/hello.txt", b"hello world")
            .unwrap();
        branch.commit("add hello.txt").unwrap();

        // Verify the commit exists on the branch via plumbing.
        let output = run_git_with_gitdir(&git_dir, &["log", "--oneline", "write-test"]).unwrap();
        assert!(output.status.success(), "git log should succeed");

        let log = String::from_utf8_lossy(&output.stdout);
        assert!(
            log.contains("add hello.txt"),
            "commit message should appear in log; got: {log}"
        );
    }
}
