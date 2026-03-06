//! Remote sync operations for the git memory branch.
//!
//! Implements fetch, push, pull-rebase and force-push on [`GitMemoryBranch`].
//! Fetch and push operate purely on refs and require no worktree.
//! Pull-rebase modifies files and therefore requires a live worktree.

use super::{run_git_in, run_git_with_gitdir, GitError, GitMemoryBranch, PushResult, SyncResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const REMOTE: &str = "origin";

// ---------------------------------------------------------------------------
// Auth-failure detection
// ---------------------------------------------------------------------------

fn is_auth_failure(stderr: &str) -> bool {
    stderr.contains("Authentication failed")
        || stderr.contains("Permission denied")
        || stderr.contains("fatal: could not read")
}

// ---------------------------------------------------------------------------
// Sync methods
// ---------------------------------------------------------------------------

impl GitMemoryBranch {
    /// Return `true` if an `origin` remote is configured for this repository.
    pub fn has_remote(&self) -> Result<bool, GitError> {
        let output = run_git_with_gitdir(&self.git_dir, &["remote", "get-url", REMOTE])?;
        Ok(output.status.success())
    }

    /// Fetch the memory branch from `origin`.
    ///
    /// Silently succeeds when no remote is configured (local-only repository).
    pub fn fetch(&self) -> Result<(), GitError> {
        if !self.has_remote()? {
            return Ok(());
        }

        let output = run_git_with_gitdir(&self.git_dir, &["fetch", REMOTE, &self.branch])?;

        if output.status.success() {
            return Ok(());
        }

        let stderr = String::from_utf8_lossy(&output.stderr);
        if is_auth_failure(&stderr) {
            return Err(GitError::AuthFailed(stderr.trim().to_string()));
        }

        Err(GitError::CommandFailed {
            command: format!("fetch {REMOTE} {}", self.branch),
            stderr: stderr.trim().to_string(),
            exit_code: output.status.code().unwrap_or(-1),
        })
    }

    /// Push the memory branch to `origin`.
    ///
    /// Returns [`PushResult::NothingToPush`] when the remote is already
    /// up-to-date, and [`PushResult::Rejected`] on a non-fast-forward rejection.
    pub fn push(&self) -> Result<PushResult, GitError> {
        self.run_push(&["push", REMOTE, &self.branch])
    }

    /// Force-push the memory branch using `--force-with-lease`.
    ///
    /// Safe to call for embedding-cache updates where overwrites are expected.
    pub fn push_force_with_lease(&self) -> Result<PushResult, GitError> {
        self.run_push(&["push", "--force-with-lease", REMOTE, &self.branch])
    }

    /// Pull changes from `origin/<branch>` and rebase local commits on top.
    ///
    /// Requires a live worktree; one is created lazily if absent.
    pub fn pull_rebase(&self) -> Result<SyncResult, GitError> {
        self.ensure_worktree()?;

        let output = run_git_in(
            &self.worktree_path,
            &["pull", "--rebase", REMOTE, &self.branch],
        )?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{stdout}{stderr}");

        if output.status.success() {
            if combined.contains("Already up to date") {
                return Ok(SyncResult::NothingToSync);
            }
            return Ok(SyncResult::Success);
        }

        if is_auth_failure(&stderr) {
            return Err(GitError::AuthFailed(stderr.trim().to_string()));
        }

        if combined.contains("CONFLICT") || combined.contains("could not apply") {
            let files = extract_conflict_files(&combined);
            let _ = run_git_in(&self.worktree_path, &["rebase", "--abort"]);
            return Ok(SyncResult::Conflicts(files));
        }

        Err(GitError::CommandFailed {
            command: format!("pull --rebase {REMOTE} {}", self.branch),
            stderr: stderr.trim().to_string(),
            exit_code: output.status.code().unwrap_or(-1),
        })
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn run_push(&self, args: &[&str]) -> Result<PushResult, GitError> {
        let output = run_git_with_gitdir(&self.git_dir, args)?;

        let stderr = String::from_utf8_lossy(&output.stderr);

        if output.status.success() {
            if stderr.contains("Everything up-to-date") {
                return Ok(PushResult::NothingToPush);
            }
            return Ok(PushResult::Success);
        }

        if is_auth_failure(&stderr) {
            return Err(GitError::AuthFailed(stderr.trim().to_string()));
        }

        if stderr.contains("rejected") || stderr.contains("[rejected]") {
            return Ok(PushResult::Rejected);
        }

        Err(GitError::CommandFailed {
            command: args.join(" "),
            stderr: stderr.trim().to_string(),
            exit_code: output.status.code().unwrap_or(-1),
        })
    }
}

// ---------------------------------------------------------------------------
// Conflict-file extraction
// ---------------------------------------------------------------------------

/// Extract conflicting file paths from git's rebase output.
///
/// Git prints `CONFLICT (content): Merge conflict in <path>` for each file.
fn extract_conflict_files(output: &str) -> Vec<String> {
    output
        .lines()
        .filter_map(|line| {
            if !line.contains("CONFLICT") {
                return None;
            }
            // The path follows "Merge conflict in " — use find to split once.
            if let Some(idx) = line.find("Merge conflict in ") {
                let path = line[idx + "Merge conflict in ".len()..].trim();
                if !path.is_empty() {
                    return Some(path.to_string());
                }
            }
            None
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    fn setup_test_repo() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let git_dir = dir.path().join(".git");
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["-c", "user.email=test@test.com", "-c", "user.name=Test"])
            .args(["commit", "--allow-empty", "-m", "init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        (dir, git_dir)
    }

    fn open_branch(git_dir: &Path) -> GitMemoryBranch {
        GitMemoryBranch::open(git_dir, Some("test-memory")).unwrap()
    }

    // -----------------------------------------------------------------------
    // has_remote
    // -----------------------------------------------------------------------

    #[test]
    fn test_has_remote_no_remote() {
        let (_dir, git_dir) = setup_test_repo();
        let branch = open_branch(&git_dir);
        assert!(!branch.has_remote().unwrap());
    }

    #[test]
    fn test_has_remote_with_remote() {
        let (_dir, git_dir) = setup_test_repo();

        std::process::Command::new("git")
            .args(["--git-dir", &git_dir.to_string_lossy()])
            .args(["remote", "add", "origin", "https://example.com/repo.git"])
            .output()
            .unwrap();

        let branch = open_branch(&git_dir);
        assert!(branch.has_remote().unwrap());
    }

    // -----------------------------------------------------------------------
    // fetch
    // -----------------------------------------------------------------------

    #[test]
    fn test_fetch_no_remote_ok() {
        let (_dir, git_dir) = setup_test_repo();
        let branch = open_branch(&git_dir);
        assert!(branch.fetch().is_ok());
    }

    // -----------------------------------------------------------------------
    // push
    // -----------------------------------------------------------------------

    #[test]
    fn test_push_no_remote() {
        let (_dir, git_dir) = setup_test_repo();

        // Create the branch so push has something to push.
        std::process::Command::new("git")
            .args(["--git-dir", &git_dir.to_string_lossy()])
            .args(["branch", "test-memory"])
            .output()
            .unwrap();

        let branch = open_branch(&git_dir);
        let result = branch.push();
        assert!(
            result.is_err(),
            "expected error when pushing without a remote configured"
        );
    }

    #[test]
    fn test_push_between_local_repos() {
        // Bare "server" repository.
        let server_dir = tempfile::tempdir().unwrap();
        std::process::Command::new("git")
            .args(["init", "--bare"])
            .current_dir(server_dir.path())
            .output()
            .unwrap();

        // Client repository with an initial commit.
        let (client_dir, client_git_dir) = setup_test_repo();

        // Wire the server as origin.
        std::process::Command::new("git")
            .args([
                "--git-dir",
                &client_git_dir.to_string_lossy(),
                "remote",
                "add",
                "origin",
                &server_dir.path().to_string_lossy(),
            ])
            .output()
            .unwrap();

        // Create the memory branch in the client.
        std::process::Command::new("git")
            .args([
                "--git-dir",
                &client_git_dir.to_string_lossy(),
                "branch",
                "test-memory",
            ])
            .output()
            .unwrap();

        let branch = open_branch(&client_git_dir);
        assert!(branch.has_remote().unwrap());

        let first = branch.push().unwrap();
        assert_eq!(first, PushResult::Success);

        // Second push with no new commits — nothing to push.
        let second = branch.push().unwrap();
        assert_eq!(second, PushResult::NothingToPush);

        // Keep dirs alive until end of test.
        drop(client_dir);
        drop(server_dir);
    }
}
