//! Remote sync operations for the git memory branch.
//!
//! Implements fetch, push, pull-rebase and force-push on [`GitMemoryBranch`].
//! Fetch and push operate purely on refs and require no worktree.
//! Pull-rebase modifies files and therefore requires a live worktree.

use super::{
    run_git_in, run_git_with_gitdir, GitError, GitMemoryBranch, PushResult, SyncResult, REMOTE,
};

// ---------------------------------------------------------------------------
// Auth-failure detection and guidance
// ---------------------------------------------------------------------------

/// True when the git stderr indicates an authentication/credential problem.
fn is_auth_failure(stderr: &str) -> bool {
    stderr.contains("Authentication failed")
        || stderr.contains("Permission denied")
        || stderr.contains("fatal: could not read")
        || stderr.contains("Host key verification failed")
        || stderr.contains("Permission denied (publickey)")
        || stderr.contains("terminal prompts disabled")
        || stderr.contains("could not read Username")
}

/// True when the git stderr indicates a network/connectivity problem.
fn is_network_failure(stderr: &str) -> bool {
    stderr.contains("Connection refused")
        || stderr.contains("Connection timed out")
        || stderr.contains("Could not resolve hostname")
        || stderr.contains("ssh: connect to host")
}

/// Build an appropriate `GitError` for auth or network failures, with
/// an actionable hint appended to the message.
fn classified_connection_error(stderr: &str, command: &str) -> GitError {
    let hint = if stderr.contains("publickey") || stderr.contains("Permission denied") {
        "\n\nHint: Check that your SSH key is loaded (ssh-add -l) and that ssh-agent is running."
    } else if stderr.contains("Host key verification") {
        "\n\nHint: The host key is not trusted. Run: ssh -T git@<host> to verify and accept it."
    } else if stderr.contains("Could not resolve hostname")
        || stderr.contains("Connection refused")
        || stderr.contains("Connection timed out")
    {
        "\n\nHint: Network issue — check your internet connection and the remote URL."
    } else if stderr.contains("terminal prompts disabled")
        || stderr.contains("could not read Username")
    {
        "\n\nHint: Git tried to prompt for credentials but prompts are disabled. Configure SSH keys or a credential helper."
    } else {
        ""
    };

    let msg = format!("{}{hint}", stderr.trim());

    if is_auth_failure(stderr) {
        GitError::AuthFailed(msg)
    } else {
        GitError::CommandFailed {
            command: command.to_string(),
            stderr: msg,
            exit_code: -1,
        }
    }
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
    /// Silently succeeds when no remote is configured (local-only repository)
    /// or when the branch does not exist on the remote yet (first use).
    pub fn fetch(&self) -> Result<(), GitError> {
        if !self.has_remote()? {
            return Ok(());
        }

        let output = run_git_with_gitdir(&self.git_dir, &["fetch", REMOTE, &self.branch])?;

        if output.status.success() {
            return Ok(());
        }

        let stderr = String::from_utf8_lossy(&output.stderr);

        // Branch doesn't exist on remote yet — nothing to fetch (first push pending).
        if stderr.contains("couldn't find remote ref") {
            return Ok(());
        }

        let cmd = format!("fetch {REMOTE} {}", self.branch);
        if is_auth_failure(&stderr) || is_network_failure(&stderr) {
            return Err(classified_connection_error(&stderr, &cmd));
        }

        Err(GitError::CommandFailed {
            command: cmd,
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

        if is_auth_failure(&stderr) || is_network_failure(&stderr) {
            let cmd = format!("pull --rebase {REMOTE} {}", self.branch);
            return Err(classified_connection_error(&stderr, &cmd));
        }

        if combined.contains("CONFLICT") || combined.contains("could not apply") {
            let files = extract_conflict_files(&combined);
            if self.try_resolve_identical_conflicts(&files)? {
                return Ok(SyncResult::Success);
            }
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

    /// Attempt to auto-resolve conflicts where every conflicting file is
    /// byte-identical between ours (stage 2) and theirs (stage 3).
    ///
    /// Returns `true` when all conflicts were resolved and the rebase was
    /// continued successfully.  Returns `false` when at least one file has
    /// genuinely different content — the caller must abort the rebase.
    fn try_resolve_identical_conflicts(&self, files: &[String]) -> Result<bool, GitError> {
        if files.is_empty() {
            return Ok(false);
        }

        for file in files {
            let ours = run_git_in(&self.worktree_path, &["show", &format!(":2:{file}")])?;
            let theirs = run_git_in(&self.worktree_path, &["show", &format!(":3:{file}")])?;
            if ours.stdout != theirs.stdout {
                return Ok(false);
            }
        }

        // All conflicting files are byte-identical — accept theirs and continue.
        for file in files {
            run_git_in(&self.worktree_path, &["checkout", "--theirs", file])?;
            run_git_in(&self.worktree_path, &["add", file])?;
        }

        let continue_output = run_git_in(&self.worktree_path, &["rebase", "--continue"])?;
        Ok(continue_output.status.success())
    }

    fn run_push(&self, args: &[&str]) -> Result<PushResult, GitError> {
        let output = run_git_with_gitdir(&self.git_dir, args)?;

        let stderr = String::from_utf8_lossy(&output.stderr);

        if output.status.success() {
            if stderr.contains("Everything up-to-date") {
                return Ok(PushResult::NothingToPush);
            }
            return Ok(PushResult::Success);
        }

        if is_auth_failure(&stderr) || is_network_failure(&stderr) {
            let cmd = args.join(" ");
            return Err(classified_connection_error(&stderr, &cmd));
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
    use crate::git::test_helpers::setup_test_repo;
    use std::path::Path;

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

    // -----------------------------------------------------------------------
    // is_auth_failure — new patterns
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_auth_failure_patterns() {
        let auth_patterns = [
            "Authentication failed",
            "Permission denied",
            "fatal: could not read",
            "Host key verification failed",
            "Permission denied (publickey)",
            "terminal prompts disabled",
            "could not read Username for 'https://github.com'",
        ];
        for pattern in auth_patterns {
            assert!(
                is_auth_failure(pattern),
                "expected is_auth_failure=true for: {pattern:?}"
            );
        }
    }

    #[test]
    fn test_is_network_failure_patterns() {
        let network_patterns = [
            "Connection refused",
            "Connection timed out",
            "Could not resolve hostname example.com",
            "ssh: connect to host github.com port 22",
        ];
        for pattern in network_patterns {
            assert!(
                is_network_failure(pattern),
                "expected is_network_failure=true for: {pattern:?}"
            );
            assert!(
                !is_auth_failure(pattern),
                "network errors should NOT be auth failures: {pattern:?}"
            );
        }
    }

    #[test]
    fn test_is_auth_failure_non_auth_errors_return_false() {
        let non_auth = [
            "repository not found",
            "fatal: not a git repository",
            "error: failed to push some refs",
            "remote: Repository not found.",
        ];
        for msg in non_auth {
            assert!(
                !is_auth_failure(msg),
                "expected is_auth_failure to return false for: {msg:?}"
            );
            assert!(
                !is_network_failure(msg),
                "expected is_network_failure to return false for: {msg:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // classified_connection_error
    // -----------------------------------------------------------------------

    #[test]
    fn test_classified_connection_error_ssh_key_hint() {
        let err = classified_connection_error("Permission denied (publickey).", "push");
        match err {
            GitError::AuthFailed(msg) => {
                assert!(
                    msg.contains("ssh-add -l"),
                    "expected SSH key hint in message, got: {msg:?}"
                );
            }
            other => panic!("expected AuthFailed, got {other:?}"),
        }
    }

    #[test]
    fn test_classified_connection_error_network_hint() {
        let err = classified_connection_error(
            "ssh: connect to host github.com: Connection refused",
            "fetch",
        );
        match err {
            GitError::CommandFailed { stderr, .. } => {
                assert!(
                    stderr.contains("Network issue"),
                    "expected network hint in message, got: {stderr:?}"
                );
            }
            other => panic!("expected CommandFailed for network error, got {other:?}"),
        }
    }

    #[test]
    fn test_classified_connection_error_credential_hint() {
        let err = classified_connection_error(
            "fatal: could not read Username: terminal prompts disabled",
            "push",
        );
        match err {
            GitError::AuthFailed(msg) => {
                assert!(
                    msg.contains("credential helper"),
                    "expected credential hint in message, got: {msg:?}"
                );
            }
            other => panic!("expected AuthFailed, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // extract_conflict_files
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_conflict_files_multiple_conflicts() {
        let output = "\
CONFLICT (content): Merge conflict in src/main.rs
CONFLICT (content): Merge conflict in README.md
Auto-merging src/lib.rs
CONFLICT (modify/delete): Merge conflict in src/deleted.rs
";
        let files = extract_conflict_files(output);
        assert_eq!(
            files,
            vec!["src/main.rs", "README.md", "src/deleted.rs"],
            "conflict files did not match expected"
        );
    }

    #[test]
    fn test_extract_conflict_files_no_conflicts_returns_empty() {
        let output = "Successfully rebased and updated refs/heads/main.\n";
        let files = extract_conflict_files(output);
        assert!(
            files.is_empty(),
            "expected empty vec for output with no conflicts"
        );
    }

    // -----------------------------------------------------------------------
    // try_resolve_identical_conflicts — test helpers
    // -----------------------------------------------------------------------

    /// Run a git command in `dir` with hermetic author identity env vars.
    /// Panics on spawn failure; returns the raw `Output` (caller checks status).
    fn git_in_hermetic(dir: &std::path::Path, args: &[&str]) -> std::process::Output {
        let mut cmd = std::process::Command::new("git");
        cmd.current_dir(dir)
            .args(args)
            .env("GIT_AUTHOR_NAME", "Test")
            .env("GIT_AUTHOR_EMAIL", "test@example.com")
            .env("GIT_COMMITTER_NAME", "Test")
            .env("GIT_COMMITTER_EMAIL", "test@example.com")
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_EDITOR", ":")
            .env("GIT_SEQUENCE_EDITOR", ":")
            .env("LC_ALL", "C");
        cmd.output().expect("git command failed to spawn")
    }

    /// Store `content` as a git loose blob and return its hex SHA-1.
    fn store_git_blob(worktree: &std::path::Path, content: &[u8]) -> String {
        let mut cmd = std::process::Command::new("git");
        cmd.current_dir(worktree)
            .args(["hash-object", "-w", "--stdin"])
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("LC_ALL", "C")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped());
        let mut child = cmd.spawn().expect("git hash-object failed to spawn");
        {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(content).unwrap();
        }
        let out = child.wait_with_output().unwrap();
        let hash = String::from_utf8_lossy(&out.stdout).trim().to_string();
        assert!(!hash.is_empty(), "git hash-object produced empty hash");
        hash
    }

    /// Overwrite stages 2 and 3 for `filename` in the repo at `worktree` to
    /// use the blobs identified by `stage2_hash` and `stage3_hash`.
    fn patch_index_stages(
        worktree: &std::path::Path,
        filename: &str,
        stage2_hash: &str,
        stage3_hash: &str,
    ) {
        let index_info =
            format!("100644 {stage2_hash} 2\t{filename}\n100644 {stage3_hash} 3\t{filename}\n");
        let mut cmd = std::process::Command::new("git");
        cmd.current_dir(worktree)
            .args(["update-index", "--index-info"])
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("LC_ALL", "C")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        let mut child = cmd.spawn().expect("git update-index failed to spawn");
        {
            use std::io::Write;
            child
                .stdin
                .take()
                .unwrap()
                .write_all(index_info.as_bytes())
                .unwrap();
        }
        child.wait_with_output().unwrap();
    }

    /// Build a `GitMemoryBranch` whose `worktree_path` is `worktree` and
    /// `git_dir` is `git_dir`.  Branch name is set to "test-memory".
    fn branch_for_worktree(
        git_dir: &std::path::Path,
        worktree: &std::path::Path,
    ) -> GitMemoryBranch {
        GitMemoryBranch {
            git_dir: git_dir.to_path_buf(),
            branch: "test-memory".to_string(),
            worktree_path: worktree.to_path_buf(),
        }
    }

    /// Set up a git repo with a genuine rebase conflict where `feature_content`
    /// and `main_content` are **different**, so git actually pauses at a conflict.
    ///
    /// `feature_content` and `main_content` must be distinct byte sequences —
    /// git would skip identical changes as "already upstream".
    ///
    /// Returns `(TempDir, git_dir, worktree_path)` with:
    ///   - `git rebase main` paused at a content conflict on `filename`.
    ///   - Index stage 2 = upstream (main's HEAD), stage 3 = feature's commit.
    fn setup_different_content_conflict(
        filename: &str,
        base_content: &[u8],
        feature_content: &[u8],
        main_content: &[u8],
    ) -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
        assert_ne!(
            feature_content, main_content,
            "setup_different_content_conflict: feature and main content must differ"
        );

        let dir = tempfile::tempdir().unwrap();
        let worktree = dir.path().to_path_buf();
        let git_dir = worktree.join(".git");

        git_in_hermetic(&worktree, &["init", "-b", "main"]);
        git_in_hermetic(&worktree, &["config", "commit.gpgsign", "false"]);
        std::fs::write(worktree.join(filename), base_content).unwrap();
        git_in_hermetic(&worktree, &["add", filename]);
        git_in_hermetic(&worktree, &["commit", "-m", "base"]);

        // Feature branch — changes `filename` to `feature_content`.
        git_in_hermetic(&worktree, &["checkout", "-b", "feature"]);
        std::fs::write(worktree.join(filename), feature_content).unwrap();
        git_in_hermetic(&worktree, &["add", filename]);
        git_in_hermetic(&worktree, &["commit", "-m", "feature change"]);

        // Main — changes `filename` to `main_content` (different).
        git_in_hermetic(&worktree, &["checkout", "main"]);
        std::fs::write(worktree.join(filename), main_content).unwrap();
        git_in_hermetic(&worktree, &["add", filename]);
        git_in_hermetic(&worktree, &["commit", "-m", "main change"]);

        // Rebase feature onto main.  Exit status is non-zero on conflict — that's expected.
        git_in_hermetic(&worktree, &["checkout", "feature"]);
        let _ = git_in_hermetic(&worktree, &["rebase", "main"]);

        // Assert the repo is genuinely in conflict state.
        let status = git_in_hermetic(&worktree, &["status", "--short"]);
        let status_out = String::from_utf8_lossy(&status.stdout);
        assert!(
            status_out.contains("UU"),
            "expected rebase conflict (UU) for filename={filename:?}, got status: {status_out}"
        );

        (dir, git_dir, worktree)
    }

    /// Set up a rebase conflict where both stage 2 and stage 3 of `filename`
    /// contain identical bytes (`resolved_content`).
    ///
    /// Because git auto-skips commits whose result is already in the new base,
    /// a normal rebase cannot produce stage2 == stage3.  Instead we:
    ///   1. Create a genuine conflict with sentinel strings (different content).
    ///   2. Overwrite both stages via `git update-index --index-info`.
    ///   3. Write conflict markers to the working-tree file so that
    ///      `git checkout --theirs` can subsequently write the clean file.
    ///
    /// Returns `(TempDir, git_dir, worktree_path)` ready for
    /// `try_resolve_identical_conflicts`.
    fn setup_identical_stages_conflict(
        filename: &str,
        resolved_content: &[u8],
    ) -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
        let (dir, git_dir, worktree) = setup_different_content_conflict(
            filename,
            b"SENTINEL_BASE\n",
            b"SENTINEL_FEATURE\n",
            b"SENTINEL_MAIN\n",
        );

        let blob_hash = store_git_blob(&worktree, resolved_content);
        patch_index_stages(&worktree, filename, &blob_hash, &blob_hash);

        // Write conflict markers so `git checkout --theirs` has something to replace.
        let markers = "<<<<<<< HEAD\nSENTINEL_MAIN\n=======\nSENTINEL_FEATURE\n>>>>>>> feature\n";
        std::fs::write(worktree.join(filename), markers.as_bytes()).unwrap();

        (dir, git_dir, worktree)
    }

    // -----------------------------------------------------------------------
    // try_resolve_identical_conflicts — empty file list
    // -----------------------------------------------------------------------

    /// An empty conflict list must return `false` — there is nothing to resolve,
    /// and calling `rebase --continue` on a clean tree would be wrong.
    #[test]
    fn test_try_resolve_identical_conflicts_empty_list_returns_false() {
        let (_dir, git_dir) = setup_test_repo();
        let branch = open_branch(&git_dir);
        let result = branch.try_resolve_identical_conflicts(&[]).unwrap();
        assert!(
            !result,
            "empty conflict list must return false (nothing to resolve)"
        );
    }

    // -----------------------------------------------------------------------
    // try_resolve_identical_conflicts — non-identical content (must NOT resolve)
    // -----------------------------------------------------------------------

    /// When ours and theirs differ, the function must return `false` — it must
    /// NOT auto-resolve a genuine conflict.  This is the data-integrity guard.
    #[test]
    fn test_try_resolve_identical_conflicts_different_content_returns_false() {
        let (_dir, git_dir, worktree) = setup_different_content_conflict(
            "data.md",
            b"ORIGINAL\n",
            b"FEATURE_EDIT\n",
            b"MAIN_EDIT\n",
        );

        let branch = branch_for_worktree(&git_dir, &worktree);
        let result = branch
            .try_resolve_identical_conflicts(&["data.md".to_string()])
            .unwrap();

        assert!(
            !result,
            "non-identical conflict must NOT be auto-resolved (data integrity)"
        );
    }

    /// Short-circuit: with two files listed, a differing first file must cause
    /// immediate `false` without touching the second file.
    #[test]
    fn test_try_resolve_identical_conflicts_short_circuits_on_first_difference() {
        let (_dir, git_dir, worktree) = setup_different_content_conflict(
            "notes.md",
            b"ORIGINAL\n",
            b"FEATURE_EDIT\n",
            b"MAIN_EDIT\n",
        );

        let branch = branch_for_worktree(&git_dir, &worktree);
        // Passing the same file twice: the first iteration will find a difference
        // and return false immediately.
        let result = branch
            .try_resolve_identical_conflicts(&["notes.md".to_string(), "notes.md".to_string()])
            .unwrap();

        assert!(!result, "any differing file must cause false return");
    }

    /// A trailing-newline difference (`content\n` vs `content`) is a genuine
    /// byte-level difference.  The function must NOT treat it as identical.
    ///
    /// Single-character differences can confuse git's "already upstream" skip,
    /// so we patch the index manually to inject the exact bytes we want.
    #[test]
    fn test_try_resolve_identical_conflicts_trailing_newline_difference_is_not_identical() {
        let (_dir, git_dir, worktree) = setup_different_content_conflict(
            "ws.md",
            b"ORIGINAL\n",
            b"SENTINEL_A\n",
            b"SENTINEL_B\n",
        );

        // Overwrite stage 2/3 with the strings that differ only in trailing newline.
        let hash_with_newline = store_git_blob(&worktree, b"content\n");
        let hash_without_newline = store_git_blob(&worktree, b"content");
        patch_index_stages(
            &worktree,
            "ws.md",
            &hash_with_newline,
            &hash_without_newline,
        );

        let branch = branch_for_worktree(&git_dir, &worktree);
        let result = branch
            .try_resolve_identical_conflicts(&["ws.md".to_string()])
            .unwrap();

        assert!(
            !result,
            "trailing-newline difference must not be treated as identical"
        );
    }

    /// Internal whitespace difference (`hello world` vs `hello  world`) is a
    /// genuine byte difference and must not be auto-resolved.
    ///
    /// Single-line, whitespace-near-identical files are susceptible to git's
    /// "already upstream" skip, so we patch the index manually after a real
    /// conflict to isolate the exact bytes we want to compare.
    #[test]
    fn test_try_resolve_identical_conflicts_internal_whitespace_difference_is_not_identical() {
        let (_dir, git_dir, worktree) = setup_different_content_conflict(
            "spaces.md",
            b"ORIGINAL\n",
            b"SENTINEL_A\n", // content that reliably triggers a real conflict
            b"SENTINEL_B\n",
        );

        // Overwrite stage 2 and 3 with the whitespace-differing strings.
        let hash_one_space = store_git_blob(&worktree, b"hello world\n");
        let hash_two_spaces = store_git_blob(&worktree, b"hello  world\n");
        patch_index_stages(&worktree, "spaces.md", &hash_one_space, &hash_two_spaces);

        let branch = branch_for_worktree(&git_dir, &worktree);
        let result = branch
            .try_resolve_identical_conflicts(&["spaces.md".to_string()])
            .unwrap();

        assert!(
            !result,
            "internal whitespace difference must not be treated as identical"
        );
    }

    // -----------------------------------------------------------------------
    // try_resolve_identical_conflicts — identical content (happy path)
    // -----------------------------------------------------------------------

    /// When both stages are byte-identical, the conflict is spurious.
    /// The function must resolve it, continue the rebase, and return `true`.
    #[test]
    fn test_try_resolve_identical_conflicts_identical_content_returns_true() {
        let (_dir, git_dir, worktree) =
            setup_identical_stages_conflict("memory.md", b"identical update\n");

        let branch = branch_for_worktree(&git_dir, &worktree);
        let result = branch
            .try_resolve_identical_conflicts(&["memory.md".to_string()])
            .unwrap();

        assert!(
            result,
            "identical stage 2/3 must be auto-resolved and rebase continued"
        );

        // Rebase completion: the rebase-merge state directory must be gone.
        // (REBASE_HEAD is a stale file that git leaves behind even after a
        // successful rebase, so we check the state dir instead.)
        let rebase_merge_dir = worktree.join(".git").join("rebase-merge");
        assert!(
            !rebase_merge_dir.exists(),
            "rebase-merge dir must be gone after successful rebase --continue"
        );
    }

    /// After resolution the working-tree file must not contain conflict markers.
    #[test]
    fn test_try_resolve_identical_conflicts_resolved_file_has_no_markers() {
        let (_dir, git_dir, worktree) =
            setup_identical_stages_conflict("clean.md", b"clean resolved content\n");

        let branch = branch_for_worktree(&git_dir, &worktree);
        branch
            .try_resolve_identical_conflicts(&["clean.md".to_string()])
            .unwrap();

        let text =
            String::from_utf8_lossy(&std::fs::read(worktree.join("clean.md")).unwrap()).to_string();
        assert!(
            !text.contains("<<<<<<<"),
            "resolved file must not contain conflict markers, got: {text:?}"
        );
        assert!(
            !text.contains(">>>>>>>"),
            "resolved file must not contain conflict markers, got: {text:?}"
        );
    }

    /// Byte-identical content that includes trailing spaces must also resolve.
    #[test]
    fn test_try_resolve_identical_conflicts_whitespace_identical_resolves() {
        let content = b"line with trailing spaces   \n";
        let (_dir, git_dir, worktree) = setup_identical_stages_conflict("spaces_same.md", content);

        let branch = branch_for_worktree(&git_dir, &worktree);
        let result = branch
            .try_resolve_identical_conflicts(&["spaces_same.md".to_string()])
            .unwrap();

        assert!(
            result,
            "byte-identical content (with whitespace) must be auto-resolved"
        );
    }
}
