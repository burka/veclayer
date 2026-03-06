//! Git-based memory branch operations.
//!
//! Provides workspace-safe access to a dedicated memory branch (e.g., `veclayer-memory`)
//! within a user's project repository. All operations use git plumbing for reads and a
//! separate worktree for writes — the user's working directory is never touched.

use std::fmt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

pub mod branch_config;
pub mod detect;
pub mod embedding_cache;
pub mod markdown;
pub mod memory_store;
pub mod plumbing;
pub mod sync;
pub mod worktree;

/// Default branch name for the memory branch.
pub const DEFAULT_BRANCH: &str = "veclayer-memory";

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur during git operations.
#[derive(Debug)]
pub enum GitError {
    /// Git is not installed or not in PATH.
    NotInstalled,
    /// The path is not inside a git repository.
    NotARepo,
    /// The specified branch does not exist.
    BranchNotFound(String),
    /// Authentication failed (SSH key, credential helper, etc.).
    AuthFailed(String),
    /// Rebase produced merge conflicts.
    ConflictOnRebase(Vec<String>),
    /// Remote has diverged and fast-forward is not possible.
    RemoteDiverged,
    /// Worktree operation failed.
    WorktreeError(String),
    /// A git command failed with a non-zero exit code.
    CommandFailed {
        command: String,
        stderr: String,
        exit_code: i32,
    },
    /// An I/O error occurred while spawning or communicating with a git subprocess.
    Io(std::io::Error),
}

impl fmt::Display for GitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotInstalled => write!(f, "git is not installed or not in PATH"),
            Self::NotARepo => write!(f, "not inside a git repository"),
            Self::BranchNotFound(b) => write!(f, "branch '{b}' not found"),
            Self::AuthFailed(msg) => write!(f, "git authentication failed: {msg}"),
            Self::ConflictOnRebase(files) => {
                write!(f, "rebase conflicts in: {}", files.join(", "))
            }
            Self::RemoteDiverged => write!(f, "remote has diverged; pull required before push"),
            Self::WorktreeError(msg) => write!(f, "worktree error: {msg}"),
            Self::CommandFailed {
                command,
                stderr,
                exit_code,
            } => write!(f, "git {command} failed (exit {exit_code}): {stderr}"),
            Self::Io(e) => write!(f, "I/O error during git subprocess: {e}"),
        }
    }
}

impl std::error::Error for GitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Result enums
// ---------------------------------------------------------------------------

/// Outcome of a `git push`.
#[derive(Debug, PartialEq, Eq)]
pub enum PushResult {
    Success,
    NothingToPush,
    /// Remote rejected the push (typically means remote diverged).
    Rejected,
}

/// Outcome of a `git pull --rebase`.
#[derive(Debug)]
pub enum SyncResult {
    Success,
    NothingToSync,
    Conflicts(Vec<String>),
}

// ---------------------------------------------------------------------------
// Core struct
// ---------------------------------------------------------------------------

/// Operates on a memory branch without touching the user's workspace.
///
/// Reads use git plumbing (`git show`, `git ls-tree`) directly against the object
/// store — no checkout required. Writes go through a dedicated worktree under
/// `~/.cache/veclayer/`, created lazily on first write.
pub struct GitMemoryBranch {
    /// Path to the project's `.git/` directory.
    pub(crate) git_dir: PathBuf,
    /// Branch name (e.g., `veclayer-memory`).
    pub(crate) branch: String,
    /// Path where the worktree lives (e.g., `~/.cache/veclayer/<hash>/`).
    pub(crate) worktree_path: PathBuf,
}

impl GitMemoryBranch {
    /// Open a memory branch for the given repository.
    ///
    /// `project_git_dir` should be the `.git/` directory of the user's project.
    /// If `branch` is `None`, defaults to [`DEFAULT_BRANCH`].
    pub fn open(project_git_dir: &Path, branch: Option<&str>) -> Result<Self, GitError> {
        let git_dir = project_git_dir.to_path_buf();
        if !git_dir.exists() {
            return Err(GitError::NotARepo);
        }

        let branch = branch.unwrap_or(DEFAULT_BRANCH).to_string();
        let worktree_path = worktree::worktree_path(&git_dir, &branch);

        Ok(Self {
            git_dir,
            branch,
            worktree_path,
        })
    }

    /// Check whether the memory branch exists in the repository.
    pub fn branch_exists(&self) -> Result<bool, GitError> {
        let output = run_git_with_gitdir(&self.git_dir, &["rev-parse", "--verify", &self.branch])?;
        Ok(output.status.success())
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Apply standard environment variables to a git command:
/// - `GIT_TERMINAL_PROMPT=0` / `GIT_ASKPASS=echo` — prevent credential prompts
///   that would hang the process in non-interactive contexts.
/// - `LC_ALL=C` — ensure locale-independent output for string-based parsing.
pub(crate) fn apply_git_env(cmd: &mut Command) -> &mut Command {
    cmd.env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_ASKPASS", "echo")
        .env("LC_ALL", "C")
}

/// Run a git command using `--git-dir` to target a specific repository
/// without requiring a working directory inside it.
pub(crate) fn run_git_with_gitdir(git_dir: &Path, args: &[&str]) -> Result<Output, GitError> {
    let mut cmd = Command::new("git");
    cmd.arg("--git-dir").arg(git_dir).args(args);
    apply_git_env(&mut cmd).output().map_err(map_io_error)
}

/// Run a git command with the working directory set to `cwd`.
pub(crate) fn run_git_in(cwd: &Path, args: &[&str]) -> Result<Output, GitError> {
    let mut cmd = Command::new("git");
    cmd.current_dir(cwd).args(args);
    apply_git_env(&mut cmd).output().map_err(map_io_error)
}

/// Check whether a git error's stderr indicates a file-not-found condition.
///
/// These strings are produced by git under `LC_ALL=C` (the locale we always set).
/// Each pattern corresponds to a distinct error path in git:
///   - "does not exist in" — `git show` when the path is absent on the ref
///   - "exists on disk, but not in" — `git diff` / `git show` path mismatch
///   - "Path '" — `git ls-files` / `git cat-file` path not found (prefix form)
///   - "unknown revision or path not in the working tree" — generic ref/path failure
pub(crate) fn is_file_not_found(stderr: &str) -> bool {
    stderr.contains("does not exist in")
        || stderr.contains("exists on disk, but not in")
        || stderr.contains("Path '")
        || stderr.contains("unknown revision or path not in the working tree")
}

/// Map a subprocess spawn/IO error to the appropriate [`GitError`].
pub(crate) fn map_io_error(e: std::io::Error) -> GitError {
    if e.kind() == std::io::ErrorKind::NotFound {
        GitError::NotInstalled
    } else {
        GitError::Io(e)
    }
}

/// Check that a git command succeeded, returning a [`GitError::CommandFailed`] if not.
pub(crate) fn check_output(output: &Output, command: &str) -> Result<(), GitError> {
    if output.status.success() {
        Ok(())
    } else {
        Err(GitError::CommandFailed {
            command: command.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            exit_code: output.status.code().unwrap_or(-1),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // is_file_not_found — pattern documentation and regression tests
    // -----------------------------------------------------------------------

    /// git show <branch>:<path> when `path` is absent on the branch
    /// e.g. "fatal: Path 'missing.md' does not exist in 'main'"
    #[test]
    fn test_is_file_not_found_does_not_exist_in() {
        assert!(is_file_not_found(
            "fatal: Path 'missing.md' does not exist in 'main'"
        ));
    }

    /// git show / git diff when the path is present on disk but not in the index
    /// e.g. "error: 'foo.txt' exists on disk, but not in 'HEAD'"
    #[test]
    fn test_is_file_not_found_exists_on_disk_but_not_in() {
        assert!(is_file_not_found(
            "error: 'foo.txt' exists on disk, but not in 'HEAD'"
        ));
    }

    /// git ls-files and similar plumbing when using the "Path '<name>'" prefix form
    /// e.g. "error: Path 'bar.txt' is in index, stage 2"  — also matches generic
    /// path-related errors like "Path 'x' not found"
    #[test]
    fn test_is_file_not_found_path_prefix() {
        assert!(is_file_not_found("error: Path 'bar.txt' not found"));
    }

    /// Generic ref-or-path failure produced by `git rev-parse` and `git show`
    /// e.g. "fatal: ambiguous argument 'HEAD:x': unknown revision or path not in the working tree"
    #[test]
    fn test_is_file_not_found_unknown_revision() {
        assert!(is_file_not_found(
            "fatal: ambiguous argument 'HEAD:x': unknown revision or path not in the working tree"
        ));
    }

    /// Unrelated stderr output must not match.
    #[test]
    fn test_is_file_not_found_rejects_unrelated() {
        assert!(!is_file_not_found("fatal: repository not found"));
        assert!(!is_file_not_found("error: Authentication failed"));
        assert!(!is_file_not_found(""));
    }

    // -----------------------------------------------------------------------
    // map_io_error — Io variant carries source
    // -----------------------------------------------------------------------

    #[test]
    fn test_map_io_error_not_found_becomes_not_installed() {
        let e = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
        assert!(matches!(map_io_error(e), GitError::NotInstalled));
    }

    #[test]
    fn test_map_io_error_other_becomes_io_variant() {
        let e = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        assert!(matches!(map_io_error(e), GitError::Io(_)));
    }

    #[test]
    fn test_git_error_io_source_is_chained() {
        use std::error::Error as _;
        let inner = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe broke");
        let err = GitError::Io(inner);
        assert!(err.source().is_some());
    }

    #[test]
    fn test_git_error_other_variants_have_no_source() {
        use std::error::Error as _;
        assert!(GitError::NotInstalled.source().is_none());
        assert!(GitError::NotARepo.source().is_none());
        assert!(GitError::RemoteDiverged.source().is_none());
    }
}
