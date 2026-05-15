//! Shared test helpers for git module tests.

use std::path::PathBuf;

/// Create a temporary git repository with an initial empty commit on `main`
/// and an orphan `veclayer-memory` branch containing two committed files:
/// `test.md` and `second.md`.
///
/// The caller must hold the returned `TempDir` alive for the duration of the
/// test; dropping it removes the directory.
///
/// Using environment variables for author identity is more reliable than
/// `-c` flags because it works even when the system has no global git config.
pub(crate) fn setup_test_repo() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let git_dir = dir.path().join(".git");

    let run = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .current_dir(dir.path())
            .env("GIT_AUTHOR_NAME", "Test")
            .env("GIT_AUTHOR_EMAIL", "test@example.com")
            .env("GIT_COMMITTER_NAME", "Test")
            .env("GIT_COMMITTER_EMAIL", "test@example.com")
            .output()
            .expect("git command failed")
    };

    run(&["init", "-b", "main"]);
    // Disable commit signing so the temp repo is hermetic: tests must not depend
    // on the host's global `commit.gpgsign` / signing-key configuration.
    run(&["config", "commit.gpgsign", "false"]);
    run(&["commit", "--allow-empty", "-m", "init"]);
    run(&["checkout", "--orphan", "veclayer-memory"]);
    run(&["rm", "-rf", "."]);

    std::fs::write(dir.path().join("test.md"), b"hello world").unwrap();
    std::fs::write(dir.path().join("second.md"), b"second file").unwrap();
    run(&["add", "test.md", "second.md"]);
    run(&["commit", "-m", "add test files"]);
    run(&["checkout", "main"]);

    (dir, git_dir)
}
