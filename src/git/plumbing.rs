//! Git plumbing read operations for the memory branch.
//!
//! All methods operate directly on the git object store via `--git-dir` — no
//! worktree checkout is required for reads.

use super::{check_output, map_io_error, run_git_with_gitdir, GitError, GitMemoryBranch};

impl GitMemoryBranch {
    /// Read the raw bytes of a file on the memory branch.
    ///
    /// Uses `git show <branch>:<path>` — works for both text and binary files.
    ///
    /// # Errors
    ///
    /// Returns [`GitError::BranchNotFound`] if the branch does not exist.
    /// Returns [`GitError::CommandFailed`] if `path` does not exist on the branch.
    pub fn read_file(&self, path: &str) -> Result<Vec<u8>, GitError> {
        self.assert_branch_exists()?;

        let object_ref = format!("{}:{}", self.branch, path);
        let output = run_git_with_gitdir(&self.git_dir, &["show", &object_ref])?;

        check_output(&output, &format!("show {}:{}", self.branch, path))?;
        Ok(output.stdout)
    }

    /// List all file paths tracked on the memory branch.
    ///
    /// Uses `git ls-tree -r --name-only <branch>`.
    ///
    /// # Errors
    ///
    /// Returns [`GitError::BranchNotFound`] if the branch does not exist.
    pub fn list_files(&self) -> Result<Vec<String>, GitError> {
        self.assert_branch_exists()?;

        let output = run_git_with_gitdir(
            &self.git_dir,
            &["ls-tree", "-r", "--name-only", &self.branch],
        )?;

        check_output(&output, &format!("ls-tree {}", self.branch))?;

        let paths = String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter(|l| !l.is_empty())
            .map(str::to_owned)
            .collect();

        Ok(paths)
    }

    /// Read multiple files from the memory branch in a single `git cat-file --batch` call.
    ///
    /// Returns a vector of `(path, content)` pairs for files that exist on the branch.
    /// Missing objects are silently skipped.
    pub fn read_files_batch(&self, paths: &[&str]) -> Result<Vec<(String, Vec<u8>)>, GitError> {
        if paths.is_empty() {
            return Ok(Vec::new());
        }

        self.assert_branch_exists()?;

        let stdin_input: String = paths
            .iter()
            .map(|p| format!("{}:{}\n", self.branch, p))
            .collect();

        let output = run_git_with_gitdir_stdin(
            &self.git_dir,
            &["cat-file", "--batch"],
            stdin_input.as_bytes(),
        )?;

        // cat-file --batch always exits 0; errors appear inline as "missing" lines.
        parse_cat_file_batch_output(&output.stdout, paths)
    }

    /// Return [`GitError::BranchNotFound`] if the memory branch does not exist.
    fn assert_branch_exists(&self) -> Result<(), GitError> {
        if !self.branch_exists()? {
            return Err(GitError::BranchNotFound(self.branch.clone()));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// cat-file --batch parser
// ---------------------------------------------------------------------------

/// Parse the raw stdout of `git cat-file --batch`.
///
/// Each object header has the form `<sha1> <type> <size>\n` followed by exactly
/// `<size>` bytes of content and a trailing `\n`. Missing objects produce a line
/// of the form `<ref> missing\n`.
fn parse_cat_file_batch_output(
    raw: &[u8],
    paths: &[&str],
) -> Result<Vec<(String, Vec<u8>)>, GitError> {
    let mut results = Vec::new();
    let mut cursor = 0usize;
    let mut path_index = 0usize;

    while cursor < raw.len() && path_index < paths.len() {
        let path = paths[path_index];
        path_index += 1;

        let Some(header_end) = find_newline(raw, cursor) else {
            break;
        };

        let header = std::str::from_utf8(&raw[cursor..header_end])
            .unwrap_or("")
            .trim();
        cursor = header_end + 1; // advance past '\n'

        if header.ends_with(" missing") {
            // Object not present on the branch — skip silently.
            continue;
        }

        // Header format: "<sha1> <type> <size>"
        let size = match parse_object_size(header) {
            Some(s) => s,
            None => continue,
        };

        if cursor + size > raw.len() {
            break;
        }

        let content = raw[cursor..cursor + size].to_vec();
        cursor += size;

        // Skip the trailing newline that git appends after each object body.
        if cursor < raw.len() && raw[cursor] == b'\n' {
            cursor += 1;
        }

        results.push((path.to_owned(), content));
    }

    Ok(results)
}

/// Find the byte offset of the next `\n` at or after `from`, or `None`.
fn find_newline(buf: &[u8], from: usize) -> Option<usize> {
    buf[from..]
        .iter()
        .position(|&b| b == b'\n')
        .map(|p| from + p)
}

/// Parse the object size from a `git cat-file --batch` header line.
///
/// Expected format: `<sha1> <type> <size>`
fn parse_object_size(header: &str) -> Option<usize> {
    header.rsplit_once(' ')?.1.parse::<usize>().ok()
}

// ---------------------------------------------------------------------------
// Stdin-capable git runner
// ---------------------------------------------------------------------------

/// Run a git command with `--git-dir` and data written to its stdin.
fn run_git_with_gitdir_stdin(
    git_dir: &std::path::Path,
    args: &[&str],
    stdin_data: &[u8],
) -> Result<std::process::Output, GitError> {
    use std::io::Write as _;
    use std::process::{Command, Stdio};

    let git_dir_str = git_dir.to_string_lossy();
    let mut full_args = vec!["--git-dir", &git_dir_str];
    full_args.extend_from_slice(args);

    let mut child = Command::new("git")
        .args(&full_args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| map_io_error(e, &full_args))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(stdin_data)
            .map_err(|e| map_io_error(e, &full_args))?;
    }

    child
        .wait_with_output()
        .map_err(|e| map_io_error(e, &full_args))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::git::GitMemoryBranch;
    use std::path::PathBuf;

    /// Create a temporary git repository with an orphan `veclayer-memory` branch
    /// containing two committed files: `test.md` and `second.md`.
    ///
    /// The caller must hold the returned `TempDir` alive for the duration of the
    /// test; dropping it removes the directory.
    fn setup_test_repo() -> (tempfile::TempDir, PathBuf) {
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

        run(&["init"]);
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

    #[test]
    fn test_read_file_from_branch() {
        let (_dir, git_dir) = setup_test_repo();
        let branch = GitMemoryBranch::open(&git_dir, Some("veclayer-memory")).unwrap();

        let content = branch.read_file("test.md").unwrap();
        assert_eq!(content, b"hello world");
    }

    #[test]
    fn test_list_files() {
        let (_dir, git_dir) = setup_test_repo();
        let branch = GitMemoryBranch::open(&git_dir, Some("veclayer-memory")).unwrap();

        let mut files = branch.list_files().unwrap();
        files.sort();

        assert_eq!(files, vec!["second.md", "test.md"]);
    }

    #[test]
    fn test_read_files_batch() {
        let (_dir, git_dir) = setup_test_repo();
        let branch = GitMemoryBranch::open(&git_dir, Some("veclayer-memory")).unwrap();

        let paths = ["test.md", "second.md"];
        let mut results = branch.read_files_batch(&paths).unwrap();
        results.sort_by(|a, b| a.0.cmp(&b.0));

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "second.md");
        assert_eq!(results[0].1, b"second file");
        assert_eq!(results[1].0, "test.md");
        assert_eq!(results[1].1, b"hello world");
    }

    #[test]
    fn test_read_files_batch_skips_missing() {
        let (_dir, git_dir) = setup_test_repo();
        let branch = GitMemoryBranch::open(&git_dir, Some("veclayer-memory")).unwrap();

        let paths = ["test.md", "does-not-exist.md", "second.md"];
        let results = branch.read_files_batch(&paths).unwrap();

        let found_paths: Vec<&str> = results.iter().map(|(p, _)| p.as_str()).collect();
        assert!(found_paths.contains(&"test.md"));
        assert!(found_paths.contains(&"second.md"));
        assert!(!found_paths.contains(&"does-not-exist.md"));
    }

    #[test]
    fn test_read_file_nonexistent_branch() {
        let (_dir, git_dir) = setup_test_repo();
        let branch = GitMemoryBranch::open(&git_dir, Some("nonexistent-branch")).unwrap();

        let result = branch.read_file("test.md");
        assert!(
            matches!(
                result,
                Err(GitError::BranchNotFound(_)) | Err(GitError::CommandFailed { .. })
            ),
            "expected BranchNotFound or CommandFailed, got: {:?}",
            result
        );
    }

    #[test]
    fn test_read_file_nonexistent_path() {
        let (_dir, git_dir) = setup_test_repo();
        let branch = GitMemoryBranch::open(&git_dir, Some("veclayer-memory")).unwrap();

        let result = branch.read_file("no-such-file.md");
        assert!(
            matches!(result, Err(GitError::CommandFailed { .. })),
            "expected CommandFailed, got: {:?}",
            result
        );
    }

    #[test]
    fn test_read_files_batch_empty_input() {
        let (_dir, git_dir) = setup_test_repo();
        let branch = GitMemoryBranch::open(&git_dir, Some("veclayer-memory")).unwrap();

        let results = branch.read_files_batch(&[]).unwrap();
        assert!(results.is_empty());
    }
}
