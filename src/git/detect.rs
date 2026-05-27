use std::path::Path;

#[derive(Debug, Clone, Default)]
pub struct GitProject {
    pub remote: Option<String>,
    pub branch: Option<String>,
}

pub fn detect(start_dir: &Path) -> GitProject {
    let git_root = match find_git_root(start_dir) {
        Some(root) => root,
        None => return GitProject::default(),
    };

    let remote = get_remote(&git_root);
    let branch = get_branch(&git_root);

    GitProject { remote, branch }
}

/// Walk upward from `start_dir` to find the nearest `.git` directory.
/// Returns the repository root (parent of `.git`).
pub fn find_git_root(start_dir: &Path) -> Option<std::path::PathBuf> {
    let mut dir = start_dir.canonicalize().ok()?;
    loop {
        let git_path = dir.join(".git");
        if git_path.exists() {
            return Some(dir.to_path_buf());
        }
        dir = dir.parent()?.to_path_buf();
    }
}

/// Resolve the `.git` directory itself (handles both normal repos and worktrees).
pub fn find_git_dir(start_dir: &Path) -> Option<std::path::PathBuf> {
    let root = find_git_root(start_dir)?;
    let git_path = root.join(".git");
    if git_path.is_dir() {
        Some(git_path)
    } else if git_path.is_file() {
        // Worktree: .git is a file pointing to the actual git dir
        let content = std::fs::read_to_string(&git_path).ok()?;
        let gitdir = content.strip_prefix("gitdir: ")?.trim();
        let resolved = if Path::new(gitdir).is_absolute() {
            std::path::PathBuf::from(gitdir)
        } else {
            root.join(gitdir).canonicalize().ok()?
        };
        Some(resolved)
    } else {
        None
    }
}

fn get_remote(git_root: &std::path::Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .current_dir(git_root)
        .args(["remote", "get-url", "origin"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let url = String::from_utf8(output.stdout).ok()?;
    let normalized = normalize_remote(url.trim());
    if normalized.is_empty() {
        return None;
    }
    Some(normalized)
}

fn get_branch(git_root: &std::path::Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .current_dir(git_root)
        .args(["branch", "--show-current"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let branch = String::from_utf8(output.stdout).ok()?.trim().to_string();
    if branch.is_empty() {
        return None;
    }
    Some(branch)
}

/// Strip an optional `userinfo@` prefix that appears before the first `/`.
///
/// Handles plain usernames (`user@host/path`) and user:password pairs
/// (`user:token@host/path`).  An `@` that only appears after the first `/`
/// (e.g. a path segment like `host/a@b`) is left untouched.
fn strip_userinfo(s: &str) -> &str {
    let slash_pos = s.find('/').unwrap_or(s.len());
    // Only strip when @ is strictly before the first /.  Use rfind (last @),
    // not find: per RFC 3986 the userinfo ends at the LAST @ before the
    // authority, so `user:p@ss@host/x` must strip `user:p@ss@`, not just
    // `user:p@`.
    if let Some(at_pos) = s[..slash_pos].rfind('@') {
        &s[at_pos + 1..]
    } else {
        s
    }
}

fn normalize_remote(url: &str) -> String {
    let normalized: &str = if let Some(rest) = url.strip_prefix("git@") {
        // git@github.com:org/repo  →  colon replaced by slash below
        // No userinfo to strip; the "git@" is already removed.
        rest
    } else if let Some(rest) = url.strip_prefix("https://") {
        strip_userinfo(rest)
    } else if let Some(rest) = url.strip_prefix("http://") {
        strip_userinfo(rest)
    } else if let Some(rest) = url.strip_prefix("ssh://") {
        strip_userinfo(rest)
    } else {
        url
    };

    let mut result = normalized.to_string();

    // git@ SCP syntax uses a colon as host/path separator — normalise to slash.
    if url.starts_with("git@") {
        if let Some(idx) = result.find(':') {
            result.replace_range(idx..=idx, "/");
        }
    }

    if result.ends_with(".git") {
        result.truncate(result.len() - 4);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── existing green tests (must remain passing) ────────────────────────────

    #[test]
    fn test_normalize_ssh_remote() {
        assert_eq!(
            normalize_remote("git@github.com:org/repo.git"),
            "github.com/org/repo"
        );
    }

    #[test]
    fn test_normalize_https_remote() {
        assert_eq!(
            normalize_remote("https://github.com/org/repo.git"),
            "github.com/org/repo"
        );
    }

    #[test]
    fn test_normalize_no_suffix() {
        assert_eq!(
            normalize_remote("https://github.com/org/repo"),
            "github.com/org/repo"
        );
    }

    #[test]
    fn test_normalize_ssh_protocol() {
        assert_eq!(
            normalize_remote("ssh://git@github.com/org/repo.git"),
            "github.com/org/repo"
        );
    }

    #[test]
    fn test_normalize_gitlab_nested() {
        assert_eq!(
            normalize_remote("git@gitlab.com:org/sub/repo.git"),
            "gitlab.com/org/sub/repo"
        );
    }

    #[test]
    fn test_normalize_ssh_without_user() {
        assert_eq!(
            normalize_remote("ssh://github.com/org/repo.git"),
            "github.com/org/repo"
        );
    }

    #[test]
    fn test_normalize_ssh_without_user_with_user_prefix() {
        // e.g. ssh://deploy@github.com/org/repo.git — non-git@ user variant
        assert_eq!(
            normalize_remote("ssh://deploy@github.com/org/repo.git"),
            "github.com/org/repo"
        );
    }

    // ── security fix: credential stripping ───────────────────────────────────

    /// PAT embedded as password must be stripped — this is the primary security fix.
    #[test]
    fn test_normalize_https_with_user_and_token_strips_credentials() {
        assert_eq!(
            normalize_remote("https://user:token@github.com/org/repo.git"),
            "github.com/org/repo"
        );
    }

    /// Plain username (no password) in https must also be stripped.
    #[test]
    fn test_normalize_https_with_bare_username_strips_credentials() {
        assert_eq!(normalize_remote("https://user@host/x"), "host/x");
    }

    /// http:// (not https://) with embedded token must be stripped.
    #[test]
    fn test_normalize_http_with_token_strips_credentials() {
        assert_eq!(normalize_remote("http://tok@host/x.git"), "host/x");
    }

    // ── strip_userinfo edge cases ─────────────────────────────────────────────

    /// An @ that appears only inside a path segment (after the first /) must
    /// NOT be treated as a userinfo separator.
    #[test]
    fn test_normalize_at_sign_only_in_path_is_preserved() {
        // After stripping "https://" we have "host/a@b" — the @ is after the
        // first slash, so nothing should be stripped.
        assert_eq!(normalize_remote("https://host/a@b"), "host/a@b");
    }

    /// Verify strip_userinfo directly: @ only in path → untouched.
    #[test]
    fn test_strip_userinfo_at_in_path_only() {
        assert_eq!(strip_userinfo("host/a@b"), "host/a@b");
    }

    /// Verify strip_userinfo directly: no @ at all → untouched.
    #[test]
    fn test_strip_userinfo_no_at() {
        assert_eq!(strip_userinfo("github.com/org/repo"), "github.com/org/repo");
    }

    /// Verify strip_userinfo directly: user:pass@host → host/...
    #[test]
    fn test_strip_userinfo_user_pass() {
        assert_eq!(
            strip_userinfo("user:pass@github.com/org/repo"),
            "github.com/org/repo"
        );
    }

    /// Verify strip_userinfo directly: user@host/path → host/path.
    #[test]
    fn test_strip_userinfo_bare_user() {
        assert_eq!(strip_userinfo("user@host/x"), "host/x");
    }

    #[test]
    fn test_find_git_dir_worktree() {
        use std::process::Command;

        let base_dir = tempfile::tempdir().expect("tempdir");
        let base_path = base_dir.path();

        // Initialise a bare-ish repo with a commit so worktrees work.
        Command::new("git")
            .args(["init", "-b", "main"])
            .current_dir(base_path)
            .status()
            .expect("git init");
        Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(base_path)
            .status()
            .expect("git config email");
        Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(base_path)
            .status()
            .expect("git config name");
        Command::new("git")
            .args(["commit", "--allow-empty", "-m", "init"])
            .current_dir(base_path)
            .status()
            .expect("git commit");

        // Create a linked worktree.
        let worktree_path = base_path.join("wt");
        Command::new("git")
            .args(["worktree", "add", "wt", "-b", "wt-branch"])
            .current_dir(base_path)
            .status()
            .expect("git worktree add");

        let git_dir =
            find_git_dir(&worktree_path).expect("find_git_dir should resolve worktree .git file");

        // The resolved git dir must sit inside .git/worktrees/<name>.
        let expected_suffix = std::path::Path::new(".git").join("worktrees").join("wt");
        assert!(
            git_dir.ends_with(&expected_suffix),
            "expected git_dir to end with {expected_suffix:?}, got {git_dir:?}"
        );
        assert!(
            git_dir.is_dir(),
            "resolved git dir must exist as a directory"
        );
    }

    #[test]
    fn test_detect_in_this_repo() {
        let project = detect(std::path::Path::new("."));
        assert!(project.remote.is_some() || project.branch.is_some());
    }
}
