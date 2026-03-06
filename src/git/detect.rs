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

fn normalize_remote(url: &str) -> String {
    let mut normalized = url.to_string();

    if normalized.starts_with("git@") {
        normalized = normalized.replacen("git@", "", 1);
        if let Some(idx) = normalized.find(':') {
            normalized.replace_range(idx..=idx, "/");
        }
    } else if normalized.starts_with("https://") {
        normalized = normalized.replacen("https://", "", 1);
    } else if normalized.starts_with("http://") {
        normalized = normalized.replacen("http://", "", 1);
    } else if normalized.starts_with("ssh://git@") {
        normalized = normalized.replacen("ssh://git@", "", 1);
    } else if normalized.starts_with("ssh://") {
        normalized = normalized.replacen("ssh://", "", 1);
        // Strip optional user@ prefix (e.g. "user@github.com/org/repo")
        if let Some(at_idx) = normalized.find('@') {
            let slash_idx = normalized.find('/').unwrap_or(normalized.len());
            if at_idx < slash_idx {
                normalized = normalized[at_idx + 1..].to_string();
            }
        }
    }

    if normalized.ends_with(".git") {
        normalized.truncate(normalized.len() - 4);
    }

    normalized
}

#[cfg(test)]
mod tests {
    use super::*;

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
