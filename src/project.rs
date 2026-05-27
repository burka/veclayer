//! Multi-method project detection with git-remote-first priority.
//!
//! Detection methods (in priority order):
//! 1. **ENV var** (`VECLAYER_PROJECT`) — explicit override, highest priority
//! 2. **Git remote** — derived from `git remote get-url origin`, becomes project name
//! 3. **OpenClaw agent ID** — from `VECLAYER_AGENT_ID` env var (OpenClaw context)
//! 4. **CWD pattern** — walk-up from CWD looking for `.veclayer/` directory
//! 5. **Config file** — `[[match]]` rules in user config
//!
//! Project stores are auto-created at `~/.veclayer/projects/<remote>/` when
//! a git remote is detected but no local `.veclayer/` exists.

use std::path::{Path, PathBuf};

#[cfg(feature = "config")]
use crate::config::UserConfig;

/// Source of project detection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectSource {
    /// Detected from `VECLAYER_PROJECT` environment variable.
    EnvVar,
    /// Derived from git remote URL (e.g. `github.com/org/repo`).
    GitRemote,
    /// OpenClaw agent context (`VECLAYER_AGENT_ID`).
    OpenClawAgent,
    /// Found via walk-up from CWD looking for `.veclayer/`.
    LocalConfig,
    /// Matched via `[[match]]` rules in user config.
    ConfigMatch,
    /// No project detected.
    None,
}

/// The subset of [`ProjectSource`] values whose data directory is resolved by
/// [`resolve_project_data_dir`].  `LocalConfig` (path comes from the walk-up
/// result) and `None` are intentionally excluded — they can never reach the
/// resolver, and this type makes that invariant compile-time rather than
/// runtime.
#[derive(Debug, Clone, Copy)]
enum ResolvableSource {
    EnvVar,
    GitRemote,
    OpenClawAgent,
    ConfigMatch,
}

/// Result of project detection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectDetection {
    /// The detected project name (None if no project found).
    pub project: Option<String>,
    /// Which source provided the detection.
    pub source: ProjectSource,
    /// The data directory for this project (if any).
    pub data_dir: Option<PathBuf>,
}

/// Detect project using multiple methods in priority order.
///
/// Returns the first non-None detection with its source and resolved data directory.
pub fn detect_project(start_dir: &Path) -> ProjectDetection {
    // 1. ENV var override — highest priority
    if let Some(project) = std::env::var_os("VECLAYER_PROJECT") {
        if !project.is_empty() {
            let project_str = project.to_string_lossy();
            let data_dir = resolve_project_data_dir(&project_str, ResolvableSource::EnvVar);
            return ProjectDetection {
                project: Some(project_str.into_owned()),
                source: ProjectSource::EnvVar,
                data_dir,
            };
        }
    }

    // 2. Git remote — primary identifier for project-first architecture
    let git_info = crate::git::detect::detect(start_dir);
    if let Some(remote) = &git_info.remote {
        let data_dir = resolve_project_data_dir(remote, ResolvableSource::GitRemote);
        return ProjectDetection {
            project: Some(remote.clone()),
            source: ProjectSource::GitRemote,
            data_dir,
        };
    }

    // 3. OpenClaw agent ID — only if VECLAYER_AGENT_ID is set
    if let Some(agent_id) = std::env::var_os("VECLAYER_AGENT_ID") {
        if !agent_id.is_empty() {
            let agent_str = agent_id.to_string_lossy();
            // Agent IDs from OpenClaw look like "agent_xxx" — use as project name
            if agent_str.starts_with("agent_") {
                let data_dir =
                    resolve_project_data_dir(&agent_str, ResolvableSource::OpenClawAgent);
                return ProjectDetection {
                    project: Some(agent_str.into_owned()),
                    source: ProjectSource::OpenClawAgent,
                    data_dir,
                };
            }
        }
    }

    // 4. CWD pattern — walk-up for .veclayer/ directory
    if let Some((data_dir, project_config)) = discover_local_project(start_dir) {
        if let Some(project) = project_config.project {
            return ProjectDetection {
                project: Some(project),
                source: ProjectSource::LocalConfig,
                data_dir: Some(data_dir),
            };
        }
    }

    // 5. Config file match rules
    #[cfg(feature = "config")]
    {
        let user_config = UserConfig::discover();
        if let Some(project) = user_config.resolve(start_dir, git_info.remote.as_deref()).project {
            let data_dir = resolve_project_data_dir(&project, ResolvableSource::ConfigMatch);
            return ProjectDetection {
                project: Some(project),
                source: ProjectSource::ConfigMatch,
                data_dir,
            };
        }
    }

    ProjectDetection {
        project: None,
        source: ProjectSource::None,
        data_dir: None,
    }
}

/// Walk up from `start_dir` looking for a `.veclayer/` directory.
fn discover_local_project(start_dir: &Path) -> Option<(PathBuf, crate::config::ProjectConfig)> {
    #[cfg(feature = "config")]
    {
        return crate::config::discover_project(start_dir);
    }
    #[cfg(not(feature = "config"))]
    {
        let _ = start_dir;
        None
    }
}

/// Resolve the data directory for a project.
///
/// For git-remote, config-match, and openclaw-agent sources, auto-creates
/// `~/.veclayer/projects/<project>/`.  For env-var, uses `VECLAYER_DATA_DIR`
/// if set, otherwise `~/.veclayer/projects/<project>/`.
///
/// `LocalConfig` and `None` are intentionally absent from [`ResolvableSource`]
/// and therefore cannot be passed here.  `LocalConfig`'s data directory comes
/// directly from the walk-up result in `detect_project`; `None` means no
/// project was detected at all.  The type enforces this at compile time.
fn resolve_project_data_dir(project: &str, source: ResolvableSource) -> Option<PathBuf> {
    let base_dirs = directories::BaseDirs::new()?;
    let veclayer_home = base_dirs.home_dir().join(".veclayer");

    let data_dir = match source {
        ResolvableSource::GitRemote
        | ResolvableSource::ConfigMatch
        | ResolvableSource::OpenClawAgent => {
            // Auto-create per-remote store: ~/.veclayer/projects/<remote>/
            let projects_dir = veclayer_home.join("projects").join(project);
            if !projects_dir.exists() {
                std::fs::create_dir_all(&projects_dir).ok()?;
            }
            projects_dir
        }
        ResolvableSource::EnvVar => {
            // ENV var — use explicit path if set, otherwise per-project store
            if let Some(explicit) = std::env::var_os("VECLAYER_DATA_DIR") {
                PathBuf::from(explicit)
            } else {
                let projects_dir = veclayer_home.join("projects").join(project);
                if !projects_dir.exists() {
                    std::fs::create_dir_all(&projects_dir).ok()?;
                }
                projects_dir
            }
        }
    };

    Some(data_dir)
}

/// Returns the global memory directory `~/.veclayer/global/`.
///
/// This is only used when explicitly requested via `global: true` on MCP tools.
pub fn global_data_dir() -> Option<PathBuf> {
    let base_dirs = directories::BaseDirs::new()?;
    let global_dir = base_dirs.home_dir().join(".veclayer").join("global");
    Some(global_dir)
}

/// Check if a project name refers to the global store.
pub fn is_global_project(project: Option<&str>) -> bool {
    project == Some("global")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Clear every env variable that influences project detection so tests
    /// don't leak state into one another.
    fn clear_project_env() {
        std::env::remove_var("VECLAYER_PROJECT");
        std::env::remove_var("VECLAYER_AGENT_ID");
        std::env::remove_var("VECLAYER_DATA_DIR");
    }

    /// Create a temp directory that contains a `.veclayer/` sub-directory and
    /// optionally a `config.toml` inside it.  Returns the `TempDir` guard
    /// (caller must keep it alive) and the path to the temp dir.
    #[cfg(feature = "config")]
    fn make_local_veclayer_dir(
        config_toml: Option<&str>,
    ) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::TempDir::new().unwrap();
        let veclayer_dir = dir.path().join(".veclayer");
        std::fs::create_dir_all(&veclayer_dir).unwrap();
        if let Some(toml) = config_toml {
            std::fs::write(veclayer_dir.join("config.toml"), toml).unwrap();
        }
        let root = dir.path().to_path_buf();
        (dir, root)
    }

    // ── existing tests (must remain passing) ─────────────────────────────────

    #[test]
    fn test_no_project_when_not_in_git_repo() {
        clear_project_env();
        // Use /var/tmp which is unlikely to match any config rules
        let result = detect_project(Path::new("/var/tmp"));
        assert_eq!(result.project, None);
        assert_eq!(result.source, ProjectSource::None);
    }

    #[test]
    fn test_env_var_overrides_git_remote() {
        clear_project_env();
        std::env::set_var("VECLAYER_PROJECT", "env-project");
        let result = detect_project(Path::new("/tmp"));
        std::env::remove_var("VECLAYER_PROJECT");

        assert_eq!(result.project.as_deref(), Some("env-project"));
        assert_eq!(result.source, ProjectSource::EnvVar);
    }

    #[test]
    fn test_is_global_project() {
        assert!(is_global_project(Some("global")));
        assert!(!is_global_project(Some("my-project")));
        assert!(!is_global_project(None));
    }

    // ── local-dir resolution ──────────────────────────────────────────────────

    /// When a `.veclayer/config.toml` with an explicit `project` field exists
    /// in the start_dir, detect_project must return LocalConfig with that name
    /// and the data_dir pointing at the `.veclayer/` directory itself.
    #[cfg(feature = "config")]
    #[test]
    fn test_local_dir_resolution_explicit_project_name() {
        clear_project_env();
        let (_guard, root) = make_local_veclayer_dir(Some(r#"project = "myapp""#));

        let result = detect_project(&root);

        assert_eq!(result.project.as_deref(), Some("myapp"));
        assert_eq!(result.source, ProjectSource::LocalConfig);
        // data_dir must be the .veclayer/ sub-directory that was found
        assert_eq!(result.data_dir, Some(root.join(".veclayer")));
    }

    /// When a `.veclayer/` directory exists but has no `config.toml`, the
    /// project name falls back to None (no git remote in the temp dir) and
    /// detection continues to the next method — ending in ProjectSource::None
    /// since there is no git remote and no other signals.
    #[cfg(feature = "config")]
    #[test]
    fn test_local_dir_no_config_toml_no_git_remote_falls_through() {
        clear_project_env();
        let (_guard, root) = make_local_veclayer_dir(None);

        let result = detect_project(&root);

        // No project name can be inferred without a git remote or explicit config.
        assert_eq!(result.project, None);
        assert_eq!(result.source, ProjectSource::None);
    }

    /// Walk-up: start from a sub-directory; the `.veclayer/` directory is in
    /// the parent.  Detection must still find it and report LocalConfig.
    #[cfg(feature = "config")]
    #[test]
    fn test_local_dir_resolution_walk_up_from_subdirectory() {
        clear_project_env();
        let (_guard, root) = make_local_veclayer_dir(Some(r#"project = "walkup""#));

        // Create a deeply nested sub-directory — no .veclayer/ here.
        let sub = root.join("a").join("b").join("c");
        std::fs::create_dir_all(&sub).unwrap();

        let result = detect_project(&sub);

        assert_eq!(result.project.as_deref(), Some("walkup"));
        assert_eq!(result.source, ProjectSource::LocalConfig);
        assert_eq!(result.data_dir, Some(root.join(".veclayer")));
    }

    // ── fallback precedence ───────────────────────────────────────────────────

    /// ENV var (priority 1) must win over a local `.veclayer/config.toml`
    /// (priority 4).
    #[cfg(feature = "config")]
    #[test]
    fn test_env_var_wins_over_local_config() {
        clear_project_env();
        let (_guard, root) = make_local_veclayer_dir(Some(r#"project = "local-project""#));

        std::env::set_var("VECLAYER_PROJECT", "env-wins");
        let result = detect_project(&root);
        std::env::remove_var("VECLAYER_PROJECT");

        assert_eq!(result.project.as_deref(), Some("env-wins"));
        assert_eq!(result.source, ProjectSource::EnvVar);
    }

    /// OpenClaw agent ID (priority 3) must win over a local `.veclayer/`
    /// directory (priority 4) when no git remote is present.
    #[cfg(feature = "config")]
    #[test]
    fn test_openclaw_agent_wins_over_local_config() {
        clear_project_env();
        let (_guard, root) = make_local_veclayer_dir(Some(r#"project = "local-project""#));

        std::env::set_var("VECLAYER_AGENT_ID", "agent_abc123");
        let result = detect_project(&root);
        std::env::remove_var("VECLAYER_AGENT_ID");

        assert_eq!(result.project.as_deref(), Some("agent_abc123"));
        assert_eq!(result.source, ProjectSource::OpenClawAgent);
    }

    // ── error / missing cases ─────────────────────────────────────────────────

    /// An empty `VECLAYER_PROJECT` env var must be treated as unset, so
    /// detection falls through to the next method.
    #[test]
    fn test_empty_env_var_is_ignored() {
        clear_project_env();
        std::env::set_var("VECLAYER_PROJECT", "");

        let result = detect_project(Path::new("/var/tmp"));
        std::env::remove_var("VECLAYER_PROJECT");

        // The empty var must not produce EnvVar source.
        assert_ne!(result.source, ProjectSource::EnvVar);
    }

    /// An `VECLAYER_AGENT_ID` that does not begin with `agent_` must be
    /// ignored — only the canonical OpenClaw format is accepted.
    #[test]
    fn test_openclaw_agent_id_without_prefix_is_ignored() {
        clear_project_env();

        std::env::set_var("VECLAYER_AGENT_ID", "session_xyz");
        let result = detect_project(Path::new("/var/tmp"));
        std::env::remove_var("VECLAYER_AGENT_ID");

        assert_ne!(result.source, ProjectSource::OpenClawAgent);
    }

    /// An empty `VECLAYER_AGENT_ID` env var must be treated as unset.
    #[test]
    fn test_empty_openclaw_agent_id_is_ignored() {
        clear_project_env();

        std::env::set_var("VECLAYER_AGENT_ID", "");
        let result = detect_project(Path::new("/var/tmp"));
        std::env::remove_var("VECLAYER_AGENT_ID");

        assert_ne!(result.source, ProjectSource::OpenClawAgent);
    }

    // ── resolve_project_data_dir ──────────────────────────────────────────────

    // NOTE: there is no test for "None source returns None" because
    // ResolvableSource cannot represent None — the type system is the proof.

    /// When VECLAYER_DATA_DIR is set, EnvVar source must return that path
    /// exactly, without creating any directories.
    #[test]
    fn test_resolve_data_dir_env_var_respects_explicit_data_dir() {
        clear_project_env();
        let tmp = tempfile::TempDir::new().unwrap();
        let explicit = tmp.path().join("explicit");
        std::fs::create_dir_all(&explicit).unwrap();

        std::env::set_var("VECLAYER_DATA_DIR", &explicit);
        let result = resolve_project_data_dir("ignored", ResolvableSource::EnvVar);
        std::env::remove_var("VECLAYER_DATA_DIR");

        assert_eq!(result, Some(explicit));
    }
}
