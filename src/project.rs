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
            let data_dir = resolve_project_data_dir(&project_str, ProjectSource::EnvVar);
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
        let data_dir = resolve_project_data_dir(remote, ProjectSource::GitRemote);
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
                    resolve_project_data_dir(&agent_str, ProjectSource::OpenClawAgent);
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
            let data_dir = resolve_project_data_dir(&project, ProjectSource::ConfigMatch);
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
/// For git-remote and config-match sources, auto-creates `~/.veclayer/projects/<project>/`.
/// For local-config, returns the `.veclayer/` directory.
/// For env-var, uses `~/.veclayer/projects/<project>/`.
fn resolve_project_data_dir(project: &str, source: ProjectSource) -> Option<PathBuf> {
    let base_dirs = directories::BaseDirs::new()?;
    let veclayer_home = base_dirs.home_dir().join(".veclayer");

    let data_dir = match source {
        ProjectSource::LocalConfig => {
            // Local .veclayer/ — don't auto-create, just return it
            return None;
        }
        ProjectSource::GitRemote | ProjectSource::ConfigMatch | ProjectSource::OpenClawAgent => {
            // Auto-create per-remote store: ~/.veclayer/projects/<remote>/
            let projects_dir = veclayer_home.join("projects").join(project);
            if !projects_dir.exists() {
                std::fs::create_dir_all(&projects_dir).ok()?;
            }
            projects_dir
        }
        ProjectSource::EnvVar => {
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
        ProjectSource::None => return None,
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

    #[test]
    fn test_no_project_when_not_in_git_repo() {
        // Ensure VECLAYER_PROJECT is not set to avoid interference from other tests
        std::env::remove_var("VECLAYER_PROJECT");
        // Use /var/tmp which is unlikely to match any config rules
        let result = detect_project(Path::new("/var/tmp"));
        assert_eq!(result.project, None);
        assert_eq!(result.source, ProjectSource::None);
    }

    #[test]
    fn test_env_var_overrides_git_remote() {
        // Set env var temporarily
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
}
