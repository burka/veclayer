//! Project and user-config discovery/resolution logic.

#[cfg(feature = "config")]
use std::path::{Path, PathBuf};

#[cfg(feature = "config")]
use tracing::warn;

#[cfg(feature = "config")]
use std::collections::HashMap;

#[cfg(feature = "config")]
use serde::Deserialize;

#[cfg(feature = "config")]
use super::types::{MatchOverride, ProjectConfig, ResolvedConfig, ResolvedScope, ScopeConfig};

/// User-level configuration with global defaults and match-based overrides.
#[cfg(feature = "config")]
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct UserConfig {
    pub data_dir: Option<String>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub read_only: Option<bool>,
    pub project: Option<String>,
    #[serde(rename = "match")]
    pub matches: Vec<MatchOverride>,
    /// Named scope definitions keyed by scope name.
    pub scopes: HashMap<String, ScopeConfig>,
}

#[cfg(feature = "config")]
impl UserConfig {
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(contents) => match toml::from_str::<Self>(&contents) {
                Ok(mut config) => {
                    config.expand_paths();
                    config
                }
                Err(e) => {
                    eprintln!(
                        "veclayer: Malformed user config {}: {} — using defaults",
                        path.display(),
                        e
                    );
                    Self::default()
                }
            },
            Err(e) => {
                eprintln!(
                    "veclayer: Could not read user config {}: {}",
                    path.display(),
                    e
                );
                Self::default()
            }
        }
    }

    /// Expand tilde (`~`) in path-like fields of the global config.
    fn expand_paths(&mut self) {
        if let Some(ref d) = self.data_dir {
            self.data_dir = Some(shellexpand::tilde(d).into_owned());
        }
    }

    /// Discover and load user config from standard locations.
    ///
    /// Uses [`user_config_path`] for resolution, with special handling for
    /// `VECLAYER_USER_CONFIG`: warns and returns defaults if the file is missing.
    pub fn discover() -> Self {
        // Special case: explicit env var → warn if file missing (don't fall through)
        if let Ok(path) = std::env::var("VECLAYER_USER_CONFIG") {
            let p = Path::new(&path);
            if p.exists() {
                return Self::load(p);
            }
            eprintln!(
                "veclayer: VECLAYER_USER_CONFIG is set to '{}' but the file does not exist — using defaults",
                path
            );
            return Self::default();
        }

        // Standard lookup: load if the resolved path exists, else defaults
        let path = user_config_path();
        if path.exists() {
            Self::load(&path)
        } else {
            Self::default()
        }
    }

    /// Resolve config for a given directory and optional git remote, merging globals
    /// and matching overrides.
    ///
    /// Each `[[match]]` entry can have a `path` glob and/or `git-remote` regex.
    /// Either matcher triggering counts as a match (OR logic).
    /// All matching overrides are applied in declaration order; last match wins per field.
    pub fn resolve(&self, cwd: &Path, git_remote: Option<&str>) -> ResolvedConfig {
        let cwd_str = cwd.to_string_lossy();

        let mut resolved = ResolvedConfig {
            project: self.project.clone(),
            data_dir: self.data_dir.clone(),
            host: self.host.clone(),
            port: self.port,
            read_only: self.read_only,
            scopes: Vec::new(),
            storage: None,
            push: None,
        };

        let mut match_scope_names: Vec<String> = Vec::new();

        for override_ in &self.matches {
            if override_.matches(cwd_str.as_ref(), git_remote) {
                if override_.project.is_some() {
                    resolved.project = override_.project.clone();
                }
                if override_.data_dir.is_some() {
                    resolved.data_dir = override_.data_dir.clone();
                }
                if override_.host.is_some() {
                    resolved.host = override_.host.clone();
                }
                if override_.port.is_some() {
                    resolved.port = override_.port;
                }
                if override_.read_only.is_some() {
                    resolved.read_only = override_.read_only;
                }
                for scope_name in &override_.scopes {
                    if !match_scope_names.contains(scope_name) {
                        match_scope_names.push(scope_name.clone());
                    }
                }
            }
        }

        resolved.scopes = self.resolve_scopes(&[], &match_scope_names);
        resolved
    }

    /// Resolve named scopes from the user config's `[scopes]` map.
    ///
    /// Produces a deduplicated union of `project_scopes` and `match_scopes`,
    /// preserving declaration order (project scopes first). Scope names not
    /// found in `self.scopes` are warned about and skipped.
    pub fn resolve_scopes(
        &self,
        project_scopes: &[String],
        match_scopes: &[String],
    ) -> Vec<ResolvedScope> {
        let mut seen: Vec<String> = Vec::new();
        for name in project_scopes.iter().chain(match_scopes.iter()) {
            if !seen.contains(name) {
                seen.push(name.clone());
            }
        }

        seen.into_iter()
            .filter_map(|name| match self.scopes.get(&name) {
                Some(scope_config) => Some(ResolvedScope {
                    name: name.clone(),
                    storage: scope_config.storage.clone(),
                    branch: scope_config
                        .branch
                        .clone()
                        .unwrap_or_else(|| "veclayer-memory".to_string()),
                    push: scope_config
                        .push
                        .clone()
                        .unwrap_or_else(|| "manual".to_string()),
                }),
                None => {
                    warn!(
                        "Unknown scope '{}' — skipping (not defined in [scopes])",
                        name
                    );
                    None
                }
            })
            .collect()
    }
}

/// Walk up from `start_dir` looking for a `.veclayer/` directory.
/// Returns `(data_dir, project_config)` if found.
#[cfg(feature = "config")]
pub fn discover_project(start_dir: &Path) -> Option<(PathBuf, ProjectConfig)> {
    let git_info = crate::git::detect::detect(start_dir);

    // Stop walk-up at $HOME — ~/.veclayer/ is the user config fallback,
    // not a project-local store.
    let home = directories::BaseDirs::new().map(|b| b.home_dir().to_path_buf());

    let mut dir = start_dir;
    loop {
        // Don't look inside $HOME itself — only below it
        if home.as_deref() == Some(dir) {
            return None;
        }

        let candidate = dir.join(".veclayer");
        if candidate.is_dir() {
            let config_path = candidate.join("config.toml");
            let mut project_config = if config_path.exists() {
                let contents = match std::fs::read_to_string(&config_path) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!(
                            "veclayer: Failed to read {}: {} — fix or remove the file",
                            config_path.display(),
                            e
                        );
                        return None;
                    }
                };
                match toml::from_str(&contents) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!(
                            "veclayer: Invalid TOML in {}: {} — fix the syntax",
                            config_path.display(),
                            e
                        );
                        return None;
                    }
                }
            } else {
                ProjectConfig::default()
            };

            if project_config.project.is_none() {
                project_config.project = git_info.remote.clone();
            }
            project_config.branch = git_info.branch.clone();

            return Some((candidate, project_config));
        }
        dir = dir.parent()?;
    }
}

/// Return the path to the user config file, using the same lookup order as
/// [`UserConfig::discover`], but without loading or creating the file.
#[cfg(feature = "config")]
pub fn user_config_path() -> PathBuf {
    if let Ok(path) = std::env::var("VECLAYER_USER_CONFIG") {
        return PathBuf::from(path);
    }

    if let Ok(config_home) = std::env::var("XDG_CONFIG_HOME") {
        return PathBuf::from(config_home).join("veclayer/config.toml");
    }

    if let Some(base) = directories::BaseDirs::new() {
        return base.config_dir().join("veclayer/config.toml");
    }

    // BaseDirs failed — try $HOME manually
    if let Some(home) = std::env::var("HOME").ok().map(PathBuf::from) {
        return home.join(".veclayer/config.toml");
    }

    PathBuf::from(".veclayer/config.toml")
}
