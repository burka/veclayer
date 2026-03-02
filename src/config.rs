//! Configuration with 12-factor layered resolution: ENV > TOML file > Defaults.
//!
//! Config file lookup order:
//! 1. `$VECLAYER_CONFIG` (explicit path)
//! 2. `<data_dir>/veclayer.toml`
//! 3. `./veclayer.toml`
//!
//! User config lookup order (for match-based overrides):
//! 1. `$VECLAYER_USER_CONFIG` (explicit path)
//! 2. `$XDG_CONFIG_HOME/veclayer/config.toml`
//! 3. `$HOME/.config/veclayer/config.toml`
//! 4. `$HOME/.veclayer/config.toml`
//!
//! Any field set in the environment always wins over the config file.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use tracing::warn;

pub const GLOB_MATCH_OPTIONS: glob::MatchOptions = glob::MatchOptions {
    case_sensitive: true,
    require_literal_separator: true,
    require_literal_leading_dot: false,
};

/// Runtime configuration for VecLayer.
#[derive(Debug, Clone)]
pub struct Config {
    /// Directory where VecLayer stores its data (LanceDB files)
    pub data_dir: PathBuf,

    /// Embedder to use
    pub embedder: EmbedderConfig,

    /// LLM provider for the think/sleep cycle
    pub llm: LlmConfig,

    /// Whether to run in read-only mode
    pub read_only: bool,

    /// Port for the HTTP/MCP server
    pub port: u16,

    /// Host to bind the server to
    pub host: String,

    /// Number of top-level results to fetch in hierarchical search
    pub search_top_k: usize,

    /// Number of children to fetch per parent in hierarchical search
    pub search_children_k: usize,

    /// Project scope for memory isolation (None = no scoping)
    pub project: Option<String>,

    /// Git branch for branch-scoped entries (auto-detected)
    pub branch: Option<String>,
}

#[derive(Debug, Clone)]
pub enum EmbedderConfig {
    FastEmbed { model: String },
    Ollama { model: String, base_url: String },
}

// --- TOML file schema (all fields optional) ---

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct FileConfig {
    data_dir: Option<String>,
    host: Option<String>,
    port: Option<u16>,
    read_only: Option<bool>,
    search_top_k: Option<usize>,
    search_children_k: Option<usize>,
    embedder: Option<FileEmbedderConfig>,
    llm: Option<FileLlmConfig>,
}

/// Unified match override: path glob and/or git-remote regex, plus config fields.
/// At least one matcher (path or git-remote) must be present.
#[derive(Debug, Clone)]
pub struct MatchOverride {
    pub path: Option<glob::Pattern>,
    #[allow(dead_code)]
    pub git_remote: Option<regex::Regex>,
    pub project: Option<String>,
    pub data_dir: Option<String>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub read_only: Option<bool>,
}

impl MatchOverride {
    /// Check if this override matches the given cwd and/or git remote (OR logic).
    pub fn matches(&self, cwd_str: &str, git_remote: Option<&str>) -> bool {
        let path_match = self
            .path
            .as_ref()
            .is_some_and(|p| p.matches_with(cwd_str, GLOB_MATCH_OPTIONS));
        let remote_match = self
            .git_remote
            .as_ref()
            .is_some_and(|re| git_remote.is_some_and(|r| re.is_match(r)));
        path_match || remote_match
    }

    pub fn path_matches(&self, cwd_str: &str) -> bool {
        self.path
            .as_ref()
            .is_some_and(|p| p.matches_with(cwd_str, GLOB_MATCH_OPTIONS))
    }

    pub fn remote_matches(&self, git_remote: Option<&str>) -> bool {
        self.git_remote
            .as_ref()
            .is_some_and(|re| git_remote.is_some_and(|r| re.is_match(r)))
    }
}

impl<'de> Deserialize<'de> for MatchOverride {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Debug, Deserialize)]
        struct Raw {
            path: Option<String>,
            #[serde(rename = "git-remote")]
            git_remote: Option<String>,
            project: Option<String>,
            data_dir: Option<String>,
            host: Option<String>,
            port: Option<u16>,
            read_only: Option<bool>,
        }

        let raw = Raw::deserialize(deserializer)?;

        if raw.path.is_none() && raw.git_remote.is_none() {
            return Err(serde::de::Error::custom(
                "[[match]] requires at least one of 'path' or 'git-remote'",
            ));
        }

        let path = raw
            .path
            .map(|d| {
                let expanded = shellexpand::tilde(&d).to_string();
                glob::Pattern::new(&expanded).map_err(serde::de::Error::custom)
            })
            .transpose()?;

        let git_remote = raw
            .git_remote
            .map(|p| regex::Regex::new(&p).map_err(serde::de::Error::custom))
            .transpose()?;

        let data_dir = raw.data_dir.map(|d| shellexpand::tilde(&d).into_owned());

        Ok(MatchOverride {
            path,
            git_remote,
            project: raw.project,
            data_dir,
            host: raw.host,
            port: raw.port,
            read_only: raw.read_only,
        })
    }
}

/// User-level configuration with global defaults and match-based overrides.
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
}

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
        let cwd_str = cwd
            .to_str()
            .expect("Current working directory is not valid UTF-8");

        let mut resolved = ResolvedConfig {
            project: self.project.clone(),
            data_dir: self.data_dir.clone(),
            host: self.host.clone(),
            port: self.port,
            read_only: self.read_only,
        };

        for override_ in &self.matches {
            if override_.matches(cwd_str, git_remote) {
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
            }
        }

        resolved
    }
}

/// Resolved configuration from user config (globals + path match).
#[derive(Debug, Clone, Default)]
pub struct ResolvedConfig {
    pub project: Option<String>,
    pub data_dir: Option<String>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub read_only: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct FileLlmConfig {
    /// "ollama" or "openai"
    #[serde(default = "default_llm_provider")]
    provider: String,
    model: Option<String>,
    base_url: Option<String>,
    api_key: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<usize>,
}

fn default_llm_provider() -> String {
    "ollama".to_string()
}

#[derive(Debug, Deserialize)]
struct FileEmbedderConfig {
    /// "fastembed" or "ollama"
    #[serde(rename = "type", default = "default_embedder_type")]
    embedder_type: String,
    model: Option<String>,
    base_url: Option<String>,
}

fn default_embedder_type() -> String {
    "fastembed".to_string()
}

impl FileConfig {
    /// Try to load from a TOML file. Returns default (all-None) on any error.
    fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(contents) => match toml::from_str(&contents) {
                Ok(config) => config,
                Err(e) => {
                    warn!(
                        "Malformed config file {}: {} — using defaults",
                        path.display(),
                        e
                    );
                    Self::default()
                }
            },
            Err(e) => {
                warn!("Could not read config file {}: {}", path.display(), e);
                Self::default()
            }
        }
    }

    /// Find and load the config file, if one exists.
    fn discover(data_dir_hint: Option<&Path>) -> Self {
        // 1. Explicit path from ENV
        if let Ok(path) = std::env::var("VECLAYER_CONFIG") {
            let p = Path::new(&path);
            if p.exists() {
                return Self::load(p);
            }
        }

        // 2. Inside data_dir
        if let Some(dir) = data_dir_hint {
            let candidate = dir.join("veclayer.toml");
            if candidate.exists() {
                return Self::load(&candidate);
            }
        }

        // 3. Current working directory
        let cwd = Path::new("./veclayer.toml");
        if cwd.exists() {
            return Self::load(cwd);
        }

        Self::default()
    }
}

// --- Hardcoded defaults ---

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 8080;
const DEFAULT_SEARCH_TOP_K: usize = 5;
const DEFAULT_SEARCH_CHILDREN_K: usize = 3;
const DEFAULT_FASTEMBED_MODEL: &str = "BAAI/bge-small-en-v1.5";
const DEFAULT_OLLAMA_MODEL: &str = "nomic-embed-text";
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

impl Config {
    /// Build config with full layered resolution: ENV > TOML file > Defaults.
    pub fn new() -> Self {
        // Resolve data_dir first (needed for TOML file discovery)
        let data_dir_env = std::env::var("VECLAYER_DATA_DIR").ok();

        // Load TOML file (uses data_dir hint for discovery)
        let file = FileConfig::discover(data_dir_env.as_ref().map(Path::new));

        // Layer: ENV > TOML > Default
        let data_dir = data_dir_env
            .or(file.data_dir)
            .map(PathBuf::from)
            .unwrap_or_else(crate::default_data_dir);

        let host = env_or("VECLAYER_HOST", file.host, DEFAULT_HOST.to_string());

        let port = env_parse("VECLAYER_PORT")
            .or(file.port)
            .unwrap_or(DEFAULT_PORT);

        let read_only = env_bool("VECLAYER_READ_ONLY")
            .or(file.read_only)
            .unwrap_or(false);

        let search_top_k = env_parse("VECLAYER_SEARCH_TOP_K")
            .or(file.search_top_k)
            .unwrap_or(DEFAULT_SEARCH_TOP_K);

        let search_children_k = env_parse("VECLAYER_SEARCH_CHILDREN_K")
            .or(file.search_children_k)
            .unwrap_or(DEFAULT_SEARCH_CHILDREN_K);

        let embedder = Self::resolve_embedder(file.embedder);
        let llm = Self::resolve_llm(file.llm);

        Self {
            data_dir,
            embedder,
            llm,
            read_only,
            port,
            host,
            search_top_k,
            search_children_k,
            project: None,
            branch: None,
        }
    }

    fn resolve_embedder(file_embedder: Option<FileEmbedderConfig>) -> EmbedderConfig {
        let embedder_type = env_or(
            "VECLAYER_EMBEDDER",
            file_embedder.as_ref().map(|e| e.embedder_type.clone()),
            "fastembed".to_string(),
        );

        match embedder_type.as_str() {
            "ollama" => {
                let model = env_or(
                    "VECLAYER_OLLAMA_MODEL",
                    file_embedder.as_ref().and_then(|e| e.model.clone()),
                    DEFAULT_OLLAMA_MODEL.to_string(),
                );
                let base_url = env_or(
                    "VECLAYER_OLLAMA_URL",
                    file_embedder.as_ref().and_then(|e| e.base_url.clone()),
                    DEFAULT_OLLAMA_URL.to_string(),
                );
                EmbedderConfig::Ollama { model, base_url }
            }
            _ => {
                let model = env_or(
                    "VECLAYER_FASTEMBED_MODEL",
                    file_embedder.as_ref().and_then(|e| e.model.clone()),
                    DEFAULT_FASTEMBED_MODEL.to_string(),
                );
                EmbedderConfig::FastEmbed { model }
            }
        }
    }

    fn resolve_llm(file_llm: Option<FileLlmConfig>) -> LlmConfig {
        let provider = env_or(
            "VECLAYER_LLM_PROVIDER",
            file_llm.as_ref().map(|l| l.provider.clone()),
            "ollama".to_string(),
        );
        let model = env_or(
            "VECLAYER_LLM_MODEL",
            file_llm.as_ref().and_then(|l| l.model.clone()),
            "llama3.2".to_string(),
        );
        let base_url = env_or(
            "VECLAYER_LLM_BASE_URL",
            file_llm.as_ref().and_then(|l| l.base_url.clone()),
            "http://localhost:11434".to_string(),
        );
        if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
            tracing::error!(
                "LLM base_url must start with http:// or https://, got: {base_url} — \
                 falling back to default"
            );
        }
        let api_key = std::env::var("VECLAYER_LLM_API_KEY")
            .ok()
            .or_else(|| file_llm.as_ref().and_then(|l| l.api_key.clone()));
        let is_loopback = base_url.contains("localhost") || base_url.contains("127.0.0.1");
        if api_key.is_some() && !base_url.starts_with("https://") && !is_loopback {
            tracing::warn!(
                "LLM base_url uses cleartext HTTP with an API key to a non-loopback host — \
                 credentials may be transmitted in the clear"
            );
        }
        let temperature = env_parse("VECLAYER_LLM_TEMPERATURE")
            .or(file_llm.as_ref().and_then(|l| l.temperature))
            .unwrap_or(0.7);
        let max_tokens = env_parse("VECLAYER_LLM_MAX_TOKENS")
            .or(file_llm.as_ref().and_then(|l| l.max_tokens))
            .unwrap_or(4096);

        LlmConfig {
            provider,
            model,
            base_url,
            api_key,
            temperature,
            max_tokens,
        }
    }

    pub fn with_data_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.data_dir = path.into();
        self
    }

    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    pub fn with_project(mut self, project: Option<String>) -> Self {
        self.project = project;
        self
    }

    pub fn with_branch(mut self, branch: Option<String>) -> Self {
        self.branch = branch;
        self
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        EmbedderConfig::FastEmbed {
            model: DEFAULT_FASTEMBED_MODEL.to_string(),
        }
    }
}

/// Configuration for the LLM provider (always available, even without the `llm` feature).
#[derive(Clone)]
pub struct LlmConfig {
    /// Provider type: "ollama" or "openai"
    pub provider: String,
    /// Model name (e.g. "llama3.2", "gpt-4o", "claude-sonnet-4-20250514")
    pub model: String,
    /// Base URL for the API
    pub base_url: String,
    /// API key (required for OpenAI-compatible providers)
    pub api_key: Option<String>,
    /// Sampling temperature
    pub temperature: f32,
    /// Maximum tokens in the response
    pub max_tokens: usize,
}

impl std::fmt::Debug for LlmConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmConfig")
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .finish()
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "ollama".to_string(),
            model: "llama3.2".to_string(),
            base_url: "http://localhost:11434".to_string(),
            api_key: None,
            temperature: 0.7,
            max_tokens: 4096,
        }
    }
}

/// Discovered project configuration from `.veclayer/config.toml`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ProjectConfig {
    /// Project name for memory isolation (explicit or auto-detected)
    pub project: Option<String>,

    /// Git branch (auto-detected, not from config file)
    #[serde(skip)]
    pub branch: Option<String>,
}

/// Walk up from `start_dir` looking for a `.veclayer/` directory.
/// Returns `(data_dir, project_config)` if found.
pub fn discover_project(start_dir: &Path) -> Option<(PathBuf, ProjectConfig)> {
    let git_info = crate::git_detect::detect(start_dir);

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
                let contents = std::fs::read_to_string(&config_path).unwrap_or_else(|e| {
                    panic!(
                        "Failed to read {}: {} — fix or remove the file",
                        config_path.display(),
                        e
                    )
                });
                toml::from_str(&contents).unwrap_or_else(|e| {
                    panic!(
                        "Invalid TOML in {}: {} — fix the syntax",
                        config_path.display(),
                        e
                    )
                })
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

/// Append a `[[match]]` block to the user config file.
///
/// At least one of `git_remote` or `path_glob` must be `Some`.
/// Parent directories are created if they do not exist.
/// Returns the path of the config file that was written.
pub fn append_match_to_user_config(
    git_remote: Option<&str>,
    path_glob: Option<&str>,
    project: &str,
) -> crate::Result<PathBuf> {
    if git_remote.is_none() && path_glob.is_none() {
        return Err(crate::Error::config(
            "at least one of git_remote or path_glob must be provided",
        ));
    }

    let config_path = user_config_path();

    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let existing = match std::fs::read_to_string(&config_path) {
        Ok(content) => content,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e.into()),
    };

    let mut block = String::from("[[match]]\n");
    if let Some(remote) = git_remote {
        block.push_str(&format!("git-remote = \"{}\"\n", remote));
    }
    if let Some(glob) = path_glob {
        block.push_str(&format!("path = \"{}\"\n", glob));
    }
    block.push_str(&format!("project = \"{}\"\n", project));

    // Build the final content: preserve existing, add a blank-line separator, append block.
    if !existing.is_empty() {
        let trimmed = existing.trim_end_matches('\n');
        let final_content = format!("{trimmed}\n\n{block}");
        std::fs::write(&config_path, final_content)?;
    } else {
        std::fs::write(&config_path, &block)?;
    }

    Ok(config_path)
}

// --- Helpers for ENV > TOML > Default resolution ---

/// Return env var if set, else TOML value if present, else default.
fn env_or(key: &str, file_val: Option<String>, default: String) -> String {
    std::env::var(key).ok().or(file_val).unwrap_or(default)
}

/// Parse env var as T. Returns None if unset or unparseable.
fn env_parse<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

/// Parse env var as boolean ("true"/"1" = true, anything else = false).
fn env_bool(key: &str) -> Option<bool> {
    std::env::var(key).ok().map(|v| v == "true" || v == "1")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_config_defaults() {
        // Clear env vars to test pure defaults
        // (can't fully clear since tests run in parallel, but verify structure)
        let config = Config::new();
        assert!(!config.data_dir.as_os_str().is_empty());
        assert!(!config.host.is_empty());
        assert!(config.port > 0);
        assert_eq!(config.search_top_k, 5);
        assert_eq!(config.search_children_k, 3);
    }

    #[test]
    fn test_config_builder_chain() {
        let config = Config::new()
            .with_data_dir("/data")
            .with_host("localhost")
            .with_port(9000)
            .with_read_only(true);

        assert_eq!(config.data_dir, Path::new("/data"));
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 9000);
        assert!(config.read_only);
    }

    #[test]
    fn test_embedder_config_default_fastembed() {
        let embedder = EmbedderConfig::default();
        assert!(
            matches!(embedder, EmbedderConfig::FastEmbed { ref model } if model == DEFAULT_FASTEMBED_MODEL),
            "Expected FastEmbed variant with default model"
        );
    }

    #[test]
    fn test_file_config_load_toml() {
        let dir = tempfile::TempDir::new().unwrap();
        let toml_path = dir.path().join("veclayer.toml");
        let mut file = std::fs::File::create(&toml_path).unwrap();
        writeln!(
            file,
            r#"
host = "0.0.0.0"
port = 3000
search_top_k = 10

[embedder]
type = "ollama"
model = "mxbai-embed-large"
base_url = "http://gpu:11434"
"#
        )
        .unwrap();

        let fc = FileConfig::load(&toml_path);
        assert_eq!(fc.host.as_deref(), Some("0.0.0.0"));
        assert_eq!(fc.port, Some(3000));
        assert_eq!(fc.search_top_k, Some(10));
        assert!(fc.data_dir.is_none()); // not specified
        assert!(fc.read_only.is_none()); // not specified

        let emb = fc.embedder.unwrap();
        assert_eq!(emb.embedder_type, "ollama");
        assert_eq!(emb.model.as_deref(), Some("mxbai-embed-large"));
        assert_eq!(emb.base_url.as_deref(), Some("http://gpu:11434"));
    }

    #[test]
    fn test_file_config_missing_file() {
        let fc = FileConfig::load(Path::new("/nonexistent/path/veclayer.toml"));
        assert!(fc.host.is_none());
        assert!(fc.port.is_none());
    }

    #[test]
    fn test_file_config_invalid_toml() {
        let dir = tempfile::TempDir::new().unwrap();
        let toml_path = dir.path().join("veclayer.toml");
        std::fs::write(&toml_path, "this is not [valid toml {{{").unwrap();

        let fc = FileConfig::load(&toml_path);
        // Should gracefully return defaults (all None)
        assert!(fc.host.is_none());
        assert!(fc.port.is_none());
    }

    #[test]
    fn test_file_config_partial_toml() {
        let dir = tempfile::TempDir::new().unwrap();
        let toml_path = dir.path().join("veclayer.toml");
        std::fs::write(&toml_path, "port = 4444\n").unwrap();

        let fc = FileConfig::load(&toml_path);
        assert_eq!(fc.port, Some(4444));
        assert!(fc.host.is_none());
        assert!(fc.data_dir.is_none());
    }

    #[test]
    fn test_env_or_helper() {
        // With no env var set for this unique key, should use file_val or default
        let result = env_or(
            "VECLAYER_TEST_NONEXISTENT_KEY_12345",
            Some("file".to_string()),
            "default".to_string(),
        );
        assert_eq!(result, "file");

        let result2 = env_or(
            "VECLAYER_TEST_NONEXISTENT_KEY_12345",
            None,
            "default".to_string(),
        );
        assert_eq!(result2, "default");
    }

    #[test]
    fn test_config_clone() {
        let config1 = Config::new().with_data_dir("/test").with_port(9999);
        let config2 = config1.clone();

        assert_eq!(config1.data_dir, config2.data_dir);
        assert_eq!(config1.port, config2.port);
        assert_eq!(config1.host, config2.host);
        assert_eq!(config1.read_only, config2.read_only);
    }

    #[test]
    fn test_config_debug_format() {
        let config = Config::new();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("Config"));
    }

    #[test]
    fn test_discover_project_walk_up() {
        let dir = tempfile::TempDir::new().unwrap();
        let veclayer_dir = dir.path().join(".veclayer");
        std::fs::create_dir_all(&veclayer_dir).unwrap();

        // With config.toml
        let config_path = veclayer_dir.join("config.toml");
        std::fs::write(&config_path, "project = \"myproject\"\n").unwrap();

        // Discover from the root
        let result = discover_project(dir.path());
        assert!(result.is_some());
        let (found_dir, config) = result.unwrap();
        assert_eq!(found_dir, veclayer_dir);
        assert_eq!(config.project.as_deref(), Some("myproject"));

        // Discover from a subdirectory
        let sub = dir.path().join("src").join("deep");
        std::fs::create_dir_all(&sub).unwrap();
        let result = discover_project(&sub);
        assert!(result.is_some());
        let (found_dir, config) = result.unwrap();
        assert_eq!(found_dir, veclayer_dir);
        assert_eq!(config.project.as_deref(), Some("myproject"));
    }

    #[test]
    fn test_discover_project_no_config() {
        let dir = tempfile::TempDir::new().unwrap();
        let veclayer_dir = dir.path().join(".veclayer");
        std::fs::create_dir_all(&veclayer_dir).unwrap();

        // No config.toml
        let result = discover_project(dir.path());
        assert!(result.is_some());
        let (found_dir, config) = result.unwrap();
        assert_eq!(found_dir, veclayer_dir);
        assert!(config.project.is_none());
    }

    #[test]
    fn test_discover_project_not_found() {
        let dir = tempfile::TempDir::new().unwrap();
        // No .veclayer/ anywhere
        let result = discover_project(dir.path());
        assert!(result.is_none());
    }

    #[test]
    #[should_panic(expected = "Invalid TOML")]
    fn test_discover_project_bad_toml_panics() {
        let dir = tempfile::TempDir::new().unwrap();
        let veclayer_dir = dir.path().join(".veclayer");
        std::fs::create_dir_all(&veclayer_dir).unwrap();
        std::fs::write(veclayer_dir.join("config.toml"), "not valid {{{ toml").unwrap();

        // Must panic — fail fast, fail loud
        discover_project(dir.path());
    }

    #[test]
    fn test_user_config_default() {
        let config = UserConfig::default();
        assert!(config.matches.is_empty());
        assert!(config.project.is_none());
        assert!(config.data_dir.is_none());
    }

    #[test]
    fn test_match_override_tilde_expansion() {
        let dir = tempfile::TempDir::new().unwrap();
        let toml_path = dir.path().join("user.toml");
        std::fs::write(
            &toml_path,
            r#"
[[match]]
path = "~/work/damalo*"
project = "damalo"
"#,
        )
        .unwrap();

        let config = UserConfig::load(&toml_path);
        assert_eq!(config.matches.len(), 1);
        assert_eq!(config.matches[0].project.as_deref(), Some("damalo"));
    }

    #[test]
    fn test_match_override_absolute_path() {
        let dir = tempfile::TempDir::new().unwrap();
        let toml_path = dir.path().join("user.toml");
        std::fs::write(
            &toml_path,
            r#"
[[match]]
path = "/tmp/test*"
project = "test"
"#,
        )
        .unwrap();

        let config = UserConfig::load(&toml_path);
        assert_eq!(config.matches.len(), 1);
        assert_eq!(config.matches[0].project.as_deref(), Some("test"));
    }

    #[test]
    fn test_resolve_single_path_match() {
        let mut config = UserConfig::default();
        config.matches.push(MatchOverride {
            path: Some(glob::Pattern::new("/tmp/test/*").unwrap()),
            git_remote: None,
            project: Some("test".to_string()),
            data_dir: Some("/tmp/test-data".to_string()),
            host: None,
            port: None,
            read_only: Some(true),
        });

        let resolved = config.resolve(Path::new("/tmp/test/something"), None);
        assert_eq!(resolved.project.as_deref(), Some("test"));
        assert_eq!(resolved.data_dir.as_deref(), Some("/tmp/test-data"));
        assert_eq!(resolved.read_only, Some(true));
    }

    #[test]
    fn test_resolve_no_match() {
        let mut config = UserConfig::default();
        config.matches.push(MatchOverride {
            path: Some(glob::Pattern::new("/tmp/test/*").unwrap()),
            git_remote: None,
            project: Some("test".to_string()),
            data_dir: None,
            host: None,
            port: None,
            read_only: None,
        });

        let resolved = config.resolve(Path::new("/other/path"), None);
        assert!(resolved.project.is_none());
        assert!(resolved.data_dir.is_none());
    }

    #[test]
    fn test_resolve_multiple_match_last_wins() {
        let mut config = UserConfig::default();

        config.matches.push(MatchOverride {
            path: Some(glob::Pattern::new("/tmp/test/**").unwrap()),
            git_remote: None,
            project: Some("first".to_string()),
            data_dir: Some("/first".to_string()),
            host: None,
            port: None,
            read_only: Some(false),
        });

        config.matches.push(MatchOverride {
            path: Some(glob::Pattern::new("/tmp/test/specific").unwrap()),
            git_remote: None,
            project: Some("second".to_string()),
            data_dir: Some("/second".to_string()),
            host: None,
            port: None,
            read_only: Some(true),
        });

        let resolved = config.resolve(Path::new("/tmp/test/specific"), None);
        assert_eq!(resolved.project.as_deref(), Some("second"));
        assert_eq!(resolved.data_dir.as_deref(), Some("/second"));
        assert_eq!(resolved.read_only, Some(true));
    }

    #[test]
    fn test_resolve_partial_override() {
        let mut config = UserConfig {
            project: Some("global".to_string()),
            ..Default::default()
        };

        config.matches.push(MatchOverride {
            path: Some(glob::Pattern::new("/tmp/*").unwrap()),
            git_remote: None,
            project: None,
            data_dir: Some("/tmp/data".to_string()),
            host: None,
            port: None,
            read_only: Some(true),
        });

        let resolved = config.resolve(Path::new("/tmp/test"), None);
        assert_eq!(resolved.project.as_deref(), Some("global"));
        assert_eq!(resolved.data_dir.as_deref(), Some("/tmp/data"));
        assert_eq!(resolved.read_only, Some(true));
    }

    #[test]
    fn test_match_override_invalid_path_pattern() {
        let dir = tempfile::TempDir::new().unwrap();
        let toml_path = dir.path().join("user.toml");
        std::fs::write(
            &toml_path,
            r#"
[[match]]
path = "[[invalid"
project = "test"
"#,
        )
        .unwrap();

        let config = UserConfig::load(&toml_path);
        assert!(config.matches.is_empty());
    }

    #[test]
    fn test_match_override_no_matcher_rejected() {
        let dir = tempfile::TempDir::new().unwrap();
        let toml_path = dir.path().join("user.toml");
        std::fs::write(
            &toml_path,
            r#"
[[match]]
project = "orphan"
"#,
        )
        .unwrap();

        // Should fail to parse — at least one matcher required
        let config = UserConfig::load(&toml_path);
        assert!(config.matches.is_empty());
    }

    // BUG-2: tilde in global data_dir must be expanded after load
    #[test]
    fn test_user_config_global_data_dir_tilde_expanded() {
        let dir = tempfile::TempDir::new().unwrap();
        let toml_path = dir.path().join("user.toml");
        std::fs::write(&toml_path, "data_dir = \"~/.veclayer\"\n").unwrap();

        let config = UserConfig::load(&toml_path);
        let data_dir = config.data_dir.expect("data_dir should be set");
        assert!(
            !data_dir.starts_with('~'),
            "data_dir '{}' should not start with '~' after tilde expansion",
            data_dir
        );
    }

    // BUG-2: tilde in match override data_dir must be expanded during deserialization
    #[test]
    fn test_match_override_data_dir_tilde_expanded() {
        let dir = tempfile::TempDir::new().unwrap();
        let toml_path = dir.path().join("user.toml");
        std::fs::write(
            &toml_path,
            "[[match]]\npath = \"/tmp/work\"\ndata_dir = \"~/.veclayer\"\n",
        )
        .unwrap();

        let config = UserConfig::load(&toml_path);
        let data_dir = config.matches[0]
            .data_dir
            .as_deref()
            .expect("match override data_dir should be set");
        assert!(
            !data_dir.starts_with('~'),
            "match override data_dir '{}' should not start with '~' after tilde expansion",
            data_dir
        );
    }

    // BUG-3: explicit VECLAYER_USER_CONFIG pointing to nonexistent file must not fall through
    #[test]
    #[serial_test::serial]
    fn test_discover_user_config_nonexistent_env_returns_defaults() {
        let original = std::env::var("VECLAYER_USER_CONFIG").ok();

        std::env::set_var(
            "VECLAYER_USER_CONFIG",
            "/nonexistent/path/that/does/not/exist.toml",
        );
        let config = UserConfig::discover();
        assert!(
            config.matches.is_empty(),
            "should return default (empty matches)"
        );
        assert!(
            config.data_dir.is_none(),
            "should return default (no data_dir)"
        );

        match original {
            Some(v) => std::env::set_var("VECLAYER_USER_CONFIG", v),
            None => std::env::remove_var("VECLAYER_USER_CONFIG"),
        }
    }

    #[test]
    fn test_match_git_remote_only() {
        let toml_str = r#"
[[match]]
git-remote = "(?i)damalo"
project = "damalo"

[[match]]
git-remote = "github\\.com/myorg/"
project = "myorg"
"#;
        let config: UserConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.matches.len(), 2);

        // git-remote match, no path
        let resolved = config.resolve(Path::new("/other"), Some("github.com/Damalo/some-repo"));
        assert_eq!(resolved.project.as_deref(), Some("damalo"));

        let resolved = config.resolve(Path::new("/other"), Some("github.com/myorg/tool"));
        assert_eq!(resolved.project.as_deref(), Some("myorg"));

        let resolved = config.resolve(Path::new("/other"), Some("github.com/unrelated/repo"));
        assert!(resolved.project.is_none());
    }

    #[test]
    fn test_match_last_wins_with_remote() {
        let toml_str = r#"
[[match]]
git-remote = "specific-repo"
project = "specific"

[[match]]
git-remote = ".*"
project = "catch-all"
"#;
        let config: UserConfig = toml::from_str(toml_str).unwrap();
        // Last match wins: catch-all matches everything, so it always wins
        let resolved = config.resolve(Path::new("/tmp"), Some("github.com/org/specific-repo"));
        assert_eq!(resolved.project.as_deref(), Some("catch-all"));

        let resolved = config.resolve(Path::new("/tmp"), Some("github.com/org/other"));
        assert_eq!(resolved.project.as_deref(), Some("catch-all"));
    }

    #[test]
    fn test_match_or_logic_both_matchers() {
        let toml_str = r#"
[[match]]
path = "/home/flob/work/damalo*"
git-remote = "(?i)damalo"
project = "damalo"
"#;
        let config: UserConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.matches.len(), 1);

        // Path matches, no remote
        let resolved = config.resolve(Path::new("/home/flob/work/damalo-app"), None);
        assert_eq!(resolved.project.as_deref(), Some("damalo"));

        // Remote matches, different path
        let resolved = config.resolve(Path::new("/other/path"), Some("github.com/Damalo/repo"));
        assert_eq!(resolved.project.as_deref(), Some("damalo"));

        // Both match
        let resolved = config.resolve(
            Path::new("/home/flob/work/damalo-app"),
            Some("github.com/Damalo/repo"),
        );
        assert_eq!(resolved.project.as_deref(), Some("damalo"));

        // Neither matches
        let resolved = config.resolve(Path::new("/other/path"), Some("github.com/other/repo"));
        assert!(resolved.project.is_none());
    }

    #[test]
    fn test_match_no_remote_provided() {
        let config = UserConfig::default();
        let resolved = config.resolve(Path::new("/tmp"), None);
        assert!(resolved.project.is_none());
    }

    // NIT-3: * must not cross path separators (require_literal_separator = true)
    #[test]
    fn test_resolve_star_does_not_cross_separator() {
        let mut config = UserConfig::default();

        config.matches.push(MatchOverride {
            path: Some(glob::Pattern::new("/tmp/work*").unwrap()),
            git_remote: None,
            project: Some("shallow".to_string()),
            data_dir: None,
            host: None,
            port: None,
            read_only: None,
        });

        // /tmp/work/deep has a slash after the * position — must not match
        let resolved_deep = config.resolve(Path::new("/tmp/work/deep"), None);
        assert!(
            resolved_deep.project.is_none(),
            "* should not cross / (got {:?})",
            resolved_deep.project
        );

        // /tmp/workspace has no slash after the * position — must match
        let resolved_shallow = config.resolve(Path::new("/tmp/workspace"), None);
        assert_eq!(
            resolved_shallow.project.as_deref(),
            Some("shallow"),
            "* should match within a single path component"
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_append_match_to_user_config() {
        let dir = tempfile::TempDir::new().unwrap();
        let config_path = dir.path().join("config.toml");

        // Use env var to point to our temp file
        std::env::set_var("VECLAYER_USER_CONFIG", config_path.to_str().unwrap());

        let result = append_match_to_user_config(
            Some("github.com/org/repo"),
            Some("/home/user/work/project*"),
            "myproject",
        );

        std::env::remove_var("VECLAYER_USER_CONFIG");

        let path = result.unwrap();
        assert_eq!(path, config_path);

        let contents = std::fs::read_to_string(&config_path).unwrap();
        assert!(contents.contains("[[match]]"));
        assert!(contents.contains("git-remote = \"github.com/org/repo\""));
        assert!(contents.contains("path = \"/home/user/work/project*\""));
        assert!(contents.contains("project = \"myproject\""));

        // Verify it round-trips through UserConfig::load
        let loaded = UserConfig::load(&config_path);
        assert_eq!(loaded.matches.len(), 1);
        assert_eq!(loaded.matches[0].project.as_deref(), Some("myproject"));
    }
}
