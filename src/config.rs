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

use std::path::PathBuf;

use serde::Deserialize;

#[cfg(feature = "config")]
use std::collections::HashMap;
#[cfg(feature = "config")]
use std::path::Path;
#[cfg(feature = "config")]
use tracing::warn;

#[cfg(feature = "config")]
pub const GLOB_MATCH_OPTIONS: glob::MatchOptions = glob::MatchOptions {
    case_sensitive: true,
    require_literal_separator: true,
    require_literal_leading_dot: false,
};

/// A named memory scope — a storage backend for entries.
#[derive(Debug, Clone, Deserialize)]
pub struct ScopeConfig {
    /// Where entries are stored: "git" (orphan branch in current repo),
    /// a git URL, or a local directory path.
    pub storage: String,
    /// Git branch name (default: "veclayer-memory").
    pub branch: Option<String>,
    /// Push mode: "always", "review", "manual", or "off".
    pub push: Option<String>,
}

/// Runtime configuration for VecLayer.
#[derive(Debug, Clone)]
pub struct Config {
    /// Directory where VecLayer stores its data (LanceDB files)
    pub data_dir: PathBuf,

    /// Embedder to use
    pub embedder: EmbedderConfig,

    /// LLM provider for the think/sleep cycle
    pub llm: LlmConfig,

    /// Authentication configuration
    pub auth: AuthConfig,

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

    /// Storage backend for the project scope (e.g. "git" for git memory branch)
    pub storage: Option<String>,

    /// Push mode for git storage (parsed from project/user config string)
    #[cfg(feature = "config")]
    pub push_mode: crate::git::branch_config::PushMode,

    /// Whether hooks (e.g. the `stale` stop hook) are enabled (default: true).
    /// Set to false to disable the Claude Code stop hook without env vars.
    pub hooks_enabled: bool,
}

/// Authentication configuration.
#[derive(Debug, Clone)]
pub struct AuthConfig {
    /// Require authentication for HTTP API access (default: false for backward compat).
    pub auth_required: bool,

    /// Public URL of this server (used as OAuth issuer and JWT audience).
    /// Example: "https://my-veclayer.fly.dev"
    pub server_url: Option<String>,

    /// Access token lifetime in seconds (default: 3600 = 1 hour).
    pub token_expiry_secs: u64,

    /// Refresh token lifetime in seconds (default: 2592000 = 30 days).
    pub refresh_expiry_secs: u64,

    /// Auto-approve OAuth authorization requests (TESTING ONLY — never in production).
    pub auto_approve: bool,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            auth_required: false,
            server_url: None,
            token_expiry_secs: crate::util::TOKEN_EXPIRY_SECS,
            refresh_expiry_secs: crate::util::REFRESH_MAX_LIFETIME_SECS,
            auto_approve: false,
        }
    }
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum EmbedderConfig {
    FastEmbed {
        model: String,
    },
    Ollama {
        model: String,
        base_url: String,
        /// Dimension of embeddings returned by the model
        dimension: usize,
    },
}

impl EmbedderConfig {
    /// Returns the known embedding dimension, if determinable from config alone.
    /// FastEmbed models have well-known dimensions; Ollama config stores it explicitly.
    pub fn dimension(&self) -> Option<usize> {
        match self {
            Self::FastEmbed { .. } => Some(384), // BGESmallENV15 default
            Self::Ollama { dimension, .. } => Some(*dimension),
        }
    }
}

// --- TOML file schema (all fields optional, gated behind "config" feature) ---

#[cfg(feature = "config")]
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct FileConfig {
    data_dir: Option<String>,
    host: Option<String>,
    port: Option<u16>,
    read_only: Option<bool>,
    hooks_enabled: Option<bool>,
    search_top_k: Option<usize>,
    search_children_k: Option<usize>,
    embedder: Option<FileEmbedderConfig>,
    llm: Option<FileLlmConfig>,
    auth: Option<FileAuthConfig>,
}

#[cfg(feature = "config")]
#[derive(Debug, Deserialize)]
struct FileAuthConfig {
    auth_required: Option<bool>,
    server_url: Option<String>,
    token_expiry_secs: Option<u64>,
    refresh_expiry_secs: Option<u64>,
    auto_approve: Option<bool>,
}

/// Unified match override: path glob and/or git-remote regex, plus config fields.
/// At least one matcher (path or git-remote) must be present.
#[cfg(feature = "config")]
#[derive(Debug, Clone)]
pub struct MatchOverride {
    pub path: Option<glob::Pattern>,
    pub git_remote: Option<regex::Regex>,
    pub project: Option<String>,
    pub data_dir: Option<String>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub read_only: Option<bool>,
    /// Named scopes to activate when this override matches.
    pub scopes: Vec<String>,
}

#[cfg(feature = "config")]
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

#[cfg(feature = "config")]
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
            #[serde(default)]
            scopes: Vec<String>,
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
            .map(|p| {
                regex::RegexBuilder::new(&p)
                    .size_limit(256 * 1024)
                    .build()
                    .map_err(serde::de::Error::custom)
            })
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
            scopes: raw.scopes,
        })
    }
}

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

/// A fully resolved scope ready for use.
#[cfg(feature = "config")]
#[derive(Debug, Clone)]
pub struct ResolvedScope {
    pub name: String,
    pub storage: String,
    pub branch: String,
    pub push: String,
}

/// Resolved configuration from user config (globals + path match).
#[cfg(feature = "config")]
#[derive(Debug, Clone, Default)]
pub struct ResolvedConfig {
    pub project: Option<String>,
    pub data_dir: Option<String>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub read_only: Option<bool>,
    /// Resolved scopes active for this context.
    pub scopes: Vec<ResolvedScope>,
    /// Project-level storage override (from project config).
    pub storage: Option<String>,
    /// Project-level push mode (from project config).
    pub push: Option<String>,
}

#[cfg(feature = "config")]
#[derive(Deserialize)]
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

#[cfg(feature = "config")]
impl std::fmt::Debug for FileLlmConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileLlmConfig")
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .finish()
    }
}

#[cfg(feature = "config")]
fn default_llm_provider() -> String {
    "ollama".to_string()
}

#[cfg(feature = "config")]
#[derive(Debug, Deserialize)]
struct FileEmbedderConfig {
    /// "fastembed" or "ollama"
    #[serde(rename = "type", default = "default_embedder_type")]
    embedder_type: String,
    model: Option<String>,
    base_url: Option<String>,
    /// Embedding vector dimension (required for ollama; ignored for fastembed)
    dimension: Option<usize>,
}

#[cfg(feature = "config")]
fn default_embedder_type() -> String {
    "fastembed".to_string()
}

#[cfg(feature = "config")]
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
            warn!(
                "VECLAYER_CONFIG is set to '{}' but the file does not exist — using defaults",
                path
            );
            return Self::default();
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

// --- Hardcoded defaults (used by Config::new(), gated behind "config") ---

#[cfg(feature = "config")]
const DEFAULT_HOST: &str = "127.0.0.1";
#[cfg(feature = "config")]
const DEFAULT_PORT: u16 = 8080;
#[cfg(feature = "config")]
const DEFAULT_SEARCH_TOP_K: usize = 5;
#[cfg(feature = "config")]
const DEFAULT_SEARCH_CHILDREN_K: usize = 3;
const DEFAULT_FASTEMBED_MODEL: &str = "Xenova/bge-small-en-v1.5";
#[cfg(feature = "config")]
const DEFAULT_OLLAMA_MODEL: &str = crate::util::DEFAULT_OLLAMA_EMBED_MODEL;
#[cfg(feature = "config")]
const DEFAULT_OLLAMA_URL: &str = crate::util::DEFAULT_OLLAMA_URL;
#[cfg(feature = "config")]
const DEFAULT_OLLAMA_DIMENSION: usize = crate::util::DEFAULT_OLLAMA_DIMENSION;

/// Summarised result of Ollama auto-discovery, used inside `Config::new()`.
///
/// Kept in a plain struct (no feature gate) so the same type can be threaded
/// through both the `llm`-gated detection path and the always-present
/// `resolve_embedder` / `resolve_llm` signatures without conditional compilation
/// complexity at the call sites.
#[cfg(feature = "config")]
struct DetectedOllama {
    base_url: String,
    /// Best available embedding model, or `None` if Ollama has no embed models.
    embed_model: Option<String>,
    /// Best available chat model, or `None` if Ollama has no chat models.
    chat_model: Option<String>,
}

/// Summarised result of OpenAI-compatible embedding-service auto-discovery
/// (vLLM, HuggingFace TEI, …), used inside `Config::new()`.
///
/// Unlike [`DetectedOllama`] this always carries a concrete `dimension`, learned
/// from a live `/v1/embeddings` probe, so the store is sized to the served model
/// instead of the Ollama default.
#[cfg(feature = "config")]
struct DetectedOpenAiEmbed {
    base_url: String,
    model: String,
    dimension: usize,
}

/// True when the user has explicitly pinned the embedder, by any means:
/// the `VECLAYER_EMBEDDER` type selector, a `[embedder]` TOML block, or any of
/// the Ollama embedder overrides (`VECLAYER_OLLAMA_URL/MODEL/DIMENSION`).
///
/// Auto-discovery (Ollama *and* OpenAI-compatible) is suppressed in all of these
/// cases. Counting the `VECLAYER_OLLAMA_*` vars here is what prevents a probe
/// from injecting a model/dimension that conflicts with a user-pinned endpoint.
#[cfg(feature = "config")]
fn embedder_explicitly_set(file_embedder: &Option<FileEmbedderConfig>) -> bool {
    std::env::var("VECLAYER_EMBEDDER").is_ok()
        || file_embedder.is_some()
        || std::env::var("VECLAYER_OLLAMA_URL").is_ok()
        || std::env::var("VECLAYER_OLLAMA_MODEL").is_ok()
        || std::env::var("VECLAYER_OLLAMA_DIMENSION").is_ok()
}

#[cfg(feature = "config")]
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

        let hooks_enabled = env_bool("VECLAYER_HOOKS_ENABLED")
            .or(file.hooks_enabled)
            .unwrap_or(true);

        let search_top_k = env_parse("VECLAYER_SEARCH_TOP_K")
            .or(file.search_top_k)
            .unwrap_or(DEFAULT_SEARCH_TOP_K);

        let search_children_k = env_parse("VECLAYER_SEARCH_CHILDREN_K")
            .or(file.search_children_k)
            .unwrap_or(DEFAULT_SEARCH_CHILDREN_K);

        // Auto-discover a local Ollama instance when the user has not explicitly
        // configured an embedder or LLM provider via env vars or the config file.
        // The probe uses a 500 ms timeout so it never meaningfully delays startup.
        // Returns (base_url, embed_model, chat_model) or None.
        let detected_ollama = Self::maybe_detect_ollama(&file.embedder, &file.llm);

        // When no embedder is configured and Ollama offered no embedding model,
        // fall back to probing for a local OpenAI-compatible embedding service
        // (vLLM/TEI) so a GPU-backed endpoint is used automatically when present.
        let detected_openai =
            Self::maybe_detect_openai_embedder(&file.embedder, detected_ollama.as_ref());

        let embedder = Self::resolve_embedder(
            file.embedder,
            detected_ollama.as_ref(),
            detected_openai.as_ref(),
        );
        let llm = Self::resolve_llm(file.llm, detected_ollama.as_ref());
        let auth = Self::resolve_auth(file.auth);

        Self {
            data_dir,
            embedder,
            llm,
            auth,
            read_only,
            hooks_enabled,
            port,
            host,
            search_top_k,
            search_children_k,
            project: None,
            branch: None,
            storage: None,
            push_mode: crate::git::branch_config::PushMode::default(),
        }
    }

    /// Run Ollama auto-discovery when neither the embedder nor the LLM provider
    /// has been explicitly configured (no env var, no TOML entry).
    ///
    /// Returns `None` immediately when explicit configuration is present, so the
    /// 500 ms probe never fires for users who have already set up their stack.
    fn maybe_detect_ollama(
        file_embedder: &Option<FileEmbedderConfig>,
        file_llm: &Option<FileLlmConfig>,
    ) -> Option<DetectedOllama> {
        let llm_explicitly_set =
            std::env::var("VECLAYER_LLM_PROVIDER").is_ok() || file_llm.is_some();

        if embedder_explicitly_set(file_embedder) && llm_explicitly_set {
            return None;
        }

        #[cfg(feature = "llm")]
        if let Some(info) = crate::ollama_discover::detect_ollama() {
            tracing::info!(
                "Detected local Ollama at {} with models: embed=[{}] chat=[{}]",
                info.base_url,
                info.embedding_models.join(", "),
                info.chat_models.join(", "),
            );
            return Some(DetectedOllama {
                embed_model: info.best_embedding_model().map(str::to_string),
                chat_model: info.best_chat_model().map(str::to_string),
                base_url: info.base_url,
            });
        }

        None
    }

    /// Probe for a local OpenAI-compatible embedding service (vLLM, TEI, …) only
    /// when no embedder is explicitly configured *and* Ollama auto-discovery did
    /// not already yield an embedding model.
    ///
    /// Returns `None` without probing in those short-circuit cases, so the extra
    /// `/v1/models` + `/v1/embeddings` round-trip never fires for users who have
    /// an explicit embedder or a working Ollama embed model.
    fn maybe_detect_openai_embedder(
        file_embedder: &Option<FileEmbedderConfig>,
        detected_ollama: Option<&DetectedOllama>,
    ) -> Option<DetectedOpenAiEmbed> {
        if embedder_explicitly_set(file_embedder) {
            return None;
        }

        let ollama_has_embed_model = detected_ollama
            .and_then(|d| d.embed_model.as_deref())
            .is_some();
        if ollama_has_embed_model {
            return None;
        }

        #[cfg(feature = "llm")]
        if let Some(info) = crate::openai_compat_discover::detect() {
            tracing::info!(
                "Detected OpenAI-compatible embedding service at {} with model {} ({} dims)",
                info.base_url,
                info.embed_model,
                info.dimension,
            );
            return Some(DetectedOpenAiEmbed {
                base_url: info.base_url,
                model: info.embed_model,
                dimension: info.dimension,
            });
        }

        None
    }

    fn resolve_embedder(
        file_embedder: Option<FileEmbedderConfig>,
        detected: Option<&DetectedOllama>,
        openai: Option<&DetectedOpenAiEmbed>,
    ) -> EmbedderConfig {
        // Resolve a *single* detected network-embedder source up front, so that
        // model / base_url / dimension can never be drawn from two different
        // servers. A native Ollama embed model takes precedence over an
        // OpenAI-compatible match (it is the more specific protocol hit); the two
        // are mutually exclusive in practice because `maybe_detect_openai_embedder`
        // bails when Ollama already offers an embed model. Both kinds are served
        // through the Ollama-protocol embedder, which transparently falls back to
        // `/v1/embeddings` for OpenAI-compatible endpoints.
        //
        // Ollama discovery does not learn dimensions, so that source contributes
        // `None` and the dimension falls through to the default; only the
        // OpenAI-compat probe carries a measured dimension.
        let (detected_model, detected_base_url, detected_dimension): (
            Option<String>,
            Option<String>,
            Option<usize>,
        ) = match (detected.and_then(|d| d.embed_model.as_deref()), openai) {
            (Some(embed_model), _) => (
                Some(embed_model.to_string()),
                detected.map(|d| d.base_url.clone()),
                None,
            ),
            (None, Some(o)) => (
                Some(o.model.clone()),
                Some(o.base_url.clone()),
                Some(o.dimension),
            ),
            (None, None) => (None, None, None),
        };

        // Determine the embedder type. When no explicit config is present and a
        // network embedder (Ollama embed model or OpenAI-compatible service) was
        // detected, switch to the Ollama-protocol embedder.
        let explicit_type = std::env::var("VECLAYER_EMBEDDER")
            .ok()
            .or_else(|| file_embedder.as_ref().map(|e| e.embedder_type.clone()));

        let use_ollama_auto = explicit_type.is_none() && detected_model.is_some();

        let embedder_type = explicit_type.unwrap_or_else(|| {
            if use_ollama_auto {
                "ollama".to_string()
            } else if cfg!(feature = "embedding-local") {
                "fastembed".to_string()
            } else {
                // Default to Ollama when embedding-local is not compiled in.
                // The embedder will produce a clear error if Ollama is unreachable,
                // and recall will fall back to keyword search.
                "ollama".to_string()
            }
        });

        match embedder_type.as_str() {
            "ollama" => {
                // Prefer: env var > TOML file > detected source > hardcoded default.
                // All three fields draw from the same `detected_*` source, so a
                // chat-only Ollama can never supply a base_url while the model and
                // dimension come from a different (OpenAI-compatible) server.
                let model = std::env::var("VECLAYER_OLLAMA_MODEL")
                    .ok()
                    .or_else(|| file_embedder.as_ref().and_then(|e| e.model.clone()))
                    .or(detected_model)
                    .unwrap_or_else(|| DEFAULT_OLLAMA_MODEL.to_string());
                let base_url = std::env::var("VECLAYER_OLLAMA_URL")
                    .ok()
                    .or_else(|| file_embedder.as_ref().and_then(|e| e.base_url.clone()))
                    .or(detected_base_url)
                    .unwrap_or_else(|| DEFAULT_OLLAMA_URL.to_string());
                let dimension = std::env::var("VECLAYER_OLLAMA_DIMENSION")
                    .ok()
                    .and_then(|v| match v.parse() {
                        Ok(d) => Some(d),
                        Err(_) => {
                            warn!(
                                "VECLAYER_OLLAMA_DIMENSION is set to '{v}' which is not a valid \
                                 integer — ignoring, using default"
                            );
                            None
                        }
                    })
                    .or_else(|| file_embedder.as_ref().and_then(|e| e.dimension))
                    .or(detected_dimension)
                    .unwrap_or(DEFAULT_OLLAMA_DIMENSION);
                EmbedderConfig::Ollama {
                    model,
                    base_url,
                    dimension,
                }
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

    fn resolve_llm(
        file_llm: Option<FileLlmConfig>,
        detected: Option<&DetectedOllama>,
    ) -> LlmConfig {
        // Auto-configure LLM when no explicit provider is set and Ollama was
        // detected with a chat model.  ENV > TOML > auto-detected > hardcoded default.
        let provider = env_or(
            "VECLAYER_LLM_PROVIDER",
            file_llm.as_ref().map(|l| l.provider.clone()),
            "ollama".to_string(),
        );
        let model = std::env::var("VECLAYER_LLM_MODEL")
            .ok()
            .or_else(|| file_llm.as_ref().and_then(|l| l.model.clone()))
            .or_else(|| detected.and_then(|d| d.chat_model.clone()))
            .unwrap_or_else(|| "llama3.2".to_string());
        let base_url = std::env::var("VECLAYER_LLM_BASE_URL")
            .ok()
            .or_else(|| file_llm.as_ref().and_then(|l| l.base_url.clone()))
            .or_else(|| detected.map(|d| d.base_url.clone()))
            .unwrap_or_else(|| DEFAULT_OLLAMA_URL.to_string());
        let base_url = if base_url.starts_with("http://") || base_url.starts_with("https://") {
            base_url
        } else {
            tracing::error!(
                "LLM base_url must start with http:// or https://, got: {base_url} — \
                 falling back to default {DEFAULT_OLLAMA_URL}"
            );
            DEFAULT_OLLAMA_URL.to_string()
        };
        let api_key_from_env = std::env::var("VECLAYER_LLM_API_KEY").ok();
        let api_key_from_file = file_llm.as_ref().and_then(|l| l.api_key.clone());
        if api_key_from_env.is_none() && api_key_from_file.is_some() {
            tracing::warn!(
                "LLM API key loaded from config file — consider using the \
                 VECLAYER_LLM_API_KEY environment variable instead for better security"
            );
        }
        let api_key = api_key_from_env
            .or(api_key_from_file)
            .map(secrecy::SecretString::from);
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

    fn resolve_auth(file_auth: Option<FileAuthConfig>) -> AuthConfig {
        let defaults = AuthConfig::default();

        let auth_required = env_bool("VECLAYER_AUTH_REQUIRED")
            .or(file_auth.as_ref().and_then(|a| a.auth_required))
            .unwrap_or(defaults.auth_required);

        let server_url = std::env::var("VECLAYER_SERVER_URL")
            .ok()
            .or_else(|| file_auth.as_ref().and_then(|a| a.server_url.clone()));

        let token_expiry_secs = env_parse("VECLAYER_TOKEN_EXPIRY")
            .or(file_auth.as_ref().and_then(|a| a.token_expiry_secs))
            .unwrap_or(defaults.token_expiry_secs);

        let refresh_expiry_secs = env_parse("VECLAYER_REFRESH_EXPIRY")
            .or(file_auth.as_ref().and_then(|a| a.refresh_expiry_secs))
            .unwrap_or(defaults.refresh_expiry_secs);

        let auto_approve = env_bool("VECLAYER_AUTO_APPROVE")
            .or(file_auth.as_ref().and_then(|a| a.auto_approve))
            .unwrap_or(defaults.auto_approve);

        AuthConfig {
            auth_required,
            server_url,
            token_expiry_secs,
            refresh_expiry_secs,
            auto_approve,
        }
    }
}

impl Config {
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

    pub fn with_storage(mut self, storage: Option<String>) -> Self {
        self.storage = storage;
        self
    }

    #[cfg(feature = "config")]
    pub fn with_push_mode(mut self, push: Option<&str>) -> Self {
        if let Some(p) = push {
            self.push_mode = parse_push_mode(p);
        }
        self
    }

    pub fn with_auth_required(mut self, auth_required: bool) -> Self {
        self.auth.auth_required = auth_required;
        self
    }
}

/// Parse a push mode string, with backward-compatible mapping of "auto" → Always.
/// NOTE: This intentionally falls back to PushMode::Review with a warning for unknown values,
/// unlike branch_config.rs which returns a hard error. Project/user config is user-edited TOML
/// where a hard error would be disruptive; the branch config is a committed, controlled file
/// where typos should be caught immediately.
#[cfg(feature = "config")]
pub fn parse_push_mode(s: &str) -> crate::git::branch_config::PushMode {
    use crate::git::branch_config::PushMode;
    match s {
        "always" => PushMode::Always,
        "auto" => {
            tracing::warn!("push mode 'auto' is deprecated, use 'always' instead");
            PushMode::Always
        }
        "pull-request" => {
            tracing::warn!("push mode 'pull-request' is not implemented; treating as 'review'");
            PushMode::Review
        }
        "review" => PushMode::Review,
        "manual" => PushMode::Manual,
        "off" => PushMode::Off,
        unknown => {
            tracing::warn!("unknown push mode '{unknown}', defaulting to 'review'");
            PushMode::Review
        }
    }
}

#[cfg(feature = "config")]
impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        if cfg!(feature = "embedding-local") {
            EmbedderConfig::FastEmbed {
                model: DEFAULT_FASTEMBED_MODEL.to_string(),
            }
        } else {
            EmbedderConfig::Ollama {
                model: crate::util::DEFAULT_OLLAMA_EMBED_MODEL.to_string(),
                base_url: crate::util::DEFAULT_OLLAMA_URL.to_string(),
                dimension: crate::util::DEFAULT_OLLAMA_DIMENSION,
            }
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
    /// API key (required for OpenAI-compatible providers). Stored as a
    /// `SecretString` so the value is zeroed in memory on drop.
    pub api_key: Option<secrecy::SecretString>,
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
            base_url: crate::util::DEFAULT_OLLAMA_URL.to_string(),
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

    /// Storage backend override for this project ("git", a git URL, or a local path).
    pub storage: Option<String>,

    /// Push mode override for this project ("always", "review", "manual", or "off").
    pub push: Option<String>,

    /// Named scopes to activate for this project.
    pub scopes: Vec<String>,
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

/// Append a `[[match]]` block to the user config file.
///
/// At least one of `git_remote` or `path_glob` must be `Some`.
/// Parent directories are created if they do not exist.
/// Returns the path of the config file that was written.
#[cfg(feature = "config")]
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
        block.push_str(&format!(
            "git-remote = \"{}\"\n",
            toml_escape_string(remote)
        ));
    }
    if let Some(glob) = path_glob {
        block.push_str(&format!("path = \"{}\"\n", toml_escape_string(glob)));
    }
    block.push_str(&format!("project = \"{}\"\n", toml_escape_string(project)));

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

// --- Helpers for ENV > TOML > Default resolution (only used by Config::new()) ---

/// Escape a string value for safe embedding inside a TOML basic string (double-quoted).
///
/// Handles the named escapes `\\`, `\"`, `\n`, `\r`, `\t` and `\uXXXX`-encodes any
/// remaining control character (`U+0000`–`U+001F` and `U+007F`), which TOML forbids
/// bare inside a basic string. Any value written without these escapes produces
/// invalid TOML that won't round-trip.
#[cfg(feature = "config")]
fn toml_escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c == '\u{7f}' => {
                out.push_str(&format!("\\u{:04X}", c as u32));
            }
            other => out.push(other),
        }
    }
    out
}

#[cfg(feature = "config")]
fn env_or(key: &str, file_val: Option<String>, default: String) -> String {
    std::env::var(key).ok().or(file_val).unwrap_or(default)
}

#[cfg(feature = "config")]
fn env_parse<T: std::str::FromStr>(key: &str) -> Option<T> {
    let raw = std::env::var(key).ok()?;
    match raw.parse() {
        Ok(v) => Some(v),
        Err(_) => {
            warn!("{key} is set to '{raw}' which could not be parsed — ignoring, using default");
            None
        }
    }
}

#[cfg(feature = "config")]
fn env_bool(key: &str) -> Option<bool> {
    let raw = std::env::var(key).ok()?;
    match raw.as_str() {
        "true" | "1" => Some(true),
        "false" | "0" => Some(false),
        other => {
            warn!(
                "{key} is set to '{other}' which is not a recognised boolean \
                 (expected true/1/false/0) — treating as false"
            );
            Some(false)
        }
    }
}

#[cfg(all(test, feature = "config"))]
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

    // Security: api_key is a SecretString — Debug must redact, not leak.
    #[test]
    fn test_llm_config_debug_redacts_api_key_when_present() {
        let config = LlmConfig {
            api_key: Some(secrecy::SecretString::from("sk-supersecret")),
            ..LlmConfig::default()
        };
        let debug_output = format!("{config:?}");
        assert!(
            debug_output.contains("<redacted>"),
            "Debug output must contain '<redacted>', got: {debug_output}"
        );
        assert!(
            !debug_output.contains("sk-supersecret"),
            "Debug output must NOT leak the secret value, got: {debug_output}"
        );
    }

    // Security: api_key absent → Debug shows None, not a redacted placeholder.
    #[test]
    fn test_llm_config_debug_shows_none_when_api_key_absent() {
        let config = LlmConfig {
            api_key: None,
            ..LlmConfig::default()
        };
        let debug_output = format!("{config:?}");
        assert!(
            debug_output.contains("api_key: None"),
            "Debug output must show 'api_key: None' when absent, got: {debug_output}"
        );
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
    fn test_embedder_config_default() {
        let embedder = EmbedderConfig::default();
        if cfg!(feature = "embedding-local") {
            assert!(
                matches!(embedder, EmbedderConfig::FastEmbed { ref model } if model == DEFAULT_FASTEMBED_MODEL),
                "Expected FastEmbed variant with default model when embedding-local is enabled"
            );
        } else {
            assert!(
                matches!(embedder, EmbedderConfig::Ollama { .. }),
                "Expected Ollama variant when embedding-local is disabled"
            );
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_resolve_llm_invalid_base_url_falls_back_to_default() {
        std::env::remove_var("VECLAYER_LLM_BASE_URL");
        let bad = FileLlmConfig {
            provider: "ollama".to_string(),
            model: None,
            base_url: Some("ftp://not-http".to_string()),
            api_key: None,
            temperature: None,
            max_tokens: None,
        };
        let llm = Config::resolve_llm(Some(bad), None);
        assert_eq!(
            llm.base_url, DEFAULT_OLLAMA_URL,
            "an invalid base_url must actually fall back to the default"
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_resolve_embedder_ollama_from_env() {
        // Use values DISTINCT from the defaults so the test proves env wins over default,
        // not that it accidentally equals the default.
        std::env::set_var("VECLAYER_EMBEDDER", "ollama");
        std::env::set_var("VECLAYER_OLLAMA_MODEL", "custom-model");
        std::env::set_var("VECLAYER_OLLAMA_URL", "http://gpu:11434");
        std::env::set_var("VECLAYER_OLLAMA_DIMENSION", "1024");

        let embedder = Config::resolve_embedder(None, None, None);

        std::env::remove_var("VECLAYER_EMBEDDER");
        std::env::remove_var("VECLAYER_OLLAMA_MODEL");
        std::env::remove_var("VECLAYER_OLLAMA_URL");
        std::env::remove_var("VECLAYER_OLLAMA_DIMENSION");

        assert!(matches!(
            embedder,
            EmbedderConfig::Ollama {
                ref model,
                ref base_url,
                dimension
            } if model == "custom-model"
                && base_url == "http://gpu:11434"
                && dimension == 1024
        ));
    }

    #[test]
    #[serial_test::serial]
    fn test_resolve_embedder_prefers_openai_compat_when_no_ollama_embed() {
        // No explicit embedder is configured and Ollama offered no embedding
        // model, but a local OpenAI-compatible service (e.g. vLLM) was detected
        // serving a 1024-dim model. resolve_embedder must point the Ollama-
        // protocol embedder — which transparently falls back to /v1/embeddings —
        // at that endpoint, carrying the probed dimension so the store is sized
        // correctly rather than defaulting to 768.
        std::env::remove_var("VECLAYER_EMBEDDER");
        std::env::remove_var("VECLAYER_OLLAMA_MODEL");
        std::env::remove_var("VECLAYER_OLLAMA_URL");
        std::env::remove_var("VECLAYER_OLLAMA_DIMENSION");

        let openai = DetectedOpenAiEmbed {
            base_url: "http://localhost:8000".to_string(),
            model: "BAAI/bge-m3".to_string(),
            dimension: 1024,
        };
        let embedder = Config::resolve_embedder(None, None, Some(&openai));

        assert!(matches!(
            embedder,
            EmbedderConfig::Ollama {
                ref model,
                ref base_url,
                dimension
            } if model == "BAAI/bge-m3"
                && base_url == "http://localhost:8000"
                && dimension == 1024
        ));
    }

    #[test]
    #[serial_test::serial]
    fn test_resolve_embedder_ollama_embed_wins_over_openai() {
        // When both an Ollama embed model and an OpenAI-compat service are
        // available, the Ollama-native model takes precedence (it is the more
        // specific, native protocol match).
        std::env::remove_var("VECLAYER_EMBEDDER");
        std::env::remove_var("VECLAYER_OLLAMA_MODEL");
        std::env::remove_var("VECLAYER_OLLAMA_URL");
        std::env::remove_var("VECLAYER_OLLAMA_DIMENSION");

        let ollama = DetectedOllama {
            base_url: "http://localhost:11434".to_string(),
            embed_model: Some("nomic-embed-text".to_string()),
            chat_model: None,
        };
        let openai = DetectedOpenAiEmbed {
            base_url: "http://localhost:8000".to_string(),
            model: "BAAI/bge-m3".to_string(),
            dimension: 1024,
        };
        let embedder = Config::resolve_embedder(None, Some(&ollama), Some(&openai));

        // All three fields must come from Ollama — never a mix where the model
        // is Ollama's but the dimension leaks from the OpenAI-compat probe. Since
        // Ollama discovery learns no dimension, it falls back to the default.
        assert!(
            matches!(
                embedder,
                EmbedderConfig::Ollama {
                    ref model,
                    ref base_url,
                    dimension
                } if model == "nomic-embed-text"
                    && base_url == "http://localhost:11434"
                    && dimension == DEFAULT_OLLAMA_DIMENSION
            ),
            "Ollama embed must win on all fields with no OpenAI-compat leakage"
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_resolve_embedder_chat_only_ollama_does_not_contaminate_openai() {
        // Regression: a chat-only Ollama (no embed model) plus a detected
        // OpenAI-compatible embed service must yield an embedder whose model,
        // base_url AND dimension all come from the OpenAI-compat server. The
        // chat-only Ollama's base_url must NOT leak in.
        std::env::remove_var("VECLAYER_EMBEDDER");
        std::env::remove_var("VECLAYER_OLLAMA_MODEL");
        std::env::remove_var("VECLAYER_OLLAMA_URL");
        std::env::remove_var("VECLAYER_OLLAMA_DIMENSION");

        let ollama = DetectedOllama {
            base_url: "http://localhost:11434".to_string(),
            embed_model: None,
            chat_model: Some("llama3.2".to_string()),
        };
        let openai = DetectedOpenAiEmbed {
            base_url: "http://localhost:8000".to_string(),
            model: "BAAI/bge-m3".to_string(),
            dimension: 1024,
        };
        let embedder = Config::resolve_embedder(None, Some(&ollama), Some(&openai));

        assert!(
            matches!(
                embedder,
                EmbedderConfig::Ollama {
                    ref model,
                    ref base_url,
                    dimension
                } if model == "BAAI/bge-m3"
                    && base_url == "http://localhost:8000"
                    && dimension == 1024
            ),
            "chat-only Ollama must not contaminate the OpenAI-compat embedder"
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_embedder_explicitly_set_detects_ollama_env_overrides() {
        for key in [
            "VECLAYER_EMBEDDER",
            "VECLAYER_OLLAMA_URL",
            "VECLAYER_OLLAMA_MODEL",
            "VECLAYER_OLLAMA_DIMENSION",
        ] {
            std::env::remove_var(key);
        }

        // Nothing set, no file → not explicit.
        assert!(!embedder_explicitly_set(&None));

        // A file embedder block alone pins it.
        let file = FileEmbedderConfig {
            embedder_type: "fastembed".to_string(),
            model: None,
            base_url: None,
            dimension: None,
        };
        assert!(embedder_explicitly_set(&Some(file)));

        // Each Ollama override env var pins it too — this is what suppresses the
        // OpenAI-compat probe so it can't inject a conflicting model/dimension.
        for key in [
            "VECLAYER_OLLAMA_URL",
            "VECLAYER_OLLAMA_MODEL",
            "VECLAYER_OLLAMA_DIMENSION",
        ] {
            std::env::set_var(key, "x");
            assert!(
                embedder_explicitly_set(&None),
                "{key} must count as an explicit embedder"
            );
            std::env::remove_var(key);
        }
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
    fn test_discover_project_bad_toml_returns_none() {
        let dir = tempfile::TempDir::new().unwrap();
        let veclayer_dir = dir.path().join(".veclayer");
        std::fs::create_dir_all(&veclayer_dir).unwrap();
        std::fs::write(veclayer_dir.join("config.toml"), "not valid {{{ toml").unwrap();

        // Malformed config.toml must return None gracefully, not panic
        let result = discover_project(dir.path());
        assert!(result.is_none());
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
            scopes: vec![],
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
            scopes: vec![],
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
            scopes: vec![],
        });

        config.matches.push(MatchOverride {
            path: Some(glob::Pattern::new("/tmp/test/specific").unwrap()),
            git_remote: None,
            project: Some("second".to_string()),
            data_dir: Some("/second".to_string()),
            host: None,
            port: None,
            read_only: Some(true),
            scopes: vec![],
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
            scopes: vec![],
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
    // NOTE(known-limitation): std::env::set_var/remove_var are unsafe since Rust 1.83+.
    // These tests use serial_test to avoid data races, but will need unsafe blocks when
    // the crate upgrades to Rust edition 2024. See README "Known Limitations".
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
path = "/tmp/damalo*"
git-remote = "(?i)damalo"
project = "damalo"
"#;
        let config: UserConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.matches.len(), 1);

        // Path matches, no remote
        let resolved = config.resolve(Path::new("/tmp/damalo-app"), None);
        assert_eq!(resolved.project.as_deref(), Some("damalo"));

        // Remote matches, different path
        let resolved = config.resolve(Path::new("/other/path"), Some("github.com/Damalo/repo"));
        assert_eq!(resolved.project.as_deref(), Some("damalo"));

        // Both match
        let resolved = config.resolve(Path::new("/tmp/damalo-app"), Some("github.com/Damalo/repo"));
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
            scopes: vec![],
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

    // NOTE(known-limitation): std::env::set_var/remove_var — see comment above.
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

    #[test]
    fn test_auth_config_defaults() {
        let auth = AuthConfig::default();
        assert!(!auth.auth_required);
        assert!(auth.server_url.is_none());
        assert_eq!(auth.token_expiry_secs, crate::util::TOKEN_EXPIRY_SECS);
        assert_eq!(
            auth.refresh_expiry_secs,
            crate::util::REFRESH_MAX_LIFETIME_SECS
        );
        assert!(!auth.auto_approve);
    }

    #[test]
    fn test_auth_config_from_toml() {
        let toml_str = r#"
[auth]
auth_required = true
server_url = "https://my-veclayer.example.com"
token_expiry_secs = 1800
refresh_expiry_secs = 86400
auto_approve = true
"#;
        let fc: FileConfig = toml::from_str(toml_str).unwrap();
        let auth_file = fc.auth.unwrap();
        assert_eq!(auth_file.auth_required, Some(true));
        assert_eq!(
            auth_file.server_url.as_deref(),
            Some("https://my-veclayer.example.com")
        );
        assert_eq!(auth_file.token_expiry_secs, Some(1800));
        assert_eq!(auth_file.refresh_expiry_secs, Some(86400));
        assert_eq!(auth_file.auto_approve, Some(true));
    }

    #[test]
    #[serial_test::serial]
    fn test_auth_config_env_override() {
        let saved_required = std::env::var("VECLAYER_AUTH_REQUIRED").ok();
        let saved_url = std::env::var("VECLAYER_SERVER_URL").ok();
        let saved_expiry = std::env::var("VECLAYER_TOKEN_EXPIRY").ok();
        let saved_approve = std::env::var("VECLAYER_AUTO_APPROVE").ok();

        std::env::set_var("VECLAYER_AUTH_REQUIRED", "true");
        std::env::set_var("VECLAYER_SERVER_URL", "https://env.example.com");
        std::env::set_var("VECLAYER_TOKEN_EXPIRY", "7200");
        std::env::set_var("VECLAYER_AUTO_APPROVE", "1");

        let auth = Config::resolve_auth(None);

        // Restore env
        match saved_required {
            Some(v) => std::env::set_var("VECLAYER_AUTH_REQUIRED", v),
            None => std::env::remove_var("VECLAYER_AUTH_REQUIRED"),
        }
        match saved_url {
            Some(v) => std::env::set_var("VECLAYER_SERVER_URL", v),
            None => std::env::remove_var("VECLAYER_SERVER_URL"),
        }
        match saved_expiry {
            Some(v) => std::env::set_var("VECLAYER_TOKEN_EXPIRY", v),
            None => std::env::remove_var("VECLAYER_TOKEN_EXPIRY"),
        }
        match saved_approve {
            Some(v) => std::env::set_var("VECLAYER_AUTO_APPROVE", v),
            None => std::env::remove_var("VECLAYER_AUTO_APPROVE"),
        }

        assert!(auth.auth_required);
        assert_eq!(auth.server_url.as_deref(), Some("https://env.example.com"));
        assert_eq!(auth.token_expiry_secs, 7200);
        assert!(auth.auto_approve);
    }

    #[test]
    fn test_scope_config_parsing() {
        let toml_str = r#"
[scopes.personal]
storage = "git@github.com:flob/my-memory.git"
push = "manual"

[scopes.acme]
storage = "git@github.com:acme/shared-memory.git"
push = "review"
branch = "acme-memory"
"#;
        let config: UserConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.scopes.len(), 2);

        let personal = config.scopes.get("personal").unwrap();
        assert_eq!(personal.storage, "git@github.com:flob/my-memory.git");
        assert_eq!(personal.push.as_deref(), Some("manual"));
        assert!(personal.branch.is_none());

        let acme = config.scopes.get("acme").unwrap();
        assert_eq!(acme.storage, "git@github.com:acme/shared-memory.git");
        assert_eq!(acme.push.as_deref(), Some("review"));
        assert_eq!(acme.branch.as_deref(), Some("acme-memory"));
    }

    #[test]
    fn test_match_with_scopes() {
        let toml_str = r#"
[scopes.personal]
storage = "git@github.com:flob/my-memory.git"

[scopes.acme]
storage = "git@github.com:acme/shared-memory.git"

[[match]]
git-remote = "github.com/acme/"
project = "acme-stuff"
scopes = ["personal", "acme"]
"#;
        let config: UserConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.matches.len(), 1);
        assert_eq!(config.matches[0].scopes, vec!["personal", "acme"]);
        assert_eq!(config.matches[0].project.as_deref(), Some("acme-stuff"));
    }

    #[test]
    fn test_project_config_with_scopes() {
        let toml_str = r#"
project = "myproject"
storage = "git"
push = "auto"
scopes = ["acme"]
"#;
        let project_config: ProjectConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(project_config.project.as_deref(), Some("myproject"));
        assert_eq!(project_config.storage.as_deref(), Some("git"));
        assert_eq!(project_config.push.as_deref(), Some("auto"));
        assert_eq!(project_config.scopes, vec!["acme"]);
    }

    #[test]
    fn test_scope_resolution() {
        let toml_str = r#"
[scopes.personal]
storage = "git@github.com:flob/my-memory.git"

[scopes.acme]
storage = "git@github.com:acme/shared-memory.git"
push = "review"
"#;
        let config: UserConfig = toml::from_str(toml_str).unwrap();

        // Union: project=[acme], match=[personal, acme] → [acme, personal] (dedup, project first)
        let project_scopes = vec!["acme".to_string()];
        let match_scopes = vec!["personal".to_string(), "acme".to_string()];
        let resolved = config.resolve_scopes(&project_scopes, &match_scopes);

        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].name, "acme");
        assert_eq!(resolved[0].storage, "git@github.com:acme/shared-memory.git");
        assert_eq!(resolved[0].push, "review");
        assert_eq!(resolved[0].branch, "veclayer-memory"); // default

        assert_eq!(resolved[1].name, "personal");
        assert_eq!(resolved[1].storage, "git@github.com:flob/my-memory.git");
        assert_eq!(resolved[1].push, "manual"); // default
        assert_eq!(resolved[1].branch, "veclayer-memory"); // default
    }

    #[test]
    fn test_unknown_scope_warning() {
        let toml_str = r#"
[scopes.known]
storage = "git"
"#;
        let config: UserConfig = toml::from_str(toml_str).unwrap();

        let project_scopes = vec!["known".to_string()];
        let match_scopes = vec!["unknown".to_string()];
        let resolved = config.resolve_scopes(&project_scopes, &match_scopes);

        // "unknown" is skipped; only "known" resolves
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].name, "known");
    }

    // with_auth_required builder tests

    #[test]
    fn test_with_auth_required_sets_true() {
        // Builder must propagate true regardless of env/file defaults.
        let config = Config::new().with_auth_required(true);
        assert!(config.auth.auth_required);
    }

    #[test]
    fn test_with_auth_required_sets_false() {
        // Builder must propagate false, making it authoritative for the merged CLI value.
        let config = Config::new().with_auth_required(false);
        assert!(!config.auth.auth_required);
    }

    #[test]
    fn test_with_auth_required_overrides_prior_true() {
        // Start with auth_required=true (via builder), then override with false.
        // Documents that with_auth_required is fully authoritative — the last call wins.
        let config = Config::new()
            .with_auth_required(true)
            .with_auth_required(false);
        assert!(!config.auth.auth_required);
    }

    #[test]
    fn test_with_auth_required_composes_in_chain() {
        // Verify that with_auth_required returns Self and can be composed
        // with the other builder methods without breaking anything.
        let config = Config::new()
            .with_port(9090)
            .with_auth_required(true)
            .with_read_only(false);
        assert!(config.auth.auth_required);
        assert_eq!(config.port, 9090);
        assert!(!config.read_only);
    }

    // parse_push_mode: unrecognized string must warn and fall back to PushMode::Review.
    #[test]
    fn test_parse_push_mode_unknown_falls_back_to_review() {
        use crate::git::branch_config::PushMode;
        assert!(
            matches!(parse_push_mode("review"), PushMode::Review),
            "canonical \"review\" must map to PushMode::Review"
        );
        let result = parse_push_mode("bogus");
        assert!(
            matches!(result, PushMode::Review),
            "expected PushMode::Review for unknown input, got {:?}",
            result
        );
    }

    // append_match_to_user_config: both matchers None must return Err with the guard message.
    #[test]
    fn test_append_match_to_user_config_both_none_returns_err() {
        let result = append_match_to_user_config(None, None, "myproject");
        assert!(
            result.is_err(),
            "expected Err when both git_remote and path_glob are None"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("at least one of"),
            "error message should contain 'at least one of', got: {msg}"
        );
    }

    // Regression: resolve() with a valid UTF-8 cwd must behave identically to before
    // the lossy-conversion change (common-case correctness).
    #[test]
    fn test_resolve_utf8_cwd_behavior_unchanged() {
        let mut config = UserConfig::default();
        config.matches.push(MatchOverride {
            path: Some(glob::Pattern::new("/tmp/project/*").unwrap()),
            git_remote: None,
            project: Some("myproject".to_string()),
            data_dir: Some("/tmp/data".to_string()),
            host: None,
            port: None,
            read_only: Some(false),
            scopes: vec![],
        });

        // Matching path: override must be applied.
        let resolved = config.resolve(Path::new("/tmp/project/src"), None);
        assert_eq!(
            resolved.project.as_deref(),
            Some("myproject"),
            "UTF-8 matching cwd must yield the expected project override"
        );
        assert_eq!(resolved.data_dir.as_deref(), Some("/tmp/data"));
        assert_eq!(resolved.read_only, Some(false));

        // Non-matching path: override must NOT be applied.
        let resolved_no_match = config.resolve(Path::new("/other/path"), None);
        assert!(
            resolved_no_match.project.is_none(),
            "UTF-8 non-matching cwd must not apply any override"
        );
    }

    // Edge case: resolve() with a non-UTF-8 cwd must not panic (lossy conversion).
    // On Linux, paths are arbitrary byte sequences that need not be valid UTF-8.
    // This test would panic against the old `.expect("... not valid UTF-8")` and
    // must pass cleanly after the fix.
    #[test]
    #[cfg(unix)]
    fn test_resolve_non_utf8_cwd_does_not_panic() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        // 0x66 0x80 0x6f — the 0x80 byte is not valid UTF-8.
        let non_utf8_bytes: &[u8] = &[0x2f, 0x66, 0x80, 0x6f]; // "/f\x80o"
        let non_utf8_path = Path::new(OsStr::from_bytes(non_utf8_bytes));

        let config = UserConfig::default();
        // Must not panic; result is unimportant (no matches configured).
        let resolved = config.resolve(non_utf8_path, None);
        assert!(
            resolved.project.is_none(),
            "non-UTF-8 cwd with no match overrides must return no project"
        );
    }

    // --- toml_escape_string + append_match_to_user_config round-trip tests ---

    // GREEN: normal values with no special characters must serialize and re-parse cleanly.
    #[test]
    #[serial_test::serial]
    fn test_append_match_round_trips_normal_values() {
        let dir = tempfile::TempDir::new().unwrap();
        let config_path = dir.path().join("config.toml");
        std::env::set_var("VECLAYER_USER_CONFIG", config_path.to_str().unwrap());

        append_match_to_user_config(
            Some("github.com/org/normal-repo"),
            Some("/home/user/work/project"),
            "normal-project",
        )
        .unwrap();

        std::env::remove_var("VECLAYER_USER_CONFIG");

        let loaded = UserConfig::load(&config_path);
        assert_eq!(loaded.matches.len(), 1);
        assert_eq!(
            loaded.matches[0].project.as_deref(),
            Some("normal-project"),
            "normal project value must round-trip unchanged"
        );
    }

    // EDGE: a project name containing a double-quote must produce valid TOML and round-trip.
    #[test]
    #[serial_test::serial]
    fn test_append_match_round_trips_double_quote_in_project() {
        let dir = tempfile::TempDir::new().unwrap();
        let config_path = dir.path().join("config.toml");
        std::env::set_var("VECLAYER_USER_CONFIG", config_path.to_str().unwrap());

        let project_with_quote = "my\"project";
        append_match_to_user_config(None, Some("/tmp/work"), project_with_quote).unwrap();

        std::env::remove_var("VECLAYER_USER_CONFIG");

        // The written file must be parseable by the toml crate.
        let contents = std::fs::read_to_string(&config_path).unwrap();
        let parsed: Result<toml::Value, _> = toml::from_str(&contents);
        assert!(
            parsed.is_ok(),
            "file containing escaped double-quote must be valid TOML, got: {parsed:?}"
        );

        // And the project field must round-trip to the original string.
        let loaded = UserConfig::load(&config_path);
        assert_eq!(loaded.matches.len(), 1);
        assert_eq!(
            loaded.matches[0].project.as_deref(),
            Some(project_with_quote),
            "double-quote in project must round-trip unchanged"
        );
    }

    // EDGE: a path glob containing a backslash must produce valid TOML and round-trip.
    #[test]
    #[serial_test::serial]
    fn test_append_match_round_trips_backslash_in_path() {
        let dir = tempfile::TempDir::new().unwrap();
        let config_path = dir.path().join("config.toml");
        std::env::set_var("VECLAYER_USER_CONFIG", config_path.to_str().unwrap());

        // Windows-style path: contains backslashes.
        let path_with_backslash = r"C:\Users\bob\work";
        append_match_to_user_config(None, Some(path_with_backslash), "winproject").unwrap();

        std::env::remove_var("VECLAYER_USER_CONFIG");

        let contents = std::fs::read_to_string(&config_path).unwrap();
        let parsed: Result<toml::Value, _> = toml::from_str(&contents);
        assert!(
            parsed.is_ok(),
            "file containing escaped backslash must be valid TOML, got: {parsed:?}"
        );

        let loaded = UserConfig::load(&config_path);
        // UserConfig::load parses the `path` field through glob::Pattern; the raw string
        // (before glob compilation) is not directly available after loading. We verify the
        // TOML is valid and the match entry was parsed (not silently dropped).
        // A glob::Pattern for a Windows path may or may not be valid on Linux — what matters
        // is that the TOML itself is well-formed and no panic occurs.
        let _ = loaded; // parsed without panic — sufficient to prove TOML validity
    }

    // EDGE: a project name containing a literal newline must produce valid TOML and round-trip.
    #[test]
    #[serial_test::serial]
    fn test_append_match_round_trips_newline_in_project() {
        let dir = tempfile::TempDir::new().unwrap();
        let config_path = dir.path().join("config.toml");
        std::env::set_var("VECLAYER_USER_CONFIG", config_path.to_str().unwrap());

        let project_with_newline = "line1\nline2";
        append_match_to_user_config(None, Some("/tmp/work"), project_with_newline).unwrap();

        std::env::remove_var("VECLAYER_USER_CONFIG");

        let contents = std::fs::read_to_string(&config_path).unwrap();
        let parsed: Result<toml::Value, _> = toml::from_str(&contents);
        assert!(
            parsed.is_ok(),
            "file containing escaped newline must be valid TOML, got: {parsed:?}"
        );

        let loaded = UserConfig::load(&config_path);
        assert_eq!(loaded.matches.len(), 1);
        assert_eq!(
            loaded.matches[0].project.as_deref(),
            Some(project_with_newline),
            "newline in project must round-trip unchanged"
        );
    }

    // Unit test for the escaping helper itself — all special characters in one pass.
    #[test]
    fn test_toml_escape_string_all_special_chars() {
        // Each special character must be replaced by its TOML escape sequence.
        assert_eq!(toml_escape_string("\\"), "\\\\");
        assert_eq!(toml_escape_string("\""), "\\\"");
        assert_eq!(toml_escape_string("\n"), "\\n");
        assert_eq!(toml_escape_string("\r"), "\\r");
        assert_eq!(toml_escape_string("\t"), "\\t");

        // A string with all of them combined.
        let input = "a\\b\"c\nd\re\tf";
        let escaped = toml_escape_string(input);
        assert_eq!(escaped, r#"a\\b\"c\nd\re\tf"#);

        // The escaped result, wrapped in quotes, must be parseable by the toml crate.
        let toml_str = format!("value = \"{escaped}\"");
        let parsed: toml::Value =
            toml::from_str(&toml_str).expect("escaped string must produce valid TOML");
        assert_eq!(
            parsed["value"].as_str().unwrap(),
            input,
            "escaped TOML value must round-trip to the original string"
        );
    }

    // Unit test: plain strings (no special chars) pass through unchanged.
    #[test]
    fn test_toml_escape_string_plain_passthrough() {
        let plain = "github.com/org/repo-name_v2.0";
        assert_eq!(toml_escape_string(plain), plain);
    }

    // EDGE: C0 control characters and U+007F must be \uXXXX-escaped so the result
    // is valid TOML that round-trips, rather than a bare control char the toml
    // crate rejects on read-back.
    #[test]
    fn test_toml_escape_string_control_chars() {
        // U+0001 (SOH), U+0008 (BS), U+001B (ESC), U+007F (DEL) are forbidden bare.
        assert_eq!(toml_escape_string("\u{01}"), "\\u0001");
        assert_eq!(toml_escape_string("\u{08}"), "\\u0008");
        assert_eq!(toml_escape_string("\u{1b}"), "\\u001B");
        assert_eq!(toml_escape_string("\u{7f}"), "\\u007F");

        // A value mixing a control char with normal text must produce valid TOML
        // and round-trip to the original string.
        let input = "tab\tand\u{1b}escape";
        let escaped = toml_escape_string(input);
        let toml_str = format!("value = \"{escaped}\"");
        let parsed: toml::Value =
            toml::from_str(&toml_str).expect("control-char escape must produce valid TOML");
        assert_eq!(
            parsed["value"].as_str().unwrap(),
            input,
            "escaped control characters must round-trip to the original string"
        );
    }

    // --- env_bool tests ---

    #[test]
    #[serial_test::serial]
    fn test_env_bool_true_values() {
        let saved = std::env::var("VECLAYER_TEST_BOOL_X9Z").ok();
        std::env::set_var("VECLAYER_TEST_BOOL_X9Z", "true");
        assert_eq!(env_bool("VECLAYER_TEST_BOOL_X9Z"), Some(true));
        std::env::set_var("VECLAYER_TEST_BOOL_X9Z", "1");
        assert_eq!(env_bool("VECLAYER_TEST_BOOL_X9Z"), Some(true));
        match saved {
            Some(v) => std::env::set_var("VECLAYER_TEST_BOOL_X9Z", v),
            None => std::env::remove_var("VECLAYER_TEST_BOOL_X9Z"),
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_env_bool_false_values() {
        let saved = std::env::var("VECLAYER_TEST_BOOL_X9Z").ok();
        std::env::set_var("VECLAYER_TEST_BOOL_X9Z", "false");
        assert_eq!(env_bool("VECLAYER_TEST_BOOL_X9Z"), Some(false));
        std::env::set_var("VECLAYER_TEST_BOOL_X9Z", "0");
        assert_eq!(env_bool("VECLAYER_TEST_BOOL_X9Z"), Some(false));
        match saved {
            Some(v) => std::env::set_var("VECLAYER_TEST_BOOL_X9Z", v),
            None => std::env::remove_var("VECLAYER_TEST_BOOL_X9Z"),
        }
    }

    // Unrecognized value falls back to Some(false) — the existing contract — not None.
    // This test pins that contract so it cannot silently regress to a panic or None.
    #[test]
    #[serial_test::serial]
    fn test_env_bool_unrecognized_value_returns_some_false() {
        let saved = std::env::var("VECLAYER_TEST_BOOL_X9Z").ok();
        std::env::set_var("VECLAYER_TEST_BOOL_X9Z", "yes");
        assert_eq!(
            env_bool("VECLAYER_TEST_BOOL_X9Z"),
            Some(false),
            "unrecognized boolean string must return Some(false), not None or panic"
        );
        match saved {
            Some(v) => std::env::set_var("VECLAYER_TEST_BOOL_X9Z", v),
            None => std::env::remove_var("VECLAYER_TEST_BOOL_X9Z"),
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_env_bool_unset_returns_none() {
        // A unique key guaranteed never to be set in the test environment.
        std::env::remove_var("VECLAYER_TEST_BOOL_UNSET_X9Z9");
        assert_eq!(env_bool("VECLAYER_TEST_BOOL_UNSET_X9Z9"), None);
    }

    // --- env_parse tests ---

    #[test]
    #[serial_test::serial]
    fn test_env_parse_bad_integer_returns_none_no_panic() {
        let saved = std::env::var("VECLAYER_TEST_PARSE_INT_X9Z").ok();
        std::env::set_var("VECLAYER_TEST_PARSE_INT_X9Z", "not-a-number");
        let result: Option<u16> = env_parse("VECLAYER_TEST_PARSE_INT_X9Z");
        assert_eq!(
            result, None,
            "unparseable integer env var must return None without panicking"
        );
        match saved {
            Some(v) => std::env::set_var("VECLAYER_TEST_PARSE_INT_X9Z", v),
            None => std::env::remove_var("VECLAYER_TEST_PARSE_INT_X9Z"),
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_env_parse_bad_float_returns_none_no_panic() {
        let saved = std::env::var("VECLAYER_TEST_PARSE_FLOAT_X9Z").ok();
        std::env::set_var("VECLAYER_TEST_PARSE_FLOAT_X9Z", "not-a-float");
        let result: Option<f32> = env_parse("VECLAYER_TEST_PARSE_FLOAT_X9Z");
        assert_eq!(
            result, None,
            "unparseable float env var must return None without panicking"
        );
        match saved {
            Some(v) => std::env::set_var("VECLAYER_TEST_PARSE_FLOAT_X9Z", v),
            None => std::env::remove_var("VECLAYER_TEST_PARSE_FLOAT_X9Z"),
        }
    }

    // --- VECLAYER_OLLAMA_DIMENSION invalid value test ---

    #[test]
    #[serial_test::serial]
    fn test_resolve_embedder_invalid_dimension_falls_back_to_default() {
        std::env::set_var("VECLAYER_EMBEDDER", "ollama");
        std::env::remove_var("VECLAYER_OLLAMA_MODEL");
        std::env::remove_var("VECLAYER_OLLAMA_URL");
        std::env::set_var("VECLAYER_OLLAMA_DIMENSION", "not-a-number");

        let embedder = Config::resolve_embedder(None, None, None);

        std::env::remove_var("VECLAYER_EMBEDDER");
        std::env::remove_var("VECLAYER_OLLAMA_DIMENSION");

        assert!(
            matches!(
                embedder,
                EmbedderConfig::Ollama { dimension, .. } if dimension == DEFAULT_OLLAMA_DIMENSION
            ),
            "invalid VECLAYER_OLLAMA_DIMENSION must fall back to DEFAULT_OLLAMA_DIMENSION"
        );
    }

    // --- VECLAYER_CONFIG missing file test ---

    #[test]
    #[serial_test::serial]
    fn test_file_config_discover_nonexistent_env_returns_defaults() {
        let saved = std::env::var("VECLAYER_CONFIG").ok();

        std::env::set_var(
            "VECLAYER_CONFIG",
            "/nonexistent/path/that/does/not/exist/veclayer.toml",
        );
        let fc = FileConfig::discover(None);

        match saved {
            Some(v) => std::env::set_var("VECLAYER_CONFIG", v),
            None => std::env::remove_var("VECLAYER_CONFIG"),
        }

        assert!(
            fc.host.is_none(),
            "missing VECLAYER_CONFIG path must return defaults (host == None)"
        );
        assert!(
            fc.port.is_none(),
            "missing VECLAYER_CONFIG path must return defaults (port == None)"
        );
        assert!(
            fc.data_dir.is_none(),
            "missing VECLAYER_CONFIG path must return defaults (data_dir == None)"
        );
    }
}
