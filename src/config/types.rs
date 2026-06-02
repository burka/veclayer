//! Configuration data types: structs, enums, and their impls.

use std::path::PathBuf;

use serde::Deserialize;

#[cfg(feature = "config")]
use std::path::Path;
#[cfg(feature = "config")]
use tracing::warn;

/// Glob matching options used for path patterns in `[[match]]` overrides.
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
pub(super) struct FileConfig {
    pub(super) data_dir: Option<String>,
    pub(super) host: Option<String>,
    pub(super) port: Option<u16>,
    pub(super) read_only: Option<bool>,
    pub(super) hooks_enabled: Option<bool>,
    pub(super) search_top_k: Option<usize>,
    pub(super) search_children_k: Option<usize>,
    pub(super) embedder: Option<FileEmbedderConfig>,
    pub(super) llm: Option<FileLlmConfig>,
    pub(super) auth: Option<FileAuthConfig>,
}

#[cfg(feature = "config")]
#[derive(Debug, Deserialize)]
pub(super) struct FileAuthConfig {
    pub(super) auth_required: Option<bool>,
    pub(super) server_url: Option<String>,
    pub(super) token_expiry_secs: Option<u64>,
    pub(super) refresh_expiry_secs: Option<u64>,
    pub(super) auto_approve: Option<bool>,
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
pub(super) struct FileLlmConfig {
    /// "ollama" or "openai"
    #[serde(default = "default_llm_provider")]
    pub(super) provider: String,
    pub(super) model: Option<String>,
    pub(super) base_url: Option<String>,
    pub(super) api_key: Option<String>,
    pub(super) temperature: Option<f32>,
    pub(super) max_tokens: Option<usize>,
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
pub(super) struct FileEmbedderConfig {
    /// "fastembed" or "ollama"
    #[serde(rename = "type", default = "default_embedder_type")]
    pub(super) embedder_type: String,
    pub(super) model: Option<String>,
    pub(super) base_url: Option<String>,
    /// Embedding vector dimension (required for ollama; ignored for fastembed)
    pub(super) dimension: Option<usize>,
}

#[cfg(feature = "config")]
fn default_embedder_type() -> String {
    "fastembed".to_string()
}

#[cfg(feature = "config")]
impl FileConfig {
    /// Try to load from a TOML file. Returns default (all-None) on any error.
    pub(super) fn load(path: &Path) -> Self {
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
    pub(super) fn discover(data_dir_hint: Option<&Path>) -> Self {
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
pub(super) const DEFAULT_HOST: &str = "127.0.0.1";
#[cfg(feature = "config")]
pub(super) const DEFAULT_PORT: u16 = 8080;
#[cfg(feature = "config")]
pub(super) const DEFAULT_SEARCH_TOP_K: usize = 5;
#[cfg(feature = "config")]
pub(super) const DEFAULT_SEARCH_CHILDREN_K: usize = 3;
pub(super) const DEFAULT_FASTEMBED_MODEL: &str = "Xenova/bge-small-en-v1.5";
#[cfg(feature = "config")]
pub(super) const DEFAULT_OLLAMA_MODEL: &str = crate::util::DEFAULT_OLLAMA_EMBED_MODEL;
#[cfg(feature = "config")]
pub(super) const DEFAULT_OLLAMA_URL: &str = crate::util::DEFAULT_OLLAMA_URL;
#[cfg(feature = "config")]
pub(super) const DEFAULT_OLLAMA_DIMENSION: usize = crate::util::DEFAULT_OLLAMA_DIMENSION;

/// Summarised result of Ollama auto-discovery, used inside `Config::new()`.
///
/// Kept in a plain struct (no feature gate) so the same type can be threaded
/// through both the `llm`-gated detection path and the always-present
/// `resolve_embedder` / `resolve_llm` signatures without conditional compilation
/// complexity at the call sites.
#[cfg(feature = "config")]
pub(super) struct DetectedOllama {
    pub(super) base_url: String,
    /// Best available embedding model, or `None` if Ollama has no embed models.
    pub(super) embed_model: Option<String>,
    /// Best available chat model, or `None` if Ollama has no chat models.
    pub(super) chat_model: Option<String>,
}

/// Summarised result of OpenAI-compatible embedding-service auto-discovery
/// (vLLM, HuggingFace TEI, …), used inside `Config::new()`.
///
/// Unlike [`DetectedOllama`] this always carries a concrete `dimension`, learned
/// from a live `/v1/embeddings` probe, so the store is sized to the served model
/// instead of the Ollama default.
#[cfg(feature = "config")]
pub(super) struct DetectedOpenAiEmbed {
    pub(super) base_url: String,
    pub(super) model: String,
    pub(super) dimension: usize,
}

/// True when the user has explicitly pinned the embedder, by any means:
/// the `VECLAYER_EMBEDDER` type selector, a `[embedder]` TOML block, or any of
/// the Ollama embedder overrides (`VECLAYER_OLLAMA_URL/MODEL/DIMENSION`).
///
/// Auto-discovery (Ollama *and* OpenAI-compatible) is suppressed in all of these
/// cases. Counting the `VECLAYER_OLLAMA_*` vars here is what prevents a probe
/// from injecting a model/dimension that conflicts with a user-pinned endpoint.
#[cfg(feature = "config")]
pub(super) fn embedder_explicitly_set(file_embedder: &Option<FileEmbedderConfig>) -> bool {
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

    pub(super) fn resolve_embedder(
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

    pub(super) fn resolve_llm(
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

    pub(super) fn resolve_auth(file_auth: Option<FileAuthConfig>) -> AuthConfig {
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

// --- Helpers for ENV > TOML > Default resolution (only used by Config::new()) ---

#[cfg(feature = "config")]
pub(super) fn env_or(key: &str, file_val: Option<String>, default: String) -> String {
    std::env::var(key).ok().or(file_val).unwrap_or(default)
}

#[cfg(feature = "config")]
pub(super) fn env_parse<T: std::str::FromStr>(key: &str) -> Option<T> {
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
pub(super) fn env_bool(key: &str) -> Option<bool> {
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
