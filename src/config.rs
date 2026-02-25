//! Configuration with 12-factor layered resolution: ENV > TOML file > Defaults.
//!
//! Config file lookup order:
//! 1. `$VECLAYER_CONFIG` (explicit path)
//! 2. `<data_dir>/veclayer.toml`
//! 3. `./veclayer.toml`
//!
//! Any field set in the environment always wins over the config file.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use tracing::warn;

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
        let api_key = std::env::var("VECLAYER_LLM_API_KEY")
            .ok()
            .or_else(|| file_llm.as_ref().and_then(|l| l.api_key.clone()));
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
#[derive(Debug, Clone)]
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
    /// Project name for memory isolation
    pub project: Option<String>,
}

/// Walk up from `start_dir` looking for a `.veclayer/` directory.
/// Returns `(data_dir, project_config)` if found.
pub fn discover_project(start_dir: &Path) -> Option<(PathBuf, ProjectConfig)> {
    let mut dir = start_dir;
    loop {
        let candidate = dir.join(".veclayer");
        if candidate.is_dir() {
            let config_path = candidate.join("config.toml");
            let project_config = if config_path.exists() {
                match std::fs::read_to_string(&config_path) {
                    Ok(contents) => toml::from_str(&contents).unwrap_or_default(),
                    Err(_) => ProjectConfig::default(),
                }
            } else {
                ProjectConfig::default()
            };
            return Some((candidate, project_config));
        }
        dir = dir.parent()?;
    }
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
}
