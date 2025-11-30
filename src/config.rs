use std::path::PathBuf;

/// Configuration for VecLayer, following 12-factor app principles
#[derive(Debug, Clone)]
pub struct Config {
    /// Directory where VecLayer stores its data (LanceDB files)
    pub data_dir: PathBuf,

    /// Embedder to use: "fastembed" or "ollama"
    pub embedder: EmbedderConfig,

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
}

#[derive(Debug, Clone)]
pub enum EmbedderConfig {
    FastEmbed { model: String },
    Ollama { model: String, base_url: String },
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from(
                std::env::var("VECLAYER_DATA_DIR")
                    .unwrap_or_else(|_| "./veclayer-data".to_string()),
            ),
            embedder: EmbedderConfig::default(),
            read_only: std::env::var("VECLAYER_READ_ONLY")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
            port: std::env::var("VECLAYER_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8080),
            host: std::env::var("VECLAYER_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
            search_top_k: std::env::var("VECLAYER_SEARCH_TOP_K")
                .ok()
                .and_then(|k| k.parse().ok())
                .unwrap_or(5),
            search_children_k: std::env::var("VECLAYER_SEARCH_CHILDREN_K")
                .ok()
                .and_then(|k| k.parse().ok())
                .unwrap_or(3),
        }
    }
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        let embedder =
            std::env::var("VECLAYER_EMBEDDER").unwrap_or_else(|_| "fastembed".to_string());

        match embedder.as_str() {
            "ollama" => EmbedderConfig::Ollama {
                model: std::env::var("VECLAYER_OLLAMA_MODEL")
                    .unwrap_or_else(|_| "nomic-embed-text".to_string()),
                base_url: std::env::var("VECLAYER_OLLAMA_URL")
                    .unwrap_or_else(|_| "http://localhost:11434".to_string()),
            },
            _ => EmbedderConfig::FastEmbed {
                model: std::env::var("VECLAYER_FASTEMBED_MODEL")
                    .unwrap_or_else(|_| "BAAI/bge-small-en-v1.5".to_string()),
            },
        }
    }
}

impl Config {
    pub fn new() -> Self {
        Self::default()
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
}
