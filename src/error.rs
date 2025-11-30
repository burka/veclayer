use thiserror::Error;

/// Result type for VecLayer operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur in VecLayer
#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parsing error: {0}")]
    Parse(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Vector store error: {0}")]
    Store(String),

    #[error("Search error: {0}")]
    Search(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Summarization error: {0}")]
    Summarization(String),

    #[error("Clustering error: {0}")]
    Clustering(String),
}

impl Error {
    pub fn parse(msg: impl Into<String>) -> Self {
        Self::Parse(msg.into())
    }

    pub fn embedding(msg: impl Into<String>) -> Self {
        Self::Embedding(msg.into())
    }

    pub fn store(msg: impl Into<String>) -> Self {
        Self::Store(msg.into())
    }

    pub fn search(msg: impl Into<String>) -> Self {
        Self::Search(msg.into())
    }

    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }

    pub fn summarization(msg: impl Into<String>) -> Self {
        Self::Summarization(msg.into())
    }

    pub fn clustering(msg: impl Into<String>) -> Self {
        Self::Clustering(msg.into())
    }
}
