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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_parse() {
        let err = Error::parse("test error");
        assert!(matches!(err, Error::Parse(_)));
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_error_embedding() {
        let err = Error::embedding("embed failed");
        assert!(matches!(err, Error::Embedding(_)));
    }

    #[test]
    fn test_error_store() {
        let err = Error::store("store error");
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn test_error_search() {
        let err = Error::search("search failed");
        assert!(matches!(err, Error::Search(_)));
    }

    #[test]
    fn test_error_config() {
        let err = Error::config("invalid config");
        assert!(matches!(err, Error::Config(_)));
    }

    #[test]
    fn test_error_not_found() {
        let err = Error::not_found("item missing");
        assert!(matches!(err, Error::NotFound(_)));
    }

    #[test]
    fn test_error_summarization() {
        let err = Error::summarization("summary failed");
        assert!(matches!(err, Error::Summarization(_)));
    }

    #[test]
    fn test_error_clustering() {
        let err = Error::clustering("cluster error");
        assert!(matches!(err, Error::Clustering(_)));
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Io(_)));
    }

    #[test]
    fn test_error_display() {
        let err = Error::parse("test message");
        let display = format!("{}", err);
        assert!(display.contains("Parsing error"));
        assert!(display.contains("test message"));
    }
}
