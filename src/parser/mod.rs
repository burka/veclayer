//! Document parsing traits and implementations.
//!
//! [`DocumentParser`] is the core trait: implementations receive raw document
//! text and emit a flat list of [`HierarchicalChunk`]s with parent-child IDs
//! already wired. [`MarkdownParser`] handles Markdown (`.md`) files.

mod markdown;

pub use markdown::MarkdownParser;

use crate::{HierarchicalChunk, Result};
use std::path::Path;

/// Trait for parsing documents into hierarchical chunks.
/// Different implementations handle different document formats.
pub trait DocumentParser: Send + Sync {
    /// Parse a document from its content string.
    /// Returns a vector of hierarchical chunks with parent-child relationships.
    fn parse(&self, content: &str, source_file: &str) -> Result<Vec<HierarchicalChunk>>;

    /// Parse a document from a file path.
    fn parse_file(&self, path: &Path) -> Result<Vec<HierarchicalChunk>> {
        let content = std::fs::read_to_string(path)?;
        let source = path.to_string_lossy().to_string();
        self.parse(&content, &source)
    }

    /// Get the file extensions this parser supports.
    fn supported_extensions(&self) -> &[&str];

    /// Check if this parser can handle a given file.
    fn can_parse(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| {
                self.supported_extensions()
                    .iter()
                    .any(|supported| supported.eq_ignore_ascii_case(ext))
            })
            .unwrap_or(false)
    }
}

/// Box type for dynamic dispatch of parsers
pub type BoxedParser = Box<dyn DocumentParser>;

// Implement DocumentParser for Arc<T> where T: DocumentParser
crate::arc_impl!(DocumentParser {
    fn parse(&self, content: &str, source_file: &str) -> Result<Vec<HierarchicalChunk>>;
    fn parse_file(&self, path: &Path) -> Result<Vec<HierarchicalChunk>>;
    fn supported_extensions(&self) -> &[&str];
    fn can_parse(&self, path: &Path) -> bool;
});

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Simple test parser for testing trait defaults
    struct TestParser;

    impl DocumentParser for TestParser {
        fn parse(&self, content: &str, source_file: &str) -> Result<Vec<HierarchicalChunk>> {
            Ok(vec![HierarchicalChunk::new(
                content.to_string(),
                crate::ChunkLevel::CONTENT,
                None,
                "test".to_string(),
                source_file.to_string(),
            )])
        }

        fn supported_extensions(&self) -> &[&str] {
            &["txt", "test"]
        }
    }

    #[test]
    fn test_can_parse_supported_extension() {
        let parser = TestParser;
        assert!(parser.can_parse(Path::new("file.txt")));
        assert!(parser.can_parse(Path::new("file.test")));
        assert!(parser.can_parse(Path::new("file.TXT"))); // case insensitive
    }

    #[test]
    fn test_can_parse_unsupported_extension() {
        let parser = TestParser;
        assert!(!parser.can_parse(Path::new("file.md")));
        assert!(!parser.can_parse(Path::new("file.rs")));
    }

    #[test]
    fn test_can_parse_no_extension() {
        let parser = TestParser;
        assert!(!parser.can_parse(Path::new("README")));
    }

    #[test]
    fn test_parse_file() {
        let parser = TestParser;

        // Create temp file
        let mut temp = NamedTempFile::new().unwrap();
        writeln!(temp, "Hello, World!").unwrap();

        let chunks = parser.parse_file(temp.path()).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("Hello"));
    }

    #[test]
    fn test_parse_file_not_found() {
        let parser = TestParser;
        let result = parser.parse_file(Path::new("/nonexistent/file.txt"));
        assert!(result.is_err());
    }
}
