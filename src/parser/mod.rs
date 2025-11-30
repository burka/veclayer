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
