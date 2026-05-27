use pulldown_cmark::{Event, HeadingLevel, Parser, Tag, TagEnd};

use super::DocumentParser;
use crate::{ChunkLevel, HierarchicalChunk, Result};

/// Parser for Markdown documents using pulldown-cmark.
/// Extracts hierarchical structure based on headings (H1-H6).
pub struct MarkdownParser {
    /// Minimum content length to create a chunk
    min_chunk_size: usize,
}

impl MarkdownParser {
    pub fn new() -> Self {
        Self { min_chunk_size: 10 }
    }

    pub fn with_min_chunk_size(mut self, size: usize) -> Self {
        self.min_chunk_size = size;
        self
    }

    fn heading_level_to_chunk_level(level: HeadingLevel) -> ChunkLevel {
        match level {
            HeadingLevel::H1 => ChunkLevel::H1,
            HeadingLevel::H2 => ChunkLevel::H2,
            HeadingLevel::H3 => ChunkLevel::H3,
            HeadingLevel::H4 => ChunkLevel::H4,
            HeadingLevel::H5 => ChunkLevel::H5,
            HeadingLevel::H6 => ChunkLevel::H6,
        }
    }
}

impl Default for MarkdownParser {
    fn default() -> Self {
        Self::new()
    }
}

/// State for tracking the parsing context
struct ParseState {
    /// Stack of parent chunks at each heading level
    heading_stack: Vec<(ChunkLevel, String, String)>, // (level, id, heading_text)

    /// Currently accumulating text
    current_text: String,

    /// Current heading text being parsed
    current_heading: Option<String>,

    /// Are we inside a heading?
    in_heading: bool,

    /// Current heading level being parsed
    current_heading_level: Option<ChunkLevel>,

    /// All chunks produced
    chunks: Vec<HierarchicalChunk>,

    /// Source file name
    source_file: String,
}

impl ParseState {
    fn new(source_file: String) -> Self {
        Self {
            heading_stack: Vec::new(),
            current_text: String::new(),
            current_heading: None,
            in_heading: false,
            current_heading_level: None,
            chunks: Vec::new(),
            source_file,
        }
    }

    /// Get the current parent ID based on the heading stack
    fn current_parent_id(&self) -> Option<String> {
        self.heading_stack.last().map(|(_, id, _)| id.clone())
    }

    /// Get the current path based on the heading stack
    fn current_path(&self) -> String {
        if self.heading_stack.is_empty() {
            return String::new();
        }
        self.heading_stack
            .iter()
            .map(|(_, _, text)| text.as_str())
            .collect::<Vec<_>>()
            .join(" > ")
    }

    /// Pop headings from the stack that are at the same or lower level
    fn pop_to_level(&mut self, level: ChunkLevel) {
        while let Some((stack_level, _, _)) = self.heading_stack.last() {
            if stack_level.depth() >= level.depth() {
                self.heading_stack.pop();
            } else {
                break;
            }
        }
    }

    /// Flush any accumulated content as a content chunk
    fn flush_content(&mut self, min_size: usize) {
        let content = self.current_text.trim().to_string();
        if content.len() >= min_size {
            let path = self.current_path();
            let parent_id = self.current_parent_id();

            let chunk = HierarchicalChunk::new(
                content,
                ChunkLevel::CONTENT,
                parent_id,
                path,
                self.source_file.clone(),
            );
            self.chunks.push(chunk);
        }
        self.current_text.clear();
    }

    /// Create a heading chunk and push it onto the stack
    fn create_heading_chunk(&mut self, level: ChunkLevel, heading: String) {
        // Pop any headings at the same or lower level
        self.pop_to_level(level);

        let path = if self.heading_stack.is_empty() {
            heading.clone()
        } else {
            format!("{} > {}", self.current_path(), heading)
        };
        let parent_id = self.current_parent_id();

        let chunk = HierarchicalChunk::new(
            heading.clone(),
            level,
            parent_id,
            path,
            self.source_file.clone(),
        )
        .with_heading(heading.clone());

        let chunk_id = chunk.id.clone();
        self.chunks.push(chunk);

        // Push onto the stack
        self.heading_stack.push((level, chunk_id, heading));
    }
}

impl DocumentParser for MarkdownParser {
    fn parse(&self, content: &str, source_file: &str) -> Result<Vec<HierarchicalChunk>> {
        let parser = Parser::new(content);
        let mut state = ParseState::new(source_file.to_string());

        for event in parser {
            match event {
                Event::Start(Tag::Heading { level, .. }) => {
                    // Flush any accumulated content before the heading
                    state.flush_content(self.min_chunk_size);
                    state.in_heading = true;
                    state.current_heading_level =
                        Some(MarkdownParser::heading_level_to_chunk_level(level));
                    state.current_heading = Some(String::new());
                }

                Event::End(TagEnd::Heading(_)) => {
                    if let (Some(level), Some(heading)) = (
                        state.current_heading_level.take(),
                        state.current_heading.take(),
                    ) {
                        let heading = heading.trim().to_string();
                        if !heading.is_empty() {
                            state.create_heading_chunk(level, heading);
                        }
                    }
                    state.in_heading = false;
                }

                Event::Text(text) | Event::Code(text) => {
                    if state.in_heading {
                        if let Some(ref mut heading) = state.current_heading {
                            heading.push_str(&text);
                        }
                    } else {
                        state.current_text.push_str(&text);
                        state.current_text.push(' ');
                    }
                }

                Event::SoftBreak | Event::HardBreak if !state.in_heading => {
                    state.current_text.push(' ');
                }

                Event::End(TagEnd::Paragraph) if !state.in_heading => {
                    state.current_text.push('\n');
                }

                Event::End(TagEnd::CodeBlock) => {
                    state.current_text.push('\n');
                }

                Event::Start(Tag::CodeBlock(_)) => {
                    state.current_text.push_str("\n```\n");
                }

                _ => {}
            }
        }

        // Flush any remaining content
        state.flush_content(self.min_chunk_size);

        Ok(state.chunks)
    }

    fn supported_extensions(&self) -> &[&str] {
        &["md", "markdown"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_markdown() {
        let parser = MarkdownParser::new();
        let content = r#"
# Chapter 1

This is the introduction.

## Section 1.1

Details about section 1.1.

## Section 1.2

Details about section 1.2.

# Chapter 2

Content for chapter 2.
"#;

        let chunks = parser.parse(content, "test.md").unwrap();

        // Should have: H1 Chapter 1, content, H2 Section 1.1, content, H2 Section 1.2, content, H1 Chapter 2, content
        assert!(!chunks.is_empty());

        // Find Chapter 1
        let chapter1 = chunks.iter().find(|c| c.content == "Chapter 1").unwrap();
        assert_eq!(chapter1.level, ChunkLevel::H1);
        assert!(chapter1.parent_id.is_none());

        // Find Section 1.1
        let section11 = chunks.iter().find(|c| c.content == "Section 1.1").unwrap();
        assert_eq!(section11.level, ChunkLevel::H2);
        assert_eq!(section11.parent_id, Some(chapter1.id.clone()));
        assert_eq!(section11.path, "Chapter 1 > Section 1.1");
    }

    #[test]
    fn test_deep_nesting() {
        let parser = MarkdownParser::new();
        let content = r#"
# H1

## H2

### H3

#### H4

Deep content here.
"#;

        let chunks = parser.parse(content, "test.md").unwrap();

        let h4 = chunks.iter().find(|c| c.content == "H4").unwrap();
        assert_eq!(h4.level, ChunkLevel::H4);
        assert_eq!(h4.path, "H1 > H2 > H3 > H4");
    }

    #[test]
    fn test_heading_level_jump() {
        let parser = MarkdownParser::new();
        let content = r#"
# Chapter 1

## Section 1.1

# Chapter 2

## Section 2.1
"#;

        let chunks = parser.parse(content, "test.md").unwrap();

        // Section 2.1 should have Chapter 2 as parent, not Section 1.1
        let chapter2 = chunks.iter().find(|c| c.content == "Chapter 2").unwrap();
        let section21 = chunks.iter().find(|c| c.content == "Section 2.1").unwrap();
        assert_eq!(section21.parent_id, Some(chapter2.id.clone()));
    }

    // --- Edge / robustness tests ---

    #[test]
    fn test_empty_string_yields_empty_chunk_list() {
        // The flush_content min-size guard (default 10) filters out empty/tiny
        // strings, so an empty document must produce zero chunks.
        let parser = MarkdownParser::new();
        let chunks = parser.parse("", "f.md").unwrap();
        assert!(
            chunks.is_empty(),
            "expected empty chunk list for empty input, got {} chunk(s)",
            chunks.len()
        );
    }

    #[test]
    fn test_whitespace_only_yields_empty_chunk_list() {
        // Whitespace-only input has no headings, and its body trims to the
        // empty string (length 0 < min_chunk_size), so the result must be empty.
        let parser = MarkdownParser::new();
        let inputs = ["   ", "\n", "\n\n\n", "  \t  \n  "];
        for input in &inputs {
            let chunks = parser.parse(input, "f.md").unwrap();
            assert!(
                chunks.is_empty(),
                "expected empty chunk list for whitespace-only input {:?}, got {} chunk(s)",
                input,
                chunks.len()
            );
        }
    }

    #[test]
    fn test_heading_only_document_no_body_chunk() {
        // A document consisting solely of "# Title\n" has one heading and no
        // body text.  The parser should emit exactly one chunk:
        //   - level  == H1
        //   - content == "Title"
        //   - heading == Some("Title")
        //   - parent_id is None  (root-level)
        // No CONTENT-level chunk should appear because there is no body text.
        let parser = MarkdownParser::new();
        let chunks = parser.parse("# Title\n", "f.md").unwrap();

        assert_eq!(
            chunks.len(),
            1,
            "expected exactly 1 chunk for heading-only document, got {}: {:?}",
            chunks.len(),
            chunks
                .iter()
                .map(|c| (&c.level, &c.content))
                .collect::<Vec<_>>()
        );

        let heading_chunk = &chunks[0];
        assert_eq!(
            heading_chunk.level,
            ChunkLevel::H1,
            "chunk level should be H1"
        );
        assert_eq!(
            heading_chunk.content, "Title",
            "chunk content should equal the heading text"
        );
        assert_eq!(
            heading_chunk.heading,
            Some("Title".to_string()),
            "heading field should be populated for a heading chunk"
        );
        assert!(
            heading_chunk.parent_id.is_none(),
            "root heading should have no parent"
        );
    }
}
