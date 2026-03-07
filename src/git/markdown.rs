//! Markdown + YAML frontmatter serializer/deserializer for git memory branch entries.
//!
//! Converts between [`Entry`] (the canonical in-memory type) and the on-disk
//! Markdown format used on the memory branch:
//!
//! ```text
//! ---
//! id: 2b0bd4c
//! created: 2026-02-21T10:15:00Z
//! perspectives: [decisions, knowledge]
//! visibility: always
//! relations:
//!   derived_from: 6c121fe
//!   supersedes: a3f8812
//! parent: bf81639
//! expires: 2026-04-04T09:00:00Z
//! impression: Search is slow
//! impression_strength: 0.6
//! summarizes: [6c121fe, 4a2b3c1]
//! ---
//! # Heading
//!
//! Body text...
//! ```
//!
//! **Rules:**
//! - `id` is the 7-char short hash of content.
//! - Fields at their defaults are omitted (empty perspectives, "normal" visibility, etc.).
//! - Relations use compact form: single target → string, multiple → array.
//! - `entry_type` is omitted when `raw` (the default). Old files without the field parse as `raw`.
//! - `level` is omitted when `h1` (the default for headed entries). Old files without the field
//!   fall back to heading-based inference: heading present → H1, no heading → Content.

use std::collections::BTreeMap;

use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::chunk::visibility::NORMAL;
use crate::chunk::{short_id, ChunkLevel, ChunkRelation, EntryType};
use crate::entry::Entry;
use crate::git::GitError;

/// Source label for entries loaded from the git memory branch.
const GIT_SOURCE: &str = "[git]";

// ---------------------------------------------------------------------------
// StringOrVec — compact single/multi relation target
// ---------------------------------------------------------------------------

/// A YAML field that can be either a single string or a list of strings.
///
/// Serialized as a bare string when there is exactly one element,
/// and as a YAML sequence when there are multiple.
#[derive(Debug, Clone, PartialEq)]
struct StringOrVec(Vec<String>);

impl StringOrVec {
    fn single(s: impl Into<String>) -> Self {
        Self(vec![s.into()])
    }

    fn multiple(v: Vec<String>) -> Self {
        Self(v)
    }
}

impl Serialize for StringOrVec {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.0.len() == 1 {
            serializer.serialize_str(&self.0[0])
        } else {
            self.0.serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for StringOrVec {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;

        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = StringOrVec;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a string or a sequence of strings")
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(StringOrVec::single(v))
            }

            fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> {
                Ok(StringOrVec::single(v))
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element::<String>()? {
                    items.push(item);
                }
                Ok(StringOrVec::multiple(items))
            }
        }

        deserializer.deserialize_any(Visitor)
    }
}

// ---------------------------------------------------------------------------
// Frontmatter helpers
// ---------------------------------------------------------------------------

fn is_normal_visibility(v: &Option<String>) -> bool {
    match v {
        Some(s) => s == NORMAL,
        None => true,
    }
}

fn skip_impression_strength(v: &Option<f32>) -> bool {
    match v {
        // Use exact equality: values very close to but not equal to 1.0
        // (e.g. from float arithmetic) should still be serialized.
        Some(s) => *s == 1.0_f32,
        None => true,
    }
}

fn is_raw_entry_type(t: &EntryType) -> bool {
    *t == EntryType::Raw
}

fn is_h1_level(l: &FrontmatterLevel) -> bool {
    *l == FrontmatterLevel::H1
}

/// Human-readable level representation for YAML frontmatter.
///
/// `H1` is the default (omitted when serializing) for backwards compatibility
/// with old files that relied on heading-based inference. Variants mirror the
/// `ChunkLevel` constants that are meaningful for git-stored entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum FrontmatterLevel {
    #[default]
    H1,
    H2,
    H3,
    H4,
    H5,
    H6,
    Content,
}

impl From<ChunkLevel> for FrontmatterLevel {
    fn from(level: ChunkLevel) -> Self {
        match level.0 {
            1 => Self::H1,
            2 => Self::H2,
            3 => Self::H3,
            4 => Self::H4,
            5 => Self::H5,
            6 => Self::H6,
            _ => Self::Content,
        }
    }
}

impl From<FrontmatterLevel> for ChunkLevel {
    fn from(level: FrontmatterLevel) -> Self {
        match level {
            FrontmatterLevel::H1 => ChunkLevel::H1,
            FrontmatterLevel::H2 => ChunkLevel::H2,
            FrontmatterLevel::H3 => ChunkLevel::H3,
            FrontmatterLevel::H4 => ChunkLevel::H4,
            FrontmatterLevel::H5 => ChunkLevel::H5,
            FrontmatterLevel::H6 => ChunkLevel::H6,
            FrontmatterLevel::Content => ChunkLevel::CONTENT,
        }
    }
}

// ---------------------------------------------------------------------------
// Frontmatter struct
// ---------------------------------------------------------------------------

/// YAML frontmatter parsed from / serialized to the `---` block.
#[derive(Debug, Serialize, Deserialize)]
struct Frontmatter {
    id: String,
    created: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    perspectives: Vec<String>,
    #[serde(default, skip_serializing_if = "is_normal_visibility")]
    visibility: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    relations: BTreeMap<String, StringOrVec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parent: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expires: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    impression: Option<String>,
    #[serde(default, skip_serializing_if = "skip_impression_strength")]
    impression_strength: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    summarizes: Vec<String>,
    /// Entry type annotation. Omitted when `raw` (the default) for backwards
    /// compatibility — old files without this field parse as `raw`.
    #[serde(default, skip_serializing_if = "is_raw_entry_type")]
    entry_type: EntryType,
    /// Chunk level. Omitted when `h1` (the default) for backwards compatibility
    /// — old files without this field fall back to heading-based inference.
    #[serde(default, skip_serializing_if = "is_h1_level")]
    level: FrontmatterLevel,
}

// ---------------------------------------------------------------------------
// Timestamp helpers
// ---------------------------------------------------------------------------

fn unix_to_iso8601(unix: i64) -> Result<String, GitError> {
    let dt = Utc
        .timestamp_opt(unix, 0)
        .single()
        .ok_or_else(|| GitError::CommandFailed {
            command: "format datetime".to_string(),
            stderr: format!("unix timestamp {unix} out of representable range"),
            exit_code: 1,
        })?;
    Ok(dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
}

fn iso8601_to_unix(s: &str) -> Result<i64, GitError> {
    s.parse::<DateTime<Utc>>()
        .map(|dt| dt.timestamp())
        .map_err(|e| GitError::CommandFailed {
            command: "parse datetime".to_string(),
            stderr: format!("invalid datetime '{s}': {e}"),
            exit_code: 1,
        })
}

// ---------------------------------------------------------------------------
// Entry ↔ Frontmatter conversion
// ---------------------------------------------------------------------------

fn entry_to_frontmatter(entry: &Entry) -> Result<Frontmatter, GitError> {
    let id = short_id(&entry.content_id()).to_string();
    let created = unix_to_iso8601(entry.created_at)?;

    let visibility = if entry.visibility == NORMAL {
        None
    } else {
        Some(entry.visibility.clone())
    };

    let relations = relations_to_map(&entry.relations);

    let parent = entry.parent_id.as_deref().map(|p| short_id(p).to_string());

    let expires = entry.expires_at.map(unix_to_iso8601).transpose()?;

    // Impression strength is only meaningful when an impression hint exists.
    // Omit when at the default (exactly 1.0) or when there's no hint.
    let impression_strength =
        if entry.impression_hint.is_some() && entry.impression_strength != 1.0_f32 {
            Some(entry.impression_strength)
        } else {
            None
        };

    Ok(Frontmatter {
        id,
        created,
        perspectives: entry.perspectives.clone(),
        visibility,
        relations,
        parent,
        expires,
        impression: entry.impression_hint.clone(),
        impression_strength,
        summarizes: entry.summarizes.clone(),
        entry_type: entry.entry_type,
        level: FrontmatterLevel::from(entry.level),
    })
}

fn frontmatter_to_entry(fm: Frontmatter, body: &str) -> Result<Entry, GitError> {
    let created_at = iso8601_to_unix(&fm.created)?;

    let expires_at = fm.expires.as_deref().map(iso8601_to_unix).transpose()?;

    let (heading, raw_content) = extract_heading(body);
    let content = raw_content.trim_end().to_string();

    // Use the explicit entry_type when set. For old files that lack the field,
    // `serde(default)` produces `EntryType::Raw`; fall back to heuristic
    // inference so those files still parse correctly.
    let entry_type = if fm.entry_type != EntryType::Raw {
        fm.entry_type
    } else if fm.impression.is_some() {
        EntryType::Impression
    } else if !fm.summarizes.is_empty() {
        EntryType::Summary
    } else {
        EntryType::Raw
    };

    // Use the explicit level when set (not the default H1). For old files that
    // lack the field, `serde(default)` produces `FrontmatterLevel::H1`; fall
    // back to heading-based inference so those files still parse correctly.
    let level = if fm.level != FrontmatterLevel::H1 {
        ChunkLevel::from(fm.level)
    } else if heading.is_some() {
        ChunkLevel::H1
    } else {
        ChunkLevel::CONTENT
    };

    let relations = map_to_relations(fm.relations);

    let impression_strength = fm.impression_strength.unwrap_or(1.0);

    let visibility = fm.visibility.unwrap_or_else(|| NORMAL.to_string());

    Ok(Entry {
        content,
        entry_type,
        source: GIT_SOURCE.to_string(),
        created_at,
        perspectives: fm.perspectives,
        relations,
        summarizes: fm.summarizes,
        heading,
        parent_id: fm.parent,
        impression_hint: fm.impression,
        impression_strength,
        expires_at,
        visibility,
        level,
        path: String::new(),
    })
}

// ---------------------------------------------------------------------------
// Relation conversion helpers
// ---------------------------------------------------------------------------

fn relations_to_map(relations: &[ChunkRelation]) -> BTreeMap<String, StringOrVec> {
    let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for rel in relations {
        map.entry(rel.kind.clone())
            .or_default()
            .push(short_id(&rel.target_id).to_string());
    }
    map.into_iter()
        .map(|(kind, targets)| {
            let value = if targets.len() == 1 {
                StringOrVec::single(targets.into_iter().next().unwrap())
            } else {
                StringOrVec::multiple(targets)
            };
            (kind, value)
        })
        .collect()
}

fn map_to_relations(map: BTreeMap<String, StringOrVec>) -> Vec<ChunkRelation> {
    let mut relations = Vec::new();
    for (kind, targets) in map {
        for target_id in targets.0 {
            relations.push(ChunkRelation::new(kind.clone(), target_id));
        }
    }
    relations
}

// ---------------------------------------------------------------------------
// Heading extraction / insertion
// ---------------------------------------------------------------------------

/// Extract an H1 heading from the body if present.
///
/// Expects the body to start with `# Heading\n\n`. Returns `(Some(heading), rest)`
/// or `(None, body)` if no heading is found.
fn extract_heading(body: &str) -> (Option<String>, String) {
    if let Some(rest) = body.strip_prefix("# ") {
        if let Some(newline_pos) = rest.find('\n') {
            let heading = rest[..newline_pos].trim_end_matches('\r').to_string();
            let remaining = rest[newline_pos + 1..]
                .trim_start_matches(['\r', '\n'])
                .to_string();
            return (Some(heading), remaining);
        }
    }
    (None, body.to_string())
}

/// Prepend the heading to the body if present.
fn prepend_heading(heading: &Option<String>, content: &str) -> String {
    match heading {
        Some(h) => format!("# {h}\n\n{content}"),
        None => content.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse Markdown with YAML frontmatter into an [`Entry`] and raw body string.
///
/// The body is the text after the closing `---` marker. The heading (if present)
/// is stripped from the body and stored in `entry.heading`.
pub fn parse(markdown: &[u8]) -> Result<(Entry, String), GitError> {
    let text = std::str::from_utf8(markdown).map_err(|e| GitError::CommandFailed {
        command: "parse markdown".to_string(),
        stderr: format!("invalid UTF-8: {e}"),
        exit_code: 1,
    })?;

    // Normalize CRLF → LF so all downstream code sees uniform line endings.
    let normalized;
    let text = if text.contains('\r') {
        normalized = text.replace("\r\n", "\n");
        &normalized
    } else {
        text
    };

    let (yaml, body) = split_frontmatter(text)?;

    let fm: Frontmatter = serde_yml::from_str(yaml).map_err(|e| GitError::CommandFailed {
        command: "parse frontmatter".to_string(),
        stderr: format!("invalid YAML: {e}"),
        exit_code: 1,
    })?;

    let entry = frontmatter_to_entry(fm, body)?;
    Ok((entry, body.trim_end().to_string()))
}

/// Render an [`Entry`] as a Markdown string with YAML frontmatter.
pub fn render(entry: &Entry) -> Result<String, GitError> {
    let fm = entry_to_frontmatter(entry)?;
    let yaml = serde_yml::to_string(&fm).map_err(|e| GitError::CommandFailed {
        command: "render frontmatter".to_string(),
        stderr: format!("YAML serialization failed: {e}"),
        exit_code: 1,
    })?;
    let body = prepend_heading(&entry.heading, &entry.content);
    Ok(format!("---\n{yaml}---\n{body}"))
}

/// Generate the filename for an entry: `slug-shortid.md`.
///
/// Slug is derived from the heading (preferred) or the first ~50 chars of content.
/// When both are empty or yield only non-alphanumeric characters the file is named
/// `shortid.md` (no leading hyphen).
pub fn entry_filename(entry: &Entry) -> String {
    let source = entry.heading.as_deref().unwrap_or(&entry.content);
    let slug = slugify(source, 50);
    let hash = entry.content_id();
    let id = short_id(&hash);
    if slug.is_empty() {
        format!("{id}.md")
    } else {
        format!("{slug}-{id}.md")
    }
}

/// Return the directory for an entry based on its first perspective.
///
/// The perspective name is sanitized via `slugify` to ensure it is safe as a
/// filesystem directory component (no `/`, `\0`, spaces, or other special chars).
/// Falls back to `_unsorted` if no perspectives are set or if sanitization yields
/// an empty string.
pub fn entry_directory(entry: &Entry) -> String {
    match entry.perspectives.first() {
        Some(perspective) => {
            let sanitized = slugify(perspective, 64);
            if sanitized.is_empty() {
                "_unsorted".to_string()
            } else {
                sanitized
            }
        }
        None => "_unsorted".to_string(),
    }
}

/// Return the full relative path for an entry: `directory/filename`.
pub fn entry_path(entry: &Entry) -> String {
    format!("{}/{}", entry_directory(entry), entry_filename(entry))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Split a document into `(yaml, body)` at the `---` frontmatter delimiters.
///
/// CRLF is normalized to LF by the caller before this function is invoked,
/// so only LF line endings appear here.
///
/// # Delimiter assumption
///
/// The closing delimiter is found by searching for the literal byte sequence
/// `\n---\n`, which is the standard frontmatter convention used by Jekyll,
/// Hugo, and most static-site generators.  This approach assumes that no YAML
/// *value* inside the frontmatter block itself contains a bare `\n---\n`
/// sequence.  That assumption is safe in practice because `serde_yml` always
/// quotes or block-indents multi-line strings, so any value that contains
/// `---` on its own line will be wrapped in quotes or indented, preventing a
/// false match.
fn split_frontmatter(text: &str) -> Result<(&str, &str), GitError> {
    let text = text.trim_start();

    let after_open = text
        .strip_prefix("---\n")
        .ok_or_else(|| GitError::CommandFailed {
            command: "parse frontmatter".to_string(),
            stderr: "missing opening '---' delimiter".to_string(),
            exit_code: 1,
        })?;

    // Find the closing `---` delimiter. Handles both mid-file and end-of-file.
    if let Some(close_pos) = after_open.find("\n---\n") {
        let yaml = &after_open[..close_pos];
        let body = after_open[close_pos + "\n---\n".len()..].trim_start_matches('\n');
        Ok((yaml, body))
    } else if after_open.ends_with("\n---") {
        // Closing delimiter at EOF with no trailing newline
        let yaml_end = after_open.rfind("\n---").unwrap();
        let yaml = &after_open[..yaml_end];
        Ok((yaml, ""))
    } else {
        Err(GitError::CommandFailed {
            command: "parse frontmatter".to_string(),
            stderr: "missing closing '---' delimiter".to_string(),
            exit_code: 1,
        })
    }
}

/// Convert a string to a URL/filename-safe slug.
///
/// - Lowercased
/// - Non-alphanumeric chars → `-`
/// - Consecutive hyphens collapsed
/// - Leading/trailing hyphens trimmed
/// - Truncated to `max_chars`
fn slugify(s: &str, max_chars: usize) -> String {
    let lower = s.to_lowercase();
    let mut slug = String::with_capacity(lower.len());
    let mut last_was_hyphen = false;

    for ch in lower.chars() {
        if ch.is_alphanumeric() {
            slug.push(ch);
            last_was_hyphen = false;
        } else if !last_was_hyphen && !slug.is_empty() {
            slug.push('-');
            last_was_hyphen = true;
        }
    }

    // Trim trailing hyphen
    let slug = slug.trim_end_matches('-');

    // Truncate at max_chars on char boundaries (safe for Unicode)
    let char_count = slug.chars().count();
    if char_count <= max_chars {
        return slug.to_string();
    }

    let truncated: String = slug.chars().take(max_chars).collect();
    truncated.trim_end_matches('-').to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{ChunkLevel, ChunkRelation, EntryType};

    fn minimal_entry() -> Entry {
        Entry {
            content: "Just some content here.".to_string(),
            entry_type: EntryType::Raw,
            source: GIT_SOURCE.to_string(),
            created_at: 1_740_132_900, // 2026-02-21T10:15:00Z
            perspectives: vec![],
            relations: vec![],
            summarizes: vec![],
            heading: None,
            parent_id: None,
            impression_hint: None,
            impression_strength: 1.0,
            expires_at: None,
            visibility: "normal".to_string(),
            level: ChunkLevel::CONTENT,
            path: String::new(),
        }
    }

    fn full_entry() -> Entry {
        Entry {
            content: "Content body here.".to_string(),
            entry_type: EntryType::Raw,
            source: GIT_SOURCE.to_string(),
            created_at: 1_740_132_900,
            perspectives: vec!["decisions".to_string(), "knowledge".to_string()],
            relations: vec![
                ChunkRelation::new("derived_from", "6c121fe"),
                ChunkRelation::new("supersedes", "a3f8812"),
            ],
            summarizes: vec![],
            heading: Some("Dual ID Scheme".to_string()),
            parent_id: Some("bf81639longer".to_string()),
            impression_hint: None,
            impression_strength: 1.0,
            expires_at: Some(1_743_753_600), // 2025-04-04T09:00:00Z approx
            visibility: "always".to_string(),
            level: ChunkLevel::H1,
            path: String::new(),
        }
    }

    // -----------------------------------------------------------------------
    // render tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_render_minimal() {
        let entry = minimal_entry();
        let rendered = render(&entry).unwrap();

        assert!(rendered.starts_with("---\n"));
        assert!(rendered.contains("id:"));
        assert!(rendered.contains("created:"));
        // Defaults should be omitted
        assert!(!rendered.contains("perspectives:"));
        assert!(!rendered.contains("visibility:"));
        assert!(!rendered.contains("relations:"));
        assert!(!rendered.contains("parent:"));
        assert!(!rendered.contains("expires:"));
        assert!(!rendered.contains("impression:"));
        assert!(!rendered.contains("impression_strength:"));
        assert!(!rendered.contains("summarizes:"));
        // entry_type: raw is the default and must be omitted
        assert!(!rendered.contains("entry_type:"));
        // Body appears after closing ---
        assert!(rendered.contains("Just some content here."));
    }

    #[test]
    fn test_render_full() {
        let entry = full_entry();
        let rendered = render(&entry).unwrap();

        assert!(rendered.contains("id:"));
        assert!(rendered.contains("created:"));
        assert!(rendered.contains("perspectives:"));
        assert!(rendered.contains("decisions"));
        assert!(rendered.contains("knowledge"));
        assert!(rendered.contains("visibility: always"));
        assert!(rendered.contains("relations:"));
        assert!(rendered.contains("derived_from:"));
        assert!(rendered.contains("6c121fe"));
        assert!(rendered.contains("supersedes:"));
        assert!(rendered.contains("a3f8812"));
        assert!(rendered.contains("parent: bf81639"));
        assert!(rendered.contains("expires:"));
        // Body has heading
        assert!(rendered.contains("# Dual ID Scheme"));
        assert!(rendered.contains("Content body here."));
    }

    #[test]
    fn test_render_impression_strength_omitted_at_default() {
        let mut entry = minimal_entry();
        entry.impression_hint = Some("curious".to_string());
        entry.impression_strength = 1.0;
        let rendered = render(&entry).unwrap();
        assert!(!rendered.contains("impression_strength:"));
        assert!(rendered.contains("impression: curious"));
    }

    #[test]
    fn test_render_impression_strength_included_when_not_default() {
        let mut entry = minimal_entry();
        entry.impression_hint = Some("uncertain".to_string());
        entry.impression_strength = 0.6;
        let rendered = render(&entry).unwrap();
        assert!(rendered.contains("impression_strength:"));
        assert!(rendered.contains("0.6"));
    }

    #[test]
    fn test_render_meta_entry_type() {
        let mut entry = minimal_entry();
        entry.entry_type = EntryType::Meta;
        let rendered = render(&entry).unwrap();
        assert!(rendered.contains("entry_type: meta"));
    }

    #[test]
    fn test_render_summary_entry_type() {
        let mut entry = minimal_entry();
        entry.entry_type = EntryType::Summary;
        let rendered = render(&entry).unwrap();
        assert!(rendered.contains("entry_type: summary"));
    }

    // -----------------------------------------------------------------------
    // parse tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_minimal() {
        let markdown = b"---\nid: abc1234\ncreated: 2026-02-21T10:15:00Z\n---\nJust content.\n";
        let (entry, body) = parse(markdown).unwrap();

        assert_eq!(entry.source, "[git]");
        assert_eq!(entry.content, "Just content.");
        assert_eq!(entry.entry_type, EntryType::Raw);
        assert_eq!(entry.visibility, "normal");
        assert!(entry.heading.is_none());
        assert!(entry.perspectives.is_empty());
        assert!(entry.relations.is_empty());
        assert_eq!(body, "Just content.");
    }

    #[test]
    fn test_parse_full() {
        let markdown = concat!(
            "---\n",
            "id: 2b0bd4c\n",
            "created: 2026-02-21T10:15:00Z\n",
            "perspectives: [decisions, knowledge]\n",
            "visibility: always\n",
            "relations:\n",
            "  derived_from: 6c121fe\n",
            "  supersedes: a3f8812\n",
            "parent: bf81639\n",
            "expires: 2026-04-04T09:00:00Z\n",
            "impression: Search is slow\n",
            "impression_strength: 0.6\n",
            "summarizes: [6c121fe, 4a2b3c1]\n",
            "---\n",
            "# Dual ID Scheme\n",
            "\n",
            "Content body here...\n",
        );
        let markdown = markdown.as_bytes();
        let (entry, _body) = parse(markdown).unwrap();

        assert_eq!(entry.source, "[git]");
        assert_eq!(entry.heading, Some("Dual ID Scheme".to_string()));
        assert_eq!(entry.content, "Content body here...");
        assert_eq!(entry.entry_type, EntryType::Impression);
        assert_eq!(entry.visibility, "always");
        assert_eq!(entry.perspectives, vec!["decisions", "knowledge"]);
        assert_eq!(entry.parent_id, Some("bf81639".to_string()));
        assert_eq!(entry.impression_hint, Some("Search is slow".to_string()));
        assert!((entry.impression_strength - 0.6).abs() < 0.001);
        assert_eq!(entry.summarizes, vec!["6c121fe", "4a2b3c1"]);
        assert_eq!(entry.level, ChunkLevel::H1);

        assert_eq!(entry.relations.len(), 2);
        let kinds: Vec<&str> = entry.relations.iter().map(|r| r.kind.as_str()).collect();
        assert!(kinds.contains(&"derived_from"));
        assert!(kinds.contains(&"supersedes"));
    }

    #[test]
    fn test_parse_explicit_entry_type_meta() {
        let markdown = concat!(
            "---\n",
            "id: abc1234\n",
            "created: 2026-02-21T10:15:00Z\n",
            "entry_type: meta\n",
            "---\n",
            "Meta entry body.\n",
        );
        let (entry, _) = parse(markdown.as_bytes()).unwrap();
        assert_eq!(entry.entry_type, EntryType::Meta);
    }

    #[test]
    fn test_parse_explicit_entry_type_summary() {
        let markdown = concat!(
            "---\n",
            "id: abc1234\n",
            "created: 2026-02-21T10:15:00Z\n",
            "entry_type: summary\n",
            "---\n",
            "Summary entry body.\n",
        );
        let (entry, _) = parse(markdown.as_bytes()).unwrap();
        assert_eq!(entry.entry_type, EntryType::Summary);
    }

    #[test]
    fn test_parse_missing_entry_type_defaults_to_raw() {
        // Old files without entry_type must parse as Raw.
        let markdown = b"---\nid: abc1234\ncreated: 2026-02-21T10:15:00Z\n---\nContent.\n";
        let (entry, _) = parse(markdown).unwrap();
        assert_eq!(entry.entry_type, EntryType::Raw);
    }

    // -----------------------------------------------------------------------
    // Roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_roundtrip() {
        let original = full_entry();
        let rendered = render(&original).unwrap();
        let (parsed, _body) = parse(rendered.as_bytes()).unwrap();

        assert_eq!(parsed.heading, original.heading);
        assert_eq!(parsed.content, original.content);
        assert_eq!(parsed.perspectives, original.perspectives);
        assert_eq!(parsed.visibility, original.visibility);
        assert_eq!(parsed.parent_id, Some("bf81639".to_string())); // short_id applied
        assert_eq!(parsed.summarizes, original.summarizes);
        // Timestamps survive the round-trip (second granularity)
        assert_eq!(parsed.created_at, original.created_at);
        assert_eq!(parsed.expires_at, original.expires_at);

        // Relations survive (order may differ in map iteration)
        assert_eq!(parsed.relations.len(), original.relations.len());
    }

    #[test]
    fn test_roundtrip_impression() {
        let mut entry = minimal_entry();
        entry.impression_hint = Some("curious".to_string());
        entry.impression_strength = 0.4;

        let rendered = render(&entry).unwrap();
        let (parsed, _) = parse(rendered.as_bytes()).unwrap();

        assert_eq!(parsed.impression_hint, Some("curious".to_string()));
        assert!((parsed.impression_strength - 0.4).abs() < 0.001);
        assert_eq!(parsed.entry_type, EntryType::Impression);
    }

    #[test]
    fn test_roundtrip_meta_entry_type() {
        let mut entry = minimal_entry();
        entry.entry_type = EntryType::Meta;

        let rendered = render(&entry).unwrap();
        let (parsed, _) = parse(rendered.as_bytes()).unwrap();

        assert_eq!(parsed.entry_type, EntryType::Meta);
    }

    #[test]
    fn test_roundtrip_summary_entry_type() {
        let mut entry = minimal_entry();
        entry.entry_type = EntryType::Summary;
        entry.summarizes = vec!["abc1234".to_string()];

        let rendered = render(&entry).unwrap();
        let (parsed, _) = parse(rendered.as_bytes()).unwrap();

        assert_eq!(parsed.entry_type, EntryType::Summary);
    }

    // -----------------------------------------------------------------------
    // entry_filename
    // -----------------------------------------------------------------------

    #[test]
    fn test_entry_filename_from_heading() {
        let content = "Content body here.";
        let id = short_id(&crate::chunk::content_hash(content)).to_string();
        let entry = Entry {
            content: content.to_string(),
            heading: Some("Dual ID Scheme".to_string()),
            ..minimal_entry()
        };
        let filename = entry_filename(&entry);
        assert_eq!(filename, format!("dual-id-scheme-{id}.md"));
    }

    #[test]
    fn test_entry_filename_from_content() {
        let long_content = "This is a very long piece of content that definitely exceeds fifty characters by a wide margin.";
        let entry = Entry {
            content: long_content.to_string(),
            heading: None,
            ..minimal_entry()
        };
        let filename = entry_filename(&entry);
        // Should be truncated slug + id + .md
        // Slug is max 50 chars before the -id suffix
        let hash = crate::chunk::content_hash(long_content);
        let id = short_id(&hash);
        assert!(filename.ends_with(&format!("-{id}.md")));
        let slug_part = filename.strip_suffix(&format!("-{id}.md")).unwrap();
        assert!(slug_part.len() <= 50);
        assert!(!slug_part.starts_with('-'));
        assert!(!slug_part.ends_with('-'));
    }

    #[test]
    fn test_entry_filename_no_trailing_hyphen_in_slug() {
        let entry = Entry {
            content: "abc".to_string(),
            heading: Some("Hello World".to_string()),
            ..minimal_entry()
        };
        let filename = entry_filename(&entry);
        assert!(filename.starts_with("hello-world-"));
    }

    // -----------------------------------------------------------------------
    // entry_directory
    // -----------------------------------------------------------------------

    #[test]
    fn test_entry_directory() {
        let entry = Entry {
            perspectives: vec!["decisions".to_string(), "knowledge".to_string()],
            ..minimal_entry()
        };
        assert_eq!(entry_directory(&entry), "decisions");
    }

    #[test]
    fn test_entry_directory_unsorted() {
        let entry = minimal_entry();
        assert_eq!(entry_directory(&entry), "_unsorted");
    }

    #[test]
    fn test_entry_directory_special_chars() {
        // Slash: must not appear in directory name — slugified away.
        let entry_slash = Entry {
            perspectives: vec!["foo/bar".to_string()],
            ..minimal_entry()
        };
        let dir = entry_directory(&entry_slash);
        assert!(
            !dir.contains('/'),
            "directory must not contain '/': got '{dir}'"
        );
        assert!(!dir.is_empty(), "directory must not be empty");

        // Spaces: converted to hyphens.
        let entry_space = Entry {
            perspectives: vec!["my decisions".to_string()],
            ..minimal_entry()
        };
        let dir = entry_directory(&entry_space);
        assert!(!dir.contains(' '), "directory must not contain spaces");

        // Unicode: non-alphanumeric Unicode characters become hyphens; alphanumeric
        // Unicode chars (e.g. letters in other scripts) are kept.
        let entry_unicode = Entry {
            perspectives: vec!["déci/sions".to_string()],
            ..minimal_entry()
        };
        let dir = entry_directory(&entry_unicode);
        assert!(
            !dir.contains('/'),
            "directory must not contain '/': got '{dir}'"
        );
        assert!(!dir.is_empty(), "directory must not be empty");
    }

    // -----------------------------------------------------------------------
    // relations compact format
    // -----------------------------------------------------------------------

    #[test]
    fn test_relations_compact_format_single() {
        let mut entry = minimal_entry();
        entry.relations = vec![ChunkRelation::new("derived_from", "6c121fe")];
        let rendered = render(&entry).unwrap();
        // Single target must be serialized as a scalar (possibly quoted), not as a list
        assert!(rendered.contains("derived_from:"));
        assert!(rendered.contains("6c121fe"));
        // Must not be a sequence item
        assert!(!rendered.contains("- 6c121fe"));
    }

    #[test]
    fn test_relations_compact_format_multiple() {
        let mut entry = minimal_entry();
        entry.relations = vec![
            ChunkRelation::new("related_to", "aaaaaaa"),
            ChunkRelation::new("related_to", "bbbbbbb"),
        ];
        let rendered = render(&entry).unwrap();
        // Multiple targets must be rendered as a YAML list
        assert!(rendered.contains("related_to:"));
        assert!(rendered.contains("- aaaaaaa") || rendered.contains("[aaaaaaa"));
    }

    // -----------------------------------------------------------------------
    // StringOrVec serde
    // -----------------------------------------------------------------------

    #[test]
    fn test_string_or_vec_serialize_single() {
        let v = StringOrVec::single("abc");
        let yaml = serde_yml::to_string(&v).unwrap();
        assert_eq!(yaml.trim(), "abc");
    }

    #[test]
    fn test_string_or_vec_serialize_multiple() {
        let v = StringOrVec::multiple(vec!["abc".to_string(), "def".to_string()]);
        let yaml = serde_yml::to_string(&v).unwrap();
        assert!(yaml.contains("abc") && yaml.contains("def"));
    }

    #[test]
    fn test_string_or_vec_deserialize_string() {
        let v: StringOrVec = serde_yml::from_str("abc").unwrap();
        assert_eq!(v.0, vec!["abc"]);
    }

    #[test]
    fn test_string_or_vec_deserialize_list() {
        let v: StringOrVec = serde_yml::from_str("- abc\n- def\n").unwrap();
        assert_eq!(v.0, vec!["abc", "def"]);
    }

    // -----------------------------------------------------------------------
    // slugify
    // -----------------------------------------------------------------------

    #[test]
    fn test_slugify_basic() {
        assert_eq!(slugify("Hello World", 50), "hello-world");
    }

    #[test]
    fn test_slugify_special_chars() {
        assert_eq!(slugify("Foo & Bar: Baz!", 50), "foo-bar-baz");
    }

    #[test]
    fn test_slugify_truncation() {
        let long = "a".repeat(60);
        let slug = slugify(&long, 50);
        assert!(slug.len() <= 50);
    }

    #[test]
    fn test_slugify_no_leading_trailing_hyphens() {
        let slug = slugify("  hello  ", 50);
        assert!(!slug.starts_with('-'));
        assert!(!slug.ends_with('-'));
    }

    // -----------------------------------------------------------------------
    // CRLF (Windows line endings)
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_crlf_line_endings() {
        let markdown =
            "---\r\nid: abc1234\r\ncreated: 2026-02-21T10:15:00Z\r\n---\r\nJust content.\r\n";
        let (entry, body) = parse(markdown.as_bytes()).unwrap();

        assert_eq!(entry.source, "[git]");
        assert_eq!(entry.content, "Just content.");
        assert_eq!(body, "Just content.");
    }

    #[test]
    fn test_parse_crlf_with_heading_and_body() {
        let markdown = "---\r\nid: 2b0bd4c\r\ncreated: 2026-02-21T10:15:00Z\r\nperspectives: [decisions]\r\n---\r\n# My Heading\r\n\r\nBody text here.\r\n";
        let (entry, _body) = parse(markdown.as_bytes()).unwrap();

        assert_eq!(entry.heading, Some("My Heading".to_string()));
        assert!(entry.content.contains("Body text here."));
        assert_eq!(entry.perspectives, vec!["decisions"]);
    }

    #[test]
    fn test_parse_crlf_at_eof_no_trailing_newline() {
        let markdown = "---\r\nid: abc1234\r\ncreated: 2026-02-21T10:15:00Z\r\n---";
        let (entry, body) = parse(markdown.as_bytes()).unwrap();

        assert_eq!(entry.source, "[git]");
        assert!(body.is_empty());
    }

    #[test]
    fn test_roundtrip_parses_with_crlf() {
        // Render produces LF; verify CRLF version also parses back.
        let original = full_entry();
        let rendered_lf = render(&original).unwrap();
        let rendered_crlf = rendered_lf.replace('\n', "\r\n");

        let (parsed, _body) = parse(rendered_crlf.as_bytes()).unwrap();
        assert_eq!(parsed.heading, original.heading);
        assert_eq!(parsed.perspectives, original.perspectives);
        assert_eq!(parsed.visibility, original.visibility);
    }

    // -----------------------------------------------------------------------
    // T13: Empty content / heading edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_entry_filename_empty_content_no_heading() {
        // Entry with empty content and no heading: slug is empty, so the
        // filename must degenerate gracefully to just `<id>.md` — no leading
        // hyphen and a non-empty result.
        let entry = Entry {
            content: String::new(),
            heading: None,
            ..minimal_entry()
        };
        let filename = entry_filename(&entry);
        assert!(!filename.is_empty(), "filename must not be empty");
        assert!(
            !filename.starts_with('-'),
            "filename must not start with '-'"
        );
        assert!(filename.ends_with(".md"), "filename must end with .md");
    }

    #[test]
    fn test_entry_filename_empty_heading() {
        // Entry with an empty heading string: `slugify("")` returns `""`, so
        // `entry_filename` falls back to the empty slug path.  The filename
        // must not start with a hyphen.
        let entry = Entry {
            content: "Some content.".to_string(),
            heading: Some(String::new()),
            ..minimal_entry()
        };
        let filename = entry_filename(&entry);
        assert!(!filename.is_empty(), "filename must not be empty");
        assert!(
            !filename.starts_with('-'),
            "filename must not start with '-'"
        );
        assert!(filename.ends_with(".md"), "filename must end with .md");
    }

    #[test]
    fn test_roundtrip_empty_content() {
        // An entry with empty content must survive a render → parse cycle.
        // The heading is preserved and the content field round-trips as empty.
        // Note: `parse` returns the *raw* post-frontmatter body (which includes
        // the heading line), so we only check `parsed.content`.
        let entry = Entry {
            content: String::new(),
            heading: Some("Empty Body".to_string()),
            ..minimal_entry()
        };
        let rendered = render(&entry).unwrap();
        let (parsed, _body) = parse(rendered.as_bytes()).unwrap();

        assert_eq!(parsed.heading, entry.heading);
        assert_eq!(parsed.content, "");
    }

    #[test]
    fn test_roundtrip_whitespace_only_content() {
        // Whitespace-only content is trimmed on parse (`trim_end`), so the
        // round-tripped content will be empty rather than `"   \n  "`.
        let entry = Entry {
            content: "   \n  ".to_string(),
            heading: Some("Whitespace Only".to_string()),
            ..minimal_entry()
        };
        let rendered = render(&entry).unwrap();
        let (parsed, _body) = parse(rendered.as_bytes()).unwrap();

        assert_eq!(parsed.heading, entry.heading);
        // `trim_end` is applied on parse; the whitespace is gone.
        assert_eq!(parsed.content, "");
    }

    // -----------------------------------------------------------------------
    // F13: serde_yml quotes multi-line strings — no bare \n---\n in YAML
    // -----------------------------------------------------------------------

    #[test]
    fn test_serde_yml_quotes_multiline_strings() {
        // If serde_yml did NOT quote multi-line values, a Frontmatter whose
        // `impression` field contained `\n---\n` would fool `split_frontmatter`
        // into treating it as the closing delimiter.  Verify that does not happen.
        let fm = Frontmatter {
            id: "abc1234".to_string(),
            created: "2026-02-21T10:15:00Z".to_string(),
            perspectives: vec![],
            visibility: None,
            relations: BTreeMap::new(),
            parent: None,
            expires: None,
            impression: Some("line one\n---\nline two".to_string()),
            impression_strength: None,
            summarizes: vec![],
            entry_type: EntryType::Raw,
            level: FrontmatterLevel::H1,
        };

        let yaml = serde_yml::to_string(&fm).unwrap();
        // The bare sequence that would fool the parser must not appear verbatim.
        assert!(
            !yaml.contains("\n---\n"),
            "serde_yml must quote the multi-line value; bare \\n---\\n found in: {yaml}"
        );
    }

    // -----------------------------------------------------------------------
    // B24: Level roundtrip — Content level must survive git sync
    // -----------------------------------------------------------------------

    #[test]
    fn test_render_content_level_writes_level_field() {
        // An inline-stored (Content level) entry must emit `level: content`
        // so that re-ingestion via sync does not promote it to H1.
        let entry = minimal_entry(); // level: ChunkLevel::CONTENT, no heading
        let rendered = render(&entry).unwrap();
        assert!(
            rendered.contains("level: content"),
            "expected 'level: content' in frontmatter; got:\n{rendered}"
        );
    }

    #[test]
    fn test_parse_explicit_content_level() {
        // A file with `level: content` must parse as ChunkLevel::CONTENT,
        // regardless of whether a heading is present.
        let markdown = concat!(
            "---\n",
            "id: abc1234\n",
            "created: 2026-02-21T10:15:00Z\n",
            "level: content\n",
            "---\n",
            "# Appears to have a heading\n",
            "\n",
            "But level is content.\n",
        );
        let (entry, _) = parse(markdown.as_bytes()).unwrap();
        assert_eq!(
            entry.level,
            ChunkLevel::CONTENT,
            "explicit 'level: content' must override heading-based inference"
        );
    }

    #[test]
    fn test_parse_old_file_without_level_falls_back_to_heading_inference() {
        // Old files lack the `level` field. They must fall back to the
        // heading-based heuristic: heading present → H1, no heading → Content.
        let with_heading = concat!(
            "---\n",
            "id: abc1234\n",
            "created: 2026-02-21T10:15:00Z\n",
            "---\n",
            "# A Heading\n",
            "\n",
            "Body.\n",
        );
        let (entry_h1, _) = parse(with_heading.as_bytes()).unwrap();
        assert_eq!(
            entry_h1.level,
            ChunkLevel::H1,
            "old file with heading must infer H1"
        );

        let without_heading = concat!(
            "---\n",
            "id: abc1234\n",
            "created: 2026-02-21T10:15:00Z\n",
            "---\n",
            "No heading, just body.\n",
        );
        let (entry_content, _) = parse(without_heading.as_bytes()).unwrap();
        assert_eq!(
            entry_content.level,
            ChunkLevel::CONTENT,
            "old file without heading must infer Content"
        );
    }

    #[test]
    fn test_roundtrip_content_level_preserved() {
        // A Content-level entry must survive a full render → parse cycle with
        // its level intact.
        let entry = minimal_entry(); // level: ChunkLevel::CONTENT
        let rendered = render(&entry).unwrap();
        let (parsed, _) = parse(rendered.as_bytes()).unwrap();
        assert_eq!(
            parsed.level,
            ChunkLevel::CONTENT,
            "Content level must survive render → parse roundtrip"
        );
    }

    #[test]
    fn test_h1_level_omitted_from_frontmatter() {
        // H1 is the default; it must be omitted to keep old files clean.
        let entry = full_entry(); // level: ChunkLevel::H1, has heading
        let rendered = render(&entry).unwrap();
        assert!(
            !rendered.contains("level:"),
            "H1 level must be omitted from frontmatter; got:\n{rendered}"
        );
    }
}
