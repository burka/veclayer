//! Render MCP tool results as readable markdown for agent consumption.
//!
//! The MCP text field is read as a string by agents. Returning raw JSON
//! forces the agent to mentally parse `{"chunk":{"content":"# Heading\\n..."}}`.
//! Instead, we render results as markdown so content is directly readable,
//! metadata is inline and minimal, and IDs are available for follow-up.

use super::types::*;
use crate::chunk::short_id;

const EMBEDDING_PENDING_LABEL: &str = "embedding pending";

/// Format recall results as readable markdown.
pub fn format_recall(query: Option<&str>, results: &[SearchResultResponse]) -> String {
    if results.is_empty() {
        return match query {
            Some(q) => format!("No results for \"{}\".", q),
            None => "No entries found.".to_string(),
        };
    }

    let mut out = String::new();

    for (i, r) in results.iter().enumerate() {
        if i > 0 {
            out.push_str("\n---\n\n");
        }

        // Header: number, heading or first line, relevance tier
        let title = r
            .chunk
            .heading
            .as_deref()
            .unwrap_or_else(|| first_line(&r.chunk.content));
        out.push_str(&format!("### {}. {} ({})\n", i + 1, title, r.relevance));

        // Hierarchy breadcrumbs (when available)
        if !r.hierarchy_path.is_empty() {
            let crumbs: Vec<&str> = r
                .hierarchy_path
                .iter()
                .map(|c| {
                    c.heading
                        .as_deref()
                        .unwrap_or_else(|| first_line(&c.content))
                })
                .collect();
            out.push_str(&format!("> {}\n", crumbs.join(" › ")));
        }

        // Metadata blockquote — visually separated from content
        let mut meta = vec![format!("`{}`", short_id(&r.chunk.id))];
        if r.chunk.entry_type != "raw" {
            meta.push(r.chunk.entry_type.clone());
        }
        if !r.chunk.perspectives.is_empty() {
            meta.push(r.chunk.perspectives.join(", "));
        }
        if r.chunk.source_file != "[agent]" && r.chunk.source_file != "[inline]" {
            meta.push(r.chunk.source_file.clone());
        }
        // Include raw score for relevance debugging
        meta.push(format!("{:.2}", r.score));
        if r.chunk.embedding_pending {
            meta.push(EMBEDDING_PENDING_LABEL.to_string());
        }
        out.push_str(&format!("> {}\n", meta.join(" · ")));

        // Content: render directly as markdown
        out.push('\n');
        out.push_str(r.chunk.content.trim());
        out.push('\n');

        // Children (condensed)
        if !r.children.is_empty() {
            out.push_str("\n**Children:**\n");
            for child in &r.children {
                let child_title = child
                    .heading
                    .as_deref()
                    .unwrap_or_else(|| first_line(&child.content));
                out.push_str(&format!("- {} `{}`\n", child_title, short_id(&child.id)));
            }
        }
    }

    // Footer
    out.push_str(&format!(
        "\n_{} result(s). Use `focus(id)` to drill into any entry._\n",
        results.len()
    ));

    out
}

/// Format focus results as readable markdown.
pub fn format_focus(response: &FocusResponse) -> String {
    let mut out = String::new();

    let node = &response.node;
    let title = node
        .heading
        .as_deref()
        .unwrap_or_else(|| first_line(&node.content));

    // Header
    out.push_str(&format!("## {}\n", title));

    // Metadata blockquote — visually separated from content
    let mut meta = vec![
        format!("`{}`", short_id(&node.id)),
        node.level.clone(),
        node.entry_type.clone(),
    ];
    if node.visibility != "normal" {
        meta.push(node.visibility.clone());
    }
    if !node.perspectives.is_empty() {
        meta.push(node.perspectives.join(", "));
    }
    if node.source_file != "[agent]" && node.source_file != "[inline]" {
        meta.push(node.source_file.clone());
    }
    if node.embedding_pending {
        meta.push("embedding pending".to_string());
    }
    out.push_str(&format!("> {}\n", meta.join(" · ")));

    // Full content
    out.push('\n');
    out.push_str(node.content.trim());
    out.push('\n');

    // Children
    if !response.children.is_empty() {
        out.push_str(&format!("\n### Children ({})\n\n", response.children.len()));
        for child in &response.children {
            let child_title = child
                .chunk
                .heading
                .as_deref()
                .unwrap_or_else(|| first_line(&child.chunk.content));
            let relevance_hint = child
                .relevance
                .map(|r| format!(" [{:.2}]", r))
                .unwrap_or_default();

            out.push_str(&format!("**{}**{}\n", child_title, relevance_hint));

            // Child metadata blockquote
            let mut meta = vec![format!("`{}`", short_id(&child.chunk.id))];
            if child.chunk.entry_type != "raw" {
                meta.push(child.chunk.entry_type.clone());
            }
            if !child.chunk.perspectives.is_empty() {
                meta.push(child.chunk.perspectives.join(", "));
            }
            if child.chunk.embedding_pending {
                meta.push(EMBEDDING_PENDING_LABEL.to_string());
            }
            out.push_str(&format!("> {}\n", meta.join(" · ")));

            // Show child content (trimmed preview for children, full would be too long)
            let preview = content_preview(&child.chunk.content, 300);
            out.push_str(preview);
            out.push_str("\n\n");
        }
    } else {
        out.push_str("\n_(no children)_\n");
    }

    out
}

/// First non-empty line of content (for use as title fallback).
fn first_line(s: &str) -> &str {
    s.lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("(untitled)")
        .trim()
}

/// Truncate content for preview, preserving line structure.
/// Uses char boundaries to avoid panics on multi-byte content.
fn content_preview(s: &str, max_chars: usize) -> &str {
    let trimmed = s.trim();
    if trimmed.len() <= max_chars {
        return trimmed;
    }
    // Floor to a char boundary at or before max_chars
    let boundary = floor_char_boundary(trimmed, max_chars);
    let truncated = &trimmed[..boundary];
    // Prefer breaking at a newline for cleaner output
    if let Some(last_newline) = truncated.rfind('\n') {
        truncated[..last_newline].trim_end()
    } else {
        truncated
    }
}

/// Find the largest byte index <= `index` that is a char boundary.
/// (Equivalent to str::floor_char_boundary, stabilised in Rust 1.82+)
fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Format store status as readable markdown for `veclayer://status` and `think(status)`.
pub fn format_store_status(
    stats: &crate::store::StoreStats,
    aging_config: &crate::aging::AgingConfig,
) -> String {
    let mut md = String::from("## Store Status\n\n");
    md.push_str(&format!("- **Total entries:** {}\n", stats.total_chunks));
    md.push_str(&format!(
        "- **Source files:** {}\n",
        stats.source_files.len()
    ));

    if !stats.chunks_by_level.is_empty() {
        md.push_str("\n### Entries by level\n\n");
        for level in 1..=7 {
            if let Some(count) = stats.chunks_by_level.get(&level) {
                let name = if level <= 6 {
                    format!("H{level}")
                } else {
                    "Content".to_string()
                };
                md.push_str(&format!("- {name}: {count}\n"));
            }
        }
    }

    if !stats.source_files.is_empty() {
        md.push_str("\n### Source files\n\n");
        for file in &stats.source_files {
            md.push_str(&format!("- {file}\n"));
        }
    }

    md.push_str(&format!(
        "\n### Aging policy\n\n- Degrade {} → '{}' after {} days\n",
        aging_config.degrade_from.join("/"),
        aging_config.degrade_to,
        aging_config.degrade_after_days,
    ));

    if stats.pending_embeddings > 0 {
        let eta = super::embed_worker::eta_seconds(stats.pending_embeddings);
        md.push_str("\n### Pending embeddings\n\n");
        md.push_str(&format!("- **Pending:** {}\n", stats.pending_embeddings));
        md.push_str(&format!("- **Estimated completion:** ~{eta}s\n"));
    }

    md
}

/// Format hot entries with salience scores for the `veclayer://hot` resource.
pub fn format_hot_entries(
    chunks: &[crate::HierarchicalChunk],
    top: &[(usize, crate::salience::SalienceScore)],
) -> String {
    let mut md = String::from("## Hot Entries\n\n");
    for (idx, score) in top {
        let chunk = &chunks[*idx];
        let heading = chunk.heading.as_deref().unwrap_or("(no heading)");
        let short = short_id(&chunk.id);
        let perspectives = if chunk.perspectives.is_empty() {
            String::new()
        } else {
            format!(" ({})", chunk.perspectives.join(", "))
        };
        let preview = content_preview(&chunk.content, 120);
        md.push_str(&format!(
            "- **{heading}** [{:.3}] `{short}`{perspectives}\n  {preview}\n",
            score.composite,
        ));
    }
    md.push_str(&format!("\n_{} entry(ies)._\n", top.len()));
    md
}

/// Format a full entry with children for the `veclayer://entries/{id}` resource.
pub fn format_entry_detail(
    chunk: &crate::HierarchicalChunk,
    children: &[crate::HierarchicalChunk],
) -> String {
    let heading = chunk.heading.as_deref().unwrap_or("(no heading)");
    let mut md = format!("## {heading}\n\n");

    // Content
    md.push_str(chunk.content.trim());
    md.push_str("\n\n");

    // Metadata block
    md.push_str("### Metadata\n\n");
    md.push_str(&format!("- **ID:** `{}`\n", chunk.id));
    md.push_str(&format!("- **Type:** {}\n", chunk.entry_type));
    md.push_str(&format!("- **Visibility:** {}\n", chunk.visibility));
    md.push_str(&format!("- **Level:** {}\n", chunk.level));
    md.push_str(&format!("- **Source:** {}\n", chunk.source_file));
    if !chunk.perspectives.is_empty() {
        md.push_str(&format!(
            "- **Perspectives:** {}\n",
            chunk.perspectives.join(", ")
        ));
    }
    if let Some(parent) = &chunk.parent_id {
        md.push_str(&format!("- **Parent:** `{}`\n", short_id(parent)));
    }

    // Access profile
    let ap = &chunk.access_profile;
    md.push_str(&format!(
        "- **Access:** total={}, hour={}, day={}, week={}\n",
        ap.total, ap.hour, ap.day, ap.week
    ));

    // Relations
    if !chunk.relations.is_empty() {
        md.push_str(&format!("\n### Relations ({})\n\n", chunk.relations.len()));
        for rel in &chunk.relations {
            md.push_str(&format!(
                "- {} → `{}`\n",
                rel.kind,
                short_id(&rel.target_id)
            ));
        }
    }

    // Children
    if !children.is_empty() {
        md.push_str(&format!("\n### Children ({})\n\n", children.len()));
        for child in children {
            let child_heading = child.heading.as_deref().unwrap_or("(no heading)");
            let child_short = short_id(&child.id);
            let preview = content_preview(&child.content, 120);
            md.push_str(&format!(
                "- **{child_heading}** `{child_short}`\n  {preview}\n"
            ));
        }
    }

    md
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(id: &str, content: &str, heading: Option<&str>) -> ChunkResponse {
        ChunkResponse {
            id: id.to_string(),
            content: content.to_string(),
            level: "H2".to_string(),
            entry_type: "raw".to_string(),
            path: "test".to_string(),
            source_file: "[agent]".to_string(),
            heading: heading.map(String::from),
            parent_id: None,
            visibility: "normal".to_string(),
            perspectives: vec![],
            access: AccessProfileResponse {
                hour: 0,
                day: 0,
                week: 0,
                month: 0,
                year: 0,
                total: 0,
            },
            embedding_pending: false,
        }
    }

    #[test]
    fn recall_empty() {
        let out = format_recall(Some("test"), &[]);
        assert_eq!(out, "No results for \"test\".");
    }

    #[test]
    fn recall_single_result() {
        let results = vec![SearchResultResponse {
            chunk: make_chunk(
                "abc1234deadbeef",
                "# Design\n\nWe chose Rust.",
                Some("Design"),
            ),
            score: 0.85,
            relevance: "strong".to_string(),
            hierarchy_path: vec![],
            children: vec![],
        }];
        let out = format_recall(Some("architecture"), &results);
        assert!(out.contains("### 1. Design (strong)"));
        assert!(out.contains("> `abc1234`")); // Metadata in blockquote
        assert!(out.contains("0.85")); // Raw score in metadata
        assert!(out.contains("We chose Rust."));
        assert!(!out.contains("\"content\"")); // No JSON
    }

    #[test]
    fn recall_with_children() {
        let child = make_chunk("child123deadbeef", "Child content here", Some("Subsection"));
        let results = vec![SearchResultResponse {
            chunk: make_chunk("parent123deadbeef", "Parent content", Some("Main")),
            score: 0.7,
            relevance: "moderate".to_string(),
            hierarchy_path: vec![],
            children: vec![child],
        }];
        let out = format_recall(Some("query"), &results);
        assert!(out.contains("**Children:**"));
        assert!(out.contains("Subsection"));
        assert!(out.contains("`child12`"));
    }

    #[test]
    fn focus_format() {
        let node = make_chunk(
            "node123deadbeef",
            "Full node content\nwith multiple lines",
            Some("My Entry"),
        );
        let child_chunk = make_chunk("child456deadbeef", "Child details", Some("Detail"));
        let response = FocusResponse {
            node,
            children: vec![FocusChild {
                chunk: child_chunk,
                relevance: Some(0.92),
            }],
        };
        let out = format_focus(&response);
        assert!(out.contains("## My Entry"));
        assert!(out.contains("> `node123`")); // Parent metadata in blockquote
        assert!(out.contains("Full node content\nwith multiple lines"));
        assert!(out.contains("### Children (1)"));
        assert!(out.contains("[0.92]"));
        assert!(out.contains("> `child45`")); // Child metadata in blockquote
    }

    #[test]
    fn perspectives_shown_in_recall() {
        let mut chunk = make_chunk("abc1234deadbeef", "Content", Some("Title"));
        chunk.perspectives = vec!["decisions".to_string(), "learnings".to_string()];
        let results = vec![SearchResultResponse {
            chunk,
            score: 0.5,
            relevance: "strong".to_string(),
            hierarchy_path: vec![],
            children: vec![],
        }];
        let out = format_recall(Some("q"), &results);
        assert!(out.contains("> `abc1234` · decisions, learnings · 0.50"));
    }

    #[test]
    fn focus_hides_normal_visibility() {
        let node = make_chunk("abc1234deadbeef", "Content", Some("Title"));
        let response = FocusResponse {
            node,
            children: vec![],
        };
        let out = format_focus(&response);
        // "normal" visibility should be omitted (default, not interesting)
        assert!(!out.contains("normal"));
        assert!(out.contains("> `abc1234` · H2 · raw"));
    }

    #[test]
    fn focus_shows_non_normal_visibility() {
        let mut node = make_chunk("abc1234deadbeef", "Content", Some("Title"));
        node.visibility = "always".to_string();
        let response = FocusResponse {
            node,
            children: vec![],
        };
        let out = format_focus(&response);
        assert!(out.contains("always"));
    }

    #[test]
    fn recall_hierarchy_breadcrumbs() {
        let ancestor = make_chunk("root000deadbeef", "Root content", Some("Root"));
        let parent = make_chunk("par0000deadbeef", "Parent content", Some("Parent"));
        let results = vec![SearchResultResponse {
            chunk: make_chunk("leaf000deadbeef", "Leaf content", Some("Leaf")),
            score: 0.6,
            relevance: "strong".to_string(),
            hierarchy_path: vec![ancestor, parent],
            children: vec![],
        }];
        let out = format_recall(Some("q"), &results);
        assert!(out.contains("> Root › Parent")); // Breadcrumb line
        assert!(out.contains("> `leaf000`")); // Metadata line follows
    }

    #[test]
    fn recall_no_breadcrumbs_when_path_empty() {
        let results = vec![SearchResultResponse {
            chunk: make_chunk("abc1234deadbeef", "Content", Some("Title")),
            score: 0.5,
            relevance: "strong".to_string(),
            hierarchy_path: vec![],
            children: vec![],
        }];
        let out = format_recall(Some("q"), &results);
        assert!(!out.contains("›")); // No breadcrumb separator
    }

    #[test]
    fn recall_score_in_metadata() {
        let results = vec![SearchResultResponse {
            chunk: make_chunk("abc1234deadbeef", "Content", Some("Title")),
            score: 0.42,
            relevance: "moderate".to_string(),
            hierarchy_path: vec![],
            children: vec![],
        }];
        let out = format_recall(Some("q"), &results);
        assert!(out.contains("0.42"));
    }

    #[test]
    fn content_preview_utf8_safe() {
        // 3-byte char: "é" is 2 bytes, "日" is 3 bytes
        let s = "Hello 日本語 world";
        // Truncate mid-character — must not panic
        let preview = content_preview(s, 8); // byte 8 is inside "本"
        assert!(!preview.is_empty());
        // Verify it's valid UTF-8 (would fail at compile time if not, but
        // the real risk is a panic from the slice)
        assert!(preview.len() <= 8);
    }

    #[test]
    fn content_preview_emoji_safe() {
        let s = "Design 🚀 choices for the system";
        let preview = content_preview(s, 10); // byte 10 is inside the rocket emoji (4 bytes)
        assert!(!preview.is_empty());
    }

    #[test]
    fn focus_child_has_blockquote_metadata() {
        let node = make_chunk("node000deadbeef", "Node content", Some("Node"));
        let mut child = make_chunk("child00deadbeef", "Child content", Some("Child"));
        child.entry_type = "summary".to_string();
        child.perspectives = vec!["decisions".to_string()];
        let response = FocusResponse {
            node,
            children: vec![FocusChild {
                chunk: child,
                relevance: Some(0.8),
            }],
        };
        let out = format_focus(&response);
        assert!(out.contains("> `child00` · summary · decisions"));
    }

    #[test]
    fn recall_embedding_pending_shown() {
        let mut chunk = make_chunk("abc1234deadbeef", "Content", Some("Title"));
        chunk.embedding_pending = true;
        let results = vec![SearchResultResponse {
            chunk,
            score: 0.5,
            relevance: "strong".to_string(),
            hierarchy_path: vec![],
            children: vec![],
        }];
        let out = format_recall(Some("q"), &results);
        assert!(out.contains("embedding pending"));
    }

    #[test]
    fn recall_embedding_not_pending_hidden() {
        let chunk = make_chunk("abc1234deadbeef", "Content", Some("Title"));
        // embedding_pending defaults to false in make_chunk
        let results = vec![SearchResultResponse {
            chunk,
            score: 0.5,
            relevance: "strong".to_string(),
            hierarchy_path: vec![],
            children: vec![],
        }];
        let out = format_recall(Some("q"), &results);
        assert!(!out.contains("embedding pending"));
    }
}
