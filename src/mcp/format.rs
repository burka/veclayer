//! Render MCP tool results as readable markdown for agent consumption.
//!
//! The MCP text field is read as a string by agents. Returning raw JSON
//! forces the agent to mentally parse `{"chunk":{"content":"# Heading\\n..."}}`.
//! Instead, we render results as markdown so content is directly readable,
//! metadata is inline and minimal, and IDs are available for follow-up.

use super::types::*;
use crate::chunk::short_id;

const EMBEDDING_PENDING_LABEL: &str = "embedding pending";

/// Default recall limit, mirroring [`super::types::default_limit`].
/// Format recall results as readable markdown.
///
/// `requested_limit` is the caller's requested result count; the footer hints
/// that more may be available only when the result set fills that limit exactly.
pub fn format_recall(
    query: Option<&str>,
    results: &[SearchResultResponse],
    requested_limit: usize,
) -> String {
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

    // Footer — hint only when the result set fills the requested limit exactly
    // (a smaller set means the store had nothing more to give).
    let cap_hint = if results.len() >= requested_limit {
        " (more may be available — increase `limit` / top_k)"
    } else {
        ""
    };
    out.push_str(&format!(
        "\n_{} result(s){}. Use `focus(id)` to drill into any entry._\n",
        results.len(),
        cap_hint
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
    if let Some(hint) = &node.impression_hint {
        let strength = node
            .impression_strength
            .map(|s| format!(" {:.2}", s))
            .unwrap_or_default();
        meta.push(format!("impression: {}{}", hint, strength));
    }
    out.push_str(&format!("> {}\n", meta.join(" · ")));

    // Full content
    out.push('\n');
    out.push_str(node.content.trim());
    out.push('\n');

    // Relations — the outgoing link graph
    if !node.relations.is_empty() {
        out.push_str(&format!("\n### Relations ({})\n\n", node.relations.len()));
        for rel in &node.relations {
            out.push_str(&format!(
                "- {} → `{}`\n",
                rel.kind,
                short_id(&rel.target_id)
            ));
        }
    }

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
///
/// Pass `embedder_config` to append an `## Embedding` section with provider details.
pub fn format_store_status(
    stats: &crate::store::StoreStats,
    aging_config: &crate::aging::AgingConfig,
    embedder_config: Option<&crate::config::EmbedderConfig>,
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

    if let Some(config) = embedder_config {
        md.push_str(&format_embedding_section(config, stats.pending_embeddings));
    }

    md
}

/// Format the `## Embedding` section for the status resource.
fn format_embedding_section(config: &crate::config::EmbedderConfig, pending: usize) -> String {
    let provider = match config {
        crate::config::EmbedderConfig::Ollama {
            model, base_url, ..
        } => format!("Ollama ({model} @ {base_url})"),
        crate::config::EmbedderConfig::FastEmbed { model } => {
            format!("FastEmbed ({model})")
        }
    };

    let dimension = config
        .dimension()
        .map_or_else(|| "unknown".to_string(), |d| d.to_string());

    let pending_line = if pending == 1 {
        "1 entry awaiting embeddings".to_string()
    } else {
        format!("{pending} entries awaiting embeddings")
    };

    format!(
        "\n## Embedding\n\n- Provider: {provider}\n- Dimension: {dimension}\n- Pending: {pending_line}\n"
    )
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
            relations: vec![],
            impression_hint: None,
            impression_strength: None,
        }
    }

    /// Wrap a chunk in a SearchResultResponse with default score/relevance and
    /// format it via format_recall.
    fn fmt_recall(chunk: ChunkResponse) -> String {
        let results = vec![SearchResultResponse {
            chunk,
            score: 0.5,
            relevance: "strong".to_string(),
            hierarchy_path: vec![],
            children: vec![],
        }];
        format_recall(Some("q"), &results, 5)
    }

    #[test]
    fn recall_empty() {
        let out = format_recall(Some("test"), &[], 5);
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
        let out = format_recall(Some("architecture"), &results, 5);
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
        let out = format_recall(Some("query"), &results, 5);
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
    fn focus_renders_relations() {
        let mut node = make_chunk("node123deadbeef", "Content", Some("Linked Entry"));
        node.relations = vec![
            RelationResponse {
                kind: "supersedes".to_string(),
                target_id: "old999deadbeef".to_string(),
            },
            RelationResponse {
                kind: "related_to".to_string(),
                target_id: "rel888deadbeef".to_string(),
            },
        ];
        let response = FocusResponse {
            node,
            children: vec![],
        };
        let out = format_focus(&response);
        assert!(out.contains("### Relations (2)"));
        assert!(out.contains("- supersedes → `old999d`"));
        assert!(out.contains("- related_to → `rel888d`"));
    }

    #[test]
    fn focus_omits_relations_section_when_empty() {
        let node = make_chunk("node123deadbeef", "Content", Some("No Links"));
        let response = FocusResponse {
            node,
            children: vec![],
        };
        let out = format_focus(&response);
        assert!(!out.contains("### Relations"));
    }

    #[test]
    fn focus_renders_impression_hint_and_strength() {
        let mut node = make_chunk("imp123deadbeef", "A hunch", Some("Impression"));
        node.entry_type = "impression".to_string();
        node.impression_hint = Some("uncertain".to_string());
        node.impression_strength = Some(0.35);
        let response = FocusResponse {
            node,
            children: vec![],
        };
        let out = format_focus(&response);
        assert!(out.contains("impression: uncertain 0.35"));
    }

    #[test]
    fn focus_omits_impression_when_absent() {
        let node = make_chunk("node123deadbeef", "Content", Some("Raw Entry"));
        let response = FocusResponse {
            node,
            children: vec![],
        };
        let out = format_focus(&response);
        assert!(!out.contains("impression:"));
    }

    #[test]
    fn perspectives_shown_in_recall() {
        let mut chunk = make_chunk("abc1234deadbeef", "Content", Some("Title"));
        chunk.perspectives = vec!["decisions".to_string(), "learnings".to_string()];
        let out = fmt_recall(chunk);
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
        let out = format_recall(Some("q"), &results, 5);
        assert!(out.contains("> Root › Parent")); // Breadcrumb line
        assert!(out.contains("> `leaf000`")); // Metadata line follows
    }

    #[test]
    fn recall_no_breadcrumbs_when_path_empty() {
        let chunk = make_chunk("abc1234deadbeef", "Content", Some("Title"));
        let out = fmt_recall(chunk);
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
        let out = format_recall(Some("q"), &results, 5);
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
        let out = fmt_recall(chunk);
        assert!(out.contains("embedding pending"));
    }

    #[test]
    fn recall_embedding_not_pending_hidden() {
        let chunk = make_chunk("abc1234deadbeef", "Content", Some("Title"));
        // embedding_pending defaults to false in make_chunk
        let out = fmt_recall(chunk);
        assert!(!out.contains("embedding pending"));
    }

    // ── format_recall edge cases ─────────────────────────────────────────

    #[test]
    fn recall_empty_no_query_shows_no_entries_found() {
        let out = format_recall(None, &[], 5);
        assert_eq!(out, "No entries found.");
    }

    #[test]
    fn recall_result_uses_first_line_as_title_when_no_heading() {
        let mut chunk = make_chunk("abc1234deadbeef", "First line\nSecond line", None);
        chunk.heading = None;
        let out = fmt_recall(chunk);
        assert!(out.contains("### 1. First line (strong)"));
    }

    #[test]
    fn recall_source_file_shown_for_non_agent_files() {
        let mut chunk = make_chunk("abc1234deadbeef", "Content", Some("Title"));
        chunk.source_file = "docs/design.md".to_string();
        let out = fmt_recall(chunk);
        assert!(out.contains("docs/design.md"));
    }

    #[test]
    fn recall_inline_source_file_omitted() {
        let mut chunk = make_chunk("abc1234deadbeef", "Content", Some("Title"));
        chunk.source_file = "[inline]".to_string();
        let out = fmt_recall(chunk);
        assert!(!out.contains("[inline]"));
    }

    #[test]
    fn recall_multiple_results_separated_by_divider() {
        let results = vec![
            SearchResultResponse {
                chunk: make_chunk("aaa1234deadbeef", "Content A", Some("Entry A")),
                score: 0.8,
                relevance: "strong".to_string(),
                hierarchy_path: vec![],
                children: vec![],
            },
            SearchResultResponse {
                chunk: make_chunk("bbb1234deadbeef", "Content B", Some("Entry B")),
                score: 0.4,
                relevance: "moderate".to_string(),
                hierarchy_path: vec![],
                children: vec![],
            },
        ];
        let out = format_recall(Some("q"), &results, 5);
        assert!(out.contains("### 1. Entry A"));
        assert!(out.contains("### 2. Entry B"));
        assert!(out.contains("\n---\n"));
        assert!(out.contains("_2 result(s)."));
    }

    #[test]
    fn recall_entry_type_shown_for_non_raw() {
        let mut chunk = make_chunk("abc1234deadbeef", "Content", Some("Title"));
        chunk.entry_type = "summary".to_string();
        let out = fmt_recall(chunk);
        assert!(out.contains("summary"));
    }

    #[test]
    fn recall_raw_entry_type_omitted_from_metadata() {
        let chunk = make_chunk("abc1234deadbeef", "Content", Some("Title"));
        let out = fmt_recall(chunk);
        // "· raw" should NOT appear as a separate token in the metadata
        assert!(!out.contains("· raw\n") && !out.contains("· raw ·"));
    }

    // ── format_focus edge cases ──────────────────────────────────────────

    #[test]
    fn focus_no_children_shows_no_children_message() {
        let node = make_chunk("node000deadbeef", "Content", Some("Node"));
        let response = FocusResponse {
            node,
            children: vec![],
        };
        let out = format_focus(&response);
        assert!(out.contains("_(no children)_"));
    }

    #[test]
    fn focus_child_without_relevance_shows_no_score_bracket() {
        let node = make_chunk("node000deadbeef", "Node content", Some("Node"));
        let child = make_chunk("child00deadbeef", "Child content", Some("Child"));
        let response = FocusResponse {
            node,
            children: vec![FocusChild {
                chunk: child,
                relevance: None,
            }],
        };
        let out = format_focus(&response);
        assert!(!out.contains("[0."));
    }

    #[test]
    fn focus_uses_first_line_for_node_without_heading() {
        let mut node = make_chunk("node000deadbeef", "First heading line\nMore content", None);
        node.heading = None;
        let response = FocusResponse {
            node,
            children: vec![],
        };
        let out = format_focus(&response);
        assert!(out.contains("## First heading line"));
    }

    // ── format_store_status ──────────────────────────────────────────────

    #[test]
    fn format_store_status_empty_store() {
        use crate::aging::AgingConfig;
        use crate::store::StoreStats;
        use std::collections::HashMap;

        let stats = StoreStats {
            total_chunks: 0,
            chunks_by_level: HashMap::new(),
            source_files: vec![],
            pending_embeddings: 0,
        };
        let aging = AgingConfig::default();
        let out = format_store_status(&stats, &aging, None);
        assert!(out.contains("## Store Status"));
        assert!(out.contains("Total entries:** 0"));
        assert!(out.contains("Aging policy"));
        assert!(!out.contains("Pending embeddings"));
    }

    #[test]
    fn format_store_status_with_entries_and_pending() {
        use crate::aging::AgingConfig;
        use crate::store::StoreStats;
        use std::collections::HashMap;

        let mut by_level = HashMap::new();
        by_level.insert(1u8, 3usize);
        by_level.insert(7u8, 5usize);

        let stats = StoreStats {
            total_chunks: 8,
            chunks_by_level: by_level,
            source_files: vec!["docs/a.md".to_string(), "docs/b.md".to_string()],
            pending_embeddings: 32,
        };
        let aging = AgingConfig {
            degrade_after_days: 14,
            degrade_to: "deep_only".to_string(),
            degrade_from: vec!["normal".to_string()],
            ..AgingConfig::default()
        };
        let out = format_store_status(&stats, &aging, None);
        assert!(out.contains("Total entries:** 8"));
        assert!(out.contains("H1: 3"));
        assert!(out.contains("Content: 5"));
        assert!(out.contains("docs/a.md"));
        assert!(out.contains("docs/b.md"));
        assert!(out.contains("Pending embeddings"));
        assert!(out.contains("Pending:** 32"));
        assert!(out.contains("14 days"));
        assert!(out.contains("deep_only"));
    }

    #[test]
    fn format_store_status_level_7_shown_as_content() {
        use crate::aging::AgingConfig;
        use crate::store::StoreStats;
        use std::collections::HashMap;

        let mut by_level = HashMap::new();
        by_level.insert(7u8, 42usize);

        let stats = StoreStats {
            total_chunks: 42,
            chunks_by_level: by_level,
            source_files: vec![],
            pending_embeddings: 0,
        };
        let out = format_store_status(&stats, &AgingConfig::default(), None);
        assert!(out.contains("Content: 42"));
        assert!(!out.contains("H7"));
    }

    // ── format_hot_entries ───────────────────────────────────────────────

    #[test]
    fn format_hot_entries_empty() {
        let chunks: Vec<crate::HierarchicalChunk> = vec![];
        let top: Vec<(usize, crate::salience::SalienceScore)> = vec![];
        let out = super::format_hot_entries(&chunks, &top);
        assert!(out.contains("## Hot Entries"));
        assert!(out.contains("_0 entry(ies)._"));
    }

    #[test]
    fn format_hot_entries_shows_score() {
        use crate::salience::SalienceScore;
        use crate::test_helpers::make_test_chunk;

        let chunk = make_test_chunk("abc1234deadbeef1234567890abcdef12345678", "Test content");
        let chunks = vec![chunk];
        let score = SalienceScore {
            composite: 0.75,
            interaction: 0.8,
            perspective: 0.7,
            revision: 0.0,
        };
        let top = vec![(0usize, score)];
        let out = super::format_hot_entries(&chunks, &top);
        assert!(out.contains("0.750"));
        assert!(out.contains("_1 entry(ies)._"));
    }

    #[test]
    fn format_hot_entries_with_perspectives() {
        use crate::salience::SalienceScore;
        use crate::test_helpers::make_test_chunk;

        let mut chunk = make_test_chunk("abc1234deadbeef1234567890abcdef12345678", "Content");
        chunk.perspectives = vec!["decisions".to_string(), "learnings".to_string()];
        let chunks = vec![chunk];
        let score = SalienceScore {
            composite: 0.5,
            interaction: 0.5,
            perspective: 0.5,
            revision: 0.0,
        };
        let top = vec![(0usize, score)];
        let out = super::format_hot_entries(&chunks, &top);
        assert!(out.contains("decisions, learnings"));
    }

    // ── format_entry_detail ──────────────────────────────────────────────

    #[test]
    fn format_entry_detail_basic() {
        use crate::test_helpers::make_test_chunk;

        let chunk = make_test_chunk("abc1234deadbeef1234567890abcdef12345678", "Entry content");
        let children: Vec<crate::HierarchicalChunk> = vec![];
        let out = super::format_entry_detail(&chunk, &children);
        assert!(out.contains("Entry content"));
        assert!(out.contains("### Metadata"));
        assert!(out.contains("**ID:**"));
        assert!(out.contains("**Visibility:**"));
    }

    #[test]
    fn format_entry_detail_with_relations() {
        use crate::test_helpers::make_test_chunk;

        let mut chunk = make_test_chunk("abc1234deadbeef1234567890abcdef12345678", "Content");
        chunk.relations.push(crate::ChunkRelation::new(
            "supersedes",
            "older-entry-id-abc123456",
        ));
        let children: Vec<crate::HierarchicalChunk> = vec![];
        let out = super::format_entry_detail(&chunk, &children);
        assert!(out.contains("### Relations (1)"));
        assert!(out.contains("supersedes"));
    }

    #[test]
    fn format_entry_detail_with_children() {
        use crate::test_helpers::make_test_chunk;

        let parent = make_test_chunk("abc1234deadbeef1234567890abcdef12345678", "Parent");
        let child = make_test_chunk("child12deadbeef1234567890abcdef123456789", "Child content");
        let out = super::format_entry_detail(&parent, &[child]);
        assert!(out.contains("### Children (1)"));
        assert!(out.contains("Child content"));
    }

    #[test]
    fn format_entry_detail_with_perspectives() {
        use crate::test_helpers::make_test_chunk;

        let mut chunk = make_test_chunk("abc1234deadbeef1234567890abcdef12345678", "Content");
        chunk.perspectives = vec!["decisions".to_string()];
        let out = super::format_entry_detail(&chunk, &[]);
        assert!(out.contains("**Perspectives:** decisions"));
    }

    #[test]
    fn format_entry_detail_no_children_section_when_empty() {
        use crate::test_helpers::make_test_chunk;

        let chunk = make_test_chunk("abc1234deadbeef1234567890abcdef12345678", "Content");
        let out = super::format_entry_detail(&chunk, &[]);
        assert!(!out.contains("### Children"));
    }

    // ── content_preview ──────────────────────────────────────────────────

    #[test]
    fn content_preview_short_content_returned_in_full() {
        let s = "Short content";
        let preview = content_preview(s, 300);
        assert_eq!(preview, s);
    }

    #[test]
    fn content_preview_truncates_at_char_boundary() {
        let s = "Hello 日本語 world and more text";
        let preview = content_preview(s, 9);
        assert!(s.is_char_boundary(preview.len()) || preview.len() <= 9);
        assert!(!preview.is_empty());
    }

    // ── first_line helper via recall ─────────────────────────────────────

    #[test]
    fn recall_uses_untitled_for_blank_content() {
        let mut chunk = make_chunk("abc1234deadbeef", "\n\n  \n", None);
        chunk.heading = None;
        let out = fmt_recall(chunk);
        assert!(out.contains("(untitled)"));
    }

    // ── Fix B: cap hint in MCP footer ────────────────────────────────────

    /// Helper: build N identical-looking (but distinct-id) SearchResultResponse entries.
    fn make_results(n: usize) -> Vec<SearchResultResponse> {
        (0..n)
            .map(|i| SearchResultResponse {
                chunk: make_chunk(
                    &format!("{:016x}", i),
                    &format!("Content {i}"),
                    Some(&format!("Entry {i}")),
                ),
                score: 0.5,
                relevance: "strong".to_string(),
                hierarchy_path: vec![],
                children: vec![],
            })
            .collect()
    }

    #[test]
    fn recall_footer_shows_cap_hint_when_result_set_fills_requested_limit() {
        // Requested 5, got 5 → the store may have more → hint shown.
        let results = make_results(5);
        let out = format_recall(Some("q"), &results, 5);
        assert!(
            out.contains("more may be available"),
            "expected cap hint in: {out}"
        );
    }

    #[test]
    fn recall_footer_omits_cap_hint_below_requested_limit() {
        // Requested 5, got 4 → store had nothing more → no hint.
        let results = make_results(4);
        let out = format_recall(Some("q"), &results, 5);
        assert!(
            !out.contains("more may be available"),
            "unexpected cap hint in: {out}"
        );
    }

    #[test]
    fn recall_footer_shows_cap_hint_when_higher_limit_filled() {
        // Caller raised top_k to 10 and got 10 → hint shown.
        let results = make_results(10);
        let out = format_recall(Some("q"), &results, 10);
        assert!(
            out.contains("more may be available"),
            "expected cap hint in: {out}"
        );
    }

    #[test]
    fn recall_footer_omits_cap_hint_when_higher_limit_not_filled() {
        // Regression: caller raised top_k to 10 but only 8 matched. The old
        // logic anchored the hint to a hardcoded default of 5 (8 >= 5) and
        // wrongly claimed more may be available. Anchored to the real limit,
        // 8 < 10 → no hint.
        let results = make_results(8);
        let out = format_recall(Some("q"), &results, 10);
        assert!(
            !out.contains("more may be available"),
            "cap hint must not fire when result set is below the requested limit: {out}"
        );
    }
}
