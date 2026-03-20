//! MCP resource endpoints — browsable, application-controlled access to the knowledge store.
//!
//! Exposes a `veclayer://` URI scheme with static resources (status, perspectives,
//! hot, recent, identity) and two URI templates (perspectives/{id}, entries/{id}).

use std::path::Path;

use chrono::{DateTime, Utc};
use rmcp::model::{
    AnnotateAble, RawResource, RawResourceTemplate, ReadResourceResult, Resource, ResourceContents,
    ResourceTemplate, Role,
};

use crate::store::{StoreBackend, VectorStore};

use super::format;
use super::tools::passes_scope_filter;

// ---------------------------------------------------------------------------
// Static resources
// ---------------------------------------------------------------------------

/// Return the 5 fixed resources with annotations.
pub fn static_resources() -> Vec<Resource> {
    let now = Utc::now();

    vec![
        make_resource(
            "veclayer://status",
            "Store status",
            "Entry counts, levels, sources, aging policy, pending embeddings",
            0.5,
            now,
            vec![Role::User, Role::Assistant],
        ),
        make_resource(
            "veclayer://perspectives",
            "Perspectives",
            "All perspectives with hints and builtin flags",
            0.4,
            now,
            vec![Role::Assistant],
        ),
        make_resource(
            "veclayer://hot",
            "Hot entries",
            "Top 10 entries by salience score",
            0.7,
            now,
            vec![Role::Assistant],
        ),
        make_resource(
            "veclayer://recent",
            "Recent entries",
            "Last 10 entries by creation time",
            0.6,
            now,
            vec![Role::Assistant],
        ),
        make_resource(
            "veclayer://identity",
            "Identity briefing",
            "Live identity briefing — core knowledge, open threads, learnings",
            0.8,
            now,
            vec![Role::Assistant],
        ),
    ]
}

/// Return the 2 URI templates.
pub fn templates() -> Vec<ResourceTemplate> {
    let now = Utc::now();

    vec![
        make_template(
            "veclayer://perspectives/{perspective_id}",
            "Perspective entries",
            "Browse the 10 most recent entries in a perspective",
            0.3,
            now,
        ),
        make_template(
            "veclayer://entries/{entry_id}",
            "Entry detail",
            "Full entry with children, relations, and access profile",
            0.5,
            now,
        ),
    ]
}

// ---------------------------------------------------------------------------
// Read dispatch
// ---------------------------------------------------------------------------

/// Dispatch a `resources/read` request by URI.
pub async fn read(
    uri: &str,
    store: &StoreBackend,
    data_dir: &Path,
    project: Option<&str>,
    branch: Option<&str>,
) -> Result<ReadResourceResult, rmcp::ErrorData> {
    let path = uri
        .strip_prefix("veclayer://")
        .ok_or_else(|| rmcp::ErrorData::invalid_params("URI must start with veclayer://", None))?;

    match path {
        "status" => read_status(uri, store, data_dir).await,
        "perspectives" => read_perspectives(uri, data_dir),
        "hot" => read_hot(uri, store, project, branch).await,
        "recent" => read_recent(uri, store, project, branch).await,
        "identity" => read_identity(uri, store, data_dir, project, branch).await,
        other => {
            if let Some(perspective_id) = other.strip_prefix("perspectives/") {
                read_perspective_entries(uri, store, data_dir, perspective_id, project, branch)
                    .await
            } else if let Some(entry_id) = other.strip_prefix("entries/") {
                read_entry(uri, store, entry_id).await
            } else {
                Err(rmcp::ErrorData::invalid_params(
                    format!("Unknown resource URI: {uri}"),
                    None,
                ))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Individual resource readers
// ---------------------------------------------------------------------------

async fn read_status(
    uri: &str,
    store: &StoreBackend,
    data_dir: &Path,
) -> Result<ReadResourceResult, rmcp::ErrorData> {
    let stats = store.stats().await.map_err(|e| {
        rmcp::ErrorData::internal_error(format!("Failed to read store stats: {e}"), None)
    })?;
    let aging_config = crate::aging::AgingConfig::load(data_dir);
    let md = format::format_store_status(&stats, &aging_config);
    Ok(text_resource(uri, &md))
}

fn read_perspectives(uri: &str, data_dir: &Path) -> Result<ReadResourceResult, rmcp::ErrorData> {
    let perspectives = crate::perspective::load(data_dir).map_err(|e| {
        rmcp::ErrorData::internal_error(format!("Failed to load perspectives: {e}"), None)
    })?;

    if perspectives.is_empty() {
        return Ok(text_resource(uri, "No perspectives defined."));
    }

    let mut md = String::from("## Perspectives\n\n");
    for p in &perspectives {
        let tag = if p.builtin { " [builtin]" } else { "" };
        md.push_str(&format!("- **{}** — {}{}\n", p.id, p.hint, tag));
    }
    md.push_str(&format!("\n{} perspective(s) total.", perspectives.len()));

    Ok(text_resource(uri, &md))
}

async fn read_hot(
    uri: &str,
    store: &StoreBackend,
    project: Option<&str>,
    branch: Option<&str>,
) -> Result<ReadResourceResult, rmcp::ErrorData> {
    let hot = store.get_hot_chunks(20).await.map_err(|e| {
        rmcp::ErrorData::internal_error(format!("Failed to get hot chunks: {e}"), None)
    })?;

    let filtered: Vec<_> = hot
        .into_iter()
        .filter(|c| passes_scope_filter(c, project, branch))
        .collect();

    if filtered.is_empty() {
        return Ok(text_resource(uri, "No entries to display."));
    }

    let weights = crate::salience::SalienceWeights::default();
    let top = crate::salience::top_salient(&filtered, &weights, 10);
    let md = format::format_hot_entries(&filtered, &top);

    Ok(text_resource(uri, &md))
}

async fn read_recent(
    uri: &str,
    store: &StoreBackend,
    project: Option<&str>,
    branch: Option<&str>,
) -> Result<ReadResourceResult, rmcp::ErrorData> {
    let entries = store
        .list_entries(None, None, None, 20)
        .await
        .map_err(|e| {
            rmcp::ErrorData::internal_error(format!("Failed to list entries: {e}"), None)
        })?;

    let filtered: Vec<_> = entries
        .into_iter()
        .filter(|c| passes_scope_filter(c, project, branch))
        .take(10)
        .collect();

    if filtered.is_empty() {
        return Ok(text_resource(uri, "No entries found."));
    }

    let mut md = String::from("## Recent Entries\n\n");
    for chunk in &filtered {
        let heading = chunk.heading.as_deref().unwrap_or("(no heading)");
        let short = crate::chunk::short_id(&chunk.id);
        let created = format_epoch(chunk.access_profile.created_at);
        md.push_str(&format!("- **{heading}** `{short}` — {created}\n"));
    }
    md.push_str(&format!("\n_{} entry(ies)._\n", filtered.len()));

    Ok(text_resource(uri, &md))
}

async fn read_identity(
    uri: &str,
    store: &StoreBackend,
    data_dir: &Path,
    project: Option<&str>,
    branch: Option<&str>,
) -> Result<ReadResourceResult, rmcp::ErrorData> {
    let snapshot = crate::identity::compute_identity(store, data_dir, project, branch)
        .await
        .map_err(|e| {
            rmcp::ErrorData::internal_error(format!("Failed to compute identity: {e}"), None)
        })?;

    let priming = crate::identity::generate_priming(&snapshot);
    if priming.is_empty() {
        return Ok(text_resource(uri, "No identity data yet."));
    }

    Ok(text_resource(uri, &priming))
}

async fn read_perspective_entries(
    uri: &str,
    store: &StoreBackend,
    data_dir: &Path,
    perspective_id: &str,
    project: Option<&str>,
    branch: Option<&str>,
) -> Result<ReadResourceResult, rmcp::ErrorData> {
    // Validate the perspective exists
    let perspectives = crate::perspective::load(data_dir).map_err(|e| {
        rmcp::ErrorData::internal_error(format!("Failed to load perspectives: {e}"), None)
    })?;
    if !perspectives.iter().any(|p| p.id == perspective_id) {
        return Err(rmcp::ErrorData::invalid_params(
            format!("Perspective '{perspective_id}' not found"),
            None,
        ));
    }

    let entries = store
        .list_entries(Some(perspective_id), None, None, 20)
        .await
        .map_err(|e| {
            rmcp::ErrorData::internal_error(format!("Failed to list entries: {e}"), None)
        })?;

    let filtered: Vec<_> = entries
        .into_iter()
        .filter(|c| passes_scope_filter(c, project, branch))
        .take(10)
        .collect();

    let mut md = format!("## Perspective: {perspective_id}\n\n");
    if filtered.is_empty() {
        md.push_str("No entries in this perspective.\n");
    } else {
        for chunk in &filtered {
            let heading = chunk.heading.as_deref().unwrap_or("(no heading)");
            let short = crate::chunk::short_id(&chunk.id);
            let created = format_epoch(chunk.access_profile.created_at);
            md.push_str(&format!("- **{heading}** `{short}` — {created}\n"));
        }
        md.push_str(&format!("\n_{} entry(ies)._\n", filtered.len()));
    }

    Ok(text_resource(uri, &md))
}

async fn read_entry(
    uri: &str,
    store: &StoreBackend,
    entry_id: &str,
) -> Result<ReadResourceResult, rmcp::ErrorData> {
    // Resolve short ID to full ID, then fetch
    let chunk = store
        .get_by_id_prefix(entry_id)
        .await
        .map_err(|e| {
            rmcp::ErrorData::invalid_params(format!("Failed to resolve entry ID: {e}"), None)
        })?
        .ok_or_else(|| {
            rmcp::ErrorData::invalid_params(format!("Entry '{entry_id}' not found"), None)
        })?;

    let children = store.get_children(&chunk.id).await.map_err(|e| {
        rmcp::ErrorData::internal_error(format!("Failed to get children: {e}"), None)
    })?;

    let md = format::format_entry_detail(&chunk, &children);
    Ok(text_resource(uri, &md))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_resource(
    uri: &str,
    name: &str,
    description: &str,
    priority: f32,
    timestamp: DateTime<Utc>,
    audience: Vec<Role>,
) -> Resource {
    RawResource::new(uri, name)
        .with_description(description)
        .with_mime_type("text/markdown")
        .with_priority(priority)
        .with_timestamp(timestamp)
        .with_audience(audience)
}

fn make_template(
    uri_template: &str,
    name: &str,
    description: &str,
    priority: f32,
    timestamp: DateTime<Utc>,
) -> ResourceTemplate {
    RawResourceTemplate::new(uri_template, name)
        .with_description(description)
        .with_mime_type("text/markdown")
        .with_priority(priority)
        .with_timestamp(timestamp)
        .with_audience(vec![Role::Assistant])
}

fn text_resource(uri: &str, text: &str) -> ReadResourceResult {
    ReadResourceResult::new(vec![ResourceContents::TextResourceContents {
        uri: uri.to_string(),
        mime_type: Some("text/markdown".to_string()),
        text: text.to_string(),
        meta: None,
    }])
}

/// Format a unix epoch timestamp as a human-readable date string.
fn format_epoch(epoch: i64) -> String {
    DateTime::from_timestamp(epoch, 0)
        .map(|dt: DateTime<Utc>| dt.format("%Y-%m-%d %H:%M UTC").to_string())
        .unwrap_or_else(|| "(unknown)".to_string())
}

#[cfg(all(test, feature = "store-lance"))]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ── static_resources ────────────────────────────────────────────────

    #[test]
    fn static_resources_returns_five_entries() {
        let resources = static_resources();
        assert_eq!(resources.len(), 5);
    }

    #[test]
    fn static_resources_uris_are_veclayer_scheme() {
        for r in static_resources() {
            assert!(
                r.raw.uri.starts_with("veclayer://"),
                "URI '{}' should use veclayer:// scheme",
                r.raw.uri
            );
        }
    }

    #[test]
    fn static_resources_all_have_text_markdown_mime() {
        for r in static_resources() {
            assert_eq!(
                r.raw.mime_type.as_deref(),
                Some("text/markdown"),
                "Resource '{}' should have text/markdown MIME",
                r.raw.uri
            );
        }
    }

    #[test]
    fn static_resources_known_uris_present() {
        let uris: Vec<String> = static_resources()
            .into_iter()
            .map(|r| r.raw.uri.clone())
            .collect();
        assert!(uris.contains(&"veclayer://status".to_string()));
        assert!(uris.contains(&"veclayer://perspectives".to_string()));
        assert!(uris.contains(&"veclayer://hot".to_string()));
        assert!(uris.contains(&"veclayer://recent".to_string()));
        assert!(uris.contains(&"veclayer://identity".to_string()));
    }

    #[test]
    fn static_resources_have_audience_set() {
        let resources = static_resources();
        for r in &resources {
            let has_audience = r
                .annotations
                .as_ref()
                .and_then(|a| a.audience.as_ref())
                .map(|a| !a.is_empty())
                .unwrap_or(false);
            assert!(
                has_audience,
                "Resource '{}' should have audience set",
                r.raw.uri
            );
        }
    }

    // ── templates ────────────────────────────────────────────────────────

    #[test]
    fn templates_returns_two_entries() {
        let tmpl = templates();
        assert_eq!(tmpl.len(), 2);
    }

    #[test]
    fn templates_have_uri_params() {
        let tmpl = templates();
        let uris: Vec<&str> = tmpl.iter().map(|t| t.raw.uri_template.as_str()).collect();
        assert!(uris.iter().any(|u| u.contains("{perspective_id}")));
        assert!(uris.iter().any(|u| u.contains("{entry_id}")));
    }

    #[test]
    fn templates_all_have_text_markdown_mime() {
        for t in templates() {
            assert_eq!(
                t.raw.mime_type.as_deref(),
                Some("text/markdown"),
                "Template '{}' should have text/markdown MIME",
                t.raw.uri_template
            );
        }
    }

    #[test]
    fn templates_have_audience_set() {
        for t in templates() {
            let has_audience = t
                .annotations
                .as_ref()
                .and_then(|a| a.audience.as_ref())
                .map(|a| !a.is_empty())
                .unwrap_or(false);
            assert!(
                has_audience,
                "Template '{}' should have audience",
                t.raw.uri_template
            );
        }
    }

    // ── read dispatch ────────────────────────────────────────────────────

    #[tokio::test]
    async fn read_rejects_non_veclayer_uri() {
        let dir = TempDir::new().unwrap();
        let store = std::sync::Arc::new(
            crate::store::StoreBackend::open_metadata(dir.path(), false)
                .await
                .unwrap(),
        );
        let result = read("https://example.com/foo", &store, dir.path(), None, None).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{:?}", err).contains("veclayer://"));
    }

    #[tokio::test]
    async fn read_rejects_unknown_path() {
        let dir = TempDir::new().unwrap();
        let store = std::sync::Arc::new(
            crate::store::StoreBackend::open_metadata(dir.path(), false)
                .await
                .unwrap(),
        );
        let result = read(
            "veclayer://unknown_path_xyz",
            &store,
            dir.path(),
            None,
            None,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn read_status_returns_markdown() {
        let dir = TempDir::new().unwrap();
        let store = std::sync::Arc::new(
            crate::store::StoreBackend::open_metadata(dir.path(), false)
                .await
                .unwrap(),
        );
        let result = read("veclayer://status", &store, dir.path(), None, None)
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("## Store Status"));
    }

    #[tokio::test]
    async fn read_perspectives_returns_default_perspectives() {
        let dir = TempDir::new().unwrap();
        let store = std::sync::Arc::new(
            crate::store::StoreBackend::open_metadata(dir.path(), false)
                .await
                .unwrap(),
        );
        let result = read("veclayer://perspectives", &store, dir.path(), None, None)
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("## Perspectives"));
        assert!(text.contains("decisions"));
    }

    #[tokio::test]
    async fn read_hot_returns_no_entries_when_store_empty() {
        let dir = TempDir::new().unwrap();
        let store = std::sync::Arc::new(
            crate::store::StoreBackend::open_metadata(dir.path(), false)
                .await
                .unwrap(),
        );
        let result = read("veclayer://hot", &store, dir.path(), None, None)
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("No entries"));
    }

    #[tokio::test]
    async fn read_hot_returns_entries_when_store_has_data() {
        let dir = TempDir::new().unwrap();
        let store = std::sync::Arc::new(
            crate::store::StoreBackend::open(dir.path(), 384, false)
                .await
                .unwrap(),
        );
        let mut chunk =
            crate::test_helpers::make_test_chunk("hotentry001", "Very important decision");
        chunk.access_profile.record_access();
        store.insert_chunks(vec![chunk]).await.unwrap();

        let result = read("veclayer://hot", &store, dir.path(), None, None)
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(!text.contains("No entries"));
    }

    #[tokio::test]
    async fn read_hot_with_project_filter_excludes_unscoped() {
        let dir = TempDir::new().unwrap();
        let store = std::sync::Arc::new(
            crate::store::StoreBackend::open(dir.path(), 384, false)
                .await
                .unwrap(),
        );
        let mut chunk =
            crate::test_helpers::make_test_chunk("hotentry002", "Unscoped knowledge entry");
        chunk.perspectives = vec!["project:other-project".to_string()];
        store.insert_chunks(vec![chunk]).await.unwrap();

        let result = read(
            "veclayer://hot",
            &store,
            dir.path(),
            Some("my-project"),
            None,
        )
        .await
        .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("No entries"));
    }

    #[tokio::test]
    async fn read_recent_returns_no_entries_when_store_empty() {
        let dir = TempDir::new().unwrap();
        let store = std::sync::Arc::new(
            crate::store::StoreBackend::open_metadata(dir.path(), false)
                .await
                .unwrap(),
        );
        let result = read("veclayer://recent", &store, dir.path(), None, None)
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("No entries"));
    }

    #[tokio::test]
    async fn read_recent_returns_entries_when_store_has_data() {
        let dir = TempDir::new().unwrap();
        let store = std::sync::Arc::new(
            crate::store::StoreBackend::open(dir.path(), 384, false)
                .await
                .unwrap(),
        );
        let chunk = crate::test_helpers::make_test_chunk("recententry01", "Recent knowledge entry");
        store.insert_chunks(vec![chunk]).await.unwrap();

        let result = read("veclayer://recent", &store, dir.path(), None, None)
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("## Recent Entries"));
        assert!(text.contains("entry(ies)"));
    }

    #[tokio::test]
    async fn read_recent_with_project_filter_excludes_unscoped() {
        let dir = TempDir::new().unwrap();
        let store = std::sync::Arc::new(
            crate::store::StoreBackend::open(dir.path(), 384, false)
                .await
                .unwrap(),
        );
        let mut chunk =
            crate::test_helpers::make_test_chunk("recententry02", "Other project entry");
        chunk.perspectives = vec!["project:other-project".to_string()];
        store.insert_chunks(vec![chunk]).await.unwrap();

        let result = read(
            "veclayer://recent",
            &store,
            dir.path(),
            Some("my-project"),
            None,
        )
        .await
        .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("No entries found"));
    }

    #[tokio::test]
    async fn read_identity_returns_no_identity_when_store_empty() {
        let dir = TempDir::new().unwrap();
        let store = std::sync::Arc::new(
            crate::store::StoreBackend::open_metadata(dir.path(), false)
                .await
                .unwrap(),
        );
        let result = read("veclayer://identity", &store, dir.path(), None, None)
            .await
            .unwrap();
        let text = extract_text(&result);
        // Empty store → no identity data
        assert!(!text.is_empty());
    }

    #[tokio::test]
    async fn read_perspective_entries_rejects_unknown_perspective() {
        let dir = TempDir::new().unwrap();
        let store = std::sync::Arc::new(
            crate::store::StoreBackend::open_metadata(dir.path(), false)
                .await
                .unwrap(),
        );
        let result = read(
            "veclayer://perspectives/nonexistent_xyz",
            &store,
            dir.path(),
            None,
            None,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn read_perspective_entries_accepts_builtin_perspective() {
        let dir = TempDir::new().unwrap();
        let store = std::sync::Arc::new(
            crate::store::StoreBackend::open_metadata(dir.path(), false)
                .await
                .unwrap(),
        );
        let result = read(
            "veclayer://perspectives/decisions",
            &store,
            dir.path(),
            None,
            None,
        )
        .await
        .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("## Perspective: decisions"));
    }

    #[tokio::test]
    async fn read_perspective_entries_shows_entries() {
        let dir = TempDir::new().unwrap();
        let store = std::sync::Arc::new(
            crate::store::StoreBackend::open(dir.path(), 384, false)
                .await
                .unwrap(),
        );
        let mut chunk =
            crate::test_helpers::make_test_chunk("perspentry01", "A decision about databases");
        chunk.perspectives = vec!["decisions".to_string()];
        store.insert_chunks(vec![chunk]).await.unwrap();

        let result = read(
            "veclayer://perspectives/decisions",
            &store,
            dir.path(),
            None,
            None,
        )
        .await
        .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("## Perspective: decisions"));
        assert!(text.contains("entry(ies)"));
    }

    #[tokio::test]
    async fn read_entry_returns_error_for_unknown_id() {
        let dir = TempDir::new().unwrap();
        let store = std::sync::Arc::new(
            crate::store::StoreBackend::open_metadata(dir.path(), false)
                .await
                .unwrap(),
        );
        let result = read(
            "veclayer://entries/nonexistent000",
            &store,
            dir.path(),
            None,
            None,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn read_entry_returns_detail_for_known_entry() {
        let dir = TempDir::new().unwrap();
        let store = std::sync::Arc::new(
            crate::store::StoreBackend::open(dir.path(), 384, false)
                .await
                .unwrap(),
        );
        let chunk = crate::test_helpers::make_test_chunk("abc1230000", "Design decision: use Rust");
        store.insert_chunks(vec![chunk]).await.unwrap();

        let result = read(
            "veclayer://entries/abc1230000",
            &store,
            dir.path(),
            None,
            None,
        )
        .await
        .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("Design decision: use Rust"));
    }

    // ── format_epoch ─────────────────────────────────────────────────────

    #[test]
    fn format_epoch_known_timestamp() {
        // 2024-01-15 12:00:00 UTC = 1705320000
        let result = format_epoch(1705320000);
        assert!(result.contains("2024-01-15"));
        assert!(result.contains("UTC"));
    }

    #[test]
    fn format_epoch_zero_is_unix_epoch() {
        let result = format_epoch(0);
        assert!(result.contains("1970-01-01"));
    }

    #[test]
    fn format_epoch_negative_out_of_range_returns_unknown() {
        let result = format_epoch(i64::MIN);
        assert_eq!(result, "(unknown)");
    }

    // ── helpers ──────────────────────────────────────────────────────────

    /// Extract text content from a ReadResourceResult.
    fn extract_text(result: &ReadResourceResult) -> String {
        result
            .contents
            .iter()
            .filter_map(|c| {
                if let rmcp::model::ResourceContents::TextResourceContents { text, .. } = c {
                    Some(text.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("")
    }
}
