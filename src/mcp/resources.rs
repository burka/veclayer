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
