//! Sync command: pull entries from git-based memory scopes into the local index.

use std::path::Path;

use crate::config::ResolvedScope;
use crate::entry::Entry;
use crate::git::detect;
use crate::git::memory_store::MemoryStore;
use crate::git::PushResult;
use crate::store::VectorStore as _;

/// Sync entries from all configured git-backed scopes.
///
/// For each scope with `storage = "git"` (same-repo orphan branch), opens the
/// [`MemoryStore`], loads all entries, and reports the count. Remote git URLs
/// are not yet supported and are reported as skipped.
///
/// This is a read-only skeleton: entries are loaded and counted but not yet
/// inserted into LanceDB (that requires the embedder and is a separate step).
pub async fn sync(_data_dir: &Path, scopes: &[ResolvedScope]) -> crate::Result<()> {
    if scopes.is_empty() {
        println!("No scopes configured. Add scopes to your config:");
        println!("  veclayer init --share    # enable git memory for this project");
        return Ok(());
    }

    println!("Checking {} scope(s)...", scopes.len());

    let cwd = std::env::current_dir()?;

    for scope in scopes {
        sync_scope(&cwd, scope);
    }

    Ok(())
}

fn sync_scope(cwd: &Path, scope: &ResolvedScope) {
    if scope.storage == "git" {
        sync_local_git_scope(cwd, scope);
    } else if is_remote_git_url(&scope.storage) {
        println!(
            "  {} — remote storage not yet supported ({})",
            scope.name, scope.storage
        );
    } else {
        println!("  {} — unknown storage type: {}", scope.name, scope.storage);
    }
}

fn sync_local_git_scope(cwd: &Path, scope: &ResolvedScope) {
    let Some(git_dir) = detect::find_git_dir(cwd) else {
        println!("  {} — skipped (not a git repository)", scope.name);
        return;
    };

    match MemoryStore::open(&git_dir, Some(&scope.branch)) {
        Ok(store) => report_loaded_entries(scope, &store),
        Err(e) => println!("  {} — error opening: {}", scope.name, e),
    }
}

fn report_loaded_entries(scope: &ResolvedScope, store: &MemoryStore) {
    match store.load() {
        Ok(entries) => println!(
            "  {} — {} entries from branch '{}'",
            scope.name,
            entries.len(),
            scope.branch,
        ),
        Err(e) => println!("  {} — error loading: {}", scope.name, e),
    }
}

// ---------------------------------------------------------------------------
// Filters for migrate
// ---------------------------------------------------------------------------

/// Filters that can be applied when migrating entries from LanceDB to git.
#[derive(Debug, Default)]
pub struct MigrateFilters {
    /// Only include entries with this perspective.
    pub perspective: Option<String>,
    /// Exclude entries with this perspective.
    pub exclude_perspective: Option<String>,
    /// Only include entries created after this Unix epoch timestamp.
    pub since: Option<i64>,
}

impl MigrateFilters {
    /// Return `true` if `entry` passes all active filters.
    pub fn accepts(&self, entry: &crate::HierarchicalChunk) -> bool {
        if let Some(ref p) = self.perspective {
            if !entry.perspectives.contains(p) {
                return false;
            }
        }
        if let Some(ref ep) = self.exclude_perspective {
            if entry.perspectives.contains(ep) {
                return false;
            }
        }
        if let Some(since_ts) = self.since {
            if entry.access_profile.created_at < since_ts {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Migrate
// ---------------------------------------------------------------------------

/// Export all local LanceDB entries to the git memory branch.
///
/// Reads every entry from the local store, converts it to [`Entry`] via
/// [`Entry::from_chunk`], and writes each one as a Markdown file on the
/// `veclayer-memory` branch. Prints a summary when done.
///
/// Entries are filtered by `filters` before writing.
pub async fn migrate(data_dir: &Path, filters: &MigrateFilters) -> crate::Result<()> {
    let cwd = std::env::current_dir()?;

    let git_dir = detect::find_git_dir(&cwd).ok_or_else(|| {
        crate::Error::InvalidOperation(
            "Not a git project. Run `veclayer init --share` first.".into(),
        )
    })?;

    let git_store = MemoryStore::open(&git_dir, None).map_err(|e| {
        crate::Error::InvalidOperation(format!("Failed to open memory branch: {e}"))
    })?;

    let store = crate::store::StoreBackend::open_metadata(data_dir, true).await?;
    let all_chunks = store.list_entries(None, None, None, usize::MAX).await?;

    if all_chunks.is_empty() {
        println!("No local entries to migrate.");
        return Ok(());
    }

    let filtered: Vec<_> = all_chunks.iter().filter(|c| filters.accepts(c)).collect();

    if filtered.is_empty() {
        println!("No entries match the specified filters.");
        return Ok(());
    }

    println!(
        "Migrating {} entries to veclayer-memory branch...",
        filtered.len()
    );
    if filtered.len() < all_chunks.len() {
        println!(
            "  ({} skipped by filters)",
            all_chunks.len() - filtered.len()
        );
    }

    let mut success = 0usize;
    let mut failed = 0usize;

    for chunk in &filtered {
        let entry = Entry::from_chunk(chunk);
        match git_store.store_entry(&entry) {
            Ok(_) => success += 1,
            Err(e) => {
                let short = &chunk.id[..7.min(chunk.id.len())];
                tracing::warn!("Failed to migrate entry {short}: {e}");
                failed += 1;
            }
        }
    }

    println!("  {success} entries written as Markdown");
    if failed > 0 {
        println!("  {failed} entries failed");
    }
    println!();
    println!("Entries are committed locally. Push the branch with:");
    println!("  veclayer sync --push");

    Ok(())
}

// ---------------------------------------------------------------------------
// Pending
// ---------------------------------------------------------------------------

/// List entries on the local git branch that haven't been pushed to remote.
///
/// Shows the entry ID, heading (truncated), date, and perspectives for each
/// unpushed commit. If no remote tracking branch exists, all entries are listed.
pub async fn show_pending() -> crate::Result<()> {
    let cwd = std::env::current_dir()?;

    let git_dir = detect::find_git_dir(&cwd)
        .ok_or_else(|| crate::Error::InvalidOperation("Not a git repository.".into()))?;

    let git_store = MemoryStore::open(&git_dir, None).map_err(|e| {
        crate::Error::InvalidOperation(format!("Failed to open memory branch: {e}"))
    })?;

    let count = git_store.unpushed_commit_count().map_err(|e| {
        crate::Error::InvalidOperation(format!("Failed to count unpushed commits: {e}"))
    })?;

    if count == 0 {
        println!("Nothing pending — memory branch is up to date with remote.");
        return Ok(());
    }

    let entries = git_store
        .load()
        .map_err(|e| crate::Error::InvalidOperation(format!("Failed to load entries: {e}")))?;

    println!("{count} commit(s) not yet pushed to remote. Entries on branch:");
    println!();

    // Display all entries on the branch (we can't easily map commits to entries
    // without more git plumbing, so show the full current state).
    if entries.is_empty() {
        println!("  (no entries on branch)");
    } else {
        print_entry_list(&entries);
    }

    println!();
    println!("Push with: veclayer sync --push");

    Ok(())
}

/// Print a concise list of entries for display.
fn print_entry_list(entries: &[crate::entry::Entry]) {
    use chrono::{DateTime, Utc};

    for entry in entries {
        let content_id = entry.content_id();
        let id = crate::chunk::short_id(&content_id);
        let heading = entry
            .heading
            .as_deref()
            .unwrap_or_else(|| entry.content.split('\n').next().unwrap_or("(empty)"));
        let heading_preview: String = if heading.chars().count() > 60 {
            let truncated: String = heading.chars().take(60).collect();
            format!("{truncated}...")
        } else {
            heading.to_string()
        };

        let date = DateTime::<Utc>::from_timestamp(entry.created_at, 0)
            .map(|dt| dt.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| "?".to_string());

        let perspectives = if entry.perspectives.is_empty() {
            String::new()
        } else {
            format!(" [{}]", entry.perspectives.join(", "))
        };

        println!("  {id}  {date}  {heading_preview}{perspectives}");
    }
}

// ---------------------------------------------------------------------------
// Push
// ---------------------------------------------------------------------------

/// Push the local git memory branch to remote.
pub async fn push_to_remote() -> crate::Result<()> {
    let cwd = std::env::current_dir()?;

    let git_dir = detect::find_git_dir(&cwd)
        .ok_or_else(|| crate::Error::InvalidOperation("Not a git repository.".into()))?;

    let git_store = MemoryStore::open(&git_dir, None).map_err(|e| {
        crate::Error::InvalidOperation(format!("Failed to open memory branch: {e}"))
    })?;

    match git_store.push() {
        Ok(PushResult::Success) => {
            println!("Pushed veclayer-memory to remote.");
        }
        Ok(PushResult::NothingToPush) => {
            println!("Nothing to push — already up to date.");
        }
        Ok(PushResult::Rejected) => {
            return Err(crate::Error::InvalidOperation(
                "Push rejected: remote has diverged. Fetch and rebase first.".into(),
            ));
        }
        Err(crate::git::GitError::AuthFailed(msg)) => {
            return Err(crate::Error::InvalidOperation(format!(
                "Authentication failed: {msg}"
            )));
        }
        Err(e) => {
            return Err(crate::Error::InvalidOperation(format!("Push failed: {e}")));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Stage / Reject
// ---------------------------------------------------------------------------

/// Promote a local LanceDB entry to the git memory branch (for `manual` push mode).
///
/// Looks up `id` (or ID prefix) in the local LanceDB store, converts it to an
/// [`Entry`], and writes it to the git memory branch.
pub async fn stage_entry(data_dir: &Path, id: &str) -> crate::Result<()> {
    let cwd = std::env::current_dir()?;

    let git_dir = detect::find_git_dir(&cwd)
        .ok_or_else(|| crate::Error::InvalidOperation("Not a git repository.".into()))?;

    let git_store = MemoryStore::open(&git_dir, None).map_err(|e| {
        crate::Error::InvalidOperation(format!("Failed to open memory branch: {e}"))
    })?;

    let store = crate::store::StoreBackend::open_metadata(data_dir, true).await?;

    let chunk = store
        .get_by_id_prefix(id)
        .await?
        .ok_or_else(|| crate::Error::not_found(format!("Entry '{id}' not found")))?;

    let entry = Entry::from_chunk(&chunk);
    let short = crate::chunk::short_id(&chunk.id).to_string();

    git_store.store_entry(&entry).map_err(|e| {
        crate::Error::InvalidOperation(format!("Failed to stage entry {short}: {e}"))
    })?;

    let heading = entry.heading.as_deref().unwrap_or("(no heading)");
    println!("Staged {short}: {heading}");

    Ok(())
}

/// Remove an entry from the git memory branch (unstage).
///
/// Finds the entry file matching `id` (or ID prefix) on the memory branch,
/// removes it via `git rm`, and commits the removal.
pub async fn reject_entry(id: &str) -> crate::Result<()> {
    let cwd = std::env::current_dir()?;

    let git_dir = detect::find_git_dir(&cwd)
        .ok_or_else(|| crate::Error::InvalidOperation("Not a git repository.".into()))?;

    let git_store = MemoryStore::open(&git_dir, None).map_err(|e| {
        crate::Error::InvalidOperation(format!("Failed to open memory branch: {e}"))
    })?;

    git_store.remove_entry(id).map_err(|e| {
        crate::Error::InvalidOperation(format!("Failed to remove entry '{id}': {e}"))
    })?;

    println!("Removed entry '{id}' from veclayer-memory branch.");

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_remote_git_url(storage: &str) -> bool {
    storage.contains("git@")
        || storage.contains("github.com")
        || storage.contains("gitlab.com")
        || storage.ends_with(".git")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_remote_git_url_ssh() {
        assert!(is_remote_git_url("git@github.com:org/repo.git"));
    }

    #[test]
    fn test_is_remote_git_url_github() {
        assert!(is_remote_git_url("https://github.com/org/repo"));
    }

    #[test]
    fn test_is_remote_git_url_dotgit_suffix() {
        assert!(is_remote_git_url("https://example.com/repo.git"));
    }

    #[test]
    fn test_is_remote_git_url_not_remote() {
        assert!(!is_remote_git_url("git"));
        assert!(!is_remote_git_url("/local/path"));
        assert!(!is_remote_git_url("lancedb"));
    }

    #[test]
    fn test_migrate_filters_accepts_no_filter() {
        use crate::chunk::{ChunkLevel, HierarchicalChunk};
        let chunk = HierarchicalChunk::new(
            "content".to_string(),
            ChunkLevel::H1,
            None,
            String::new(),
            "src.md".to_string(),
        )
        .with_perspectives(vec!["decisions".to_string()]);

        let filters = MigrateFilters::default();
        assert!(filters.accepts(&chunk));
    }

    #[test]
    fn test_migrate_filters_perspective_include() {
        use crate::chunk::{ChunkLevel, HierarchicalChunk};
        let chunk = HierarchicalChunk::new(
            "content".to_string(),
            ChunkLevel::H1,
            None,
            String::new(),
            "src.md".to_string(),
        )
        .with_perspectives(vec!["decisions".to_string()]);

        let filters = MigrateFilters {
            perspective: Some("decisions".to_string()),
            ..Default::default()
        };
        assert!(filters.accepts(&chunk));

        let filters = MigrateFilters {
            perspective: Some("knowledge".to_string()),
            ..Default::default()
        };
        assert!(!filters.accepts(&chunk));
    }

    #[test]
    fn test_migrate_filters_perspective_exclude() {
        use crate::chunk::{ChunkLevel, HierarchicalChunk};
        let chunk = HierarchicalChunk::new(
            "content".to_string(),
            ChunkLevel::H1,
            None,
            String::new(),
            "src.md".to_string(),
        )
        .with_perspectives(vec!["decisions".to_string()]);

        let filters = MigrateFilters {
            exclude_perspective: Some("decisions".to_string()),
            ..Default::default()
        };
        assert!(!filters.accepts(&chunk));

        let filters = MigrateFilters {
            exclude_perspective: Some("knowledge".to_string()),
            ..Default::default()
        };
        assert!(filters.accepts(&chunk));
    }

    #[test]
    fn test_migrate_filters_since() {
        use crate::chunk::{ChunkLevel, HierarchicalChunk};
        let mut chunk = HierarchicalChunk::new(
            "content".to_string(),
            ChunkLevel::H1,
            None,
            String::new(),
            "src.md".to_string(),
        );
        chunk.access_profile.created_at = 1_000_000;

        let filters = MigrateFilters {
            since: Some(500_000),
            ..Default::default()
        };
        assert!(filters.accepts(&chunk));

        let filters = MigrateFilters {
            since: Some(2_000_000),
            ..Default::default()
        };
        assert!(!filters.accepts(&chunk));
    }
}
