use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, RecordBatchIterator,
    StringArray, UInt16Array, UInt32Array, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::NewColumnTransform;
use lancedb::{connect, Connection, Table};

use super::{
    CompactStats, FileLock, SearchResult, StoreStats, VectorStore, EMBEDDING_STATUS_EMBEDDED,
    EMBEDDING_STATUS_PENDING,
};
use crate::{ChunkLevel, ClusterMembership, Error, HierarchicalChunk, Result};

pub(crate) const TABLE_NAME: &str = "chunks";

/// Auto-compaction threshold: prune old versions when version count exceeds this.
const MAX_VERSIONS: usize = 50;

/// Byte offset into the source document at which this chunk starts.
/// `start_offset` and `end_offset` are not persisted in LanceDB; they are
/// populated only during the initial parsing pass and are set to zero when
/// a chunk is reconstructed from the LanceDB store.
const CHUNK_BYTE_OFFSET_NOT_STORED: usize = 0;

const EMBEDDING_EMBEDDED_FILTER: &str = "embedding_status = 'embedded'";
const EMBEDDING_PENDING_FILTER: &str = "embedding_status = 'pending'";
const EMBEDDING_EMBEDDED_SQL: &str = "'embedded'";

/// Deterministic fingerprint of an Arrow schema: hash of field names + types.
/// Changes automatically when fields are added, removed, or retyped.
/// Uses DataType's Display (delegates to Debug for non-Struct types in arrow-schema v55).
/// Stable within the ^55 pin; a major bump may change fingerprints, which is acceptable
/// since this is informational metadata, not a migration gate.
fn schema_fingerprint(schema: &Schema) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for field in schema.fields() {
        hasher.update(field.name().as_bytes());
        hasher.update(b":");
        hasher.update(format!("{}", field.data_type()).as_bytes());
        hasher.update(b"\n");
    }
    let hash = hasher.finalize();
    format!("{hash:x}")[..12].to_string()
}

/// Columns from older schema versions that are no longer in schema() but
/// may still exist in on-disk tables. These are NOT treated as "newer version".
const LEGACY_COLUMNS: &[&str] = &["last_accessed", "access_count", "is_summary"];

fn migration_default(field_name: &str) -> Option<&'static str> {
    match field_name {
        "cluster_memberships" | "summarizes" | "perspectives" | "relations" => Some("'[]'"),
        "entry_type" => Some("'raw'"),
        "visibility" => Some("'normal'"),
        "created_at" | "last_rolled" => Some("cast(0 as bigint)"),
        "access_hour" | "access_day" | "access_week" | "access_month" | "access_year" => {
            Some("cast(0 as smallint unsigned)")
        }
        "access_total" => Some("cast(0 as int unsigned)"),
        "expires_at" => Some("cast(NULL as bigint)"),
        "impression_hint" => Some("cast(NULL as string)"),
        "impression_strength" => Some("cast(1.0 as float)"),
        "embedding_status" => Some(EMBEDDING_EMBEDDED_SQL),
        _ => None,
    }
}

/// Escape a string value for safe inclusion in a LanceDB/DataFusion SQL filter.
///
/// # Safety model
///
/// LanceDB uses Apache DataFusion as its SQL engine. DataFusion string literals
/// follow standard SQL escaping rules:
///
/// - Single quotes are escaped by doubling them (`'` → `''`). This is the
///   standard SQL escaping mechanism and closes the only structural injection
///   vector in `column = '...'` expressions.
/// - Backslashes are **not** treated as escape characters by DataFusion in
///   string literals (DataFusion uses `E'...'` Postgres-style syntax for that,
///   which we never emit). Plain single-quoted literals treat `\` literally,
///   so no backslash escaping is needed.
/// - Null bytes (`\0`) cannot appear in Arrow string columns (Arrow Utf8 is
///   valid UTF-8, which excludes embedded NUL). We strip them defensively
///   so a malformed input cannot terminate the string early in any downstream
///   processing layer.
///
/// Characters that are special in SQL LIKE patterns (`%`, `_`) are **not**
/// escaped here because `eq_filter` uses `=`, not `LIKE`. For LIKE usage see
/// `like_escape_pattern`.
fn sql_escape(s: &str) -> String {
    // Strip NUL bytes (defensive; Arrow Utf8 rejects them anyway).
    // Double single quotes to prevent string literal breakout.
    s.replace('\0', "").replace('\'', "''")
}

/// Escape a value for safe use as the pattern operand in a LIKE expression.
///
/// Escapes:
/// - `\` → `\\`  (our chosen ESCAPE character, must come first)
/// - `%` → `\%`  (LIKE: match zero-or-more chars)
/// - `_` → `\_`  (LIKE: match exactly one char)
/// - `'` → `''`  (SQL string literal breakout)
/// - `\0` stripped (see `sql_escape`)
///
/// The caller is responsible for appending `ESCAPE '\'` to the LIKE clause.
fn like_escape_pattern(s: &str) -> String {
    s.replace('\0', "")
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
        .replace('\'', "''")
}

fn eq_filter(column: &str, value: &str) -> String {
    format!("{} = '{}'", column, sql_escape(value))
}

/// Format a Unix epoch timestamp as a decimal integer literal for safe
/// inclusion in a LanceDB/DataFusion filter predicate.
///
/// # Injection safety
///
/// `i64` can only produce the characters `[-0-9]`. The returned string
/// is always a bare decimal (or negative decimal) integer with no quotes,
/// parentheses, or SQL keywords — it cannot alter the predicate structure
/// even if the call site were ever refactored to accept a wider type.
///
/// Example output: `"1716816000"`, `"-3600"`, `"0"`.
fn timestamp_filter_value(ts: i64) -> String {
    let s = ts.to_string();
    debug_assert!(
        s.bytes().all(|b| b.is_ascii_digit() || b == b'-'),
        "timestamp filter literal must contain only [-0-9], got {s:?}"
    );
    s
}

/// Append `created_at` range predicates for the optional `since`/`until` bounds.
///
/// `i64::MIN` / `i64::MAX` are treated as unbounded and emit no predicate: they
/// impose no real constraint, and `i64::MIN` in particular cannot be embedded as
/// a DataFusion `Int64` literal — its magnitude (`9223372036854775808`) exceeds
/// `i64::MAX`, so the SQL parser promotes it to `Float64` and then fails the
/// `Int64` conversion. Omitting the no-op bound keeps every `i64` input safe.
fn push_timestamp_filters(filters: &mut Vec<String>, since: Option<i64>, until: Option<i64>) {
    if let Some(s) = since {
        if s > i64::MIN {
            filters.push(format!("created_at >= {}", timestamp_filter_value(s)));
        }
    }
    if let Some(u) = until {
        if u < i64::MAX {
            filters.push(format!("created_at <= {}", timestamp_filter_value(u)));
        }
    }
}

fn is_commit_conflict(e: &lancedb::Error) -> bool {
    // LanceDB wraps lance errors as "lance error: Commit conflict for version N: ..."
    // or "lance error: Retryable commit conflict for version N: ...".
    // Match case-insensitively to catch both variants.
    e.to_string()
        .to_ascii_lowercase()
        .contains("commit conflict")
}

/// Build a range filter that matches rows where `column` starts with `prefix`.
///
/// # Safety
///
/// `get_by_id_prefix` — the only caller — validates that `prefix` contains only
/// ASCII hex digits (`[0-9a-f]`) before calling this function. That input set
/// contains no SQL-special characters, so `sql_escape` is applied for defense
/// in depth but makes no material difference for valid inputs.
fn starts_with_filter(column: &str, prefix: &str) -> String {
    format!(
        "{} >= '{}' AND {} < '{}'",
        column,
        sql_escape(prefix),
        column,
        sql_escape(&prefix_upper_bound(prefix))
    )
}

/// Compute the exclusive upper bound for a prefix scan.
/// E.g. "abc" -> "abd", "ff" -> next impossible (empty = scan all).
fn prefix_upper_bound(prefix: &str) -> String {
    let mut chars: Vec<char> = prefix.chars().collect();
    // Increment the last character
    while let Some(last) = chars.pop() {
        if let Some(next) = char::from_u32(last as u32 + 1) {
            chars.push(next);
            return chars.into_iter().collect();
        }
        // Overflow (e.g. 'f' in hex is fine, but handle edge cases) — pop and try parent
    }
    // All chars overflowed — return a string that's definitely beyond any sha256 hex
    "\u{ffff}".to_string()
}

pub struct LanceStore {
    connection: Connection,
    dimension: usize,
    lock_dir: Option<PathBuf>,
}

impl LanceStore {
    /// Returns true if this store was opened in read-only mode.
    fn is_read_only(&self) -> bool {
        self.lock_dir.is_none()
    }

    /// Open the store's table and return it together with its version list.
    async fn open_table_and_versions(&self) -> Result<(Table, Vec<lancedb::table::Version>)> {
        let table = self
            .connection
            .open_table(TABLE_NAME)
            .execute()
            .await
            .map_err(|e| Error::store(format!("Failed to open table: {}", e)))?;
        let versions = table
            .list_versions()
            .await
            .map_err(|e| Error::store(format!("Failed to list versions: {}", e)))?;
        Ok((table, versions))
    }

    /// Compact fragments + prune old versions if version count exceeds MAX_VERSIONS.
    /// Keeps the last 3 versions as a safety margin for concurrent access.
    /// Below the threshold this is a near-free `list_versions()` check.
    pub(crate) async fn auto_compact_if_needed(&self) -> Result<CompactStats> {
        if self.is_read_only() {
            return Ok(CompactStats::default());
        }

        let (table, versions) = self.open_table_and_versions().await?;

        if versions.len() <= MAX_VERSIONS {
            return Ok(CompactStats::default());
        }

        Self::run_compact(&table, versions).await
    }

    /// Run compact + prune unconditionally, returning detailed stats.
    /// Used by the user-invoked `veclayer reflect prune` command and the daily
    /// background timer in the MCP server, where we want progress regardless of
    /// the version-count threshold.
    pub(crate) async fn force_compact(&self) -> Result<CompactStats> {
        if self.is_read_only() {
            return Err(Error::store("Cannot compact a read-only store".to_string()));
        }

        let (table, versions) = self.open_table_and_versions().await?;
        Self::run_compact(&table, versions).await
    }

    /// Compute the prune cutoff: the age (duration before `now`) of the
    /// `keep`-th most recent version, so pruning removes everything older than
    /// that and the newest `keep` versions are retained.
    ///
    /// `timestamps_desc` must be sorted newest-first. Returns zero when there
    /// are `keep` or fewer versions, or when the cutoff timestamp is in the
    /// future relative to `now` (clock skew) — never a negative delta, which
    /// lance would reject.
    fn compute_prune_cutoff(
        timestamps_desc: &[chrono::DateTime<chrono::Utc>],
        keep: usize,
        now: chrono::DateTime<chrono::Utc>,
    ) -> chrono::TimeDelta {
        if keep == 0 || timestamps_desc.len() <= keep {
            return chrono::TimeDelta::zero();
        }
        let cutoff = timestamps_desc[keep - 1];
        (now - cutoff).max(chrono::TimeDelta::zero())
    }

    /// Compact fragments and prune old versions, keeping the last `KEEP_VERSIONS`.
    ///
    /// Returns an error if either the fragment compaction or the version prune
    /// fails. This is deliberate: a silently swallowed failure on the only
    /// space-reclaiming path lets the store grow without bound (issue #92), so
    /// callers must be able to observe — and surface — a broken compaction.
    async fn run_compact(
        table: &Table,
        versions: Vec<lancedb::table::Version>,
    ) -> Result<CompactStats> {
        const KEEP_VERSIONS: usize = 3;

        // Cutoff = age of the KEEP_VERSIONS-th most recent version, so prune
        // removes everything older than that (keeping the newest KEEP_VERSIONS).
        let mut sorted = versions;
        sorted.sort_by_key(|b| std::cmp::Reverse(b.version));
        let total_versions = sorted.len();
        let timestamps: Vec<chrono::DateTime<chrono::Utc>> =
            sorted.iter().map(|v| v.timestamp).collect();
        let older_than = Self::compute_prune_cutoff(&timestamps, KEEP_VERSIONS, chrono::Utc::now());

        tracing::info!(
            "Auto-compact: {} versions, compacting fragments and pruning older than {:?} (keep {})",
            total_versions,
            older_than,
            KEEP_VERSIONS
        );

        use lancedb::table::OptimizeAction;
        let mut stats = CompactStats::default();
        let mut errors: Vec<String> = Vec::new();

        // 1. Compact fragments — merges small files and materializes deletions.
        //    This is what physically reclaims space from updates and tombstoned rows.
        match table
            .optimize(OptimizeAction::Compact {
                options: lancedb::table::CompactionOptions::default(),
                remap_options: None,
            })
            .await
        {
            Ok(s) => {
                if let Some(c) = s.compaction {
                    stats.fragments_removed = c.fragments_removed as u64;
                    stats.fragments_added = c.fragments_added as u64;
                    stats.files_removed = c.files_removed as u64;
                    stats.files_added = c.files_added as u64;
                }
            }
            Err(e) => {
                tracing::warn!("Auto-compact: fragment compaction failed: {}", e);
                errors.push(format!("fragment compaction failed: {e}"));
            }
        }

        // 2. Prune old version manifests + their orphaned data files.
        match table
            .optimize(OptimizeAction::Prune {
                older_than: Some(older_than),
                delete_unverified: Some(true),
                error_if_tagged_old_versions: None,
            })
            .await
        {
            Ok(s) => {
                if let Some(p) = s.prune {
                    stats.versions_removed = p.old_versions;
                    stats.bytes_reclaimed = p.bytes_removed;
                }
            }
            Err(e) => {
                tracing::warn!("Auto-compact: version prune failed: {}", e);
                errors.push(format!("version prune failed: {e}"));
            }
        }

        tracing::info!(
            "Auto-compact done: {} versions removed, {} fragments compacted into {}, {} bytes reclaimed",
            stats.versions_removed,
            stats.fragments_removed,
            stats.fragments_added,
            stats.bytes_reclaimed
        );

        // Surface failures instead of swallowing them: an unbounded store grows
        // because a persistently failing compaction looked like success (#92).
        if errors.is_empty() {
            Ok(stats)
        } else {
            Err(Error::store(format!(
                "compaction incomplete ({total_versions} versions): {}",
                errors.join("; ")
            )))
        }
    }

    pub async fn open(path: impl AsRef<Path>, dimension: usize, read_only: bool) -> Result<Self> {
        let path = path.as_ref();
        let lock_dir = (!read_only).then(|| path.to_path_buf());

        std::fs::create_dir_all(path)?;

        let uri = path.to_string_lossy().to_string();
        let connection = connect(&uri)
            .execute()
            .await
            .map_err(|e| Error::store(format!("Failed to connect to LanceDB: {}", e)))?;

        // If the table already exists, use its dimension to avoid schema mismatches.
        // This allows opening a store created with a different embedder configuration
        // (e.g., Ollama 768-dim vs FastEmbed 384-dim) without errors.
        let effective_dimension = Self::detect_dimension(path).await.unwrap_or(dimension);

        let store = Self {
            connection,
            dimension: effective_dimension,
            lock_dir,
        };

        // Hold the write lock during schema creation/migration so two concurrent
        // opens cannot race on table creation or column add/drop (issue #70).
        let _migration_lock = if !read_only {
            let dir = path.to_path_buf();
            Some(
                tokio::task::spawn_blocking(move || FileLock::acquire_blocking(&dir))
                    .await
                    .map_err(|e| Error::store(format!("lock task failed: {e}")))??,
            )
        } else {
            None
        };
        store.ensure_table().await?;

        // On first open after deploying the auto-compaction fix: if the store has a
        // wildly excessive version count (>500, e.g. 57k on lumi), do a one-time
        // aggressive compact+prune in the background so the next invocation starts clean.
        // This is fire-and-forget — errors are logged but never block store use.
        let conn = store.connection.clone();
        if !read_only {
            tokio::spawn(async move {
                if let Ok(table) = conn.open_table(TABLE_NAME).execute().await {
                    if let Ok(versions) = table.list_versions().await {
                        if versions.len() > 500 {
                            tracing::warn!(
                                "Store has {} old versions -- running one-time aggressive compact+prune",
                                versions.len()
                            );
                            match Self::run_compact(&table, versions).await {
                                Ok(s) => tracing::info!(
                                    "One-time compact done: {} versions, {} fragments merged, {} bytes reclaimed",
                                    s.versions_removed, s.fragments_removed, s.bytes_reclaimed
                                ),
                                Err(e) => tracing::warn!("One-time compact failed: {}", e),
                            }
                        }
                    }
                }
            });
        }

        Ok(store)
    }

    /// Open the store for metadata-only operations.
    ///
    /// Reads the embedding dimension from an existing table's schema if present,
    /// otherwise falls back to a sensible default. This avoids requiring an
    /// active embedder or config resolution for browse/list operations.
    /// Detect the embedding dimension from an existing LanceDB table's schema.
    async fn detect_dimension(path: &std::path::Path) -> Option<usize> {
        use arrow_schema::DataType;

        let uri = path.to_string_lossy().to_string();
        let conn = connect(&uri).execute().await.ok()?;
        let table = conn.open_table(TABLE_NAME).execute().await.ok()?;
        let schema = table.schema().await.ok()?;
        let field = schema.field_with_name("embedding").ok()?;
        match field.data_type() {
            DataType::FixedSizeList(_, size) => Some(*size as usize),
            _ => None,
        }
    }

    pub async fn open_metadata(path: impl AsRef<Path>, read_only: bool) -> Result<Self> {
        let path = path.as_ref();
        let db_path = path.join(format!("{TABLE_NAME}.lance"));
        let dimension = if db_path.exists() {
            Self::detect_dimension(path).await.unwrap_or(384)
        } else {
            384
        };
        Self::open(path, dimension, read_only).await
    }

    // Single chokepoint: all 6 mutating trait methods route here, so rejecting
    // writes for read-only stores is enforced in one place.
    async fn with_write_lock<F, Fut, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let dir = match self.lock_dir.as_ref() {
            Some(dir) => dir.clone(),
            None => {
                return Err(Error::store(
                    "write rejected: store was opened in read-only mode",
                ))
            }
        };

        let _lock = tokio::task::spawn_blocking(move || FileLock::acquire_blocking(&dir))
            .await
            .map_err(|e| Error::store(format!("lock task failed: {e}")))??;

        f().await
    }

    /// Retry a LanceDB write operation on commit conflict.
    ///
    /// LanceDB uses optimistic concurrency — concurrent UpdateConfig transactions
    /// can conflict. This reopens the table on each retry to get a fresh view of
    /// the latest committed version, then retries with short backoff.
    /// Returns the number of rows updated by the successful attempt, so callers
    /// can distinguish a no-op (filter matched nothing) from a real update.
    async fn retry_on_conflict<F, Fut>(&self, op_name: &str, f: F) -> Result<u64>
    where
        F: Fn(Table) -> Fut,
        Fut: std::future::Future<
            Output = std::result::Result<lancedb::table::UpdateResult, lancedb::Error>,
        >,
    {
        let max_attempts = 5; // up to 4 retries: 10ms, 20ms, 40ms, 80ms ≈ 150ms total
        let mut delay = tokio::time::Duration::from_millis(10);

        for attempt in 0..max_attempts {
            let table = self.get_table().await?;
            match f(table).await {
                Ok(result) => return Ok(result.rows_updated),
                Err(e) if attempt + 1 < max_attempts && is_commit_conflict(&e) => {
                    tracing::debug!(
                        "{op_name}: commit conflict (attempt {attempt}), retrying in {delay:?}"
                    );
                    tokio::time::sleep(delay).await;
                    delay *= 2;
                }
                Err(e) => {
                    return Err(Error::store(format!("Failed to {op_name}: {e}")));
                }
            }
        }
        Err(Error::store(format!(
            "Failed to {op_name}: exhausted {max_attempts} attempts"
        )))
    }

    fn schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.dimension as i32,
                ),
                false,
            ),
            Field::new("level", DataType::UInt8, false),
            Field::new("parent_id", DataType::Utf8, true),
            Field::new("path", DataType::Utf8, false),
            Field::new("source_file", DataType::Utf8, false),
            Field::new("heading", DataType::Utf8, true),
            Field::new("cluster_memberships", DataType::Utf8, false),
            Field::new("entry_type", DataType::Utf8, false),
            Field::new("summarizes", DataType::Utf8, false),
            Field::new("perspectives", DataType::Utf8, false),
            Field::new("visibility", DataType::Utf8, false),
            Field::new("relations", DataType::Utf8, false),
            Field::new("created_at", DataType::Int64, false),
            Field::new("last_rolled", DataType::Int64, false),
            Field::new("access_hour", DataType::UInt16, false),
            Field::new("access_day", DataType::UInt16, false),
            Field::new("access_week", DataType::UInt16, false),
            Field::new("access_month", DataType::UInt16, false),
            Field::new("access_year", DataType::UInt16, false),
            Field::new("access_total", DataType::UInt32, false),
            Field::new("expires_at", DataType::Int64, true),
            Field::new("impression_hint", DataType::Utf8, true),
            Field::new("impression_strength", DataType::Float32, false),
            Field::new("embedding_status", DataType::Utf8, false),
        ]))
    }

    async fn ensure_table(&self) -> Result<()> {
        let read_only = self.is_read_only();

        let tables = self
            .connection
            .table_names()
            .execute()
            .await
            .map_err(|e| Error::store(format!("Failed to list tables: {}", e)))?;

        if !tables.contains(&TABLE_NAME.to_string()) {
            // Bootstrapping an empty store is allowed even read-only: maintenance
            // commands (reflect/compact) and cross-project reads open a possibly
            // never-initialized path read-only and expect a graceful empty store,
            // not an error. The read-only contract forbids mutating *existing*
            // data (migrations below), not creating a fresh empty table.
            let schema = self.schema();
            self.connection
                .create_empty_table(TABLE_NAME, schema)
                .execute()
                .await
                .map_err(|e| Error::store(format!("Failed to create table: {}", e)))?;
            // create_empty_table is the only write allowed read-only (so maintenance
            // and cross-project reads see a graceful empty store). stamp_version
            // writes manifest metadata and is cosmetic, so it stays read-write only.
            if !read_only {
                let table = self.get_table().await?;
                Self::stamp_version(&table, &self.schema()).await?;
            }
            return Ok(());
        }

        // Table exists — check for schema drift
        let table = self.get_table().await?;
        let current_schema = table
            .schema()
            .await
            .map_err(|e| Error::store(format!("Failed to read table schema: {}", e)))?;
        let expected_schema = self.schema();

        let current_fields: HashSet<&str> = current_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        let expected_fields: HashSet<&str> = expected_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();

        // 1. Columns we need but table doesn't have → add them (write) or error (read-only)
        let missing: Vec<(String, String)> = expected_schema
            .fields()
            .iter()
            .filter(|f| !current_fields.contains(f.name().as_str()))
            .map(|f| {
                let default = migration_default(f.name()).ok_or_else(|| {
                    Error::store(format!(
                        "Schema migration: core field '{}' missing from table. \
                         Store may be corrupted or from an incompatible version.",
                        f.name()
                    ))
                })?;
                Ok((f.name().clone(), default.to_string()))
            })
            .collect::<Result<_>>()?;

        if !missing.is_empty() {
            let names: Vec<&str> = missing.iter().map(|(n, _)| n.as_str()).collect();
            if read_only {
                return Err(Error::store(format!(
                    "cannot open store in read-only mode: the store has an older schema \
                     and requires migration (missing columns: {:?}). \
                     Open it read-write first to migrate.",
                    names
                )));
            }
            tracing::info!("migrating store: adding columns {:?}", names);
            table
                .add_columns(NewColumnTransform::SqlExpressions(missing), None)
                .await
                .map_err(|e| Error::store(format!("Schema migration failed: {}", e)))?;
        }

        // Read-only opens never migrate an existing store. A store where legacy
        // `is_summary` still exists alongside `entry_type` is valid to read; the
        // data migration and legacy-column cleanup happen on the next read-write open.
        if !read_only {
            // 2. Migrate data from legacy columns into new columns before dropping
            if current_fields.contains("is_summary") {
                tracing::info!("migrating data: is_summary → entry_type");
                table
                    .update()
                    .column("entry_type", "'summary'")
                    .only_if("is_summary = true AND entry_type = 'raw'")
                    .execute()
                    .await
                    .map_err(|e| {
                        Error::store(format!("Data migration is_summary failed: {}", e))
                    })?;
            }

            // 3. Legacy columns still in table → drop them (safe after data migration)
            let legacy_to_drop: Vec<&str> = current_fields
                .iter()
                .filter(|f| LEGACY_COLUMNS.contains(f))
                .copied()
                .collect();

            if !legacy_to_drop.is_empty() {
                tracing::info!(
                    "migrating store: dropping legacy columns {:?}",
                    legacy_to_drop
                );
                table
                    .drop_columns(&legacy_to_drop)
                    .await
                    .map_err(|e| Error::store(format!("Legacy column removal failed: {}", e)))?;
            }
        }

        // 4. Columns present in both but with different types → incompatible
        for expected_field in expected_schema.fields() {
            if let Ok(current_field) = current_schema.field_with_name(expected_field.name()) {
                if current_field.data_type() != expected_field.data_type() {
                    return Err(Error::store(format!(
                        "Column '{}' has type {:?} in store but {:?} in this client. \
                         Store was created with a different embedder dimension. \
                         Run `veclayer rebuild-index` to migrate to the current model.",
                        expected_field.name(),
                        current_field.data_type(),
                        expected_field.data_type()
                    )));
                }
            }
        }

        // 5. Columns table has but we don't expect (excluding legacy) → newer store
        let unexpected: Vec<&str> = current_fields
            .iter()
            .filter(|f| !expected_fields.contains(*f) && !LEGACY_COLUMNS.contains(f))
            .copied()
            .collect();

        if !unexpected.is_empty() {
            let stored_commit = current_schema
                .metadata()
                .get("veclayer::commit")
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());
            return Err(Error::store(format!(
                "Store has columns not recognized by this client: {:?}. \
                 Store was written by version: {}. \
                 Please update veclayer to a newer version.",
                unexpected, stored_commit
            )));
        }

        // 6. Stamp current version (write — skip when read-only)
        if !read_only {
            Self::stamp_version(&table, &expected_schema).await?;
        }

        Ok(())
    }

    /// Write schema fingerprint and build commit into Arrow schema metadata.
    /// Requires a native (local) LanceDB table — silently skips for remote connections.
    /// Skips the write entirely when the fingerprint is already current (avoids
    /// UpdateConfig conflicts between concurrent processes).
    async fn stamp_version(table: &Table, schema: &Schema) -> Result<()> {
        if let Some(native) = table.as_native() {
            let fingerprint = schema_fingerprint(schema);

            // Skip if already stamped with the same fingerprint — avoids a
            // replace_schema_metadata write that can conflict with concurrent opens.
            let current_schema = table
                .schema()
                .await
                .map_err(|e| Error::store(format!("Failed to read schema: {}", e)))?;
            if current_schema
                .metadata()
                .get("veclayer::schema_fingerprint")
                .is_some_and(|f| f == &fingerprint)
            {
                return Ok(());
            }

            let commit_info = format!(
                "{} ({})",
                env!("VECLAYER_GIT_HASH"),
                env!("VECLAYER_GIT_DATE"),
            );
            native
                .replace_schema_metadata(vec![
                    ("veclayer::schema_fingerprint".to_string(), fingerprint),
                    ("veclayer::commit".to_string(), commit_info),
                ])
                .await
                .map_err(|e| Error::store(format!("Failed to stamp version: {}", e)))?;
        }
        Ok(())
    }

    async fn get_table(&self) -> Result<Table> {
        self.connection
            .open_table(TABLE_NAME)
            .execute()
            .await
            .map_err(|e| Error::store(format!("Failed to open table: {}", e)))
    }

    fn chunks_to_batch(&self, chunks: &[HierarchicalChunk]) -> Result<RecordBatch> {
        let ids: Vec<&str> = chunks.iter().map(|c| c.id.as_str()).collect();
        let contents: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        let levels: Vec<u8> = chunks.iter().map(|c| c.level.depth()).collect();
        let parent_ids: Vec<Option<&str>> = chunks.iter().map(|c| c.parent_id.as_deref()).collect();
        let paths: Vec<&str> = chunks.iter().map(|c| c.path.as_str()).collect();
        let source_files: Vec<&str> = chunks.iter().map(|c| c.source_file.as_str()).collect();
        let headings: Vec<Option<&str>> = chunks.iter().map(|c| c.heading.as_deref()).collect();

        let cluster_memberships: Vec<String> = chunks
            .iter()
            .map(|c| {
                serde_json::to_string(&c.cluster_memberships)
                    .map_err(|e| Error::store(format!("serialize cluster_memberships: {}", e)))
            })
            .collect::<Result<_>>()?;
        let entry_type: Vec<String> = chunks.iter().map(|c| c.entry_type.to_string()).collect();
        let summarizes: Vec<String> = chunks
            .iter()
            .map(|c| {
                serde_json::to_string(&c.summarizes)
                    .map_err(|e| Error::store(format!("serialize summarizes: {}", e)))
            })
            .collect::<Result<_>>()?;
        let perspectives: Vec<String> = chunks
            .iter()
            .map(|c| {
                serde_json::to_string(&c.perspectives)
                    .map_err(|e| Error::store(format!("serialize perspectives: {}", e)))
            })
            .collect::<Result<_>>()?;

        let visibility: Vec<String> = chunks.iter().map(|c| c.visibility.clone()).collect();
        let relations: Vec<String> = chunks
            .iter()
            .map(|c| {
                serde_json::to_string(&c.relations)
                    .map_err(|e| Error::store(format!("serialize relations: {}", e)))
            })
            .collect::<Result<_>>()?;
        let created_at: Vec<i64> = chunks.iter().map(|c| c.access_profile.created_at).collect();
        let last_rolled: Vec<i64> = chunks
            .iter()
            .map(|c| c.access_profile.last_rolled)
            .collect();
        let access_hour: Vec<u16> = chunks.iter().map(|c| c.access_profile.hour).collect();
        let access_day: Vec<u16> = chunks.iter().map(|c| c.access_profile.day).collect();
        let access_week: Vec<u16> = chunks.iter().map(|c| c.access_profile.week).collect();
        let access_month: Vec<u16> = chunks.iter().map(|c| c.access_profile.month).collect();
        let access_year: Vec<u16> = chunks.iter().map(|c| c.access_profile.year).collect();
        let access_total: Vec<u32> = chunks.iter().map(|c| c.access_profile.total).collect();
        let expires_at: Vec<Option<i64>> = chunks.iter().map(|c| c.expires_at).collect();
        let impression_hint: Vec<Option<&str>> = chunks
            .iter()
            .map(|c| c.impression_hint.as_deref())
            .collect();
        let impression_strength: Vec<f32> = chunks.iter().map(|c| c.impression_strength).collect();

        let mut embedding_values: Vec<f32> = Vec::with_capacity(chunks.len() * self.dimension);
        let mut embedding_status: Vec<&str> = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            if let Some(ref emb) = chunk.embedding {
                if emb.len() != self.dimension {
                    return Err(Error::store(format!(
                        "Embedding dimension mismatch: expected {}, got {}. \
                     Run `veclayer rebuild-index` to re-embed all entries with the current model.",
                        self.dimension,
                        emb.len()
                    )));
                }
                embedding_values.extend(emb);
                embedding_status.push(EMBEDDING_STATUS_EMBEDDED);
            } else {
                // Zero-vector for pending entries — self-excludes from cosine similarity (score ≈ 0.0)
                embedding_values.extend(std::iter::repeat_n(0.0f32, self.dimension));
                embedding_status.push(EMBEDDING_STATUS_PENDING);
            }
        }

        let values = Float32Array::from(embedding_values);
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let embedding_array =
            FixedSizeListArray::new(field, self.dimension as i32, Arc::new(values), None);

        RecordBatch::try_new(
            self.schema(),
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(contents)),
                Arc::new(embedding_array),
                Arc::new(UInt8Array::from(levels)),
                Arc::new(StringArray::from(parent_ids)),
                Arc::new(StringArray::from(paths)),
                Arc::new(StringArray::from(source_files)),
                Arc::new(StringArray::from(headings)),
                Arc::new(StringArray::from(cluster_memberships)),
                Arc::new(StringArray::from(entry_type)),
                Arc::new(StringArray::from(summarizes)),
                Arc::new(StringArray::from(perspectives)),
                Arc::new(StringArray::from(visibility)),
                Arc::new(StringArray::from(relations)),
                Arc::new(Int64Array::from(created_at)),
                Arc::new(Int64Array::from(last_rolled)),
                Arc::new(UInt16Array::from(access_hour)),
                Arc::new(UInt16Array::from(access_day)),
                Arc::new(UInt16Array::from(access_week)),
                Arc::new(UInt16Array::from(access_month)),
                Arc::new(UInt16Array::from(access_year)),
                Arc::new(UInt32Array::from(access_total)),
                Arc::new(Int64Array::from(expires_at)),
                Arc::new(StringArray::from(impression_hint)),
                Arc::new(Float32Array::from(impression_strength)),
                Arc::new(StringArray::from(embedding_status)),
            ],
        )
        .map_err(|e| Error::store(format!("Failed to create record batch: {}", e)))
    }

    fn extract_column<'a, T: 'static>(
        batch: &'a RecordBatch,
        index: usize,
        name: &str,
    ) -> Result<&'a T> {
        batch
            .column(index)
            .as_any()
            .downcast_ref::<T>()
            .ok_or_else(|| Error::store(format!("Invalid {} column", name)))
    }

    /// Collect all chunks from a sequence of record batches.
    fn collect_chunks(&self, batches: &[RecordBatch]) -> Result<Vec<HierarchicalChunk>> {
        let mut chunks = Vec::new();
        for batch in batches {
            chunks.extend(self.batch_to_chunks(batch)?);
        }
        Ok(chunks)
    }

    fn batch_to_chunks(&self, batch: &RecordBatch) -> Result<Vec<HierarchicalChunk>> {
        let ids = Self::extract_column::<StringArray>(batch, 0, "id")?;
        let contents = Self::extract_column::<StringArray>(batch, 1, "content")?;
        let embeddings = Self::extract_column::<FixedSizeListArray>(batch, 2, "embedding")?;
        let levels = Self::extract_column::<UInt8Array>(batch, 3, "level")?;
        let parent_ids = Self::extract_column::<StringArray>(batch, 4, "parent_id")?;
        let paths = Self::extract_column::<StringArray>(batch, 5, "path")?;
        let source_files = Self::extract_column::<StringArray>(batch, 6, "source_file")?;
        let headings = Self::extract_column::<StringArray>(batch, 7, "heading")?;

        let cluster_memberships_col = batch
            .column_by_name("cluster_memberships")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let entry_type_col = batch
            .column_by_name("entry_type")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let summarizes_col = batch
            .column_by_name("summarizes")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let perspectives_col = batch
            .column_by_name("perspectives")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());

        let visibility_col = batch
            .column_by_name("visibility")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let relations_col = batch
            .column_by_name("relations")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let created_at_col = batch
            .column_by_name("created_at")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let last_rolled_col = batch
            .column_by_name("last_rolled")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let access_hour_col = batch
            .column_by_name("access_hour")
            .and_then(|c| c.as_any().downcast_ref::<UInt16Array>());
        let access_day_col = batch
            .column_by_name("access_day")
            .and_then(|c| c.as_any().downcast_ref::<UInt16Array>());
        let access_week_col = batch
            .column_by_name("access_week")
            .and_then(|c| c.as_any().downcast_ref::<UInt16Array>());
        let access_month_col = batch
            .column_by_name("access_month")
            .and_then(|c| c.as_any().downcast_ref::<UInt16Array>());
        let access_year_col = batch
            .column_by_name("access_year")
            .and_then(|c| c.as_any().downcast_ref::<UInt16Array>());
        let access_total_col = batch
            .column_by_name("access_total")
            .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
        let legacy_last_accessed_col = batch
            .column_by_name("last_accessed")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let legacy_access_count_col = batch
            .column_by_name("access_count")
            .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
        let expires_at_col = batch
            .column_by_name("expires_at")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let impression_hint_col = batch
            .column_by_name("impression_hint")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let impression_strength_col = batch
            .column_by_name("impression_strength")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>());
        let embedding_status_col = batch
            .column_by_name("embedding_status")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());

        let mut chunks = Vec::with_capacity(batch.num_rows());

        for i in 0..batch.num_rows() {
            let is_pending = embedding_status_col
                .is_some_and(|col| !col.is_null(i) && col.value(i) == EMBEDDING_STATUS_PENDING);

            let embedding: Option<Vec<f32>> = if is_pending {
                None
            } else {
                let embedding_array = embeddings.value(i);
                let embedding_values = embedding_array
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| Error::store("Invalid embedding values"))?;
                Some(
                    (0..embedding_values.len())
                        .map(|j| embedding_values.value(j))
                        .collect(),
                )
            };

            let cluster_memberships: Vec<ClusterMembership> = cluster_memberships_col
                .and_then(|col| {
                    if col.is_null(i) {
                        None
                    } else {
                        serde_json::from_str(col.value(i)).ok()
                    }
                })
                .unwrap_or_default();

            let entry_type = entry_type_col
                .and_then(|col| {
                    if col.is_null(i) {
                        None
                    } else {
                        Some(col.value(i).parse().unwrap_or_default())
                    }
                })
                .unwrap_or_default();

            let summarizes: Vec<String> = summarizes_col
                .and_then(|col| {
                    if col.is_null(i) {
                        None
                    } else {
                        serde_json::from_str(col.value(i)).ok()
                    }
                })
                .unwrap_or_default();

            let perspectives: Vec<String> = perspectives_col
                .and_then(|col| {
                    if col.is_null(i) {
                        None
                    } else {
                        serde_json::from_str(col.value(i)).ok()
                    }
                })
                .unwrap_or_default();

            let visibility: String = visibility_col
                .and_then(|col| {
                    if col.is_null(i) {
                        None
                    } else {
                        Some(col.value(i).to_string())
                    }
                })
                .unwrap_or_else(|| crate::chunk::visibility::NORMAL.to_string());

            let relations: Vec<crate::chunk::ChunkRelation> = relations_col
                .and_then(|col| {
                    if col.is_null(i) {
                        None
                    } else {
                        serde_json::from_str(col.value(i)).ok()
                    }
                })
                .unwrap_or_default();

            let access_profile = if access_hour_col.is_some() {
                crate::chunk::AccessProfile {
                    created_at: created_at_col.map(|col| col.value(i)).unwrap_or(0),
                    last_rolled: last_rolled_col.map(|col| col.value(i)).unwrap_or(0),
                    hour: access_hour_col.map(|col| col.value(i)).unwrap_or(0),
                    day: access_day_col.map(|col| col.value(i)).unwrap_or(0),
                    week: access_week_col.map(|col| col.value(i)).unwrap_or(0),
                    month: access_month_col.map(|col| col.value(i)).unwrap_or(0),
                    year: access_year_col.map(|col| col.value(i)).unwrap_or(0),
                    total: access_total_col.map(|col| col.value(i)).unwrap_or(0),
                }
            } else {
                let created_at = created_at_col.map(|col| col.value(i)).unwrap_or(0);
                let last_accessed = legacy_last_accessed_col
                    .map(|col| col.value(i))
                    .unwrap_or(0);
                let access_count = legacy_access_count_col.map(|col| col.value(i)).unwrap_or(0);
                crate::chunk::AccessProfile {
                    created_at,
                    last_rolled: last_accessed,
                    hour: 0,
                    day: 0,
                    week: 0,
                    month: 0,
                    year: 0,
                    total: access_count,
                }
            };

            let expires_at = expires_at_col.and_then(|col| {
                if col.is_null(i) {
                    None
                } else {
                    Some(col.value(i))
                }
            });

            chunks.push(HierarchicalChunk {
                id: ids.value(i).to_string(),
                content: contents.value(i).to_string(),
                embedding,
                level: ChunkLevel(levels.value(i)),
                parent_id: if parent_ids.is_null(i) {
                    None
                } else {
                    Some(parent_ids.value(i).to_string())
                },
                path: paths.value(i).to_string(),
                source_file: source_files.value(i).to_string(),
                heading: if headings.is_null(i) {
                    None
                } else {
                    Some(headings.value(i).to_string())
                },
                start_offset: CHUNK_BYTE_OFFSET_NOT_STORED,
                end_offset: CHUNK_BYTE_OFFSET_NOT_STORED,
                cluster_memberships,
                entry_type,
                summarizes,
                perspectives,
                visibility,
                relations,
                access_profile,
                expires_at,
                impression_hint: impression_hint_col.and_then(|col| {
                    if col.is_null(i) {
                        None
                    } else {
                        Some(col.value(i).to_string())
                    }
                }),
                impression_strength: impression_strength_col
                    .map(|col| col.value(i))
                    .unwrap_or(1.0),
            });
        }

        Ok(chunks)
    }
}

impl VectorStore for LanceStore {
    async fn insert_chunks(&self, chunks: Vec<HierarchicalChunk>) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        self.with_write_lock(|| async {
            let batch = self.chunks_to_batch(&chunks)?;
            let table = self.get_table().await?;

            table
                .add(Box::new(RecordBatchIterator::new(
                    vec![Ok(batch)],
                    self.schema(),
                )))
                .execute()
                .await
                .map_err(|e| Error::store(format!("Failed to insert chunks: {}", e)))?;

            // Fire-and-forget auto-compaction after writes
            if let Err(e) = self.auto_compact_if_needed().await {
                tracing::warn!("Auto-compact after insert failed (non-fatal): {}", e);
            }

            Ok(())
        })
        .await
    }

    async fn search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        level_filter: Option<ChunkLevel>,
        perspectives: &[&str],
    ) -> Result<Vec<SearchResult>> {
        let table = self.get_table().await?;

        let query_vec: Vec<f32> = query_embedding.to_vec();

        // Use cosine distance so the score (`1.0 - distance`) is cosine
        // similarity in [-1, 1] — matching the SQLite backend and the
        // documented blend-score contract. LanceDB defaults to L2 otherwise.
        let mut query = table
            .query()
            .nearest_to(query_vec)
            .map_err(|e| Error::search(format!("Failed to create nearest neighbor query: {}", e)))?
            .distance_type(lancedb::DistanceType::Cosine);

        query = query.only_if(EMBEDDING_EMBEDDED_FILTER);
        if let Some(level) = level_filter {
            query = query.only_if(format!("level = {}", level.depth()));
        }
        if !perspectives.is_empty() {
            // Perspectives are stored as a JSON array (e.g. `["decisions","knowledge"]`).
            // We match by looking for `"<name>` anywhere in the serialized array.
            // `like_escape_pattern` escapes LIKE metacharacters (%, _) so a perspective
            // name containing those characters matches exactly rather than broadly.
            let clauses: Vec<String> = perspectives
                .iter()
                .map(|p| {
                    format!(
                        "perspectives LIKE '%\"{}%' ESCAPE '\\'",
                        like_escape_pattern(p)
                    )
                })
                .collect();
            query = query.only_if(format!("({})", clauses.join(" OR ")));
        }

        let results = query
            .limit(limit)
            .execute()
            .await
            .map_err(|e| Error::search(format!("Failed to execute search: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| Error::search(format!("Failed to collect results: {}", e)))?;

        let mut search_results = Vec::new();
        for batch in results {
            let distances: Option<&Float32Array> = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref());

            let chunks = self.batch_to_chunks(&batch)?;
            for (i, chunk) in chunks.into_iter().enumerate() {
                // Cosine distance is in [0, 2]; `1.0 - distance` is cosine
                // similarity. Clamp to [-1, 1] to absorb floating-point error
                // (identical vectors can yield a tiny negative distance).
                let score = distances
                    .map(|d| (1.0 - d.value(i)).clamp(-1.0, 1.0))
                    .unwrap_or(1.0);
                search_results.push(SearchResult { chunk, score });
            }
        }

        Ok(search_results)
    }

    async fn get_children(&self, parent_id: &str) -> Result<Vec<HierarchicalChunk>> {
        let table = self.get_table().await?;

        let results = table
            .query()
            .only_if(eq_filter("parent_id", parent_id))
            .execute()
            .await
            .map_err(|e| Error::search(format!("Failed to query children: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| Error::search(format!("Failed to collect children: {}", e)))?;

        // LanceDB returns batches in unspecified order; sort by id so the
        // result order is deterministic across calls and backends.
        let mut chunks = self.collect_chunks(&results)?;
        chunks.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(chunks)
    }

    async fn get_by_id(&self, id: &str) -> Result<Option<HierarchicalChunk>> {
        let table = self.get_table().await?;

        let results = table
            .query()
            .only_if(eq_filter("id", id))
            .limit(1)
            .execute()
            .await
            .map_err(|e| Error::search(format!("Failed to query by id: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| Error::search(format!("Failed to collect result: {}", e)))?;

        if let Some(batch) = results.first() {
            let chunks = self.batch_to_chunks(batch)?;
            Ok(chunks.into_iter().next())
        } else {
            Ok(None)
        }
    }

    async fn get_by_id_prefix(&self, prefix: &str) -> Result<Option<HierarchicalChunk>> {
        // Validate: IDs are SHA-256 hex, so only allow hex characters
        if prefix.is_empty() || !prefix.chars().all(|c| c.is_ascii_hexdigit()) {
            return Ok(None);
        }

        // Try exact match first
        if let Some(chunk) = self.get_by_id(prefix).await? {
            return Ok(Some(chunk));
        }

        // Fall back to prefix scan
        let table = self.get_table().await?;

        let results = table
            .query()
            .only_if(starts_with_filter("id", prefix))
            .limit(2) // fetch 2 to detect ambiguity
            .execute()
            .await
            .map_err(|e| Error::search(format!("Failed to query by id prefix: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| Error::search(format!("Failed to collect prefix results: {}", e)))?;

        let chunks = self.collect_chunks(&results)?;

        match chunks.len() {
            0 => Ok(None),
            1 => Ok(Some(chunks.into_iter().next().unwrap())),
            _ => Err(Error::config(format!(
                "Ambiguous prefix '{}': matches {} entries. Use a longer prefix.",
                prefix,
                chunks.len()
            ))),
        }
    }

    async fn get_by_source(&self, source_file: &str) -> Result<Vec<HierarchicalChunk>> {
        let table = self.get_table().await?;

        let results = table
            .query()
            .only_if(eq_filter("source_file", source_file))
            .execute()
            .await
            .map_err(|e| Error::search(format!("Failed to query by source: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| Error::search(format!("Failed to collect results: {}", e)))?;

        // Sort by id for a deterministic, backend-independent result order.
        let mut chunks = self.collect_chunks(&results)?;
        chunks.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(chunks)
    }

    async fn delete_by_source(&self, source_file: &str) -> Result<usize> {
        self.with_write_lock(|| async {
            let table = self.get_table().await?;

            let before = self.get_by_source(source_file).await?.len();

            table
                .delete(&eq_filter("source_file", source_file))
                .await
                .map_err(|e| Error::store(format!("Failed to delete by source: {}", e)))?;

            Ok(before)
        })
        .await
    }

    async fn stats(&self) -> Result<StoreStats> {
        let table = self.get_table().await?;

        let results = table
            .query()
            .execute()
            .await
            .map_err(|e| Error::search(format!("Failed to query stats: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| Error::search(format!("Failed to collect stats: {}", e)))?;

        let mut total_chunks = 0;
        let mut chunks_by_level: HashMap<u8, usize> = HashMap::new();
        let mut source_file_set: HashSet<String> = HashSet::new();
        let mut pending_embeddings = 0usize;

        for batch in &results {
            total_chunks += batch.num_rows();

            let levels = Self::extract_column::<UInt8Array>(batch, 3, "level")?;
            let sources = Self::extract_column::<StringArray>(batch, 6, "source_file")?;
            let status_col = batch
                .column_by_name("embedding_status")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            for i in 0..batch.num_rows() {
                let level = levels.value(i);
                *chunks_by_level.entry(level).or_insert(0) += 1;
                source_file_set.insert(sources.value(i).to_string());

                if status_col
                    .is_some_and(|col| !col.is_null(i) && col.value(i) == EMBEDDING_STATUS_PENDING)
                {
                    pending_embeddings += 1;
                }
            }
        }

        Ok(StoreStats {
            total_chunks,
            chunks_by_level,
            source_files: source_file_set.into_iter().collect(),
            pending_embeddings,
        })
    }

    async fn update_access_profiles(
        &self,
        updates: Vec<(String, crate::AccessProfile)>,
    ) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }

        self.with_write_lock(|| async {
            for (chunk_id, profile) in &updates {
                let filter = eq_filter("id", chunk_id);

                self.retry_on_conflict("update access profile", |table| {
                    let filter = filter.clone();
                    async move {
                        table
                            .update()
                            .column("last_rolled", profile.last_rolled.to_string())
                            .column("access_hour", profile.hour.to_string())
                            .column("access_day", profile.day.to_string())
                            .column("access_week", profile.week.to_string())
                            .column("access_month", profile.month.to_string())
                            .column("access_year", profile.year.to_string())
                            .column("access_total", profile.total.to_string())
                            .only_if(filter)
                            .execute()
                            .await
                    }
                })
                .await?;
            }

            Ok(())
        })
        .await
    }

    async fn update_visibility(&self, chunk_id: &str, visibility: &str) -> Result<()> {
        self.with_write_lock(|| async {
            let filter = eq_filter("id", chunk_id);

            let rows_updated = self
                .retry_on_conflict("update visibility", |table| {
                    let filter = filter.clone();
                    async move {
                        table
                            .update()
                            .column("visibility", format!("'{}'", sql_escape(visibility)))
                            .only_if(filter)
                            .execute()
                            .await
                    }
                })
                .await?;

            if rows_updated == 0 {
                return Err(Error::store(format!(
                    "update_visibility: chunk '{}' not found",
                    chunk_id
                )));
            }

            Ok(())
        })
        .await
    }

    async fn add_relation(&self, chunk_id: &str, relation: crate::ChunkRelation) -> Result<()> {
        self.with_write_lock(|| async {
            let chunk = self
                .get_by_id(chunk_id)
                .await?
                .ok_or_else(|| Error::store(format!("Chunk not found: {}", chunk_id)))?;

            let mut relations = chunk.relations;
            relations.push(relation);

            let relations_json = serde_json::to_string(&relations)
                .map_err(|e| Error::store(format!("serialize relations: {}", e)))?;

            let filter = eq_filter("id", chunk_id);

            self.retry_on_conflict("add relation", |table| {
                let filter = filter.clone();
                let relations_json = relations_json.clone();
                async move {
                    table
                        .update()
                        .column("relations", format!("'{}'", sql_escape(&relations_json)))
                        .only_if(filter)
                        .execute()
                        .await
                }
            })
            .await?;

            Ok(())
        })
        .await
    }

    async fn get_hot_chunks(&self, limit: usize) -> Result<Vec<HierarchicalChunk>> {
        let table = self.get_table().await?;

        let results = table
            .query()
            .only_if("access_total > 0")
            .execute()
            .await
            .map_err(|e| Error::search(format!("Failed to query hot chunks: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| Error::search(format!("Failed to collect hot chunks: {}", e)))?;

        let mut all_chunks = self.collect_chunks(&results)?;

        all_chunks.sort_by(|a, b| {
            b.access_profile
                .total
                .cmp(&a.access_profile.total)
                .then(b.access_profile.hour.cmp(&a.access_profile.hour))
        });

        all_chunks.truncate(limit);
        Ok(all_chunks)
    }

    async fn get_stale_chunks(
        &self,
        stale_seconds: i64,
        limit: usize,
    ) -> Result<Vec<HierarchicalChunk>> {
        let now = crate::chunk::now_epoch_secs();
        let cutoff = now - stale_seconds;

        let table = self.get_table().await?;

        let filter = format!(
            "last_rolled < {} AND access_hour = 0 AND access_day = 0 AND access_week = 0 AND (visibility = 'normal' OR visibility = 'always')",
            cutoff
        );

        let results = table
            .query()
            .only_if(filter)
            .execute()
            .await
            .map_err(|e| Error::search(format!("Failed to query stale chunks: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| Error::search(format!("Failed to collect stale chunks: {}", e)))?;

        let mut all_chunks = self.collect_chunks(&results)?;

        all_chunks.sort_by_key(|c| c.access_profile.last_rolled);

        all_chunks.truncate(limit);
        Ok(all_chunks)
    }

    async fn search_text(
        &self,
        query: &str,
        perspectives: &[&str],
        since: Option<i64>,
        until: Option<i64>,
        limit: usize,
    ) -> Result<Vec<HierarchicalChunk>> {
        let table = self.get_table().await?;

        let mut filters = Vec::new();

        // Split query into words and require all of them (AND logic)
        let words: Vec<&str> = query.split_whitespace().collect();
        for word in &words {
            filters.push(format!(
                "content LIKE '%{}%' ESCAPE '\\'",
                like_escape_pattern(word)
            ));
        }

        if !perspectives.is_empty() {
            let clauses: Vec<String> = perspectives
                .iter()
                .map(|p| {
                    format!(
                        "perspectives LIKE '%\"{}%' ESCAPE '\\'",
                        like_escape_pattern(p)
                    )
                })
                .collect();
            filters.push(format!("({})", clauses.join(" OR ")));
        }
        push_timestamp_filters(&mut filters, since, until);

        let mut q = table.query();
        if !filters.is_empty() {
            q = q.only_if(filters.join(" AND "));
        }

        let results = q
            .execute()
            .await
            .map_err(|e| Error::search(format!("Failed to search text: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| Error::search(format!("Failed to collect text search results: {}", e)))?;

        let mut all_chunks = self.collect_chunks(&results)?;

        all_chunks.sort_by(|a, b| {
            b.access_profile
                .created_at
                .cmp(&a.access_profile.created_at)
        });
        all_chunks.truncate(limit);

        Ok(all_chunks)
    }

    async fn list_entries(
        &self,
        perspectives: &[&str],
        since: Option<i64>,
        until: Option<i64>,
        limit: usize,
    ) -> Result<Vec<HierarchicalChunk>> {
        let table = self.get_table().await?;

        let mut filters = Vec::new();
        if !perspectives.is_empty() {
            let clauses: Vec<String> = perspectives
                .iter()
                .map(|p| {
                    format!(
                        "perspectives LIKE '%\"{}%' ESCAPE '\\'",
                        like_escape_pattern(p)
                    )
                })
                .collect();
            filters.push(format!("({})", clauses.join(" OR ")));
        }
        push_timestamp_filters(&mut filters, since, until);

        let mut query = table.query();
        if !filters.is_empty() {
            query = query.only_if(filters.join(" AND "));
        }

        let results = query
            .execute()
            .await
            .map_err(|e| Error::search(format!("Failed to list entries: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| Error::search(format!("Failed to collect entries: {}", e)))?;

        let mut all_chunks = self.collect_chunks(&results)?;

        all_chunks.sort_by(|a, b| {
            b.access_profile
                .created_at
                .cmp(&a.access_profile.created_at)
        });
        all_chunks.truncate(limit);

        Ok(all_chunks)
    }

    async fn get_pending_embeddings(&self, limit: usize) -> Result<Vec<HierarchicalChunk>> {
        let table = self.get_table().await?;

        let results = table
            .query()
            .only_if(EMBEDDING_PENDING_FILTER)
            .limit(limit)
            .execute()
            .await
            .map_err(|e| Error::search(format!("Failed to query pending embeddings: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| Error::search(format!("Failed to collect pending embeddings: {}", e)))?;

        self.collect_chunks(&results)
    }

    /// Updates the stored embedding vectors for the given chunk IDs.
    ///
    /// # Atomicity warning — NON-ATOMIC delete-then-add
    ///
    /// Internally this function performs a `table.delete(filter)` followed by a
    /// `table.add(...)`.  These two operations are **not** wrapped in a single
    /// LanceDB transaction:
    ///
    /// * If `delete` succeeds but `add` fails, the affected rows are permanently
    ///   absent from the LanceDB index for the current session.
    /// * Callers **must** treat an `Err` return as "index possibly inconsistent".
    ///   The blob store is the source of truth; a full `rebuild_index` will
    ///   reconstruct the index from blobs and restore consistency.
    /// * This limitation is intentional — id-based delete filters make an
    ///   add-before-delete swap unsafe (duplicate rows) and a proper atomic
    ///   replace requires a LanceDB-level merge operation.  That rework is
    ///   tracked separately.
    /// * The delete-succeeds/add-fails data-loss path is not unit-tested:
    ///   injecting a post-delete failure would require a fault hook the
    ///   LanceDB API does not expose. The pinning tests cover the reachable
    ///   behaviors (partial/unknown-id skips return `Ok`).
    ///
    /// # Unknown IDs are silently skipped
    ///
    /// IDs that are not present in the LanceDB table are logged at `WARN` level
    /// and skipped.  The call still returns `Ok(())` in that case, and any IDs
    /// that *were* found are updated normally.
    async fn batch_update_embeddings(&self, updates: Vec<(String, Vec<f32>)>) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }

        self.with_write_lock(|| async {
            // IDs come from entries already stored in LanceDB (SHA-256 hex strings),
            // so they contain only [0-9a-f] in practice. `sql_escape` is applied
            // for defense in depth in case a caller passes a non-canonical ID.
            let id_list: String = updates
                .iter()
                .map(|(id, _)| format!("'{}'", sql_escape(id)))
                .collect::<Vec<_>>()
                .join(", ");
            let filter = format!("id IN ({})", id_list);

            let table = self.get_table().await?;

            let results = table
                .query()
                .only_if(&filter)
                .execute()
                .await
                .map_err(|e| {
                    Error::search(format!(
                        "Failed to query chunks for batch embedding update: {}",
                        e
                    ))
                })?
                .try_collect::<Vec<_>>()
                .await
                .map_err(|e| {
                    Error::search(format!(
                        "Failed to collect chunks for batch embedding update: {}",
                        e
                    ))
                })?;

            let mut chunks_by_id: HashMap<String, HierarchicalChunk> = self
                .collect_chunks(&results)?
                .into_iter()
                .map(|c| (c.id.clone(), c))
                .collect();

            let mut updated_chunks = Vec::with_capacity(updates.len());
            for (chunk_id, embedding) in updates {
                match chunks_by_id.remove(&chunk_id) {
                    Some(mut chunk) => {
                        chunk.embedding = Some(embedding);
                        updated_chunks.push(chunk);
                    }
                    None => {
                        tracing::warn!(
                            "Chunk not found for batch embedding update, skipping: {}",
                            chunk_id
                        );
                    }
                }
            }

            if updated_chunks.is_empty() {
                return Ok(());
            }

            table.delete(&filter).await.map_err(|e| {
                Error::store(format!(
                    "Failed to delete for batch embedding update: {}",
                    e
                ))
            })?;

            let batch = self.chunks_to_batch(&updated_chunks)?;
            table
                .add(Box::new(RecordBatchIterator::new(
                    vec![Ok(batch)],
                    self.schema(),
                )))
                .execute()
                .await
                .map_err(|e| {
                    Error::store(format!(
                        "Failed to reinsert after batch embedding update: {}",
                        e
                    ))
                })?;

            Ok(())
        })
        .await?;

        // Fire-and-forget auto-compaction after writes
        if let Err(e) = self.auto_compact_if_needed().await {
            tracing::warn!("Auto-compact after batch update failed (non-fatal): {}", e);
        }

        Ok(())
    }

    async fn count_pending_embeddings(&self) -> Result<usize> {
        let table = self.get_table().await?;

        let results = table
            .query()
            .only_if(EMBEDDING_PENDING_FILTER)
            .select(lancedb::query::Select::columns(&["id"]))
            .execute()
            .await
            .map_err(|e| Error::search(format!("Failed to count pending embeddings: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| Error::search(format!("Failed to collect pending count: {}", e)))?;

        Ok(results.iter().map(|b| b.num_rows()).sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_store() -> (LanceStore, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let store = LanceStore::open(temp_dir.path(), 384, false).await.unwrap();
        (store, temp_dir)
    }

    fn create_test_chunk(id: &str, content: &str, level: ChunkLevel) -> HierarchicalChunk {
        let mut chunk = HierarchicalChunk::new(
            content.to_string(),
            level,
            None,
            "test".to_string(),
            "test.md".to_string(),
        );
        chunk.id = id.to_string();
        chunk.embedding = Some(vec![0.1; 384]);
        chunk
    }

    fn create_test_chunk_with_parent(
        id: &str,
        content: &str,
        level: ChunkLevel,
        parent_id: &str,
    ) -> HierarchicalChunk {
        let mut chunk = HierarchicalChunk::new(
            content.to_string(),
            level,
            Some(parent_id.to_string()),
            format!("parent > {}", id),
            "test.md".to_string(),
        );
        chunk.id = id.to_string();
        chunk.embedding = Some(vec![0.2; 384]);
        chunk
    }

    #[tokio::test]
    async fn test_concurrent_read_write_access() {
        let temp_dir = TempDir::new().unwrap();
        let store1 = LanceStore::open(temp_dir.path(), 384, false).await.unwrap();
        let store2 = LanceStore::open(temp_dir.path(), 384, false).await.unwrap();

        let stats1 = store1.stats().await.unwrap();
        let stats2 = store2.stats().await.unwrap();
        assert_eq!(stats1.total_chunks, 0);
        assert_eq!(stats2.total_chunks, 0);

        let chunk = create_test_chunk("concurrent-1", "Concurrent content", ChunkLevel::H1);
        store1.insert_chunks(vec![chunk]).await.unwrap();

        let retrieved = store2.get_by_id("concurrent-1").await.unwrap();
        assert!(retrieved.is_some());
    }

    #[tokio::test]
    async fn test_insert_and_get_by_id() {
        let (store, _temp) = create_test_store().await;

        let chunk = create_test_chunk("test-1", "Test content", ChunkLevel::H1);
        store.insert_chunks(vec![chunk.clone()]).await.unwrap();

        let retrieved = store.get_by_id("test-1").await.unwrap();
        assert!(retrieved.is_some());

        let retrieved_chunk = retrieved.unwrap();
        assert_eq!(retrieved_chunk.id, "test-1");
        assert_eq!(retrieved_chunk.content, "Test content");
        assert_eq!(retrieved_chunk.level, ChunkLevel::H1);
        assert!(retrieved_chunk.embedding.is_some());
    }

    #[tokio::test]
    async fn test_get_by_id_not_found() {
        let (store, _temp) = create_test_store().await;

        let result = store.get_by_id("nonexistent").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_insert_empty_chunks() {
        let (store, _temp) = create_test_store().await;

        let result = store.insert_chunks(vec![]).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_search() {
        let (store, _temp) = create_test_store().await;

        let chunk1 = create_test_chunk("search-1", "First chunk", ChunkLevel::H1);
        let chunk2 = create_test_chunk("search-2", "Second chunk", ChunkLevel::H2);
        let chunk3 = create_test_chunk("search-3", "Third chunk", ChunkLevel::H1);

        store
            .insert_chunks(vec![chunk1, chunk2, chunk3])
            .await
            .unwrap();

        let query_embedding = vec![0.1; 384];
        let results = store.search(&query_embedding, 2, None, &[]).await.unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].score >= 0.0 && results[0].score <= 1.0);
    }

    #[tokio::test]
    async fn test_search_with_level_filter() {
        let (store, _temp) = create_test_store().await;

        let chunk1 = create_test_chunk("filter-1", "H1 chunk", ChunkLevel::H1);
        let chunk2 = create_test_chunk("filter-2", "H2 chunk", ChunkLevel::H2);
        let chunk3 = create_test_chunk("filter-3", "Another H1", ChunkLevel::H1);

        store
            .insert_chunks(vec![chunk1, chunk2, chunk3])
            .await
            .unwrap();

        let query_embedding = vec![0.1; 384];
        let results = store
            .search(&query_embedding, 10, Some(ChunkLevel::H1), &[])
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.chunk.level == ChunkLevel::H1));
    }

    #[tokio::test]
    async fn test_get_children() {
        let (store, _temp) = create_test_store().await;

        let parent = create_test_chunk("parent-1", "Parent chunk", ChunkLevel::H1);
        let child1 =
            create_test_chunk_with_parent("child-1", "Child 1", ChunkLevel::H2, "parent-1");
        let child2 =
            create_test_chunk_with_parent("child-2", "Child 2", ChunkLevel::H2, "parent-1");
        let unrelated = create_test_chunk("unrelated", "Unrelated", ChunkLevel::H1);

        store
            .insert_chunks(vec![parent, child1, child2, unrelated])
            .await
            .unwrap();

        let children = store.get_children("parent-1").await.unwrap();

        assert_eq!(children.len(), 2);
        assert!(children
            .iter()
            .all(|c| c.parent_id.as_deref() == Some("parent-1")));
        // Results are sorted by id for a deterministic, backend-independent order.
        let ids: Vec<&str> = children.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, ["child-1", "child-2"]);
    }

    #[tokio::test]
    async fn test_get_children_no_children() {
        let (store, _temp) = create_test_store().await;

        let parent = create_test_chunk("lonely-parent", "Lonely", ChunkLevel::H1);
        store.insert_chunks(vec![parent]).await.unwrap();

        let children = store.get_children("lonely-parent").await.unwrap();
        assert_eq!(children.len(), 0);
    }

    #[tokio::test]
    async fn test_get_by_source() {
        let (store, _temp) = create_test_store().await;

        let mut chunk1 = create_test_chunk("src-1", "From file1", ChunkLevel::H1);
        chunk1.source_file = "file1.md".to_string();

        let mut chunk2 = create_test_chunk("src-2", "Also file1", ChunkLevel::H2);
        chunk2.source_file = "file1.md".to_string();

        let mut chunk3 = create_test_chunk("src-3", "From file2", ChunkLevel::H1);
        chunk3.source_file = "file2.md".to_string();

        store
            .insert_chunks(vec![chunk1, chunk2, chunk3])
            .await
            .unwrap();

        let from_file1 = store.get_by_source("file1.md").await.unwrap();
        assert_eq!(from_file1.len(), 2);
        assert!(from_file1.iter().all(|c| c.source_file == "file1.md"));

        let from_file2 = store.get_by_source("file2.md").await.unwrap();
        assert_eq!(from_file2.len(), 1);
        assert_eq!(from_file2[0].id, "src-3");
    }

    #[tokio::test]
    async fn test_delete_by_source() {
        let (store, _temp) = create_test_store().await;

        let mut chunk1 = create_test_chunk("del-1", "Delete me", ChunkLevel::H1);
        chunk1.source_file = "delete.md".to_string();

        let mut chunk2 = create_test_chunk("del-2", "Keep me", ChunkLevel::H1);
        chunk2.source_file = "keep.md".to_string();

        store.insert_chunks(vec![chunk1, chunk2]).await.unwrap();

        let deleted_count = store.delete_by_source("delete.md").await.unwrap();
        assert_eq!(deleted_count, 1);

        let remaining = store.get_by_source("delete.md").await.unwrap();
        assert_eq!(remaining.len(), 0);

        let kept = store.get_by_source("keep.md").await.unwrap();
        assert_eq!(kept.len(), 1);
    }

    #[tokio::test]
    async fn test_delete_nonexistent_source() {
        let (store, _temp) = create_test_store().await;

        let deleted_count = store.delete_by_source("nonexistent.md").await.unwrap();
        assert_eq!(deleted_count, 0);
    }

    #[tokio::test]
    async fn test_stats() {
        let (store, _temp) = create_test_store().await;

        let mut chunk1 = create_test_chunk("stats-1", "H1 chunk", ChunkLevel::H1);
        chunk1.source_file = "file1.md".to_string();

        let mut chunk2 = create_test_chunk("stats-2", "H2 chunk", ChunkLevel::H2);
        chunk2.source_file = "file1.md".to_string();

        let mut chunk3 = create_test_chunk("stats-3", "Another H1", ChunkLevel::H1);
        chunk3.source_file = "file2.md".to_string();

        store
            .insert_chunks(vec![chunk1, chunk2, chunk3])
            .await
            .unwrap();

        let stats = store.stats().await.unwrap();

        assert_eq!(stats.total_chunks, 3);
        assert_eq!(*stats.chunks_by_level.get(&1).unwrap(), 2);
        assert_eq!(*stats.chunks_by_level.get(&2).unwrap(), 1);
        assert_eq!(stats.source_files.len(), 2);
        assert!(stats.source_files.contains(&"file1.md".to_string()));
        assert!(stats.source_files.contains(&"file2.md".to_string()));
    }

    #[tokio::test]
    async fn test_stats_empty_store() {
        let (store, _temp) = create_test_store().await;

        let stats = store.stats().await.unwrap();
        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.chunks_by_level.len(), 0);
        assert_eq!(stats.source_files.len(), 0);
    }

    #[tokio::test]
    async fn test_cluster_fields_roundtrip() {
        let (store, _temp) = create_test_store().await;

        let mut chunk = create_test_chunk("cluster-1", "Cluster test", ChunkLevel::H1);
        chunk.cluster_memberships = vec![
            ClusterMembership::new("cluster-a", 0.8),
            ClusterMembership::new("cluster-b", 0.6),
        ];
        chunk.entry_type = crate::chunk::EntryType::Summary;
        chunk.summarizes = vec!["chunk-1".to_string(), "chunk-2".to_string()];

        store.insert_chunks(vec![chunk.clone()]).await.unwrap();

        let retrieved = store.get_by_id("cluster-1").await.unwrap().unwrap();

        assert_eq!(retrieved.cluster_memberships.len(), 2);
        assert_eq!(retrieved.cluster_memberships[0].cluster_id, "cluster-a");
        assert!((retrieved.cluster_memberships[0].probability - 0.8).abs() < 0.01);
        assert_eq!(retrieved.cluster_memberships[1].cluster_id, "cluster-b");
        assert!((retrieved.cluster_memberships[1].probability - 0.6).abs() < 0.01);
        assert!(retrieved.is_summary());
        assert_eq!(retrieved.summarizes.len(), 2);
        assert!(retrieved.summarizes.contains(&"chunk-1".to_string()));
        assert!(retrieved.summarizes.contains(&"chunk-2".to_string()));
    }

    #[tokio::test]
    async fn test_cluster_fields_empty() {
        let (store, _temp) = create_test_store().await;

        let chunk = create_test_chunk("cluster-2", "No clusters", ChunkLevel::H1);
        store.insert_chunks(vec![chunk]).await.unwrap();

        let retrieved = store.get_by_id("cluster-2").await.unwrap().unwrap();

        assert_eq!(retrieved.cluster_memberships.len(), 0);
        assert!(!retrieved.is_summary());
        assert_eq!(retrieved.summarizes.len(), 0);
    }

    #[tokio::test]
    async fn test_sql_injection_protection() {
        let (store, _temp) = create_test_store().await;

        let mut chunk = create_test_chunk("injection-test", "Test", ChunkLevel::H1);
        chunk.source_file = "'; DROP TABLE chunks; --".to_string();

        store.insert_chunks(vec![chunk]).await.unwrap();

        let result = store.get_by_source("'; DROP TABLE chunks; --").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_embedding_dimension_validation() {
        let (store, _temp) = create_test_store().await;

        let mut chunk = create_test_chunk("dim-test", "Wrong dimension", ChunkLevel::H1);
        chunk.embedding = Some(vec![0.1; 256]);

        let result = store.insert_chunks(vec![chunk]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_missing_embedding_inserts_as_pending() {
        let (store, _temp) = create_test_store().await;

        let mut chunk = create_test_chunk("no-emb", "No embedding", ChunkLevel::H1);
        chunk.embedding = None;

        // Should succeed — inserts with zero-vector + pending status
        store.insert_chunks(vec![chunk]).await.unwrap();

        // Round-trips with embedding: None
        let retrieved = store.get_by_id("no-emb").await.unwrap().unwrap();
        assert_eq!(retrieved.id, "no-emb");
        assert!(retrieved.embedding.is_none());
    }

    #[tokio::test]
    async fn test_pending_chunk_excluded_from_vector_search() {
        let (store, _temp) = create_test_store().await;

        // Insert one embedded and one pending chunk
        let embedded = create_test_chunk("emb-1", "Embedded chunk", ChunkLevel::H1);
        let mut pending = create_test_chunk("pend-1", "Pending chunk", ChunkLevel::H1);
        pending.embedding = None;

        store.insert_chunks(vec![embedded, pending]).await.unwrap();

        // Vector search should only return the embedded chunk
        let query = vec![0.1f32; 384];
        let results = store.search(&query, 10, None, &[]).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.id, "emb-1");
    }

    #[tokio::test]
    async fn test_get_pending_embeddings() {
        let (store, _temp) = create_test_store().await;

        let embedded = create_test_chunk("emb-2", "Embedded", ChunkLevel::H1);
        let mut pending1 = create_test_chunk("pend-2", "Pending A", ChunkLevel::H1);
        pending1.embedding = None;
        let mut pending2 = create_test_chunk("pend-3", "Pending B", ChunkLevel::H1);
        pending2.embedding = None;

        store
            .insert_chunks(vec![embedded, pending1, pending2])
            .await
            .unwrap();

        let pending = store.get_pending_embeddings(10).await.unwrap();
        assert_eq!(pending.len(), 2);
        assert!(pending.iter().all(|c| c.embedding.is_none()));
    }

    #[tokio::test]
    async fn test_update_embedding_makes_searchable() {
        let (store, _temp) = create_test_store().await;

        let mut pending = create_test_chunk("upd-1", "To be embedded", ChunkLevel::H1);
        pending.embedding = None;
        store.insert_chunks(vec![pending]).await.unwrap();

        // Not searchable yet
        let query = vec![0.1f32; 384];
        let results = store.search(&query, 10, None, &[]).await.unwrap();
        assert!(results.is_empty());

        // Update with real embedding
        store
            .batch_update_embeddings(vec![("upd-1".to_string(), vec![0.1f32; 384])])
            .await
            .unwrap();

        // Now searchable
        let results = store.search(&query, 10, None, &[]).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.id, "upd-1");
        assert!(results[0].chunk.embedding.is_some());
    }

    #[tokio::test]
    async fn test_stats_reflect_pending_count() {
        let (store, _temp) = create_test_store().await;

        let embedded = create_test_chunk("s-emb", "Embedded", ChunkLevel::H1);
        let mut pending = create_test_chunk("s-pend", "Pending", ChunkLevel::H1);
        pending.embedding = None;

        store.insert_chunks(vec![embedded, pending]).await.unwrap();

        let stats = store.stats().await.unwrap();
        assert_eq!(stats.total_chunks, 2);
        assert_eq!(stats.pending_embeddings, 1);
    }

    #[tokio::test]
    async fn test_count_pending_embeddings() {
        let (store, _temp) = create_test_store().await;

        let embedded = create_test_chunk("cp-emb", "Embedded", ChunkLevel::H1);
        let mut p1 = create_test_chunk("cp-p1", "Pending 1", ChunkLevel::H1);
        p1.embedding = None;
        let mut p2 = create_test_chunk("cp-p2", "Pending 2", ChunkLevel::H1);
        p2.embedding = None;

        store.insert_chunks(vec![embedded, p1, p2]).await.unwrap();

        assert_eq!(store.count_pending_embeddings().await.unwrap(), 2);

        // After updating one, count decreases
        store
            .batch_update_embeddings(vec![("cp-p1".to_string(), vec![0.1f32; 384])])
            .await
            .unwrap();
        assert_eq!(store.count_pending_embeddings().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_update_visibility() {
        let (store, _temp) = create_test_store().await;

        let chunk = create_test_chunk("vis-1", "Visibility test", ChunkLevel::H1);
        store.insert_chunks(vec![chunk]).await.unwrap();

        let before = store.get_by_id("vis-1").await.unwrap().unwrap();
        assert_eq!(before.visibility, "normal");

        store.update_visibility("vis-1", "always").await.unwrap();
        let after = store.get_by_id("vis-1").await.unwrap().unwrap();
        assert_eq!(after.visibility, "always");

        store.update_visibility("vis-1", "deep_only").await.unwrap();
        let after2 = store.get_by_id("vis-1").await.unwrap().unwrap();
        assert_eq!(after2.visibility, "deep_only");
    }

    #[tokio::test]
    async fn test_update_visibility_custom_value() {
        let (store, _temp) = create_test_store().await;

        let chunk = create_test_chunk("vis-2", "Custom visibility", ChunkLevel::H1);
        store.insert_chunks(vec![chunk]).await.unwrap();

        store.update_visibility("vis-2", "draft").await.unwrap();
        let after = store.get_by_id("vis-2").await.unwrap().unwrap();
        assert_eq!(after.visibility, "draft");
    }

    #[tokio::test]
    async fn test_add_relation() {
        let (store, _temp) = create_test_store().await;

        let chunk1 = create_test_chunk("rel-1", "Source chunk", ChunkLevel::H1);
        let chunk2 = create_test_chunk("rel-2", "Target chunk", ChunkLevel::H1);
        store.insert_chunks(vec![chunk1, chunk2]).await.unwrap();

        let relation = crate::ChunkRelation::superseded_by("rel-2");
        store.add_relation("rel-1", relation).await.unwrap();

        let after = store.get_by_id("rel-1").await.unwrap().unwrap();
        assert_eq!(after.relations.len(), 1);
        assert_eq!(after.relations[0].kind, "superseded_by");
        assert_eq!(after.relations[0].target_id, "rel-2");
    }

    #[tokio::test]
    async fn test_add_multiple_relations() {
        let (store, _temp) = create_test_store().await;

        let chunk = create_test_chunk("multi-rel", "Multi-relation", ChunkLevel::H1);
        store.insert_chunks(vec![chunk]).await.unwrap();

        store
            .add_relation("multi-rel", crate::ChunkRelation::related_to("a"))
            .await
            .unwrap();
        store
            .add_relation("multi-rel", crate::ChunkRelation::derived_from("b"))
            .await
            .unwrap();

        let after = store.get_by_id("multi-rel").await.unwrap().unwrap();
        assert_eq!(after.relations.len(), 2);
    }

    #[tokio::test]
    async fn test_add_relation_nonexistent_chunk() {
        let (store, _temp) = create_test_store().await;

        let relation = crate::ChunkRelation::related_to("target");
        let result = store.add_relation("nonexistent", relation).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_update_access_profiles() {
        let (store, _temp) = create_test_store().await;

        let chunk = create_test_chunk("access-1", "Access test", ChunkLevel::H1);
        store.insert_chunks(vec![chunk]).await.unwrap();

        let before = store.get_by_id("access-1").await.unwrap().unwrap();
        assert_eq!(before.access_profile.total, 0);
        assert_eq!(before.access_profile.hour, 0);

        let mut profile = before.access_profile.clone();
        profile.hour = 3;
        profile.total = 3;
        profile.last_rolled = profile.created_at + 100;

        store
            .update_access_profiles(vec![("access-1".to_string(), profile)])
            .await
            .unwrap();

        let after = store.get_by_id("access-1").await.unwrap().unwrap();
        assert_eq!(after.access_profile.hour, 3);
        assert_eq!(after.access_profile.total, 3);
    }

    #[tokio::test]
    async fn test_update_access_profiles_empty() {
        let (store, _temp) = create_test_store().await;

        let result = store.update_access_profiles(vec![]).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_visibility_roundtrip() {
        let (store, _temp) = create_test_store().await;

        let mut chunk = create_test_chunk("vis-rt", "Roundtrip test", ChunkLevel::H1);
        chunk.visibility = "always".to_string();
        store.insert_chunks(vec![chunk]).await.unwrap();

        let retrieved = store.get_by_id("vis-rt").await.unwrap().unwrap();
        assert_eq!(retrieved.visibility, "always");
    }

    #[tokio::test]
    async fn test_relations_roundtrip() {
        let (store, _temp) = create_test_store().await;

        let mut chunk = create_test_chunk("rel-rt", "Relations roundtrip", ChunkLevel::H1);
        chunk.relations = vec![
            crate::ChunkRelation::superseded_by("newer"),
            crate::ChunkRelation::related_to("sibling"),
        ];
        store.insert_chunks(vec![chunk]).await.unwrap();

        let retrieved = store.get_by_id("rel-rt").await.unwrap().unwrap();
        assert_eq!(retrieved.relations.len(), 2);
        assert_eq!(retrieved.relations[0].kind, "superseded_by");
        assert_eq!(retrieved.relations[1].kind, "related_to");
    }

    #[tokio::test]
    async fn test_access_profile_roundtrip() {
        let (store, _temp) = create_test_store().await;

        let mut chunk = create_test_chunk("ap-rt", "AccessProfile roundtrip", ChunkLevel::H1);
        chunk.access_profile.hour = 5;
        chunk.access_profile.day = 10;
        chunk.access_profile.week = 20;
        chunk.access_profile.month = 50;
        chunk.access_profile.year = 100;
        chunk.access_profile.total = 185;
        store.insert_chunks(vec![chunk]).await.unwrap();

        let retrieved = store.get_by_id("ap-rt").await.unwrap().unwrap();
        assert_eq!(retrieved.access_profile.hour, 5);
        assert_eq!(retrieved.access_profile.day, 10);
        assert_eq!(retrieved.access_profile.week, 20);
        assert_eq!(retrieved.access_profile.month, 50);
        assert_eq!(retrieved.access_profile.year, 100);
        assert_eq!(retrieved.access_profile.total, 185);
    }

    #[tokio::test]
    async fn test_update_visibility_sql_injection() {
        let (store, _temp) = create_test_store().await;

        let chunk = create_test_chunk("sqli-vis", "SQL injection test", ChunkLevel::H1);
        store.insert_chunks(vec![chunk]).await.unwrap();

        store
            .update_visibility("sqli-vis", "'; DROP TABLE chunks; --")
            .await
            .unwrap();

        let after = store.get_by_id("sqli-vis").await.unwrap().unwrap();
        assert!(after.visibility.contains("DROP TABLE"));
    }

    #[tokio::test]
    async fn test_get_hot_chunks_empty() {
        let (store, _temp) = create_test_store().await;

        let hot = store.get_hot_chunks(10).await.unwrap();
        assert!(hot.is_empty());
    }

    #[tokio::test]
    async fn test_get_hot_chunks_sorted() {
        let (store, _temp) = create_test_store().await;

        let mut chunk1 = create_test_chunk("hot-1", "Low access", ChunkLevel::H1);
        chunk1.access_profile.total = 5;
        chunk1.access_profile.hour = 1;

        let mut chunk2 = create_test_chunk("hot-2", "High access", ChunkLevel::H1);
        chunk2.access_profile.total = 50;
        chunk2.access_profile.hour = 10;

        let mut chunk3 = create_test_chunk("hot-3", "No access", ChunkLevel::H1);
        chunk3.access_profile.total = 0;

        store
            .insert_chunks(vec![chunk1, chunk2, chunk3])
            .await
            .unwrap();

        let hot = store.get_hot_chunks(10).await.unwrap();
        assert_eq!(hot.len(), 2);
        assert_eq!(hot[0].id, "hot-2");
        assert_eq!(hot[1].id, "hot-1");
    }

    #[tokio::test]
    async fn test_get_hot_chunks_limit() {
        let (store, _temp) = create_test_store().await;

        let mut chunk1 = create_test_chunk("lim-1", "A", ChunkLevel::H1);
        chunk1.access_profile.total = 10;
        let mut chunk2 = create_test_chunk("lim-2", "B", ChunkLevel::H1);
        chunk2.access_profile.total = 20;
        let mut chunk3 = create_test_chunk("lim-3", "C", ChunkLevel::H1);
        chunk3.access_profile.total = 30;

        store
            .insert_chunks(vec![chunk1, chunk2, chunk3])
            .await
            .unwrap();

        let hot = store.get_hot_chunks(2).await.unwrap();
        assert_eq!(hot.len(), 2);
        assert_eq!(hot[0].id, "lim-3");
        assert_eq!(hot[1].id, "lim-2");
    }

    #[tokio::test]
    async fn test_get_stale_chunks_empty() {
        let (store, _temp) = create_test_store().await;

        let stale = store.get_stale_chunks(86_400, 10).await.unwrap();
        assert!(stale.is_empty());
    }

    #[tokio::test]
    async fn test_get_stale_chunks_filters_active() {
        let (store, _temp) = create_test_store().await;

        let now = crate::chunk::now_epoch_secs();

        let mut active = create_test_chunk("active", "Active chunk", ChunkLevel::H1);
        active.access_profile.last_rolled = now;
        active.access_profile.hour = 3;
        active.access_profile.total = 3;

        let mut stale = create_test_chunk("stale", "Stale chunk", ChunkLevel::H1);
        stale.access_profile.last_rolled = now - 90 * 86_400;
        stale.access_profile.hour = 0;
        stale.access_profile.day = 0;
        stale.access_profile.week = 0;
        stale.access_profile.total = 10;

        store.insert_chunks(vec![active, stale]).await.unwrap();

        let result = store.get_stale_chunks(30 * 86_400, 10).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "stale");
    }

    #[tokio::test]
    async fn test_get_stale_chunks_excludes_deep_only() {
        let (store, _temp) = create_test_store().await;

        let now = crate::chunk::now_epoch_secs();

        let mut chunk = create_test_chunk("deep", "Already archived", ChunkLevel::H1);
        chunk.visibility = "deep_only".to_string();
        chunk.access_profile.last_rolled = now - 90 * 86_400;
        chunk.access_profile.hour = 0;
        chunk.access_profile.day = 0;
        chunk.access_profile.week = 0;

        store.insert_chunks(vec![chunk]).await.unwrap();

        let result = store.get_stale_chunks(30 * 86_400, 10).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_list_entries_empty() {
        let (store, _temp) = create_test_store().await;

        let result = store.list_entries(&[], None, None, 10).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_list_entries_sorted_newest_first() {
        let (store, _temp) = create_test_store().await;

        let mut chunk1 = create_test_chunk("list-1", "First", ChunkLevel::H1);
        chunk1.access_profile.created_at = 1000;

        let mut chunk2 = create_test_chunk("list-2", "Second", ChunkLevel::H1);
        chunk2.access_profile.created_at = 2000;

        let mut chunk3 = create_test_chunk("list-3", "Third", ChunkLevel::H1);
        chunk3.access_profile.created_at = 3000;

        store
            .insert_chunks(vec![chunk1, chunk2, chunk3])
            .await
            .unwrap();

        let result = store.list_entries(&[], None, None, 10).await.unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].id, "list-3");
        assert_eq!(result[1].id, "list-2");
        assert_eq!(result[2].id, "list-1");
    }

    #[tokio::test]
    async fn test_list_entries_with_perspective_filter() {
        let (store, _temp) = create_test_store().await;

        let mut chunk1 = create_test_chunk("lp-1", "Decisions", ChunkLevel::H1);
        chunk1.perspectives = vec!["decisions".to_string()];

        let mut chunk2 = create_test_chunk("lp-2", "Knowledge", ChunkLevel::H1);
        chunk2.perspectives = vec!["knowledge".to_string()];

        store.insert_chunks(vec![chunk1, chunk2]).await.unwrap();

        let result = store
            .list_entries(&["decisions"], None, None, 10)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "lp-1");
    }

    #[tokio::test]
    async fn test_list_entries_with_time_range() {
        let (store, _temp) = create_test_store().await;

        let mut chunk1 = create_test_chunk("lt-1", "First", ChunkLevel::H1);
        chunk1.access_profile.created_at = 1000;

        let mut chunk2 = create_test_chunk("lt-2", "Second", ChunkLevel::H1);
        chunk2.access_profile.created_at = 2000;

        let mut chunk3 = create_test_chunk("lt-3", "Third", ChunkLevel::H1);
        chunk3.access_profile.created_at = 3000;

        store
            .insert_chunks(vec![chunk1, chunk2, chunk3])
            .await
            .unwrap();

        let result = store
            .list_entries(&[], Some(1500), Some(2500), 10)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "lt-2");
    }

    #[tokio::test]
    async fn test_list_entries_limit() {
        let (store, _temp) = create_test_store().await;

        let mut chunk1 = create_test_chunk("ll-1", "First", ChunkLevel::H1);
        chunk1.access_profile.created_at = 1000;

        let mut chunk2 = create_test_chunk("ll-2", "Second", ChunkLevel::H1);
        chunk2.access_profile.created_at = 2000;

        let mut chunk3 = create_test_chunk("ll-3", "Third", ChunkLevel::H1);
        chunk3.access_profile.created_at = 3000;

        store
            .insert_chunks(vec![chunk1, chunk2, chunk3])
            .await
            .unwrap();

        let result = store.list_entries(&[], None, None, 2).await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, "ll-3");
        assert_eq!(result[1].id, "ll-2");
    }

    #[tokio::test]
    async fn test_impression_fields_roundtrip() {
        let (store, _temp) = create_test_store().await;

        let mut chunk = create_test_chunk("imp-1", "An impression", ChunkLevel::H1);
        chunk.entry_type = crate::chunk::EntryType::Impression;
        chunk.impression_hint = Some("uncertain".to_string());
        chunk.impression_strength = 0.4;
        store.insert_chunks(vec![chunk]).await.unwrap();

        let retrieved = store.get_by_id("imp-1").await.unwrap().unwrap();
        assert_eq!(retrieved.entry_type, crate::chunk::EntryType::Impression);
        assert_eq!(retrieved.impression_hint.as_deref(), Some("uncertain"));
        assert!((retrieved.impression_strength - 0.4).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_impression_fields_default() {
        let (store, _temp) = create_test_store().await;

        let chunk = create_test_chunk("imp-2", "No impression fields", ChunkLevel::H1);
        store.insert_chunks(vec![chunk]).await.unwrap();

        let retrieved = store.get_by_id("imp-2").await.unwrap().unwrap();
        assert_eq!(retrieved.impression_hint, None);
        assert!((retrieved.impression_strength - 1.0).abs() < 0.001);
    }

    // --- Schema migration tests ---

    /// Every non-core field in schema() must have a migration_default().
    #[tokio::test]
    async fn test_migration_default_coverage() {
        let core_fields = [
            "id",
            "content",
            "embedding",
            "level",
            "parent_id",
            "path",
            "source_file",
            "heading",
        ];
        let (store, _temp) = create_test_store().await;
        let schema = store.schema();
        for field in schema.fields() {
            if core_fields.contains(&field.name().as_str()) {
                continue;
            }
            assert!(
                migration_default(field.name()).is_some(),
                "Field '{}' has no migration_default()",
                field.name()
            );
        }
    }

    /// Create a table missing impression columns, then open LanceStore.
    /// Verify the columns were added and data is readable.
    #[tokio::test]
    async fn test_adds_missing_columns() {
        let temp_dir = TempDir::new().unwrap();
        let uri = temp_dir.path().to_string_lossy().to_string();
        let connection = connect(&uri).execute().await.unwrap();

        // Create a schema missing impression_hint and impression_strength
        let old_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 384),
                false,
            ),
            Field::new("level", DataType::UInt8, false),
            Field::new("parent_id", DataType::Utf8, true),
            Field::new("path", DataType::Utf8, false),
            Field::new("source_file", DataType::Utf8, false),
            Field::new("heading", DataType::Utf8, true),
            Field::new("cluster_memberships", DataType::Utf8, false),
            Field::new("entry_type", DataType::Utf8, false),
            Field::new("summarizes", DataType::Utf8, false),
            Field::new("perspectives", DataType::Utf8, false),
            Field::new("visibility", DataType::Utf8, false),
            Field::new("relations", DataType::Utf8, false),
            Field::new("created_at", DataType::Int64, false),
            Field::new("last_rolled", DataType::Int64, false),
            Field::new("access_hour", DataType::UInt16, false),
            Field::new("access_day", DataType::UInt16, false),
            Field::new("access_week", DataType::UInt16, false),
            Field::new("access_month", DataType::UInt16, false),
            Field::new("access_year", DataType::UInt16, false),
            Field::new("access_total", DataType::UInt32, false),
            Field::new("expires_at", DataType::Int64, true),
            // impression_hint and impression_strength intentionally omitted
        ]));

        connection
            .create_empty_table(TABLE_NAME, old_schema)
            .execute()
            .await
            .unwrap();

        // Now open via LanceStore — should auto-migrate
        let store = LanceStore::open(temp_dir.path(), 384, false).await.unwrap();

        // Verify we can insert and read back with the new fields
        let mut chunk = create_test_chunk("mig-1", "Migrated chunk", ChunkLevel::H1);
        chunk.impression_hint = Some("test".to_string());
        chunk.impression_strength = 0.5;
        store.insert_chunks(vec![chunk]).await.unwrap();

        let retrieved = store.get_by_id("mig-1").await.unwrap().unwrap();
        assert_eq!(retrieved.impression_hint.as_deref(), Some("test"));
        assert!((retrieved.impression_strength - 0.5).abs() < 0.001);
    }

    /// Fresh store → close → reopen. No errors.
    #[tokio::test]
    async fn test_noop_when_current() {
        let temp_dir = TempDir::new().unwrap();

        // First open creates the table
        let store = LanceStore::open(temp_dir.path(), 384, false).await.unwrap();
        let chunk = create_test_chunk("noop-1", "Noop test", ChunkLevel::H1);
        store.insert_chunks(vec![chunk]).await.unwrap();
        drop(store);

        // Second open should be a no-op migration
        let store2 = LanceStore::open(temp_dir.path(), 384, false).await.unwrap();
        let retrieved = store2.get_by_id("noop-1").await.unwrap().unwrap();
        assert_eq!(retrieved.content, "Noop test");
    }

    /// Table has an unknown column → error with "update veclayer".
    #[tokio::test]
    async fn test_old_client_detects_newer_store() {
        let temp_dir = TempDir::new().unwrap();

        // Create a store with normal schema first
        let store = LanceStore::open(temp_dir.path(), 384, false).await.unwrap();
        drop(store);

        // Add an unknown column to simulate a newer version
        let uri = temp_dir.path().to_string_lossy().to_string();
        let connection = connect(&uri).execute().await.unwrap();
        let table = connection.open_table(TABLE_NAME).execute().await.unwrap();
        table
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![(
                    "future_column".to_string(),
                    "cast(NULL as string)".to_string(),
                )]),
                None,
            )
            .await
            .unwrap();
        drop(table);
        drop(connection);

        // Now open should fail
        let result = LanceStore::open(temp_dir.path(), 384, false).await;
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("Expected error for newer store, but open succeeded"),
        };
        assert!(
            err_msg.contains("update veclayer"),
            "Error should mention update: {}",
            err_msg
        );
    }

    /// Table has a legacy column (last_accessed) → should NOT trigger error.
    #[tokio::test]
    async fn test_legacy_columns_not_flagged() {
        let temp_dir = TempDir::new().unwrap();

        // Create a store with normal schema
        let store = LanceStore::open(temp_dir.path(), 384, false).await.unwrap();
        drop(store);

        // Add a legacy column
        let uri = temp_dir.path().to_string_lossy().to_string();
        let connection = connect(&uri).execute().await.unwrap();
        let table = connection.open_table(TABLE_NAME).execute().await.unwrap();
        table
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![(
                    "last_accessed".to_string(),
                    "cast(0 as bigint)".to_string(),
                )]),
                None,
            )
            .await
            .unwrap();
        drop(table);
        drop(connection);

        // Should open fine — legacy columns are ignored
        let store2 = LanceStore::open(temp_dir.path(), 384, false).await;
        assert!(store2.is_ok(), "Legacy column should not cause error");
    }

    /// Column present in both schemas but with different type → error.
    #[tokio::test]
    async fn test_type_mismatch_detected() {
        let temp_dir = TempDir::new().unwrap();

        // Create a table with visibility as Int64 instead of Utf8
        let uri = temp_dir.path().to_string_lossy().to_string();
        std::fs::create_dir_all(temp_dir.path()).unwrap();
        let connection = connect(&uri).execute().await.unwrap();

        let bad_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 384),
                false,
            ),
            Field::new("level", DataType::UInt8, false),
            Field::new("parent_id", DataType::Utf8, true),
            Field::new("path", DataType::Utf8, false),
            Field::new("source_file", DataType::Utf8, false),
            Field::new("heading", DataType::Utf8, true),
            Field::new("cluster_memberships", DataType::Utf8, false),
            Field::new("entry_type", DataType::Utf8, false),
            Field::new("summarizes", DataType::Utf8, false),
            Field::new("perspectives", DataType::Utf8, false),
            // Wrong type: Int64 instead of Utf8
            Field::new("visibility", DataType::Int64, false),
            Field::new("relations", DataType::Utf8, false),
            Field::new("created_at", DataType::Int64, false),
            Field::new("last_rolled", DataType::Int64, false),
            Field::new("access_hour", DataType::UInt16, false),
            Field::new("access_day", DataType::UInt16, false),
            Field::new("access_week", DataType::UInt16, false),
            Field::new("access_month", DataType::UInt16, false),
            Field::new("access_year", DataType::UInt16, false),
            Field::new("access_total", DataType::UInt32, false),
            Field::new("expires_at", DataType::Int64, true),
            Field::new("impression_hint", DataType::Utf8, true),
            Field::new("impression_strength", DataType::Float32, false),
        ]));

        connection
            .create_empty_table(TABLE_NAME, bad_schema)
            .execute()
            .await
            .unwrap();
        drop(connection);

        let result = LanceStore::open(temp_dir.path(), 384, false).await;
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("Expected error for type mismatch, but open succeeded"),
        };
        assert!(
            err_msg.contains("visibility"),
            "Error should mention the mismatched column: {}",
            err_msg
        );
        assert!(
            err_msg.contains("different"),
            "Error should mention different schema: {}",
            err_msg
        );
    }

    /// Verify version metadata is stamped.
    #[tokio::test]
    async fn test_version_metadata_stamped() {
        let temp_dir = TempDir::new().unwrap();
        let store = LanceStore::open(temp_dir.path(), 384, false).await.unwrap();

        let table = store.get_table().await.unwrap();
        let schema = table.schema().await.unwrap();
        let metadata = schema.metadata();

        let expected_fp = schema_fingerprint(&store.schema());
        assert_eq!(
            metadata.get("veclayer::schema_fingerprint"),
            Some(&expected_fp)
        );
        assert!(
            metadata.contains_key("veclayer::commit"),
            "Should have commit info"
        );
    }

    #[test]
    fn test_schema_fingerprint_deterministic() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("value", DataType::Int64, false),
        ]);
        let fp1 = schema_fingerprint(&schema);
        let fp2 = schema_fingerprint(&schema);
        assert_eq!(fp1, fp2);
        assert_eq!(fp1.len(), 12, "Fingerprint should be 12 hex chars");
    }

    #[test]
    fn test_schema_fingerprint_changes_on_field_add() {
        let schema_a = Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("value", DataType::Int64, false),
        ]);
        let schema_b = Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("value", DataType::Int64, false),
            Field::new("extra", DataType::Utf8, true),
        ]);
        assert_ne!(schema_fingerprint(&schema_a), schema_fingerprint(&schema_b));
    }

    /// `open_metadata` creates the table on first call and is re-entrant for
    /// both read-write and read-only opens on the same directory.
    #[tokio::test]
    async fn test_open_metadata_basic() {
        let temp_dir = TempDir::new().unwrap();

        // First call (read-write) — must create the table and succeed.
        let store_rw = LanceStore::open_metadata(temp_dir.path(), false)
            .await
            .expect("open_metadata read-write must succeed on a fresh directory");

        let stats = store_rw
            .stats()
            .await
            .expect("stats on empty metadata store must succeed");
        assert_eq!(stats.total_chunks, 0);

        // Second call (read-only) on the same directory — must also succeed.
        let store_ro = LanceStore::open_metadata(temp_dir.path(), true)
            .await
            .expect("open_metadata read-only must succeed when table already exists");

        let stats_ro = store_ro
            .stats()
            .await
            .expect("stats on read-only metadata store must succeed");
        assert_eq!(stats_ro.total_chunks, 0);
    }

    /// A read-only store has `lock_dir = None`, so `with_write_lock` takes the
    /// `None` branch and issues no FileLock at all. Concretely: opening a second
    /// read-write store while a read-only store is open must succeed immediately
    /// without any lock contention, confirming the read-only path never acquires
    /// the lock file.
    #[tokio::test]
    async fn test_read_only_store_skips_write_lock() {
        let temp_dir = TempDir::new().unwrap();

        // Seed: create the table via a read-write open.
        let store_rw = LanceStore::open(temp_dir.path(), 384, false)
            .await
            .expect("initial read-write open must succeed");

        let chunk = create_test_chunk("ro-lock-1", "Seed chunk", ChunkLevel::H1);
        store_rw.insert_chunks(vec![chunk]).await.unwrap();

        // Open a read-only handle — this must not hold any lock file.
        let _store_ro = LanceStore::open(temp_dir.path(), 384, true)
            .await
            .expect("read-only open must succeed");

        // A fresh read-write open on the same directory must succeed immediately:
        // if the read-only store had held the lock this would block until timeout.
        let store_rw2 = LanceStore::open(temp_dir.path(), 384, false)
            .await
            .expect("second read-write open must not be blocked by the read-only handle");

        // Confirm the read-write handle is fully functional.
        let chunk2 = create_test_chunk("ro-lock-2", "Written after ro open", ChunkLevel::H1);
        store_rw2
            .insert_chunks(vec![chunk2])
            .await
            .expect("write via second read-write store must succeed");

        let retrieved = store_rw2
            .get_by_id("ro-lock-2")
            .await
            .unwrap()
            .expect("chunk must be readable back");
        assert_eq!(retrieved.content, "Written after ro open");
    }

    // ── read-only write-rejection tests ──────────────────────────────────────

    /// Helper: open a seeded read-only LanceStore. Returns the store and the
    /// TempDir (kept alive by the caller).
    async fn open_seeded_read_only(id: &str) -> (LanceStore, TempDir) {
        let dir = TempDir::new().unwrap();
        let rw = LanceStore::open(dir.path(), 384, false).await.unwrap();
        rw.insert_chunks(vec![create_test_chunk(id, "seed", ChunkLevel::H1)])
            .await
            .unwrap();
        // Release the write lock before the read-only open so the two handles
        // don't contend on the same FileLock.
        drop(rw);
        let ro = LanceStore::open(dir.path(), 384, true).await.unwrap();
        (ro, dir)
    }

    #[tokio::test]
    async fn test_read_only_store_rejects_insert_chunks() {
        let (ro, _dir) = open_seeded_read_only("ro-ins-seed").await;
        let chunk = create_test_chunk("ro-ins-new", "should not appear", ChunkLevel::H1);
        let err = ro.insert_chunks(vec![chunk]).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("write rejected: store was opened in read-only mode"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_read_only_store_rejects_delete_by_source() {
        let (ro, _dir) = open_seeded_read_only("ro-del-seed").await;
        let err = ro.delete_by_source("test.md").await.unwrap_err();
        assert!(
            err.to_string()
                .contains("write rejected: store was opened in read-only mode"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_read_only_store_rejects_update_visibility() {
        let (ro, _dir) = open_seeded_read_only("ro-vis-seed").await;
        let err = ro
            .update_visibility("ro-vis-seed", "private")
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("write rejected: store was opened in read-only mode"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_read_only_store_rejects_add_relation() {
        let (ro, _dir) = open_seeded_read_only("ro-rel-seed").await;
        let relation = crate::ChunkRelation::related_to("other");
        let err = ro.add_relation("ro-rel-seed", relation).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("write rejected: store was opened in read-only mode"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_read_only_store_rejects_batch_update_embeddings() {
        let (ro, _dir) = open_seeded_read_only("ro-emb-seed").await;
        let updates = vec![("ro-emb-seed".to_string(), vec![0.2_f32; 384])];
        let err = ro.batch_update_embeddings(updates).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("write rejected: store was opened in read-only mode"),
            "unexpected error: {err}"
        );
    }

    /// Rejected insert must leave contents unchanged: seed 1 row, reopen
    /// read-only, attempt insert → Err, reopen read-write → still 1 row.
    #[tokio::test]
    async fn test_read_only_rejected_insert_leaves_contents_unchanged() {
        let dir = TempDir::new().unwrap();

        // Seed one row via a read-write handle.
        let rw = LanceStore::open(dir.path(), 384, false).await.unwrap();
        rw.insert_chunks(vec![create_test_chunk(
            "ro-stable-seed",
            "original",
            ChunkLevel::H1,
        )])
        .await
        .unwrap();
        drop(rw);

        // Read-only handle: insert must be rejected.
        let ro = LanceStore::open(dir.path(), 384, true).await.unwrap();
        let chunk = create_test_chunk("ro-stable-new", "should not appear", ChunkLevel::H1);
        let err = ro.insert_chunks(vec![chunk]).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("write rejected: store was opened in read-only mode"),
            "unexpected error: {err}"
        );
        drop(ro);

        // Re-open read-write and confirm exactly 1 row remains.
        let rw2 = LanceStore::open(dir.path(), 384, false).await.unwrap();
        let stats = rw2.stats().await.unwrap();
        assert_eq!(
            stats.total_chunks, 1,
            "rejected insert must not change row count"
        );
        let original = rw2
            .get_by_id("ro-stable-seed")
            .await
            .unwrap()
            .expect("original row must still exist");
        assert_eq!(original.content, "original");
    }

    #[tokio::test]
    async fn test_read_only_store_rejects_update_access_profiles() {
        let (ro, _dir) = open_seeded_read_only("ro-acc-seed").await;
        // Non-empty updates: the empty case short-circuits to Ok before the lock.
        let updates = vec![("ro-acc-seed".to_string(), crate::AccessProfile::new())];
        let err = ro.update_access_profiles(updates).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("write rejected: store was opened in read-only mode"),
            "unexpected error: {err}"
        );
    }

    /// Multiple concurrent `open()` calls on the same fresh directory must all
    /// succeed. The migration lock serialises table creation so only one caller
    /// creates the table while others wait and see it already present.
    #[tokio::test]
    async fn test_concurrent_open_serialized_by_migration_lock() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();

        let p1 = path.clone();
        let p2 = path.clone();
        let p3 = path.clone();
        let p4 = path.clone();

        let (r1, r2, r3, r4) = tokio::join!(
            LanceStore::open(p1, 384, false),
            LanceStore::open(p2, 384, false),
            LanceStore::open(p3, 384, false),
            LanceStore::open(p4, 384, false),
        );

        let store1 = r1.expect("concurrent open #1 must succeed");
        let _store2 = r2.expect("concurrent open #2 must succeed");
        let _store3 = r3.expect("concurrent open #3 must succeed");
        let store4 = r4.expect("concurrent open #4 must succeed");

        // Write via one handle, read via another — confirms a single coherent table.
        let chunk = create_test_chunk("concurrent-open-1", "Raced into existence", ChunkLevel::H1);
        store1
            .insert_chunks(vec![chunk])
            .await
            .expect("insert after concurrent open must succeed");

        let retrieved = store4
            .get_by_id("concurrent-open-1")
            .await
            .expect("get_by_id must not error")
            .expect("chunk written by store1 must be visible via store4");
        assert_eq!(retrieved.content, "Raced into existence");
    }

    // --- is_commit_conflict unit tests ---

    #[test]
    fn test_is_commit_conflict_exact_capitalized() {
        // Matches the CommitConflict lance variant: "Commit conflict for version N"
        // wrapped as "lance error: Commit conflict for version N: ..."
        let e = lancedb::Error::Runtime {
            message: "Commit conflict for version 42: details here".to_string(),
        };
        assert!(
            is_commit_conflict(&e),
            "CommitConflict-style message must match"
        );
    }

    #[test]
    fn test_is_commit_conflict_lowercase() {
        // Matches the RetryableCommitConflict variant (lower-c in "commit")
        let e = lancedb::Error::Runtime {
            message: "commit conflict for version 7: retryable".to_string(),
        };
        assert!(
            is_commit_conflict(&e),
            "lowercase 'commit conflict' must match"
        );
    }

    #[test]
    fn test_is_commit_conflict_retryable_prefix() {
        // The full retryable message starts with "Retryable commit conflict"
        let e = lancedb::Error::Runtime {
            message: "Retryable commit conflict for version 3: source, loc".to_string(),
        };
        assert!(
            is_commit_conflict(&e),
            "'Retryable commit conflict' must match"
        );
    }

    #[test]
    fn test_is_commit_conflict_other_variant() {
        // lancedb::Error::Other displays as the raw message — verify it also works
        let e = lancedb::Error::Other {
            message: "lance error: Commit conflict for version 1: something".to_string(),
            source: None,
        };
        assert!(
            is_commit_conflict(&e),
            "Other variant with 'commit conflict' in message must match"
        );
    }

    #[test]
    fn test_is_commit_conflict_unrelated_error() {
        let e = lancedb::Error::Runtime {
            message: "some unrelated I/O failure".to_string(),
        };
        assert!(!is_commit_conflict(&e), "Unrelated error must not match");
    }

    #[test]
    fn test_is_commit_conflict_empty_message() {
        let e = lancedb::Error::Runtime {
            message: String::new(),
        };
        assert!(!is_commit_conflict(&e), "Empty message must not match");
    }

    #[test]
    fn test_is_commit_conflict_partial_word_no_match() {
        // "commitment" contains "commit" but not "commit conflict"
        let e = lancedb::Error::Runtime {
            message: "commitment to version control".to_string(),
        };
        assert!(!is_commit_conflict(&e), "'commitment' alone must not match");
    }

    #[test]
    fn test_is_commit_conflict_mixed_case() {
        // Case-insensitive check: "COMMIT CONFLICT" must match
        let e = lancedb::Error::Runtime {
            message: "COMMIT CONFLICT for version 10".to_string(),
        };
        assert!(
            is_commit_conflict(&e),
            "All-caps 'COMMIT CONFLICT' must match (case-insensitive)"
        );
    }

    // --- retry_on_conflict / public-API integration tests ---

    /// `update_access_profiles` with an empty slice returns Ok immediately —
    /// no table access is attempted.
    #[tokio::test]
    async fn test_update_access_profiles_empty_returns_ok() {
        let (store, _temp) = create_test_store().await;
        let result = store.update_access_profiles(vec![]).await;
        assert!(result.is_ok(), "empty update must return Ok");
    }

    /// `update_visibility` on a nonexistent chunk ID returns Err — a zero-row
    /// update means the target is missing, matching the SQLite backend and the
    /// `add_relation` contract (both reject unknown chunk IDs).
    #[tokio::test]
    async fn test_update_visibility_nonexistent_id_returns_err() {
        let (store, _temp) = create_test_store().await;
        let result = store
            .update_visibility("no-such-id-000000000000000000000000000000", "always")
            .await;
        assert!(
            result.is_err(),
            "update on nonexistent chunk must return Err, got Ok"
        );
    }

    /// Normal `update_access_profiles` call on existing data succeeds on the
    /// first attempt (green path through `retry_on_conflict`).
    #[tokio::test]
    async fn test_update_access_profiles_green_path() {
        let (store, _temp) = create_test_store().await;
        let chunk = create_test_chunk("uap-green-1", "Green path", ChunkLevel::H1);
        store.insert_chunks(vec![chunk]).await.unwrap();

        let before = store.get_by_id("uap-green-1").await.unwrap().unwrap();
        let mut profile = before.access_profile.clone();
        profile.total = 7;
        profile.hour = 2;

        store
            .update_access_profiles(vec![("uap-green-1".to_string(), profile)])
            .await
            .expect("update_access_profiles must succeed");

        let after = store.get_by_id("uap-green-1").await.unwrap().unwrap();
        assert_eq!(after.access_profile.total, 7);
        assert_eq!(after.access_profile.hour, 2);
    }

    /// Normal `update_visibility` succeeds on first attempt.
    #[tokio::test]
    async fn test_update_visibility_green_path() {
        let (store, _temp) = create_test_store().await;
        let chunk = create_test_chunk("uv-green-1", "Visibility green path", ChunkLevel::H1);
        store.insert_chunks(vec![chunk]).await.unwrap();

        store
            .update_visibility("uv-green-1", "deep_only")
            .await
            .expect("update_visibility must succeed");

        let after = store.get_by_id("uv-green-1").await.unwrap().unwrap();
        assert_eq!(after.visibility, "deep_only");
    }

    /// Normal `add_relation` succeeds on first attempt.
    #[tokio::test]
    async fn test_add_relation_green_path() {
        let (store, _temp) = create_test_store().await;
        let chunk = create_test_chunk("ar-green-1", "Relation green path", ChunkLevel::H1);
        let target = create_test_chunk("ar-green-2", "Target", ChunkLevel::H1);
        store.insert_chunks(vec![chunk, target]).await.unwrap();

        store
            .add_relation("ar-green-1", crate::ChunkRelation::related_to("ar-green-2"))
            .await
            .expect("add_relation must succeed");

        let after = store.get_by_id("ar-green-1").await.unwrap().unwrap();
        assert_eq!(after.relations.len(), 1);
        assert_eq!(after.relations[0].target_id, "ar-green-2");
    }

    /// Spawn two tasks that each call `update_visibility` on the same chunk.
    /// Both must succeed — `retry_on_conflict` handles any optimistic-concurrency
    /// conflict that arises from the simultaneous writes.
    #[tokio::test]
    async fn test_concurrent_update_visibility_both_succeed() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();

        // Seed data with a single shared open
        let seed = LanceStore::open(&path, 384, false).await.unwrap();
        let chunk = create_test_chunk("conc-vis-1", "Concurrent visibility", ChunkLevel::H1);
        seed.insert_chunks(vec![chunk]).await.unwrap();
        drop(seed);

        // Two independent stores writing to the same directory concurrently
        let p1 = path.clone();
        let p2 = path.clone();

        let (r1, r2) = tokio::join!(
            async move {
                let s = LanceStore::open(&p1, 384, false).await.unwrap();
                s.update_visibility("conc-vis-1", "always").await
            },
            async move {
                let s = LanceStore::open(&p2, 384, false).await.unwrap();
                s.update_visibility("conc-vis-1", "deep_only").await
            }
        );

        assert!(r1.is_ok(), "concurrent update #1 must succeed: {:?}", r1);
        assert!(r2.is_ok(), "concurrent update #2 must succeed: {:?}", r2);

        // The row exists, so neither update may spuriously hit the zero-row Err
        // guard; the surviving value must be one of the two writers' inputs.
        let verify = LanceStore::open(&path, 384, false).await.unwrap();
        let final_vis = verify
            .get_by_id("conc-vis-1")
            .await
            .unwrap()
            .unwrap()
            .visibility;
        assert!(
            final_vis == "always" || final_vis == "deep_only",
            "final visibility must be one of the concurrent writes, got {final_vis:?}"
        );
    }

    /// Concurrent store + recall: one task writes new chunks while another
    /// calls `stats()` repeatedly. Both must complete without errors.
    #[tokio::test]
    async fn test_concurrent_store_and_recall() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();

        // Create the table first
        let seed = LanceStore::open(&path, 384, false).await.unwrap();
        drop(seed);

        let p_writer = path.clone();
        let p_reader = path.clone();

        let writer = tokio::spawn(async move {
            let s = LanceStore::open(&p_writer, 384, false).await.unwrap();
            for i in 0..5u8 {
                let id = format!("conc-sr-{}", i);
                let chunk = create_test_chunk(&id, "concurrent content", ChunkLevel::H1);
                s.insert_chunks(vec![chunk]).await.unwrap();
            }
        });

        let reader = tokio::spawn(async move {
            let s = LanceStore::open(&p_reader, 384, false).await.unwrap();
            for _ in 0..5 {
                s.stats().await.unwrap();
            }
        });

        writer.await.expect("writer task must not panic");
        reader.await.expect("reader task must not panic");
    }

    // --- stamp_version skip / idempotent open tests ---

    /// Opening the same store a second time must not cause a commit conflict.
    /// The fingerprint check in `stamp_version` skips the `replace_schema_metadata`
    /// write when the metadata is already current.
    #[tokio::test]
    async fn test_idempotent_open_no_conflict() {
        let temp_dir = TempDir::new().unwrap();

        let store1 = LanceStore::open(temp_dir.path(), 384, false)
            .await
            .expect("first open must succeed");
        drop(store1);

        // Second open must succeed without any stamp_version write conflict
        let store2 = LanceStore::open(temp_dir.path(), 384, false)
            .await
            .expect("second open must succeed without commit conflict");
        drop(store2);

        // Third open for good measure
        LanceStore::open(temp_dir.path(), 384, false)
            .await
            .expect("third open must succeed");
    }

    /// After the first open stamps the fingerprint, the metadata key must be
    /// present and unchanged after a second open (no duplicate write happened).
    #[tokio::test]
    async fn test_stamp_version_fingerprint_stable_across_opens() {
        let temp_dir = TempDir::new().unwrap();

        let store1 = LanceStore::open(temp_dir.path(), 384, false).await.unwrap();
        let table1 = store1.get_table().await.unwrap();
        let fp_after_first = table1
            .schema()
            .await
            .unwrap()
            .metadata()
            .get("veclayer::schema_fingerprint")
            .cloned()
            .expect("fingerprint must be set after first open");
        // Record the table version after first open
        let version_after_first = table1.version().await.unwrap();
        drop(table1);
        drop(store1);

        let store2 = LanceStore::open(temp_dir.path(), 384, false).await.unwrap();
        let table2 = store2.get_table().await.unwrap();
        let fp_after_second = table2
            .schema()
            .await
            .unwrap()
            .metadata()
            .get("veclayer::schema_fingerprint")
            .cloned()
            .expect("fingerprint must still be set after second open");
        let version_after_second = table2.version().await.unwrap();

        assert_eq!(
            fp_after_first, fp_after_second,
            "fingerprint must not change on second open"
        );
        assert_eq!(
            version_after_first, version_after_second,
            "table version must not advance on second open (stamp_version skipped the write)"
        );
    }

    /// Four concurrent opens on the same fresh directory must all succeed.
    /// This exercises the stamp_version skip path under contention — if the
    /// skip were absent, concurrent `replace_schema_metadata` calls would
    /// produce commit conflicts.
    #[tokio::test]
    async fn test_concurrent_opens_no_commit_conflict() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();

        // Warm the table (first open creates it and stamps the fingerprint)
        LanceStore::open(&path, 384, false).await.unwrap();

        // Now open four handles simultaneously — each runs stamp_version
        // and must exit via the "fingerprint already current" early-return
        // rather than attempting a conflicting write
        let p1 = path.clone();
        let p2 = path.clone();
        let p3 = path.clone();
        let p4 = path.clone();

        let (r1, r2, r3, r4) = tokio::join!(
            LanceStore::open(p1, 384, false),
            LanceStore::open(p2, 384, false),
            LanceStore::open(p3, 384, false),
            LanceStore::open(p4, 384, false),
        );

        r1.expect("concurrent reopen #1 must succeed");
        r2.expect("concurrent reopen #2 must succeed");
        r3.expect("concurrent reopen #3 must succeed");
        r4.expect("concurrent reopen #4 must succeed");
    }

    // --- Edge-case tests ---

    /// `update_access_profiles` with a chunk ID that has no matching rows
    /// returns Ok — LanceDB update with no rows matched is not an error.
    #[tokio::test]
    async fn test_update_access_profiles_nonexistent_id_is_ok() {
        let (store, _temp) = create_test_store().await;
        let profile = crate::AccessProfile::default();
        let result = store
            .update_access_profiles(vec![(
                "nonexistent-chunk-id-000000000000".to_string(),
                profile,
            )])
            .await;
        assert!(
            result.is_ok(),
            "update_access_profiles on missing id must not error: {:?}",
            result
        );
    }

    // Verifies the per-write-operation locking design (issue #70).
    //
    // Design: multiple LanceStore instances MAY open the same directory concurrently.
    // Mutual exclusion is enforced only around individual write operations, not for the
    // lifetime of the store. The schema migration at open time is also serialised by a
    // short-lived write lock so two simultaneous opens cannot race on table creation or
    // column migrations. After open, each write acquires the lock independently, which
    // means interleaved writes from different handles are safe and the last writer wins.
    #[tokio::test]
    async fn test_concurrent_rw_access_works_with_per_write_lock() {
        let temp_dir = TempDir::new().unwrap();

        // Both read-write opens must succeed — no process-lifetime lock is held.
        let store1 = LanceStore::open(temp_dir.path(), 384, false)
            .await
            .expect("first read-write open must succeed");
        let store2 = LanceStore::open(temp_dir.path(), 384, false)
            .await
            .expect("second read-write open must succeed (per-write locking, no open-time lock)");

        // Writes from both handles must succeed and be visible to the other.
        let chunk1 = create_test_chunk("concurrent-rw-1", "Written by store1", ChunkLevel::H1);
        store1
            .insert_chunks(vec![chunk1])
            .await
            .expect("insert via store1 must succeed");

        let chunk2 = create_test_chunk("concurrent-rw-2", "Written by store2", ChunkLevel::H1);
        store2
            .insert_chunks(vec![chunk2])
            .await
            .expect("insert via store2 must succeed");

        // Both chunks must be retrievable from either store handle.
        let from_store2 = store2
            .get_by_id("concurrent-rw-1")
            .await
            .expect("get_by_id must not error")
            .expect("chunk written by store1 must be visible via store2");
        assert_eq!(from_store2.content, "Written by store1");

        let from_store1 = store1
            .get_by_id("concurrent-rw-2")
            .await
            .expect("get_by_id must not error")
            .expect("chunk written by store2 must be visible via store1");
        assert_eq!(from_store1.content, "Written by store2");

        // Read-only access on the same directory must always succeed.
        LanceStore::open(temp_dir.path(), 384, true)
            .await
            .expect("read-only open must succeed concurrently with read-write stores");
    }

    // --- SQL filter safety unit tests ---

    /// `sql_escape` doubles single quotes and strips NUL bytes.
    #[test]
    fn test_sql_escape_single_quote() {
        assert_eq!(sql_escape("it's"), "it''s");
        assert_eq!(sql_escape("''"), "''''");
        assert_eq!(sql_escape("no quotes"), "no quotes");
    }

    #[test]
    fn test_sql_escape_strips_null_bytes() {
        assert_eq!(sql_escape("abc\0def"), "abcdef");
        assert_eq!(sql_escape("\0"), "");
    }

    #[test]
    fn test_sql_escape_backslash_unchanged() {
        // Backslashes are literal in DataFusion single-quoted strings.
        // Escaping them would corrupt the value.
        assert_eq!(sql_escape(r"a\b"), r"a\b");
    }

    /// `like_escape_pattern` escapes LIKE metacharacters (`%`, `_`, `\`)
    /// and also handles single quotes and NUL bytes.
    #[test]
    fn test_like_escape_pattern_percent() {
        // A bare % would act as wildcard; after escaping it becomes a literal.
        assert_eq!(like_escape_pattern("100%"), r"100\%");
    }

    #[test]
    fn test_like_escape_pattern_underscore() {
        assert_eq!(like_escape_pattern("a_b"), r"a\_b");
    }

    #[test]
    fn test_like_escape_pattern_backslash() {
        // Backslash is our ESCAPE char in LIKE; it must be doubled.
        assert_eq!(like_escape_pattern(r"a\b"), r"a\\b");
    }

    #[test]
    fn test_like_escape_pattern_single_quote() {
        assert_eq!(like_escape_pattern("it's"), "it''s");
    }

    #[test]
    fn test_like_escape_pattern_null_byte() {
        assert_eq!(like_escape_pattern("abc\0def"), r"abcdef");
    }

    #[test]
    fn test_like_escape_pattern_combined() {
        // "100%_done\work" → all three metacharacters at once
        assert_eq!(like_escape_pattern(r"100%_done\work"), r"100\%\_done\\work");
    }

    // --- SQL injection integration tests ---

    /// Source file names containing SQL-special characters roundtrip correctly
    /// through `get_by_source` and `delete_by_source`.
    #[tokio::test]
    async fn test_source_file_with_sql_special_chars_roundtrips() {
        let (store, _temp) = create_test_store().await;

        let tricky_sources = [
            "'; DROP TABLE chunks; --",
            "path/to/file's notes.md",
            r"C:\Users\Alice\notes.md",
            "file with % wildcard.md",
            "file_with_underscores.md",
        ];

        for (i, source) in tricky_sources.iter().enumerate() {
            let mut chunk = create_test_chunk(&format!("src-sql-{i}"), "content", ChunkLevel::H1);
            chunk.source_file = source.to_string();
            store
                .insert_chunks(vec![chunk])
                .await
                .unwrap_or_else(|e| panic!("insert failed for source {:?}: {}", source, e));

            let retrieved = store
                .get_by_source(source)
                .await
                .unwrap_or_else(|e| panic!("get_by_source failed for {:?}: {}", source, e));
            assert_eq!(
                retrieved.len(),
                1,
                "expected exactly one chunk for source {:?}, got {}",
                source,
                retrieved.len()
            );
            assert_eq!(retrieved[0].source_file, *source);

            let deleted = store
                .delete_by_source(source)
                .await
                .unwrap_or_else(|e| panic!("delete_by_source failed for {:?}: {}", source, e));
            assert_eq!(
                deleted, 1,
                "expected one deleted chunk for source {:?}",
                source
            );

            // Confirm it's really gone.
            let after = store.get_by_source(source).await.unwrap();
            assert!(
                after.is_empty(),
                "chunk for source {:?} must be gone after delete",
                source
            );
        }
    }

    /// Perspective names with LIKE metacharacters match only the intended entries,
    /// not broader sets that would result from un-escaped `%` or `_`.
    #[tokio::test]
    async fn test_perspective_filter_with_like_metacharacters() {
        let (store, _temp) = create_test_store().await;

        // "100%" should match only chunks with that exact perspective,
        // not every chunk (which a bare `%` wildcard would return).
        let mut chunk_pct =
            create_test_chunk("persp-pct", "percentage perspective", ChunkLevel::H1);
        chunk_pct.perspectives = vec!["100%".to_string()];

        let mut chunk_normal =
            create_test_chunk("persp-normal", "normal perspective", ChunkLevel::H1);
        chunk_normal.perspectives = vec!["decisions".to_string()];

        // "a_b" should not match "axb" — underscore is a literal here.
        let mut chunk_under =
            create_test_chunk("persp-under", "underscore perspective", ChunkLevel::H1);
        chunk_under.perspectives = vec!["a_b".to_string()];

        let mut chunk_axb = create_test_chunk("persp-axb", "axb perspective", ChunkLevel::H1);
        chunk_axb.perspectives = vec!["axb".to_string()];

        store
            .insert_chunks(vec![chunk_pct, chunk_normal, chunk_under, chunk_axb])
            .await
            .unwrap();

        // Filter by "100%" — must return exactly one chunk, not all chunks.
        let query_vec = vec![0.1f32; 384];
        let results = store.search(&query_vec, 10, None, &["100%"]).await.unwrap();
        assert_eq!(
            results.len(),
            1,
            "100% perspective must match exactly one chunk"
        );
        assert_eq!(results[0].chunk.id, "persp-pct");

        // list_entries path
        let listed = store.list_entries(&["100%"], None, None, 10).await.unwrap();
        assert_eq!(
            listed.len(),
            1,
            "list_entries: 100% must match exactly one chunk"
        );
        assert_eq!(listed[0].id, "persp-pct");

        // Filter by "a_b" — must not match "axb".
        let results_under = store.search(&query_vec, 10, None, &["a_b"]).await.unwrap();
        assert_eq!(
            results_under.len(),
            1,
            "a_b must match only the a_b perspective, not axb"
        );
        assert_eq!(results_under[0].chunk.id, "persp-under");
    }

    /// Chunk IDs containing SQL-special characters are stored and retrieved correctly.
    /// (In practice IDs are SHA-256 hex, but the escape layer must be correct.)
    #[tokio::test]
    async fn test_chunk_id_with_sql_special_chars() {
        let (store, _temp) = create_test_store().await;

        // IDs with a single-quote — the only SQL-special char reachable via the
        // ID column in normal operation (sha256 hex is safe, but test the guard).
        let tricky_id = "abc'def";
        let mut chunk = create_test_chunk(tricky_id, "tricky id", ChunkLevel::H1);
        chunk.id = tricky_id.to_string();
        store.insert_chunks(vec![chunk]).await.unwrap();

        let retrieved = store.get_by_id(tricky_id).await.unwrap();
        assert!(
            retrieved.is_some(),
            "chunk with quote in ID must be retrievable"
        );
        assert_eq!(retrieved.unwrap().id, tricky_id);
    }

    // ── auto_compact_if_needed ────────────────────────────────────────────────

    /// auto_compact_if_needed on a read-only store must return Ok(default) — the
    /// early-return guard must still work after the error-propagation refactor.
    #[tokio::test]
    async fn test_auto_compact_noop_on_read_only_store() {
        let temp_dir = TempDir::new().unwrap();
        // Create the table via a read-write handle first.
        let rw = LanceStore::open(temp_dir.path(), 384, false).await.unwrap();
        rw.insert_chunks(vec![create_test_chunk(
            "ro-auto-001",
            "read-only auto compact test",
            ChunkLevel::CONTENT,
        )])
        .await
        .unwrap();
        drop(rw);

        let ro = LanceStore::open(temp_dir.path(), 384, true).await.unwrap();
        let result = ro.auto_compact_if_needed().await;
        assert!(
            result.is_ok(),
            "auto_compact_if_needed on a read-only store must return Ok: {:?}",
            result
        );
        let stats = result.unwrap();
        assert_eq!(stats.fragments_removed, 0);
        assert_eq!(stats.versions_removed, 0);
    }

    /// auto_compact_if_needed on a fresh store (versions <= MAX_VERSIONS) must
    /// return Ok(default) — the below-threshold early return must still work.
    #[tokio::test]
    async fn test_auto_compact_noop_below_threshold() {
        let (store, _dir) = create_test_store().await;
        // Only the initial table version exists — well below MAX_VERSIONS.
        let result = store.auto_compact_if_needed().await;
        assert!(
            result.is_ok(),
            "auto_compact_if_needed below threshold must not error: {:?}",
            result
        );
        let stats = result.unwrap();
        assert_eq!(stats.fragments_removed, 0);
        assert_eq!(stats.versions_removed, 0);
    }

    // NOTE: the open_table/list_versions error paths in auto_compact_if_needed
    // are not covered by a dedicated test because triggering them deterministically
    // requires a corrupted or deleted store directory. The code now mirrors the
    // already-tested force_compact implementation, so confidence comes from that
    // shared structure rather than a brittle filesystem-corruption fixture.

    // ── force_compact ─────────────────────────────────────────────────────────

    /// Each insert creates a new LanceDB version. After N inserts there are N+1
    /// versions (including the initial empty table version). force_compact must
    /// reduce that count and leave the data intact.
    #[tokio::test]
    async fn test_force_compact_reduces_version_count() {
        let (store, _dir) = create_test_store().await;

        // 20 separate inserts → 20 additional versions on top of the initial one.
        for i in 0..20u32 {
            let chunk = create_test_chunk(
                &format!("prune{i:03}"),
                &format!("prunable content {i}"),
                ChunkLevel::CONTENT,
            );
            store.insert_chunks(vec![chunk]).await.unwrap();
        }

        let table = store
            .connection
            .open_table(TABLE_NAME)
            .execute()
            .await
            .unwrap();
        let versions_before = table.list_versions().await.unwrap().len();
        assert!(
            versions_before > 1,
            "expected multiple versions after 20 inserts, got {versions_before}"
        );

        let stats = store.force_compact().await.unwrap();
        assert!(
            stats.versions_removed > 0,
            "force_compact must remove at least one old version (got {})",
            stats.versions_removed
        );

        let versions_after = table.list_versions().await.unwrap().len();
        assert!(
            versions_after < versions_before,
            "force_compact must reduce version count ({versions_before} → {versions_after})"
        );

        // Data must survive the compact.
        let store_stats = store.stats().await.unwrap();
        assert_eq!(
            store_stats.total_chunks, 20,
            "all 20 chunks must still be present after compact"
        );
    }

    /// force_compact on a freshly opened store (1 version, no history) must
    /// succeed without error even when there is nothing to reclaim.
    #[tokio::test]
    async fn test_force_compact_noop_on_fresh_store() {
        let (store, _dir) = create_test_store().await;
        // No writes — only the initial table version exists.
        let result = store.force_compact().await;
        assert!(
            result.is_ok(),
            "force_compact on fresh store must not error: {:?}",
            result
        );
    }

    /// force_compact on a read-only store must return an error immediately
    /// without touching the data.
    #[tokio::test]
    async fn test_force_compact_errors_on_read_only_store() {
        let temp_dir = TempDir::new().unwrap();
        // Open once read-write to create the table, then open read-only.
        let rw = LanceStore::open(temp_dir.path(), 384, false).await.unwrap();
        rw.insert_chunks(vec![create_test_chunk(
            "ro001",
            "read-only test",
            ChunkLevel::CONTENT,
        )])
        .await
        .unwrap();
        drop(rw);

        let ro = LanceStore::open(temp_dir.path(), 384, true).await.unwrap();
        assert!(
            ro.force_compact().await.is_err(),
            "force_compact on a read-only store must return an error"
        );
    }

    #[test]
    fn test_compute_prune_cutoff_empty_returns_zero() {
        let now = chrono::Utc::now();
        assert_eq!(
            LanceStore::compute_prune_cutoff(&[], 3, now),
            chrono::TimeDelta::zero(),
            "no versions means nothing to prune"
        );
    }

    #[test]
    fn test_compute_prune_cutoff_at_or_below_keep_returns_zero() {
        let now = chrono::Utc::now();
        let ts: Vec<_> = (1..=3).map(|h| now - chrono::TimeDelta::hours(h)).collect();
        // len == keep: every version is within the keep window → nothing to prune.
        assert_eq!(
            LanceStore::compute_prune_cutoff(&ts, 3, now),
            chrono::TimeDelta::zero()
        );
        // len < keep: likewise nothing to prune.
        assert_eq!(
            LanceStore::compute_prune_cutoff(&ts[..2], 3, now),
            chrono::TimeDelta::zero()
        );
    }

    #[test]
    fn test_compute_prune_cutoff_keep_zero_returns_zero() {
        let now = chrono::Utc::now();
        let ts: Vec<_> = (1..=3).map(|h| now - chrono::TimeDelta::hours(h)).collect();
        // keep == 0 must not collapse to "prune everything" — guard returns zero.
        assert_eq!(
            LanceStore::compute_prune_cutoff(&ts, 0, now),
            chrono::TimeDelta::zero()
        );
    }

    #[test]
    fn test_compute_prune_cutoff_normal_uses_keep_th_timestamp() {
        let now = chrono::Utc::now();
        // Newest-first: 1h, 2h, 3h, 4h ago.
        let ts: Vec<_> = (1..=4).map(|h| now - chrono::TimeDelta::hours(h)).collect();
        // keep=3 → cutoff is the 3rd most recent (3h ago), so older_than == 3h and
        // only the 4h-old version falls outside the keep window.
        assert_eq!(
            LanceStore::compute_prune_cutoff(&ts, 3, now),
            chrono::TimeDelta::hours(3)
        );
    }

    #[test]
    fn test_compute_prune_cutoff_keep_one_uses_most_recent() {
        let now = chrono::Utc::now();
        // keep=1 with 2 versions: cutoff is the single most recent (index 0, 1h ago),
        // so older_than == 1h and the 2h-old version is pruned. Guards the sharpest
        // index edge — keep-1 == 0 must read the newest, not panic or off-by-one.
        let ts = vec![
            now - chrono::TimeDelta::hours(1),
            now - chrono::TimeDelta::hours(2),
        ];
        assert_eq!(
            LanceStore::compute_prune_cutoff(&ts, 1, now),
            chrono::TimeDelta::hours(1)
        );
    }

    #[test]
    fn test_compute_prune_cutoff_future_timestamp_clamps_to_zero() {
        let now = chrono::Utc::now();
        // Clock skew: the keep-th most recent version is timestamped in the future.
        let ts = vec![
            now + chrono::TimeDelta::hours(2),
            now + chrono::TimeDelta::hours(1),
            now - chrono::TimeDelta::hours(1),
        ];
        // keep=2 → cutoff = ts[1] = now+1h → now - cutoff is negative → clamped to
        // zero so lance never receives a negative duration.
        assert_eq!(
            LanceStore::compute_prune_cutoff(&ts, 2, now),
            chrono::TimeDelta::zero()
        );
    }

    #[tokio::test]
    async fn test_batch_update_embeddings_skips_unknown_id() {
        let (store, _temp) = create_test_store().await;

        // Insert one chunk without an embedding so it starts as pending.
        let mut chunk = create_test_chunk("real-id", "Real chunk content", ChunkLevel::H1);
        chunk.embedding = None;
        store.insert_chunks(vec![chunk]).await.unwrap();

        // Verify the chunk is pending (not yet searchable).
        let query = vec![0.1f32; 384];
        let results_before = store.search(&query, 10, None, &[]).await.unwrap();
        assert!(
            results_before.is_empty(),
            "pending chunk must not appear in vector search"
        );

        // Call batch_update_embeddings with one real id and one ghost id.
        let new_embedding = vec![0.1f32; 384];
        let result = store
            .batch_update_embeddings(vec![
                ("real-id".to_string(), new_embedding.clone()),
                ("ghost-id".to_string(), vec![0.9f32; 384]),
            ])
            .await;

        // Must return Ok — unknown ids are skipped, not an error.
        assert!(
            result.is_ok(),
            "ghost id must not cause an error: {:?}",
            result
        );

        // The real chunk must still exist and have the updated embedding.
        let retrieved = store
            .get_by_id("real-id")
            .await
            .unwrap()
            .expect("real chunk must still be present after update");
        assert_eq!(retrieved.id, "real-id");
        assert_eq!(
            retrieved.embedding.as_deref(),
            Some(new_embedding.as_slice()),
            "embedding must be updated to the new vector"
        );

        // The real chunk must now be searchable.
        let results_after = store.search(&query, 10, None, &[]).await.unwrap();
        assert_eq!(
            results_after.len(),
            1,
            "real chunk must be searchable after embedding update"
        );
        assert_eq!(results_after[0].chunk.id, "real-id");

        // The ghost id must not have created a spurious row.
        let ghost = store.get_by_id("ghost-id").await.unwrap();
        assert!(
            ghost.is_none(),
            "ghost id must not have been inserted into the store"
        );
    }

    // ── timestamp_filter_value unit tests ─────────────────────────────────────

    /// Positive timestamps produce a plain decimal string with no SQL-special chars.
    #[test]
    fn test_timestamp_filter_value_positive() {
        assert_eq!(timestamp_filter_value(0), "0");
        assert_eq!(timestamp_filter_value(1), "1");
        assert_eq!(timestamp_filter_value(1_716_816_000), "1716816000");
        assert_eq!(timestamp_filter_value(i64::MAX), i64::MAX.to_string());
    }

    /// Negative timestamps (pre-epoch or timezone offsets) produce a leading minus
    /// and nothing else — no quotes, operators, or keywords.
    #[test]
    fn test_timestamp_filter_value_negative() {
        assert_eq!(timestamp_filter_value(-1), "-1");
        assert_eq!(timestamp_filter_value(-3600), "-3600");
        assert_eq!(timestamp_filter_value(i64::MIN), i64::MIN.to_string());
    }

    /// The output contains only characters from `[-0-9]` — impossible to break
    /// out of a numeric position in a filter predicate regardless of value.
    #[test]
    fn test_timestamp_filter_value_only_numeric_chars() {
        for ts in [i64::MIN, -1, 0, 1, i64::MAX] {
            let s = timestamp_filter_value(ts);
            assert!(
                s.chars().all(|c| c.is_ascii_digit() || c == '-'),
                "timestamp_filter_value({ts}) = {s:?} contains non-numeric chars"
            );
            // No SQL keywords or metacharacters.
            assert!(!s.contains('\''), "must not contain single quote");
            assert!(!s.contains(' '), "must not contain space");
            assert!(!s.contains(';'), "must not contain semicolon");
        }
    }

    // ── search_text tests ─────────────────────────────────────────────────────

    /// Empty query string returns all chunks (no word filters applied).
    #[tokio::test]
    async fn test_search_text_empty_query_returns_all() {
        let (store, _temp) = create_test_store().await;

        let chunk1 = create_test_chunk("st-all-1", "Alpha content", ChunkLevel::H1);
        let chunk2 = create_test_chunk("st-all-2", "Beta content", ChunkLevel::H1);
        store.insert_chunks(vec![chunk1, chunk2]).await.unwrap();

        let results = store.search_text("", &[], None, None, 10).await.unwrap();
        let ids: Vec<&str> = results.iter().map(|c| c.id.as_str()).collect();
        assert!(
            ids.contains(&"st-all-1") && ids.contains(&"st-all-2"),
            "empty query must return all inserted chunks, got {ids:?}"
        );
    }

    /// Query with no matching content returns an empty vec, not an error.
    #[tokio::test]
    async fn test_search_text_no_matches_returns_empty() {
        let (store, _temp) = create_test_store().await;

        let chunk = create_test_chunk("st-nm-1", "Only this content", ChunkLevel::H1);
        store.insert_chunks(vec![chunk]).await.unwrap();

        let results = store
            .search_text("xyzzy_not_present", &[], None, None, 10)
            .await
            .unwrap();
        assert!(
            results.is_empty(),
            "non-matching query must return empty results"
        );
    }

    /// k larger than the table size returns all matching rows, not an error.
    #[tokio::test]
    async fn test_search_text_k_larger_than_table_returns_all() {
        let (store, _temp) = create_test_store().await;

        let chunk1 = create_test_chunk("st-big-1", "Rust programming", ChunkLevel::H1);
        let chunk2 = create_test_chunk("st-big-2", "Rust memory safety", ChunkLevel::H1);
        store.insert_chunks(vec![chunk1, chunk2]).await.unwrap();

        // Limit of 9999 — much larger than the 2 rows in the table.
        let results = store
            .search_text("Rust", &[], None, None, 9999)
            .await
            .unwrap();
        assert_eq!(
            results.len(),
            2,
            "limit larger than table size must return all matching rows"
        );
    }

    /// `search_text` respects the `limit` parameter.
    #[tokio::test]
    async fn test_search_text_limit_is_respected() {
        let (store, _temp) = create_test_store().await;

        let mut chunk1 = create_test_chunk("st-lim-1", "common word alpha", ChunkLevel::H1);
        chunk1.access_profile.created_at = 3000;
        let mut chunk2 = create_test_chunk("st-lim-2", "common word beta", ChunkLevel::H1);
        chunk2.access_profile.created_at = 2000;
        let mut chunk3 = create_test_chunk("st-lim-3", "common word gamma", ChunkLevel::H1);
        chunk3.access_profile.created_at = 1000;

        store
            .insert_chunks(vec![chunk1, chunk2, chunk3])
            .await
            .unwrap();

        let results = store
            .search_text("common", &[], None, None, 2)
            .await
            .unwrap();
        assert_eq!(results.len(), 2, "limit=2 must return at most 2 results");
    }

    /// Results are sorted newest-first (descending `created_at`).
    #[tokio::test]
    async fn test_search_text_ordering_newest_first() {
        let (store, _temp) = create_test_store().await;

        let mut chunk_old = create_test_chunk("st-ord-old", "searchable content", ChunkLevel::H1);
        chunk_old.access_profile.created_at = 1000;

        let mut chunk_mid = create_test_chunk("st-ord-mid", "searchable content", ChunkLevel::H1);
        chunk_mid.access_profile.created_at = 2000;

        let mut chunk_new = create_test_chunk("st-ord-new", "searchable content", ChunkLevel::H1);
        chunk_new.access_profile.created_at = 3000;

        store
            .insert_chunks(vec![chunk_old, chunk_mid, chunk_new])
            .await
            .unwrap();

        let results = store
            .search_text("searchable", &[], None, None, 10)
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(
            results[0].id, "st-ord-new",
            "first result must be the newest chunk"
        );
        assert_eq!(
            results[1].id, "st-ord-mid",
            "second result must be the middle chunk"
        );
        assert_eq!(
            results[2].id, "st-ord-old",
            "third result must be the oldest chunk"
        );
    }

    /// `since` filter excludes chunks created before the boundary.
    #[tokio::test]
    async fn test_search_text_since_filter() {
        let (store, _temp) = create_test_store().await;

        let mut before = create_test_chunk("st-since-before", "needle content", ChunkLevel::H1);
        before.access_profile.created_at = 999;

        let mut at_boundary = create_test_chunk("st-since-at", "needle content", ChunkLevel::H1);
        at_boundary.access_profile.created_at = 1000;

        let mut after = create_test_chunk("st-since-after", "needle content", ChunkLevel::H1);
        after.access_profile.created_at = 1001;

        store
            .insert_chunks(vec![before, at_boundary, after])
            .await
            .unwrap();

        // since=1000: rows with created_at >= 1000 are included.
        let results = store
            .search_text("needle", &[], Some(1000), None, 10)
            .await
            .unwrap();

        assert_eq!(
            results.len(),
            2,
            "since=1000 must return rows at and after boundary"
        );
        let ids: Vec<&str> = results.iter().map(|c| c.id.as_str()).collect();
        assert!(
            ids.contains(&"st-since-at"),
            "boundary row must be included"
        );
        assert!(
            ids.contains(&"st-since-after"),
            "after row must be included"
        );
        assert!(
            !ids.contains(&"st-since-before"),
            "before row must be excluded"
        );
    }

    /// `until` filter excludes chunks created after the boundary.
    #[tokio::test]
    async fn test_search_text_until_filter() {
        let (store, _temp) = create_test_store().await;

        let mut before = create_test_chunk("st-until-before", "marker content", ChunkLevel::H1);
        before.access_profile.created_at = 999;

        let mut at_boundary = create_test_chunk("st-until-at", "marker content", ChunkLevel::H1);
        at_boundary.access_profile.created_at = 1000;

        let mut after = create_test_chunk("st-until-after", "marker content", ChunkLevel::H1);
        after.access_profile.created_at = 1001;

        store
            .insert_chunks(vec![before, at_boundary, after])
            .await
            .unwrap();

        // until=1000: rows with created_at <= 1000 are included.
        let results = store
            .search_text("marker", &[], None, Some(1000), 10)
            .await
            .unwrap();

        assert_eq!(
            results.len(),
            2,
            "until=1000 must return rows at and before boundary"
        );
        let ids: Vec<&str> = results.iter().map(|c| c.id.as_str()).collect();
        assert!(
            ids.contains(&"st-until-before"),
            "before row must be included"
        );
        assert!(
            ids.contains(&"st-until-at"),
            "boundary row must be included"
        );
        assert!(
            !ids.contains(&"st-until-after"),
            "after row must be excluded"
        );
    }

    /// Boundary timestamps at i64::MIN and i64::MAX do not panic or cause a
    /// filter-injection — they emit as plain decimal integers and execute safely.
    #[tokio::test]
    async fn test_search_text_extreme_timestamps_do_not_panic() {
        let (store, _temp) = create_test_store().await;

        let mut chunk = create_test_chunk("st-ext-1", "extreme content", ChunkLevel::H1);
        chunk.access_profile.created_at = 5000;
        store.insert_chunks(vec![chunk]).await.unwrap();

        // i64::MIN as `since` — everything has created_at >= i64::MIN, so all rows returned.
        let results_min = store
            .search_text("extreme", &[], Some(i64::MIN), None, 10)
            .await;
        assert!(
            results_min.is_ok(),
            "since=i64::MIN must not error: {:?}",
            results_min
        );
        assert_eq!(results_min.unwrap().len(), 1);

        // i64::MAX as `until` — everything has created_at <= i64::MAX, so all rows returned.
        let results_max = store
            .search_text("extreme", &[], None, Some(i64::MAX), 10)
            .await;
        assert!(
            results_max.is_ok(),
            "until=i64::MAX must not error: {:?}",
            results_max
        );
        assert_eq!(results_max.unwrap().len(), 1);
    }

    // ── batch_update_embeddings atomicity / partial-failure tests ─────────────

    /// Partial-failure semantics (documented): when batch_update_embeddings is
    /// called with a mix of known and unknown IDs, the known IDs are updated and
    /// the call returns Ok. The unknown IDs are silently skipped (logged WARN).
    ///
    /// This test pins the documented behavior: partial success is not an error;
    /// the rows that were found are correctly updated and no spurious rows appear.
    #[tokio::test]
    async fn test_batch_update_embeddings_partial_success_is_ok() {
        let (store, _temp) = create_test_store().await;

        let mut chunk_a = create_test_chunk("bue-a", "chunk a", ChunkLevel::H1);
        chunk_a.embedding = None;
        let mut chunk_b = create_test_chunk("bue-b", "chunk b", ChunkLevel::H1);
        chunk_b.embedding = None;

        store.insert_chunks(vec![chunk_a, chunk_b]).await.unwrap();

        // Mix: two real IDs + one ghost. The ghost is silently skipped.
        let result = store
            .batch_update_embeddings(vec![
                ("bue-a".to_string(), vec![0.1f32; 384]),
                ("bue-ghost".to_string(), vec![0.5f32; 384]),
                ("bue-b".to_string(), vec![0.2f32; 384]),
            ])
            .await;

        assert!(
            result.is_ok(),
            "mixed known/unknown IDs must return Ok: {:?}",
            result
        );

        // Both real chunks must have been updated and be searchable.
        let retrieved_a = store.get_by_id("bue-a").await.unwrap().unwrap();
        assert!(
            retrieved_a.embedding.is_some(),
            "bue-a must have an embedding after update"
        );

        let retrieved_b = store.get_by_id("bue-b").await.unwrap().unwrap();
        assert!(
            retrieved_b.embedding.is_some(),
            "bue-b must have an embedding after update"
        );

        // The ghost must not have been inserted.
        assert!(
            store.get_by_id("bue-ghost").await.unwrap().is_none(),
            "ghost ID must not have been inserted"
        );
    }

    /// All-unknown batch returns Ok with a no-op: nothing is deleted or inserted.
    #[tokio::test]
    async fn test_batch_update_embeddings_all_unknown_is_noop() {
        let (store, _temp) = create_test_store().await;

        let chunk = create_test_chunk("bue-noop-1", "untouched", ChunkLevel::H1);
        store.insert_chunks(vec![chunk]).await.unwrap();

        let result = store
            .batch_update_embeddings(vec![
                ("ghost-x".to_string(), vec![0.1f32; 384]),
                ("ghost-y".to_string(), vec![0.2f32; 384]),
            ])
            .await;

        assert!(
            result.is_ok(),
            "all-unknown batch must return Ok: {:?}",
            result
        );

        // The existing chunk must still be present and unmodified.
        let still_there = store.get_by_id("bue-noop-1").await.unwrap();
        assert!(
            still_there.is_some(),
            "existing chunk must not be affected by all-unknown batch"
        );
    }

    // ── start_offset / end_offset const tests ─────────────────────────────────

    /// Chunks reconstructed from LanceDB always have start_offset == end_offset == 0,
    /// because byte offsets are not persisted in the store.
    /// This test pins the value of CHUNK_BYTE_OFFSET_NOT_STORED and confirms
    /// that round-tripped chunks carry the documented zero default.
    #[test]
    fn test_chunk_byte_offset_not_stored_is_zero() {
        assert_eq!(
            CHUNK_BYTE_OFFSET_NOT_STORED, 0,
            "CHUNK_BYTE_OFFSET_NOT_STORED must be 0"
        );
    }

    #[tokio::test]
    async fn test_round_tripped_chunk_has_zero_byte_offsets() {
        let (store, _temp) = create_test_store().await;

        // Insert a chunk that was created with non-zero offsets during parsing.
        let mut chunk = create_test_chunk("offsets-1", "Offset test content", ChunkLevel::H1);
        chunk.start_offset = 42;
        chunk.end_offset = 99;
        store.insert_chunks(vec![chunk]).await.unwrap();

        // When read back from LanceDB the offsets must be zeroed (not persisted).
        let retrieved = store.get_by_id("offsets-1").await.unwrap().unwrap();
        assert_eq!(
            retrieved.start_offset, CHUNK_BYTE_OFFSET_NOT_STORED,
            "start_offset must be {} after LanceDB round-trip",
            CHUNK_BYTE_OFFSET_NOT_STORED
        );
        assert_eq!(
            retrieved.end_offset, CHUNK_BYTE_OFFSET_NOT_STORED,
            "end_offset must be {} after LanceDB round-trip",
            CHUNK_BYTE_OFFSET_NOT_STORED
        );
    }

    // ── ensure_table read-only contract tests ────────────────────────────────

    /// A read-only open of a FRESH (never-initialized) directory must succeed
    /// and yield a readable, empty store. Maintenance commands (reflect/compact)
    /// and cross-project reads rely on this graceful-empty bootstrap.
    #[tokio::test]
    async fn test_read_only_open_fresh_dir_yields_empty_store() {
        let dir = TempDir::new().unwrap();
        let ro = LanceStore::open(dir.path(), 384, true)
            .await
            .unwrap_or_else(|e| panic!("read-only open of fresh dir must succeed, got: {e}"));
        let stats = ro.stats().await.expect("stats on empty store must work");
        assert_eq!(stats.total_chunks, 0, "fresh store must be empty");

        // The read-only bootstrap creates the empty table but must skip the
        // cosmetic stamp_version write, so it leaves strictly fewer LanceDB
        // versions than a read-write bootstrap (which additionally stamps schema
        // metadata). Comparing the two avoids coupling to internal version counts.
        let (_t, ro_versions) = ro.open_table_and_versions().await.unwrap();
        let rw_dir = TempDir::new().unwrap();
        let rw = LanceStore::open(rw_dir.path(), 384, false).await.unwrap();
        let (_t2, rw_versions) = rw.open_table_and_versions().await.unwrap();
        assert!(
            ro_versions.len() < rw_versions.len(),
            "read-only bootstrap must skip stamp_version (ro={}, rw={})",
            ro_versions.len(),
            rw_versions.len()
        );
    }

    /// A read-only open of an ALREADY-SEEDED, current-schema store must succeed,
    /// be readable, and run NO migration: the LanceDB version count observed
    /// through the read-only handle itself must equal the pre-open count.
    #[tokio::test]
    async fn test_read_only_open_seeded_store_does_not_migrate() {
        let dir = TempDir::new().unwrap();

        // Seed one row via a read-write handle and capture the version count.
        let rw = LanceStore::open(dir.path(), 384, false).await.unwrap();
        rw.insert_chunks(vec![create_test_chunk(
            "ro-immutable-seed",
            "seeded",
            ChunkLevel::H1,
        )])
        .await
        .unwrap();
        let (_t, versions_before) = rw.open_table_and_versions().await.unwrap();
        let count_before = versions_before.len();
        drop(rw);

        // Read-only open must succeed and read back the seeded row.
        let ro = LanceStore::open(dir.path(), 384, true)
            .await
            .unwrap_or_else(|e| panic!("read-only open of seeded store must succeed, got: {e}"));
        let chunk = ro
            .get_by_id("ro-immutable-seed")
            .await
            .expect("get_by_id must not error")
            .expect("seeded chunk must be present");
        assert_eq!(chunk.content, "seeded");

        // Measure the version count through the read-only handle itself — a
        // read that cannot itself write — so the assertion is independent of any
        // later read-write reopen.
        let (_t2, versions_after) = ro.open_table_and_versions().await.unwrap();
        assert_eq!(
            versions_after.len(),
            count_before,
            "read-only open must not create new LanceDB versions (was {}, now {})",
            count_before,
            versions_after.len()
        );
    }
}
