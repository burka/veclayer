//! SQLite backend for VectorStore.
//!
//! Stores all chunk data in a single `chunks` table using standard SQL types.
//! Embeddings are stored as little-endian f32 BLOBs. JSON columns hold
//! perspectives, relations, cluster_memberships, and summarizes.
//!
//! WAL mode is enabled on open for better concurrent read performance.

use std::path::Path;
use std::sync::{Arc, Mutex};

use rusqlite::{params, Connection, OptionalExtension};

use super::{
    FileLock, SearchResult, StoreStats, VectorStore, EMBEDDING_STATUS_EMBEDDED,
    EMBEDDING_STATUS_PENDING,
};
use crate::access_profile::now_epoch_secs;
use crate::chunk::EntryType;
use crate::Error;
use crate::{
    AccessProfile, ChunkLevel, ChunkRelation, ClusterMembership, HierarchicalChunk, Result,
};

/// Build a parameterized LIKE pattern that matches a perspective stored as a
/// JSON string element (e.g. `["decisions","learnings"]`).
///
/// The returned string has the form `%"perspective"%` with LIKE wildcards
/// (`%`, `_`) and backslash escaped so the caller can bind it as a query
/// parameter with `ESCAPE '\'`.  SQL structural characters are never
/// interpolated into the query text, so injection is impossible regardless of
/// what the caller passes in.
fn perspective_like_pattern(perspective: &str) -> String {
    let escaped = perspective
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_");
    format!("%\"{escaped}\"%")
}

// --- Schema ---

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS chunks (
    id                  TEXT PRIMARY KEY,
    content             TEXT NOT NULL,
    embedding           BLOB,
    embedding_status    TEXT NOT NULL DEFAULT 'pending',
    level               INTEGER NOT NULL,
    parent_id           TEXT,
    path                TEXT NOT NULL,
    source_file         TEXT NOT NULL,
    heading             TEXT,
    start_offset        INTEGER NOT NULL DEFAULT 0,
    end_offset          INTEGER NOT NULL DEFAULT 0,
    doc_type            TEXT NOT NULL DEFAULT 'memory',
    entry_type          TEXT NOT NULL DEFAULT 'raw',
    visibility          TEXT NOT NULL DEFAULT 'normal',
    perspectives        TEXT NOT NULL DEFAULT '[]',
    relations           TEXT NOT NULL DEFAULT '[]',
    cluster_memberships TEXT NOT NULL DEFAULT '[]',
    summarizes          TEXT NOT NULL DEFAULT '[]',
    created_at          INTEGER NOT NULL,
    last_rolled         INTEGER NOT NULL,
    access_hour         INTEGER NOT NULL DEFAULT 0,
    access_day          INTEGER NOT NULL DEFAULT 0,
    access_week         INTEGER NOT NULL DEFAULT 0,
    access_month        INTEGER NOT NULL DEFAULT 0,
    access_year         INTEGER NOT NULL DEFAULT 0,
    access_total        INTEGER NOT NULL DEFAULT 0,
    expires_at          INTEGER,
    impression_hint     TEXT,
    impression_strength REAL NOT NULL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_file);
CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks(embedding_status);
CREATE INDEX IF NOT EXISTS idx_chunks_access ON chunks(access_total);
CREATE INDEX IF NOT EXISTS idx_chunks_created ON chunks(created_at);
CREATE INDEX IF NOT EXISTS idx_chunks_last_rolled ON chunks(last_rolled);
";

// --- Struct ---

/// SQLite-backed vector store.
///
/// Thread-safe via `Arc<Mutex<Connection>>`. All blocking operations are
/// dispatched to `tokio::task::spawn_blocking` to avoid stalling the async
/// executor.
pub(crate) struct SqliteStore {
    conn: Arc<Mutex<Connection>>,
    /// Expected embedding dimension. Every inserted or updated embedding must
    /// have exactly this many `f32` elements.
    dimension: usize,
    /// When `true`, all write operations are rejected with a store error.
    read_only: bool,
    _lock: Option<FileLock>,
}

// --- Binary helpers ---

fn vec_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for &f in v {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    bytes
}

fn bytes_to_vec(b: &[u8]) -> rusqlite::Result<Vec<f32>> {
    if !b.len().is_multiple_of(4) {
        return Err(rusqlite::Error::FromSqlConversionFailure(
            0,
            rusqlite::types::Type::Blob,
            format!("embedding blob length {} is not a multiple of 4", b.len()).into(),
        ));
    }
    Ok(b.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk is exactly 4 bytes")))
        .collect())
}

// --- Row mapping ---

fn entry_type_from_str(s: &str) -> EntryType {
    s.parse().unwrap_or_default()
}

/// Convert an `i64` column value to a narrower integer type, failing loudly on
/// out-of-range data instead of silently truncating with an `as` cast.
fn narrow<T: TryFrom<i64>>(value: i64, column: &str) -> rusqlite::Result<T> {
    T::try_from(value).map_err(|_| {
        rusqlite::Error::FromSqlConversionFailure(
            0,
            rusqlite::types::Type::Integer,
            format!("column '{column}' value {value} is out of range").into(),
        )
    })
}

fn row_to_chunk(row: &rusqlite::Row) -> rusqlite::Result<HierarchicalChunk> {
    let id: String = row.get("id")?;
    let content: String = row.get("content")?;
    let embedding_blob: Option<Vec<u8>> = row.get("embedding")?;
    let embedding_status: String = row.get("embedding_status")?;
    let level_depth: u8 = narrow(row.get::<_, i64>("level")?, "level")?;
    let parent_id: Option<String> = row.get("parent_id")?;
    let path: String = row.get("path")?;
    let source_file: String = row.get("source_file")?;
    let heading: Option<String> = row.get("heading")?;
    let start_offset: usize = narrow(row.get::<_, i64>("start_offset")?, "start_offset")?;
    let end_offset: usize = narrow(row.get::<_, i64>("end_offset")?, "end_offset")?;
    let entry_type_str: String = row.get("entry_type")?;
    let visibility: String = row.get("visibility")?;
    let perspectives_json: String = row.get("perspectives")?;
    let relations_json: String = row.get("relations")?;
    let cluster_memberships_json: String = row.get("cluster_memberships")?;
    let summarizes_json: String = row.get("summarizes")?;
    let created_at: i64 = row.get("created_at")?;
    let last_rolled: i64 = row.get("last_rolled")?;
    let access_hour: u16 = narrow(row.get::<_, i64>("access_hour")?, "access_hour")?;
    let access_day: u16 = narrow(row.get::<_, i64>("access_day")?, "access_day")?;
    let access_week: u16 = narrow(row.get::<_, i64>("access_week")?, "access_week")?;
    let access_month: u16 = narrow(row.get::<_, i64>("access_month")?, "access_month")?;
    let access_year: u16 = narrow(row.get::<_, i64>("access_year")?, "access_year")?;
    let access_total: u32 = narrow(row.get::<_, i64>("access_total")?, "access_total")?;
    let expires_at: Option<i64> = row.get("expires_at")?;
    let impression_hint: Option<String> = row.get("impression_hint")?;
    let impression_strength: f32 = row.get::<_, f64>("impression_strength")? as f32;

    let embedding = if embedding_status == EMBEDDING_STATUS_PENDING {
        None
    } else {
        embedding_blob.map(|b| bytes_to_vec(&b)).transpose()?
    };

    let perspectives: Vec<String> = serde_json::from_str(&perspectives_json).unwrap_or_default();
    let relations: Vec<ChunkRelation> = serde_json::from_str(&relations_json).unwrap_or_default();
    let cluster_memberships: Vec<ClusterMembership> =
        serde_json::from_str(&cluster_memberships_json).unwrap_or_default();
    let summarizes: Vec<String> = serde_json::from_str(&summarizes_json).unwrap_or_default();

    Ok(HierarchicalChunk {
        id,
        content,
        embedding,
        level: ChunkLevel(level_depth),
        parent_id,
        path,
        source_file,
        heading,
        start_offset,
        end_offset,
        cluster_memberships,
        entry_type: entry_type_from_str(&entry_type_str),
        summarizes,
        visibility,
        perspectives,
        relations,
        access_profile: AccessProfile {
            created_at,
            last_rolled,
            hour: access_hour,
            day: access_day,
            week: access_week,
            month: access_month,
            year: access_year,
            total: access_total,
        },
        expires_at,
        impression_hint,
        impression_strength,
    })
}

// --- Constructor ---

impl SqliteStore {
    /// Open or create a SQLite store at `path/store.db`.
    ///
    /// Acquires an exclusive file lock for write mode. Sets WAL journal mode
    /// and runs the schema migration before returning.
    pub(crate) async fn open(
        path: impl AsRef<Path>,
        dimension: usize,
        read_only: bool,
    ) -> Result<Self> {
        let dir = path.as_ref().to_path_buf();
        let db_path = dir.join("store.db");

        // Acquire write lock before creating the connection so concurrent opens
        // cannot race on schema creation.
        let lock = if !read_only {
            let lock_dir = dir.clone();
            let acquired =
                tokio::task::spawn_blocking(move || FileLock::acquire_blocking(&lock_dir))
                    .await
                    .map_err(|e| Error::store(format!("lock task join error: {e}")))?
                    .map_err(|e| Error::store(format!("failed to acquire store lock: {e}")))?;
            Some(acquired)
        } else {
            None
        };

        let db_path_clone = db_path.clone();
        let conn = tokio::task::spawn_blocking(move || -> Result<Connection> {
            std::fs::create_dir_all(db_path_clone.parent().unwrap_or(Path::new(".")))?;

            let flags = if read_only {
                rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI
            } else {
                rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
                    | rusqlite::OpenFlags::SQLITE_OPEN_CREATE
                    | rusqlite::OpenFlags::SQLITE_OPEN_URI
            };

            let conn = Connection::open_with_flags(&db_path_clone, flags)
                .map_err(|e| Error::store(format!("failed to open SQLite: {e}")))?;

            conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
                .map_err(|e| Error::store(format!("PRAGMA setup failed: {e}")))?;

            if !read_only {
                conn.execute_batch(SCHEMA_SQL)
                    .map_err(|e| Error::store(format!("schema migration failed: {e}")))?;
            }

            Ok(conn)
        })
        .await
        .map_err(|e| Error::store(format!("spawn_blocking join error: {e}")))??;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            dimension,
            read_only,
            _lock: lock,
        })
    }

    /// Run a blocking closure on the connection, off the async executor.
    async fn with_conn<T, F>(&self, f: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&Connection) -> Result<T> + Send + 'static,
    {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let guard = conn.lock().map_err(|_| Error::store("mutex poisoned"))?;
            f(&guard)
        })
        .await
        .map_err(|e| Error::store(format!("spawn_blocking join: {e}")))?
    }

    /// Run a blocking closure that needs a mutable connection (for transactions).
    async fn with_conn_mut<T, F>(&self, f: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut Connection) -> Result<T> + Send + 'static,
    {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let mut guard = conn.lock().map_err(|_| Error::store("mutex poisoned"))?;
            f(&mut guard)
        })
        .await
        .map_err(|e| Error::store(format!("spawn_blocking join: {e}")))?
    }
}

// --- VectorStore implementation ---

impl VectorStore for SqliteStore {
    fn insert_chunks(
        &self,
        chunks: Vec<HierarchicalChunk>,
    ) -> impl std::future::Future<Output = Result<()>> + Send {
        let read_only = self.read_only;
        let dimension = self.dimension;
        self.with_conn_mut(move |conn| {
            if read_only {
                return Err(Error::store(
                    "write rejected: store was opened in read-only mode",
                ));
            }
            for chunk in &chunks {
                if let Some(emb) = chunk.embedding.as_deref() {
                    if emb.len() != dimension {
                        return Err(Error::store(format!(
                            "dimension mismatch in insert_chunks: expected {dimension}, got {}",
                            emb.len()
                        )));
                    }
                }
            }
            let tx = conn
                .transaction()
                .map_err(|e| Error::store(format!("begin transaction: {e}")))?;

            {
                let mut stmt = tx
                    .prepare(
                        "INSERT OR REPLACE INTO chunks (
                            id, content, embedding, embedding_status,
                            level, parent_id, path, source_file, heading,
                            start_offset, end_offset, entry_type, visibility,
                            perspectives, relations, cluster_memberships, summarizes,
                            created_at, last_rolled,
                            access_hour, access_day, access_week, access_month, access_year, access_total,
                            expires_at, impression_hint, impression_strength
                        ) VALUES (
                            ?1, ?2, ?3, ?4,
                            ?5, ?6, ?7, ?8, ?9,
                            ?10, ?11, ?12, ?13,
                            ?14, ?15, ?16, ?17,
                            ?18, ?19,
                            ?20, ?21, ?22, ?23, ?24, ?25,
                            ?26, ?27, ?28
                        )",
                    )
                    .map_err(|e| Error::store(format!("insert_chunks prepare: {e}")))?;

                for chunk in &chunks {
                    let (embedding_blob, status) = match &chunk.embedding {
                        Some(emb) => (Some(vec_to_bytes(emb)), EMBEDDING_STATUS_EMBEDDED),
                        None => (None, EMBEDDING_STATUS_PENDING),
                    };

                    let perspectives = serde_json::to_string(&chunk.perspectives)
                        .unwrap_or_else(|_| "[]".to_string());
                    let relations = serde_json::to_string(&chunk.relations)
                        .unwrap_or_else(|_| "[]".to_string());
                    let cluster_memberships = serde_json::to_string(&chunk.cluster_memberships)
                        .unwrap_or_else(|_| "[]".to_string());
                    let summarizes = serde_json::to_string(&chunk.summarizes)
                        .unwrap_or_else(|_| "[]".to_string());

                    stmt.execute(params![
                        chunk.id,
                        chunk.content,
                        embedding_blob,
                        status,
                        chunk.level.0 as i64,
                        chunk.parent_id,
                        chunk.path,
                        chunk.source_file,
                        chunk.heading,
                        chunk.start_offset as i64,
                        chunk.end_offset as i64,
                        chunk.entry_type.to_string(),
                        chunk.visibility,
                        perspectives,
                        relations,
                        cluster_memberships,
                        summarizes,
                        chunk.access_profile.created_at,
                        chunk.access_profile.last_rolled,
                        chunk.access_profile.hour as i64,
                        chunk.access_profile.day as i64,
                        chunk.access_profile.week as i64,
                        chunk.access_profile.month as i64,
                        chunk.access_profile.year as i64,
                        chunk.access_profile.total as i64,
                        chunk.expires_at,
                        chunk.impression_hint,
                        chunk.impression_strength as f64,
                    ])
                    .map_err(|e| Error::store(format!("insert chunk {}: {e}", chunk.id)))?;
                }
            } // drop stmt before commit

            tx.commit()
                .map_err(|e| Error::store(format!("commit insert_chunks: {e}")))?;
            Ok(())
        })
    }

    fn search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        level_filter: Option<ChunkLevel>,
        perspectives: &[&str],
    ) -> impl std::future::Future<Output = Result<Vec<SearchResult>>> + Send {
        let query_vec = query_embedding.to_vec();
        let perspectives_owned: Vec<String> = perspectives.iter().map(|s| s.to_string()).collect();

        self.with_conn(move |conn| {
            let mut sql = String::from("SELECT * FROM chunks WHERE embedding_status = 'embedded'");
            let mut bound: Vec<rusqlite::types::Value> = Vec::new();

            if let Some(level) = level_filter {
                bound.push(rusqlite::types::Value::Integer(i64::from(level.0)));
                sql.push_str(&format!(" AND level = ?{}", bound.len()));
            }

            if !perspectives_owned.is_empty() {
                let clause_start = bound.len() + 1;
                for p in &perspectives_owned {
                    bound.push(rusqlite::types::Value::Text(perspective_like_pattern(p)));
                }
                let clauses: Vec<String> = (clause_start..=bound.len())
                    .map(|i| format!("perspectives LIKE ?{i} ESCAPE '\\'"))
                    .collect();
                sql.push_str(&format!(" AND ({})", clauses.join(" OR ")));
            }

            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| Error::store(format!("search prepare: {e}")))?;

            let rows: Vec<HierarchicalChunk> = stmt
                .query_map(rusqlite::params_from_iter(bound), row_to_chunk)
                .map_err(|e| Error::store(format!("search query: {e}")))?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(|e| Error::store(format!("row parse: {e}")))?;

            let mut scored: Vec<SearchResult> = rows
                .into_iter()
                .filter_map(|chunk| {
                    let score = chunk
                        .embedding
                        .as_ref()
                        .map(|emb| crate::search::cosine_similarity(&query_vec, emb))?;
                    Some(SearchResult { chunk, score })
                })
                .collect();

            crate::chunk::sort_f32_desc(&mut scored, |r| r.score);
            scored.truncate(limit);

            Ok(scored)
        })
    }

    fn get_children(
        &self,
        parent_id: &str,
    ) -> impl std::future::Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        let parent_id = parent_id.to_owned();

        self.with_conn(move |conn| {
            let mut stmt = conn
                .prepare("SELECT * FROM chunks WHERE parent_id = ?1")
                .map_err(|e| Error::store(format!("get_children prepare: {e}")))?;

            let chunks: Vec<HierarchicalChunk> = stmt
                .query_map(params![parent_id], row_to_chunk)
                .map_err(|e| Error::store(format!("get_children query: {e}")))?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(|e| Error::store(format!("row parse: {e}")))?;

            Ok(chunks)
        })
    }

    fn get_by_id(
        &self,
        id: &str,
    ) -> impl std::future::Future<Output = Result<Option<HierarchicalChunk>>> + Send {
        let id = id.to_owned();

        self.with_conn(move |conn| {
            let result = conn
                .query_row(
                    "SELECT * FROM chunks WHERE id = ?1",
                    params![id],
                    row_to_chunk,
                )
                .optional()
                .map_err(|e| Error::store(format!("get_by_id query: {e}")))?;

            Ok(result)
        })
    }

    fn get_by_id_prefix(
        &self,
        prefix: &str,
    ) -> impl std::future::Future<Output = Result<Option<HierarchicalChunk>>> + Send {
        let prefix = prefix.to_owned();

        self.with_conn(move |conn| {
            // Validate hex-only prefix before touching the DB
            if !prefix.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(Error::store(format!(
                    "invalid id prefix '{}': must contain only hex digits",
                    prefix
                )));
            }

            // Try exact match first
            let exact = conn
                .query_row(
                    "SELECT * FROM chunks WHERE id = ?1",
                    params![prefix],
                    row_to_chunk,
                )
                .optional()
                .map_err(|e| Error::store(format!("get_by_id_prefix exact: {e}")))?;

            if let Some(chunk) = exact {
                return Ok(Some(chunk));
            }

            // Fall back to prefix scan, fetch up to 2 to detect ambiguity
            let like_pattern = format!("{}%", prefix);
            let mut stmt = conn
                .prepare("SELECT * FROM chunks WHERE id LIKE ?1 LIMIT 2")
                .map_err(|e| Error::store(format!("get_by_id_prefix prepare: {e}")))?;

            let matches: Vec<HierarchicalChunk> = stmt
                .query_map(params![like_pattern], row_to_chunk)
                .map_err(|e| Error::store(format!("get_by_id_prefix query: {e}")))?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(|e| Error::store(format!("row parse: {e}")))?;

            match matches.len() {
                0 => Ok(None),
                1 => Ok(Some(matches.into_iter().next().unwrap())),
                _ => Err(Error::store(format!(
                    "ambiguous id prefix '{}': matches multiple chunks",
                    prefix
                ))),
            }
        })
    }

    fn get_by_source(
        &self,
        source_file: &str,
    ) -> impl std::future::Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        let source_file = source_file.to_owned();

        self.with_conn(move |conn| {
            let mut stmt = conn
                .prepare("SELECT * FROM chunks WHERE source_file = ?1")
                .map_err(|e| Error::store(format!("get_by_source prepare: {e}")))?;

            let chunks: Vec<HierarchicalChunk> = stmt
                .query_map(params![source_file], row_to_chunk)
                .map_err(|e| Error::store(format!("get_by_source query: {e}")))?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(|e| Error::store(format!("row parse: {e}")))?;

            Ok(chunks)
        })
    }

    fn delete_by_source(
        &self,
        source_file: &str,
    ) -> impl std::future::Future<Output = Result<usize>> + Send {
        let source_file = source_file.to_owned();
        let read_only = self.read_only;

        self.with_conn(move |conn| {
            if read_only {
                return Err(Error::store(
                    "write rejected: store was opened in read-only mode",
                ));
            }
            let changed = conn
                .execute(
                    "DELETE FROM chunks WHERE source_file = ?1",
                    params![source_file],
                )
                .map_err(|e| Error::store(format!("delete_by_source: {e}")))?;

            Ok(changed)
        })
    }

    fn stats(&self) -> impl std::future::Future<Output = Result<StoreStats>> + Send {
        self.with_conn(move |conn| {
            let mut stmt = conn
                .prepare("SELECT level, COUNT(*) FROM chunks GROUP BY level")
                .map_err(|e| Error::store(format!("stats by_level prepare: {e}")))?;

            let chunks_by_level: std::collections::HashMap<u8, usize> = stmt
                .query_map([], |row| {
                    let level: i64 = row.get(0)?;
                    let count: i64 = row.get(1)?;
                    Ok((narrow::<u8>(level, "level")?, count as usize))
                })
                .map_err(|e| Error::store(format!("stats by_level query: {e}")))?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(|e| Error::store(format!("row parse: {e}")))?
                .into_iter()
                .collect();

            let total_chunks = chunks_by_level.values().sum();

            let mut stmt = conn
                .prepare("SELECT DISTINCT source_file FROM chunks ORDER BY source_file")
                .map_err(|e| Error::store(format!("stats source_files prepare: {e}")))?;

            let source_files: Vec<String> = stmt
                .query_map([], |row| row.get::<_, String>(0))
                .map_err(|e| Error::store(format!("stats source_files query: {e}")))?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(|e| Error::store(format!("row parse: {e}")))?;

            let pending_embeddings: usize =
                conn.query_row(
                    "SELECT COUNT(*) FROM chunks WHERE embedding_status = 'pending'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .map_err(|e| Error::store(format!("stats pending: {e}")))? as usize;

            Ok(StoreStats {
                total_chunks,
                chunks_by_level,
                source_files,
                pending_embeddings,
            })
        })
    }

    fn update_access_profiles(
        &self,
        updates: Vec<(String, AccessProfile)>,
    ) -> impl std::future::Future<Output = Result<()>> + Send {
        let read_only = self.read_only;
        self.with_conn_mut(move |conn| {
            if read_only {
                return Err(Error::store(
                    "write rejected: store was opened in read-only mode",
                ));
            }
            let tx = conn
                .transaction()
                .map_err(|e| Error::store(format!("begin transaction: {e}")))?;

            {
                let mut stmt = tx
                    .prepare(
                        "UPDATE chunks SET
                            created_at = ?2,
                            last_rolled = ?3,
                            access_hour = ?4,
                            access_day = ?5,
                            access_week = ?6,
                            access_month = ?7,
                            access_year = ?8,
                            access_total = ?9
                         WHERE id = ?1",
                    )
                    .map_err(|e| Error::store(format!("update_access_profiles prepare: {e}")))?;

                for (id, profile) in &updates {
                    stmt.execute(params![
                        id,
                        profile.created_at,
                        profile.last_rolled,
                        profile.hour as i64,
                        profile.day as i64,
                        profile.week as i64,
                        profile.month as i64,
                        profile.year as i64,
                        profile.total as i64,
                    ])
                    .map_err(|e| Error::store(format!("update_access_profiles {id}: {e}")))?;
                }
            } // drop stmt before commit

            tx.commit()
                .map_err(|e| Error::store(format!("commit update_access_profiles: {e}")))?;
            Ok(())
        })
    }

    fn update_visibility(
        &self,
        chunk_id: &str,
        visibility: &str,
    ) -> impl std::future::Future<Output = Result<()>> + Send {
        let chunk_id = chunk_id.to_owned();
        let visibility = visibility.to_owned();
        let read_only = self.read_only;

        self.with_conn(move |conn| {
            if read_only {
                return Err(Error::store(
                    "write rejected: store was opened in read-only mode",
                ));
            }
            conn.execute(
                "UPDATE chunks SET visibility = ?2 WHERE id = ?1",
                params![chunk_id, visibility],
            )
            .map_err(|e| Error::store(format!("update_visibility: {e}")))?;
            Ok(())
        })
    }

    fn add_relation(
        &self,
        chunk_id: &str,
        relation: ChunkRelation,
    ) -> impl std::future::Future<Output = Result<()>> + Send {
        let chunk_id = chunk_id.to_owned();
        let read_only = self.read_only;

        self.with_conn(move |conn| {
            if read_only {
                return Err(Error::store(
                    "write rejected: store was opened in read-only mode",
                ));
            }
            let relation_json = serde_json::to_string(&relation)
                .map_err(|e| Error::store(format!("serialize relation: {e}")))?;
            let updated = conn
                .execute(
                    "UPDATE chunks SET relations = json_insert(relations, '$[#]', json(?2)) WHERE id = ?1",
                    params![chunk_id, relation_json],
                )
                .map_err(|e| Error::store(format!("add_relation: {e}")))?;
            if updated == 0 {
                return Err(Error::store(format!(
                    "add_relation: chunk '{}' not found",
                    chunk_id
                )));
            }
            Ok(())
        })
    }

    fn get_hot_chunks(
        &self,
        limit: usize,
    ) -> impl std::future::Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        self.with_conn(move |conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT * FROM chunks
                     WHERE access_total > 0
                     ORDER BY access_total DESC, access_hour DESC
                     LIMIT ?1",
                )
                .map_err(|e| Error::store(format!("get_hot_chunks prepare: {e}")))?;

            let chunks: Vec<HierarchicalChunk> = stmt
                .query_map(params![limit as i64], row_to_chunk)
                .map_err(|e| Error::store(format!("get_hot_chunks query: {e}")))?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(|e| Error::store(format!("row parse: {e}")))?;

            Ok(chunks)
        })
    }

    fn get_stale_chunks(
        &self,
        stale_seconds: i64,
        limit: usize,
    ) -> impl std::future::Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        self.with_conn(move |conn| {
            let cutoff = now_epoch_secs() - stale_seconds;
            let mut stmt = conn
                .prepare(
                    "SELECT * FROM chunks
                     WHERE last_rolled < ?1
                       AND access_hour = 0
                       AND access_day = 0
                       AND access_week = 0
                       AND (visibility = 'normal' OR visibility = 'always')
                     ORDER BY last_rolled ASC
                     LIMIT ?2",
                )
                .map_err(|e| Error::store(format!("get_stale_chunks prepare: {e}")))?;

            let chunks: Vec<HierarchicalChunk> = stmt
                .query_map(params![cutoff, limit as i64], row_to_chunk)
                .map_err(|e| Error::store(format!("get_stale_chunks query: {e}")))?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(|e| Error::store(format!("row parse: {e}")))?;

            Ok(chunks)
        })
    }

    fn search_text(
        &self,
        query: &str,
        perspectives: &[&str],
        since: Option<i64>,
        until: Option<i64>,
        limit: usize,
    ) -> impl std::future::Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        let perspectives_owned: Vec<String> = perspectives.iter().map(|s| s.to_string()).collect();
        let query_owned = query.to_string();

        self.with_conn(move |conn| {
            let mut conditions: Vec<String> = Vec::new();
            let mut bound: Vec<rusqlite::types::Value> = Vec::new();

            // Split query into words and require all of them (AND logic)
            let words: Vec<&str> = query_owned.split_whitespace().collect();
            if !words.is_empty() {
                for word in &words {
                    let escaped = word
                        .replace('\\', "\\\\")
                        .replace('%', "\\%")
                        .replace('_', "\\_");
                    bound.push(rusqlite::types::Value::Text(format!("%{escaped}%")));
                    conditions.push(format!("content LIKE ?{} ESCAPE '\\'", bound.len()));
                }
            }

            if !perspectives_owned.is_empty() {
                let clause_start = bound.len() + 1;
                for p in &perspectives_owned {
                    bound.push(rusqlite::types::Value::Text(perspective_like_pattern(p)));
                }
                let clauses: Vec<String> = (clause_start..=bound.len())
                    .map(|i| format!("perspectives LIKE ?{i} ESCAPE '\\'"))
                    .collect();
                conditions.push(format!("({})", clauses.join(" OR ")));
            }
            if let Some(ts) = since {
                bound.push(rusqlite::types::Value::Integer(ts));
                conditions.push(format!("created_at >= ?{}", bound.len()));
            }
            if let Some(ts) = until {
                bound.push(rusqlite::types::Value::Integer(ts));
                conditions.push(format!("created_at <= ?{}", bound.len()));
            }

            let where_clause = if conditions.is_empty() {
                String::new()
            } else {
                format!(" WHERE {}", conditions.join(" AND "))
            };

            bound.push(rusqlite::types::Value::Integer(limit as i64));
            let sql = format!(
                "SELECT * FROM chunks{where_clause} ORDER BY created_at DESC LIMIT ?{}",
                bound.len()
            );

            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| Error::store(format!("search_text prepare: {e}")))?;

            let chunks: Vec<HierarchicalChunk> = stmt
                .query_map(rusqlite::params_from_iter(bound), row_to_chunk)
                .map_err(|e| Error::store(format!("search_text query: {e}")))?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(|e| Error::store(format!("row parse: {e}")))?;

            Ok(chunks)
        })
    }

    fn list_entries(
        &self,
        perspectives: &[&str],
        since: Option<i64>,
        until: Option<i64>,
        limit: usize,
    ) -> impl std::future::Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        let perspectives_owned: Vec<String> = perspectives.iter().map(|s| s.to_string()).collect();

        self.with_conn(move |conn| {
            let mut conditions: Vec<String> = Vec::new();
            let mut bound: Vec<rusqlite::types::Value> = Vec::new();

            if !perspectives_owned.is_empty() {
                let clause_start = bound.len() + 1;
                for p in &perspectives_owned {
                    bound.push(rusqlite::types::Value::Text(perspective_like_pattern(p)));
                }
                let clauses: Vec<String> = (clause_start..=bound.len())
                    .map(|i| format!("perspectives LIKE ?{i} ESCAPE '\\'"))
                    .collect();
                conditions.push(format!("({})", clauses.join(" OR ")));
            }
            if let Some(ts) = since {
                bound.push(rusqlite::types::Value::Integer(ts));
                conditions.push(format!("created_at >= ?{}", bound.len()));
            }
            if let Some(ts) = until {
                bound.push(rusqlite::types::Value::Integer(ts));
                conditions.push(format!("created_at <= ?{}", bound.len()));
            }

            let where_clause = if conditions.is_empty() {
                String::new()
            } else {
                format!(" WHERE {}", conditions.join(" AND "))
            };

            bound.push(rusqlite::types::Value::Integer(limit as i64));
            let sql = format!(
                "SELECT * FROM chunks{where_clause} ORDER BY created_at DESC LIMIT ?{}",
                bound.len()
            );

            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| Error::store(format!("list_entries prepare: {e}")))?;

            let chunks: Vec<HierarchicalChunk> = stmt
                .query_map(rusqlite::params_from_iter(bound), row_to_chunk)
                .map_err(|e| Error::store(format!("list_entries query: {e}")))?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(|e| Error::store(format!("row parse: {e}")))?;

            Ok(chunks)
        })
    }

    fn get_pending_embeddings(
        &self,
        limit: usize,
    ) -> impl std::future::Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        self.with_conn(move |conn| {
            let mut stmt = conn
                .prepare("SELECT * FROM chunks WHERE embedding_status = 'pending' LIMIT ?1")
                .map_err(|e| Error::store(format!("get_pending_embeddings prepare: {e}")))?;

            let chunks: Vec<HierarchicalChunk> = stmt
                .query_map(params![limit as i64], row_to_chunk)
                .map_err(|e| Error::store(format!("get_pending_embeddings query: {e}")))?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(|e| Error::store(format!("row parse: {e}")))?;

            Ok(chunks)
        })
    }

    fn batch_update_embeddings(
        &self,
        updates: Vec<(String, Vec<f32>)>,
    ) -> impl std::future::Future<Output = Result<()>> + Send {
        let read_only = self.read_only;
        let dimension = self.dimension;
        self.with_conn_mut(move |conn| {
            if read_only {
                return Err(Error::store(
                    "write rejected: store was opened in read-only mode",
                ));
            }
            for (id, emb) in &updates {
                if emb.len() != dimension {
                    return Err(Error::store(format!(
                        "dimension mismatch in batch_update_embeddings for '{id}': expected {dimension}, got {}",
                        emb.len()
                    )));
                }
            }
            let tx = conn
                .transaction()
                .map_err(|e| Error::store(format!("begin transaction: {e}")))?;

            {
                let mut stmt = tx
                    .prepare(
                        "UPDATE chunks SET embedding = ?2, embedding_status = 'embedded' WHERE id = ?1",
                    )
                    .map_err(|e| Error::store(format!("batch_update_embeddings prepare: {e}")))?;

                for (id, embedding) in &updates {
                    let blob = vec_to_bytes(embedding);
                    stmt.execute(params![id, blob])
                        .map_err(|e| Error::store(format!("batch_update_embeddings {id}: {e}")))?;
                }
            } // drop stmt before commit

            tx.commit()
                .map_err(|e| Error::store(format!("commit batch_update_embeddings: {e}")))?;
            Ok(())
        })
    }

    fn count_pending_embeddings(&self) -> impl std::future::Future<Output = Result<usize>> + Send {
        self.with_conn(move |conn| {
            let count: usize = conn
                .query_row(
                    "SELECT COUNT(*) FROM chunks WHERE embedding_status = 'pending'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .map_err(|e| Error::store(format!("count_pending_embeddings: {e}")))?
                as usize;
            Ok(count)
        })
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access_profile::now_epoch_secs;
    #[allow(unused_imports)]
    use crate::chunk::{visibility, EntryType};
    use tempfile::TempDir;

    // --- bytes_to_vec ---

    #[test]
    fn bytes_to_vec_happy_path_round_trips_floats() {
        let input = vec![1.0f32, 2.0, 3.0];
        let blob = vec_to_bytes(&input);
        let output = bytes_to_vec(&blob).expect("valid blob must decode");
        assert_eq!(output, input);
    }

    #[test]
    fn bytes_to_vec_empty_blob_yields_empty_vec() {
        // Length 0 is a valid multiple of 4.
        let output = bytes_to_vec(&[]).expect("empty blob must succeed");
        assert!(output.is_empty());
    }

    #[test]
    fn bytes_to_vec_non_multiple_of_4_returns_error() {
        // 5 bytes: not a multiple of 4 — must fail, not silently truncate.
        let corrupt: Vec<u8> = vec![0x00, 0x01, 0x02, 0x03, 0x04];
        let result = bytes_to_vec(&corrupt);
        assert!(result.is_err(), "non-multiple-of-4 blob must return Err");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("not a multiple of 4"),
            "error must mention 'not a multiple of 4', got: {msg}"
        );
    }

    #[test]
    fn test_narrow_accepts_in_range_and_rejects_overflow() {
        assert_eq!(narrow::<u8>(7, "level").unwrap(), 7u8);
        assert_eq!(narrow::<u16>(60_000, "access_hour").unwrap(), 60_000u16);
        // Out-of-range values must error, not silently truncate.
        assert!(narrow::<u8>(300, "level").is_err());
        assert!(narrow::<u16>(100_000, "access_hour").is_err());
        assert!(narrow::<usize>(-1, "start_offset").is_err());
    }

    async fn create_test_store() -> (SqliteStore, TempDir) {
        let dir = TempDir::new().expect("tempdir");
        let store = SqliteStore::open(dir.path(), 4, false)
            .await
            .expect("open store");
        (store, dir)
    }

    fn make_chunk(content: &str, source: &str) -> HierarchicalChunk {
        HierarchicalChunk::new(
            content.to_string(),
            ChunkLevel::H1,
            None,
            "root".to_string(),
            source.to_string(),
        )
    }

    fn make_chunk_with_embedding(content: &str, source: &str, emb: Vec<f32>) -> HierarchicalChunk {
        make_chunk(content, source).with_embedding(emb)
    }

    // --- crud_roundtrip ---

    #[tokio::test]
    async fn crud_roundtrip() {
        let (store, _dir) = create_test_store().await;

        let chunk = make_chunk("Hello, SQLite!", "test.md");
        let chunk_id = chunk.id.clone();
        let short = &chunk_id[..7];

        store.insert_chunks(vec![chunk]).await.expect("insert");

        // get_by_id
        let fetched = store
            .get_by_id(&chunk_id)
            .await
            .expect("get_by_id")
            .expect("chunk present");

        assert_eq!(fetched.id, chunk_id);
        assert_eq!(fetched.content, "Hello, SQLite!");
        assert_eq!(fetched.source_file, "test.md");

        // get_by_id_prefix with short id
        let by_prefix = store
            .get_by_id_prefix(short)
            .await
            .expect("get_by_id_prefix")
            .expect("chunk via prefix");

        assert_eq!(by_prefix.id, chunk_id);

        // not found
        let missing = store.get_by_id("no-such-id").await.expect("query ok");
        assert!(missing.is_none());
    }

    // --- search_and_perspective ---

    #[tokio::test]
    async fn search_and_perspective() {
        let (store, _dir) = create_test_store().await;

        let c1 = make_chunk_with_embedding("decisions chunk", "a.md", vec![1.0, 0.0, 0.0, 0.0])
            .with_perspective("decisions");
        let c2 = make_chunk_with_embedding("learnings chunk", "b.md", vec![0.0, 1.0, 0.0, 0.0])
            .with_perspective("learnings");
        let c3 = make_chunk_with_embedding("other chunk", "c.md", vec![0.8, 0.2, 0.0, 0.0]);

        store.insert_chunks(vec![c1, c2, c3]).await.expect("insert");

        // Unfiltered search — query is close to [1, 0, 0, 0]
        let results = store
            .search(&[1.0, 0.0, 0.0, 0.0], 10, None, &[])
            .await
            .expect("search");

        assert!(!results.is_empty(), "expected results");

        // Perspective-filtered: only "decisions"
        let filtered = store
            .search(&[1.0, 0.0, 0.0, 0.0], 10, None, &["decisions"])
            .await
            .expect("search with perspective");

        assert_eq!(filtered.len(), 1);
        assert!(filtered[0]
            .chunk
            .perspectives
            .contains(&"decisions".to_string()));

        // Level filter — level 2 returns nothing (all inserted at H1 = depth 1)
        let level_filtered = store
            .search(&[1.0, 0.0, 0.0, 0.0], 10, Some(ChunkLevel::H2), &[])
            .await
            .expect("level filter search");

        assert!(level_filtered.is_empty());
    }

    // --- stats ---

    #[tokio::test]
    async fn stats() {
        let (store, _dir) = create_test_store().await;

        let c1 = make_chunk("a", "file1.md");
        let c2 = make_chunk("b", "file1.md");
        let c3 = HierarchicalChunk::new(
            "c".to_string(),
            ChunkLevel::H2,
            None,
            "root".to_string(),
            "file2.md".to_string(),
        );

        store.insert_chunks(vec![c1, c2, c3]).await.expect("insert");

        let s = store.stats().await.expect("stats");

        assert_eq!(s.total_chunks, 3);
        assert_eq!(s.pending_embeddings, 3); // none have embeddings
        assert!(s.source_files.contains(&"file1.md".to_string()));
        assert!(s.source_files.contains(&"file2.md".to_string()));
        assert_eq!(*s.chunks_by_level.get(&1).unwrap(), 2);
        assert_eq!(*s.chunks_by_level.get(&2).unwrap(), 1);
    }

    // --- access_profiles ---

    #[tokio::test]
    async fn access_profiles() {
        let (store, _dir) = create_test_store().await;

        let chunk = make_chunk("access test", "ap.md");
        let id = chunk.id.clone();
        store.insert_chunks(vec![chunk]).await.expect("insert");

        let now = now_epoch_secs();
        let profile = AccessProfile {
            created_at: now - 100,
            last_rolled: now - 50,
            hour: 3,
            day: 5,
            week: 7,
            month: 2,
            year: 1,
            total: 18,
        };

        store
            .update_access_profiles(vec![(id.clone(), profile.clone())])
            .await
            .expect("update");

        let fetched = store.get_by_id(&id).await.expect("query").expect("chunk");
        let ap = &fetched.access_profile;

        assert_eq!(ap.hour, 3);
        assert_eq!(ap.day, 5);
        assert_eq!(ap.week, 7);
        assert_eq!(ap.month, 2);
        assert_eq!(ap.year, 1);
        assert_eq!(ap.total, 18);
    }

    // --- pending_embeddings ---

    #[tokio::test]
    async fn pending_embeddings() {
        let (store, _dir) = create_test_store().await;

        // Insert two chunks without embeddings
        let c1 = make_chunk("pending a", "pa.md");
        let c2 = make_chunk("pending b", "pb.md");
        let id1 = c1.id.clone();
        let id2 = c2.id.clone();

        store.insert_chunks(vec![c1, c2]).await.expect("insert");

        let count = store.count_pending_embeddings().await.expect("count");
        assert_eq!(count, 2);

        let pending = store.get_pending_embeddings(10).await.expect("get_pending");
        assert_eq!(pending.len(), 2);

        // Batch update with real embeddings
        let updates = vec![
            (id1.clone(), vec![0.1f32, 0.2, 0.3, 0.4]),
            (id2.clone(), vec![0.5f32, 0.6, 0.7, 0.8]),
        ];
        store
            .batch_update_embeddings(updates)
            .await
            .expect("batch update");

        let count_after = store.count_pending_embeddings().await.expect("count after");
        assert_eq!(count_after, 0);

        // Should now be searchable
        let results = store
            .search(&[0.1, 0.2, 0.3, 0.4], 5, None, &[])
            .await
            .expect("search after batch update");

        assert_eq!(results.len(), 2);
        // Top result should be the one with embedding close to query
        assert_eq!(results[0].chunk.id, id1);
    }

    // --- stale_and_hot ---

    #[tokio::test]
    async fn stale_and_hot() {
        let (store, _dir) = create_test_store().await;

        let now = now_epoch_secs();

        // Stale chunk: last_rolled in the distant past, no recent accesses
        let mut stale = make_chunk("stale chunk", "stale.md");
        stale.access_profile.last_rolled = now - 10_000;
        stale.access_profile.created_at = now - 10_000;

        // Hot chunk: many accesses
        let mut hot = make_chunk("hot chunk", "hot.md");
        hot.access_profile.total = 100;
        hot.access_profile.hour = 10;

        // Recent chunk: last_rolled is recent, no accesses
        let mut recent = make_chunk("recent chunk", "recent.md");
        recent.access_profile.last_rolled = now;

        store
            .insert_chunks(vec![stale.clone(), hot.clone(), recent.clone()])
            .await
            .expect("insert");

        // get_hot_chunks
        let hot_results = store.get_hot_chunks(5).await.expect("get_hot");
        assert!(!hot_results.is_empty());
        assert_eq!(hot_results[0].id, hot.id);

        // get_stale_chunks: stale_seconds=1000, so cutoff = now-1000
        // Only 'stale' has last_rolled < now-1000 and no recent hour/day/week access
        let stale_results = store.get_stale_chunks(1000, 5).await.expect("get_stale");
        assert!(!stale_results.is_empty());
        assert!(stale_results.iter().any(|c| c.id == stale.id));
        // Hot and recent should not appear (hot has access_total but no hour/day/week
        // requirement; let's verify stale chunk is there)
        assert!(!stale_results.iter().any(|c| c.id == recent.id));
    }

    // --- delete_by_source ---

    #[tokio::test]
    async fn delete_by_source() {
        let (store, _dir) = create_test_store().await;

        let c1 = make_chunk("del a", "delete-me.md");
        let c2 = make_chunk("del b", "delete-me.md");
        let c3 = make_chunk("keep", "keep.md");

        store.insert_chunks(vec![c1, c2, c3]).await.expect("insert");

        let deleted = store
            .delete_by_source("delete-me.md")
            .await
            .expect("delete");
        assert_eq!(deleted, 2);

        let s = store.stats().await.expect("stats");
        assert_eq!(s.total_chunks, 1);
        assert!(s.source_files.contains(&"keep.md".to_string()));
        assert!(!s.source_files.contains(&"delete-me.md".to_string()));
    }

    // --- insert_chunks: batch, pending status ---

    #[tokio::test]
    async fn insert_batch_without_embeddings_marks_pending() {
        let (store, _dir) = create_test_store().await;

        let chunks: Vec<HierarchicalChunk> = (0..5)
            .map(|i| make_chunk(&format!("content {i}"), "batch.md"))
            .collect();

        store.insert_chunks(chunks).await.expect("insert batch");

        let count = store.count_pending_embeddings().await.expect("count");
        assert_eq!(count, 5);

        let s = store.stats().await.expect("stats");
        assert_eq!(s.total_chunks, 5);
        assert_eq!(s.pending_embeddings, 5);
    }

    #[tokio::test]
    async fn insert_chunk_with_embedding_is_not_pending() {
        let (store, _dir) = create_test_store().await;

        let chunk = make_chunk_with_embedding("embedded", "emb.md", vec![1.0, 0.0, 0.0, 0.0]);
        store.insert_chunks(vec![chunk]).await.expect("insert");

        let count = store.count_pending_embeddings().await.expect("count");
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn insert_empty_batch_is_noop() {
        let (store, _dir) = create_test_store().await;

        store.insert_chunks(vec![]).await.expect("insert empty");

        let s = store.stats().await.expect("stats");
        assert_eq!(s.total_chunks, 0);
    }

    #[tokio::test]
    async fn duplicate_insert_replaces_chunk() {
        let (store, _dir) = create_test_store().await;

        let mut chunk = make_chunk("original", "dup.md");
        let id = chunk.id.clone();
        store
            .insert_chunks(vec![chunk.clone()])
            .await
            .expect("insert 1");

        // Replace with updated content (same id since INSERT OR REPLACE)
        chunk.visibility = "always".to_string();
        store.insert_chunks(vec![chunk]).await.expect("insert 2");

        let s = store.stats().await.expect("stats");
        assert_eq!(s.total_chunks, 1, "no duplicate rows");

        let fetched = store.get_by_id(&id).await.expect("query").expect("chunk");
        assert_eq!(fetched.visibility, "always");
    }

    // --- search: empty store, top-k, cosine ranking ---

    #[tokio::test]
    async fn search_empty_store_returns_empty() {
        let (store, _dir) = create_test_store().await;

        let results = store
            .search(&[1.0, 0.0, 0.0, 0.0], 10, None, &[])
            .await
            .expect("search");

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn search_pending_chunks_excluded() {
        let (store, _dir) = create_test_store().await;

        // Chunk without embedding stays pending and must not appear in search
        let chunk = make_chunk("no embedding", "ne.md");
        store.insert_chunks(vec![chunk]).await.expect("insert");

        let results = store
            .search(&[1.0, 0.0, 0.0, 0.0], 10, None, &[])
            .await
            .expect("search");

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn search_top_k_limits_results() {
        let (store, _dir) = create_test_store().await;

        let chunks: Vec<HierarchicalChunk> = (0..10)
            .map(|i| {
                make_chunk_with_embedding(
                    &format!("chunk {i}"),
                    "topk.md",
                    vec![i as f32, 0.0, 0.0, 0.0],
                )
            })
            .collect();

        store.insert_chunks(chunks).await.expect("insert");

        let results = store
            .search(&[1.0, 0.0, 0.0, 0.0], 3, None, &[])
            .await
            .expect("search");

        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn search_results_ordered_by_cosine_similarity() {
        let (store, _dir) = create_test_store().await;

        // c_close is almost identical to query; c_far is orthogonal
        let c_close = make_chunk_with_embedding("close", "sim.md", vec![1.0, 0.0, 0.0, 0.0]);
        let c_far = make_chunk_with_embedding("far", "sim.md", vec![0.0, 1.0, 0.0, 0.0]);

        store
            .insert_chunks(vec![c_close.clone(), c_far.clone()])
            .await
            .expect("insert");

        let results = store
            .search(&[1.0, 0.0, 0.0, 0.0], 10, None, &[])
            .await
            .expect("search");

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk.id, c_close.id, "closest first");
        assert!(results[0].score >= results[1].score, "scores descending");
    }

    // --- get_children ---

    #[tokio::test]
    async fn get_children_returns_direct_children() {
        let (store, _dir) = create_test_store().await;

        let parent = make_chunk("parent", "tree.md");
        let parent_id = parent.id.clone();

        let mut child1 = make_chunk("child 1", "tree.md");
        child1.parent_id = Some(parent_id.clone());

        let mut child2 = make_chunk("child 2", "tree.md");
        child2.parent_id = Some(parent_id.clone());

        let unrelated = make_chunk("unrelated", "tree.md");

        store
            .insert_chunks(vec![parent, child1.clone(), child2.clone(), unrelated])
            .await
            .expect("insert");

        let children = store.get_children(&parent_id).await.expect("get_children");

        assert_eq!(children.len(), 2);
        let ids: Vec<&str> = children.iter().map(|c| c.id.as_str()).collect();
        assert!(ids.contains(&child1.id.as_str()));
        assert!(ids.contains(&child2.id.as_str()));
    }

    #[tokio::test]
    async fn get_children_leaf_node_returns_empty() {
        let (store, _dir) = create_test_store().await;

        let leaf = make_chunk("leaf node", "leaf.md");
        let leaf_id = leaf.id.clone();
        store.insert_chunks(vec![leaf]).await.expect("insert");

        let children = store.get_children(&leaf_id).await.expect("get_children");
        assert!(children.is_empty());
    }

    #[tokio::test]
    async fn get_children_unknown_parent_returns_empty() {
        let (store, _dir) = create_test_store().await;

        let children = store
            .get_children("nonexistent-parent")
            .await
            .expect("get_children");

        assert!(children.is_empty());
    }

    // --- get_by_id_prefix: ambiguous, non-hex ---

    #[tokio::test]
    async fn get_by_id_prefix_ambiguous_returns_error() {
        let (store, _dir) = create_test_store().await;

        // Force two chunks that share the same 4-char prefix by using fixed IDs
        let mut c1 = make_chunk("alpha", "ambig.md");
        c1.id = "aaaa1111bbbbccccddddeeeeffffaaaabbbbccccddddeeeeffffaaaaaaaaaaaa".to_string();

        let mut c2 = make_chunk("beta", "ambig.md");
        c2.id = "aaaa2222bbbbccccddddeeeeffffaaaabbbbccccddddeeeeffffaaaaaaaaaaaa".to_string();

        store.insert_chunks(vec![c1, c2]).await.expect("insert");

        let result = store.get_by_id_prefix("aaaa").await;
        assert!(result.is_err(), "ambiguous prefix must return error");
    }

    #[tokio::test]
    async fn get_by_id_prefix_non_hex_returns_error() {
        let (store, _dir) = create_test_store().await;

        let result = store.get_by_id_prefix("xyz!").await;
        assert!(result.is_err(), "non-hex prefix must return error");
    }

    #[tokio::test]
    async fn get_by_id_prefix_missing_returns_none() {
        let (store, _dir) = create_test_store().await;

        let result = store.get_by_id_prefix("deadbeef").await.expect("query ok");

        assert!(result.is_none());
    }

    // --- get_by_source ---

    #[tokio::test]
    async fn get_by_source_returns_all_matching_chunks() {
        let (store, _dir) = create_test_store().await;

        let c1 = make_chunk("source a1", "src.md");
        let c2 = make_chunk("source a2", "src.md");
        let c3 = make_chunk("other source", "other.md");

        store.insert_chunks(vec![c1, c2, c3]).await.expect("insert");

        let chunks = store.get_by_source("src.md").await.expect("get_by_source");
        assert_eq!(chunks.len(), 2);
        assert!(chunks.iter().all(|c| c.source_file == "src.md"));
    }

    #[tokio::test]
    async fn get_by_source_unknown_file_returns_empty() {
        let (store, _dir) = create_test_store().await;

        let chunks = store
            .get_by_source("does-not-exist.md")
            .await
            .expect("get_by_source");

        assert!(chunks.is_empty());
    }

    // --- update_visibility ---

    #[tokio::test]
    async fn update_visibility_persists_new_value() {
        let (store, _dir) = create_test_store().await;

        let chunk = make_chunk("visibility test", "vis.md");
        let id = chunk.id.clone();
        store.insert_chunks(vec![chunk]).await.expect("insert");

        store
            .update_visibility(&id, "deep_only")
            .await
            .expect("update_visibility");

        let fetched = store.get_by_id(&id).await.expect("query").expect("chunk");
        assert_eq!(fetched.visibility, "deep_only");
    }

    #[tokio::test]
    async fn update_visibility_custom_value_works() {
        let (store, _dir) = create_test_store().await;

        let chunk = make_chunk("custom vis", "cvis.md");
        let id = chunk.id.clone();
        store.insert_chunks(vec![chunk]).await.expect("insert");

        store
            .update_visibility(&id, "archived")
            .await
            .expect("update_visibility");

        let fetched = store.get_by_id(&id).await.expect("query").expect("chunk");
        assert_eq!(fetched.visibility, "archived");
    }

    // --- add_relation ---

    #[tokio::test]
    async fn add_relation_appends_to_chunk() {
        let (store, _dir) = create_test_store().await;

        let chunk = make_chunk("relation source", "rel.md");
        let id = chunk.id.clone();
        store.insert_chunks(vec![chunk]).await.expect("insert");

        let rel = ChunkRelation::superseded_by("target-id-001");
        store.add_relation(&id, rel).await.expect("add_relation");

        let fetched = store.get_by_id(&id).await.expect("query").expect("chunk");
        assert_eq!(fetched.relations.len(), 1);
        assert_eq!(fetched.relations[0].kind, "superseded_by");
        assert_eq!(fetched.relations[0].target_id, "target-id-001");
    }

    #[tokio::test]
    async fn add_relation_accumulates_multiple_relations() {
        let (store, _dir) = create_test_store().await;

        let chunk = make_chunk("multi-rel", "mrel.md");
        let id = chunk.id.clone();
        store.insert_chunks(vec![chunk]).await.expect("insert");

        store
            .add_relation(&id, ChunkRelation::related_to("a"))
            .await
            .expect("rel 1");
        store
            .add_relation(&id, ChunkRelation::related_to("b"))
            .await
            .expect("rel 2");
        store
            .add_relation(&id, ChunkRelation::superseded_by("c"))
            .await
            .expect("rel 3");

        let fetched = store.get_by_id(&id).await.expect("query").expect("chunk");
        assert_eq!(fetched.relations.len(), 3);
    }

    #[tokio::test]
    async fn add_relation_missing_chunk_returns_error() {
        let (store, _dir) = create_test_store().await;

        let result = store
            .add_relation("nonexistent-id", ChunkRelation::related_to("x"))
            .await;

        assert!(result.is_err(), "missing chunk must error");
    }

    // --- list_entries ---

    #[tokio::test]
    async fn list_entries_returns_all_chunks_ordered_by_created_at_desc() {
        let (store, _dir) = create_test_store().await;

        let now = now_epoch_secs();

        let mut old = make_chunk("old entry", "list.md");
        old.access_profile.created_at = now - 1000;
        old.access_profile.last_rolled = now - 1000;

        let mut mid = make_chunk("mid entry", "list.md");
        mid.access_profile.created_at = now - 500;
        mid.access_profile.last_rolled = now - 500;

        let mut recent = make_chunk("recent entry", "list.md");
        recent.access_profile.created_at = now;
        recent.access_profile.last_rolled = now;

        store
            .insert_chunks(vec![old.clone(), mid.clone(), recent.clone()])
            .await
            .expect("insert");

        let entries = store
            .list_entries(&[], None, None, 100)
            .await
            .expect("list_entries");

        assert_eq!(entries.len(), 3);
        // Most recent first
        assert_eq!(entries[0].id, recent.id);
        assert_eq!(entries[2].id, old.id);
    }

    #[tokio::test]
    async fn list_entries_limit_respected() {
        let (store, _dir) = create_test_store().await;

        let chunks: Vec<HierarchicalChunk> = (0..10)
            .map(|i| make_chunk(&format!("entry {i}"), "lim.md"))
            .collect();

        store.insert_chunks(chunks).await.expect("insert");

        let entries = store
            .list_entries(&[], None, None, 4)
            .await
            .expect("list_entries");

        assert_eq!(entries.len(), 4);
    }

    #[tokio::test]
    async fn list_entries_perspective_filter() {
        let (store, _dir) = create_test_store().await;

        let c_dec = make_chunk("a decision", "lp.md").with_perspective("decisions");
        let c_learn = make_chunk("a learning", "lp.md").with_perspective("learnings");
        let c_none = make_chunk("no perspective", "lp.md");

        store
            .insert_chunks(vec![c_dec.clone(), c_learn, c_none])
            .await
            .expect("insert");

        let entries = store
            .list_entries(&["decisions"], None, None, 100)
            .await
            .expect("list_entries");

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, c_dec.id);
    }

    #[tokio::test]
    async fn list_entries_since_filter() {
        let (store, _dir) = create_test_store().await;

        let now = now_epoch_secs();

        let mut before = make_chunk("before", "ts.md");
        before.access_profile.created_at = now - 2000;
        before.access_profile.last_rolled = now - 2000;

        let mut after = make_chunk("after", "ts.md");
        after.access_profile.created_at = now - 100;
        after.access_profile.last_rolled = now - 100;

        store
            .insert_chunks(vec![before, after.clone()])
            .await
            .expect("insert");

        let entries = store
            .list_entries(&[], Some(now - 500), None, 100)
            .await
            .expect("list_entries since");

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, after.id);
    }

    #[tokio::test]
    async fn list_entries_until_filter() {
        let (store, _dir) = create_test_store().await;

        let now = now_epoch_secs();

        let mut before = make_chunk("before", "until.md");
        before.access_profile.created_at = now - 2000;
        before.access_profile.last_rolled = now - 2000;

        let mut after = make_chunk("after", "until.md");
        after.access_profile.created_at = now - 100;
        after.access_profile.last_rolled = now - 100;

        store
            .insert_chunks(vec![before.clone(), after])
            .await
            .expect("insert");

        let entries = store
            .list_entries(&[], None, Some(now - 500), 100)
            .await
            .expect("list_entries until");

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, before.id);
    }

    /// Verify that `perspective_like_pattern` escapes `%` so that a perspective
    /// like `"te%st"` does NOT act as a SQL wildcard and match everything.
    ///
    /// Two chunks are inserted:
    ///   - `"test"` perspective  — the naive unescaped wildcard `%"te%st"%` would match this
    ///   - `"te%st"` perspective — the only chunk that should match the literal query
    ///
    /// Querying list_entries with `"te%st"` must return exactly 1 result (the
    /// literal match), not 2 (which would indicate the `%` was treated as a wildcard).
    #[tokio::test]
    async fn perspective_like_percent_not_treated_as_wildcard() {
        let (store, _dir) = create_test_store().await;

        let c_plain = make_chunk("plain test perspective", "pct.md").with_perspective("test");
        let c_literal =
            make_chunk("literal percent perspective", "pct.md").with_perspective("te%st");

        store
            .insert_chunks(vec![c_plain, c_literal.clone()])
            .await
            .expect("insert");

        // Query using the literal perspective "te%st". If `%` is not escaped in
        // the LIKE pattern, both chunks would match (naive pattern `%"te%st"%`
        // matches `["test"]` because `%` acts as wildcard). With correct escaping
        // only the chunk whose perspective IS literally "te%st" must be returned.
        let entries = store
            .list_entries(&["te%st"], None, None, 100)
            .await
            .expect("list_entries");

        assert_eq!(
            entries.len(),
            1,
            "only the literal 'te%st' perspective chunk must match; \
             got {} (unescaped % would match both chunks)",
            entries.len()
        );
        assert_eq!(entries[0].id, c_literal.id);
    }

    #[tokio::test]
    async fn list_entries_empty_store_returns_empty() {
        let (store, _dir) = create_test_store().await;

        let entries = store
            .list_entries(&[], None, None, 100)
            .await
            .expect("list_entries");

        assert!(entries.is_empty());
    }

    // --- insert: round-trips metadata fields ---

    #[tokio::test]
    async fn insert_preserves_all_metadata_fields() {
        let (store, _dir) = create_test_store().await;

        let mut chunk =
            make_chunk_with_embedding("rich chunk", "meta.md", vec![0.1, 0.2, 0.3, 0.4]);
        chunk.heading = Some("Rich Heading".to_string());
        chunk.start_offset = 42;
        chunk.end_offset = 99;
        chunk.entry_type = EntryType::Meta;
        chunk.visibility = "always".to_string();
        chunk.perspectives = vec!["decisions".to_string(), "learnings".to_string()];
        chunk.summarizes = vec!["abc".to_string()];
        chunk.impression_hint = Some("confident".to_string());
        chunk.impression_strength = 0.75;
        chunk.expires_at = Some(9_999_999);

        let id = chunk.id.clone();
        store.insert_chunks(vec![chunk]).await.expect("insert");

        let fetched = store.get_by_id(&id).await.expect("query").expect("chunk");

        assert_eq!(fetched.heading, Some("Rich Heading".to_string()));
        assert_eq!(fetched.start_offset, 42);
        assert_eq!(fetched.end_offset, 99);
        assert_eq!(fetched.entry_type, EntryType::Meta);
        assert_eq!(fetched.visibility, "always");
        assert_eq!(fetched.perspectives, vec!["decisions", "learnings"]);
        assert_eq!(fetched.summarizes, vec!["abc"]);
        assert_eq!(fetched.impression_hint, Some("confident".to_string()));
        assert!((fetched.impression_strength - 0.75_f32).abs() < 1e-5);
        assert_eq!(fetched.expires_at, Some(9_999_999));
    }

    // --- get_hot_chunks: respects ordering and limit ---

    #[tokio::test]
    async fn get_hot_chunks_empty_store_returns_empty() {
        let (store, _dir) = create_test_store().await;

        let hot = store.get_hot_chunks(5).await.expect("get_hot");
        assert!(hot.is_empty());
    }

    #[tokio::test]
    async fn get_hot_chunks_limit_respected() {
        let (store, _dir) = create_test_store().await;

        let chunks: Vec<HierarchicalChunk> = (1..=5)
            .map(|i| {
                let mut c = make_chunk(&format!("hot {i}"), "hot_lim.md");
                c.access_profile.total = i * 10;
                c.access_profile.hour = i as u16;
                c
            })
            .collect();

        store.insert_chunks(chunks).await.expect("insert");

        let hot = store.get_hot_chunks(2).await.expect("get_hot");
        assert_eq!(hot.len(), 2);
        // Highest total first
        assert!(
            hot[0].access_profile.total >= hot[1].access_profile.total,
            "ordered descending"
        );
    }

    // --- get_stale_chunks: respects visibility filter ---

    #[tokio::test]
    async fn get_stale_chunks_excludes_deep_only_visibility() {
        let (store, _dir) = create_test_store().await;

        let now = now_epoch_secs();

        let mut stale_normal = make_chunk("stale normal", "sn.md");
        stale_normal.access_profile.last_rolled = now - 5000;
        stale_normal.access_profile.created_at = now - 5000;
        // default visibility is "normal"

        let mut stale_deep = make_chunk("stale deep", "sd.md");
        stale_deep.access_profile.last_rolled = now - 5000;
        stale_deep.access_profile.created_at = now - 5000;
        stale_deep.visibility = "deep_only".to_string();

        store
            .insert_chunks(vec![stale_normal.clone(), stale_deep.clone()])
            .await
            .expect("insert");

        let stale = store.get_stale_chunks(1000, 10).await.expect("get_stale");

        let ids: Vec<&str> = stale.iter().map(|c| c.id.as_str()).collect();
        assert!(
            ids.contains(&stale_normal.id.as_str()),
            "normal should appear"
        );
        assert!(
            !ids.contains(&stale_deep.id.as_str()),
            "deep_only must not appear"
        );
    }

    // --- stats: empty store ---

    #[tokio::test]
    async fn stats_empty_store() {
        let (store, _dir) = create_test_store().await;

        let s = store.stats().await.expect("stats");
        assert_eq!(s.total_chunks, 0);
        assert_eq!(s.pending_embeddings, 0);
        assert!(s.source_files.is_empty());
        assert!(s.chunks_by_level.is_empty());
    }

    // --- zero-vector edge case ---

    #[tokio::test]
    async fn search_zero_vector_does_not_panic() {
        let (store, _dir) = create_test_store().await;

        let chunk = make_chunk_with_embedding("zero vec", "zv.md", vec![0.0, 0.0, 0.0, 0.0]);
        store.insert_chunks(vec![chunk]).await.expect("insert");

        // Searching with a zero query should not panic (cosine of zero vector is undefined)
        let _results = store
            .search(&[0.0, 0.0, 0.0, 0.0], 5, None, &[])
            .await
            .expect("search with zero vector");
    }

    // --- SQL injection resistance ---

    /// A crafted perspective containing SQL metacharacters must not allow the
    /// attacker to break out of the LIKE pattern and execute arbitrary SQL.
    ///
    /// Before the fix, the clause was built by string interpolation, so a value
    /// like `x%" OR 1=1--` would expand to:
    ///
    ///   perspectives LIKE '%"x%" OR 1=1--"%' ESCAPE '\'
    ///
    /// causing the LIKE to short-circuit and every row to match.
    ///
    /// With parameterized queries the entire pattern is a single bound value;
    /// the SQL parser never sees the user string, so injection is impossible.
    #[tokio::test]
    async fn search_perspective_sql_injection_returns_no_false_positives() {
        let (store, _dir) = create_test_store().await;

        // Insert a chunk that has a legitimate "safe" perspective.
        let safe_chunk =
            make_chunk_with_embedding("safe chunk", "safe.md", vec![1.0, 0.0, 0.0, 0.0])
                .with_perspective("safe");

        store.insert_chunks(vec![safe_chunk]).await.expect("insert");

        // Crafted payloads that would previously break out of the LIKE context.
        let injection_attempts = [
            r#"x%" OR 1=1--"#,
            r#"' OR '1'='1"#,
            r#"safe"%' OR '1'='1' --"#,
            r#"%" OR embedding_status='embedded' --"#,
        ];

        for payload in &injection_attempts {
            let results = store
                .search(&[1.0, 0.0, 0.0, 0.0], 100, None, &[payload])
                .await
                .unwrap_or_else(|_| vec![]);

            assert!(
                results.is_empty(),
                "injection payload '{payload}' leaked rows through search perspective filter"
            );
        }
    }

    #[tokio::test]
    async fn list_entries_perspective_sql_injection_returns_no_false_positives() {
        let (store, _dir) = create_test_store().await;

        let safe_chunk = make_chunk("safe entry", "safe.md").with_perspective("safe");
        store.insert_chunks(vec![safe_chunk]).await.expect("insert");

        let injection_attempts = [
            r#"x%" OR 1=1--"#,
            r#"' OR '1'='1"#,
            r#"safe"%' OR '1'='1' --"#,
        ];

        for payload in &injection_attempts {
            let entries = store
                .list_entries(&[payload], None, None, 100)
                .await
                .unwrap_or_else(|_| vec![]);

            assert!(
                entries.is_empty(),
                "injection payload '{payload}' leaked rows through list_entries perspective filter"
            );
        }
    }

    /// LIKE wildcards in perspective names must be treated as literals, not
    /// as pattern characters, so they cannot be used to match unintended rows.
    #[tokio::test]
    async fn search_perspective_like_wildcards_treated_as_literals() {
        let (store, _dir) = create_test_store().await;

        let chunk = make_chunk_with_embedding("wildcard chunk", "wc.md", vec![1.0, 0.0, 0.0, 0.0])
            .with_perspective("decisions");
        store.insert_chunks(vec![chunk]).await.expect("insert");

        // A LIKE wildcard "%" should NOT match "decisions".
        let results = store
            .search(&[1.0, 0.0, 0.0, 0.0], 100, None, &["%"])
            .await
            .expect("search");

        assert!(
            results.is_empty(),
            "bare '%' perspective should not match any stored chunk"
        );
    }

    // --- dimension enforcement ---

    #[tokio::test]
    async fn insert_correct_dimension_succeeds() {
        // dimension=4; a 4-element embedding must insert without error.
        let (store, _dir) = create_test_store().await;
        let chunk = make_chunk_with_embedding("good dim", "dim.md", vec![0.1, 0.2, 0.3, 0.4]);
        store
            .insert_chunks(vec![chunk])
            .await
            .expect("correct dimension should succeed");
    }

    #[tokio::test]
    async fn insert_wrong_dimension_returns_error() {
        // dimension=4; a 3-element embedding must be rejected.
        let (store, _dir) = create_test_store().await;
        let chunk = make_chunk_with_embedding("bad dim", "dim.md", vec![0.1, 0.2, 0.3]);
        let result = store.insert_chunks(vec![chunk]).await;
        assert!(result.is_err(), "wrong-dimension insert must error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("dimension mismatch"),
            "error message should mention 'dimension mismatch', got: {msg}"
        );
    }

    #[tokio::test]
    async fn insert_oversized_dimension_returns_error() {
        // dimension=4; a 5-element embedding must be rejected.
        let (store, _dir) = create_test_store().await;
        let chunk = make_chunk_with_embedding("over dim", "dim.md", vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let result = store.insert_chunks(vec![chunk]).await;
        assert!(result.is_err(), "oversized-dimension insert must error");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("dimension mismatch"), "got: {msg}");
    }

    #[tokio::test]
    async fn insert_pending_chunk_without_embedding_skips_dimension_check() {
        // A chunk with no embedding (pending) must always be accepted regardless
        // of the store's configured dimension.
        let (store, _dir) = create_test_store().await;
        let chunk = make_chunk("no embedding yet", "pending.md"); // embedding = None
        store
            .insert_chunks(vec![chunk])
            .await
            .expect("pending chunk with no embedding should succeed");
    }

    #[tokio::test]
    async fn batch_update_embeddings_wrong_dimension_returns_error() {
        // Insert a pending chunk then try to update it with the wrong dimension.
        let (store, _dir) = create_test_store().await;
        let chunk = make_chunk("pending", "bdim.md");
        let id = chunk.id.clone();
        store.insert_chunks(vec![chunk]).await.expect("insert");

        // 3-element vector into a dimension=4 store must fail.
        let result = store
            .batch_update_embeddings(vec![(id, vec![0.1, 0.2, 0.3])])
            .await;
        assert!(result.is_err(), "wrong-dimension batch update must error");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("dimension mismatch"), "got: {msg}");
    }

    #[tokio::test]
    async fn batch_update_embeddings_correct_dimension_succeeds() {
        let (store, _dir) = create_test_store().await;
        let chunk = make_chunk("pending", "bdim_ok.md");
        let id = chunk.id.clone();
        store.insert_chunks(vec![chunk]).await.expect("insert");

        store
            .batch_update_embeddings(vec![(id, vec![0.1, 0.2, 0.3, 0.4])])
            .await
            .expect("correct dimension batch update should succeed");
    }

    // --- read_only enforcement ---

    /// Open an existing DB in read-only mode. The DB must already be populated
    /// (schema created) by a prior writable open, otherwise the read-only open
    /// would fail at the SQLite level.
    async fn create_read_only_store(dir: &TempDir) -> SqliteStore {
        // First open: create schema and insert a seed chunk.
        {
            let rw = SqliteStore::open(dir.path(), 4, false)
                .await
                .expect("open rw");
            let seed = make_chunk("seed", "seed.md");
            rw.insert_chunks(vec![seed]).await.expect("seed insert");
        }
        // Second open: read-only.
        SqliteStore::open(dir.path(), 4, true)
            .await
            .expect("open read-only")
    }

    #[tokio::test]
    async fn read_only_store_allows_reads() {
        let dir = TempDir::new().expect("tempdir");
        let store = create_read_only_store(&dir).await;

        // stats() is a read — must succeed.
        let s = store.stats().await.expect("stats on read-only store");
        assert_eq!(s.total_chunks, 1);
    }

    #[tokio::test]
    async fn read_only_store_rejects_insert_chunks() {
        let dir = TempDir::new().expect("tempdir");
        let store = create_read_only_store(&dir).await;

        let chunk = make_chunk("blocked", "ro.md");
        let result = store.insert_chunks(vec![chunk]).await;
        assert!(result.is_err(), "insert on read-only store must error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("read-only"),
            "error should mention read-only, got: {msg}"
        );
    }

    #[tokio::test]
    async fn read_only_store_rejects_delete_by_source() {
        let dir = TempDir::new().expect("tempdir");
        let store = create_read_only_store(&dir).await;

        let result = store.delete_by_source("seed.md").await;
        assert!(result.is_err(), "delete on read-only store must error");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("read-only"), "got: {msg}");
    }

    #[tokio::test]
    async fn read_only_store_rejects_update_visibility() {
        let dir = TempDir::new().expect("tempdir");
        let store = create_read_only_store(&dir).await;

        // We don't need a valid chunk id — the read-only check fires first.
        let result = store.update_visibility("any-id", "archived").await;
        assert!(
            result.is_err(),
            "update_visibility on read-only store must error"
        );
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("read-only"), "got: {msg}");
    }

    #[tokio::test]
    async fn read_only_store_rejects_update_access_profiles() {
        let dir = TempDir::new().expect("tempdir");
        let store = create_read_only_store(&dir).await;

        let now = now_epoch_secs();
        let profile = AccessProfile {
            created_at: now,
            last_rolled: now,
            hour: 0,
            day: 0,
            week: 0,
            month: 0,
            year: 0,
            total: 0,
        };
        let result = store
            .update_access_profiles(vec![("any-id".to_string(), profile)])
            .await;
        assert!(
            result.is_err(),
            "update_access_profiles on read-only store must error"
        );
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("read-only"), "got: {msg}");
    }

    #[tokio::test]
    async fn read_only_store_rejects_add_relation() {
        let dir = TempDir::new().expect("tempdir");
        let store = create_read_only_store(&dir).await;

        let result = store
            .add_relation("any-id", ChunkRelation::related_to("other"))
            .await;
        assert!(
            result.is_err(),
            "add_relation on read-only store must error"
        );
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("read-only"), "got: {msg}");
    }

    #[tokio::test]
    async fn read_only_store_rejects_batch_update_embeddings() {
        let dir = TempDir::new().expect("tempdir");
        let store = create_read_only_store(&dir).await;

        let result = store
            .batch_update_embeddings(vec![("any-id".to_string(), vec![0.1, 0.2, 0.3, 0.4])])
            .await;
        assert!(
            result.is_err(),
            "batch_update_embeddings on read-only store must error"
        );
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("read-only"), "got: {msg}");
    }

    // --- open: error paths ---

    /// Writing random bytes to `store.db` before calling `open` must cause
    /// `open` to return `Err` with a non-empty message.  The file name is the
    /// same literal used by the production code (`store.db`).
    ///
    /// SQLite detects the corruption either at `Connection::open_with_flags`
    /// (header magic mismatch) or at the first `execute_batch` call (WAL
    /// PRAGMA or schema migration).  Either way the result must be `Err`, not a
    /// panic or a silent success.
    #[tokio::test]
    async fn open_corrupt_db_returns_error() {
        let dir = TempDir::new().expect("tempdir");

        // Plant garbage bytes at the exact path open() will use.
        let db_path = dir.path().join("store.db");
        std::fs::write(
            &db_path,
            b"THIS IS NOT A VALID SQLITE DATABASE FILE\xFF\xFE\x00",
        )
        .expect("write corrupt file");

        let result = SqliteStore::open(dir.path(), 4, false).await;

        assert!(result.is_err(), "opening a corrupt DB must return Err");
        // Use .err().unwrap() to avoid the T: Debug bound on unwrap_err().
        let msg = result.err().unwrap().to_string();
        assert!(
            !msg.is_empty(),
            "error message must be non-empty on corrupt DB"
        );
    }

    /// Opening in read-only mode when `store.db` does not yet exist must return
    /// `Err`.  This exercises the `SQLITE_OPEN_READ_ONLY` flag path in `open`,
    /// which differs from the corrupt-DB path: here the file is simply absent,
    /// so SQLite refuses to create it.
    #[tokio::test]
    async fn open_readonly_nonexistent_db_returns_error() {
        let dir = TempDir::new().expect("tempdir");
        // Do NOT create store.db — the directory exists but the file does not.

        let result = SqliteStore::open(dir.path(), 4, true).await;

        assert!(
            result.is_err(),
            "opening a non-existent DB in read-only mode must return Err"
        );
        // Use .err().unwrap() to avoid the T: Debug bound on unwrap_err().
        let msg = result.err().unwrap().to_string();
        assert!(
            !msg.is_empty(),
            "error message must be non-empty when DB file is absent"
        );
    }

    // --- search_text ---

    /// Happy path: a single inserted chunk is returned when its content
    /// contains the queried word.
    #[tokio::test]
    async fn search_text_happy_path_returns_matching_chunk() {
        let (store, _dir) = create_test_store().await;

        let hit = make_chunk("the quick brown fox", "st.md");
        let miss = make_chunk("pack my box with five dozen liquor jugs", "st.md");
        let hit_id = hit.id.clone();
        store.insert_chunks(vec![hit, miss]).await.expect("insert");

        let results = store
            .search_text("fox", &[], None, None, 10)
            .await
            .expect("search_text");

        assert_eq!(results.len(), 1, "only the matching chunk is returned");
        assert_eq!(results[0].id, hit_id);
    }

    /// Multi-word AND logic: only chunks whose content contains BOTH words are
    /// returned; a chunk with only one of the words is excluded.
    #[tokio::test]
    async fn search_text_multi_word_requires_all_words() {
        let (store, _dir) = create_test_store().await;

        // Contains both "apple" and "orange"
        let both = make_chunk("apple and orange together", "st.md");
        // Contains only "apple"
        let only_apple = make_chunk("apple alone here", "st.md");
        // Contains only "orange"
        let only_orange = make_chunk("orange on its own", "st.md");

        let both_id = both.id.clone();
        store
            .insert_chunks(vec![both, only_apple, only_orange])
            .await
            .expect("insert");

        let results = store
            .search_text("apple orange", &[], None, None, 10)
            .await
            .expect("search_text multi-word");

        assert_eq!(
            results.len(),
            1,
            "only the chunk with BOTH words is returned"
        );
        assert_eq!(results[0].id, both_id);
    }

    /// Empty query returns all rows up to `limit` — no WHERE clause is built
    /// when there are no words and no other filters.
    #[tokio::test]
    async fn search_text_empty_query_returns_all_up_to_limit() {
        let (store, _dir) = create_test_store().await;

        let chunks: Vec<_> = (0..5)
            .map(|i| make_chunk(&format!("entry {i}"), "st.md"))
            .collect();
        store.insert_chunks(chunks).await.expect("insert");

        // limit=10 — should return all 5
        let all = store
            .search_text("", &[], None, None, 10)
            .await
            .expect("search_text empty unlimited");
        assert_eq!(all.len(), 5, "empty query returns every chunk");

        // limit=3 — should cap at 3
        let capped = store
            .search_text("", &[], None, None, 3)
            .await
            .expect("search_text empty capped");
        assert_eq!(capped.len(), 3, "limit is respected for empty query");
    }

    /// LIKE-escape correctness for `%`: a chunk containing a literal `%` is
    /// found only when the query contains a literal `%`, and that `%` does NOT
    /// act as a SQL wildcard matching arbitrary content.
    ///
    /// Without proper escaping, `search_text("50%", …)` would use the pattern
    /// `%50%%` which matches anything containing "50" followed by any suffix —
    /// hitting both the "50% off" chunk AND the "50 cents" chunk.  With correct
    /// escaping (`ESCAPE '\'`) only the literal-`%` chunk is returned.
    #[tokio::test]
    async fn search_text_percent_is_not_treated_as_wildcard() {
        let (store, _dir) = create_test_store().await;

        // This chunk contains a literal '%'
        let literal_pct = make_chunk("discount 50% off today", "escape.md");
        // This chunk contains "50" but no '%' — must NOT be matched
        let no_pct = make_chunk("50 cents change", "escape.md");

        let literal_pct_id = literal_pct.id.clone();
        store
            .insert_chunks(vec![literal_pct, no_pct])
            .await
            .expect("insert");

        let results = store
            .search_text("50%", &[], None, None, 10)
            .await
            .expect("search_text percent escape");

        assert_eq!(
            results.len(),
            1,
            "unescaped '%' would match 'no_pct' too — exactly 1 result means escape works"
        );
        assert_eq!(
            results[0].id, literal_pct_id,
            "the returned chunk must be the one with a literal '%'"
        );
    }

    /// LIKE-escape correctness for `_`: a chunk containing a literal `_` is
    /// found only when the query contains a literal `_`, and that `_` does NOT
    /// act as a single-character SQL wildcard.
    ///
    /// Without escaping, `search_text("a_c", …)` would match "abc", "a-c",
    /// "a_c", etc.  With proper escaping only "a_c" is returned.
    #[tokio::test]
    async fn search_text_underscore_is_not_treated_as_wildcard() {
        let (store, _dir) = create_test_store().await;

        // Content with a literal underscore matching the query exactly
        let with_under = make_chunk("config key a_c defined", "escape.md");
        // Content whose middle character differs — must NOT match literal "a_c"
        let without_under = make_chunk("config key abc defined", "escape.md");

        let with_under_id = with_under.id.clone();
        store
            .insert_chunks(vec![with_under, without_under])
            .await
            .expect("insert");

        let results = store
            .search_text("a_c", &[], None, None, 10)
            .await
            .expect("search_text underscore escape");

        assert_eq!(
            results.len(),
            1,
            "unescaped '_' would also match 'abc' — exactly 1 result means escape works"
        );
        assert_eq!(
            results[0].id, with_under_id,
            "the returned chunk must be the one with a literal '_'"
        );
    }

    // --- stats: level validation ---

    /// Happy path: stats() correctly aggregates (level, count) pairs for rows
    /// whose level values are in the valid u8 range.
    #[tokio::test]
    async fn stats_groups_by_level_correctly() {
        let (store, _dir) = create_test_store().await;

        // Two H1 chunks (level=1) and one H2 chunk (level=2).
        let c1 = make_chunk("a", "l.md");
        let c2 = make_chunk("b", "l.md");
        let c3 = HierarchicalChunk::new(
            "c".to_string(),
            ChunkLevel::H2,
            None,
            "root".to_string(),
            "l.md".to_string(),
        );
        store.insert_chunks(vec![c1, c2, c3]).await.expect("insert");

        let s = store.stats().await.expect("stats");

        assert_eq!(s.total_chunks, 3);
        assert_eq!(*s.chunks_by_level.get(&1).expect("level 1 present"), 2);
        assert_eq!(*s.chunks_by_level.get(&2).expect("level 2 present"), 1);
    }

    /// Error path: a row whose `level` column holds an out-of-range value (256)
    /// causes `stats()` to return `Err` rather than silently wrapping 256 → 0
    /// (the old `level as u8` behaviour).
    ///
    /// The test is discriminating: the old `as u8` cast would map 256 → 0 and
    /// return `Ok(...)`, so a `Ok` result is a regression indicator.  The fixed
    /// `narrow::<u8>` call returns `FromSqlConversionFailure` for any value that
    /// does not fit in a `u8`.
    ///
    /// To plant the bad row we bypass the normal insert path and write directly
    /// via a raw SQL `execute` against the same connection the store uses.
    #[tokio::test]
    async fn stats_out_of_range_level_returns_error() {
        let (store, _dir) = create_test_store().await;

        // Insert a valid row first so we know the schema is initialised, then
        // overwrite its level to 256 via a raw UPDATE.
        let chunk = make_chunk("sentinel", "oor.md");
        let id = chunk.id.clone();
        store.insert_chunks(vec![chunk]).await.expect("insert");

        // Directly mutate the level to an out-of-range value via the internal
        // connection.  `mod tests` is a child of this module so private fields
        // are accessible.
        {
            let conn = store.conn.lock().expect("lock");
            conn.execute(
                "UPDATE chunks SET level = 256 WHERE id = ?1",
                rusqlite::params![id],
            )
            .expect("raw UPDATE to plant out-of-range level");
        }

        let result = store.stats().await;

        assert!(
            result.is_err(),
            "stats() must return Err when a row has level=256 (would wrap to 0 with `as u8`)"
        );
    }
}
