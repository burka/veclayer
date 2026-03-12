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

/// Escape SQL LIKE wildcards (`%`, `_`) and single quotes for safe interpolation
/// into a LIKE pattern. The caller must append `ESCAPE '\'` to the clause.
fn escape_like(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
        .replace('\'', "''")
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
    /// Dimension hint stored for future validation use.
    #[allow(dead_code)]
    dimension: usize,
    /// Read-only flag preserved for guard assertions and open/close logic.
    #[allow(dead_code)]
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

fn bytes_to_vec(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk is exactly 4 bytes")))
        .collect()
}

// --- Row mapping ---

fn entry_type_from_str(s: &str) -> EntryType {
    match s {
        "summary" => EntryType::Summary,
        "meta" => EntryType::Meta,
        "impression" => EntryType::Impression,
        _ => EntryType::Raw,
    }
}

fn row_to_chunk(row: &rusqlite::Row) -> rusqlite::Result<HierarchicalChunk> {
    let id: String = row.get("id")?;
    let content: String = row.get("content")?;
    let embedding_blob: Option<Vec<u8>> = row.get("embedding")?;
    let embedding_status: String = row.get("embedding_status")?;
    let level_depth: u8 = row.get::<_, i64>("level")? as u8;
    let parent_id: Option<String> = row.get("parent_id")?;
    let path: String = row.get("path")?;
    let source_file: String = row.get("source_file")?;
    let heading: Option<String> = row.get("heading")?;
    let start_offset: usize = row.get::<_, i64>("start_offset")? as usize;
    let end_offset: usize = row.get::<_, i64>("end_offset")? as usize;
    let entry_type_str: String = row.get("entry_type")?;
    let visibility: String = row.get("visibility")?;
    let perspectives_json: String = row.get("perspectives")?;
    let relations_json: String = row.get("relations")?;
    let cluster_memberships_json: String = row.get("cluster_memberships")?;
    let summarizes_json: String = row.get("summarizes")?;
    let created_at: i64 = row.get("created_at")?;
    let last_rolled: i64 = row.get("last_rolled")?;
    let access_hour: u16 = row.get::<_, i64>("access_hour")? as u16;
    let access_day: u16 = row.get::<_, i64>("access_day")? as u16;
    let access_week: u16 = row.get::<_, i64>("access_week")? as u16;
    let access_month: u16 = row.get::<_, i64>("access_month")? as u16;
    let access_year: u16 = row.get::<_, i64>("access_year")? as u16;
    let access_total: u32 = row.get::<_, i64>("access_total")? as u32;
    let expires_at: Option<i64> = row.get("expires_at")?;
    let impression_hint: Option<String> = row.get("impression_hint")?;
    let impression_strength: f32 = row.get::<_, f64>("impression_strength")? as f32;

    let embedding = if embedding_status == EMBEDDING_STATUS_PENDING {
        None
    } else {
        embedding_blob.map(|b| bytes_to_vec(&b))
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
        self.with_conn_mut(move |conn| {
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
        perspective: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Vec<SearchResult>>> + Send {
        let query_vec = query_embedding.to_vec();
        let perspective_owned = perspective.map(str::to_owned);

        self.with_conn(move |conn| {
            let mut sql = String::from("SELECT * FROM chunks WHERE embedding_status = 'embedded'");

            if let Some(level) = level_filter {
                sql.push_str(&format!(" AND level = {}", level.0));
            }

            if let Some(ref p) = perspective_owned {
                let escaped = escape_like(p);
                sql.push_str(&format!(
                    " AND perspectives LIKE '%\"{}\"%' ESCAPE '\\'",
                    escaped
                ));
            }

            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| Error::store(format!("search prepare: {e}")))?;

            let rows: Vec<HierarchicalChunk> = stmt
                .query_map([], row_to_chunk)
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

            scored.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
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

        self.with_conn(move |conn| {
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
                    Ok((level as u8, count as usize))
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
        self.with_conn_mut(move |conn| {
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

        self.with_conn(move |conn| {
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

        self.with_conn(move |conn| {
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

    fn list_entries(
        &self,
        perspective: Option<&str>,
        since: Option<i64>,
        until: Option<i64>,
        limit: usize,
    ) -> impl std::future::Future<Output = Result<Vec<HierarchicalChunk>>> + Send {
        let perspective_owned = perspective.map(str::to_owned);

        self.with_conn(move |conn| {
            let mut conditions: Vec<String> = Vec::new();
            let mut values: Vec<i64> = Vec::new();
            let mut idx = 1usize;

            if let Some(ref p) = perspective_owned {
                let escaped = escape_like(p);
                conditions.push(format!("perspectives LIKE '%\"{}\"%' ESCAPE '\\'", escaped));
            }
            if let Some(ts) = since {
                conditions.push(format!("created_at >= ?{idx}"));
                values.push(ts);
                idx += 1;
            }
            if let Some(ts) = until {
                conditions.push(format!("created_at <= ?{idx}"));
                values.push(ts);
                idx += 1;
            }

            let where_clause = if conditions.is_empty() {
                String::new()
            } else {
                format!(" WHERE {}", conditions.join(" AND "))
            };

            let sql =
                format!("SELECT * FROM chunks{where_clause} ORDER BY created_at DESC LIMIT ?{idx}");
            values.push(limit as i64);

            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| Error::store(format!("list_entries prepare: {e}")))?;

            let chunks: Vec<HierarchicalChunk> = stmt
                .query_map(rusqlite::params_from_iter(values), row_to_chunk)
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
        self.with_conn_mut(move |conn| {
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
            .search(&[1.0, 0.0, 0.0, 0.0], 10, None, None)
            .await
            .expect("search");

        assert!(!results.is_empty(), "expected results");

        // Perspective-filtered: only "decisions"
        let filtered = store
            .search(&[1.0, 0.0, 0.0, 0.0], 10, None, Some("decisions"))
            .await
            .expect("search with perspective");

        assert_eq!(filtered.len(), 1);
        assert!(filtered[0]
            .chunk
            .perspectives
            .contains(&"decisions".to_string()));

        // Level filter — level 2 returns nothing (all inserted at H1 = depth 1)
        let level_filtered = store
            .search(&[1.0, 0.0, 0.0, 0.0], 10, Some(ChunkLevel::H2), None)
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
            .search(&[0.1, 0.2, 0.3, 0.4], 5, None, None)
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
}
