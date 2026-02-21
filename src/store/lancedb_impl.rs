use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, RecordBatchIterator,
    StringArray, UInt16Array, UInt32Array, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{connect, Connection, Table};

use super::{FileLock, SearchResult, StoreStats, VectorStore};
use crate::{ChunkLevel, ClusterMembership, Error, HierarchicalChunk, Result};

const TABLE_NAME: &str = "chunks";

fn sql_escape(s: &str) -> String {
    s.replace('\'', "''")
}

fn eq_filter(column: &str, value: &str) -> String {
    format!("{} = '{}'", column, sql_escape(value))
}

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
    _lock: Option<FileLock>,
}

impl LanceStore {
    pub async fn open(path: impl AsRef<Path>, dimension: usize, read_only: bool) -> Result<Self> {
        let path = path.as_ref();
        let lock = (!read_only).then(|| FileLock::acquire(path)).transpose()?;

        std::fs::create_dir_all(path)?;

        let uri = path.to_string_lossy().to_string();
        let connection = connect(&uri)
            .execute()
            .await
            .map_err(|e| Error::store(format!("Failed to connect to LanceDB: {}", e)))?;

        let store = Self {
            connection,
            dimension,
            _lock: lock,
        };

        store.ensure_table().await?;

        Ok(store)
    }

    pub async fn open_metadata(path: impl AsRef<Path>, read_only: bool) -> Result<Self> {
        let path = path.as_ref();
        let lock = (!read_only).then(|| FileLock::acquire(path)).transpose()?;

        std::fs::create_dir_all(path)?;

        let uri = path.to_string_lossy().to_string();
        let connection = connect(&uri)
            .execute()
            .await
            .map_err(|e| Error::store(format!("Failed to connect to LanceDB: {}", e)))?;

        let store = Self {
            connection,
            dimension: 384,
            _lock: lock,
        };

        store.ensure_table().await?;

        Ok(store)
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
        ]))
    }

    async fn ensure_table(&self) -> Result<()> {
        let tables = self
            .connection
            .table_names()
            .execute()
            .await
            .map_err(|e| Error::store(format!("Failed to list tables: {}", e)))?;

        if !tables.contains(&TABLE_NAME.to_string()) {
            let schema = self.schema();

            self.connection
                .create_empty_table(TABLE_NAME, schema)
                .execute()
                .await
                .map_err(|e| Error::store(format!("Failed to create table: {}", e)))?;
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
        for chunk in chunks {
            if let Some(ref emb) = chunk.embedding {
                if emb.len() != self.dimension {
                    return Err(Error::store(format!(
                        "Embedding dimension mismatch: expected {}, got {}",
                        self.dimension,
                        emb.len()
                    )));
                }
                embedding_values.extend(emb);
            } else {
                return Err(Error::store("Chunk missing embedding"));
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

        let mut chunks = Vec::with_capacity(batch.num_rows());

        for i in 0..batch.num_rows() {
            let embedding_array = embeddings.value(i);
            let embedding_values = embedding_array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| Error::store("Invalid embedding values"))?;
            let embedding: Vec<f32> = (0..embedding_values.len())
                .map(|j| embedding_values.value(j))
                .collect();

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
                        match col.value(i) {
                            "summary" => Some(crate::chunk::EntryType::Summary),
                            "meta" => Some(crate::chunk::EntryType::Meta),
                            "impression" => Some(crate::chunk::EntryType::Impression),
                            _ => Some(crate::chunk::EntryType::Raw),
                        }
                    }
                })
                .unwrap_or(crate::chunk::EntryType::Raw);

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
                embedding: Some(embedding),
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
                start_offset: 0,
                end_offset: 0,
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

        Ok(())
    }

    async fn search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        level_filter: Option<ChunkLevel>,
    ) -> Result<Vec<SearchResult>> {
        let table = self.get_table().await?;

        let query_vec: Vec<f32> = query_embedding.to_vec();

        let mut query = table.query().nearest_to(query_vec).map_err(|e| {
            Error::search(format!("Failed to create nearest neighbor query: {}", e))
        })?;

        if let Some(level) = level_filter {
            query = query.only_if(format!("level = {}", level.depth()));
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
                let score = distances.map(|d| 1.0 - d.value(i)).unwrap_or(1.0);
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

        let mut chunks = Vec::new();
        for batch in results {
            chunks.extend(self.batch_to_chunks(&batch)?);
        }

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

        let mut chunks = Vec::new();
        for batch in &results {
            chunks.extend(self.batch_to_chunks(batch)?);
        }

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
            .only_if(format!("source_file = '{}'", sql_escape(source_file)))
            .execute()
            .await
            .map_err(|e| Error::search(format!("Failed to query by source: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| Error::search(format!("Failed to collect results: {}", e)))?;

        let mut chunks = Vec::new();
        for batch in results {
            chunks.extend(self.batch_to_chunks(&batch)?);
        }

        Ok(chunks)
    }

    async fn delete_by_source(&self, source_file: &str) -> Result<usize> {
        let table = self.get_table().await?;

        let before = self.get_by_source(source_file).await?.len();

        table
            .delete(&format!("source_file = '{}'", sql_escape(source_file)))
            .await
            .map_err(|e| Error::store(format!("Failed to delete by source: {}", e)))?;

        Ok(before)
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
        let mut source_files: Vec<String> = Vec::new();

        for batch in results {
            total_chunks += batch.num_rows();

            let levels = Self::extract_column::<UInt8Array>(&batch, 3, "level")?;
            let sources = Self::extract_column::<StringArray>(&batch, 6, "source_file")?;

            for i in 0..batch.num_rows() {
                let level = levels.value(i);
                *chunks_by_level.entry(level).or_insert(0) += 1;

                let source = sources.value(i).to_string();
                if !source_files.contains(&source) {
                    source_files.push(source);
                }
            }
        }

        Ok(StoreStats {
            total_chunks,
            chunks_by_level,
            source_files,
        })
    }

    async fn update_access_profiles(
        &self,
        updates: Vec<(String, crate::AccessProfile)>,
    ) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }

        let table = self.get_table().await?;

        for (chunk_id, profile) in updates {
            let filter = eq_filter("id", &chunk_id);

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
                .map_err(|e| Error::store(format!("Failed to update access profile: {}", e)))?;
        }

        Ok(())
    }

    async fn update_visibility(&self, chunk_id: &str, visibility: &str) -> Result<()> {
        let table = self.get_table().await?;
        let filter = eq_filter("id", chunk_id);

        table
            .update()
            .column("visibility", format!("'{}'", sql_escape(visibility)))
            .only_if(filter)
            .execute()
            .await
            .map_err(|e| Error::store(format!("Failed to update visibility: {}", e)))?;

        Ok(())
    }

    async fn add_relation(&self, chunk_id: &str, relation: crate::ChunkRelation) -> Result<()> {
        let chunk = self
            .get_by_id(chunk_id)
            .await?
            .ok_or_else(|| Error::store(format!("Chunk not found: {}", chunk_id)))?;

        let mut relations = chunk.relations;
        relations.push(relation);

        let relations_json = serde_json::to_string(&relations)
            .map_err(|e| Error::store(format!("serialize relations: {}", e)))?;

        let table = self.get_table().await?;
        let filter = eq_filter("id", chunk_id);

        table
            .update()
            .column("relations", format!("'{}'", sql_escape(&relations_json)))
            .only_if(filter)
            .execute()
            .await
            .map_err(|e| Error::store(format!("Failed to add relation: {}", e)))?;

        Ok(())
    }

    async fn search_by_perspective(
        &self,
        query_embedding: &[f32],
        limit: usize,
        perspective: &str,
    ) -> Result<Vec<SearchResult>> {
        let table = self.get_table().await?;
        let query_vec: Vec<f32> = query_embedding.to_vec();

        let escaped = sql_escape(perspective);
        let filter = format!("perspectives LIKE '%\"{}%'", escaped);

        let query = table
            .query()
            .nearest_to(query_vec)
            .map_err(|e| Error::search(format!("Failed to create perspective search: {}", e)))?
            .only_if(filter)
            .limit(limit);

        let results = query
            .execute()
            .await
            .map_err(|e| Error::search(format!("Failed to execute perspective search: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| Error::search(format!("Failed to collect perspective results: {}", e)))?;

        let mut search_results = Vec::new();
        for batch in results {
            let distances: Option<&Float32Array> = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref());

            let chunks = self.batch_to_chunks(&batch)?;
            for (i, chunk) in chunks.into_iter().enumerate() {
                let score = distances.map(|d| 1.0 - d.value(i)).unwrap_or(1.0);
                search_results.push(SearchResult { chunk, score });
            }
        }

        Ok(search_results)
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

        let mut all_chunks = Vec::new();
        for batch in results {
            let mut chunks = self.batch_to_chunks(&batch)?;
            all_chunks.append(&mut chunks);
        }

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

        let mut all_chunks = Vec::new();
        for batch in results {
            let mut chunks = self.batch_to_chunks(&batch)?;
            all_chunks.append(&mut chunks);
        }

        all_chunks.sort_by_key(|c| c.access_profile.last_rolled);

        all_chunks.truncate(limit);
        Ok(all_chunks)
    }

    async fn list_entries(
        &self,
        perspective: Option<&str>,
        since: Option<i64>,
        until: Option<i64>,
        limit: usize,
    ) -> Result<Vec<HierarchicalChunk>> {
        let table = self.get_table().await?;

        let mut filters = Vec::new();
        if let Some(p) = perspective {
            let escaped = sql_escape(p);
            filters.push(format!("perspectives LIKE '%\"{}%'", escaped));
        }
        if let Some(s) = since {
            filters.push(format!("created_at >= {}", s));
        }
        if let Some(u) = until {
            filters.push(format!("created_at <= {}", u));
        }

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

        let mut all_chunks = Vec::new();
        for batch in results {
            all_chunks.extend(self.batch_to_chunks(&batch)?);
        }

        all_chunks.sort_by(|a, b| {
            b.access_profile
                .created_at
                .cmp(&a.access_profile.created_at)
        });
        all_chunks.truncate(limit);

        Ok(all_chunks)
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
        let results = store.search(&query_embedding, 2, None).await.unwrap();

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
            .search(&query_embedding, 10, Some(ChunkLevel::H1))
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
        assert!(children.iter().any(|c| c.id == "child-1"));
        assert!(children.iter().any(|c| c.id == "child-2"));
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
    async fn test_missing_embedding() {
        let (store, _temp) = create_test_store().await;

        let mut chunk = create_test_chunk("no-emb", "No embedding", ChunkLevel::H1);
        chunk.embedding = None;

        let result = store.insert_chunks(vec![chunk]).await;
        assert!(result.is_err());
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

        let result = store.list_entries(None, None, None, 10).await.unwrap();
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

        let result = store.list_entries(None, None, None, 10).await.unwrap();
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
            .list_entries(Some("decisions"), None, None, 10)
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
            .list_entries(None, Some(1500), Some(2500), 10)
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

        let result = store.list_entries(None, None, None, 2).await.unwrap();
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
}
