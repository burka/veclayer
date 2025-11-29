use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
    StringArray, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{connect, Connection, Table};

use super::{SearchResult, StoreStats, VectorStore};
use crate::{ChunkLevel, Error, HierarchicalChunk, Result};

const TABLE_NAME: &str = "chunks";

/// LanceDB-based vector store implementation.
/// Stores all data in a local directory, no external services needed.
pub struct LanceStore {
    connection: Connection,
    dimension: usize,
}

impl LanceStore {
    /// Open or create a LanceDB store at the given path
    pub async fn open(path: impl AsRef<Path>, dimension: usize) -> Result<Self> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;

        let uri = path.to_string_lossy().to_string();
        let connection = connect(&uri)
            .execute()
            .await
            .map_err(|e| Error::store(format!("Failed to connect to LanceDB: {}", e)))?;

        let store = Self {
            connection,
            dimension,
        };

        // Ensure table exists
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
            // Create empty table with schema
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
        let parent_ids: Vec<Option<&str>> =
            chunks.iter().map(|c| c.parent_id.as_deref()).collect();
        let paths: Vec<&str> = chunks.iter().map(|c| c.path.as_str()).collect();
        let source_files: Vec<&str> = chunks.iter().map(|c| c.source_file.as_str()).collect();
        let headings: Vec<Option<&str>> = chunks.iter().map(|c| c.heading.as_deref()).collect();

        // Build embeddings as FixedSizeList
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
        let embedding_array = FixedSizeListArray::new(
            field,
            self.dimension as i32,
            Arc::new(values),
            None,
        );

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
            ],
        )
        .map_err(|e| Error::store(format!("Failed to create record batch: {}", e)))
    }

    fn batch_to_chunks(&self, batch: &RecordBatch) -> Result<Vec<HierarchicalChunk>> {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::store("Invalid id column"))?;
        let contents = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::store("Invalid content column"))?;
        let embeddings = batch
            .column(2)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| Error::store("Invalid embedding column"))?;
        let levels = batch
            .column(3)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .ok_or_else(|| Error::store("Invalid level column"))?;
        let parent_ids = batch
            .column(4)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::store("Invalid parent_id column"))?;
        let paths = batch
            .column(5)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::store("Invalid path column"))?;
        let source_files = batch
            .column(6)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::store("Invalid source_file column"))?;
        let headings = batch
            .column(7)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::store("Invalid heading column"))?;

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
            // Get distance column if present
            let distances: Option<&Float32Array> = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref());

            let chunks = self.batch_to_chunks(&batch)?;
            for (i, chunk) in chunks.into_iter().enumerate() {
                let score = distances
                    .map(|d| 1.0 - d.value(i)) // Convert distance to similarity
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
            .only_if(format!("parent_id = '{}'", parent_id.replace('\'', "''")))
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
            .only_if(format!("id = '{}'", id.replace('\'', "''")))
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

    async fn get_by_source(&self, source_file: &str) -> Result<Vec<HierarchicalChunk>> {
        let table = self.get_table().await?;

        let results = table
            .query()
            .only_if(format!(
                "source_file = '{}'",
                source_file.replace('\'', "''")
            ))
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

        // Get count before delete
        let before = self.get_by_source(source_file).await?.len();

        table
            .delete(&format!(
                "source_file = '{}'",
                source_file.replace('\'', "''")
            ))
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

            let levels = batch
                .column(3)
                .as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or_else(|| Error::store("Invalid level column"))?;

            let sources = batch
                .column(6)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| Error::store("Invalid source_file column"))?;

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
}
