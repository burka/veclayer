/// Shared test helpers for use across all test modules.
///
/// This module is compiled only when running tests.

#[cfg(test)]
use crate::store::VectorStore;

/// Returns the embedding dimension that Config::new() would resolve.
/// Uses the real config resolution so tests match production behavior.
#[cfg(test)]
pub(crate) fn config_dimension() -> usize {
    let config = crate::Config::new();
    crate::embedder::from_config(&config.embedder)
        .map(|e| e.dimension())
        .unwrap_or(384)
}

#[cfg(test)]
pub(crate) fn make_test_chunk(id: &str, content: &str) -> crate::HierarchicalChunk {
    make_test_chunk_dim(id, content, 384)
}

#[cfg(test)]
pub(crate) fn make_test_chunk_dim(
    id: &str,
    content: &str,
    dimension: usize,
) -> crate::HierarchicalChunk {
    crate::HierarchicalChunk {
        id: id.to_string(),
        content: content.to_string(),
        embedding: Some(vec![0.0f32; dimension]),
        level: crate::chunk::ChunkLevel(1),
        parent_id: None,
        path: "test".to_string(),
        source_file: "test".to_string(),
        heading: Some(format!("Heading for {}", id)),
        start_offset: 0,
        end_offset: 0,
        cluster_memberships: vec![],
        entry_type: crate::chunk::EntryType::Raw,
        summarizes: vec![],
        perspectives: vec![],
        visibility: "normal".to_string(),
        relations: vec![],
        access_profile: crate::AccessProfile::new(),
        expires_at: None,
        impression_hint: None,
        impression_strength: 1.0,
    }
}

/// Open a test store and insert the given chunks, returning (store, temp_dir_guard).
///
/// The caller must keep the returned guard alive for the lifetime of the store.
/// Using `let (store, _guard) = test_store_with_chunks(...).await;` is the
/// idiomatic way to ensure the directory isn't deleted while the store is in use.
#[cfg(test)]
pub(crate) async fn test_store_with_chunks(
    chunks: Vec<crate::HierarchicalChunk>,
) -> (std::sync::Arc<crate::store::StoreBackend>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let store = crate::store::StoreBackend::open(dir.path(), 384, false)
        .await
        .unwrap();
    store.insert_chunks(chunks).await.unwrap();
    (std::sync::Arc::new(store), dir)
}
