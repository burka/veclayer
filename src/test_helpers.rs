/// Shared test helpers for use across all test modules.
///
/// This module is compiled only when running tests.
#[cfg(test)]
use crate::store::VectorStore;

/// Returns the embedding dimension that Config::new() would resolve.
/// Uses the real config resolution so tests match production behavior.
#[cfg(all(test, feature = "config"))]
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

/// Deterministic in-memory embedder for hermetic tests.
///
/// Produces a constant vector of the configured dimension so tests never depend
/// on a network embedding service. Use [`MockEmbedder::failing`] to exercise
/// embedding-failure paths.
#[cfg(test)]
pub(crate) struct MockEmbedder {
    dimension: usize,
    fail: bool,
}

#[cfg(test)]
impl MockEmbedder {
    pub(crate) fn new(dimension: usize) -> Self {
        Self {
            dimension,
            fail: false,
        }
    }

    /// An embedder whose `embed` always returns an error, for import-failure tests.
    pub(crate) fn failing(dimension: usize) -> Self {
        Self {
            dimension,
            fail: true,
        }
    }
}

#[cfg(test)]
impl crate::Embedder for MockEmbedder {
    fn embed<'a>(
        &'a self,
        texts: &'a [&'a str],
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = crate::Result<Vec<Vec<f32>>>> + Send + 'a>,
    > {
        let fail = self.fail;
        let dimension = self.dimension;
        let count = texts.len();
        Box::pin(async move {
            if fail {
                return Err(crate::Error::config("mock embedder failure"));
            }
            Ok(vec![vec![1.0; dimension]; count])
        })
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "mock"
    }
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

/// Open a test store and insert the given chunks, returning (store, data_dir, temp_dir_guard).
///
/// The caller must keep the returned guard alive for the lifetime of the store.
/// Using `let (store, _data_dir, _guard) = test_store_with_chunks(...).await;` is the
/// idiomatic way to ensure the directory isn't deleted while the store is in use.
#[cfg(test)]
pub(crate) async fn test_store_with_chunks(
    chunks: Vec<crate::HierarchicalChunk>,
) -> (
    std::sync::Arc<crate::store::StoreBackend>,
    std::path::PathBuf,
    tempfile::TempDir,
) {
    let (store, data_dir, dir) = test_store_setup("full", 384).await;
    store.insert_chunks(chunks).await.unwrap();
    (store, data_dir, dir)
}

/// Open a test metadata store, returning (store, data_dir, temp_dir_guard).
///
/// For use with read-only store operations that don't require full embeddings.
#[cfg(all(test, feature = "store-lance", feature = "mcp"))]
pub(crate) async fn test_store_meta() -> (
    std::sync::Arc<crate::store::StoreBackend>,
    std::path::PathBuf,
    tempfile::TempDir,
) {
    let (store, data_dir, dir) = test_store_setup("metadata", 384).await;
    (store, data_dir, dir)
}

/// Shared tempdir + store open helper. kind is "full" or "metadata".
async fn test_store_setup(
    kind: &str,
    dimension: usize,
) -> (
    std::sync::Arc<crate::store::StoreBackend>,
    std::path::PathBuf,
    tempfile::TempDir,
) {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_path_buf();
    let store = match kind {
        "full" => crate::store::StoreBackend::open(&data_dir, dimension, false)
            .await
            .unwrap(),
        #[cfg(feature = "store-lance")]
        "metadata" => crate::store::StoreBackend::open_metadata(&data_dir, false)
            .await
            .unwrap(),
        _ => unreachable!(),
    };
    (std::sync::Arc::new(store), data_dir, dir)
}

/// Assert that a file has unix permission bits 0o600 (owner read/write only).
#[cfg(all(unix, test, feature = "auth"))]
pub(crate) fn assert_file_mode_600(path: &std::path::Path, context: &str) {
    use std::os::unix::fs::MetadataExt;
    let meta = std::fs::metadata(path).unwrap();
    let mode = meta.mode() & 0o777;
    assert_eq!(mode, 0o600, "{}: expected 0o600, got 0o{mode:o}", context);
}

/// Create a test Entry with the given content and visibility.
#[cfg(test)]
pub(crate) fn make_test_entry(content: &str, visibility: &str) -> crate::Entry {
    crate::Entry {
        content: content.to_string(),
        entry_type: crate::chunk::EntryType::Raw,
        source: "test".to_string(),
        created_at: 0,
        perspectives: vec![],
        relations: vec![],
        summarizes: vec![],
        heading: None,
        parent_id: None,
        impression_hint: None,
        impression_strength: 1.0,
        expires_at: None,
        visibility: visibility.to_string(),
        level: crate::chunk::ChunkLevel(7),
        path: String::new(),
    }
}

/// Create a test StoredBlob with the given content, visibility, and empty embeddings.
#[cfg(test)]
pub(crate) fn test_blob(content: &str, visibility: &str) -> crate::StoredBlob {
    crate::StoredBlob {
        entry: make_test_entry(content, visibility),
        embeddings: vec![],
    }
}
