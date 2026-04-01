//! Agent-configurable aging rules for automatic visibility degradation.
//!
//! Aging now considers salience: high-salience entries are protected
//! from degradation even when they haven't been accessed recently.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::chunk::now_epoch_secs;
use crate::salience::{self, SalienceWeights};
use crate::{Result, VectorStore};

const AGING_CONFIG_FILE: &str = "aging_config.json";

/// Aging configuration: rules for automatic visibility degradation.
///
/// The agent sets these rules via the `configure_aging` MCP tool.
/// `apply_aging` then executes the rules, degrading chunks that match.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgingConfig {
    /// Number of days without access before a chunk is degraded.
    /// Default: 30
    pub degrade_after_days: u32,
    /// Visibility to assign to degraded chunks.
    /// Default: "deep_only"
    pub degrade_to: String,
    /// Only degrade chunks with these visibilities.
    /// Default: ["normal"]
    pub degrade_from: Vec<String>,
    /// Minimum salience score to protect an entry from degradation.
    /// Entries with salience >= this threshold are kept even when stale.
    /// Default: 0.15
    #[serde(default = "default_salience_protection")]
    pub salience_protection: f32,
}

fn default_salience_protection() -> f32 {
    0.15
}

impl Default for AgingConfig {
    fn default() -> Self {
        Self {
            degrade_after_days: 30,
            degrade_to: "deep_only".to_string(),
            degrade_from: vec!["normal".to_string()],
            salience_protection: default_salience_protection(),
        }
    }
}

impl AgingConfig {
    /// Load from the data directory. Returns default if no config exists.
    #[must_use]
    pub fn load(data_dir: &Path) -> Self {
        let path = data_dir.join(AGING_CONFIG_FILE);
        if path.exists() {
            std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default()
        } else {
            Self::default()
        }
    }

    /// Save to the data directory.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or file writing fails.
    pub fn save(&self, data_dir: &Path) -> Result<()> {
        let path = data_dir.join(AGING_CONFIG_FILE);
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| crate::Error::config(format!("Failed to serialize aging config: {e}")))?;
        std::fs::write(&path, json)?;
        Ok(())
    }

    /// Threshold in seconds.
    #[must_use]
    pub fn stale_seconds(&self) -> i64 {
        i64::from(self.degrade_after_days) * 86_400
    }
}

/// Result of applying aging rules.
#[derive(Debug, Clone, Serialize)]
pub struct AgingResult {
    /// Number of chunks that were degraded.
    pub degraded_count: usize,
    /// IDs of chunks that were degraded.
    pub degraded_ids: Vec<String>,
}

/// Apply aging rules: find stale chunks and degrade their visibility.
///
/// Salience protection: entries with composite salience >= `salience_protection`
/// are skipped even when stale, preserving high-value knowledge.
///
/// # Errors
///
/// Returns an error if store operations fail.
pub async fn apply_aging<S: VectorStore>(store: &S, config: &AgingConfig) -> Result<AgingResult> {
    let now = now_epoch_secs();
    let cutoff_secs = config.stale_seconds();
    let weights = SalienceWeights::default();

    let stale = store.get_stale_chunks(cutoff_secs, 500).await?;

    let mut degraded_ids = Vec::new();

    for chunk in &stale {
        // Only degrade chunks whose current visibility is in the degrade_from list
        if !config.degrade_from.contains(&chunk.visibility) {
            continue;
        }

        // Check that the chunk is truly stale (no recent activity)
        let total_recent = u32::from(chunk.access_profile.hour)
            + u32::from(chunk.access_profile.day)
            + u32::from(chunk.access_profile.week)
            + u32::from(chunk.access_profile.month);

        let age_since_roll = now - chunk.access_profile.last_rolled;

        if total_recent > 0 || age_since_roll < cutoff_secs {
            continue;
        }

        // Salience protection: high-salience entries survive aging
        let salience_score = salience::compute(chunk, &weights);
        if salience_score.composite >= config.salience_protection {
            continue;
        }

        store
            .update_visibility(&chunk.id, &config.degrade_to)
            .await
            .map_err(|e| {
                crate::Error::store(format!("Failed to degrade chunk {}: {}", chunk.id, e))
            })?;
        degraded_ids.push(chunk.id.clone());
    }

    Ok(AgingResult {
        degraded_count: degraded_ids.len(),
        degraded_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tempfile::TempDir;

    use crate::access_profile::AccessProfile;
    use crate::chunk::{ChunkLevel, HierarchicalChunk};
    use crate::store::{SearchResult, StoreStats};
    use crate::{ChunkRelation, Result};

    // --- AgingConfig tests ---

    #[test]
    fn test_aging_config_default() {
        let config = AgingConfig::default();
        assert_eq!(config.degrade_after_days, 30);
        assert_eq!(config.degrade_to, "deep_only");
        assert_eq!(config.degrade_from, vec!["normal"]);
        assert_eq!(config.stale_seconds(), 30 * 86_400);
        assert!((config.salience_protection - 0.15).abs() < 0.001);
    }

    #[test]
    fn test_aging_config_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let config = AgingConfig {
            degrade_after_days: 14,
            degrade_to: "archived".to_string(),
            degrade_from: vec!["normal".to_string(), "seasonal".to_string()],
            salience_protection: 0.2,
        };

        config.save(temp_dir.path()).unwrap();
        let loaded = AgingConfig::load(temp_dir.path());

        assert_eq!(loaded.degrade_after_days, 14);
        assert_eq!(loaded.degrade_to, "archived");
        assert_eq!(loaded.degrade_from.len(), 2);
    }

    #[test]
    fn test_aging_config_load_missing() {
        let temp_dir = TempDir::new().unwrap();
        let loaded = AgingConfig::load(temp_dir.path());
        assert_eq!(loaded.degrade_after_days, 30); // default
    }

    #[test]
    fn test_aging_config_save_and_load_roundtrip_preserves_all_fields() {
        let temp_dir = TempDir::new().unwrap();
        let config = AgingConfig {
            degrade_after_days: 7,
            degrade_to: "seasonal".to_string(),
            degrade_from: vec!["normal".to_string()],
            salience_protection: 0.5,
        };
        config.save(temp_dir.path()).unwrap();
        let loaded = AgingConfig::load(temp_dir.path());
        assert_eq!(loaded.degrade_after_days, 7);
        assert_eq!(loaded.degrade_to, "seasonal");
        assert_eq!(loaded.degrade_from, vec!["normal"]);
        assert!((loaded.salience_protection - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_aging_config_load_corrupted_falls_back_to_default() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join(AGING_CONFIG_FILE);
        std::fs::write(&path, b"this is not valid json").unwrap();
        let loaded = AgingConfig::load(temp_dir.path());
        // Corrupted file silently falls back to default
        assert_eq!(loaded.degrade_after_days, 30);
    }

    #[test]
    fn test_stale_seconds_one_day() {
        let config = AgingConfig {
            degrade_after_days: 1,
            ..AgingConfig::default()
        };
        assert_eq!(config.stale_seconds(), 86_400);
    }

    #[test]
    fn test_stale_seconds_zero_days() {
        let config = AgingConfig {
            degrade_after_days: 0,
            ..AgingConfig::default()
        };
        assert_eq!(config.stale_seconds(), 0);
    }

    #[test]
    fn test_stale_seconds_large_value() {
        let config = AgingConfig {
            degrade_after_days: 365,
            ..AgingConfig::default()
        };
        assert_eq!(config.stale_seconds(), 365 * 86_400);
    }

    // --- AgingResult serde ---

    #[test]
    fn test_aging_result_serializes_to_json() {
        let result = AgingResult {
            degraded_count: 2,
            degraded_ids: vec!["abc".to_string(), "def".to_string()],
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("degraded_count"));
        assert!(json.contains("abc"));
    }

    #[test]
    fn test_aging_result_empty() {
        let result = AgingResult {
            degraded_count: 0,
            degraded_ids: vec![],
        };
        assert_eq!(result.degraded_count, 0);
        assert!(result.degraded_ids.is_empty());
    }

    // --- apply_aging integration tests ---

    /// Minimal in-memory VectorStore for aging tests.
    struct MockStore {
        stale_chunks: Vec<HierarchicalChunk>,
        degraded: Arc<Mutex<Vec<(String, String)>>>,
    }

    impl MockStore {
        fn new(stale_chunks: Vec<HierarchicalChunk>) -> Self {
            Self {
                stale_chunks,
                degraded: Arc::new(Mutex::new(vec![])),
            }
        }

        fn degraded_ids(&self) -> Vec<String> {
            self.degraded
                .lock()
                .unwrap()
                .iter()
                .map(|(id, _)| id.clone())
                .collect()
        }
    }

    impl crate::VectorStore for MockStore {
        async fn insert_chunks(&self, _chunks: Vec<HierarchicalChunk>) -> Result<()> {
            Ok(())
        }

        async fn search(
            &self,
            _query_embedding: &[f32],
            _limit: usize,
            _level_filter: Option<ChunkLevel>,
            _perspectives: &[&str],
        ) -> Result<Vec<SearchResult>> {
            Ok(vec![])
        }

        async fn get_children(&self, _parent_id: &str) -> Result<Vec<HierarchicalChunk>> {
            Ok(vec![])
        }

        async fn get_by_id(&self, _id: &str) -> Result<Option<HierarchicalChunk>> {
            Ok(None)
        }

        async fn get_by_id_prefix(&self, _prefix: &str) -> Result<Option<HierarchicalChunk>> {
            Ok(None)
        }

        async fn get_by_source(&self, _source_file: &str) -> Result<Vec<HierarchicalChunk>> {
            Ok(vec![])
        }

        async fn delete_by_source(&self, _source_file: &str) -> Result<usize> {
            Ok(0)
        }

        async fn stats(&self) -> Result<StoreStats> {
            Ok(StoreStats::default())
        }

        async fn update_access_profiles(
            &self,
            _updates: Vec<(String, AccessProfile)>,
        ) -> Result<()> {
            Ok(())
        }

        async fn update_visibility(&self, chunk_id: &str, visibility: &str) -> Result<()> {
            self.degraded
                .lock()
                .unwrap()
                .push((chunk_id.to_string(), visibility.to_string()));
            Ok(())
        }

        async fn add_relation(&self, _chunk_id: &str, _relation: ChunkRelation) -> Result<()> {
            Ok(())
        }

        async fn get_hot_chunks(&self, _limit: usize) -> Result<Vec<HierarchicalChunk>> {
            Ok(vec![])
        }

        async fn get_stale_chunks(
            &self,
            _stale_seconds: i64,
            _limit: usize,
        ) -> Result<Vec<HierarchicalChunk>> {
            Ok(self.stale_chunks.clone())
        }

        async fn search_text(
            &self,
            _query: &str,
            _perspectives: &[&str],
            _since: Option<i64>,
            _until: Option<i64>,
            _limit: usize,
        ) -> Result<Vec<HierarchicalChunk>> {
            Ok(vec![])
        }

        async fn list_entries(
            &self,
            _perspectives: &[&str],
            _since: Option<i64>,
            _until: Option<i64>,
            _limit: usize,
        ) -> Result<Vec<HierarchicalChunk>> {
            Ok(vec![])
        }

        async fn get_pending_embeddings(&self, _limit: usize) -> Result<Vec<HierarchicalChunk>> {
            Ok(vec![])
        }

        async fn batch_update_embeddings(&self, _updates: Vec<(String, Vec<f32>)>) -> Result<()> {
            Ok(())
        }

        async fn count_pending_embeddings(&self) -> Result<usize> {
            Ok(0)
        }
    }

    /// Build a truly stale chunk: last_rolled far in the past, no recent accesses.
    fn stale_chunk(id_suffix: &str, visibility: &str) -> HierarchicalChunk {
        let far_past = 1_000_000_i64; // epoch seconds well in the past
        let mut chunk = HierarchicalChunk::new(
            format!("stale content {}", id_suffix),
            ChunkLevel::CONTENT,
            None,
            String::new(),
            "test.md".to_string(),
        );
        chunk.visibility = visibility.to_string();
        // Place last_rolled far in the past so age_since_roll > cutoff_secs
        chunk.access_profile.last_rolled = far_past;
        chunk
    }

    #[tokio::test]
    async fn test_apply_aging_degrades_stale_normal_chunk() {
        let chunk = stale_chunk("a", "normal");
        let store = MockStore::new(vec![chunk.clone()]);
        let config = AgingConfig::default();
        let result = apply_aging(&store, &config).await.unwrap();
        assert_eq!(result.degraded_count, 1);
        assert_eq!(result.degraded_ids, vec![chunk.id]);
        assert_eq!(store.degraded_ids().len(), 1);
    }

    #[tokio::test]
    async fn test_apply_aging_skips_wrong_visibility() {
        // A stale chunk with visibility "deep_only" should NOT be degraded
        // because degrade_from = ["normal"] by default.
        let chunk = stale_chunk("b", "deep_only");
        let result = apply_aging_chunk(&chunk, AgingConfig::default()).await;
        assert_eq!(result.degraded_count, 0);
    }

    #[tokio::test]
    async fn test_apply_aging_skips_high_salience_chunk() {
        // A stale chunk with maximum perspectives achieves composite = 0.25
        // (perspective=1.0, w_perspective=0.25, no interaction/revision).
        // Set protection threshold just below that to verify it's protected.
        let mut chunk = stale_chunk("c", "normal");
        chunk.perspectives = (0..8).map(|i| format!("p{}", i)).collect();
        let config = AgingConfig {
            salience_protection: 0.20, // 0.25 composite > 0.20 threshold → protected
            ..AgingConfig::default()
        };
        let result = apply_aging_chunk(&chunk, config).await;
        assert_eq!(result.degraded_count, 0);
    }

    #[tokio::test]
    async fn test_apply_aging_no_stale_chunks() {
        let store = MockStore::new(vec![]);
        let config = AgingConfig::default();
        let result = apply_aging(&store, &config).await.unwrap();
        assert_eq!(result.degraded_count, 0);
        assert!(result.degraded_ids.is_empty());
    }

    #[tokio::test]
    async fn test_apply_aging_skips_chunk_with_recent_activity() {
        // hour > 0 means recent activity — should not be degraded.
        let mut chunk = stale_chunk("d", "normal");
        chunk.access_profile.hour = 1;
        let result = apply_aging_chunk(&chunk, AgingConfig::default()).await;
        assert_eq!(result.degraded_count, 0);
    }

    /// Apply aging to a single chunk and return the result.
    async fn apply_aging_chunk(
        chunk: &crate::HierarchicalChunk,
        config: AgingConfig,
    ) -> crate::aging::AgingResult {
        let store = MockStore::new(vec![chunk.clone()]);
        apply_aging(&store, &config).await.unwrap()
    }
}
