//! Agent-configurable aging rules for automatic visibility degradation.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::chunk::now_epoch_secs;
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
}

impl Default for AgingConfig {
    fn default() -> Self {
        Self {
            degrade_after_days: 30,
            degrade_to: "deep_only".to_string(),
            degrade_from: vec!["normal".to_string()],
        }
    }
}

impl AgingConfig {
    /// Load from the data directory. Returns default if no config exists.
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
    pub fn save(&self, data_dir: &Path) -> Result<()> {
        let path = data_dir.join(AGING_CONFIG_FILE);
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| crate::Error::config(format!("Failed to serialize aging config: {}", e)))?;
        std::fs::write(&path, json)?;
        Ok(())
    }

    /// Threshold in seconds.
    pub fn stale_seconds(&self) -> i64 {
        self.degrade_after_days as i64 * 86_400
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
pub async fn apply_aging<S: VectorStore>(
    store: &S,
    config: &AgingConfig,
) -> Result<AgingResult> {
    let now = now_epoch_secs();
    let cutoff_secs = config.stale_seconds();

    let stale = store.get_stale_chunks(cutoff_secs, 500).await?;

    let mut degraded_ids = Vec::new();

    for chunk in &stale {
        // Only degrade chunks whose current visibility is in the degrade_from list
        if config.degrade_from.contains(&chunk.visibility) {
            // Check that the chunk is truly stale (no month/year activity either)
            let total_recent = chunk.access_profile.hour as u32
                + chunk.access_profile.day as u32
                + chunk.access_profile.week as u32
                + chunk.access_profile.month as u32;

            let age_since_roll = now - chunk.access_profile.last_rolled;

            if total_recent == 0 && age_since_roll >= cutoff_secs {
                store
                    .update_visibility(&chunk.id, &config.degrade_to)
                    .await?;
                degraded_ids.push(chunk.id.clone());
            }
        }
    }

    Ok(AgingResult {
        degraded_count: degraded_ids.len(),
        degraded_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_aging_config_default() {
        let config = AgingConfig::default();
        assert_eq!(config.degrade_after_days, 30);
        assert_eq!(config.degrade_to, "deep_only");
        assert_eq!(config.degrade_from, vec!["normal"]);
        assert_eq!(config.stale_seconds(), 30 * 86_400);
    }

    #[test]
    fn test_aging_config_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let config = AgingConfig {
            degrade_after_days: 14,
            degrade_to: "archived".to_string(),
            degrade_from: vec!["normal".to_string(), "seasonal".to_string()],
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
}
