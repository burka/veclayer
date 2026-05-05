//! Daily background compactor — runs in long-lived MCP server processes.
//!
//! `auto_compact_if_needed` already fires from write paths (after every ~50
//! versions accumulate). That covers active sessions but misses the case where
//! a flurry of writes leaves the store fragmented and then the user goes idle.
//! This worker wakes once per `INTERVAL` and forces a compact + prune so the
//! store self-cleans even without further writes — typically firing early in
//! the morning of long-running sessions.

use std::sync::Arc;
use std::time::Duration;

use tracing::{info, warn};

use crate::store::StoreBackend;

/// How often the daily compactor wakes. 24h matches the user's expected
/// "early in the morning" cadence for sessions started the prior day.
const INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);

/// Spawn the daily background compactor. Returns the `JoinHandle`; on server
/// shutdown the task is dropped with the runtime.
pub fn spawn(store: Arc<StoreBackend>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(INTERVAL).await;
            match store.force_compact().await {
                Ok(stats) if stats.versions_removed == 0 && stats.fragments_removed == 0 => {
                    info!("Daily compact: nothing to reclaim");
                }
                Ok(stats) => {
                    info!(
                        "Daily compact: {} versions, {} fragments merged, {} bytes reclaimed",
                        stats.versions_removed, stats.fragments_removed, stats.bytes_reclaimed
                    );
                }
                Err(e) => warn!("Daily compact failed: {e}"),
            }
        }
    })
}
