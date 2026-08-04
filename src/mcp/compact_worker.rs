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

/// How often the compactor wakes. Reduced from 24h to 2h to aggressively
/// reclaim metadata during high-churn periods (space-constrained environments).
/// Once incremental-prune (task 03) lands with true index reclaim, this can
/// revert to daily since disk space will actually be freed.
const INTERVAL: Duration = Duration::from_secs(2 * 60 * 60);

/// Run one compaction pass on `store` and log the outcome.
///
/// Extracted from the `spawn` loop so the per-iteration logic can be tested
/// without advancing a 24-hour timer.
pub(crate) async fn run_once(store: &StoreBackend) {
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

/// Spawn the daily background compactor. Returns the `JoinHandle`; on server
/// shutdown the task is dropped with the runtime.
pub fn spawn(store: Arc<StoreBackend>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(INTERVAL).await;
            run_once(&store).await;
        }
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{run_once, spawn, INTERVAL};
    use crate::store::StoreBackend;

    async fn open_tmp_store() -> (tempfile::TempDir, StoreBackend) {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let store = StoreBackend::open(tmp.path(), 384, false)
            .await
            .expect("open store");
        (tmp, store)
    }

    // ── run_once — green ──────────────────────────────────────────────────────

    /// A fresh (empty) store has nothing to compact. `run_once` must complete
    /// without panicking and without returning an error to the caller.
    #[tokio::test]
    async fn run_once_on_empty_store_completes_without_panic() {
        let (_tmp, store) = open_tmp_store().await;
        // Panics / unwrap failures inside run_once would propagate here.
        run_once(&store).await;
    }

    /// Calling `run_once` multiple times on the same store is idempotent —
    /// repeated compaction of an already-compact store must not panic.
    #[tokio::test]
    async fn run_once_is_idempotent() {
        let (_tmp, store) = open_tmp_store().await;
        run_once(&store).await;
        run_once(&store).await;
        run_once(&store).await;
    }

    // ── run_once — edge ───────────────────────────────────────────────────────

    /// An `Arc`-wrapped store (the same type passed to `spawn`) must also work
    /// correctly when dereffed into `run_once`.
    #[tokio::test]
    async fn run_once_accepts_arc_deref() {
        let (_tmp, store) = open_tmp_store().await;
        let arc_store = Arc::new(store);
        run_once(&arc_store).await;
    }

    // ── spawn — lifecycle ─────────────────────────────────────────────────────

    /// `spawn` must return a valid, non-completed `JoinHandle`. The worker task
    /// starts sleeping immediately (24 h), so the handle must still be running
    /// right after spawn — aborting it must not produce a panic.
    #[tokio::test]
    async fn spawn_returns_running_handle_that_can_be_aborted() {
        let (_tmp, store) = open_tmp_store().await;
        let handle = spawn(Arc::new(store));

        // The task should still be running (it is blocked on the first sleep).
        assert!(!handle.is_finished());

        // Aborting must not panic or block.
        handle.abort();

        // After abort the handle finishes with an `Err(JoinError::Cancelled)`.
        let result = handle.await;
        assert!(
            result.is_err(),
            "aborted task should produce a JoinError, got Ok"
        );
        assert!(
            result.unwrap_err().is_cancelled(),
            "JoinError should be a cancellation"
        );
    }

    // ── INTERVAL constant ─────────────────────────────────────────────────────

    /// The compaction interval must be exactly 24 hours. If someone accidentally
    /// changes it to minutes, this test catches the regression.
    #[test]
    fn interval_is_24_hours() {
        assert_eq!(
            INTERVAL,
            std::time::Duration::from_secs(24 * 60 * 60),
            "INTERVAL must be 24 h"
        );
    }
}
