//! Hourly background compactor — runs in long-lived MCP server processes.
//!
//! Write paths only prune old versions (`auto_prune_if_needed`): they run in
//! short-lived processes such as the `observe` hook, where a killed fragment
//! rewrite leaks its staging file. Fragment compaction therefore happens only
//! in long-lived or user-invoked contexts: `serve` startup, this worker, and
//! `veclayer prune`. This worker wakes once per `INTERVAL` and forces a
//! compact + prune so the store stays compacted during long sessions and
//! self-cleans even without further writes.

use std::sync::Arc;
use std::time::Duration;

use tracing::{info, warn};

use crate::store::StoreBackend;

/// How often the compactor wakes. `spawn` sleeps before the first pass (the
/// `serve` startup compaction covers t=0), so the interval must be well below
/// a typical MCP server lifetime or the server never compacts again. A pass
/// on an already-compact store is cheap.
const INTERVAL: Duration = Duration::from_secs(60 * 60);

/// Run one compaction pass on `store` and log the outcome.
///
/// Extracted from the `spawn` loop so the per-iteration logic can be tested
/// without advancing the interval timer.
pub(crate) async fn run_once(store: &StoreBackend) {
    match store.force_compact().await {
        Ok(stats)
            if stats.versions_removed == 0
                && stats.fragments_removed == 0
                && stats.leaked_files_removed == 0 =>
        {
            info!("Periodic compact: nothing to reclaim");
        }
        Ok(stats) => {
            info!(
                "Periodic compact: {} versions, {} fragments merged, {} bytes reclaimed, {} leaked upload files ({} bytes) removed",
                stats.versions_removed,
                stats.fragments_removed,
                stats.bytes_reclaimed,
                stats.leaked_files_removed,
                stats.leaked_bytes_reclaimed
            );
        }
        Err(e) => warn!("Periodic compact failed: {e}"),
    }
}

/// Spawn the hourly background compactor. Returns the `JoinHandle`; on server
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
    /// starts sleeping immediately (1 h), so the handle must still be running
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

    /// The compaction interval must be exactly 1 hour: short enough that MCP
    /// servers living less than a day still compact after startup, long enough
    /// that a fragment rewrite is not started constantly.
    #[test]
    fn interval_is_1_hour() {
        assert_eq!(
            INTERVAL,
            std::time::Duration::from_secs(60 * 60),
            "INTERVAL must be 1 h"
        );
    }
}
