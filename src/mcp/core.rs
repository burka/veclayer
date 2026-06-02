//! Session-invariant server dependencies shared between [`McpHandler`] and
//! [`AppState`](super::http::AppState).
//!
//! `ServerCore` is created once at server startup and wrapped in `Arc` so both
//! the MCP handler and the HTTP state can hold a cheap clone without duplicating
//! the underlying allocations.

use std::path::PathBuf;
use std::sync::Arc;

use crate::blob_store::BlobStore;
use crate::store::StoreBackend;
use crate::Embedder;

/// Session-invariant dependencies shared by every MCP session and every HTTP
/// request handler.
///
/// Fields here are set at startup and never mutated afterwards.  Per-session
/// fields (project, branch, git_store, push_mode, …) stay on the outer type.
pub struct ServerCore {
    pub store: Arc<StoreBackend>,
    pub embedder: Arc<dyn Embedder + Send + Sync>,
    pub embedder_config: crate::config::EmbedderConfig,
    pub blob_store: Arc<BlobStore>,
    pub data_dir: PathBuf,
}
