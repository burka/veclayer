//! MCP stdio transport — rmcp-based.

use std::sync::Arc;

use rmcp::{transport::stdio, ServiceExt};
use tracing::info;

use crate::auth::capability::Capability;
use crate::blob_store::BlobStore;
use crate::embedder;
use crate::store::StoreBackend;
use crate::{Config, Embedder, Result};

use super::handler::McpHandler;

/// Run the MCP server on stdio.
pub async fn run_stdio(config: Config) -> Result<()> {
    info!("Starting MCP stdio server...");

    // Build shared state (same initialization as before)
    let embedder: Arc<dyn Embedder + Send + Sync> =
        Arc::from(embedder::from_config(&config.embedder)?);
    let dimension = embedder.dimension();
    let store = StoreBackend::open(&config.data_dir, dimension, config.read_only).await?;
    let store = Arc::new(store);
    let blob_store = BlobStore::open(&config.data_dir)?;
    let blob_store = Arc::new(blob_store);

    // Spawn background embedding worker
    if !config.read_only {
        let _worker = super::embed_worker::spawn(
            Arc::clone(&store),
            Arc::clone(&embedder),
            Arc::clone(&blob_store),
        );
        let _compact = super::compact_worker::spawn(Arc::clone(&store));
    }

    let instructions = super::compute_instructions(
        store.as_ref(),
        &config.data_dir,
        config.project.as_deref(),
        config.branch.as_deref(),
        None,
    )
    .await;

    let push_mode = config.push_mode;
    let git_store = if push_mode.uses_git() {
        super::open_git_store(&config)
    } else {
        None
    };

    // Create handler and serve via rmcp stdio transport.
    // Stdio transport is trusted (local process), so it always gets Admin capability.
    let handler = McpHandler::new(
        store,
        embedder,
        config.embedder.clone(),
        blob_store,
        config.data_dir.clone(),
        config.project.clone(),
        config.branch.clone(),
        instructions,
        Capability::Admin,
        git_store,
        push_mode,
    );

    let service = handler
        .serve(stdio())
        .await
        .map_err(|e| crate::Error::InvalidOperation(format!("MCP stdio error: {}", e)))?;

    // Block until the client disconnects
    service
        .waiting()
        .await
        .map_err(|e| crate::Error::InvalidOperation(format!("MCP stdio error: {}", e)))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// `run_stdio` is a top-level entry point that requires a fully-wired Config,
// embedder, and store — it is not unit-testable in isolation.
//
// What IS testable is the framing layer underneath: the stdio transport
// delegates to `rmcp::transport::async_rw::AsyncRwTransport`, which wraps
// any `AsyncRead + AsyncWrite` pair with `JsonRpcMessageCodec` (newline-
// framed JSON-RPC 2.0).  Because that type is public and parameterised over
// its reader/writer, we can drive it with in-memory buffers, exercising the
// same codec path that `run_stdio` uses at runtime.
//
// Feature gate: these tests require the `mcp` cargo feature (which pulls in
// `rmcp` with `transport-io` → `transport-async-rw`).
#[cfg(test)]
mod tests {
    use rmcp::{
        transport::{async_rw::AsyncRwTransport, Transport},
        RoleServer,
    };
    use tokio::io::BufReader;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build an `AsyncRwTransport` (server role) over an in-memory byte slice.
    ///
    /// A `tokio::io::sink()` is used as the write half — we only need to
    /// exercise the *read* path for receive-side tests.
    fn server_transport_from_bytes(
        data: &'static [u8],
    ) -> AsyncRwTransport<RoleServer, BufReader<&'static [u8]>, tokio::io::Sink> {
        AsyncRwTransport::new(BufReader::new(data), tokio::io::sink())
    }

    // -----------------------------------------------------------------------
    // Green: well-formed messages
    // -----------------------------------------------------------------------

    /// A single valid JSON-RPC `initialize` request (the very first message a
    /// client sends over an MCP stdio connection) is decoded correctly.
    #[tokio::test]
    async fn receive_single_valid_message() {
        let line = b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\"clientInfo\":{\"name\":\"test\",\"version\":\"0.1\"}}}\n";
        let mut transport = server_transport_from_bytes(line);
        let msg = transport.receive().await;
        assert!(
            msg.is_some(),
            "expected a parsed message, got None (EOF or parse error)"
        );
    }

    /// Multiple consecutive newline-framed messages are all decoded and
    /// delivered in order.
    #[tokio::test]
    async fn receive_sequence_of_messages() {
        // Two ping requests back-to-back.
        let data: &'static [u8] = b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n\
             {\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}\n";
        let mut transport = server_transport_from_bytes(data);

        let first = transport.receive().await;
        assert!(first.is_some(), "first ping should parse");

        let second = transport.receive().await;
        assert!(second.is_some(), "second ping should parse");
    }

    // -----------------------------------------------------------------------
    // Edge: EOF / empty input
    // -----------------------------------------------------------------------

    /// Empty input (immediate EOF) terminates cleanly — `receive()` returns
    /// `None` without panicking.
    #[tokio::test]
    async fn empty_input_returns_none_on_receive() {
        let mut transport = server_transport_from_bytes(b"");
        let result = transport.receive().await;
        assert!(
            result.is_none(),
            "EOF on empty input should yield None, not a panic"
        );
    }

    /// A stream that contains only a bare newline (blank line) does not crash
    /// and yields `None` (the codec treats a blank line as empty, not a
    /// message, and EOF follows with no data left).
    #[tokio::test]
    async fn blank_line_only_yields_none() {
        // A single blank line followed by EOF — no parseable message.
        let mut transport = server_transport_from_bytes(b"\n");
        // The codec will attempt to parse an empty string → serde error → None
        // (errors are swallowed by AsyncRwTransport::receive which logs and
        // returns None, consistent with the `inspect_err(…).ok()` pattern in
        // the rmcp source).
        let result = transport.receive().await;
        // A blank line carries no parseable message: the empty frame fails to
        // deserialize, the error is swallowed by AsyncRwTransport::receive, and
        // the stream yields None rather than panicking.
        assert!(
            result.is_none(),
            "blank line must yield None, not a message"
        );
    }

    /// A CRLF-terminated line (`\r\n`) is handled identically to `\n`.  The
    /// codec strips the trailing `\r` before parsing.
    #[tokio::test]
    async fn crlf_terminated_line_is_parsed() {
        let line = b"{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"ping\"}\r\n";
        let mut transport = server_transport_from_bytes(line);
        let msg = transport.receive().await;
        assert!(
            msg.is_some(),
            "CRLF-terminated valid JSON should parse as a message"
        );
    }

    // -----------------------------------------------------------------------
    // Error: malformed input
    // -----------------------------------------------------------------------

    /// A line of non-JSON garbage is not a panic — the codec logs the error
    /// and `receive()` returns `None` (the rmcp `AsyncRwTransport::receive`
    /// implementation swallows decode errors via `.ok()`).
    #[tokio::test]
    async fn malformed_line_is_not_a_panic() {
        let mut transport = server_transport_from_bytes(b"this is not json at all\n");
        let result = transport.receive().await;
        // Malformed input → decode error → None (not a panic, not an unwrap failure)
        assert!(
            result.is_none(),
            "malformed line should yield None, not a panic"
        );
    }

    /// Truncated / partial JSON followed by EOF does not panic.  The
    /// `decode_eof` path of the codec attempts to parse the remaining bytes
    /// as a final frame; a parse failure is swallowed the same way.
    #[tokio::test]
    async fn truncated_json_at_eof_is_not_a_panic() {
        let mut transport = server_transport_from_bytes(b"{\"jsonrpc\":\"2.0\",\"id\":99");
        let result = transport.receive().await;
        // No newline → decode_eof path → serde error → None
        assert!(
            result.is_none(),
            "truncated JSON at EOF should yield None, not a panic"
        );
    }

    /// A malformed first line followed by a valid second line: the transport
    /// stops at the first error (returns None after the bad line), consistent
    /// with rmcp's `.ok()` swallowing strategy — the transport does not
    /// silently skip and continue.  This test documents the ACTUAL behaviour.
    #[tokio::test]
    async fn bad_then_good_line_transport_stops_at_error() {
        let data: &'static [u8] = b"not-json\n{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"ping\"}\n";
        let mut transport = server_transport_from_bytes(data);
        // First receive: hits the malformed line → None (error swallowed)
        let first = transport.receive().await;
        assert!(
            first.is_none(),
            "bad first line should yield None (error swallowed by rmcp)"
        );
        // After the bad-line None, the transport is considered closed; a
        // second receive also returns None.
        let second = transport.receive().await;
        assert!(
            second.is_none(),
            "after an error/None the transport yields None again"
        );
    }
}
