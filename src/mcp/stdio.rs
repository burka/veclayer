//! MCP stdio transport — rmcp-based.

use std::sync::Arc;

use rmcp::{transport::stdio, ServiceExt};
use tracing::info;

use crate::auth::capability::Capability;
use crate::blob_store::BlobStore;
use crate::embedder;
use crate::store::StoreBackend;
use crate::{Config, Embedder, Result};

use super::core::ServerCore;
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
    let core = Arc::new(ServerCore {
        store,
        embedder,
        embedder_config: config.embedder.clone(),
        blob_store,
        data_dir: config.data_dir.clone(),
    });

    let handler = McpHandler::new(
        core,
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
        model::{
            ErrorCode, ErrorData, JsonRpcMessage, NumberOrString, ServerNotification,
            ServerRequest, ServerResult,
        },
        transport::{async_rw::AsyncRwTransport, Transport},
        RoleServer,
    };
    use tokio::io::{AsyncBufReadExt, BufReader};

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

    /// Build a loopback `AsyncRwTransport` (server role) backed by a
    /// `tokio::io::duplex` pipe.  Returns the transport together with the
    /// client-side `DuplexStream` so callers can inspect outbound bytes.
    fn duplex_server_transport() -> (
        AsyncRwTransport<
            RoleServer,
            tokio::io::ReadHalf<tokio::io::DuplexStream>,
            tokio::io::WriteHalf<tokio::io::DuplexStream>,
        >,
        tokio::io::DuplexStream,
    ) {
        let (client_side, server_side) = tokio::io::duplex(4096);
        let (server_read, server_write) = tokio::io::split(server_side);
        (
            AsyncRwTransport::new(server_read, server_write),
            client_side,
        )
    }

    /// Shorthand for building a `TxJsonRpcMessage<RoleServer>` error frame —
    /// the simplest possible outbound message (no tool-specific types needed).
    fn make_error_message(
        id: i64,
    ) -> JsonRpcMessage<ServerRequest, ServerResult, ServerNotification> {
        JsonRpcMessage::error(
            ErrorData::new(ErrorCode(0), "test error", None),
            Some(NumberOrString::Number(id)),
        )
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

    /// A malformed first line followed by a valid second line: rmcp's codec
    /// skips the unparseable line and continues, so the first `receive()`
    /// yields the *next* valid message rather than stopping at the error.
    /// This documents the ACTUAL behaviour as of rmcp 1.7 (in earlier versions
    /// the bad line surfaced as a terminal `None`; the codec now resyncs on the
    /// newline boundary and recovers, which is strictly more robust).
    #[tokio::test]
    async fn bad_then_good_line_transport_skips_to_next_valid() {
        let data: &'static [u8] = b"not-json\n{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"ping\"}\n";
        let mut transport = server_transport_from_bytes(data);
        // First receive: the malformed line is skipped and the following valid
        // request is returned.
        let first = transport.receive().await;
        assert!(
            first.is_some(),
            "rmcp 1.7 skips the bad line and returns the next valid message"
        );
        // The recovered message is the ping request with id 4.
        let msg = first.unwrap();
        let dbg = format!("{msg:?}");
        assert!(
            dbg.contains("PingRequest") && dbg.contains("Number(4)"),
            "recovered message should be the valid ping request with id 4, got: {dbg}"
        );
        // After the single valid message the stream is exhausted (EOF) → None.
        let second = transport.receive().await;
        assert!(
            second.is_none(),
            "after the only valid message the transport reaches EOF and yields None"
        );
    }

    // -----------------------------------------------------------------------
    // Edge: decode_eof — valid JSON without trailing newline
    // -----------------------------------------------------------------------

    /// A complete, valid JSON-RPC message with no trailing `\n` is decoded via
    /// the `decode_eof` path and yields `Some(msg)`.
    ///
    /// The codec's `decode_eof` implementation flushes any remaining buffer
    /// content as a final frame when the underlying I/O reaches EOF.  This is
    /// the path hit when a client sends a message with `\r` only (no `\n`) or
    /// when the write end closes without the final newline.
    #[tokio::test]
    async fn valid_json_at_eof_without_newline_is_parsed() {
        // Valid ping with no trailing newline — must go through decode_eof.
        let line = b"{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"ping\"}";
        let mut transport = server_transport_from_bytes(line);
        let msg = transport.receive().await;
        assert!(
            msg.is_some(),
            "complete JSON at EOF (no trailing newline) must yield Some via decode_eof"
        );
    }

    // -----------------------------------------------------------------------
    // Edge: whitespace-only line
    // -----------------------------------------------------------------------

    /// A line containing only ASCII spaces is not valid JSON and must yield
    /// `None` — the codec error is swallowed, same as blank-line behaviour.
    #[tokio::test]
    async fn whitespace_only_line_yields_none() {
        let mut transport = server_transport_from_bytes(b"   \n");
        let result = transport.receive().await;
        assert!(
            result.is_none(),
            "whitespace-only line should yield None (invalid JSON)"
        );
    }

    // -----------------------------------------------------------------------
    // Compatibility: non-standard notifications are silently skipped
    // -----------------------------------------------------------------------

    /// A non-standard `notifications/stderr` message (sent by some LSP-aware
    /// clients) is silently skipped by the compatibility layer.  The transport
    /// transparently consumes it and delivers the next valid message — it does
    /// NOT surface an error or return `None` for the next real message.
    #[tokio::test]
    async fn non_standard_notification_is_transparently_skipped() {
        // A non-standard notification followed immediately by a valid ping.
        let data: &'static [u8] =
            b"{\"method\":\"notifications/stderr\",\"params\":{\"content\":\"log line\"}}\n\
             {\"jsonrpc\":\"2.0\",\"id\":6,\"method\":\"ping\"}\n";
        let mut transport = server_transport_from_bytes(data);
        // The compatibility layer skips the non-standard frame; `decode` returns
        // `Ok(None)` which tells FramedRead to keep reading — the very next
        // call to `decode` finds the valid ping and returns it.  So the first
        // and only `receive()` yields the ping, not None.
        let msg = transport.receive().await.expect(
            "non-standard notification must be skipped; the following valid ping must be returned",
        );
        // Confirm it is the ping (id=6) and not a remnant of the skipped frame.
        let json = serde_json::to_value(&msg).expect("received message must serialize");
        assert_eq!(
            json["id"], 6,
            "the frame delivered after the skipped notification must be the ping with id=6"
        );
    }

    // -----------------------------------------------------------------------
    // Send path: wire serialization
    // -----------------------------------------------------------------------

    /// `send()` writes a newline-framed JSON object to the underlying writer.
    /// We verify two invariants:
    ///   1. The written bytes are valid JSON (parse without error).
    ///   2. The frame is terminated with `\n`.
    #[tokio::test]
    async fn send_writes_newline_framed_json() {
        let (mut transport, client) = duplex_server_transport();

        transport
            .send(make_error_message(42))
            .await
            .expect("send must not fail on an open transport");

        // Read exactly one newline-framed frame. We deliberately do NOT rely on
        // closing the transport to signal EOF: the transport still owns the
        // duplex read half, so the client side never sees EOF and read_to_end
        // would block forever. read_until stops at the frame's terminator.
        let mut reader = BufReader::new(client);
        let mut buf = Vec::new();
        reader
            .read_until(b'\n', &mut buf)
            .await
            .expect("reading from duplex client side must not fail");

        assert!(!buf.is_empty(), "send must produce non-empty output");
        assert_eq!(
            *buf.last().unwrap(),
            b'\n',
            "codec must terminate each frame with a newline"
        );

        // Strip the trailing newline and parse as JSON.
        let json_bytes = &buf[..buf.len() - 1];
        let parsed: serde_json::Value =
            serde_json::from_slice(json_bytes).expect("framed bytes must be valid JSON");
        assert_eq!(
            parsed["id"], 42,
            "serialized message must preserve the request id"
        );
    }

    // -----------------------------------------------------------------------
    // Close path: send after close returns NotConnected
    // -----------------------------------------------------------------------

    /// After `close()` is called the writer is dropped.  A subsequent `send()`
    /// must return `Err` with `ErrorKind::NotConnected` — it must not panic.
    #[tokio::test]
    async fn send_after_close_returns_error() {
        let (mut transport, _client) = duplex_server_transport();

        transport.close().await.expect("close must not fail");

        let result = transport.send(make_error_message(1)).await;
        assert!(
            result.is_err(),
            "send after close must return Err (transport is closed)"
        );
        assert_eq!(
            result.unwrap_err().kind(),
            std::io::ErrorKind::NotConnected,
            "error kind must be NotConnected"
        );
    }
}
