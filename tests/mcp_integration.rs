#![cfg(feature = "http")]

use std::sync::Arc;
use tempfile::TempDir;
use tokio::net::TcpListener;
use veclayer::mcp::http::{build_app, AppState};
use veclayer::{BlobStore, StoreBackend};

// ── Stub embedder ─────────────────────────────────────────────────────────────
//
// Returns zero vectors immediately so the background embed worker completes
// its first pass without any network call.  This eliminates the flaky
// sleep-poll loop in store-and-recall roundtrip tests — the embed is
// deterministic and fast, so a single yield_now() is sufficient.

struct StubEmbedder;

impl veclayer::Embedder for StubEmbedder {
    fn embed<'a>(
        &'a self,
        texts: &'a [&'a str],
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = veclayer::Result<Vec<Vec<f32>>>> + Send + 'a>,
    > {
        // Return a constant non-zero unit vector.  All documents get the same
        // embedding so cosine similarity is 1.0 for every query — every stored
        // entry is returned for every query.  This makes the roundtrip test
        // deterministic: after the worker runs, recall always finds the entry.
        let result: Vec<Vec<f32>> = texts.iter().map(|_| vec![1.0f32; 384]).collect();
        Box::pin(async move { Ok(result) })
    }

    fn dimension(&self) -> usize {
        384
    }

    fn name(&self) -> &str {
        "stub"
    }
}

/// Spin up a real HTTP server on a random port, return the base URL.
async fn spawn_server() -> (String, TempDir) {
    spawn_server_with(None, None).await
}

/// Spin up a server with optional project/branch scope.
async fn spawn_server_with(project: Option<String>, branch: Option<String>) -> (String, TempDir) {
    let (base, tmp, _) = spawn_server_with_state(project, branch).await;
    (base, tmp)
}

/// Like `spawn_server_with` but also returns the `AppState` so callers can
/// spawn the background embed worker at a controlled point in the test.
async fn spawn_server_with_state(
    project: Option<String>,
    branch: Option<String>,
) -> (String, TempDir, AppState) {
    let tmp = TempDir::new().unwrap();
    // Use the stub embedder: zero-vector, instant, no network.
    let embedder: Arc<dyn veclayer::Embedder + Send + Sync> = Arc::new(StubEmbedder);
    let dim = embedder.dimension();
    let store = Arc::new(StoreBackend::open(tmp.path(), dim, false).await.unwrap());
    let blob_store = Arc::new(BlobStore::open(tmp.path()).unwrap());

    let state = AppState {
        core: Arc::new(veclayer::mcp::core::ServerCore {
            store,
            embedder,
            embedder_config: veclayer::config::EmbedderConfig::default(),
            blob_store,
            data_dir: tmp.path().to_path_buf(),
        }),
        project,
        branch,
        auth: None,
        git_store: None,
        push_mode: veclayer::git::branch_config::PushMode::Off,
        // Use the static instructions so tests that check non-empty instructions pass.
        instructions: Arc::new(veclayer::mcp::MCP_INSTRUCTIONS.to_string()),
    };

    let returned_state = state.clone();
    let app = build_app(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    (format!("http://127.0.0.1:{port}"), tmp, returned_state)
}

/// POST a raw JSON body to `/mcp` and return `(status, headers, body_text)`.
async fn post_mcp_raw(
    client: &reqwest::Client,
    base: &str,
    session_id: Option<&str>,
    body: serde_json::Value,
) -> (reqwest::StatusCode, reqwest::header::HeaderMap, String) {
    let mut req = client
        .post(format!("{base}/mcp"))
        .header("content-type", "application/json")
        .header("accept", "application/json, text/event-stream")
        .json(&body);

    if let Some(id) = session_id {
        req = req.header("mcp-session-id", id);
    }

    let resp = req.send().await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let text = resp.text().await.unwrap();
    (status, headers, text)
}

/// POST a JSON-RPC message to `/mcp` and return `(status, headers, parsed_json)`.
///
/// The Streamable HTTP transport responds with SSE. The body contains one or
/// more `\n\n`-separated events; each event may have a `data:` line with
/// a JSON-RPC object. Returns the first parseable JSON-RPC object found.
async fn post_mcp(
    client: &reqwest::Client,
    base: &str,
    session_id: Option<&str>,
    body: serde_json::Value,
) -> (
    reqwest::StatusCode,
    reqwest::header::HeaderMap,
    serde_json::Value,
) {
    let (status, headers, text) = post_mcp_raw(client, base, session_id, body).await;
    let json = parse_mcp_response(&text);
    (status, headers, json)
}

/// Extract the first JSON-RPC object from an SSE or plain-JSON body.
///
/// Stateful mode returns `text/event-stream`. The body looks like:
///
/// ```text
/// id: 0\nretry: 3000\ndata:\n\ndata: {…}\n\n
/// ```
///
/// Scan for `data:` lines and return the first one that parses as a JSON
/// object. Falls back to treating the whole body as JSON when no `data:`
/// line is found (covers the `json_response: true` stateless mode).
fn parse_mcp_response(body: &str) -> serde_json::Value {
    for line in body.lines() {
        let trimmed = line.trim();
        if let Some(payload) = trimmed.strip_prefix("data:") {
            let payload = payload.trim();
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) {
                if v.is_object() {
                    return v;
                }
            }
        }
    }

    // Fall back: whole body might already be JSON (stateless json_response mode).
    serde_json::from_str(body).unwrap_or(serde_json::Value::Null)
}

/// Send `initialize` + `notifications/initialized`, return `(session_id, init_result_json)`.
///
/// After `initialize`, the MCP protocol requires the client to send
/// `notifications/initialized` before the server will handle any other
/// requests. Failing to do so causes the server to terminate the session.
async fn initialize_session(client: &reqwest::Client, base: &str) -> (String, serde_json::Value) {
    // Step 1: send `initialize` request.
    let (status, headers, json) = post_mcp(
        client,
        base,
        None,
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0.1"}
            }
        }),
    )
    .await;

    assert_eq!(status, 200, "initialize must return 200");

    let session_id = headers
        .get("mcp-session-id")
        .expect("Mcp-Session-Id header must be present after initialize")
        .to_str()
        .expect("Mcp-Session-Id header must be valid UTF-8")
        .to_owned();

    // Step 2: send the required `notifications/initialized` notification.
    // This is a fire-and-forget notification; the server responds with 202 Accepted.
    let (notif_status, _, _) = post_mcp_raw(
        client,
        base,
        Some(&session_id),
        serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }),
    )
    .await;
    assert_eq!(
        notif_status, 202,
        "notifications/initialized must return 202 Accepted"
    );

    (session_id, json)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial_test::serial]
async fn mcp_initialize_returns_server_info() {
    let (base, _tmp) = spawn_server().await;
    let client = reqwest::Client::new();

    let (session_id, json) = initialize_session(&client, &base).await;

    assert!(!session_id.is_empty(), "session ID must not be empty");

    let result = &json["result"];
    assert!(result.is_object(), "response must have a 'result' field");

    let server_name = result["serverInfo"]["name"]
        .as_str()
        .expect("serverInfo.name must be a string");
    assert_eq!(server_name, "veclayer");

    let instructions = result["instructions"]
        .as_str()
        .expect("result.instructions must be a string");
    assert!(!instructions.is_empty(), "instructions must not be empty");
    assert!(
        instructions.contains("VecLayer"),
        "instructions must mention VecLayer"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial_test::serial]
async fn mcp_tools_list_returns_four_tools() {
    let (base, _tmp) = spawn_server().await;
    let client = reqwest::Client::new();

    let (session_id, _) = initialize_session(&client, &base).await;

    let (status, _, json) = post_mcp(
        &client,
        &base,
        Some(&session_id),
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }),
    )
    .await;

    assert_eq!(status, 200);

    let tools = json["result"]["tools"]
        .as_array()
        .expect("result.tools must be an array");

    assert_eq!(
        tools.len(),
        4,
        "expected exactly 4 tools, got {}",
        tools.len()
    );

    let names: Vec<&str> = tools
        .iter()
        .map(|t| t["name"].as_str().expect("tool must have a name"))
        .collect();

    let expected = ["recall", "focus", "store", "think"];
    for name in expected {
        assert!(
            names.contains(&name),
            "tool '{name}' not found in list: {names:?}"
        );
    }

    // `share` must NOT be advertised until UCAN capability-token signing is
    // implemented — advertising a tool that always errors misleads agents.
    assert!(
        !names.contains(&"share"),
        "`share` must not be advertised until UCAN signing lands: {names:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial_test::serial]
async fn mcp_store_and_recall_roundtrip() {
    // Use spawn_server_with_state so we can spawn the embed worker after
    // storing, avoiding the 10-second idle sleep that fires when the worker
    // starts before any entries exist.
    let (base, _tmp, state) = spawn_server_with_state(None, None).await;
    let client = reqwest::Client::new();

    let (session_id, _) = initialize_session(&client, &base).await;

    // Store a piece of knowledge — embedding is now deferred.
    let (status, _, store_json) = post_mcp(
        &client,
        &base,
        Some(&session_id),
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "store",
                "arguments": {
                    "content": "Rust is a systems programming language",
                    "heading": "Rust"
                }
            }
        }),
    )
    .await;

    assert_eq!(status, 200);

    let is_error = store_json["result"]["isError"].as_bool().unwrap_or(false);
    assert!(!is_error, "store must not return an error: {store_json}");

    let store_text = store_json["result"]["content"][0]["text"]
        .as_str()
        .expect("store result must have text content");
    assert!(
        store_text.contains("Stored. ID:"),
        "store result must contain 'Stored. ID:', got: {store_text}"
    );

    // Spawn the embed worker now that there is a pending entry to process.
    // The StubEmbedder returns zero vectors immediately (no network), so the
    // worker's first pass completes synchronously and the entry becomes
    // searchable.  A few yields are sufficient for the task to run — no
    // 500 ms sleep poll loop required.
    let _worker = veclayer::mcp::embed_worker::spawn(
        Arc::clone(&state.core.store),
        Arc::clone(&state.core.embedder),
        Arc::clone(&state.core.blob_store),
    );

    // Yield to the tokio executor until the worker has processed the entry.
    // With a stub embedder this completes in the very first iteration, so a
    // tight bounded loop with minimal delay is deterministic and fast.
    let mut recall_text = String::new();
    for id in 4u32..=14 {
        tokio::task::yield_now().await;

        let (status, _, recall_json) = post_mcp(
            &client,
            &base,
            Some(&session_id),
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": "tools/call",
                "params": {
                    "name": "recall",
                    "arguments": {"query": "systems programming"}
                }
            }),
        )
        .await;

        assert_eq!(status, 200);

        recall_text = recall_json["result"]["content"][0]["text"]
            .as_str()
            .expect("recall result must have text content")
            .to_owned();

        if recall_text.contains("Rust") {
            break;
        }

        // One short sleep to let the worker task run if yield_now wasn't enough.
        if id < 14 {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }
    assert!(
        recall_text.contains("Rust"),
        "recall result must mention 'Rust', got: {recall_text}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial_test::serial]
async fn mcp_session_id_required_after_init() {
    let (base, _tmp) = spawn_server().await;
    let client = reqwest::Client::new();

    // Establish a session so the server is in stateful mode.
    let (_session_id, _) = initialize_session(&client, &base).await;

    // Send `tools/list` without the session ID header.
    // Without a session ID on a non-initialize request, the server returns
    // a non-200 error response (the MCP spec allows 400 Bad Request in this case).
    let (status, _, _) = post_mcp_raw(
        &client,
        &base,
        None,
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }),
    )
    .await;

    assert_ne!(
        status,
        reqwest::StatusCode::OK,
        "tools/list without session ID must not return 200"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial_test::serial]
async fn mcp_tool_descriptions_include_project_context() {
    let (base, _tmp) = spawn_server_with(Some("my-app".into()), Some("feature/auth".into())).await;
    let client = reqwest::Client::new();

    let (session_id, _) = initialize_session(&client, &base).await;

    let (status, _, json) = post_mcp(
        &client,
        &base,
        Some(&session_id),
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }),
    )
    .await;

    assert_eq!(status, 200);

    let tools = json["result"]["tools"]
        .as_array()
        .expect("result.tools must be an array");

    let recall = tools
        .iter()
        .find(|t| t["name"] == "recall")
        .expect("recall tool must exist");
    let recall_desc = recall["description"]
        .as_str()
        .expect("recall must have a description");
    assert!(
        recall_desc.contains("my-app"),
        "recall description must mention project name, got: {recall_desc}"
    );
    assert!(
        recall_desc.contains("feature/auth"),
        "recall description must mention branch, got: {recall_desc}"
    );

    let store = tools
        .iter()
        .find(|t| t["name"] == "store")
        .expect("store tool must exist");
    let store_desc = store["description"]
        .as_str()
        .expect("store must have a description");
    assert!(
        store_desc.contains("my-app"),
        "store description must mention project name, got: {store_desc}"
    );
}

// ---------------------------------------------------------------------------
// Resource tests
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial_test::serial]
async fn mcp_resources_list_returns_static_resources() {
    let (base, _tmp) = spawn_server().await;
    let client = reqwest::Client::new();

    let (session_id, init_json) = initialize_session(&client, &base).await;

    // Verify the server advertises resource capability
    let capabilities = &init_json["result"]["capabilities"];
    assert!(
        capabilities.get("resources").is_some(),
        "server must advertise resources capability"
    );

    let (status, _, json) = post_mcp(
        &client,
        &base,
        Some(&session_id),
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "resources/list",
            "params": {}
        }),
    )
    .await;

    assert_eq!(status, 200);

    let resources = json["result"]["resources"]
        .as_array()
        .expect("result.resources must be an array");

    assert_eq!(
        resources.len(),
        5,
        "expected 5 static resources, got {}",
        resources.len()
    );

    let uris: Vec<&str> = resources
        .iter()
        .map(|r| r["uri"].as_str().expect("resource must have a uri"))
        .collect();

    let expected = [
        "veclayer://status",
        "veclayer://perspectives",
        "veclayer://hot",
        "veclayer://recent",
        "veclayer://identity",
    ];
    for uri in expected {
        assert!(
            uris.contains(&uri),
            "resource URI '{uri}' not found in list: {uris:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial_test::serial]
async fn mcp_resource_templates_list() {
    let (base, _tmp) = spawn_server().await;
    let client = reqwest::Client::new();

    let (session_id, _) = initialize_session(&client, &base).await;

    let (status, _, json) = post_mcp(
        &client,
        &base,
        Some(&session_id),
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "resources/templates/list",
            "params": {}
        }),
    )
    .await;

    assert_eq!(status, 200);

    let templates = json["result"]["resourceTemplates"]
        .as_array()
        .expect("result.resourceTemplates must be an array");

    assert_eq!(
        templates.len(),
        2,
        "expected 2 resource templates, got {}",
        templates.len()
    );

    let uris: Vec<&str> = templates
        .iter()
        .map(|t| {
            t["uriTemplate"]
                .as_str()
                .expect("template must have a uriTemplate")
        })
        .collect();

    assert!(uris.contains(&"veclayer://perspectives/{perspective_id}"));
    assert!(uris.contains(&"veclayer://entries/{entry_id}"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial_test::serial]
async fn mcp_read_status_resource() {
    let (base, _tmp) = spawn_server().await;
    let client = reqwest::Client::new();

    let (session_id, _) = initialize_session(&client, &base).await;

    let (status, _, json) = post_mcp(
        &client,
        &base,
        Some(&session_id),
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 12,
            "method": "resources/read",
            "params": {"uri": "veclayer://status"}
        }),
    )
    .await;

    assert_eq!(status, 200);

    let contents = json["result"]["contents"]
        .as_array()
        .expect("result.contents must be an array");

    assert!(!contents.is_empty(), "contents must not be empty");

    let text = contents[0]["text"]
        .as_str()
        .expect("content must have text");

    assert!(
        text.contains("Store Status"),
        "status resource must contain 'Store Status', got: {text}"
    );
    assert!(
        text.contains("Total entries"),
        "status resource must contain 'Total entries', got: {text}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial_test::serial]
async fn mcp_read_entry_after_store() {
    let (base, _tmp) = spawn_server().await;
    let client = reqwest::Client::new();

    let (session_id, _) = initialize_session(&client, &base).await;

    // Store an entry first
    let (status, _, store_json) = post_mcp(
        &client,
        &base,
        Some(&session_id),
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 13,
            "method": "tools/call",
            "params": {
                "name": "store",
                "arguments": {
                    "content": "MCP resources expose knowledge as browsable data",
                    "heading": "MCP Resources"
                }
            }
        }),
    )
    .await;

    assert_eq!(status, 200);

    let store_text = store_json["result"]["content"][0]["text"]
        .as_str()
        .expect("store result must have text content");

    // The format is "Stored. ID: <short_id>. Embedding is being computed…"
    // Extract the first token after the "Stored. ID: " prefix.
    let stored_id = store_text
        .strip_prefix("Stored. ID: ")
        .unwrap_or(store_text)
        .split_whitespace()
        .next()
        .map(|s| s.trim_end_matches('.'))
        .unwrap_or(store_text);

    // Read the entry via resources/read
    let entry_uri = format!("veclayer://entries/{stored_id}");
    let (status, _, json) = post_mcp(
        &client,
        &base,
        Some(&session_id),
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 14,
            "method": "resources/read",
            "params": {"uri": entry_uri}
        }),
    )
    .await;

    assert_eq!(status, 200);

    let result = &json["result"];
    assert!(
        result.is_object(),
        "resources/read must return a result, got: {json}"
    );

    let contents = result["contents"]
        .as_array()
        .unwrap_or_else(|| panic!("result.contents must be an array, got: {json}"));

    let text = contents[0]["text"]
        .as_str()
        .expect("content must have text");

    assert!(
        text.contains("MCP Resources"),
        "entry resource must contain heading 'MCP Resources', got: {text}"
    );
    assert!(
        text.contains("browsable data"),
        "entry resource must contain stored content, got: {text}"
    );
}
