use std::sync::Arc;
use tempfile::TempDir;
use tokio::net::TcpListener;
use veclayer::mcp::http::{build_app, AppState};
use veclayer::{BlobStore, StoreBackend};

/// Spin up a real HTTP server on a random port, return the base URL.
async fn spawn_server() -> (String, TempDir) {
    let tmp = TempDir::new().unwrap();
    let embedder: Arc<dyn veclayer::Embedder + Send + Sync> = Arc::from(
        veclayer::embedder::from_config(&veclayer::config::EmbedderConfig::default()).unwrap(),
    );
    let dim = embedder.dimension();
    let store = Arc::new(StoreBackend::open(tmp.path(), dim, false).await.unwrap());
    let blob_store = Arc::new(BlobStore::open(tmp.path()).unwrap());

    let state = AppState {
        store,
        embedder,
        blob_store,
        data_dir: tmp.path().to_path_buf(),
        project: None,
        branch: None,
    };

    let app = build_app(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    (format!("http://127.0.0.1:{port}"), tmp)
}

#[tokio::test]
#[serial_test::serial]
async fn health_returns_ok() {
    let (base, _tmp) = spawn_server().await;
    let resp = reqwest::get(format!("{base}/health")).await.unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.text().await.unwrap(), "OK");
}

#[tokio::test]
#[serial_test::serial]
async fn recall_empty_store_returns_empty() {
    let (base, _tmp) = spawn_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/recall"))
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert!(body.is_empty());
}

#[tokio::test]
#[serial_test::serial]
async fn store_and_recall_roundtrip() {
    let (base, _tmp) = spawn_server().await;
    let client = reqwest::Client::new();

    // Store
    let resp = client
        .post(format!("{base}/api/store"))
        .json(&serde_json::json!({
            "content": "Rust is a systems programming language",
            "heading": "Rust",
            "perspectives": ["knowledge"]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let stored: String = resp.json().await.unwrap();
    assert!(stored.starts_with("Stored. ID:"));

    // Recall
    let resp = client
        .post(format!("{base}/api/recall"))
        .json(&serde_json::json!({"query": "systems programming"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let results: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert!(!results.is_empty());
    assert!(results[0]["chunk"]["content"]
        .as_str()
        .unwrap()
        .contains("Rust"));
}

#[tokio::test]
#[serial_test::serial]
async fn store_empty_content_returns_400() {
    let (base, _tmp) = spawn_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/store"))
        .json(&serde_json::json!({"content": ""}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("content is required"));
}

#[tokio::test]
#[serial_test::serial]
async fn malformed_json_returns_400() {
    let (base, _tmp) = spawn_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/store"))
        .header("content-type", "application/json")
        .body("not valid json{{{")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
#[serial_test::serial]
async fn stats_returns_counts() {
    let (base, _tmp) = spawn_server().await;
    let resp = reqwest::get(format!("{base}/api/stats")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["total_chunks"], 0);
}

#[tokio::test]
#[serial_test::serial]
async fn think_reflection_returns_report() {
    let (base, _tmp) = spawn_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/think"))
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
#[serial_test::serial]
async fn identity_returns_briefing() {
    let (base, _tmp) = spawn_server().await;
    let resp = reqwest::get(format!("{base}/api/identity")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["instructions"].is_string());
    assert!(body["core_entries"].is_number());
}

#[tokio::test]
#[serial_test::serial]
async fn priming_returns_plain_text() {
    let (base, _tmp) = spawn_server().await;
    let resp = reqwest::get(format!("{base}/api/priming")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(ct.contains("text/plain"));
    let text = resp.text().await.unwrap();
    assert!(text.contains("recall"));
}

#[tokio::test]
#[serial_test::serial]
async fn share_returns_token() {
    let (base, _tmp) = spawn_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/share"))
        .json(&serde_json::json!({"tree": "/"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
#[serial_test::serial]
async fn focus_nonexistent_id_returns_error() {
    let (base, _tmp) = spawn_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/focus"))
        .json(&serde_json::json!({"id": "nonexistent_id_12345"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"].is_string());
}
