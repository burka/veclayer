#![cfg(feature = "http")]

use std::sync::Arc;
use tempfile::TempDir;
use tokio::net::TcpListener;
use veclayer::mcp::http::{build_app, AppState};
use veclayer::{BlobStore, StoreBackend};

/// Build an open (no-auth) AppState backed by a temp directory.
async fn open_state(tmp: &TempDir) -> AppState {
    let embedder: Arc<dyn veclayer::Embedder + Send + Sync> = Arc::from(
        veclayer::embedder::from_config(&veclayer::config::EmbedderConfig::default()).unwrap(),
    );
    let dim = embedder.dimension();
    let store = Arc::new(StoreBackend::open(tmp.path(), dim, false).await.unwrap());
    let blob_store = Arc::new(BlobStore::open(tmp.path()).unwrap());
    AppState {
        store,
        embedder,
        blob_store,
        data_dir: tmp.path().to_path_buf(),
        project: None,
        branch: None,
        auth: None,
        git_store: None,
        push_mode: veclayer::git::branch_config::PushMode::Off,
    }
}

/// Spin up a real HTTP server on a random port, return the base URL.
async fn spawn_server() -> (String, TempDir) {
    let tmp = TempDir::new().unwrap();
    let state = open_state(&tmp).await;
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

// ─── Auth integration tests ───────────────────────────────────────────────────

mod auth {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use std::time::{SystemTime, UNIX_EPOCH};

    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;
    use tempfile::TempDir;
    use tokio::net::TcpListener;
    use veclayer::auth::capability::Capability;
    use veclayer::auth::middleware::AuthState;
    use veclayer::auth::oauth::OAuthState;
    use veclayer::auth::token::{self, Claims};
    use veclayer::auth::token_store::TokenStore;
    use veclayer::mcp::http::{AppState, AuthSetup};
    use veclayer::{BlobStore, StoreBackend};

    const SERVER_DID: &str = "did:key:zTest";

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    fn generate_key() -> SigningKey {
        SigningKey::generate(&mut OsRng)
    }

    fn mint_token(key: &SigningKey, cap: Capability) -> String {
        let t = now();
        let claims = Claims::new(
            "did:key:zClient".to_owned(),
            SERVER_DID.to_owned(),
            cap,
            t,
            t + 3600,
        );
        token::mint(key, &claims).expect("mint")
    }

    async fn spawn_auth_server(auth_required: bool) -> (String, TempDir, SigningKey) {
        let tmp = TempDir::new().unwrap();
        let key = generate_key();

        let embedder: Arc<dyn veclayer::Embedder + Send + Sync> = Arc::from(
            veclayer::embedder::from_config(&veclayer::config::EmbedderConfig::default()).unwrap(),
        );
        let dim = embedder.dimension();
        let store = Arc::new(StoreBackend::open(tmp.path(), dim, false).await.unwrap());
        let blob_store = Arc::new(BlobStore::open(tmp.path()).unwrap());

        let auth_state = AuthState {
            verifying_key: key.verifying_key(),
            server_did: SERVER_DID.to_owned(),
            auth_required,
        };
        let oauth_state = OAuthState {
            token_store: Arc::new(Mutex::new(TokenStore::open(tmp.path()).unwrap())),
            signing_key: Arc::new(key.clone()),
            server_did: SERVER_DID.to_owned(),
            server_url: "http://127.0.0.1:0".to_owned(),
            token_expiry_secs: 3600,
            refresh_expiry_secs: 86_400,
            auto_approve: false,
            device_codes: Arc::new(Mutex::new(HashMap::new())),
        };

        let state = AppState {
            store,
            embedder,
            blob_store,
            data_dir: tmp.path().to_path_buf(),
            project: None,
            branch: None,
            auth: Some(AuthSetup {
                auth_state,

                oauth_state,
            }),
            git_store: None,
            push_mode: veclayer::git::branch_config::PushMode::Off,
        };

        let app = veclayer::mcp::http::build_app(state);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        (format!("http://127.0.0.1:{port}"), tmp, key)
    }

    // ── health is always public ───────────────────────────────────────────────

    #[tokio::test]
    #[serial_test::serial]
    async fn health_public_when_auth_required() {
        let (base, _tmp, _key) = spawn_auth_server(true).await;
        let resp = reqwest::get(format!("{base}/health")).await.unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn health_public_when_auth_not_required() {
        let (base, _tmp, _key) = spawn_auth_server(false).await;
        let resp = reqwest::get(format!("{base}/health")).await.unwrap();
        assert_eq!(resp.status(), 200);
    }

    // ── auth_required=true blocks unauthenticated requests ────────────────────

    #[tokio::test]
    #[serial_test::serial]
    async fn recall_requires_auth_when_enforced() {
        let (base, _tmp, _key) = spawn_auth_server(true).await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{base}/api/recall"))
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn store_requires_auth_when_enforced() {
        let (base, _tmp, _key) = spawn_auth_server(true).await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{base}/api/store"))
            .json(&serde_json::json!({"content": "hello"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
    }

    // ── valid token passes through ────────────────────────────────────────────

    #[tokio::test]
    #[serial_test::serial]
    async fn recall_with_read_token_succeeds() {
        let (base, _tmp, key) = spawn_auth_server(true).await;
        let jwt = mint_token(&key, Capability::Read);
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{base}/api/recall"))
            .bearer_auth(&jwt)
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn store_with_read_token_is_forbidden() {
        let (base, _tmp, key) = spawn_auth_server(true).await;
        let jwt = mint_token(&key, Capability::Read);
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{base}/api/store"))
            .bearer_auth(&jwt)
            .json(&serde_json::json!({"content": "hello"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 403);
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn store_with_write_token_succeeds() {
        let (base, _tmp, key) = spawn_auth_server(true).await;
        let jwt = mint_token(&key, Capability::Write);
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{base}/api/store"))
            .bearer_auth(&jwt)
            .json(&serde_json::json!({"content": "auth integration test"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    // ── capability enforcement on read-only endpoints ─────────────────────────

    #[tokio::test]
    #[serial_test::serial]
    async fn stats_requires_auth_when_enforced() {
        let (base, _tmp, _key) = spawn_auth_server(true).await;
        let resp = reqwest::get(format!("{base}/api/stats")).await.unwrap();
        assert_eq!(resp.status(), 401);
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn identity_requires_auth_when_enforced() {
        let (base, _tmp, _key) = spawn_auth_server(true).await;
        let resp = reqwest::get(format!("{base}/api/identity")).await.unwrap();
        assert_eq!(resp.status(), 401);
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn priming_requires_auth_when_enforced() {
        let (base, _tmp, _key) = spawn_auth_server(true).await;
        let resp = reqwest::get(format!("{base}/api/priming")).await.unwrap();
        assert_eq!(resp.status(), 401);
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn stats_accessible_with_read_token() {
        let (base, _tmp, key) = spawn_auth_server(true).await;
        let jwt = mint_token(&key, Capability::Read);
        let resp = reqwest::Client::new()
            .get(format!("{base}/api/stats"))
            .bearer_auth(&jwt)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn identity_accessible_with_read_token() {
        let (base, _tmp, key) = spawn_auth_server(true).await;
        let jwt = mint_token(&key, Capability::Read);
        let resp = reqwest::Client::new()
            .get(format!("{base}/api/identity"))
            .bearer_auth(&jwt)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn priming_accessible_with_read_token() {
        let (base, _tmp, key) = spawn_auth_server(true).await;
        let jwt = mint_token(&key, Capability::Read);
        let resp = reqwest::Client::new()
            .get(format!("{base}/api/priming"))
            .bearer_auth(&jwt)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    // ── auth_required=false → open access (backward compat) ──────────────────

    #[tokio::test]
    #[serial_test::serial]
    async fn open_mode_allows_recall_without_token() {
        let (base, _tmp, _key) = spawn_auth_server(false).await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{base}/api/recall"))
            .json(&serde_json::json!({}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn open_mode_allows_store_without_token() {
        let (base, _tmp, _key) = spawn_auth_server(false).await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{base}/api/store"))
            .json(&serde_json::json!({"content": "open mode store test"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }
}
