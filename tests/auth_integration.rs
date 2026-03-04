//! End-to-end OAuth chain integration tests.
//!
//! These tests exercise the complete auth chain through the full application
//! stack: OAuth endpoints → token exchange → protected API routes.  Each test
//! goes all the way from client registration or device authorisation to a
//! working Bearer token that is accepted by `/api/*`.
//!
//! The oauth.rs unit tests verify the OAuth router in isolation with
//! `tower::ServiceExt::oneshot`.  The http_integration.rs auth module verifies
//! that pre-minted tokens are accepted or rejected correctly.  This file fills
//! the gap: obtaining a token through the live OAuth endpoints and then using
//! it against the live protected API — no shortcuts.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use reqwest::redirect::Policy;
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use tokio::net::TcpListener;
use veclayer::auth::capability::Capability;
use veclayer::auth::middleware::AuthState;
use veclayer::auth::oauth::OAuthState;
use veclayer::auth::token::{self, Claims};
use veclayer::auth::token_store::TokenStore;
use veclayer::mcp::http::{AppState, AuthSetup};
use veclayer::{BlobStore, StoreBackend};

// ── Constants ─────────────────────────────────────────────────────────────────

const SERVER_DID: &str = "did:key:zAuthIntegTest";
const REDIRECT_URI: &str = "https://app.test/callback";

// ── Helpers ───────────────────────────────────────────────────────────────────

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Generate a PKCE (verifier, S256-challenge) pair.
fn pkce_pair() -> (String, String) {
    let verifier = format!("veclayer-test-verifier-{}", uuid::Uuid::new_v4());
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
    (verifier, challenge)
}

/// Mint a JWT directly (bypasses OAuth, used only for the expired-token test).
fn mint_jwt(key: &SigningKey, cap: Capability, iat: u64, exp: u64) -> String {
    let claims = Claims::new(
        "did:key:zClient".to_owned(),
        SERVER_DID.to_owned(),
        cap,
        iat,
        exp,
    );
    token::mint(key, &claims).expect("mint")
}

/// Spin up a real TCP server with auth enabled and return (base_url, tempdir, signing_key).
///
/// `auto_approve` controls whether the authorization endpoint immediately
/// redirects with a code (true) or shows a consent page (false).
async fn spawn_auth_server(auto_approve: bool) -> (String, TempDir, SigningKey) {
    let tmp = TempDir::new().unwrap();
    let key = SigningKey::generate(&mut OsRng);

    let embedder: Arc<dyn veclayer::Embedder + Send + Sync> = Arc::from(
        veclayer::embedder::from_config(&veclayer::config::EmbedderConfig::default()).unwrap(),
    );
    let dim = embedder.dimension();
    let store = Arc::new(StoreBackend::open(tmp.path(), dim, false).await.unwrap());
    let blob_store = Arc::new(BlobStore::open(tmp.path()).unwrap());

    let auth_state = AuthState {
        verifying_key: key.verifying_key(),
        server_did: SERVER_DID.to_owned(),
        auth_required: true,
    };

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server_url = format!("http://127.0.0.1:{port}");

    let oauth_state = OAuthState {
        token_store: Arc::new(Mutex::new(TokenStore::open(tmp.path()).unwrap())),
        signing_key: Arc::new(key.clone()),
        server_did: SERVER_DID.to_owned(),
        server_url: server_url.clone(),
        token_expiry_secs: 3600,
        refresh_expiry_secs: 86_400,
        auto_approve,
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
    };

    let app = veclayer::mcp::http::build_app(state);
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    (server_url, tmp, key)
}

// ── Shared client that does NOT follow redirects ──────────────────────────────

fn no_redirect_client() -> reqwest::Client {
    reqwest::Client::builder()
        .redirect(Policy::none())
        .build()
        .unwrap()
}

// ── OAuth flow helpers ────────────────────────────────────────────────────────

/// POST /oauth/register → client_id
async fn register_client(base: &str, redirect_uri: &str) -> String {
    let resp = reqwest::Client::new()
        .post(format!("{base}/oauth/register"))
        .json(&serde_json::json!({
            "client_name": "auth_integration test client",
            "redirect_uris": [redirect_uri]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "register must return 201");
    let body: serde_json::Value = resp.json().await.unwrap();
    body["client_id"].as_str().unwrap().to_owned()
}

/// Full authorization code flow → (access_token, refresh_token).
///
/// Requires the server to be started with `auto_approve = true`.
async fn auth_code_flow(
    base: &str,
    client_id: &str,
    redirect_uri: &str,
    scope: &str,
) -> (String, String) {
    let (verifier, challenge) = pkce_pair();
    let client = no_redirect_client();

    // GET /oauth/authorize — auto-approve redirects immediately.
    let resp = client
        .get(format!("{base}/oauth/authorize"))
        .query(&[
            ("response_type", "code"),
            ("client_id", client_id),
            ("redirect_uri", redirect_uri),
            ("scope", scope),
            ("code_challenge", &challenge),
            ("code_challenge_method", "S256"),
        ])
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_redirection(),
        "authorize must redirect, got {}",
        resp.status()
    );

    let location = resp.headers()["location"].to_str().unwrap().to_owned();
    assert!(
        location.contains("code="),
        "redirect must contain code, got: {location}"
    );

    let code: String = location
        .split("code=")
        .nth(1)
        .unwrap()
        .split('&')
        .next()
        .unwrap()
        .to_owned();

    // POST /oauth/token — exchange code for tokens.
    let resp = reqwest::Client::new()
        .post(format!("{base}/oauth/token"))
        .form(&[
            ("grant_type", "authorization_code"),
            ("code", &code),
            ("code_verifier", &verifier),
            ("client_id", client_id),
            ("redirect_uri", redirect_uri),
        ])
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "token exchange must return 200");

    let body: serde_json::Value = resp.json().await.unwrap();
    let access_token = body["access_token"].as_str().unwrap().to_owned();
    let refresh_token = body["refresh_token"].as_str().unwrap().to_owned();
    assert!(!access_token.is_empty(), "access_token must not be empty");
    assert!(!refresh_token.is_empty(), "refresh_token must not be empty");

    (access_token, refresh_token)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

// ── test_health_is_public ─────────────────────────────────────────────────────

#[tokio::test]
#[serial_test::serial]
async fn test_health_is_public() {
    let (base, _tmp, _key) = spawn_auth_server(false).await;
    let resp = reqwest::get(format!("{base}/health")).await.unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.text().await.unwrap(), "OK");
}

// ── test_api_requires_auth_when_enabled ───────────────────────────────────────

#[tokio::test]
#[serial_test::serial]
async fn test_api_requires_auth_when_enabled() {
    let (base, _tmp, _key) = spawn_auth_server(false).await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/recall"))
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
    // WWW-Authenticate header signals the required scheme.
    assert!(resp
        .headers()
        .get("www-authenticate")
        .is_some_and(|v| v.to_str().unwrap().contains("Bearer")));
}

// ── test_valid_bearer_grants_access ───────────────────────────────────────────

#[tokio::test]
#[serial_test::serial]
async fn test_valid_bearer_grants_access() {
    let (base, _tmp, key) = spawn_auth_server(false).await;
    let t = now();
    let jwt = mint_jwt(&key, Capability::Read, t, t + 3600);

    let resp = reqwest::Client::new()
        .post(format!("{base}/api/recall"))
        .bearer_auth(&jwt)
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

// ── test_read_token_cannot_write ──────────────────────────────────────────────

#[tokio::test]
#[serial_test::serial]
async fn test_read_token_cannot_write() {
    let (base, _tmp, key) = spawn_auth_server(false).await;
    let t = now();
    let jwt = mint_jwt(&key, Capability::Read, t, t + 3600);

    let resp = reqwest::Client::new()
        .post(format!("{base}/api/store"))
        .bearer_auth(&jwt)
        .json(&serde_json::json!({"content": "should be blocked"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403);
}

// ── test_oauth_metadata_discoverable ─────────────────────────────────────────

#[tokio::test]
#[serial_test::serial]
async fn test_oauth_metadata_discoverable() {
    let (base, _tmp, _key) = spawn_auth_server(false).await;
    let resp = reqwest::get(format!("{base}/.well-known/oauth-authorization-server"))
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();

    // Required fields per RFC 8414.
    assert!(body["issuer"].as_str().is_some(), "issuer missing");
    assert!(
        body["authorization_endpoint"].as_str().is_some(),
        "authorization_endpoint missing"
    );
    assert!(
        body["token_endpoint"].as_str().is_some(),
        "token_endpoint missing"
    );

    // Endpoints must point at our server.
    assert!(
        body["authorization_endpoint"]
            .as_str()
            .unwrap()
            .contains("/oauth/authorize"),
        "authorization_endpoint path wrong"
    );
    assert!(
        body["token_endpoint"]
            .as_str()
            .unwrap()
            .contains("/oauth/token"),
        "token_endpoint path wrong"
    );

    // Grant types and PKCE.
    let grants = body["grant_types_supported"].as_array().unwrap();
    assert!(
        grants.iter().any(|g| g == "authorization_code"),
        "authorization_code grant missing"
    );
    assert!(
        grants.iter().any(|g| g == "refresh_token"),
        "refresh_token grant missing"
    );
    assert!(
        grants
            .iter()
            .any(|g| g == "urn:ietf:params:oauth:grant-type:device_code"),
        "device_code grant missing"
    );

    let methods = body["code_challenge_methods_supported"].as_array().unwrap();
    assert!(
        methods.iter().any(|m| m == "S256"),
        "S256 PKCE method missing"
    );
}

// ── test_oauth_full_authorization_code_flow ───────────────────────────────────
//
// Register a client → authorize (auto_approve) → exchange code for tokens →
// use access_token to call a protected API endpoint.

#[tokio::test]
#[serial_test::serial]
async fn test_oauth_full_authorization_code_flow() {
    let (base, _tmp, _key) = spawn_auth_server(true /* auto_approve */).await;

    // Register a client.
    let client_id = register_client(&base, REDIRECT_URI).await;

    // Run the full authorization code flow.
    let (access_token, _refresh_token) =
        auth_code_flow(&base, &client_id, REDIRECT_URI, "write").await;

    // The access_token must work against a protected write endpoint.
    let resp = reqwest::Client::new()
        .post(format!("{base}/api/store"))
        .bearer_auth(&access_token)
        .json(&serde_json::json!({
            "content": "stored via OAuth authorization code flow",
            "heading": "OAuth test"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "OAuth access_token must be accepted by /api/store"
    );
}

// ── test_oauth_token_refresh ──────────────────────────────────────────────────
//
// Complete the auth code flow → use refresh_token to obtain new tokens →
// verify the new access_token works and the old refresh_token is revoked.

#[tokio::test]
#[serial_test::serial]
async fn test_oauth_token_refresh() {
    let (base, _tmp, _key) = spawn_auth_server(true).await;
    let client_id = register_client(&base, REDIRECT_URI).await;
    let (access_token_1, refresh_token_1) =
        auth_code_flow(&base, &client_id, REDIRECT_URI, "read").await;

    // First access_token should work.
    let resp = reqwest::Client::new()
        .post(format!("{base}/api/recall"))
        .bearer_auth(&access_token_1)
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "first access_token must work");

    // Exchange refresh_token for new tokens.
    let resp = reqwest::Client::new()
        .post(format!("{base}/oauth/token"))
        .form(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", &refresh_token_1),
            ("client_id", &client_id),
        ])
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "refresh must return 200");

    let body: serde_json::Value = resp.json().await.unwrap();
    let access_token_2 = body["access_token"].as_str().unwrap().to_owned();
    let refresh_token_2 = body["refresh_token"].as_str().unwrap().to_owned();

    // New tokens must differ from the originals.
    assert_ne!(access_token_1, access_token_2, "access tokens must rotate");
    assert_ne!(
        refresh_token_1, refresh_token_2,
        "refresh tokens must rotate"
    );

    // New access_token must be accepted by the API.
    let resp = reqwest::Client::new()
        .post(format!("{base}/api/recall"))
        .bearer_auth(&access_token_2)
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "refreshed access_token must work");

    // Old refresh_token must be revoked.
    let resp = reqwest::Client::new()
        .post(format!("{base}/oauth/token"))
        .form(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", &refresh_token_1),
            ("client_id", &client_id),
        ])
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400, "revoked refresh_token must be rejected");
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body["error"], "invalid_grant",
        "revoked token must return invalid_grant"
    );

    // new refresh_token is not used yet, just verify it's non-empty.
    assert!(!refresh_token_2.is_empty());
}

// ── test_oauth_device_flow ────────────────────────────────────────────────────
//
// POST /oauth/device/code → simulate user approval → POST /oauth/token →
// use access_token against the API.

#[tokio::test]
#[serial_test::serial]
async fn test_oauth_device_flow() {
    let (base, _tmp, _key) = spawn_auth_server(false).await;
    let client = reqwest::Client::new();

    // Register a client (device flow requires a known client_id).
    let client_id = register_client(&base, REDIRECT_URI).await;

    // Step 1: request a device code.
    let resp = client
        .post(format!("{base}/oauth/device/code"))
        .form(&[("client_id", &client_id), ("scope", &"read".to_owned())])
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "device/code must return 200");

    let device_body: serde_json::Value = resp.json().await.unwrap();
    let device_code = device_body["device_code"].as_str().unwrap().to_owned();
    let user_code = device_body["user_code"].as_str().unwrap().to_owned();

    assert!(!device_code.is_empty(), "device_code must not be empty");
    assert!(!user_code.is_empty(), "user_code must not be empty");
    assert!(
        device_body["verification_uri"].as_str().is_some(),
        "verification_uri missing"
    );
    assert_eq!(device_body["expires_in"], 600);

    // Step 2: poll before approval — must get authorization_pending.
    let resp = client
        .post(format!("{base}/oauth/token"))
        .form(&[
            ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
            ("device_code", &device_code),
            ("client_id", &client_id),
        ])
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400, "pending poll must return 400");
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body["error"], "authorization_pending",
        "pending response must use authorization_pending error"
    );

    // Step 3: simulate user approving the device via the browser endpoint.
    let resp = client
        .post(format!("{base}/oauth/device"))
        .form(&[
            ("user_code", user_code.as_str()),
            ("scope", "read"),
            ("approved", "true"),
        ])
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "device approval must return 200");
    let html = resp.text().await.unwrap();
    assert!(
        html.contains("Authorized"),
        "success page must contain 'Authorized'"
    );

    // Step 4: poll again — must get tokens.
    let resp = client
        .post(format!("{base}/oauth/token"))
        .form(&[
            ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
            ("device_code", &device_code),
            ("client_id", &client_id),
        ])
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "post-approval poll must return 200");

    let token_body: serde_json::Value = resp.json().await.unwrap();
    let access_token = token_body["access_token"].as_str().unwrap().to_owned();
    assert!(!access_token.is_empty(), "access_token must not be empty");
    assert_eq!(token_body["token_type"], "Bearer");

    // Step 5: use the access_token against the API.
    let resp = client
        .post(format!("{base}/api/recall"))
        .bearer_auth(&access_token)
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "device-flow access_token must be accepted by the API"
    );
}

// ── test_backward_compat_no_auth ──────────────────────────────────────────────
//
// When auth = None (no AuthSetup), all API routes are fully open.

#[tokio::test]
#[serial_test::serial]
async fn test_backward_compat_no_auth() {
    // Build an open (no-auth) server using the same pattern as http_integration.rs.
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
        auth: None,
    };

    let app = veclayer::mcp::http::build_app(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let base = format!("http://127.0.0.1:{port}");

    // No Bearer token needed — recall and store must both succeed.
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{base}/api/recall"))
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "recall without token must succeed in open mode"
    );

    let resp = client
        .post(format!("{base}/api/store"))
        .json(&serde_json::json!({"content": "backward compat test"}))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "store without token must succeed in open mode"
    );
}

// ── test_expired_token_rejected ───────────────────────────────────────────────
//
// A JWT with exp in the past must be rejected with 401 by the live server.

#[tokio::test]
#[serial_test::serial]
async fn test_expired_token_rejected() {
    let (base, _tmp, key) = spawn_auth_server(false).await;
    let t = now();
    // exp in the past.
    let expired_jwt = mint_jwt(&key, Capability::Read, t - 7200, t - 3600);

    let resp = reqwest::Client::new()
        .post(format!("{base}/api/recall"))
        .bearer_auth(&expired_jwt)
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        401,
        "expired token must be rejected with 401"
    );
}

// ── test_oauth_scope_write_token_permits_store ────────────────────────────────
//
// A token obtained via OAuth with scope=write must be accepted by /api/store.

#[tokio::test]
#[serial_test::serial]
async fn test_oauth_scope_write_token_permits_store() {
    let (base, _tmp, _key) = spawn_auth_server(true).await;
    let client_id = register_client(&base, REDIRECT_URI).await;
    let (access_token, _) = auth_code_flow(&base, &client_id, REDIRECT_URI, "write").await;

    let resp = reqwest::Client::new()
        .post(format!("{base}/api/store"))
        .bearer_auth(&access_token)
        .json(&serde_json::json!({
            "content": "OAuth write scope integration test"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "write-scoped OAuth token must be accepted by /api/store"
    );
}

// ── test_oauth_scope_read_token_blocks_store ──────────────────────────────────
//
// A token obtained via OAuth with scope=read must be rejected by /api/store.

#[tokio::test]
#[serial_test::serial]
async fn test_oauth_scope_read_token_blocks_store() {
    let (base, _tmp, _key) = spawn_auth_server(true).await;
    let client_id = register_client(&base, REDIRECT_URI).await;
    let (access_token, _) = auth_code_flow(&base, &client_id, REDIRECT_URI, "read").await;

    let resp = reqwest::Client::new()
        .post(format!("{base}/api/store"))
        .bearer_auth(&access_token)
        .json(&serde_json::json!({
            "content": "this must be blocked"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        403,
        "read-scoped OAuth token must be rejected by /api/store with 403"
    );
}

// ── test_oauth_registration_empty_name_rejected ───────────────────────────────

#[tokio::test]
#[serial_test::serial]
async fn test_oauth_registration_empty_name_rejected() {
    let (base, _tmp, _key) = spawn_auth_server(false).await;

    let resp = reqwest::Client::new()
        .post(format!("{base}/oauth/register"))
        .json(&serde_json::json!({
            "client_name": "   ",
            "redirect_uris": [REDIRECT_URI]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        400,
        "empty client_name must be rejected with 400"
    );
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(
        body["error"].as_str().unwrap_or("").contains("client_name"),
        "error must mention client_name"
    );
}

// ── test_oauth_registration_empty_redirect_uris_rejected ─────────────────────

#[tokio::test]
#[serial_test::serial]
async fn test_oauth_registration_empty_redirect_uris_rejected() {
    let (base, _tmp, _key) = spawn_auth_server(false).await;

    let resp = reqwest::Client::new()
        .post(format!("{base}/oauth/register"))
        .json(&serde_json::json!({
            "client_name": "valid name",
            "redirect_uris": []
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        400,
        "empty redirect_uris must be rejected with 400"
    );
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body["error"].as_str().unwrap_or(""),
        "invalid_client_metadata",
        "error code must be invalid_client_metadata"
    );
}

// ── test_consent_form_deny_path ───────────────────────────────────────────────
//
// When auto_approve=false and the user submits the consent form with deny,
// the redirect must carry an access_denied error.

#[tokio::test]
#[serial_test::serial]
async fn test_consent_form_deny_path() {
    let (base, _tmp, _key) = spawn_auth_server(false /* auto_approve=false */).await;
    let client_id = register_client(&base, REDIRECT_URI).await;
    let (_, challenge) = pkce_pair();

    // POST /oauth/authorize simulating the deny button.
    let no_redirect = no_redirect_client();
    let resp = no_redirect
        .post(format!("{base}/oauth/authorize"))
        .form(&[
            ("client_id", client_id.as_str()),
            ("redirect_uri", REDIRECT_URI),
            ("scope", "write"),
            ("code_challenge", challenge.as_str()),
            ("state", "test-state-xyz"),
            ("approved", "false"),
        ])
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_redirection(),
        "deny must redirect, got {}",
        resp.status()
    );
    let location = resp.headers()["location"].to_str().unwrap();
    assert!(
        location.contains("error=access_denied"),
        "deny redirect must contain access_denied, got: {location}"
    );
    assert!(
        location.contains("state=test-state-xyz"),
        "state must be preserved in deny redirect"
    );
}
