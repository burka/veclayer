//! OAuth 2.0 Authorization Server — route handlers and router builder.
//!
//! Implements:
//! - RFC 6749: Authorization Code Grant (with PKCE)
//! - RFC 7591: Dynamic Client Registration
//! - RFC 8628: Device Authorization Grant
//! - OAuth 2.0 Authorization Server Metadata (RFC 8414)
//!
//! Build the router with [`oauth_router`] and merge it into the main
//! application router when ready.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::extract::{Form, Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine;
use ed25519_dalek::SigningKey;
use percent_encoding::{utf8_percent_encode, AsciiSet, CONTROLS};
use rand::{Rng, RngCore};
use serde::Deserialize;

use tracing::{info, warn};

use super::capability::Capability;
use super::token::{mint, Claims};
use super::token_store::TokenStore;
use crate::util::unix_now;

// ─── Device expiry ────────────────────────────────────────────────────────────

const DEVICE_CODE_TTL_SECS: u64 = 600;

// ─── Shared state ─────────────────────────────────────────────────────────────

/// Shared state for all OAuth endpoints.
#[derive(Clone)]
pub struct OAuthState {
    pub token_store: Arc<Mutex<TokenStore>>,
    pub signing_key: Arc<SigningKey>,
    pub server_did: String,
    pub server_url: String,
    pub token_expiry_secs: u64,
    pub refresh_expiry_secs: u64,
    pub auto_approve: bool,
    /// Pending device authorizations (in-memory, short-lived).
    pub device_codes: Arc<Mutex<HashMap<String, PendingDeviceAuth>>>,
}

// ─── Device authorization ─────────────────────────────────────────────────────

/// In-flight device authorization request.
pub struct PendingDeviceAuth {
    pub device_code: String,
    pub user_code: String,
    pub client_id: String,
    pub scope: Capability,
    pub expires_at: u64,
    /// `None` = pending, `Some(cap)` = approved with that capability.
    pub approved: Option<Capability>,
    pub denied: bool,
}

// ─── Router ───────────────────────────────────────────────────────────────────

/// Build the OAuth 2.0 router.
///
/// Merge this into the main application router:
/// ```ignore
/// let app = build_app(state).merge(oauth_router(oauth_state));
/// ```
pub fn oauth_router(state: OAuthState) -> Router {
    Router::new()
        .route(
            "/.well-known/oauth-authorization-server",
            get(metadata_handler),
        )
        .route("/oauth/register", post(register_handler))
        .route(
            "/oauth/authorize",
            get(authorize_get_handler).post(authorize_post_handler),
        )
        .route("/oauth/token", post(token_handler))
        .route("/oauth/device/code", post(device_code_handler))
        .route(
            "/oauth/device",
            get(device_page_handler).post(device_approve_handler),
        )
        .with_state(state)
}

// ─── Metadata ─────────────────────────────────────────────────────────────────

/// GET /.well-known/oauth-authorization-server
async fn metadata_handler(State(state): State<OAuthState>) -> Json<serde_json::Value> {
    let url = &state.server_url;
    Json(serde_json::json!({
        "issuer": url,
        "authorization_endpoint": format!("{url}/oauth/authorize"),
        "token_endpoint": format!("{url}/oauth/token"),
        "device_authorization_endpoint": format!("{url}/oauth/device/code"),
        "registration_endpoint": format!("{url}/oauth/register"),
        "response_types_supported": ["code"],
        "grant_types_supported": [
            "authorization_code",
            "urn:ietf:params:oauth:grant-type:device_code",
            "refresh_token"
        ],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
        "scopes_supported": ["read", "write", "admin"]
    }))
}

// ─── Dynamic Client Registration (RFC 7591) ───────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub client_name: String,
    pub redirect_uris: Vec<String>,
}

/// POST /oauth/register
/// Maximum number of registered clients before registration is rejected.
const MAX_REGISTERED_CLIENTS: usize = 100;

/// Maximum length for client names and redirect URIs.
const MAX_STRING_LENGTH: usize = 2048;

async fn register_handler(
    State(state): State<OAuthState>,
    Json(body): Json<RegisterRequest>,
) -> Response {
    if body.client_name.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "client_name is required"})),
        )
            .into_response();
    }

    if body.client_name.len() > MAX_STRING_LENGTH {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "invalid_client_metadata",
                "error_description": "client_name exceeds maximum length"
            })),
        )
            .into_response();
    }

    if body.redirect_uris.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "invalid_client_metadata",
                "error_description": "redirect_uris must not be empty"
            })),
        )
            .into_response();
    }

    if body
        .redirect_uris
        .iter()
        .any(|u| u.len() > MAX_STRING_LENGTH)
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "invalid_client_metadata",
                "error_description": "redirect_uri exceeds maximum length"
            })),
        )
            .into_response();
    }

    let mut store = state.token_store.lock().unwrap_or_else(|e| e.into_inner());

    if store.client_count() >= MAX_REGISTERED_CLIENTS {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({
                "error": "too_many_clients",
                "error_description": "maximum number of registered clients reached"
            })),
        )
            .into_response();
    }

    let client = store.register_client(&body.client_name, body.redirect_uris);

    info!(
        "OAuth client registered: {} ({})",
        client.client_id, client.client_name
    );

    (
        StatusCode::CREATED,
        Json(serde_json::json!({
            "client_id": client.client_id,
            "client_name": client.client_name,
            "redirect_uris": client.redirect_uris,
            "created_at": client.created_at,
        })),
    )
        .into_response()
}

// ─── Authorization Endpoint ───────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct AuthorizeQuery {
    pub response_type: Option<String>,
    pub client_id: Option<String>,
    pub redirect_uri: Option<String>,
    pub state: Option<String>,
    pub scope: Option<String>,
    pub code_challenge: Option<String>,
    pub code_challenge_method: Option<String>,
}

/// GET /oauth/authorize
async fn authorize_get_handler(
    State(state): State<OAuthState>,
    Query(params): Query<AuthorizeQuery>,
) -> Response {
    // Step 1: Extract client_id and redirect_uri — 400 error page if absent
    // (cannot redirect before URI is validated).
    let client_id = match &params.client_id {
        Some(id) => id.clone(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Html(error_page("Missing client_id parameter")),
            )
                .into_response()
        }
    };

    let redirect_uri = match &params.redirect_uri {
        Some(uri) => uri.clone(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Html(error_page("Missing redirect_uri parameter")),
            )
                .into_response()
        }
    };

    // Step 2: Validate client_id and redirect_uri against the registry before
    // issuing any redirect.  An unvalidated redirect_uri must never be the
    // target of an error redirect (open redirect vulnerability).
    let client_name = {
        let store = state.token_store.lock().unwrap_or_else(|e| e.into_inner());
        match store.get_client(&client_id) {
            Some(client) => {
                if !client.redirect_uris.contains(&redirect_uri) {
                    return (
                        StatusCode::BAD_REQUEST,
                        Html(error_page("redirect_uri does not match registered URIs")),
                    )
                        .into_response();
                }
                client.client_name.clone()
            }
            None => {
                return (
                    StatusCode::BAD_REQUEST,
                    Html(error_page("Unknown client_id")),
                )
                    .into_response();
            }
        }
    };

    // Step 3: Check response_type — only "code" is supported.
    if params.response_type.as_deref() != Some("code") {
        return redirect_with_error(
            &redirect_uri,
            "unsupported_response_type",
            "only response_type=code is supported",
            params.state.as_deref(),
        );
    }

    // Step 4: Require PKCE code_challenge.
    let code_challenge = match &params.code_challenge {
        Some(c) => c.clone(),
        None => {
            return redirect_with_error(
                &redirect_uri,
                "invalid_request",
                "code_challenge required",
                params.state.as_deref(),
            );
        }
    };

    // Only S256 is supported.
    if params
        .code_challenge_method
        .as_deref()
        .unwrap_or("plain")
        .to_uppercase()
        != "S256"
    {
        return redirect_with_error(
            &redirect_uri,
            "invalid_request",
            "only S256 code_challenge_method is supported",
            params.state.as_deref(),
        );
    }

    let scope_str = params.scope.as_deref().unwrap_or("read");
    let capability = match scope_str.parse::<Capability>() {
        Ok(cap) => cap,
        Err(_) => {
            return redirect_with_error(
                &redirect_uri,
                "invalid_scope",
                "unknown scope",
                params.state.as_deref(),
            );
        }
    };

    if state.auto_approve {
        let code = state
            .token_store
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .create_code(
                &client_id,
                &state.server_did,
                capability,
                &redirect_uri,
                &code_challenge,
            );

        return redirect_with_code(&redirect_uri, &code, params.state.as_deref());
    }

    // Show consent page.
    Html(consent_page(
        &client_name,
        &client_id,
        &redirect_uri,
        scope_str,
        &code_challenge,
        params.state.as_deref().unwrap_or(""),
    ))
    .into_response()
}

#[derive(Debug, Deserialize)]
pub struct ConsentForm {
    pub client_id: String,
    pub redirect_uri: String,
    pub scope: String,
    pub code_challenge: String,
    pub state: Option<String>,
    pub approved: Option<String>,
}

/// POST /oauth/authorize — handle consent form submission.
async fn authorize_post_handler(
    State(state): State<OAuthState>,
    Form(form): Form<ConsentForm>,
) -> Response {
    let oauth_state = form.state.as_deref();

    // Validate client and redirect_uri BEFORE any redirect to prevent open
    // redirect via a crafted POST with an arbitrary redirect_uri.
    {
        let store = state.token_store.lock().unwrap_or_else(|e| e.into_inner());
        match store.get_client(&form.client_id) {
            Some(client) => {
                if !client.redirect_uris.contains(&form.redirect_uri) {
                    return (
                        StatusCode::BAD_REQUEST,
                        Html(error_page("redirect_uri does not match registered URIs")),
                    )
                        .into_response();
                }
            }
            None => {
                return (
                    StatusCode::BAD_REQUEST,
                    Html(error_page("Unknown client_id")),
                )
                    .into_response();
            }
        }
    }

    let redirect_uri = form.redirect_uri.clone();

    // Deny button was pressed or `approved` field is absent.
    if form.approved.as_deref() != Some("true") {
        return redirect_with_error(
            &redirect_uri,
            "access_denied",
            "user denied access",
            oauth_state,
        );
    }

    let capability = match form.scope.parse::<Capability>() {
        Ok(cap) => cap,
        Err(_) => {
            return redirect_with_error(
                &redirect_uri,
                "invalid_scope",
                "unknown scope",
                oauth_state,
            );
        }
    };

    let code = state
        .token_store
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .create_code(
            &form.client_id,
            &state.server_did,
            capability,
            &form.redirect_uri,
            &form.code_challenge,
        );

    redirect_with_code(&redirect_uri, &code, oauth_state)
}

// ─── Token Endpoint ───────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct TokenRequest {
    pub grant_type: String,
    // authorization_code fields
    pub code: Option<String>,
    pub code_verifier: Option<String>,
    pub client_id: Option<String>,
    pub redirect_uri: Option<String>,
    // refresh_token fields
    pub refresh_token: Option<String>,
    // device_code fields
    pub device_code: Option<String>,
}

/// POST /oauth/token
async fn token_handler(
    State(state): State<OAuthState>,
    Form(form): Form<TokenRequest>,
) -> Response {
    match form.grant_type.as_str() {
        "authorization_code" => handle_auth_code_grant(state, form).await,
        "refresh_token" => handle_refresh_token_grant(state, form).await,
        "urn:ietf:params:oauth:grant-type:device_code" => {
            handle_device_code_grant(state, form).await
        }
        other => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "unsupported_grant_type",
                "error_description": format!("unsupported grant_type: {other}")
            })),
        )
            .into_response(),
    }
}

async fn handle_auth_code_grant(state: OAuthState, form: TokenRequest) -> Response {
    info!(
        "Auth code exchange for client {}",
        form.client_id.as_deref().unwrap_or("unknown")
    );
    let code = match form.code.as_deref() {
        Some(c) => c,
        None => {
            return token_error("invalid_request", "code is required");
        }
    };

    let verifier = match form.code_verifier.as_deref() {
        Some(v) => v,
        None => {
            return token_error("invalid_request", "code_verifier is required");
        }
    };

    let auth_code = {
        let mut store = state.token_store.lock().unwrap_or_else(|e| e.into_inner());
        match store.consume_code(code, verifier) {
            Ok(ac) => ac,
            Err(e) => {
                return token_error("invalid_grant", &e.to_string());
            }
        }
    };

    // Validate client_id (required for public clients per RFC 6749 §4.1.3).
    let client_id = match form.client_id.as_deref() {
        Some(cid) => cid,
        None => {
            return token_error("invalid_request", "client_id is required");
        }
    };
    if client_id != auth_code.client_id {
        return token_error("invalid_client", "client_id mismatch");
    }
    if let Some(uri) = form.redirect_uri.as_deref() {
        if uri != auth_code.redirect_uri {
            return token_error("invalid_grant", "redirect_uri mismatch");
        }
    }

    mint_token_response(
        &state,
        &auth_code.client_id,
        &auth_code.did,
        auth_code.capability,
    )
}

async fn handle_refresh_token_grant(state: OAuthState, form: TokenRequest) -> Response {
    info!("Refresh token exchange");
    let refresh = match form.refresh_token.as_deref() {
        Some(r) => r,
        None => {
            return token_error("invalid_request", "refresh_token is required");
        }
    };

    // Validate and revoke atomically to prevent TOCTOU: a concurrent request
    // cannot redeem the same token between the two operations.
    let (client_id, did, capability) = {
        let mut store = state.token_store.lock().unwrap_or_else(|e| e.into_inner());
        match store.validate_and_revoke_refresh(refresh) {
            Ok(record) => record,
            Err(e) => {
                return token_error("invalid_grant", &e.to_string());
            }
        }
    };

    // Validate client_id (required for public clients per RFC 6749 §4.1.3).
    // Note: revocation already happened — a mismatched client_id still rejects
    // the request, and the caller cannot retry since the token is already revoked.
    let form_cid = match form.client_id.as_deref() {
        Some(cid) => cid,
        None => {
            return token_error("invalid_request", "client_id is required");
        }
    };
    if form_cid != client_id {
        return token_error("invalid_client", "client_id mismatch");
    }

    mint_token_response(&state, &client_id, &did, capability)
}

async fn handle_device_code_grant(state: OAuthState, form: TokenRequest) -> Response {
    info!("Device code token exchange");
    let device_code_str = match form.device_code.as_deref() {
        Some(dc) => dc,
        None => {
            return token_error("invalid_request", "device_code is required");
        }
    };

    let now = unix_now();

    let (client_id, capability) = {
        let mut map = state.device_codes.lock().unwrap_or_else(|e| e.into_inner());

        let entry = match map.get(device_code_str) {
            Some(e) => e,
            None => {
                return token_error("invalid_grant", "unknown device_code");
            }
        };

        if now > entry.expires_at {
            map.remove(device_code_str);
            return token_error("expired_token", "device authorization expired");
        }

        if entry.denied {
            map.remove(device_code_str);
            return token_error("access_denied", "user denied access");
        }

        match entry.approved {
            None => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": "authorization_pending",
                        "error_description": "user has not yet approved the device"
                    })),
                )
                    .into_response();
            }
            Some(cap) => {
                let client_id = entry.client_id.clone();
                map.remove(device_code_str);
                (client_id, cap)
            }
        }
    };

    // For device grants the subject DID equals the client_id (no user account).
    mint_token_response(&state, &client_id, &client_id, capability)
}

/// Mint a fresh access token + refresh token pair and return the token response body.
fn mint_token_response(
    state: &OAuthState,
    client_id: &str,
    did: &str,
    capability: Capability,
) -> Response {
    let now = unix_now();
    let exp = now + state.token_expiry_secs;

    let claims = Claims::new(
        did.to_owned(),
        state.server_did.clone(),
        capability,
        now,
        exp,
    );

    let access_token = match mint(&state.signing_key, &claims) {
        Ok(t) => t,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "server_error",
                    "error_description": e.to_string()
                })),
            )
                .into_response();
        }
    };

    // Generate a random refresh token.
    let refresh_token = generate_opaque_token();
    let refresh_exp = now + state.refresh_expiry_secs;

    state
        .token_store
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .store_refresh(&refresh_token, client_id, did, capability, refresh_exp);

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": state.token_expiry_secs,
            "refresh_token": refresh_token,
            "scope": capability.to_string(),
        })),
    )
        .into_response()
}

// ─── Device Authorization (RFC 8628) ─────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct DeviceCodeRequest {
    pub client_id: Option<String>,
    pub scope: Option<String>,
}

/// POST /oauth/device/code
async fn device_code_handler(
    State(state): State<OAuthState>,
    Form(form): Form<DeviceCodeRequest>,
) -> Response {
    let client_id = match form.client_id.as_deref() {
        Some(id) => id,
        None => {
            return token_error("invalid_request", "client_id is required");
        }
    };

    // Validate client exists.
    {
        let store = state.token_store.lock().unwrap_or_else(|e| e.into_inner());
        if store.get_client(client_id).is_none() {
            return token_error("invalid_client", "unknown client_id");
        }
    }

    let scope_str = form.scope.as_deref().unwrap_or("read");
    let capability = match scope_str.parse::<Capability>() {
        Ok(cap) => cap,
        Err(_) => {
            return token_error("invalid_scope", "unknown scope");
        }
    };

    let device_code = generate_opaque_token();
    let user_code = generate_user_code();
    let expires_at = unix_now() + DEVICE_CODE_TTL_SECS;

    let pending = PendingDeviceAuth {
        device_code: device_code.clone(),
        user_code: user_code.clone(),
        client_id: client_id.to_owned(),
        scope: capability,
        expires_at,
        approved: None,
        denied: false,
    };

    state
        .device_codes
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(device_code.clone(), pending);

    let url = &state.server_url;
    let formatted_code = format_user_code(&user_code);

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "device_code": device_code,
            "user_code": formatted_code,
            "verification_uri": format!("{url}/oauth/device"),
            "verification_uri_complete": format!("{url}/oauth/device?user_code={formatted_code}"),
            "expires_in": DEVICE_CODE_TTL_SECS,
            "interval": 5,
        })),
    )
        .into_response()
}

/// GET /oauth/device — render the device verification page.
async fn device_page_handler(
    State(state): State<OAuthState>,
    Query(params): Query<HashMap<String, String>>,
) -> Html<String> {
    let prefill = params.get("user_code").cloned().unwrap_or_default();
    Html(device_verification_page(&state.server_url, &prefill))
}

#[derive(Debug, Deserialize)]
pub struct DeviceApproveForm {
    pub user_code: String,
    pub scope: Option<String>,
    pub approved: Option<String>,
}

/// POST /oauth/device — handle device approval.
async fn device_approve_handler(
    State(state): State<OAuthState>,
    Form(form): Form<DeviceApproveForm>,
) -> Html<String> {
    // Normalize user_code: strip dashes, uppercase.
    let normalized = normalize_user_code(&form.user_code);

    let scope_str = form.scope.as_deref().unwrap_or("read");
    let capability = scope_str.parse::<Capability>().unwrap_or(Capability::Read);

    let mut map = state.device_codes.lock().unwrap_or_else(|e| e.into_inner());

    // Find the matching entry by user_code.
    let entry = map
        .values_mut()
        .find(|e| normalize_user_code(&e.user_code) == normalized);

    match entry {
        None => Html(error_page("Unknown or expired user code")),
        Some(entry) if unix_now() > entry.expires_at => Html(error_page("Code has expired")),
        Some(entry) => {
            if form.approved.as_deref() == Some("true") {
                entry.approved = Some(capability);
                info!(
                    "Device authorization approved for user code {}",
                    form.user_code
                );
                Html(device_success_page())
            } else {
                entry.denied = true;
                info!(
                    "Device authorization denied for user code {}",
                    form.user_code
                );
                Html(device_denied_page())
            }
        }
    }
}

// ─── HTML templates ───────────────────────────────────────────────────────────

fn consent_page(
    client_name: &str,
    client_id: &str,
    redirect_uri: &str,
    scope: &str,
    code_challenge: &str,
    state_val: &str,
) -> String {
    let client_name = html_escape(client_name);
    let client_id = html_escape(client_id);
    let redirect_uri = html_escape(redirect_uri);
    let scope = html_escape(scope);
    let code_challenge = html_escape(code_challenge);
    let state_val = html_escape(state_val);
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Authorize Access</title>
  <style>
    body {{ font-family: sans-serif; background: #f5f5f5; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }}
    .card {{ background: #fff; border-radius: 8px; padding: 2rem; max-width: 420px; width: 100%; box-shadow: 0 2px 12px rgba(0,0,0,.12); }}
    h1 {{ font-size: 1.3rem; margin-top: 0; }}
    .scope {{ background: #eef; border-left: 4px solid #448; padding: .5rem 1rem; border-radius: 4px; margin: 1rem 0; }}
    .buttons {{ display: flex; gap: 1rem; margin-top: 1.5rem; }}
    button {{ flex: 1; padding: .7rem; border: none; border-radius: 6px; font-size: 1rem; cursor: pointer; }}
    .approve {{ background: #2a7a2a; color: #fff; }}
    .deny {{ background: #e0e0e0; color: #333; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Authorize Access</h1>
    <p><strong>{client_name}</strong> is requesting access to your VecLayer knowledge store.</p>
    <div class="scope">Requested scope: <strong>{scope}</strong></div>
    <form method="POST" action="/oauth/authorize">
      <input type="hidden" name="client_id" value="{client_id}">
      <input type="hidden" name="redirect_uri" value="{redirect_uri}">
      <input type="hidden" name="scope" value="{scope}">
      <input type="hidden" name="code_challenge" value="{code_challenge}">
      <input type="hidden" name="state" value="{state_val}">
      <div class="buttons">
        <button type="submit" name="approved" value="true" class="approve">Approve</button>
        <button type="submit" name="approved" value="false" class="deny">Deny</button>
      </div>
    </form>
  </div>
</body>
</html>"#
    )
}

fn device_verification_page(server_url: &str, prefill_code: &str) -> String {
    let _ = server_url; // available for future use if needed
    let prefill_code = html_escape(prefill_code);
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Device Authorization</title>
  <style>
    body {{ font-family: sans-serif; background: #f5f5f5; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }}
    .card {{ background: #fff; border-radius: 8px; padding: 2rem; max-width: 420px; width: 100%; box-shadow: 0 2px 12px rgba(0,0,0,.12); }}
    h1 {{ font-size: 1.3rem; margin-top: 0; }}
    label {{ display: block; margin-bottom: .3rem; font-weight: bold; }}
    input[type=text], select {{ width: 100%; box-sizing: border-box; padding: .6rem; border: 1px solid #ccc; border-radius: 4px; font-size: 1rem; margin-bottom: 1rem; }}
    .buttons {{ display: flex; gap: 1rem; margin-top: .5rem; }}
    button {{ flex: 1; padding: .7rem; border: none; border-radius: 6px; font-size: 1rem; cursor: pointer; }}
    .approve {{ background: #2a7a2a; color: #fff; }}
    .deny {{ background: #e0e0e0; color: #333; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Authorize Device</h1>
    <p>Enter the code shown on your device to grant it access.</p>
    <form method="POST" action="/oauth/device">
      <label for="user_code">Device Code</label>
      <input type="text" id="user_code" name="user_code" value="{prefill_code}" placeholder="ABCD-EFGH" required>
      <label for="scope">Access Level</label>
      <select id="scope" name="scope">
        <option value="read">read</option>
        <option value="write">write</option>
        <option value="admin">admin</option>
      </select>
      <div class="buttons">
        <button type="submit" name="approved" value="true" class="approve">Approve</button>
        <button type="submit" name="approved" value="false" class="deny">Deny</button>
      </div>
    </form>
  </div>
</body>
</html>"#
    )
}

fn device_success_page() -> String {
    r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Device Authorized</title>
  <style>
    body { font-family: sans-serif; background: #f5f5f5; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
    .card { background: #fff; border-radius: 8px; padding: 2rem; max-width: 420px; width: 100%; box-shadow: 0 2px 12px rgba(0,0,0,.12); text-align: center; }
    h1 { color: #2a7a2a; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Device Authorized</h1>
    <p>Your device has been successfully authorized. You may close this page.</p>
  </div>
</body>
</html>"#
    .to_owned()
}

fn device_denied_page() -> String {
    r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Access Denied</title>
  <style>
    body { font-family: sans-serif; background: #f5f5f5; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
    .card { background: #fff; border-radius: 8px; padding: 2rem; max-width: 420px; width: 100%; box-shadow: 0 2px 12px rgba(0,0,0,.12); text-align: center; }
    h1 { color: #c0392b; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Access Denied</h1>
    <p>You denied access to the device. You may close this page.</p>
  </div>
</body>
</html>"#
    .to_owned()
}

fn error_page(message: &str) -> String {
    let message = html_escape(message);
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Error</title>
  <style>
    body {{ font-family: sans-serif; background: #f5f5f5; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }}
    .card {{ background: #fff; border-radius: 8px; padding: 2rem; max-width: 420px; width: 100%; box-shadow: 0 2px 12px rgba(0,0,0,.12); }}
    h1 {{ color: #c0392b; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Error</h1>
    <p>{message}</p>
  </div>
</body>
</html>"#
    )
}

// ─── Redirect helpers ─────────────────────────────────────────────────────────

fn redirect_with_code(redirect_uri: &str, code: &str, state: Option<&str>) -> Response {
    let mut url = format!("{redirect_uri}?code={}", urlencoded(code));
    if let Some(s) = state {
        url.push_str(&format!("&state={}", urlencoded(s)));
    }
    Redirect::to(&url).into_response()
}

fn redirect_with_error(
    redirect_uri: &str,
    error: &str,
    description: &str,
    state: Option<&str>,
) -> Response {
    let mut url = format!(
        "{redirect_uri}?error={}&error_description={}",
        urlencoded(error),
        urlencoded(description)
    );
    if let Some(s) = state {
        url.push_str(&format!("&state={}", urlencoded(s)));
    }
    Redirect::to(&url).into_response()
}

// ─── Token error response ─────────────────────────────────────────────────────

fn token_error(error_type: &str, description: &str) -> Response {
    warn!("Token error: {error_type}: {description}");
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({
            "error": error_type,
            "error_description": description,
        })),
    )
        .into_response()
}

// ─── Utilities ────────────────────────────────────────────────────────────────

/// Generate a cryptographically random URL-safe token string.
fn generate_opaque_token() -> String {
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(bytes)
        .chars()
        .take(43)
        .collect()
}

/// Generate an 8-character uppercase user code (no dashes yet — formatted later).
///
/// Uses an unambiguous character set to avoid transcription errors.
/// 8 characters from a 32-symbol alphabet gives ~40 bits of entropy.
fn generate_user_code() -> String {
    const CHARS: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789"; // unambiguous chars
    let mut rng = rand::rngs::OsRng;
    (0..8)
        .map(|_| CHARS[rng.gen_range(0..CHARS.len())] as char)
        .collect()
}

/// Format an 8-char user code as "ABCD-EFGH" (4+4 split).
fn format_user_code(code: &str) -> String {
    let upper = code.to_uppercase().replace('-', "");
    if upper.len() >= 5 {
        format!("{}-{}", &upper[..4], &upper[4..])
    } else {
        upper
    }
}

/// Normalize a user code: strip dashes and uppercase.
fn normalize_user_code(code: &str) -> String {
    code.replace('-', "").to_uppercase()
}

/// Characters that must be percent-encoded in query-string values.
///
/// Extends the CONTROLS base with all RFC 3986 reserved and delimiter
/// characters that would otherwise break query-string parsing.
const QUERY_VALUE: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'%')
    .add(b'&')
    .add(b'+')
    .add(b'/')
    .add(b':')
    .add(b';')
    .add(b'<')
    .add(b'=')
    .add(b'>')
    .add(b'?')
    .add(b'@')
    .add(b'[')
    .add(b'\\')
    .add(b']')
    .add(b'^')
    .add(b'`')
    .add(b'{')
    .add(b'|')
    .add(b'}');

/// Percent-encode a string for safe inclusion as a query-string value.
///
/// Encodes all characters that have special meaning in query strings or
/// URLs (RFC 3986 reserved characters and delimiters), while leaving
/// unreserved characters (letters, digits, `-`, `.`, `_`, `~`) unencoded.
fn urlencoded(s: &str) -> String {
    utf8_percent_encode(s, QUERY_VALUE).to_string()
}

/// Escape a string for safe inclusion in HTML content or attribute values.
///
/// Prevents XSS by replacing characters that have special meaning in HTML.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;
    use sha2::{Digest, Sha256};
    use tempfile::TempDir;
    use tower::ServiceExt;

    use crate::auth::token::verify;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn pkce_pair() -> (String, String) {
        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk".to_owned();
        let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
        (verifier, challenge)
    }

    fn make_state(auto_approve: bool) -> (OAuthState, TempDir) {
        let dir = TempDir::new().expect("tempdir");
        let store = TokenStore::open(dir.path()).expect("open store");
        let signing_key = SigningKey::generate(&mut OsRng);

        let oauth_state = OAuthState {
            token_store: Arc::new(Mutex::new(store)),
            signing_key: Arc::new(signing_key),
            server_did: "did:key:zServer".to_owned(),
            server_url: "http://localhost:8080".to_owned(),
            token_expiry_secs: 3600,
            refresh_expiry_secs: 86400,
            auto_approve,
            device_codes: Arc::new(Mutex::new(HashMap::new())),
        };
        (oauth_state, dir)
    }

    async fn body_vec(resp: axum::response::Response) -> Vec<u8> {
        axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body bytes")
            .to_vec()
    }

    async fn body_json(resp: axum::response::Response) -> serde_json::Value {
        let bytes = body_vec(resp).await;
        serde_json::from_slice(&bytes).expect("parse JSON")
    }

    fn get_req(uri: &str) -> Request<Body> {
        Request::builder()
            .uri(uri)
            .method("GET")
            .body(Body::empty())
            .expect("request")
    }

    fn post_json(uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .uri(uri)
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .expect("request")
    }

    fn post_form(uri: &str, body: &str) -> Request<Body> {
        Request::builder()
            .uri(uri)
            .method("POST")
            .header("content-type", "application/x-www-form-urlencoded")
            .body(Body::from(body.to_owned()))
            .expect("request")
    }

    /// Register a client and return its client_id.
    async fn register_client(app: &Router, redirect_uri: &str) -> String {
        let resp = app
            .clone()
            .oneshot(post_json(
                "/oauth/register",
                serde_json::json!({
                    "client_name": "Test Client",
                    "redirect_uris": [redirect_uri]
                }),
            ))
            .await
            .expect("register");
        let json = body_json(resp).await;
        json["client_id"].as_str().expect("client_id").to_owned()
    }

    // ── test_metadata_endpoint ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_metadata_endpoint() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let resp = app
            .oneshot(get_req("/.well-known/oauth-authorization-server"))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;

        assert_eq!(json["issuer"], "http://localhost:8080");
        assert_eq!(
            json["authorization_endpoint"],
            "http://localhost:8080/oauth/authorize"
        );
        assert_eq!(json["token_endpoint"], "http://localhost:8080/oauth/token");
        assert!(json["grant_types_supported"].is_array());
        let grants = json["grant_types_supported"].as_array().unwrap();
        assert!(grants.iter().any(|g| g == "authorization_code"));
        assert!(grants.iter().any(|g| g == "refresh_token"));
        assert_eq!(json["code_challenge_methods_supported"][0], "S256");
    }

    // ── test_client_registration ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_client_registration() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let resp = app
            .oneshot(post_json(
                "/oauth/register",
                serde_json::json!({
                    "client_name": "My App",
                    "redirect_uris": ["https://example.com/callback"]
                }),
            ))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::CREATED);
        let json = body_json(resp).await;
        assert!(!json["client_id"].as_str().unwrap_or("").is_empty());
        assert_eq!(json["client_name"], "My App");
        assert_eq!(json["redirect_uris"][0], "https://example.com/callback");
    }

    // ── test_authorization_code_flow ──────────────────────────────────────────

    #[tokio::test]
    async fn test_authorization_code_flow() {
        let (state, _dir) = make_state(true /* auto_approve */);
        let signing_key = state.signing_key.clone();
        let server_did = state.server_did.clone();
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;

        let (verifier, challenge) = pkce_pair();

        // Step 1: GET /oauth/authorize — should immediately redirect with code.
        let authorize_uri = format!(
            "/oauth/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope=write&code_challenge={challenge}&code_challenge_method=S256"
        );
        let resp = app
            .clone()
            .oneshot(get_req(&authorize_uri))
            .await
            .expect("authorize");
        assert_eq!(resp.status(), StatusCode::SEE_OTHER);

        let location = resp
            .headers()
            .get("location")
            .expect("location header")
            .to_str()
            .expect("location str");
        assert!(location.starts_with(redirect_uri));
        assert!(location.contains("code="));

        let code = location
            .split("code=")
            .nth(1)
            .unwrap()
            .split('&')
            .next()
            .unwrap()
            .to_owned();

        // Step 2: POST /oauth/token — exchange code for tokens.
        let body = format!(
            "grant_type=authorization_code&code={code}&code_verifier={verifier}&client_id={client_id}&redirect_uri={redirect_uri}"
        );
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/token", &body))
            .await
            .expect("token");
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_json(resp).await;
        let access_token = json["access_token"].as_str().expect("access_token");
        assert!(!access_token.is_empty());
        assert_eq!(json["token_type"], "Bearer");
        assert!(json["refresh_token"].as_str().is_some());

        // Step 3: Verify the JWT is valid.
        let claims = verify(
            access_token,
            &signing_key.verifying_key(),
            Some(&server_did),
        )
        .expect("valid JWT");
        assert_eq!(claims.cap, Capability::Write);
    }

    // ── test_pkce_required ────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_pkce_required() {
        let (state, _dir) = make_state(true);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;

        // Authorize with a valid PKCE pair first.
        let (_, challenge) = pkce_pair();
        let authorize_uri = format!(
            "/oauth/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope=read&code_challenge={challenge}&code_challenge_method=S256"
        );
        let resp = app
            .clone()
            .oneshot(get_req(&authorize_uri))
            .await
            .expect("authorize");
        let location = resp
            .headers()
            .get("location")
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        let code = location
            .split("code=")
            .nth(1)
            .unwrap()
            .split('&')
            .next()
            .unwrap()
            .to_owned();

        // Exchange WITHOUT code_verifier.
        let body = format!(
            "grant_type=authorization_code&code={code}&client_id={client_id}&redirect_uri={redirect_uri}"
        );
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/token", &body))
            .await
            .expect("token");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "invalid_request");
    }

    // ── test_refresh_token_rotation ───────────────────────────────────────────

    #[tokio::test]
    async fn test_refresh_token_rotation() {
        let (state, _dir) = make_state(true);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (verifier, challenge) = pkce_pair();

        // Authorize and get initial tokens.
        let authorize_uri = format!(
            "/oauth/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope=read&code_challenge={challenge}&code_challenge_method=S256"
        );
        let resp = app
            .clone()
            .oneshot(get_req(&authorize_uri))
            .await
            .expect("authorize");
        let location = resp
            .headers()
            .get("location")
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        let code = location
            .split("code=")
            .nth(1)
            .unwrap()
            .split('&')
            .next()
            .unwrap()
            .to_owned();

        let body = format!("grant_type=authorization_code&code={code}&code_verifier={verifier}&client_id={client_id}&redirect_uri={redirect_uri}");
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/token", &body))
            .await
            .expect("token");
        let json = body_json(resp).await;
        let refresh1 = json["refresh_token"].as_str().unwrap().to_owned();

        // Use refresh token to get new tokens.
        let body =
            format!("grant_type=refresh_token&refresh_token={refresh1}&client_id={client_id}");
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/token", &body))
            .await
            .expect("refresh");
        assert_eq!(resp.status(), StatusCode::OK);
        let json2 = body_json(resp).await;
        let refresh2 = json2["refresh_token"].as_str().unwrap().to_owned();

        // New refresh token must differ.
        assert_ne!(refresh1, refresh2);

        // Old refresh token must now be invalid.
        let body =
            format!("grant_type=refresh_token&refresh_token={refresh1}&client_id={client_id}");
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/token", &body))
            .await
            .expect("stale refresh");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json3 = body_json(resp).await;
        assert_eq!(json3["error"], "invalid_grant");
    }

    // ── test_device_flow ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_device_flow() {
        let (state, _dir) = make_state(false);
        let signing_key = state.signing_key.clone();
        let server_did = state.server_did.clone();
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;

        // Step 1: Request device code.
        let body = format!("client_id={client_id}&scope=read");
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/device/code", &body))
            .await
            .expect("device/code");
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        let device_code = json["device_code"]
            .as_str()
            .expect("device_code")
            .to_owned();
        let user_code = json["user_code"].as_str().expect("user_code").to_owned();

        assert!(json["verification_uri"].as_str().is_some());
        assert_eq!(json["expires_in"], 600);

        // Step 2: Poll before approval — expect authorization_pending.
        let body = format!(
            "grant_type=urn:ietf:params:oauth:grant-type:device_code&device_code={device_code}&client_id={client_id}"
        );
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/token", &body))
            .await
            .expect("poll pending");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json_pending = body_json(resp).await;
        assert_eq!(json_pending["error"], "authorization_pending");

        // Step 3: Approve the device via the browser flow.
        let approve_body = format!("user_code={user_code}&scope=read&approved=true");
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/device", &approve_body))
            .await
            .expect("approve");
        assert_eq!(resp.status(), StatusCode::OK);
        let html = body_vec(resp).await;
        assert!(String::from_utf8_lossy(&html).contains("Authorized"));

        // Step 4: Poll again — expect tokens.
        let body = format!(
            "grant_type=urn:ietf:params:oauth:grant-type:device_code&device_code={device_code}&client_id={client_id}"
        );
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/token", &body))
            .await
            .expect("poll approved");
        assert_eq!(resp.status(), StatusCode::OK);
        let json_tokens = body_json(resp).await;
        let access_token = json_tokens["access_token"].as_str().expect("access_token");
        assert!(!access_token.is_empty());

        // Verify the JWT.
        let claims = verify(
            access_token,
            &signing_key.verifying_key(),
            Some(&server_did),
        )
        .expect("valid JWT");
        assert_eq!(claims.cap, Capability::Read);
    }

    // ── test_html_escape ──────────────────────────────────────────────────────

    #[test]
    fn test_html_escape_encodes_xss_vectors() {
        assert_eq!(html_escape("<script>"), "&lt;script&gt;");
        assert_eq!(html_escape("a & b"), "a &amp; b");
        assert_eq!(html_escape(r#"val="x""#), "val=&quot;x&quot;");
        assert_eq!(html_escape("it's"), "it&#x27;s");
        assert_eq!(html_escape("safe text"), "safe text");
    }

    // ── test_open_redirect_blocked ────────────────────────────────────────────

    #[tokio::test]
    async fn test_open_redirect_blocked_missing_code_challenge() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;

        // Request /oauth/authorize without code_challenge but with a valid
        // client_id — should NOT redirect to the supplied redirect_uri with
        // an error (open redirect), it should redirect to the validated URI.
        // Even after the fix (which does redirect after client validation),
        // the redirect target must be the registered redirect_uri, not
        // an attacker-controlled one.
        let evil_uri = "https://evil.com/steal";
        let authorize_uri = format!(
            "/oauth/authorize?response_type=code&client_id={client_id}&redirect_uri={evil_uri}"
        );
        let resp = app
            .clone()
            .oneshot(get_req(&authorize_uri))
            .await
            .expect("request");

        // Evil redirect_uri is not registered, so we must get a 400 error page.
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "unregistered redirect_uri must return 400, not redirect to evil.com"
        );
    }

    #[tokio::test]
    async fn test_open_redirect_blocked_invalid_client() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        // Unknown client_id with a redirect_uri — must return 400, not redirect.
        let authorize_uri = "/oauth/authorize?response_type=code&client_id=unknown-client&redirect_uri=https%3A%2F%2Fevil.com";
        let resp = app
            .clone()
            .oneshot(get_req(authorize_uri))
            .await
            .expect("request");

        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "unknown client_id must return 400"
        );
    }

    // ── test_response_type_required ───────────────────────────────────────────

    #[tokio::test]
    async fn test_unsupported_response_type_returns_error_redirect() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (_, challenge) = pkce_pair();

        let authorize_uri = format!(
            "/oauth/authorize?response_type=token&client_id={client_id}&redirect_uri={redirect_uri}&code_challenge={challenge}&code_challenge_method=S256"
        );
        let resp = app
            .clone()
            .oneshot(get_req(&authorize_uri))
            .await
            .expect("request");

        assert_eq!(
            resp.status(),
            StatusCode::SEE_OTHER,
            "unsupported response_type must redirect"
        );
        let location = resp
            .headers()
            .get("location")
            .expect("location header")
            .to_str()
            .unwrap();
        assert!(
            location.contains("error=unsupported_response_type"),
            "redirect must contain unsupported_response_type, got: {location}"
        );
    }

    // ── test_user_code_length ─────────────────────────────────────────────────

    #[test]
    fn test_user_code_is_8_chars_with_hyphen() {
        for _ in 0..20 {
            let code = generate_user_code();
            assert_eq!(code.len(), 8, "raw user code must be 8 chars");
            let formatted = format_user_code(&code);
            // Format: XXXX-XXXX
            assert_eq!(
                formatted.len(),
                9,
                "formatted user code must be 9 chars (XXXX-XXXX), got: {formatted}"
            );
            assert_eq!(
                formatted.chars().nth(4),
                Some('-'),
                "hyphen must be at position 4, got: {formatted}"
            );
        }
    }

    // ── test_urlencoded ───────────────────────────────────────────────────────

    #[test]
    fn test_urlencoded_encodes_reserved_characters() {
        assert!(urlencoded("a&b").contains("%26"));
        assert!(urlencoded("a=b").contains("%3D"));
        assert!(urlencoded("a b").contains("%20"));
        assert!(urlencoded("a#b").contains("%23"));
        assert!(urlencoded("a%b").contains("%25"));
        // Unreserved characters must NOT be encoded.
        assert_eq!(urlencoded("abc-123_~."), "abc-123_~.");
    }
}
