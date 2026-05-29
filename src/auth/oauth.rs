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

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{Form, Query, State};
use axum::http::header::{CACHE_CONTROL, PRAGMA};
use axum::http::{HeaderValue, StatusCode};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine;
use ed25519_dalek::SigningKey;
use percent_encoding::{utf8_percent_encode, AsciiSet, CONTROLS};
use rand_core::{OsRng, RngCore};
use serde::Deserialize;

use tracing::{error, info, warn};

use super::capability::Capability;
use super::token::{mint, Claims};
use super::token_store::TokenStore;
use crate::util::unix_now;

// ─── Device expiry ────────────────────────────────────────────────────────────

const DEVICE_CODE_TTL_SECS: u64 = 600;

/// Minimum seconds between consecutive token-endpoint polls for the same
/// device code (RFC 8628 §3.5 `slow_down` interval).  Must match the
/// `"interval"` value advertised in the device-code response.
const DEVICE_POLL_INTERVAL_SECS: u64 = 5;

// ─── Security helpers ─────────────────────────────────────────────────────────

/// Returns `true` when `server_url` resolves to a loopback address.
///
/// Accepts the full server URL (e.g. `http://localhost:8080`) or a bare host.
/// Matches `localhost`, `127.x.x.x`, and `::1`.
#[must_use]
pub fn is_localhost(server_url: &str) -> bool {
    // Strip scheme prefix if present.
    let after_scheme = server_url
        .strip_prefix("http://")
        .or_else(|| server_url.strip_prefix("https://"))
        .unwrap_or(server_url);

    // Strip path: take everything up to the first '/'.
    let authority = after_scheme.split('/').next().unwrap_or(after_scheme);

    // IPv6 literals in URLs are enclosed in brackets: `[::1]:8080`.
    // Bare `::1` (no brackets) is also accepted as a loopback host.
    let host = if authority.starts_with('[') {
        // Bracketed IPv6: strip `[` and take until `]`.
        authority
            .trim_start_matches('[')
            .split(']')
            .next()
            .unwrap_or(authority)
    } else if authority.contains(':') && authority.matches(':').count() > 1 {
        // Bare IPv6 address (multiple colons, no brackets) — use as-is.
        authority
    } else if let Some((h, _port)) = authority.rsplit_once(':') {
        // IPv4 or hostname with optional port: `localhost:8080` -> `localhost`.
        h
    } else {
        authority
    };

    is_loopback_host(host)
}

/// Validate a redirect URI for OAuth client registration.
///
/// Allowed:
/// - `https://` (any host, no embedded credentials or fragment)
/// - `http://localhost` or `http://127.x.x.x` (no embedded credentials or fragment)
/// - Custom app schemes (e.g. `myapp://`)
///
/// Rejected:
/// - `javascript:`, `data:`, `file:`
/// - `http://` with a non-localhost host
/// - Any URI containing embedded userinfo (`user:pass@` or `user@`)
/// - Any URI containing a fragment component (`#`)
#[must_use]
pub fn validate_redirect_uri(uri: &str) -> bool {
    let lower = uri.to_lowercase();

    // Reject dangerous schemes unconditionally.
    if lower.starts_with("javascript:") || lower.starts_with("data:") || lower.starts_with("file:")
    {
        return false;
    }

    // Reject fragment components: OAuth 2.0 (RFC 6749 §3.1.2) prohibits
    // fragments in redirect URIs.  A fragment in a registered URI cannot be
    // matched exactly and can interfere with the authorization response.
    if uri.contains('#') {
        return false;
    }

    // Reject embedded userinfo (credentials) in the authority component.
    // `https://user:pass@evil.com/steal` would otherwise pass the https check
    // and could be used for credential-harvesting open redirects.
    //
    // Strategy: check for `@` between `://` and the first `/` (the authority).
    if let Some(after_scheme) = lower
        .strip_prefix("https://")
        .or_else(|| lower.strip_prefix("http://"))
    {
        let authority = after_scheme.split('/').next().unwrap_or(after_scheme);
        if authority.contains('@') {
            return false;
        }
    }

    // HTTPS is allowed for any host (after the above checks).
    if lower.starts_with("https://") {
        return true;
    }

    // HTTP is only allowed for loopback addresses (IPv4 127.0.0.0/8, IPv6
    // `[::1]`, or `localhost`) per RFC 8252 §8.3.
    if let Some(rest) = lower.strip_prefix("http://") {
        let authority = rest.split('/').next().unwrap_or(rest);
        return is_loopback_host(authority_host(authority));
    }

    // Everything else (custom app schemes like `myapp://`) is allowed,
    // provided it is not one of the rejected schemes above.
    uri.contains("://")
}

/// Extract the bare host from an HTTP authority, dropping any `:port` suffix
/// and unwrapping an IPv6 literal: `[::1]:8080` → `::1`, `127.0.0.1:80` →
/// `127.0.0.1`, `localhost` → `localhost`.
fn authority_host(authority: &str) -> &str {
    if let Some(after_bracket) = authority.strip_prefix('[') {
        // IPv6 literal: the host is everything up to the closing bracket.
        after_bracket.split(']').next().unwrap_or(after_bracket)
    } else {
        // Hostname or IPv4: strip an optional `:port` suffix.
        authority.split(':').next().unwrap_or(authority)
    }
}

/// Returns `true` when `host` is a loopback host: the literal `localhost`, or
/// any address in IPv4 `127.0.0.0/8` or IPv6 `::1`.
///
/// Parses the host as an `IpAddr` so the 127-octet check is exact: a substring
/// match like `starts_with("127.")` would wrongly accept `127.0.0.1.evil.com`,
/// a remote host an attacker could register to receive auth codes over plain
/// HTTP. A non-numeric host that is not `localhost` fails to parse and is
/// rejected.
fn is_loopback_host(host: &str) -> bool {
    if host == "localhost" {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .is_ok_and(|ip| ip.is_loopback())
}

// ─── Bounded FIFO collections ────────────────────────────────────────────────

/// An insertion-ordered map with a hard capacity cap.
///
/// On each insert, if `len >= cap`, the single oldest entry is evicted before
/// the new one is added.  This is strictly better than `.clear()`: in-flight
/// sessions survive a burst; only the very oldest entry is displaced.
///
/// Eviction is FIFO by *first* insertion, not LRU: re-inserting an existing key
/// updates its value but keeps its original eviction position.
pub struct FifoMap<K, V> {
    map: HashMap<K, V>,
    order: VecDeque<K>,
    cap: usize,
}

impl<K: std::hash::Hash + Eq + Clone, V> FifoMap<K, V> {
    pub fn new(cap: usize) -> Self {
        assert!(cap > 0, "FifoMap capacity must be non-zero");
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            cap,
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn insert(&mut self, key: K, value: V) {
        if let Some(slot) = self.map.get_mut(&key) {
            *slot = value;
            return;
        }
        if self.map.len() >= self.cap {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            }
        }
        self.order.push_back(key.clone());
        self.map.insert(key, value);
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: std::borrow::Borrow<Q>,
        Q: std::hash::Hash + Eq + ?Sized,
    {
        if let Some(pos) = self.order.iter().position(|k| k.borrow() == key) {
            self.order.remove(pos);
        }
        self.map.remove(key)
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: std::borrow::Borrow<Q>,
        Q: std::hash::Hash + Eq + ?Sized,
    {
        self.map.get(key)
    }
}

/// An insertion-ordered set with a hard capacity cap.
///
/// On each insert, if `len >= cap`, the single oldest entry is evicted.
pub struct FifoSet<K> {
    inner: FifoMap<K, ()>,
}

impl<K: std::hash::Hash + Eq + Clone> FifoSet<K> {
    pub fn new(cap: usize) -> Self {
        Self {
            inner: FifoMap::new(cap),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn insert(&mut self, key: K) {
        self.inner.insert(key, ());
    }

    pub fn remove<Q>(&mut self, key: &Q) -> bool
    where
        K: std::borrow::Borrow<Q>,
        Q: std::hash::Hash + Eq + ?Sized,
    {
        self.inner.remove(key).is_some()
    }

    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        K: std::borrow::Borrow<Q>,
        Q: std::hash::Hash + Eq + ?Sized,
    {
        self.inner.get(key).is_some()
    }
}

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
    /// Pending consent page sessions keyed by CSRF token.
    pub pending_consents: Arc<Mutex<FifoMap<String, PendingConsent>>>,
    /// One-time CSRF tokens issued by the device verification page (GET),
    /// validated and consumed on the device approval POST.
    pub device_csrf_tokens: Arc<Mutex<FifoSet<String>>>,
}

/// Upper bound on outstanding device-page CSRF tokens.  When the cap is hit,
/// the oldest entry is evicted before the new one is inserted, so a stale
/// page simply needs reloading without disrupting all other in-flight sessions.
pub const MAX_DEVICE_CSRF_TOKENS: usize = 1024;

/// Upper bound on outstanding (unconsumed, unexpired) device authorization
/// codes. Expired entries are purged before this cap is checked, so it only
/// bites under a flood of fresh requests — bounding memory against a client
/// that loops the device-code endpoint.
const MAX_DEVICE_CODES: usize = 1024;

/// Maximum byte length accepted for a user_code submission on the device
/// approval page.  A real user code is 9 characters (XXXX-XXXX); this limit
/// prevents large allocations in `normalize_user_code` and O(n) map scans.
const MAX_USER_CODE_LENGTH: usize = 64;

// ─── Consent CSRF tracking ────────────────────────────────────────────────────

/// Maximum length for the OAuth `state` parameter (RFC 6749 does not mandate a
/// limit but unbounded values are a DoS/injection risk).
const MAX_STATE_LEN: usize = 512;

/// Upper bound on outstanding consent sessions.  When the cap is hit, the
/// oldest entry is evicted before the new one is inserted, so a burst of
/// authorizations displaces only the very oldest session rather than all of them.
pub const MAX_PENDING_CONSENTS: usize = 1024;

/// Short-lived record created when the consent page is rendered.
///
/// The `csrf_token` is embedded in the HTML form and validated on POST so that
/// a forged cross-site request cannot submit a consent decision.
pub struct PendingConsent {
    pub csrf_token: String,
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
    /// When this code was last polled at the token endpoint.  `None` on first
    /// poll.  Used to enforce RFC 8628 §3.5 `slow_down` rate limiting.
    pub last_polled: Option<Instant>,
}

// ─── Router ───────────────────────────────────────────────────────────────────

/// Build the OAuth 2.0 router.
///
/// Merge this into the main application router:
/// ```ignore
/// let app = build_app(state).merge(oauth_router(oauth_state));
/// ```
///
/// # Panics
///
/// Panics if `auto_approve=true` and `server_url` is not a localhost address.
pub fn oauth_router(state: OAuthState) -> Router {
    if state.auto_approve {
        if !is_localhost(&state.server_url) {
            tracing::error!(
                server_url = %state.server_url,
                "auto_approve=true is only allowed on localhost — refusing to start"
            );
            panic!(
                "Security misconfiguration: auto_approve=true is not permitted on non-localhost \
                 address '{}'. Set auto_approve=false or bind to localhost.",
                state.server_url
            );
        }
        warn!(
            server_url = %state.server_url,
            "auto_approve=true: OAuth consent is bypassed — do not use in production"
        );
    }

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

/// Maximum number of redirect URIs allowed per client registration.
///
/// Prevents a single registration from submitting thousands of 2 KiB URIs and
/// growing oauth_store.json unboundedly (DoS).
const MAX_REDIRECT_URIS_PER_CLIENT: usize = 10;

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

    if body.redirect_uris.len() > MAX_REDIRECT_URIS_PER_CLIENT {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "invalid_client_metadata",
                "error_description": format!(
                    "too many redirect_uris; maximum allowed is {}",
                    MAX_REDIRECT_URIS_PER_CLIENT
                )
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

    if let Some(invalid) = body
        .redirect_uris
        .iter()
        .find(|u| !validate_redirect_uri(u))
    {
        warn!("Client registration rejected: invalid redirect_uri scheme: {invalid}");
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "invalid_redirect_uri",
                "error_description": format!(
                    "redirect_uri '{}' uses a disallowed scheme; \
                     only https://, http://localhost, and custom app schemes are permitted",
                    invalid
                )
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

    let client = match store.register_client(&body.client_name, body.redirect_uris) {
        Ok(client) => client,
        Err(e) => {
            warn!("OAuth client registration failed to persist: {e}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "server_error",
                    "error_description": "failed to persist client registration"
                })),
            )
                .into_response();
        }
    };

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
            return secure_html_response_with_status(
                StatusCode::BAD_REQUEST,
                error_page("Missing client_id parameter"),
            )
        }
    };

    let redirect_uri = match &params.redirect_uri {
        Some(uri) => uri.clone(),
        None => {
            return secure_html_response_with_status(
                StatusCode::BAD_REQUEST,
                error_page("Missing redirect_uri parameter"),
            )
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
                    return secure_html_response_with_status(
                        StatusCode::BAD_REQUEST,
                        error_page("redirect_uri does not match registered URIs"),
                    );
                }
                client.client_name.clone()
            }
            None => {
                return secure_html_response_with_status(
                    StatusCode::BAD_REQUEST,
                    error_page("Unknown client_id"),
                );
            }
        }
    };

    // Step 3: Validate the optional `state` parameter length.
    if let Some(s) = params.state.as_deref() {
        if s.len() > MAX_STATE_LEN {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "invalid_request",
                    "error_description": "state parameter too long (max 512 bytes)"
                })),
            )
                .into_response();
        }
    }

    // Step 4: Check response_type — only "code" is supported.
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

    if code_challenge.len() > MAX_STRING_LENGTH {
        return redirect_with_error(
            &redirect_uri,
            "invalid_request",
            "code_challenge too long",
            params.state.as_deref(),
        );
    }

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
    let Ok(capability) = scope_str.parse::<Capability>() else {
        return redirect_with_error(
            &redirect_uri,
            "invalid_scope",
            "unknown scope",
            params.state.as_deref(),
        );
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

        return match code {
            Ok(code) => redirect_with_code(&redirect_uri, &code, params.state.as_deref()),
            Err(e) => {
                warn!("Authorization code failed to persist: {e}");
                redirect_with_error(
                    &redirect_uri,
                    "server_error",
                    "failed to persist authorization code",
                    params.state.as_deref(),
                )
            }
        };
    }

    // Generate a CSRF token for this consent session and store it server-side
    // so the POST handler can validate it (prevents cross-site consent forgery).
    let csrf_token = generate_opaque_token();
    {
        let mut consents = state
            .pending_consents
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        consents.insert(
            csrf_token.clone(),
            PendingConsent {
                csrf_token: csrf_token.clone(),
            },
        );
    }

    // Show consent page.
    secure_html_response(consent_page(
        &client_name,
        &client_id,
        &redirect_uri,
        scope_str,
        &code_challenge,
        params.state.as_deref().unwrap_or(""),
        &csrf_token,
    ))
}

#[derive(Debug, Deserialize)]
pub struct ConsentForm {
    pub client_id: String,
    pub redirect_uri: String,
    pub scope: String,
    pub code_challenge: String,
    pub state: Option<String>,
    pub approved: Option<String>,
    /// CSRF token generated when the consent page was rendered.
    ///
    /// `Option` so that a missing field produces a controlled 400 rather than
    /// axum's default 422 Unprocessable Entity.
    pub csrf_token: Option<String>,
}

/// POST /oauth/authorize — handle consent form submission.
async fn authorize_post_handler(
    State(state): State<OAuthState>,
    Form(form): Form<ConsentForm>,
) -> Response {
    let oauth_state = form.state.as_deref();

    // Validate CSRF token FIRST — before any redirect — to reject forged requests.
    // A missing or unknown token both result in 400 Bad Request.
    {
        let csrf_key = match form.csrf_token.as_deref() {
            Some(k) if !k.is_empty() => k.to_owned(),
            _ => {
                return secure_html_response_with_status(
                    StatusCode::BAD_REQUEST,
                    error_page("Invalid or missing CSRF token"),
                );
            }
        };
        let mut consents = state
            .pending_consents
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if consents.remove(&csrf_key).is_none() {
            return secure_html_response_with_status(
                StatusCode::BAD_REQUEST,
                error_page("Invalid or missing CSRF token"),
            );
        }
    }

    // Validate client and redirect_uri BEFORE any redirect to prevent open
    // redirect via a crafted POST with an arbitrary redirect_uri.
    {
        let store = state.token_store.lock().unwrap_or_else(|e| e.into_inner());
        match store.get_client(&form.client_id) {
            Some(client) => {
                if !client.redirect_uris.contains(&form.redirect_uri) {
                    return secure_html_response_with_status(
                        StatusCode::BAD_REQUEST,
                        error_page("redirect_uri does not match registered URIs"),
                    );
                }
            }
            None => {
                return secure_html_response_with_status(
                    StatusCode::BAD_REQUEST,
                    error_page("Unknown client_id"),
                );
            }
        }
    }

    let redirect_uri = form.redirect_uri.clone();

    // Validate the optional `state` parameter length — mirrors the GET-path
    // guard so an attacker who obtained a valid CSRF token cannot smuggle an
    // arbitrarily long state through the POST body.
    if let Some(s) = form.state.as_deref() {
        if s.len() > MAX_STATE_LEN {
            return secure_html_response_with_status(
                StatusCode::BAD_REQUEST,
                error_page("state parameter too long"),
            );
        }
    }

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

    // Bound the PKCE challenge before it is persisted — mirrors the GET-path
    // guard so an oversized challenge cannot be smuggled in via the consent POST.
    if form.code_challenge.len() > MAX_STRING_LENGTH {
        return redirect_with_error(
            &redirect_uri,
            "invalid_request",
            "code_challenge too long",
            oauth_state,
        );
    }

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

    match code {
        Ok(code) => redirect_with_code(&redirect_uri, &code, oauth_state),
        Err(e) => {
            warn!("Authorization code failed to persist: {e}");
            redirect_with_error(
                &redirect_uri,
                "server_error",
                "failed to persist authorization code",
                oauth_state,
            )
        }
    }
}

// ─── Token Endpoint ───────────────────────────────────────────────────────────

#[derive(Deserialize)]
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

impl std::fmt::Debug for TokenRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenRequest")
            .field("grant_type", &self.grant_type)
            .field("client_id", &self.client_id)
            .field("redirect_uri", &self.redirect_uri)
            .field("code", &self.code.as_ref().map(|_| "<redacted>"))
            .field(
                "code_verifier",
                &self.code_verifier.as_ref().map(|_| "<redacted>"),
            )
            .field(
                "refresh_token",
                &self.refresh_token.as_ref().map(|_| "<redacted>"),
            )
            .field(
                "device_code",
                &self.device_code.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
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
        other => {
            warn!("Token request with unsupported grant_type: {other}");
            token_error("unsupported_grant_type", "unsupported grant_type")
        }
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
                warn!("Authorization code exchange failed: {e}");
                return token_error("invalid_grant", "invalid authorization code");
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
    // RFC 6749 §4.1.3: when the authorization request used a redirect_uri,
    // the token request must include the identical value. Only checking it
    // when the client chooses to send it weakens code-injection defenses.
    if !auth_code.redirect_uri.is_empty() {
        match form.redirect_uri.as_deref() {
            Some(uri) if uri == auth_code.redirect_uri => {}
            Some(_) => return token_error("invalid_grant", "redirect_uri mismatch"),
            None => return token_error("invalid_request", "redirect_uri is required"),
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
    // client_id is required for public clients (RFC 6749 §4.1.3).
    let form_cid = match form.client_id.as_deref() {
        Some(cid) => cid,
        None => {
            return token_error("invalid_request", "client_id is required");
        }
    };

    // Verify the client binding BEFORE revoking, then revoke — all under one
    // lock so check-then-revoke stays atomic (no TOCTOU). A wrong/malicious
    // client_id is rejected without burning the legitimate owner's still-valid
    // token (denial-of-service vector).
    let (client_id, did, capability) = {
        let mut store = state.token_store.lock().unwrap_or_else(|e| e.into_inner());
        match store.refresh_token_client_id(refresh) {
            Ok(bound_cid) if bound_cid.as_str() == form_cid => {}
            Ok(_) => return token_error("invalid_client", "client_id mismatch"),
            Err(e) => {
                warn!("Refresh token exchange failed: {e}");
                return token_error("invalid_grant", "invalid refresh token");
            }
        }
        match store.validate_and_revoke_refresh(refresh) {
            Ok(record) => record,
            Err(e) => {
                warn!("Refresh token exchange failed: {e}");
                return token_error("invalid_grant", "invalid refresh token");
            }
        }
    };

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

    // Validate client_id (required for public clients per RFC 6749 §4.1.3).
    let form_client_id = match form.client_id.as_deref() {
        Some(cid) => cid,
        None => {
            return token_error("invalid_request", "client_id is required");
        }
    };

    let now = unix_now();
    let poll_interval = Duration::from_secs(DEVICE_POLL_INTERVAL_SECS);

    let (client_id, capability) = {
        let mut map = state.device_codes.lock().unwrap_or_else(|e| e.into_inner());

        // Validate before mutating; collect what we need and drop the reference
        // before any `map.remove` call (avoids two simultaneous mutable borrows).
        let (bound_cid, expires_at, denied, approved, too_fast) = {
            let entry = match map.get(device_code_str) {
                Some(e) => e,
                None => return token_error("invalid_grant", "unknown device_code"),
            };
            let too_fast = entry
                .last_polled
                .map(|t| t.elapsed() < poll_interval)
                .unwrap_or(false);
            (
                entry.client_id.clone(),
                entry.expires_at,
                entry.denied,
                entry.approved,
                too_fast,
            )
        };

        // Reject a mismatched client_id first, before disclosing expiry/denial
        // state — a wrong client must not learn the device code's status. This
        // path deliberately returns before the last_polled stamp below: the
        // response is cheap and stateless, and stamping here would let anyone
        // who guessed the device_code pin the real client in perpetual
        // slow_down by polling with a bogus client_id.
        if form_client_id != bound_cid {
            return token_error("invalid_client", "client_id mismatch");
        }

        if now > expires_at {
            map.remove(device_code_str);
            return token_error("expired_token", "device authorization expired");
        }

        if denied {
            map.remove(device_code_str);
            return token_error("access_denied", "user denied access");
        }

        // RFC 8628 §3.5: stamp last_polled on every poll, even one that will be
        // rejected as slow_down below. The throttle window restarts from each
        // request, so a client that keeps polling under the interval keeps
        // getting slow_down until it actually waits the full interval.
        if let Some(entry) = map.get_mut(device_code_str) {
            entry.last_polled = Some(Instant::now());
        }

        if too_fast {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "slow_down",
                    "error_description": "polling too fast; wait at least 5 seconds between requests"
                })),
            )
                .into_response();
        }

        match approved {
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
                map.remove(device_code_str);
                (bound_cid, cap)
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
        state.server_did.clone(),
        did.to_owned(),
        state.server_did.clone(),
        capability,
        now,
        exp,
    );

    let access_token = match mint(&state.signing_key, &claims) {
        Ok(t) => t,
        Err(e) => {
            error!("Failed to mint access token: {e}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "server_error",
                    "error_description": "failed to issue token"
                })),
            )
                .into_response();
        }
    };

    // Generate a random refresh token.
    let refresh_token = generate_opaque_token();
    let refresh_exp = now + state.refresh_expiry_secs;

    if let Err(e) = state
        .token_store
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .store_refresh(&refresh_token, client_id, did, capability, refresh_exp)
    {
        warn!("Refresh token failed to persist: {e}");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": "server_error",
                "error_description": "failed to persist refresh token"
            })),
        )
            .into_response();
    }

    let mut resp = (
        StatusCode::OK,
        Json(serde_json::json!({
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": state.token_expiry_secs,
            "refresh_token": refresh_token,
            "scope": capability.to_string(),
        })),
    )
        .into_response();

    // RFC 6749 §5.1 — token responses MUST NOT be cached.
    let headers = resp.headers_mut();
    headers.insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert(PRAGMA, HeaderValue::from_static("no-cache"));

    resp
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
        last_polled: None,
    };

    {
        let mut map = state.device_codes.lock().unwrap_or_else(|e| e.into_inner());
        // Drop expired entries first so a steady stream of legitimate,
        // short-lived codes never trips the cap; then bound the live set.
        map.retain(|_, e| unix_now() <= e.expires_at);
        if map.len() >= MAX_DEVICE_CODES {
            return token_error(
                "temporarily_unavailable",
                "too many pending device authorizations; try again shortly",
            );
        }
        map.insert(device_code.clone(), pending);
    }

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
) -> Response {
    let prefill = params.get("user_code").cloned().unwrap_or_default();

    // Issue a one-time CSRF token so a forged cross-site POST cannot submit a
    // device-approval decision on the victim's behalf.
    let csrf_token = generate_opaque_token();
    {
        let mut tokens = state
            .device_csrf_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        tokens.insert(csrf_token.clone());
    }

    secure_html_response(device_verification_page(&prefill, &csrf_token))
}

#[derive(Debug, Deserialize)]
pub struct DeviceApproveForm {
    pub user_code: String,
    // NOTE: no `scope` field here by design.
    //
    // The authoritative requested scope is stored server-side in
    // `PendingDeviceAuth.scope` when the device calls `/oauth/device/code`.
    // Accepting a user-supplied scope in the approval form would allow
    // a malicious page or user to escalate beyond what the device requested
    // (e.g. submitting `scope=admin` for a device that only asked for `scope=read`).
    // Any `scope` key in the POST body is silently ignored by serde.
    pub approved: Option<String>,
    pub csrf_token: Option<String>,
}

/// POST /oauth/device — handle device approval.
async fn device_approve_handler(
    State(state): State<OAuthState>,
    Form(form): Form<DeviceApproveForm>,
) -> Response {
    // Validate and consume the one-time CSRF token before acting on the form,
    // so a forged cross-site POST is rejected.
    let csrf_ok = form
        .csrf_token
        .as_deref()
        .map(|t| {
            state
                .device_csrf_tokens
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(t)
        })
        .unwrap_or(false);
    if !csrf_ok {
        warn!("Device approval rejected: missing or invalid CSRF token");
        return secure_html_response(error_page(
            "Invalid or expired request. Reload the device authorization page and try again.",
        ));
    }

    // Reject oversized user_code before any allocation-heavy normalization or
    // O(n) map scan.  Real codes are at most 9 bytes (XXXX-XXXX).
    if form.user_code.len() > MAX_USER_CODE_LENGTH {
        warn!(
            "Device approval rejected: user_code too long ({} bytes)",
            form.user_code.len()
        );
        return secure_html_response(error_page("Invalid device code"));
    }

    // Normalize user_code: strip dashes, uppercase.
    let normalized = normalize_user_code(&form.user_code);

    let mut map = state.device_codes.lock().unwrap_or_else(|e| e.into_inner());

    // Find the matching entry by user_code.
    let entry = map
        .values_mut()
        .find(|e| normalize_user_code(&e.user_code) == normalized);

    match entry {
        None => secure_html_response(error_page("Unknown or expired user code")),
        Some(entry) if unix_now() > entry.expires_at => {
            secure_html_response(error_page("Code has expired"))
        }
        Some(entry) if entry.approved.is_some() || entry.denied => {
            // Already decided — a second POST must not flip or re-confirm it.
            secure_html_response(error_page("This code has already been processed"))
        }
        Some(entry) => {
            if form.approved.as_deref() == Some("true") {
                // Use the server-side stored scope, not any form-supplied value.
                // This prevents a malicious approval POST from escalating beyond
                // the scope the device originally requested.
                let granted_scope = entry.scope;
                entry.approved = Some(granted_scope);
                info!(
                    "Device authorization approved for user code {} with scope {:?}",
                    form.user_code, granted_scope
                );
                secure_html_response(device_success_page())
            } else {
                entry.denied = true;
                info!(
                    "Device authorization denied for user code {}",
                    form.user_code
                );
                secure_html_response(device_denied_page())
            }
        }
    }
}

// ─── Secure HTML response helper ─────────────────────────────────────────────

/// Wraps an HTML page string into a `Response` with anti-clickjacking and
/// content-security headers.
///
/// Every HTML response served by auth endpoints must use this helper so that
/// browsers refuse to embed the page in a frame (`X-Frame-Options: DENY`) and
/// restrict which resources the page may load (`Content-Security-Policy`).
fn secure_html_response(html: String) -> Response {
    secure_html_response_with_status(StatusCode::OK, html)
}

fn secure_html_response_with_status(status: StatusCode, html: String) -> Response {
    let mut resp = (status, Html(html)).into_response();
    let headers = resp.headers_mut();
    headers.insert(
        axum::http::header::HeaderName::from_static("x-frame-options"),
        HeaderValue::from_static("DENY"),
    );
    headers.insert(
        axum::http::header::HeaderName::from_static("content-security-policy"),
        HeaderValue::from_static(
            "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'",
        ),
    );
    resp
}

// ─── HTML templates ───────────────────────────────────────────────────────────

/// Wraps `body` HTML in the shared page shell with common CSS.
///
/// Extra per-page CSS rules can be injected via `extra_css` (pass `""` for
/// none).  Both parameters are inserted verbatim — callers are responsible for
/// escaping any user-controlled content before passing it in.
fn page_shell(title: &str, extra_css: &str, body: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: sans-serif; background: #f5f5f5; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }}
    .card {{ background: #fff; border-radius: 8px; padding: 2rem; max-width: 420px; width: 100%; box-shadow: 0 2px 12px rgba(0,0,0,.12); }}
    h1 {{ font-size: 1.3rem; margin-top: 0; }}
    button {{ flex: 1; padding: .7rem; border: none; border-radius: 6px; font-size: 1rem; cursor: pointer; }}
    .buttons {{ display: flex; gap: 1rem; margin-top: 1.5rem; }}
    .approve {{ background: #2a7a2a; color: #fff; }}
    .deny {{ background: #e0e0e0; color: #333; }}
    {extra_css}
  </style>
</head>
<body>
  <div class="card">
    {body}
  </div>
</body>
</html>"#
    )
}

fn consent_page(
    client_name: &str,
    client_id: &str,
    redirect_uri: &str,
    scope: &str,
    code_challenge: &str,
    state_val: &str,
    csrf_token: &str,
) -> String {
    let client_name = html_escape(client_name);
    let client_id = html_escape(client_id);
    let redirect_uri = html_escape(redirect_uri);
    let scope = html_escape(scope);
    let code_challenge = html_escape(code_challenge);
    let state_val = html_escape(state_val);
    // CSRF token is URL-safe alphanumeric — no HTML escaping needed, but we
    // escape defensively in case the generation scheme ever changes.
    let csrf_token = html_escape(csrf_token);
    let body = format!(
        r#"<h1>Authorize Access</h1>
    <p><strong>{client_name}</strong> is requesting access to your VecLayer knowledge store.</p>
    <div class="scope">Requested scope: <strong>{scope}</strong></div>
    <form method="POST" action="/oauth/authorize">
      <input type="hidden" name="client_id" value="{client_id}">
      <input type="hidden" name="redirect_uri" value="{redirect_uri}">
      <input type="hidden" name="scope" value="{scope}">
      <input type="hidden" name="code_challenge" value="{code_challenge}">
      <input type="hidden" name="state" value="{state_val}">
      <input type="hidden" name="csrf_token" value="{csrf_token}">
      <div class="buttons">
        <button type="submit" name="approved" value="true" class="approve">Approve</button>
        <button type="submit" name="approved" value="false" class="deny">Deny</button>
      </div>
    </form>"#
    );
    let extra_css =
        ".scope { background: #eef; border-left: 4px solid #448; padding: .5rem 1rem; border-radius: 4px; margin: 1rem 0; }";
    page_shell("Authorize Access", extra_css, &body)
}

fn device_verification_page(prefill_code: &str, csrf_token: &str) -> String {
    let prefill_code = html_escape(prefill_code);
    // CSRF token is URL-safe alphanumeric; escape defensively regardless.
    let csrf_token = html_escape(csrf_token);
    // The scope dropdown was removed: the granted scope is always the scope the
    // device originally requested (stored server-side in PendingDeviceAuth.scope).
    // Presenting a writable dropdown would mislead users into thinking they can
    // restrict or expand the grant, when in fact any submitted value is ignored.
    let body = format!(
        r#"<h1>Authorize Device</h1>
    <p>Enter the code shown on your device to grant it access.</p>
    <p class="scope-note">The access level granted will be exactly what the device requested.</p>
    <form method="POST" action="/oauth/device">
      <input type="hidden" name="csrf_token" value="{csrf_token}">
      <label for="user_code">Device Code</label>
      <input type="text" id="user_code" name="user_code" value="{prefill_code}" placeholder="ABCD-EFGH" required>
      <div class="buttons">
        <button type="submit" name="approved" value="true" class="approve">Approve</button>
        <button type="submit" name="approved" value="false" class="deny">Deny</button>
      </div>
    </form>"#
    );
    let extra_css = "label { display: block; margin-bottom: .3rem; font-weight: bold; } \
        input[type=text] { width: 100%; box-sizing: border-box; padding: .6rem; border: 1px solid #ccc; border-radius: 4px; font-size: 1rem; margin-bottom: 1rem; } \
        .scope-note { color: #555; font-size: .9rem; margin-bottom: 1rem; } \
        .buttons { margin-top: .5rem; }";
    page_shell("Device Authorization", extra_css, &body)
}

fn device_success_page() -> String {
    let body = r#"<h1 style="color:#2a7a2a">Device Authorized</h1>
    <p>Your device has been successfully authorized. You may close this page.</p>"#;
    page_shell("Device Authorized", ".card { text-align: center; }", body)
}

fn device_denied_page() -> String {
    let body = r#"<h1 style="color:#c0392b">Access Denied</h1>
    <p>You denied access to the device. You may close this page.</p>"#;
    page_shell("Access Denied", ".card { text-align: center; }", body)
}

fn error_page(message: &str) -> String {
    let message = html_escape(message);
    let body = format!(
        r#"<h1 style="color:#c0392b">Error</h1>
    <p>{message}</p>"#
    );
    page_shell("Error", "", &body)
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
    let mut resp = (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({
            "error": error_type,
            "error_description": description,
        })),
    )
        .into_response();

    // RFC 6749 §5.2 — error responses from the token endpoint also MUST NOT be
    // cached, as they may contain sensitive error details.
    let headers = resp.headers_mut();
    headers.insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert(PRAGMA, HeaderValue::from_static("no-cache"));

    resp
}

// ─── Utilities ────────────────────────────────────────────────────────────────

/// Generate a cryptographically random URL-safe token string.
fn generate_opaque_token() -> String {
    let mut bytes = [0u8; 32];
    OsRng.fill_bytes(&mut bytes);
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
///
/// Draws from `OsRng` with rejection sampling to eliminate modulo bias.  The
/// alphabet is exactly 32 symbols (256 / 32 == 8 with remainder 0), so in
/// practice no bytes are ever rejected — but the code is correct for any
/// power-of-two alphabet size.
fn generate_user_code() -> String {
    const CHARS: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789"; // 32 unambiguous chars
                                                              // Largest multiple of CHARS.len() that fits in a u8.  Bytes >= this value
                                                              // would introduce bias and are rejected.  For len=32 this equals 256 (mod
                                                              // 256 → 0), meaning the condition `byte >= threshold` is never true.
    let threshold = (256u16 - (256u16 % CHARS.len() as u16)) as u8;
    let mut code = Vec::with_capacity(8);
    let mut byte = [0u8; 1];
    while code.len() < 8 {
        OsRng.fill_bytes(&mut byte);
        if threshold == 0 || byte[0] < threshold {
            code.push(CHARS[(byte[0] as usize) % CHARS.len()] as char);
        }
    }
    code.into_iter().collect()
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
    use rand_core::OsRng;
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
            pending_consents: Arc::new(Mutex::new(FifoMap::new(MAX_PENDING_CONSENTS))),
            device_csrf_tokens: Arc::new(Mutex::new(FifoSet::new(MAX_DEVICE_CSRF_TOKENS))),
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

    /// GET the device verification page and extract its one-time CSRF token,
    /// mirroring how a real browser obtains the token before submitting.
    async fn device_csrf_token(app: &Router) -> String {
        let resp = app
            .clone()
            .oneshot(get_req("/oauth/device"))
            .await
            .expect("device page");
        let html = String::from_utf8(body_vec(resp).await).expect("utf8");
        let marker = r#"name="csrf_token" value=""#;
        let start = html.find(marker).expect("csrf input present") + marker.len();
        let end = html[start..].find('"').expect("csrf value terminator") + start;
        html[start..end].to_owned()
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

    /// Render the consent page and extract the CSRF token from the hidden form field.
    async fn get_consent_csrf_token(
        app: &Router,
        client_id: &str,
        redirect_uri: &str,
        challenge: &str,
    ) -> String {
        let uri = format!(
            "/oauth/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope=read&code_challenge={challenge}&code_challenge_method=S256"
        );
        let resp = app
            .clone()
            .oneshot(get_req(&uri))
            .await
            .expect("consent GET");
        assert_eq!(resp.status(), StatusCode::OK, "expected consent page");
        let html = String::from_utf8(body_vec(resp).await).unwrap();
        // Extract value="..." from name="csrf_token"
        html.split("name=\"csrf_token\"")
            .nth(1)
            .and_then(|s| s.split("value=\"").nth(1))
            .and_then(|s| s.split('"').next())
            .expect("csrf_token hidden field")
            .to_owned()
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
        let codes = state.device_codes.clone();
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
        let csrf = device_csrf_token(&app).await;
        let approve_body =
            format!("user_code={user_code}&scope=read&approved=true&csrf_token={csrf}");
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/device", &approve_body))
            .await
            .expect("approve");
        assert_eq!(resp.status(), StatusCode::OK);
        let html = body_vec(resp).await;
        assert!(String::from_utf8_lossy(&html).contains("Authorized"));

        // A real client waits the poll interval between requests; simulate that
        // by back-dating last_polled so this poll isn't throttled as slow_down.
        {
            let mut map = codes.lock().unwrap();
            if let Some(entry) = map.get_mut(&device_code) {
                entry.last_polled =
                    Some(Instant::now() - Duration::from_secs(DEVICE_POLL_INTERVAL_SECS + 1));
            }
        }

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

    // ── normalize_user_code ───────────────────────────────────────────────────

    #[test]
    fn test_normalize_user_code_strips_dashes_and_uppercases() {
        assert_eq!(normalize_user_code("abcd-efgh"), "ABCDEFGH");
        assert_eq!(normalize_user_code("ABCD-EFGH"), "ABCDEFGH");
        assert_eq!(normalize_user_code("abcdefgh"), "ABCDEFGH");
        assert_eq!(normalize_user_code(""), "");
    }

    // ── format_user_code edge cases ───────────────────────────────────────────

    #[test]
    fn test_format_user_code_short_code_returned_as_is() {
        let result = format_user_code("ABC");
        assert_eq!(result, "ABC");
        assert!(!result.contains('-'));
    }

    #[test]
    fn test_format_user_code_with_existing_dashes_reformatted() {
        let result = format_user_code("ABCD-EFGH");
        assert_eq!(result, "ABCD-EFGH");
    }

    // ── generate_opaque_token ─────────────────────────────────────────────────

    #[test]
    fn test_generate_opaque_token_length_and_uniqueness() {
        let t1 = generate_opaque_token();
        let t2 = generate_opaque_token();
        assert_eq!(t1.len(), 43, "opaque token must be 43 chars");
        assert_ne!(t1, t2, "tokens must be unique");
        assert!(
            t1.chars()
                .all(|c| c.is_alphanumeric() || c == '-' || c == '_'),
            "token must be URL-safe: {t1}"
        );
    }

    // ── Registration validation errors ───────────────────────────────────────

    #[tokio::test]
    async fn test_register_empty_client_name_rejected() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let resp = app
            .oneshot(post_json(
                "/oauth/register",
                serde_json::json!({
                    "client_name": "   ",
                    "redirect_uris": ["https://example.com/cb"]
                }),
            ))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "client_name is required");
    }

    #[tokio::test]
    async fn test_register_empty_redirect_uris_rejected() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let resp = app
            .oneshot(post_json(
                "/oauth/register",
                serde_json::json!({
                    "client_name": "My App",
                    "redirect_uris": []
                }),
            ))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "invalid_client_metadata");
    }

    #[tokio::test]
    async fn test_register_client_name_too_long_rejected() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let resp = app
            .oneshot(post_json(
                "/oauth/register",
                serde_json::json!({
                    "client_name": "x".repeat(MAX_STRING_LENGTH + 1),
                    "redirect_uris": ["https://example.com/cb"]
                }),
            ))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert!(json["error_description"]
            .as_str()
            .unwrap_or("")
            .contains("client_name"));
    }

    #[tokio::test]
    async fn test_register_redirect_uri_too_long_rejected() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let long_uri = format!("https://example.com/{}", "x".repeat(MAX_STRING_LENGTH));
        let resp = app
            .oneshot(post_json(
                "/oauth/register",
                serde_json::json!({
                    "client_name": "My App",
                    "redirect_uris": [long_uri]
                }),
            ))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert!(json["error_description"]
            .as_str()
            .unwrap_or("")
            .contains("redirect_uri"));
    }

    // ── Authorize GET: missing fields ─────────────────────────────────────────

    #[tokio::test]
    async fn test_authorize_get_missing_client_id_returns_400() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let resp = app
            .oneshot(get_req("/oauth/authorize?response_type=code"))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_authorize_get_missing_redirect_uri_returns_400() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let resp = app
            .oneshot(get_req("/oauth/authorize?response_type=code&client_id=xyz"))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_authorize_get_plain_challenge_method_rejected() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;

        let authorize_uri = format!(
            "/oauth/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&code_challenge=somechallenge&code_challenge_method=plain"
        );
        let resp = app
            .clone()
            .oneshot(get_req(&authorize_uri))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::SEE_OTHER);
        let location = resp.headers().get("location").unwrap().to_str().unwrap();
        assert!(location.contains("error=invalid_request"));
    }

    #[tokio::test]
    async fn test_authorize_get_invalid_scope_redirects_with_error() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (_, challenge) = pkce_pair();

        let authorize_uri = format!(
            "/oauth/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&code_challenge={challenge}&code_challenge_method=S256&scope=superuser"
        );
        let resp = app
            .clone()
            .oneshot(get_req(&authorize_uri))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::SEE_OTHER);
        let location = resp.headers().get("location").unwrap().to_str().unwrap();
        assert!(location.contains("error=invalid_scope"));
    }

    #[tokio::test]
    async fn test_authorize_get_no_pkce_redirects_with_error() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;

        let authorize_uri = format!(
            "/oauth/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope=read"
        );
        let resp = app
            .clone()
            .oneshot(get_req(&authorize_uri))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::SEE_OTHER);
        let location = resp.headers().get("location").unwrap().to_str().unwrap();
        assert!(location.contains("error=invalid_request"));
    }

    // ── Authorize GET: code_challenge length guard ────────────────────────────

    #[tokio::test]
    async fn test_authorize_get_code_challenge_too_long_redirects_with_error() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let long_challenge = "x".repeat(MAX_STRING_LENGTH + 1);

        let authorize_uri = format!(
            "/oauth/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&code_challenge={long_challenge}&code_challenge_method=S256&scope=read"
        );
        let resp = app
            .clone()
            .oneshot(get_req(&authorize_uri))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::SEE_OTHER);
        let location = resp.headers().get("location").unwrap().to_str().unwrap();
        assert!(
            location.contains("error=invalid_request"),
            "expected invalid_request in redirect, got: {location}"
        );
    }

    // ── Authorize GET: consent page ───────────────────────────────────────────

    #[tokio::test]
    async fn test_authorize_get_shows_consent_page_when_not_auto_approve() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (_, challenge) = pkce_pair();

        let authorize_uri = format!(
            "/oauth/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope=read&code_challenge={challenge}&code_challenge_method=S256"
        );
        let resp = app
            .clone()
            .oneshot(get_req(&authorize_uri))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::OK);
        let html = String::from_utf8(body_vec(resp).await).unwrap();
        assert!(html.contains("Authorize Access"));
        assert!(html.contains(&client_id));
    }

    /// The consent-session map is bounded: once it reaches MAX_PENDING_CONSENTS
    /// the next consent-page render evicts the oldest entry (FIFO) and inserts
    /// the new one, keeping the map exactly at the cap.
    /// Discriminating: without the cap the map would grow to MAX + 1.
    #[tokio::test]
    async fn test_pending_consents_map_is_bounded() {
        let (state, _dir) = make_state(false);
        let consents = state.pending_consents.clone();
        let app = oauth_router(state);
        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (_, challenge) = pkce_pair();

        {
            let mut map = consents.lock().unwrap();
            for i in 0..MAX_PENDING_CONSENTS {
                let token = format!("stale-{i}");
                map.insert(token.clone(), PendingConsent { csrf_token: token });
            }
            assert_eq!(map.len(), MAX_PENDING_CONSENTS);
        }

        let authorize_uri = format!(
            "/oauth/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope=read&code_challenge={challenge}&code_challenge_method=S256"
        );
        let resp = app.oneshot(get_req(&authorize_uri)).await.expect("request");
        assert_eq!(resp.status(), StatusCode::OK);

        // FIFO eviction: oldest entry displaced, new one inserted — map stays at cap.
        let len = consents.lock().unwrap().len();
        assert_eq!(
            len, MAX_PENDING_CONSENTS,
            "FIFO eviction must keep map at cap, got {len}"
        );
    }

    // ── Authorize POST (consent form) ─────────────────────────────────────────

    #[tokio::test]
    async fn test_authorize_post_approved_issues_code() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (_, challenge) = pkce_pair();

        let csrf = get_consent_csrf_token(&app, &client_id, redirect_uri, &challenge).await;
        let form = format!(
            "client_id={client_id}&redirect_uri={redirect_uri}&scope=read&code_challenge={challenge}&approved=true&csrf_token={csrf}"
        );
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/authorize", &form))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::SEE_OTHER);
        let location = resp.headers().get("location").unwrap().to_str().unwrap();
        assert!(
            location.contains("code="),
            "expected code in location: {location}"
        );
    }

    #[tokio::test]
    async fn test_authorize_post_denied_redirects_with_access_denied() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (_, challenge) = pkce_pair();

        let csrf = get_consent_csrf_token(&app, &client_id, redirect_uri, &challenge).await;
        let form = format!(
            "client_id={client_id}&redirect_uri={redirect_uri}&scope=read&code_challenge={challenge}&approved=false&csrf_token={csrf}"
        );
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/authorize", &form))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::SEE_OTHER);
        let location = resp.headers().get("location").unwrap().to_str().unwrap();
        assert!(location.contains("error=access_denied"));
    }

    #[tokio::test]
    async fn test_authorize_post_unknown_client_returns_400() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let form = "client_id=unknown-xyz&redirect_uri=https%3A%2F%2Fexample.com%2Fcb&scope=read&code_challenge=xxx";
        let resp = app
            .oneshot(post_form("/oauth/authorize", form))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_authorize_post_mismatched_redirect_uri_returns_400() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;

        let form = format!(
            "client_id={client_id}&redirect_uri=https://evil.com/steal&scope=read&code_challenge=xxx"
        );
        let resp = app
            .oneshot(post_form("/oauth/authorize", &form))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_authorize_post_invalid_scope_redirects_with_error() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (_, challenge) = pkce_pair();

        let csrf = get_consent_csrf_token(&app, &client_id, redirect_uri, &challenge).await;
        let form = format!(
            "client_id={client_id}&redirect_uri={redirect_uri}&scope=superuser&code_challenge={challenge}&approved=true&csrf_token={csrf}"
        );
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/authorize", &form))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::SEE_OTHER);
        let location = resp.headers().get("location").unwrap().to_str().unwrap();
        assert!(location.contains("error=invalid_scope"));
    }

    /// Even with a valid CSRF token and `approved=true`, an oversized
    /// `code_challenge` smuggled in via the consent POST must be rejected with
    /// `invalid_request` rather than persisted — mirroring the GET-path guard.
    /// Discriminating: without the POST guard this would issue a `code=`.
    #[tokio::test]
    async fn test_authorize_post_code_challenge_too_long_redirects_with_error() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (_, challenge) = pkce_pair();

        // Obtain a valid CSRF token via the consent page (rendered with a normal
        // challenge), then submit an oversized challenge in the POST body.
        let csrf = get_consent_csrf_token(&app, &client_id, redirect_uri, &challenge).await;
        let long_challenge = "x".repeat(MAX_STRING_LENGTH + 1);
        let form = format!(
            "client_id={client_id}&redirect_uri={redirect_uri}&scope=read&code_challenge={long_challenge}&approved=true&csrf_token={csrf}"
        );
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/authorize", &form))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::SEE_OTHER);
        let location = resp.headers().get("location").unwrap().to_str().unwrap();
        assert!(
            location.contains("error=invalid_request"),
            "expected invalid_request in redirect, got: {location}"
        );
    }

    /// A state value of 513 bytes submitted via the consent POST must be
    /// rejected with 400 Bad Request — mirroring the GET-path guard.
    /// Discriminating: without the POST guard this would redirect with a code.
    #[tokio::test]
    async fn test_authorize_post_rejects_state_exceeding_max_len() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (_, challenge) = pkce_pair();

        // Obtain a valid CSRF token via the consent page GET.
        let csrf = get_consent_csrf_token(&app, &client_id, redirect_uri, &challenge).await;
        let long_state = "s".repeat(MAX_STATE_LEN + 1); // 513 bytes
        let form = format!(
            "client_id={client_id}&redirect_uri={redirect_uri}&scope=read&code_challenge={challenge}&approved=true&csrf_token={csrf}&state={long_state}"
        );
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/authorize", &form))
            .await
            .expect("request");

        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "expected 400 for oversized state"
        );
    }

    /// A state value of exactly MAX_STATE_LEN (512) bytes must pass the guard
    /// and proceed normally (redirect with a code, not a 400).
    #[tokio::test]
    async fn test_authorize_post_accepts_state_at_max_len() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (_, challenge) = pkce_pair();

        let csrf = get_consent_csrf_token(&app, &client_id, redirect_uri, &challenge).await;
        let exact_state = "s".repeat(MAX_STATE_LEN); // exactly 512 bytes
        let form = format!(
            "client_id={client_id}&redirect_uri={redirect_uri}&scope=read&code_challenge={challenge}&approved=true&csrf_token={csrf}&state={exact_state}"
        );
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/authorize", &form))
            .await
            .expect("request");

        // Must NOT be rejected by the length guard — should redirect with a code.
        assert_ne!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "state of exactly MAX_STATE_LEN should pass the guard"
        );
        assert_eq!(resp.status(), StatusCode::SEE_OTHER);
        let location = resp.headers().get("location").unwrap().to_str().unwrap();
        assert!(
            location.contains("code="),
            "expected code in redirect, got: {location}"
        );
    }

    // ── Token endpoint: error paths ───────────────────────────────────────────

    #[tokio::test]
    async fn test_token_unsupported_grant_type() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let resp = app
            .oneshot(post_form("/oauth/token", "grant_type=implicit"))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "unsupported_grant_type");
    }

    #[tokio::test]
    async fn test_token_auth_code_missing_code() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let resp = app
            .oneshot(post_form(
                "/oauth/token",
                "grant_type=authorization_code&code_verifier=someverifier&client_id=abc",
            ))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "invalid_request");
    }

    #[tokio::test]
    async fn test_token_auth_code_missing_client_id() {
        let (state, _dir) = make_state(true);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (verifier, challenge) = pkce_pair();

        let authorize_uri = format!(
            "/oauth/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope=read&code_challenge={challenge}&code_challenge_method=S256"
        );
        let resp = app.clone().oneshot(get_req(&authorize_uri)).await.unwrap();
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

        // Exchange without client_id
        let body = format!("grant_type=authorization_code&code={code}&code_verifier={verifier}");
        let resp = app
            .oneshot(post_form("/oauth/token", &body))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "invalid_request");
    }

    #[tokio::test]
    async fn test_token_auth_code_client_id_mismatch() {
        let (state, _dir) = make_state(true);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (verifier, challenge) = pkce_pair();

        let authorize_uri = format!(
            "/oauth/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope=read&code_challenge={challenge}&code_challenge_method=S256"
        );
        let resp = app.clone().oneshot(get_req(&authorize_uri)).await.unwrap();
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

        let body = format!(
            "grant_type=authorization_code&code={code}&code_verifier={verifier}&client_id=wrong-client"
        );
        let resp = app
            .oneshot(post_form("/oauth/token", &body))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "invalid_client");
    }

    #[tokio::test]
    async fn test_token_auth_code_redirect_uri_mismatch() {
        let (state, _dir) = make_state(true);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (verifier, challenge) = pkce_pair();

        let authorize_uri = format!(
            "/oauth/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope=read&code_challenge={challenge}&code_challenge_method=S256"
        );
        let resp = app.clone().oneshot(get_req(&authorize_uri)).await.unwrap();
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

        let body = format!(
            "grant_type=authorization_code&code={code}&code_verifier={verifier}&client_id={client_id}&redirect_uri=https://other.com/cb"
        );
        let resp = app
            .oneshot(post_form("/oauth/token", &body))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "invalid_grant");
    }

    #[tokio::test]
    async fn test_token_refresh_missing_token() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let resp = app
            .oneshot(post_form(
                "/oauth/token",
                "grant_type=refresh_token&client_id=abc",
            ))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "invalid_request");
    }

    #[tokio::test]
    async fn test_token_refresh_missing_client_id_after_invalid_token() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        // No client_id provided — should return invalid_request
        let resp = app
            .oneshot(post_form(
                "/oauth/token",
                "grant_type=refresh_token&refresh_token=sometoken",
            ))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        // Either invalid_grant (token not found) or invalid_request (no client_id)
        // The current implementation tries to validate/revoke the token first,
        // so if the token is not found it returns invalid_grant before checking client_id.
        let error = json["error"].as_str().unwrap_or("");
        assert!(
            error == "invalid_grant" || error == "invalid_request",
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn test_token_refresh_invalid_token() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let resp = app
            .oneshot(post_form(
                "/oauth/token",
                "grant_type=refresh_token&refresh_token=nonexistent&client_id=abc",
            ))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "invalid_grant");
    }

    #[tokio::test]
    async fn test_token_refresh_client_id_mismatch() {
        let (state, _dir) = make_state(true);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (verifier, challenge) = pkce_pair();

        let authorize_uri = format!(
            "/oauth/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope=read&code_challenge={challenge}&code_challenge_method=S256"
        );
        let resp = app.clone().oneshot(get_req(&authorize_uri)).await.unwrap();
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
            .unwrap();
        let json = body_json(resp).await;
        let refresh = json["refresh_token"].as_str().unwrap().to_owned();

        // Use refresh token with wrong client_id
        let body =
            format!("grant_type=refresh_token&refresh_token={refresh}&client_id=wrong-client");
        let resp = app
            .oneshot(post_form("/oauth/token", &body))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "invalid_client");
    }

    /// Regression test for refresh-token denial-of-service: a wrong-client
    /// refresh request must NOT invalidate the token so the legitimate client
    /// can still use it.
    ///
    /// `handle_refresh_token_grant` verifies the submitted `client_id` against
    /// the token's bound client (via `refresh_token_client_id`) BEFORE revoking,
    /// all under one lock. A wrong-client attempt is rejected with
    /// `invalid_client` without burning the legitimate owner's still-valid token.
    #[tokio::test]
    async fn test_refresh_wrong_client_does_not_burn_token() {
        let (state, _dir) = make_state(true);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (verifier, challenge) = pkce_pair();

        // Step 1: Obtain a refresh token via the normal authorization flow.
        let authorize_uri = format!(
            "/oauth/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope=read&code_challenge={challenge}&code_challenge_method=S256"
        );
        let resp = app.clone().oneshot(get_req(&authorize_uri)).await.unwrap();
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
            .unwrap();
        let json = body_json(resp).await;
        let refresh_token = json["refresh_token"].as_str().unwrap().to_owned();

        // Step 2: An attacker submits the token with a wrong client_id.
        // Expected: rejected with invalid_client; token is NOT burned.
        let body = format!(
            "grant_type=refresh_token&refresh_token={refresh_token}&client_id=attacker-client"
        );
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/token", &body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(
            json["error"], "invalid_client",
            "wrong client must be rejected"
        );

        // Step 3: The legitimate client must still be able to use the token.
        // This is what FAILS with the current implementation (the token was
        // already revoked in step 2).
        let body =
            format!("grant_type=refresh_token&refresh_token={refresh_token}&client_id={client_id}");
        let resp = app.oneshot(post_form("/oauth/token", &body)).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "legitimate client must still be able to refresh after a wrong-client attempt"
        );
        let json = body_json(resp).await;
        assert!(
            json["access_token"].is_string(),
            "expected a new access token for the legitimate client"
        );
    }

    // ── Device flow: error paths ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_device_code_missing_client_id() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let resp = app
            .oneshot(post_form("/oauth/device/code", "scope=read"))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "invalid_request");
    }

    #[tokio::test]
    async fn test_device_code_unknown_client_id() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let resp = app
            .oneshot(post_form(
                "/oauth/device/code",
                "client_id=nonexistent&scope=read",
            ))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "invalid_client");
    }

    #[tokio::test]
    async fn test_device_code_invalid_scope() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;

        let body = format!("client_id={client_id}&scope=superuser");
        let resp = app
            .oneshot(post_form("/oauth/device/code", &body))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "invalid_scope");
    }

    #[tokio::test]
    async fn test_device_token_unknown_device_code() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;

        let body = format!(
            "grant_type=urn:ietf:params:oauth:grant-type:device_code&device_code=unknown&client_id={client_id}"
        );
        let resp = app
            .oneshot(post_form("/oauth/token", &body))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "invalid_grant");
    }

    #[tokio::test]
    async fn test_device_token_missing_device_code() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let body = "grant_type=urn:ietf:params:oauth:grant-type:device_code&client_id=abc";
        let resp = app
            .oneshot(post_form("/oauth/token", body))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "invalid_request");
    }

    #[tokio::test]
    async fn test_device_denied_returns_access_denied() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;

        // Request a device code
        let body = format!("client_id={client_id}&scope=read");
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/device/code", &body))
            .await
            .unwrap();
        let json = body_json(resp).await;
        let device_code = json["device_code"].as_str().unwrap().to_owned();
        let user_code = json["user_code"].as_str().unwrap().to_owned();

        // Deny the device
        let csrf = device_csrf_token(&app).await;
        let deny_body = format!("user_code={user_code}&approved=false&csrf_token={csrf}");
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/device", &deny_body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let html = String::from_utf8(body_vec(resp).await).unwrap();
        assert!(html.contains("Denied") || html.contains("denied"));

        // Poll after denial → access_denied
        let poll_body = format!(
            "grant_type=urn:ietf:params:oauth:grant-type:device_code&device_code={device_code}&client_id={client_id}"
        );
        let resp = app
            .oneshot(post_form("/oauth/token", &poll_body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "access_denied");
    }

    #[tokio::test]
    async fn test_device_page_handler_get_returns_html() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let resp = app
            .oneshot(get_req("/oauth/device"))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::OK);
        let html = String::from_utf8(body_vec(resp).await).unwrap();
        assert!(html.contains("Device Authorization") || html.contains("Authorize Device"));
    }

    #[tokio::test]
    async fn test_device_page_prefills_user_code() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let resp = app
            .oneshot(get_req("/oauth/device?user_code=ABCD-EFGH"))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::OK);
        let html = String::from_utf8(body_vec(resp).await).unwrap();
        assert!(html.contains("ABCD-EFGH"));
    }

    #[tokio::test]
    async fn test_device_approve_unknown_user_code_returns_error_page() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let csrf = device_csrf_token(&app).await;
        let resp = app
            .oneshot(post_form(
                "/oauth/device",
                &format!("user_code=ZZZZ-ZZZZ&approved=true&csrf_token={csrf}"),
            ))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::OK);
        let html = String::from_utf8(body_vec(resp).await).unwrap();
        assert!(
            html.contains("Unknown"),
            "expected unknown-code page, got: {html}"
        );
    }

    #[tokio::test]
    async fn test_device_approve_without_csrf_token_rejected() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        // A POST with no csrf_token (a forged cross-site request) must be rejected.
        let resp = app
            .oneshot(post_form(
                "/oauth/device",
                "user_code=ABCD-EFGH&approved=true",
            ))
            .await
            .expect("request");
        let html = String::from_utf8(body_vec(resp).await).unwrap();
        assert!(
            html.contains("Invalid or expired request"),
            "a CSRF-less device approval must be rejected, got: {html}"
        );
    }

    #[tokio::test]
    async fn test_device_double_approve_rejected() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);
        let client_id = register_client(&app, "https://app.example.com/cb").await;

        let resp = app
            .clone()
            .oneshot(post_form(
                "/oauth/device/code",
                &format!("client_id={client_id}&scope=read"),
            ))
            .await
            .unwrap();
        let user_code = body_json(resp).await["user_code"]
            .as_str()
            .unwrap()
            .to_owned();

        // First approval succeeds.
        let csrf = device_csrf_token(&app).await;
        let resp = app
            .clone()
            .oneshot(post_form(
                "/oauth/device",
                &format!("user_code={user_code}&scope=read&approved=true&csrf_token={csrf}"),
            ))
            .await
            .unwrap();
        assert!(String::from_utf8(body_vec(resp).await)
            .unwrap()
            .contains("Authorized"));

        // A second approval (with a fresh, valid CSRF token) must be rejected.
        let csrf2 = device_csrf_token(&app).await;
        let resp = app
            .oneshot(post_form(
                "/oauth/device",
                &format!("user_code={user_code}&scope=read&approved=true&csrf_token={csrf2}"),
            ))
            .await
            .unwrap();
        let html = String::from_utf8(body_vec(resp).await).unwrap();
        assert!(
            html.contains("already been processed"),
            "a second device approval must be rejected, got: {html}"
        );
    }

    /// Scope escalation fix: a device that requested `scope=read` receives a
    /// READ token even when the approval POST submits `scope=admin`.  The
    /// server-side stored scope (`PendingDeviceAuth.scope`) is authoritative;
    /// the form-supplied scope field is silently ignored.
    ///
    /// Discriminating: before the fix, submitting `scope=admin` in the approval
    /// form would cause `entry.approved = Some(Capability::Admin)`, minting an
    /// admin token for a device that only asked for read.
    #[tokio::test]
    async fn test_device_approve_scope_escalation_blocked() {
        let (state, _dir) = make_state(false);
        let signing_key = state.signing_key.clone();
        let server_did = state.server_did.clone();
        let app = oauth_router(state);
        let client_id = register_client(&app, "https://app.example.com/cb").await;

        // Device requests read scope.
        let resp = app
            .clone()
            .oneshot(post_form(
                "/oauth/device/code",
                &format!("client_id={client_id}&scope=read"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        let device_code = json["device_code"].as_str().unwrap().to_owned();
        let user_code = json["user_code"].as_str().unwrap().to_owned();

        // Approve with scope=admin in the form body — escalation attempt.
        // The server must ignore this field and grant only read (entry.scope).
        let csrf = device_csrf_token(&app).await;
        let resp = app
            .clone()
            .oneshot(post_form(
                "/oauth/device",
                &format!("user_code={user_code}&scope=admin&approved=true&csrf_token={csrf}"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let html = String::from_utf8(body_vec(resp).await).unwrap();
        assert!(
            html.contains("Authorized"),
            "approval must succeed (scope field is ignored), got: {html}"
        );

        // Poll — must get a token with READ capability, not admin.
        let poll_body = format!(
            "grant_type=urn:ietf:params:oauth:grant-type:device_code&device_code={device_code}&client_id={client_id}"
        );
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/token", &poll_body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        let access_token = json["access_token"].as_str().expect("access_token");
        assert!(!access_token.is_empty());

        // Verify the JWT was minted with Read, not Admin.
        let claims = verify(
            access_token,
            &signing_key.verifying_key(),
            Some(&server_did),
        )
        .expect("valid JWT");
        assert_eq!(
            claims.cap,
            Capability::Read,
            "escalation blocked: token must have Read capability, got: {:?}",
            claims.cap
        );
    }

    /// A device that requested `scope=write` must receive a WRITE token after
    /// approval, even if the form does not include a scope field.
    /// Confirms that the server-stored scope (not the form field) is authoritative.
    #[tokio::test]
    async fn test_device_approve_write_scope_granted_from_entry() {
        let (state, _dir) = make_state(false);
        let signing_key = state.signing_key.clone();
        let server_did = state.server_did.clone();
        let app = oauth_router(state);
        let client_id = register_client(&app, "https://app.example.com/cb").await;

        // Device requests write scope.
        let resp = app
            .clone()
            .oneshot(post_form(
                "/oauth/device/code",
                &format!("client_id={client_id}&scope=write"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        let device_code = json["device_code"].as_str().unwrap().to_owned();
        let user_code = json["user_code"].as_str().unwrap().to_owned();

        // Approve without any scope field — entry.scope (write) is used.
        let csrf = device_csrf_token(&app).await;
        let resp = app
            .clone()
            .oneshot(post_form(
                "/oauth/device",
                &format!("user_code={user_code}&approved=true&csrf_token={csrf}"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let html = String::from_utf8(body_vec(resp).await).unwrap();
        assert!(
            html.contains("Authorized"),
            "approval must succeed, got: {html}"
        );

        // Poll — must get a token with WRITE capability.
        let poll_body = format!(
            "grant_type=urn:ietf:params:oauth:grant-type:device_code&device_code={device_code}&client_id={client_id}"
        );
        let resp = app
            .oneshot(post_form("/oauth/token", &poll_body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        let access_token = json["access_token"].as_str().expect("access_token");
        let claims = verify(
            access_token,
            &signing_key.verifying_key(),
            Some(&server_did),
        )
        .expect("valid JWT");
        assert_eq!(
            claims.cap,
            Capability::Write,
            "device requesting write must get write token, got: {:?}",
            claims.cap
        );
    }

    /// A device approval POST that omits the `scope` field entirely must succeed —
    /// the granted scope comes from `entry.scope` (server-side), not from the
    /// form, so an absent form field has no effect.
    #[tokio::test]
    async fn test_device_approve_missing_scope_field_succeeds() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);
        let client_id = register_client(&app, "https://app.example.com/cb").await;

        let resp = app
            .clone()
            .oneshot(post_form(
                "/oauth/device/code",
                &format!("client_id={client_id}&scope=read"),
            ))
            .await
            .unwrap();
        let user_code = body_json(resp).await["user_code"]
            .as_str()
            .unwrap()
            .to_owned();

        // No `scope` field in the approval form.
        let csrf = device_csrf_token(&app).await;
        let resp = app
            .oneshot(post_form(
                "/oauth/device",
                &format!("user_code={user_code}&approved=true&csrf_token={csrf}"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let html = String::from_utf8(body_vec(resp).await).unwrap();
        assert!(
            html.contains("Authorized"),
            "absent form scope field must not block approval, got: {html}"
        );
    }

    /// When the device-code map is already full of live (unexpired) entries, a
    /// new device-code request is rejected rather than growing memory without
    /// bound. Pre-populates the shared map directly to avoid 1024 HTTP calls.
    #[tokio::test]
    async fn test_device_code_rejected_when_map_full() {
        let (state, _dir) = make_state(false);
        let codes = state.device_codes.clone();
        let app = oauth_router(state);
        let client_id = register_client(&app, "https://app.example.com/cb").await;

        let live_until = unix_now() + DEVICE_CODE_TTL_SECS;
        {
            let mut map = codes.lock().unwrap();
            for i in 0..MAX_DEVICE_CODES {
                let key = format!("dc-{i}");
                map.insert(
                    key.clone(),
                    PendingDeviceAuth {
                        device_code: key,
                        user_code: format!("UC{i:08}"),
                        client_id: client_id.clone(),
                        scope: Capability::Read,
                        expires_at: live_until,
                        approved: None,
                        denied: false,
                        last_polled: None,
                    },
                );
            }
        }

        let resp = app
            .oneshot(post_form(
                "/oauth/device/code",
                &format!("client_id={client_id}&scope=read"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "temporarily_unavailable");
    }

    /// A map full of *expired* entries must not block new requests: the handler
    /// purges expired codes before checking the cap. Discriminating: without
    /// the purge the map would be at capacity and the request would be rejected.
    #[tokio::test]
    async fn test_device_code_purges_expired_before_cap() {
        let (state, _dir) = make_state(false);
        let codes = state.device_codes.clone();
        let app = oauth_router(state);
        let client_id = register_client(&app, "https://app.example.com/cb").await;

        {
            let mut map = codes.lock().unwrap();
            for i in 0..MAX_DEVICE_CODES {
                let key = format!("expired-{i}");
                map.insert(
                    key.clone(),
                    PendingDeviceAuth {
                        device_code: key,
                        user_code: format!("EX{i:08}"),
                        client_id: client_id.clone(),
                        scope: Capability::Read,
                        expires_at: 0, // long expired
                        approved: None,
                        denied: false,
                        last_polled: None,
                    },
                );
            }
        }

        let resp = app
            .oneshot(post_form(
                "/oauth/device/code",
                &format!("client_id={client_id}&scope=read"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(
            json["device_code"].is_string(),
            "expected a new device code"
        );

        // The expired entries were purged; only the freshly inserted one remains.
        assert_eq!(codes.lock().unwrap().len(), 1);
    }

    // ── Device flow: client_id binding ───────────────────────────────────────

    /// A different client (client B) that learns client A's device_code must
    /// NOT be able to redeem it for a token.  This is the core security fix:
    /// the polling request's client_id must match the client_id embedded in
    /// the device-code entry.
    #[tokio::test]
    async fn test_device_token_wrong_client_id_rejected() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        // Register two distinct clients.
        let client_a = register_client(&app, "https://app-a.example.com/cb").await;
        let client_b = register_client(&app, "https://app-b.example.com/cb").await;

        // Client A obtains a device code.
        let resp = app
            .clone()
            .oneshot(post_form(
                "/oauth/device/code",
                &format!("client_id={client_a}&scope=read"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        let device_code = json["device_code"].as_str().unwrap().to_owned();
        let user_code = json["user_code"].as_str().unwrap().to_owned();

        // User approves.
        let csrf = device_csrf_token(&app).await;
        app.clone()
            .oneshot(post_form(
                "/oauth/device",
                &format!("user_code={user_code}&scope=read&approved=true&csrf_token={csrf}"),
            ))
            .await
            .unwrap();

        // Client B tries to redeem client A's approved device_code — must be rejected.
        let body = format!(
            "grant_type=urn:ietf:params:oauth:grant-type:device_code&device_code={device_code}&client_id={client_b}"
        );
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/token", &body))
            .await
            .expect("poll with wrong client");
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "wrong client_id must be rejected"
        );
        let json = body_json(resp).await;
        assert_eq!(
            json["error"], "invalid_client",
            "expected invalid_client, got: {json}"
        );
    }

    /// The correct client redeeming its own approved device_code must still succeed.
    #[tokio::test]
    async fn test_device_token_correct_client_id_succeeds() {
        let (state, _dir) = make_state(false);
        let signing_key = state.signing_key.clone();
        let server_did = state.server_did.clone();
        let app = oauth_router(state);

        let client_id = register_client(&app, "https://app.example.com/cb").await;

        let resp = app
            .clone()
            .oneshot(post_form(
                "/oauth/device/code",
                &format!("client_id={client_id}&scope=read"),
            ))
            .await
            .unwrap();
        let json = body_json(resp).await;
        let device_code = json["device_code"].as_str().unwrap().to_owned();
        let user_code = json["user_code"].as_str().unwrap().to_owned();

        let csrf = device_csrf_token(&app).await;
        app.clone()
            .oneshot(post_form(
                "/oauth/device",
                &format!("user_code={user_code}&scope=read&approved=true&csrf_token={csrf}"),
            ))
            .await
            .unwrap();

        let body = format!(
            "grant_type=urn:ietf:params:oauth:grant-type:device_code&device_code={device_code}&client_id={client_id}"
        );
        let resp = app
            .oneshot(post_form("/oauth/token", &body))
            .await
            .expect("poll with correct client");
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "correct client_id must succeed"
        );
        let json = body_json(resp).await;
        let access_token = json["access_token"].as_str().expect("access_token");
        assert!(!access_token.is_empty());
        let claims = verify(
            access_token,
            &signing_key.verifying_key(),
            Some(&server_did),
        )
        .expect("valid JWT");
        assert_eq!(claims.cap, Capability::Read);
    }

    /// Polling without any client_id in the form must be rejected with
    /// `invalid_request`, matching the auth-code and refresh-token paths.
    #[tokio::test]
    async fn test_device_token_missing_client_id_rejected() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let client_id = register_client(&app, "https://app.example.com/cb").await;

        let resp = app
            .clone()
            .oneshot(post_form(
                "/oauth/device/code",
                &format!("client_id={client_id}&scope=read"),
            ))
            .await
            .unwrap();
        let json = body_json(resp).await;
        let device_code = json["device_code"].as_str().unwrap().to_owned();
        let user_code = json["user_code"].as_str().unwrap().to_owned();

        let csrf = device_csrf_token(&app).await;
        app.clone()
            .oneshot(post_form(
                "/oauth/device",
                &format!("user_code={user_code}&scope=read&approved=true&csrf_token={csrf}"),
            ))
            .await
            .unwrap();

        // Poll without client_id — must be rejected.
        let body = format!(
            "grant_type=urn:ietf:params:oauth:grant-type:device_code&device_code={device_code}"
        );
        let resp = app
            .oneshot(post_form("/oauth/token", &body))
            .await
            .expect("poll without client_id");
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "missing client_id must be rejected"
        );
        let json = body_json(resp).await;
        assert_eq!(
            json["error"], "invalid_request",
            "expected invalid_request, got: {json}"
        );
    }

    // ── Device flow: expired device code ─────────────────────────────────────

    #[tokio::test]
    async fn test_device_token_expired_code() {
        let (state, _dir) = make_state(false);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = {
            let mut store = state.token_store.lock().unwrap();
            store
                .register_client("App", vec![redirect_uri.to_owned()])
                .expect("register")
                .client_id
                .clone()
        };

        let expired_device_code = "expired-device-code-xyz";
        state.device_codes.lock().unwrap().insert(
            expired_device_code.to_owned(),
            PendingDeviceAuth {
                device_code: expired_device_code.to_owned(),
                user_code: "AAAA-BBBB".to_owned(),
                client_id: client_id.clone(),
                scope: crate::auth::capability::Capability::Read,
                expires_at: 0,
                approved: None,
                denied: false,
                last_polled: None,
            },
        );

        let app = oauth_router(state);

        let body = format!(
            "grant_type=urn:ietf:params:oauth:grant-type:device_code&device_code={expired_device_code}&client_id={client_id}"
        );
        let resp = app
            .oneshot(post_form("/oauth/token", &body))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "expired_token");
    }

    // ── is_localhost ──────────────────────────────────────────────────────────

    #[test]
    fn test_is_localhost_recognizes_loopback_addresses() {
        assert!(is_localhost("http://localhost:8080"));
        assert!(is_localhost("http://127.0.0.1:8080"));
        assert!(is_localhost("http://127.1.2.3"));
        assert!(is_localhost("localhost"));
        assert!(is_localhost("127.0.0.1"));
        assert!(is_localhost("::1"));
    }

    #[test]
    fn test_is_localhost_rejects_non_loopback() {
        assert!(!is_localhost("http://example.com:8080"));
        assert!(!is_localhost("http://192.168.1.1:8080"));
        assert!(!is_localhost("http://0.0.0.0:8080"));
        assert!(!is_localhost("https://myapp.example.com"));
    }

    // ── validate_redirect_uri ─────────────────────────────────────────────────

    #[test]
    fn test_validate_redirect_uri_accepts_https() {
        assert!(validate_redirect_uri("https://example.com/callback"));
        assert!(validate_redirect_uri("https://app.myservice.io/oauth/cb"));
        assert!(validate_redirect_uri("HTTPS://example.com/cb"));
    }

    #[test]
    fn test_validate_redirect_uri_accepts_http_localhost() {
        assert!(validate_redirect_uri("http://localhost/callback"));
        assert!(validate_redirect_uri("http://localhost:3000/cb"));
        assert!(validate_redirect_uri("http://127.0.0.1/callback"));
        assert!(validate_redirect_uri("http://127.0.0.1:9000/cb"));
        // Any 127.0.0.0/8 address is loopback, not just 127.0.0.1.
        assert!(validate_redirect_uri("http://127.255.255.254/cb"));
    }

    /// IPv6 loopback `[::1]` must be accepted for http (RFC 8252 §8.3),
    /// matching the https behaviour — with and without a port.
    #[test]
    fn test_validate_redirect_uri_accepts_http_ipv6_loopback() {
        assert!(validate_redirect_uri("http://[::1]/callback"));
        assert!(validate_redirect_uri("http://[::1]:8080/cb"));
        // https loopback was already accepted; assert it stays consistent.
        assert!(validate_redirect_uri("https://[::1]/cb"));
    }

    /// A non-loopback IPv6 literal over http must be rejected.
    #[test]
    fn test_validate_redirect_uri_rejects_http_ipv6_non_loopback() {
        assert!(!validate_redirect_uri("http://[2001:db8::1]/cb"));
        assert!(!validate_redirect_uri("http://[fe80::1]:9000/cb"));
    }

    #[test]
    fn test_authority_host_extraction() {
        assert_eq!(authority_host("localhost"), "localhost");
        assert_eq!(authority_host("localhost:3000"), "localhost");
        assert_eq!(authority_host("127.0.0.1:9000"), "127.0.0.1");
        assert_eq!(authority_host("[::1]"), "::1");
        assert_eq!(authority_host("[::1]:8080"), "::1");
        assert_eq!(authority_host("[2001:db8::1]:443"), "2001:db8::1");
    }

    #[test]
    fn test_validate_redirect_uri_accepts_custom_app_schemes() {
        assert!(validate_redirect_uri("myapp://oauth/callback"));
        assert!(validate_redirect_uri("com.example.app://auth"));
    }

    #[test]
    fn test_validate_redirect_uri_rejects_javascript() {
        assert!(!validate_redirect_uri("javascript:alert(1)"));
        assert!(!validate_redirect_uri("JAVASCRIPT:alert(document.cookie)"));
    }

    #[test]
    fn test_validate_redirect_uri_rejects_data_and_file() {
        assert!(!validate_redirect_uri(
            "data:text/html,<script>alert(1)</script>"
        ));
        assert!(!validate_redirect_uri("file:///etc/passwd"));
    }

    #[test]
    fn test_validate_redirect_uri_rejects_http_non_localhost() {
        assert!(!validate_redirect_uri("http://example.com/callback"));
        assert!(!validate_redirect_uri("http://192.168.1.100/cb"));
        assert!(!validate_redirect_uri("http://0.0.0.0/cb"));
    }

    /// A host that merely *starts with* `127.` but is a real DNS name must be
    /// rejected: `http://127.0.0.1.evil.com/cb` resolves to an attacker host,
    /// so accepting it would leak auth codes over plain HTTP to a remote party.
    #[test]
    fn test_validate_redirect_uri_rejects_loopback_lookalike_host() {
        assert!(!validate_redirect_uri("http://127.0.0.1.evil.com/cb"));
        assert!(!validate_redirect_uri("http://localhost.evil.com/cb"));
        assert!(!validate_redirect_uri("http://127.0.0.1evil.com/cb"));
        // Empty authority (`http:///path`) has no host and must be rejected.
        assert!(!validate_redirect_uri("http:///path"));
    }

    /// The same loopback-lookalike guard applies to `is_localhost`.
    #[test]
    fn test_is_localhost_rejects_loopback_lookalike_host() {
        assert!(!is_localhost("http://127.0.0.1.evil.com"));
        assert!(!is_localhost("http://localhost.evil.com"));
        assert!(!is_localhost("127.0.0.1.evil.com"));
    }

    /// Embedded userinfo (credentials) in https:// URIs must be rejected.
    /// `https://user:pass@evil.com/steal` could be used for credential-harvesting
    /// open redirects if registered as a redirect URI.
    #[test]
    fn test_validate_redirect_uri_rejects_embedded_credentials() {
        // user:pass@ form
        assert!(!validate_redirect_uri("https://user:pass@evil.com/steal"));
        // user@ form (no password)
        assert!(!validate_redirect_uri("https://user@evil.com/cb"));
        // Credentials in http://localhost URI
        assert!(!validate_redirect_uri("http://user:pass@localhost/cb"));
        // Uppercase scheme — lower-cased before check
        assert!(!validate_redirect_uri("HTTPS://user@example.com/cb"));
    }

    /// Fragment components (#) in redirect URIs must be rejected per RFC 6749 §3.1.2.
    /// Fragments cannot be meaningfully matched during authorize-time validation.
    #[test]
    fn test_validate_redirect_uri_rejects_fragment() {
        assert!(!validate_redirect_uri("https://example.com/cb#fragment"));
        assert!(!validate_redirect_uri("https://example.com/cb#"));
        assert!(!validate_redirect_uri("http://localhost/cb#anchor"));
        assert!(!validate_redirect_uri("myapp://oauth/cb#frag"));
    }

    /// Verify that clean https:// URIs without credentials or fragments are still accepted.
    #[test]
    fn test_validate_redirect_uri_clean_https_still_accepted() {
        assert!(validate_redirect_uri("https://example.com/callback"));
        assert!(validate_redirect_uri(
            "https://app.myservice.io/oauth/cb?query=param"
        ));
        assert!(validate_redirect_uri("https://localhost/cb")); // https localhost is fine
    }

    /// Registration must reject a URI with embedded credentials via the HTTP endpoint.
    #[tokio::test]
    async fn test_register_rejects_redirect_uri_with_embedded_credentials() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let resp = app
            .oneshot(post_json(
                "/oauth/register",
                serde_json::json!({
                    "client_name": "Phishing Client",
                    "redirect_uris": ["https://user:pass@evil.com/steal"]
                }),
            ))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(
            json["error"], "invalid_redirect_uri",
            "embedded credentials in redirect_uri must be rejected, got: {json}"
        );
    }

    /// Registration must reject a URI containing a fragment component.
    #[tokio::test]
    async fn test_register_rejects_redirect_uri_with_fragment() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let resp = app
            .oneshot(post_json(
                "/oauth/register",
                serde_json::json!({
                    "client_name": "Fragment Client",
                    "redirect_uris": ["https://example.com/cb#fragment"]
                }),
            ))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(
            json["error"], "invalid_redirect_uri",
            "fragment in redirect_uri must be rejected, got: {json}"
        );
    }

    // ── auto_approve production guard ─────────────────────────────────────────

    #[test]
    #[should_panic(expected = "auto_approve=true is not permitted on non-localhost")]
    fn test_auto_approve_on_non_localhost_panics() {
        let dir = TempDir::new().expect("tempdir");
        let store = TokenStore::open(dir.path()).expect("open store");
        let signing_key = SigningKey::generate(&mut OsRng);

        let state = OAuthState {
            token_store: Arc::new(Mutex::new(store)),
            signing_key: Arc::new(signing_key),
            server_did: "did:key:zServer".to_owned(),
            server_url: "http://0.0.0.0:8080".to_owned(),
            token_expiry_secs: 3600,
            refresh_expiry_secs: 86400,
            auto_approve: true,
            device_codes: Arc::new(Mutex::new(HashMap::new())),
            pending_consents: Arc::new(Mutex::new(FifoMap::new(MAX_PENDING_CONSENTS))),
            device_csrf_tokens: Arc::new(Mutex::new(FifoSet::new(MAX_DEVICE_CSRF_TOKENS))),
        };

        let _ = oauth_router(state);
    }

    #[test]
    fn test_auto_approve_on_localhost_does_not_panic() {
        let (state, _dir) = make_state(true);
        // Should succeed without panicking.
        let _ = oauth_router(state);
    }

    // ── redirect_uri scheme validation in registration ────────────────────────

    #[tokio::test]
    async fn test_register_rejects_javascript_redirect_uri() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let resp = app
            .oneshot(post_json(
                "/oauth/register",
                serde_json::json!({
                    "client_name": "Evil Client",
                    "redirect_uris": ["javascript:alert(document.cookie)"]
                }),
            ))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "invalid_redirect_uri");
    }

    #[tokio::test]
    async fn test_register_accepts_https_redirect_uri() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let resp = app
            .oneshot(post_json(
                "/oauth/register",
                serde_json::json!({
                    "client_name": "Legit Client",
                    "redirect_uris": ["https://app.example.com/callback"]
                }),
            ))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::CREATED);
        let json = body_json(resp).await;
        assert!(!json["client_id"].as_str().unwrap_or("").is_empty());
    }

    #[tokio::test]
    async fn test_register_rejects_http_non_localhost_redirect_uri() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let resp = app
            .oneshot(post_json(
                "/oauth/register",
                serde_json::json!({
                    "client_name": "Bad Client",
                    "redirect_uris": ["http://evil.example.com/callback"]
                }),
            ))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "invalid_redirect_uri");
    }

    // ── HTML template helpers ─────────────────────────────────────────────────

    #[test]
    fn test_consent_page_escapes_xss_in_client_name() {
        let html = consent_page(
            "<script>alert(1)</script>",
            "safe-client-id",
            "https://example.com/cb",
            "read",
            "challenge",
            "",
            "test-csrf-token",
        );
        assert!(!html.contains("<script>alert(1)</script>"));
        assert!(html.contains("&lt;script&gt;"));
    }

    #[test]
    fn test_error_page_escapes_message() {
        let html = error_page("<b>bad</b>");
        assert!(!html.contains("<b>bad</b>"));
        assert!(html.contains("&lt;b&gt;bad&lt;/b&gt;"));
    }

    // ── redirect helpers ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_redirect_with_code_includes_state() {
        let resp = redirect_with_code(
            "https://app.example.com/cb",
            "mycode",
            Some("oauth-state-123"),
        );
        let location = resp
            .headers()
            .get("location")
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        assert!(location.contains("code=mycode"));
        assert!(location.contains("state=oauth-state-123"));
    }

    #[tokio::test]
    async fn test_redirect_with_code_no_state() {
        let resp = redirect_with_code("https://app.example.com/cb", "mycode", None);
        let location = resp
            .headers()
            .get("location")
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        assert!(location.contains("code=mycode"));
        assert!(!location.contains("state="));
    }

    #[tokio::test]
    async fn test_redirect_with_error_includes_state() {
        let resp = redirect_with_error(
            "https://app.example.com/cb",
            "access_denied",
            "user denied",
            Some("mystate"),
        );
        let location = resp
            .headers()
            .get("location")
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        assert!(location.contains("error=access_denied"));
        assert!(location.contains("state=mystate"));
    }

    // ── #25 CSRF protection ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_consent_post_missing_csrf_returns_400() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (_, challenge) = pkce_pair();

        // POST without csrf_token field at all.
        let form = format!(
            "client_id={client_id}&redirect_uri={redirect_uri}&scope=read&code_challenge={challenge}&approved=true"
        );
        let resp = app
            .oneshot(post_form("/oauth/authorize", &form))
            .await
            .expect("request");

        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "missing csrf_token must return 400"
        );
    }

    #[tokio::test]
    async fn test_consent_post_invalid_csrf_returns_400() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (_, challenge) = pkce_pair();

        // POST with a forged csrf_token that was never issued.
        let form = format!(
            "client_id={client_id}&redirect_uri={redirect_uri}&scope=read&code_challenge={challenge}&approved=true&csrf_token=forged-token-value"
        );
        let resp = app
            .oneshot(post_form("/oauth/authorize", &form))
            .await
            .expect("request");

        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "invalid csrf_token must return 400"
        );
    }

    #[tokio::test]
    async fn test_consent_post_replayed_csrf_returns_400() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (_, challenge) = pkce_pair();

        // Obtain a real CSRF token by GETting the consent page.
        let csrf = get_consent_csrf_token(&app, &client_id, redirect_uri, &challenge).await;

        let form = format!(
            "client_id={client_id}&redirect_uri={redirect_uri}&scope=read&code_challenge={challenge}&approved=true&csrf_token={csrf}"
        );

        // First POST consumes the token.
        app.clone()
            .oneshot(post_form("/oauth/authorize", &form))
            .await
            .expect("first post");

        // Second POST with the same token must fail.
        let resp = app
            .oneshot(post_form("/oauth/authorize", &form))
            .await
            .expect("second post");

        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "replayed csrf_token must return 400"
        );
    }

    // ── #26 state parameter length cap ───────────────────────────────────────

    #[tokio::test]
    async fn test_authorize_get_state_too_long_returns_400() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (_, challenge) = pkce_pair();
        let long_state = "x".repeat(513);

        let authorize_uri = format!(
            "/oauth/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope=read&code_challenge={challenge}&code_challenge_method=S256&state={long_state}"
        );
        let resp = app.oneshot(get_req(&authorize_uri)).await.expect("request");

        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "state > 512 bytes must return 400"
        );
        let json = body_json(resp).await;
        assert_eq!(json["error"], "invalid_request");
        assert!(
            json["error_description"]
                .as_str()
                .unwrap_or("")
                .contains("512"),
            "error must mention the 512-byte limit"
        );
    }

    #[tokio::test]
    async fn test_authorize_get_state_at_limit_is_accepted() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (_, challenge) = pkce_pair();
        let state_at_limit = "x".repeat(512);

        let authorize_uri = format!(
            "/oauth/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope=read&code_challenge={challenge}&code_challenge_method=S256&state={state_at_limit}"
        );
        let resp = app.oneshot(get_req(&authorize_uri)).await.expect("request");

        // 200 (consent page shown) — state length is exactly at the limit.
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "state of exactly 512 bytes must be accepted"
        );
    }

    // ── test_error_description_does_not_leak_internal_detail ─────────────────

    /// Bad authorization code: the error_description must be a stable generic
    /// string and must NOT contain internal implementation detail.
    #[tokio::test]
    async fn test_auth_code_error_description_is_generic() {
        let (state, _dir) = make_state(true);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (verifier, _) = pkce_pair();

        // Submit a completely bogus authorization code with a valid verifier.
        let body = format!(
            "grant_type=authorization_code&code=BOGUS_CODE&code_verifier={verifier}&client_id={client_id}&redirect_uri={redirect_uri}"
        );
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/token", &body))
            .await
            .expect("token");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "invalid_grant");

        let desc = json["error_description"].as_str().unwrap_or("");
        // Must equal the generic message — not a raw error string.
        assert_eq!(
            desc, "invalid authorization code",
            "error_description must be generic, got: {desc:?}"
        );
        // Must not bleed internal implementation strings.
        for forbidden in &[
            "signature",
            "InvalidToken",
            "jsonwebtoken",
            "Base64",
            "decode",
        ] {
            assert!(
                !desc.contains(forbidden),
                "error_description must not contain {forbidden:?}, got: {desc:?}"
            );
        }
    }

    /// Bad refresh token: the error_description must be a stable generic string
    /// and must NOT contain internal implementation detail.
    #[tokio::test]
    async fn test_refresh_token_error_description_is_generic() {
        let (state, _dir) = make_state(true);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;

        // Submit a completely bogus refresh token.
        let body =
            format!("grant_type=refresh_token&refresh_token=BOGUS_REFRESH&client_id={client_id}");
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/token", &body))
            .await
            .expect("refresh");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "invalid_grant");

        let desc = json["error_description"].as_str().unwrap_or("");
        // Must equal the generic message — not a raw error string.
        assert_eq!(
            desc, "invalid refresh token",
            "error_description must be generic, got: {desc:?}"
        );
        // Must not bleed internal implementation strings.
        for forbidden in &[
            "signature",
            "InvalidToken",
            "jsonwebtoken",
            "Base64",
            "decode",
            "not found",
        ] {
            assert!(
                !desc.contains(forbidden),
                "error_description must not contain {forbidden:?}, got: {desc:?}"
            );
        }
    }

    /// Unsupported grant_type: the error_description must be a stable generic
    /// string and must NOT reflect the caller-supplied grant_type value back
    /// into the response body.
    #[tokio::test]
    async fn test_unsupported_grant_type_does_not_reflect_input() {
        let (state, _dir) = make_state(true);
        let app = oauth_router(state);

        // A distinctive, caller-controlled grant_type value.
        let injected = "INJECTED_MARKER_xyzzy";
        let body = format!("grant_type={injected}");
        let resp = app
            .clone()
            .oneshot(post_form("/oauth/token", &body))
            .await
            .expect("token");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"], "unsupported_grant_type");

        let desc = json["error_description"].as_str().unwrap_or("");
        // Must equal the generic message — not echo the caller's value.
        assert_eq!(
            desc, "unsupported grant_type",
            "error_description must be generic, got: {desc:?}"
        );
        // The caller-supplied value must not appear anywhere in the response.
        assert!(
            !desc.contains(injected),
            "error_description must not reflect caller input {injected:?}, got: {desc:?}"
        );
    }

    // ── Fix 1: redirect_uri count cap ─────────────────────────────────────────

    /// Registering with more than MAX_REDIRECT_URIS_PER_CLIENT URIs is rejected
    /// with 400 invalid_client_metadata (DoS guard).
    #[tokio::test]
    async fn test_register_too_many_redirect_uris_rejected() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        // Build a list of 11 valid redirect URIs (one over the limit of 10).
        let uris: Vec<String> = (0..=MAX_REDIRECT_URIS_PER_CLIENT)
            .map(|i| format!("https://example.com/cb{i}"))
            .collect();

        let resp = app
            .oneshot(post_json(
                "/oauth/register",
                serde_json::json!({
                    "client_name": "Too Many URIs",
                    "redirect_uris": uris,
                }),
            ))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(
            json["error"], "invalid_client_metadata",
            "expected invalid_client_metadata, got: {json}"
        );
        assert!(
            json["error_description"]
                .as_str()
                .unwrap_or("")
                .contains("redirect_uri"),
            "error_description must mention redirect_uri, got: {json}"
        );
    }

    /// Registering with exactly MAX_REDIRECT_URIS_PER_CLIENT URIs (the boundary)
    /// must succeed with 201 Created.
    #[tokio::test]
    async fn test_register_exactly_max_redirect_uris_accepted() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let uris: Vec<String> = (0..MAX_REDIRECT_URIS_PER_CLIENT)
            .map(|i| format!("https://example.com/cb{i}"))
            .collect();

        let resp = app
            .oneshot(post_json(
                "/oauth/register",
                serde_json::json!({
                    "client_name": "Max URIs Client",
                    "redirect_uris": uris,
                }),
            ))
            .await
            .expect("request");

        assert_eq!(
            resp.status(),
            StatusCode::CREATED,
            "exactly MAX_REDIRECT_URIS_PER_CLIENT URIs must be accepted"
        );
        let json = body_json(resp).await;
        assert!(!json["client_id"].as_str().unwrap_or("").is_empty());
    }

    // ── Fix 2: device user_code length cap ───────────────────────────────────

    /// A device approval POST with a huge user_code (after a valid CSRF token)
    /// must be rejected with an error page — no large allocations or O(n) scan.
    #[tokio::test]
    async fn test_device_approve_oversized_user_code_rejected() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        // Obtain a genuine CSRF token via GET /oauth/device so CSRF validation passes.
        let csrf = device_csrf_token(&app).await;

        // Submit a user_code that is far beyond MAX_USER_CODE_LENGTH.
        let huge_code = "A".repeat(10_000);
        let body = format!("user_code={huge_code}&approved=true&csrf_token={csrf}");
        let resp = app
            .oneshot(post_form("/oauth/device", &body))
            .await
            .expect("request");

        // Handler returns an error page (HTML, status 200 like the other error
        // pages in device_approve_handler).
        assert_eq!(resp.status(), StatusCode::OK);
        let html = String::from_utf8(body_vec(resp).await).unwrap();
        assert!(
            html.contains("Error") || html.contains("Invalid"),
            "oversized user_code must yield an error page, got: {html}"
        );
        // Must NOT produce a success page.
        assert!(
            !html.contains("Authorized"),
            "oversized user_code must not approve the device, got: {html}"
        );
    }

    /// A normal-length user_code still goes through the full approval flow.
    /// (Reuses the existing device_approve path — exercises the green case.)
    #[tokio::test]
    async fn test_device_approve_normal_length_user_code_accepted() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        let client_id = register_client(&app, "https://app.example.com/cb").await;

        // Request a device code so a real user_code exists in the map.
        let resp = app
            .clone()
            .oneshot(post_form(
                "/oauth/device/code",
                &format!("client_id={client_id}&scope=read"),
            ))
            .await
            .unwrap();
        let user_code = body_json(resp).await["user_code"]
            .as_str()
            .unwrap()
            .to_owned();

        // Approve with the real (9-char) user_code — must succeed.
        let csrf = device_csrf_token(&app).await;
        let body = format!("user_code={user_code}&scope=read&approved=true&csrf_token={csrf}");
        let resp = app
            .oneshot(post_form("/oauth/device", &body))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::OK);
        let html = String::from_utf8(body_vec(resp).await).unwrap();
        assert!(
            html.contains("Authorized"),
            "normal-length user_code must be approved, got: {html}"
        );
    }

    // ── Fix 3: Cache-Control / Pragma on token responses ─────────────────────

    /// Successful token responses must carry Cache-Control: no-store and
    /// Pragma: no-cache (RFC 6749 §5.1).
    #[tokio::test]
    async fn test_token_success_response_has_no_store_headers() {
        let (state, _dir) = make_state(true);
        let app = oauth_router(state);

        let redirect_uri = "https://app.example.com/cb";
        let client_id = register_client(&app, redirect_uri).await;
        let (verifier, challenge) = pkce_pair();

        let authorize_uri = format!(
            "/oauth/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope=read&code_challenge={challenge}&code_challenge_method=S256"
        );
        let resp = app.clone().oneshot(get_req(&authorize_uri)).await.unwrap();
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
            .oneshot(post_form("/oauth/token", &body))
            .await
            .expect("token");

        assert_eq!(resp.status(), StatusCode::OK);

        let cc = resp
            .headers()
            .get("cache-control")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert_eq!(
            cc, "no-store",
            "token success must have Cache-Control: no-store, got: {cc:?}"
        );

        let pragma = resp
            .headers()
            .get("pragma")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert_eq!(
            pragma, "no-cache",
            "token success must have Pragma: no-cache, got: {pragma:?}"
        );
    }

    /// Error token responses must also carry Cache-Control: no-store and
    /// Pragma: no-cache (RFC 6749 §5.2).
    #[tokio::test]
    async fn test_token_error_response_has_no_store_headers() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);

        // Trigger a predictable token error: unsupported grant_type.
        let resp = app
            .oneshot(post_form("/oauth/token", "grant_type=implicit"))
            .await
            .expect("request");

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let cc = resp
            .headers()
            .get("cache-control")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert_eq!(
            cc, "no-store",
            "token error must have Cache-Control: no-store, got: {cc:?}"
        );

        let pragma = resp
            .headers()
            .get("pragma")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert_eq!(
            pragma, "no-cache",
            "token error must have Pragma: no-cache, got: {pragma:?}"
        );
    }

    // ── generate_user_code ────────────────────────────────────────────────────

    /// User codes must be 8 uppercase chars from the unambiguous alphabet and
    /// two successive calls must produce distinct codes (collision probability
    /// ~1e-12; a flaky failure here would indicate a broken RNG).
    #[test]
    fn test_generate_user_code_format_and_uniqueness() {
        const ALPHABET: &str = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
        let code1 = generate_user_code();
        let code2 = generate_user_code();

        assert_eq!(code1.len(), 8, "user code must be 8 chars, got: {code1}");
        for ch in code1.chars() {
            assert!(
                ALPHABET.contains(ch),
                "char {ch:?} not in unambiguous alphabet: {code1}"
            );
        }
        assert_ne!(code1, code2, "two successive codes must differ");
    }

    /// format_user_code must produce the XXXX-XXXX form from an 8-char raw code.
    #[test]
    fn test_format_user_code_inserts_dash() {
        let raw = generate_user_code();
        assert_eq!(raw.len(), 8);
        let formatted = format_user_code(&raw);
        assert_eq!(
            formatted.len(),
            9,
            "formatted code must be 9 chars: {formatted}"
        );
        assert_eq!(
            &formatted[4..5],
            "-",
            "dash must be at position 4: {formatted}"
        );
        assert_eq!(&formatted[..4], &raw[..4]);
        assert_eq!(&formatted[5..], &raw[4..]);
    }

    // ── Device poll rate limiting (RFC 8628 §3.5) ────────────────────────────

    /// Polling the same device code twice in rapid succession must yield
    /// `slow_down` on the second call, enforcing the 5-second interval.
    #[tokio::test]
    async fn test_device_poll_rate_limit_slow_down() {
        let (state, _dir) = make_state(false);
        let app = oauth_router(state);
        let client_id = register_client(&app, "https://app.example.com/cb").await;

        let resp = app
            .clone()
            .oneshot(post_form(
                "/oauth/device/code",
                &format!("client_id={client_id}&scope=read"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        let device_code = json["device_code"].as_str().unwrap().to_owned();

        let poll_body = format!(
            "grant_type=urn:ietf:params:oauth:grant-type:device_code&device_code={device_code}&client_id={client_id}"
        );

        // First poll: no last_polled set → authorization_pending.
        let resp1 = app
            .clone()
            .oneshot(post_form("/oauth/token", &poll_body))
            .await
            .unwrap();
        assert_eq!(resp1.status(), StatusCode::BAD_REQUEST);
        let json1 = body_json(resp1).await;
        assert_eq!(
            json1["error"], "authorization_pending",
            "first poll must return authorization_pending, got: {json1}"
        );

        // Second poll immediately: last_polled was just set → slow_down.
        let resp2 = app
            .clone()
            .oneshot(post_form("/oauth/token", &poll_body))
            .await
            .unwrap();
        assert_eq!(resp2.status(), StatusCode::BAD_REQUEST);
        let json2 = body_json(resp2).await;
        assert_eq!(
            json2["error"], "slow_down",
            "rapid second poll must return slow_down, got: {json2}"
        );
    }

    /// After the poll interval has elapsed, a pending code must return
    /// `authorization_pending` again (not `slow_down`).  Injects a backdated
    /// `last_polled` directly into the map to avoid sleeping in the test.
    #[tokio::test]
    async fn test_device_poll_after_interval_returns_authorization_pending() {
        let (state, _dir) = make_state(false);
        let codes = state.device_codes.clone();
        let app = oauth_router(state);
        let client_id = register_client(&app, "https://app.example.com/cb").await;

        let resp = app
            .clone()
            .oneshot(post_form(
                "/oauth/device/code",
                &format!("client_id={client_id}&scope=read"),
            ))
            .await
            .unwrap();
        let json = body_json(resp).await;
        let device_code = json["device_code"].as_str().unwrap().to_owned();

        // Back-date last_polled so the interval check passes.
        {
            let mut map = codes.lock().unwrap();
            if let Some(entry) = map.get_mut(&device_code) {
                entry.last_polled =
                    Some(Instant::now() - Duration::from_secs(DEVICE_POLL_INTERVAL_SECS + 1));
            }
        }

        let poll_body = format!(
            "grant_type=urn:ietf:params:oauth:grant-type:device_code&device_code={device_code}&client_id={client_id}"
        );
        let resp = app
            .oneshot(post_form("/oauth/token", &poll_body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(
            json["error"], "authorization_pending",
            "poll after interval must return authorization_pending, got: {json}"
        );
    }

    // ── FifoMap / FifoSet bounded eviction ────────────────────────────────────

    /// Inserting cap+N entries into FifoMap must evict exactly the N oldest,
    /// keeping the map size at cap with the most recent entries present.
    #[test]
    fn test_fifo_map_evicts_oldest_at_cap() {
        const CAP: usize = 4;
        let mut map: FifoMap<String, u32> = FifoMap::new(CAP);
        for i in 0..CAP + 2 {
            map.insert(format!("key-{i}"), i as u32);
        }
        assert_eq!(map.len(), CAP, "map must stay at cap after overflow");
        // Oldest two ("key-0", "key-1") must have been evicted.
        assert!(map.get("key-0").is_none(), "oldest entry must be evicted");
        assert!(
            map.get("key-1").is_none(),
            "second-oldest entry must be evicted"
        );
        // Most recent entries must still be present.
        assert_eq!(map.get("key-4"), Some(&4));
        assert_eq!(map.get("key-5"), Some(&5));
    }

    /// FifoSet bounded eviction mirrors FifoMap: oldest tokens evicted first.
    #[test]
    fn test_fifo_set_evicts_oldest_at_cap() {
        const CAP: usize = 3;
        let mut set: FifoSet<String> = FifoSet::new(CAP);
        for i in 0..CAP + 2 {
            set.insert(format!("tok-{i}"));
        }
        assert_eq!(set.len(), CAP, "set must stay at cap after overflow");
        assert!(!set.contains("tok-0"), "oldest token must be evicted");
        assert!(
            !set.contains("tok-1"),
            "second-oldest token must be evicted"
        );
        assert!(set.contains("tok-2"));
        assert!(set.contains("tok-3"));
        assert!(set.contains("tok-4"));
    }

    /// Inserting a key that already exists must update the value without
    /// affecting insertion order or growing the map.
    #[test]
    fn test_fifo_map_update_existing_key_no_eviction() {
        const CAP: usize = 3;
        let mut map: FifoMap<String, u32> = FifoMap::new(CAP);
        map.insert("a".to_owned(), 1);
        map.insert("b".to_owned(), 2);
        map.insert("a".to_owned(), 99); // update existing
        assert_eq!(map.len(), 2, "update must not grow the map");
        assert_eq!(map.get("a"), Some(&99), "value must be updated");
        assert_eq!(map.get("b"), Some(&2));

        // FIFO is by first insertion: "a" keeps its original slot, so inserting
        // a third distinct key fills the map without evicting anything.
        map.insert("c".to_owned(), 3);
        assert_eq!(map.len(), CAP);
    }

    /// device_csrf_tokens must stay bounded under a burst of device-page renders.
    #[tokio::test]
    async fn test_device_csrf_tokens_bounded_under_burst() {
        let (state, _dir) = make_state(false);
        let csrf_tokens = state.device_csrf_tokens.clone();
        let app = oauth_router(state);

        // Pre-fill to one below the cap.
        {
            let mut tokens = csrf_tokens.lock().unwrap();
            for i in 0..MAX_DEVICE_CSRF_TOKENS - 1 {
                tokens.insert(format!("old-csrf-{i}"));
            }
            assert_eq!(tokens.len(), MAX_DEVICE_CSRF_TOKENS - 1);
        }

        // Two more GET /oauth/device renders should keep the set at the cap.
        for _ in 0..2 {
            let resp = app
                .clone()
                .oneshot(get_req("/oauth/device"))
                .await
                .expect("device page");
            assert_eq!(resp.status(), StatusCode::OK);
        }

        let len = csrf_tokens.lock().unwrap().len();
        assert_eq!(
            len, MAX_DEVICE_CSRF_TOKENS,
            "device CSRF set must stay at cap, got {len}"
        );
    }
}
