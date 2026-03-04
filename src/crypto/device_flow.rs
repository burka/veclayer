//! OAuth 2.0 Device Authorization Grant client (RFC 8628).
//!
//! Provides types and polling logic for the device flow. HTTP calls are
//! injected via the [`HttpClient`] trait so this module compiles without
//! requiring `reqwest` as a direct dependency — callers supply the transport.
//!
//! # Typical usage
//!
//! ```ignore
//! let config = DeviceFlowConfig { ... };
//! let token = run_device_flow(&config, http, |auth| {
//!     println!("Visit {} and enter code {}", auth.verification_uri, auth.user_code);
//! }).await?;
//! ```

use std::collections::HashMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for a device authorization endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceFlowConfig {
    /// URL of the device authorization endpoint.
    pub device_authorization_url: String,
    /// URL of the token endpoint.
    pub token_url: String,
    /// OAuth 2.0 client identifier.
    pub client_id: String,
    /// Space-separated scopes, if required.
    pub scope: Option<String>,
}

/// Response from the device authorization endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceAuthorizationResponse {
    /// Opaque code used in subsequent token polling requests.
    pub device_code: String,
    /// Short human-friendly code displayed to the user.
    pub user_code: String,
    /// URL where the user completes authorization.
    pub verification_uri: String,
    /// Pre-filled URL including the user code (optional server extension).
    pub verification_uri_complete: Option<String>,
    /// Lifetime of `device_code` and `user_code` in seconds.
    pub expires_in: u64,
    /// Minimum polling interval in seconds (default: 5).
    pub interval: u64,
}

/// Successful token response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenResponse {
    /// Bearer token issued by the authorization server.
    pub access_token: String,
    /// Token type (typically `"Bearer"`).
    pub token_type: String,
    /// Lifetime of the access token in seconds.
    pub expires_in: Option<u64>,
    /// Refresh token, if issued.
    pub refresh_token: Option<String>,
    /// Space-separated scopes actually granted.
    pub scope: Option<String>,
}

/// Error during the device authorization flow.
#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
pub enum DeviceFlowError {
    #[error("device authorization request failed: {0}")]
    AuthorizationRequest(String),

    #[error("token polling failed: {0}")]
    TokenPoll(String),

    #[error("device code expired — user did not authorize in time")]
    ExpiredCode,

    #[error("access denied — user rejected the authorization request")]
    AccessDenied,

    #[error("server requested slower polling; back off and retry")]
    SlowDown,
}

// ──────────────────────────────────────────────────────────────────────────────
// HTTP abstraction
// ──────────────────────────────────────────────────────────────────────────────

/// Boxed future returned by [`HttpClient::post_form`].
pub type PostFormFuture<'a> = std::pin::Pin<
    Box<
        dyn std::future::Future<Output = Result<(u16, serde_json::Value), DeviceFlowError>>
            + Send
            + 'a,
    >,
>;

/// Minimal HTTP client trait used by the device flow implementation.
///
/// Implementors perform a single application/x-www-form-urlencoded POST and
/// return the response body as a JSON [`serde_json::Value`] together with the
/// HTTP status code.
///
/// # Async
///
/// The trait uses a boxed future so it can be object-safe and used without
/// generic parameters at the call sites in [`run_device_flow`].
pub trait HttpClient: Send + Sync {
    /// POST `url` with `form_fields` as the URL-encoded body.
    ///
    /// Returns `(status_code, body_json)`.
    fn post_form<'a>(
        &'a self,
        url: &'a str,
        form_fields: &'a HashMap<String, String>,
    ) -> PostFormFuture<'a>;
}

// ──────────────────────────────────────────────────────────────────────────────
// Core flow functions
// ──────────────────────────────────────────────────────────────────────────────

/// Request device authorization from the server.
///
/// Returns the authorization response containing the `user_code` to display.
pub async fn request_device_authorization(
    config: &DeviceFlowConfig,
    http: &dyn HttpClient,
) -> Result<DeviceAuthorizationResponse, DeviceFlowError> {
    let mut fields = HashMap::new();
    fields.insert("client_id".to_string(), config.client_id.clone());
    if let Some(scope) = &config.scope {
        fields.insert("scope".to_string(), scope.clone());
    }

    let (status, body) = http
        .post_form(&config.device_authorization_url, &fields)
        .await?;

    if status != 200 {
        let error_desc = body
            .get("error_description")
            .and_then(|v| v.as_str())
            .or_else(|| body.get("error").and_then(|v| v.as_str()))
            .unwrap_or("unknown error");
        return Err(DeviceFlowError::AuthorizationRequest(format!(
            "HTTP {status}: {error_desc}"
        )));
    }

    parse_device_authorization_response(&body)
}

/// Poll for token completion.
///
/// Returns `Ok(Some(token))` when the user has authorized, `Ok(None)` when
/// authorization is still pending, or `Err` on terminal failures.
pub async fn poll_for_token(
    config: &DeviceFlowConfig,
    device_code: &str,
    http: &dyn HttpClient,
) -> Result<Option<TokenResponse>, DeviceFlowError> {
    let mut fields = HashMap::new();
    fields.insert("client_id".to_string(), config.client_id.clone());
    fields.insert("device_code".to_string(), device_code.to_string());
    fields.insert(
        "grant_type".to_string(),
        "urn:ietf:params:oauth:grant-type:device_code".to_string(),
    );

    let (_status, body) = http.post_form(&config.token_url, &fields).await?;

    if let Some(access_token) = body.get("access_token").and_then(|v| v.as_str()) {
        return Ok(Some(TokenResponse {
            access_token: access_token.to_string(),
            token_type: body
                .get("token_type")
                .and_then(|v| v.as_str())
                .unwrap_or("Bearer")
                .to_string(),
            expires_in: body.get("expires_in").and_then(|v| v.as_u64()),
            refresh_token: body
                .get("refresh_token")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            scope: body
                .get("scope")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        }));
    }

    match body
        .get("error")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
    {
        "authorization_pending" => Ok(None),
        "slow_down" => Err(DeviceFlowError::SlowDown),
        "expired_token" => Err(DeviceFlowError::ExpiredCode),
        "access_denied" => Err(DeviceFlowError::AccessDenied),
        other => Err(DeviceFlowError::TokenPoll(format!(
            "unexpected error code: {other}"
        ))),
    }
}

/// Run the complete device authorization flow.
///
/// 1. Request device authorization from the server.
/// 2. Call `on_user_prompt` so the caller can display the `user_code` and
///    `verification_uri` to the user.
/// 3. Poll until the user authorizes, denies, or the code expires, respecting
///    the server-requested polling interval and `slow_down` back-offs.
pub async fn run_device_flow(
    config: &DeviceFlowConfig,
    http: &dyn HttpClient,
    on_user_prompt: impl Fn(&DeviceAuthorizationResponse),
) -> Result<TokenResponse, DeviceFlowError> {
    let auth = request_device_authorization(config, http).await?;
    on_user_prompt(&auth);

    let base_interval = auth.interval.max(5);
    let mut slow_down_offset: u64 = 0;

    loop {
        let interval_secs = base_interval + slow_down_offset;
        tokio::time::sleep(Duration::from_secs(interval_secs)).await;

        match poll_for_token(config, &auth.device_code, http).await {
            Ok(Some(token)) => return Ok(token),
            Ok(None) => {
                // Still pending — keep polling at current interval (RFC 8628 §3.5).
            }
            Err(DeviceFlowError::SlowDown) => {
                // Server requests we back off by 5 seconds; offset accumulates.
                slow_down_offset += 5;
            }
            Err(terminal) => return Err(terminal),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

fn parse_device_authorization_response(
    body: &serde_json::Value,
) -> Result<DeviceAuthorizationResponse, DeviceFlowError> {
    let required_str = |key: &str| -> Result<String, DeviceFlowError> {
        body.get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| {
                DeviceFlowError::AuthorizationRequest(format!(
                    "missing required field '{key}' in response"
                ))
            })
    };

    Ok(DeviceAuthorizationResponse {
        device_code: required_str("device_code")?,
        user_code: required_str("user_code")?,
        verification_uri: required_str("verification_uri")?,
        verification_uri_complete: body
            .get("verification_uri_complete")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        expires_in: body
            .get("expires_in")
            .and_then(|v| v.as_u64())
            .unwrap_or(300),
        interval: body.get("interval").and_then(|v| v.as_u64()).unwrap_or(5),
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Error display ──────────────────────────────────────────────────────────

    #[test]
    fn test_authorization_request_error_display() {
        let err = DeviceFlowError::AuthorizationRequest("HTTP 400: bad client".to_string());
        let msg = err.to_string();
        assert!(
            msg.contains("device authorization request failed"),
            "got: {msg}"
        );
        assert!(msg.contains("HTTP 400"), "got: {msg}");
    }

    #[test]
    fn test_token_poll_error_display() {
        let err = DeviceFlowError::TokenPoll("network timeout".to_string());
        let msg = err.to_string();
        assert!(msg.contains("token polling failed"), "got: {msg}");
        assert!(msg.contains("network timeout"), "got: {msg}");
    }

    #[test]
    fn test_expired_code_error_display() {
        let msg = DeviceFlowError::ExpiredCode.to_string();
        assert!(msg.contains("expired"), "got: {msg}");
    }

    #[test]
    fn test_access_denied_error_display() {
        let msg = DeviceFlowError::AccessDenied.to_string();
        assert!(msg.contains("denied"), "got: {msg}");
    }

    #[test]
    fn test_slow_down_error_display() {
        let msg = DeviceFlowError::SlowDown.to_string();
        assert!(msg.contains("slower"), "got: {msg}");
    }

    // ── Response parsing ───────────────────────────────────────────────────────

    #[test]
    fn test_parse_device_authorization_response_complete() {
        let body = serde_json::json!({
            "device_code": "dev123",
            "user_code": "ABCD-1234",
            "verification_uri": "https://example.com/activate",
            "verification_uri_complete": "https://example.com/activate?code=ABCD-1234",
            "expires_in": 600,
            "interval": 5
        });
        let resp = parse_device_authorization_response(&body).unwrap();
        assert_eq!(resp.device_code, "dev123");
        assert_eq!(resp.user_code, "ABCD-1234");
        assert_eq!(resp.verification_uri, "https://example.com/activate");
        assert_eq!(
            resp.verification_uri_complete.as_deref(),
            Some("https://example.com/activate?code=ABCD-1234")
        );
        assert_eq!(resp.expires_in, 600);
        assert_eq!(resp.interval, 5);
    }

    #[test]
    fn test_parse_device_authorization_response_minimal() {
        let body = serde_json::json!({
            "device_code": "dev456",
            "user_code": "XY-99",
            "verification_uri": "https://example.com/device"
        });
        let resp = parse_device_authorization_response(&body).unwrap();
        assert_eq!(resp.expires_in, 300); // default
        assert_eq!(resp.interval, 5); // default
        assert!(resp.verification_uri_complete.is_none());
    }

    #[test]
    fn test_parse_device_authorization_response_missing_device_code() {
        let body = serde_json::json!({
            "user_code": "XY-99",
            "verification_uri": "https://example.com/device"
        });
        let err = parse_device_authorization_response(&body).unwrap_err();
        assert!(matches!(err, DeviceFlowError::AuthorizationRequest(_)));
    }

    // ── Serialization roundtrip ────────────────────────────────────────────────

    #[test]
    fn test_device_flow_config_serialization_roundtrip() {
        let config = DeviceFlowConfig {
            device_authorization_url: "https://auth.example.com/device".to_string(),
            token_url: "https://auth.example.com/token".to_string(),
            client_id: "my-client".to_string(),
            scope: Some("openid profile".to_string()),
        };
        let json = serde_json::to_string(&config).unwrap();
        let restored: DeviceFlowConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.client_id, config.client_id);
        assert_eq!(restored.scope, config.scope);
    }

    #[test]
    fn test_token_response_serialization_roundtrip() {
        let token = TokenResponse {
            access_token: "tok_abc123".to_string(),
            token_type: "Bearer".to_string(),
            expires_in: Some(3600),
            refresh_token: Some("ref_xyz".to_string()),
            scope: Some("read write".to_string()),
        };
        let json = serde_json::to_string(&token).unwrap();
        let restored: TokenResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.access_token, token.access_token);
        assert_eq!(restored.refresh_token, token.refresh_token);
    }

    // ── Mock HTTP client for polling logic ────────────────────────────────────

    struct MockHttpClient {
        /// Responses to return in order for each post_form call.
        responses: std::sync::Mutex<std::collections::VecDeque<(u16, serde_json::Value)>>,
    }

    impl MockHttpClient {
        fn new(responses: Vec<(u16, serde_json::Value)>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses.into_iter().collect()),
            }
        }
    }

    impl HttpClient for MockHttpClient {
        fn post_form<'a>(
            &'a self,
            _url: &'a str,
            _fields: &'a HashMap<String, String>,
        ) -> PostFormFuture<'a> {
            let response = self
                .responses
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or((500, serde_json::json!({"error": "no more responses"})));
            Box::pin(async move { Ok(response) })
        }
    }

    #[test]
    fn test_request_device_authorization_success() {
        let http = MockHttpClient::new(vec![(
            200,
            serde_json::json!({
                "device_code": "dev_ok",
                "user_code": "USER-CODE",
                "verification_uri": "https://example.com/auth",
                "expires_in": 300,
                "interval": 5
            }),
        )]);

        let config = DeviceFlowConfig {
            device_authorization_url: "https://example.com/device".to_string(),
            token_url: "https://example.com/token".to_string(),
            client_id: "test".to_string(),
            scope: None,
        };

        let rt = tokio::runtime::Runtime::new().unwrap();
        let auth = rt
            .block_on(request_device_authorization(&config, &http))
            .unwrap();
        assert_eq!(auth.device_code, "dev_ok");
        assert_eq!(auth.user_code, "USER-CODE");
    }

    #[test]
    fn test_request_device_authorization_http_error() {
        let http = MockHttpClient::new(vec![(
            400,
            serde_json::json!({
                "error": "invalid_client",
                "error_description": "Unknown client_id"
            }),
        )]);

        let config = DeviceFlowConfig {
            device_authorization_url: "https://example.com/device".to_string(),
            token_url: "https://example.com/token".to_string(),
            client_id: "bad-client".to_string(),
            scope: None,
        };

        let rt = tokio::runtime::Runtime::new().unwrap();
        let err = rt
            .block_on(request_device_authorization(&config, &http))
            .unwrap_err();
        assert!(matches!(err, DeviceFlowError::AuthorizationRequest(_)));
    }

    #[test]
    fn test_poll_for_token_pending() {
        let http = MockHttpClient::new(vec![(
            200,
            serde_json::json!({"error": "authorization_pending"}),
        )]);

        let config = DeviceFlowConfig {
            device_authorization_url: "https://example.com/device".to_string(),
            token_url: "https://example.com/token".to_string(),
            client_id: "test".to_string(),
            scope: None,
        };

        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt
            .block_on(poll_for_token(&config, "dev_code", &http))
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_poll_for_token_success() {
        let http = MockHttpClient::new(vec![(
            200,
            serde_json::json!({
                "access_token": "at_secret",
                "token_type": "Bearer",
                "expires_in": 3600
            }),
        )]);

        let config = DeviceFlowConfig {
            device_authorization_url: "https://example.com/device".to_string(),
            token_url: "https://example.com/token".to_string(),
            client_id: "test".to_string(),
            scope: None,
        };

        let rt = tokio::runtime::Runtime::new().unwrap();
        let token = rt
            .block_on(poll_for_token(&config, "dev_code", &http))
            .unwrap()
            .expect("token should be present");
        assert_eq!(token.access_token, "at_secret");
        assert_eq!(token.token_type, "Bearer");
    }

    #[test]
    fn test_poll_for_token_expired() {
        let http = MockHttpClient::new(vec![(200, serde_json::json!({"error": "expired_token"}))]);

        let config = DeviceFlowConfig {
            device_authorization_url: "https://example.com/device".to_string(),
            token_url: "https://example.com/token".to_string(),
            client_id: "test".to_string(),
            scope: None,
        };

        let rt = tokio::runtime::Runtime::new().unwrap();
        let err = rt
            .block_on(poll_for_token(&config, "dev_code", &http))
            .unwrap_err();
        assert!(matches!(err, DeviceFlowError::ExpiredCode));
    }

    #[test]
    fn test_poll_for_token_access_denied() {
        let http = MockHttpClient::new(vec![(200, serde_json::json!({"error": "access_denied"}))]);

        let config = DeviceFlowConfig {
            device_authorization_url: "https://example.com/device".to_string(),
            token_url: "https://example.com/token".to_string(),
            client_id: "test".to_string(),
            scope: None,
        };

        let rt = tokio::runtime::Runtime::new().unwrap();
        let err = rt
            .block_on(poll_for_token(&config, "dev_code", &http))
            .unwrap_err();
        assert!(matches!(err, DeviceFlowError::AccessDenied));
    }
}
