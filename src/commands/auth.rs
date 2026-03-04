//! Authentication commands — token minting, OAuth login, and status display.

use std::collections::HashMap;
use std::io::{self, IsTerminal};
use std::path::{Path, PathBuf};

use owo_colors::{OwoColorize, Stream};
use serde::{Deserialize, Serialize};

use crate::auth::capability::Capability;
use crate::auth::token::{self, Claims};
use crate::crypto::{keypair, keystore};
use crate::util::{set_file_mode_600, unix_now};
use crate::Result;

// ──────────────────────────────────────────────────────────────────────────────
// Token cache types
// ──────────────────────────────────────────────────────────────────────────────

const TOKEN_CACHE_FILENAME: &str = "token_cache.json";

/// A locally-cached OAuth token for a remote server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedToken {
    pub access_token: String,
    pub refresh_token: Option<String>,
    /// Unix timestamp when this token expires.
    pub expires_at: u64,
    pub scope: String,
}

/// The full token cache file, keyed by server URL.
#[derive(Debug, Default, Serialize, Deserialize)]
struct TokenCache {
    servers: HashMap<String, CachedToken>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Token cache helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Returns the path to the token cache file.
pub fn cache_path(data_dir: &Path) -> PathBuf {
    data_dir.join(TOKEN_CACHE_FILENAME)
}

/// Persist a token for `server_url` into the local cache.
///
/// Writes atomically via a temporary sibling file and restricts permissions
/// to 0o600 (owner read/write only) because the file contains bearer tokens.
pub fn save_token(data_dir: &Path, server_url: &str, token: &CachedToken) -> Result<()> {
    let path = cache_path(data_dir);
    let mut cache = load_cache(&path)?;
    cache.servers.insert(server_url.to_string(), token.clone());
    let json = serde_json::to_string_pretty(&cache)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json)?;
    set_file_mode_600(&tmp)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// Load the cached token for `server_url`, if any.
pub fn load_token(data_dir: &Path, server_url: &str) -> Result<Option<CachedToken>> {
    let path = cache_path(data_dir);
    let cache = load_cache(&path)?;
    Ok(cache.servers.get(server_url).cloned())
}

/// Return all cached server tokens.
pub fn list_cached(data_dir: &Path) -> Result<HashMap<String, CachedToken>> {
    let path = cache_path(data_dir);
    let cache = load_cache(&path)?;
    Ok(cache.servers)
}

fn load_cache(path: &Path) -> Result<TokenCache> {
    if !path.exists() {
        return Ok(TokenCache::default());
    }
    let json = std::fs::read_to_string(path)?;
    let cache: TokenCache = serde_json::from_str(&json)?;
    Ok(cache)
}

// ──────────────────────────────────────────────────────────────────────────────
// Duration parsing
// ──────────────────────────────────────────────────────────────────────────────

/// Parse a duration string into seconds.
///
/// Accepts:
/// - `"Nh"` — N hours (e.g. `"1h"`, `"24h"`)
/// - `"Nd"` — N days (e.g. `"30d"`, `"7d"`)
/// - `"N"`  — N seconds (plain integer, e.g. `"3600"`)
pub fn parse_duration_secs(s: &str) -> Result<u64> {
    if let Some(hours) = s.strip_suffix('h') {
        let n: u64 = hours
            .parse()
            .map_err(|_| crate::Error::Parse(format!("invalid duration: {s}")))?;
        return Ok(n * 3600);
    }
    if let Some(days) = s.strip_suffix('d') {
        let n: u64 = days
            .parse()
            .map_err(|_| crate::Error::Parse(format!("invalid duration: {s}")))?;
        return Ok(n * 86400);
    }
    s.parse::<u64>()
        .map_err(|_| crate::Error::Parse(format!("invalid duration: {s}")))
}

// ──────────────────────────────────────────────────────────────────────────────
// auth token
// ──────────────────────────────────────────────────────────────────────────────

/// Mint a JWT access token using the local identity keypair.
///
/// Prints the raw token to stdout (nothing else) so the output can be piped.
pub async fn auth_token(
    data_dir: &Path,
    capability: &str,
    expires: &str,
    audience: Option<&str>,
) -> Result<()> {
    auth_token_with_passphrase(data_dir, capability, expires, audience, None).await
}

pub(crate) async fn auth_token_with_passphrase(
    data_dir: &Path,
    capability: &str,
    expires: &str,
    audience: Option<&str>,
    passphrase: Option<&str>,
) -> Result<()> {
    let path = keystore::keystore_path(data_dir);
    if !keystore::exists(&path) {
        return Err(crate::Error::NotFound(
            "No identity found. Run `veclayer identity init` first.".to_string(),
        ));
    }

    let passphrase = match passphrase {
        Some(p) => p.to_string(),
        None => resolve_passphrase()?,
    };

    let signing_key =
        keystore::load(&passphrase, &path).map_err(|e| crate::Error::Crypto(e.to_string()))?;

    let cap: Capability = capability
        .parse()
        .map_err(|e: String| crate::Error::InvalidOperation(e))?;

    let expiry_secs = parse_duration_secs(expires)?;

    let now = unix_now();
    let own_did = keypair::to_did(&signing_key.verifying_key());
    let aud = audience.unwrap_or(&own_did).to_string();

    let claims = Claims::new(own_did, aud, cap, now, now + expiry_secs);

    let jwt =
        token::mint(&signing_key, &claims).map_err(|e| crate::Error::Crypto(e.to_string()))?;

    println!("{jwt}");
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// auth login
// ──────────────────────────────────────────────────────────────────────────────

/// Authenticate against a remote VecLayer server using the OAuth device flow.
///
/// Requires the `llm` feature (which includes `reqwest`).
#[cfg(feature = "llm")]
pub async fn auth_login(data_dir: &Path, server_url: &str) -> Result<()> {
    use crate::crypto::device_flow::{
        run_device_flow, DeviceFlowConfig, HttpClient, PostFormFuture,
    };

    struct ReqwestHttpClient {
        client: reqwest::Client,
    }

    impl HttpClient for ReqwestHttpClient {
        fn post_form<'a>(
            &'a self,
            url: &'a str,
            fields: &'a HashMap<String, String>,
        ) -> PostFormFuture<'a> {
            Box::pin(async move {
                let params: Vec<(&str, &str)> = fields
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.as_str()))
                    .collect();
                let resp = self
                    .client
                    .post(url)
                    .form(&params)
                    .send()
                    .await
                    .map_err(|e| {
                        crate::crypto::device_flow::DeviceFlowError::AuthorizationRequest(
                            e.to_string(),
                        )
                    })?;
                let status = resp.status().as_u16();
                let body: serde_json::Value = resp.json().await.map_err(|e| {
                    crate::crypto::device_flow::DeviceFlowError::AuthorizationRequest(e.to_string())
                })?;
                Ok((status, body))
            })
        }
    }

    let http = ReqwestHttpClient {
        client: reqwest::Client::new(),
    };

    // Discover OAuth endpoints.
    let meta_url = format!("{server_url}/.well-known/oauth-authorization-server");
    let meta: serde_json::Value = http
        .client
        .get(&meta_url)
        .send()
        .await
        .map_err(|e| crate::Error::InvalidOperation(format!("metadata fetch failed: {e}")))?
        .json()
        .await
        .map_err(|e| crate::Error::InvalidOperation(format!("metadata parse failed: {e}")))?;

    let device_url = meta
        .get("device_authorization_endpoint")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            crate::Error::InvalidOperation(
                "server metadata missing device_authorization_endpoint".to_string(),
            )
        })?
        .to_string();

    let token_url = meta
        .get("token_endpoint")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            crate::Error::InvalidOperation("server metadata missing token_endpoint".to_string())
        })?
        .to_string();

    // Dynamic client registration.
    let reg_url = format!("{server_url}/oauth/register");
    let reg_resp: serde_json::Value = http
        .client
        .post(&reg_url)
        .json(&serde_json::json!({
            "client_name": "veclayer-cli",
            "grant_types": ["urn:ietf:params:oauth:grant-type:device_code"],
            "token_endpoint_auth_method": "none"
        }))
        .send()
        .await
        .map_err(|e| crate::Error::InvalidOperation(format!("client registration failed: {e}")))?
        .json()
        .await
        .map_err(|e| {
            crate::Error::InvalidOperation(format!("registration response parse failed: {e}"))
        })?;

    let client_id = reg_resp
        .get("client_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            crate::Error::InvalidOperation("registration response missing client_id".to_string())
        })?
        .to_string();

    let config = DeviceFlowConfig {
        device_authorization_url: device_url,
        token_url,
        client_id,
        scope: Some("read write".to_string()),
    };

    let token_resp = run_device_flow(&config, &http, |auth| {
        eprintln!(
            "\nOpen this URL to authorize:\n  {}\n\nOr visit {} and enter code: {}",
            auth.verification_uri_complete
                .as_deref()
                .unwrap_or(&auth.verification_uri),
            auth.verification_uri,
            auth.user_code
        );
    })
    .await
    .map_err(|e| crate::Error::InvalidOperation(e.to_string()))?;

    let now = unix_now();
    let expires_at = now + token_resp.expires_in.unwrap_or(3600);
    let scope = token_resp.scope.clone().unwrap_or_default();

    let cached = CachedToken {
        access_token: token_resp.access_token,
        refresh_token: token_resp.refresh_token,
        expires_at,
        scope: scope.clone(),
    };

    save_token(data_dir, server_url, &cached)?;
    let path = cache_path(data_dir);

    println!(
        "Logged in to {server_url}\n  Scope: {scope}\n  Token cached at: {}",
        path.display()
    );

    Ok(())
}

/// Stub for when `reqwest` is unavailable (non-`llm` builds).
#[cfg(not(feature = "llm"))]
pub async fn auth_login(_data_dir: &Path, _server_url: &str) -> Result<()> {
    eprintln!("auth login requires the 'llm' feature (includes reqwest).");
    eprintln!("Build with `cargo build` (default features) to enable it.");
    std::process::exit(1);
}

// ──────────────────────────────────────────────────────────────────────────────
// auth status
// ──────────────────────────────────────────────────────────────────────────────

/// Show current authentication state: local identity and cached server tokens.
pub async fn auth_status(data_dir: &Path) -> Result<()> {
    let ks_path = keystore::keystore_path(data_dir);
    let has_identity = keystore::exists(&ks_path);

    if has_identity {
        println!(
            "Identity: {}",
            "found".if_supports_color(Stream::Stdout, |s| s.green())
        );
        println!(
            "  Keystore: {}",
            ks_path
                .display()
                .if_supports_color(Stream::Stdout, |s| s.dimmed())
        );
    } else {
        println!(
            "Identity: {} — run `veclayer identity init` to create one",
            "none".if_supports_color(Stream::Stdout, |s| s.yellow())
        );
    }

    let cached = list_cached(data_dir)?;
    if cached.is_empty() {
        println!("Cached tokens: none");
    } else {
        println!("Cached tokens:");
        let now = unix_now();
        for (server, token) in &cached {
            let status = if token.expires_at > now {
                let remaining_secs = token.expires_at - now;
                format_expiry(remaining_secs)
                    .if_supports_color(Stream::Stdout, |s| s.green())
                    .to_string()
            } else {
                "expired"
                    .if_supports_color(Stream::Stdout, |s| s.red())
                    .to_string()
            };
            println!("  {server}");
            println!("    scope:   {}", token.scope);
            println!("    expires: {status}");
        }
    }

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

fn format_expiry(remaining_secs: u64) -> String {
    if remaining_secs >= 86400 {
        format!("expires in {}d", remaining_secs / 86400)
    } else if remaining_secs >= 3600 {
        format!("expires in {}h", remaining_secs / 3600)
    } else {
        format!("expires in {}m", remaining_secs / 60)
    }
}

fn resolve_passphrase() -> Result<String> {
    if let Ok(pass) = std::env::var("VECLAYER_PASSPHRASE") {
        return Ok(pass);
    }
    if io::stdin().is_terminal() {
        return prompt_passphrase("Enter passphrase: ");
    }
    Ok(String::new())
}

fn prompt_passphrase(prompt: &str) -> Result<String> {
    eprint!("{prompt}");
    use std::io::Write;
    io::stderr().flush()?;
    rpassword::read_password().map_err(crate::Error::Io)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;
    use crate::crypto::{keypair, keystore};

    fn temp_data_dir() -> (TempDir, PathBuf) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_path_buf();
        (dir, path)
    }

    fn init_identity(data_dir: &Path, passphrase: &str) {
        let signing_key = keypair::generate();
        let ks_path = keystore::keystore_path(data_dir);
        keystore::save(&signing_key, passphrase, &ks_path).unwrap();
    }

    // ── Duration parsing ──────────────────────────────────────────────────────

    #[test]
    fn test_parse_duration_hours() {
        assert_eq!(parse_duration_secs("1h").unwrap(), 3600);
        assert_eq!(parse_duration_secs("24h").unwrap(), 86400);
    }

    #[test]
    fn test_parse_duration_days() {
        assert_eq!(parse_duration_secs("1d").unwrap(), 86400);
        assert_eq!(parse_duration_secs("30d").unwrap(), 2_592_000);
    }

    #[test]
    fn test_parse_duration_seconds() {
        assert_eq!(parse_duration_secs("3600").unwrap(), 3600);
        assert_eq!(parse_duration_secs("0").unwrap(), 0);
    }

    #[test]
    fn test_parse_duration_invalid() {
        assert!(parse_duration_secs("abc").is_err());
        assert!(parse_duration_secs("xh").is_err());
        assert!(parse_duration_secs("yd").is_err());
    }

    // ── Token minting ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_auth_token_mints_valid_jwt() {
        let (_dir, data_dir) = temp_data_dir();
        let passphrase = "test-passphrase";
        init_identity(&data_dir, passphrase);

        // Redirect stdout so the JWT doesn't actually print.
        // We verify it by loading the key and using token::verify.
        let ks_path = keystore::keystore_path(&data_dir);
        let signing_key = keystore::load(passphrase, &ks_path).unwrap();
        let own_did = keypair::to_did(&signing_key.verifying_key());

        // Mint via the command function (with injected passphrase).
        // We can't easily capture stdout here, so we test the core logic directly.
        let expiry_secs = parse_duration_secs("1h").unwrap();
        let now = unix_now();
        let claims = Claims::new(
            own_did.clone(),
            own_did.clone(),
            crate::auth::capability::Capability::Read,
            now,
            now + expiry_secs,
        );
        let jwt = token::mint(&signing_key, &claims).unwrap();

        let recovered = token::verify(&jwt, &signing_key.verifying_key(), Some(&own_did)).unwrap();

        assert_eq!(recovered.sub, own_did);
        assert_eq!(recovered.aud, own_did);
        assert_eq!(recovered.cap, crate::auth::capability::Capability::Read);
        assert_eq!(recovered.exp, now + 3600);
    }

    #[tokio::test]
    async fn test_auth_token_no_identity_returns_error() {
        let (_dir, data_dir) = temp_data_dir();

        let err = auth_token_with_passphrase(&data_dir, "read", "1h", None, Some(""))
            .await
            .unwrap_err();

        assert!(
            err.to_string().contains("No identity found"),
            "expected 'No identity found', got: {err}"
        );
    }

    // ── Token cache ───────────────────────────────────────────────────────────

    #[test]
    fn test_token_cache_save_load() {
        let (_dir, data_dir) = temp_data_dir();

        let token = CachedToken {
            access_token: "tok_abc".to_string(),
            refresh_token: Some("ref_xyz".to_string()),
            expires_at: unix_now() + 3600,
            scope: "read write".to_string(),
        };

        save_token(&data_dir, "https://example.com", &token).unwrap();

        let loaded = load_token(&data_dir, "https://example.com")
            .unwrap()
            .expect("token should be present");

        assert_eq!(loaded.access_token, "tok_abc");
        assert_eq!(loaded.refresh_token.as_deref(), Some("ref_xyz"));
        assert_eq!(loaded.scope, "read write");
    }

    #[test]
    fn test_token_cache_file_has_restricted_permissions() {
        let (_dir, data_dir) = temp_data_dir();

        let token = CachedToken {
            access_token: "tok_secret".to_string(),
            refresh_token: None,
            expires_at: unix_now() + 3600,
            scope: "read".to_string(),
        };

        save_token(&data_dir, "https://example.com", &token).unwrap();

        let path = cache_path(&data_dir);
        assert!(path.exists(), "token cache file must exist");

        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let meta = std::fs::metadata(&path).unwrap();
            let mode = meta.mode() & 0o777;
            assert_eq!(
                mode, 0o600,
                "token cache must be owner-only (0o600), got 0o{mode:o}"
            );
        }
    }

    #[test]
    fn test_token_cache_load_missing_returns_none() {
        let (_dir, data_dir) = temp_data_dir();
        let result = load_token(&data_dir, "https://missing.example.com").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_token_cache_list_cached() {
        let (_dir, data_dir) = temp_data_dir();

        for i in 0..3 {
            let token = CachedToken {
                access_token: format!("tok_{i}"),
                refresh_token: None,
                expires_at: unix_now() + 3600,
                scope: "read".to_string(),
            };
            save_token(&data_dir, &format!("https://server{i}.example.com"), &token).unwrap();
        }

        let cached = list_cached(&data_dir).unwrap();
        assert_eq!(cached.len(), 3);
    }

    // ── auth status ───────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_auth_status_with_no_identity() {
        let (_dir, data_dir) = temp_data_dir();
        // Should complete without error and print a helpful message.
        auth_status(&data_dir).await.unwrap();
    }

    #[tokio::test]
    async fn test_auth_status_with_identity() {
        let (_dir, data_dir) = temp_data_dir();
        init_identity(&data_dir, "pass");
        auth_status(&data_dir).await.unwrap();
    }
}
