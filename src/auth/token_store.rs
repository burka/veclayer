//! Persistent OAuth token store (file-backed JSON).
//!
//! Stores registered OAuth clients, short-lived authorization codes, and
//! long-lived refresh token records. The file is written after every mutation
//! and is protected with 0o600 permissions on Unix (contains token hashes).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::capability::Capability;
use crate::util::{set_file_mode_600, unix_now};
use crate::{Error, Result};

// ─── File name ────────────────────────────────────────────────────────────────

const STORE_FILE: &str = "oauth_store.json";

/// How long (seconds) authorization codes remain valid.
const CODE_TTL_SECS: u64 = 600; // 10 minutes

// ─── Data structures ──────────────────────────────────────────────────────────

/// Registered OAuth client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredClient {
    pub client_id: String,
    pub client_name: String,
    pub redirect_uris: Vec<String>,
    pub created_at: u64,
}

/// Short-lived authorization code (for OAuth code exchange).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthCode {
    pub code: String,
    pub client_id: String,
    pub did: String,
    pub capability: Capability,
    pub redirect_uri: String,
    pub code_challenge: String,
    pub code_challenge_method: String, // always "S256"
    pub expires_at: u64,
    pub used: bool,
}

/// Long-lived refresh token record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefreshRecord {
    /// SHA-256 hash of the actual token (never store raw).
    pub token_hash: String,
    pub client_id: String,
    pub did: String,
    pub capability: Capability,
    pub expires_at: u64,
    pub revoked: bool,
}

// ─── Serializable envelope ────────────────────────────────────────────────────

/// On-disk representation — wraps all three maps for a single JSON object.
#[derive(Serialize, Deserialize, Default)]
struct StoreData {
    clients: HashMap<String, RegisteredClient>,
    codes: HashMap<String, AuthCode>,
    refresh_tokens: HashMap<String, RefreshRecord>,
}

// ─── TokenStore ───────────────────────────────────────────────────────────────

/// Persistent OAuth token store (file-backed JSON).
pub struct TokenStore {
    path: PathBuf,
    clients: HashMap<String, RegisteredClient>,
    codes: HashMap<String, AuthCode>,
    refresh_tokens: HashMap<String, RefreshRecord>,
}

impl TokenStore {
    /// Open or create a token store at `{data_dir}/oauth_store.json`.
    pub fn open(data_dir: &Path) -> Result<Self> {
        let path = data_dir.join(STORE_FILE);

        let data = if path.exists() {
            let raw = std::fs::read_to_string(&path)?;
            serde_json::from_str::<StoreData>(&raw)?
        } else {
            StoreData::default()
        };

        Ok(Self {
            path,
            clients: data.clients,
            codes: data.codes,
            refresh_tokens: data.refresh_tokens,
        })
    }

    /// Persist current state to disk.
    fn save(&self) -> Result<()> {
        let data = StoreData {
            clients: self.clients.clone(),
            codes: self.codes.clone(),
            refresh_tokens: self.refresh_tokens.clone(),
        };

        let json = serde_json::to_string_pretty(&data)?;

        // Write atomically via a temporary sibling file, then rename.
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, json)?;
        set_file_mode_600(&tmp)?;
        std::fs::rename(&tmp, &self.path)?;

        Ok(())
    }

    // ─── Client registry ──────────────────────────────────────────────────────

    /// Return the number of registered clients.
    pub fn client_count(&self) -> usize {
        self.clients.len()
    }

    /// Register a new OAuth client and persist.
    pub fn register_client(&mut self, name: &str, redirect_uris: Vec<String>) -> RegisteredClient {
        let client = RegisteredClient {
            client_id: Uuid::new_v4().to_string(),
            client_name: name.to_owned(),
            redirect_uris,
            created_at: unix_now(),
        };
        self.clients
            .insert(client.client_id.clone(), client.clone());
        // Best-effort persist; callers that need hard durability should handle
        // errors at a higher level.
        let _ = self.save();
        client
    }

    /// Look up a client by its ID.
    pub fn get_client(&self, client_id: &str) -> Option<&RegisteredClient> {
        self.clients.get(client_id)
    }

    // ─── Authorization codes ──────────────────────────────────────────────────

    /// Create a short-lived authorization code and persist.
    ///
    /// Returns the raw code string (must be delivered to the client).
    pub fn create_code(
        &mut self,
        client_id: &str,
        did: &str,
        capability: Capability,
        redirect_uri: &str,
        code_challenge: &str,
    ) -> String {
        let code = Uuid::new_v4().to_string();
        let auth_code = AuthCode {
            code: code.clone(),
            client_id: client_id.to_owned(),
            did: did.to_owned(),
            capability,
            redirect_uri: redirect_uri.to_owned(),
            code_challenge: code_challenge.to_owned(),
            code_challenge_method: "S256".to_owned(),
            expires_at: unix_now() + CODE_TTL_SECS,
            used: false,
        };
        self.codes.insert(code.clone(), auth_code);
        let _ = self.save();
        code
    }

    /// Consume an authorization code after PKCE verification.
    ///
    /// Marks the code as used (prevents replay) and persists.  Returns the
    /// consumed [`AuthCode`] so the caller can issue tokens.
    ///
    /// Errors:
    /// - `NotFound` — unknown code
    /// - `InvalidOperation` — expired, already used, or PKCE mismatch
    pub fn consume_code(&mut self, code: &str, code_verifier: &str) -> Result<AuthCode> {
        let record = self
            .codes
            .get_mut(code)
            .ok_or_else(|| Error::not_found(format!("authorization code not found: {code}")))?;

        if record.used {
            return Err(Error::InvalidOperation(
                "authorization code already used".to_owned(),
            ));
        }

        if unix_now() > record.expires_at {
            return Err(Error::InvalidOperation(
                "authorization code expired".to_owned(),
            ));
        }

        // RFC 7636 §4.1: code_verifier must be 43..=128 characters.
        if !(43..=128).contains(&code_verifier.len()) {
            return Err(Error::InvalidOperation(
                "code_verifier must be 43-128 characters (RFC 7636 §4.1)".to_owned(),
            ));
        }

        if !verify_pkce(code_verifier, &record.code_challenge) {
            return Err(Error::InvalidOperation(
                "PKCE verification failed".to_owned(),
            ));
        }

        record.used = true;
        let consumed = record.clone();
        self.save()?;
        Ok(consumed)
    }

    // ─── Refresh tokens ───────────────────────────────────────────────────────

    /// Store a refresh token (only the SHA-256 hash is persisted).
    pub fn store_refresh(
        &mut self,
        token: &str,
        client_id: &str,
        did: &str,
        capability: Capability,
        expires_at: u64,
    ) {
        let hash = sha256_hex(token);
        let record = RefreshRecord {
            token_hash: hash.clone(),
            client_id: client_id.to_owned(),
            did: did.to_owned(),
            capability,
            expires_at,
            revoked: false,
        };
        self.refresh_tokens.insert(hash, record);
        let _ = self.save();
    }

    /// Validate a refresh token (test-only; production uses `validate_and_revoke_refresh`).
    #[cfg(test)]
    fn validate_refresh(&self, token: &str) -> Result<&RefreshRecord> {
        let hash = sha256_hex(token);
        let record = self
            .refresh_tokens
            .get(&hash)
            .ok_or_else(|| Error::not_found("refresh token not found"))?;

        if record.revoked {
            return Err(Error::InvalidOperation("refresh token revoked".to_owned()));
        }

        if unix_now() > record.expires_at {
            return Err(Error::InvalidOperation("refresh token expired".to_owned()));
        }

        Ok(record)
    }

    /// Revoke a refresh token and persist (test-only; production uses `validate_and_revoke_refresh`).
    #[cfg(test)]
    fn revoke_refresh(&mut self, token: &str) {
        let hash = sha256_hex(token);
        if let Some(record) = self.refresh_tokens.get_mut(&hash) {
            record.revoked = true;
        }
        let _ = self.save();
    }

    /// Validate and atomically revoke a refresh token in a single call.
    ///
    /// Combines validation and revocation to prevent TOCTOU races: the token
    /// cannot be used by a concurrent request between the two operations.
    ///
    /// Returns `(client_id, did, capability)` on success.
    ///
    /// Errors:
    /// - `NotFound` — unknown token hash
    /// - `InvalidOperation` — token revoked or expired
    pub fn validate_and_revoke_refresh(
        &mut self,
        token: &str,
    ) -> Result<(String, String, Capability)> {
        let hash = sha256_hex(token);
        let record = self
            .refresh_tokens
            .get_mut(&hash)
            .ok_or_else(|| Error::not_found("refresh token not found"))?;

        if record.revoked {
            return Err(Error::InvalidOperation("refresh token revoked".to_owned()));
        }

        if unix_now() > record.expires_at {
            return Err(Error::InvalidOperation("refresh token expired".to_owned()));
        }

        let result = (
            record.client_id.clone(),
            record.did.clone(),
            record.capability,
        );
        record.revoked = true;
        self.save()?;
        Ok(result)
    }

    // ─── Cleanup ──────────────────────────────────────────────────────────────

    /// Remove expired or used authorization codes and expired/revoked refresh
    /// tokens.  Call periodically to keep the file small.
    pub fn purge_expired(&mut self) {
        let now = unix_now();
        self.codes.retain(|_, c| !c.used && now <= c.expires_at);
        self.refresh_tokens
            .retain(|_, r| !r.revoked && now <= r.expires_at);
        let _ = self.save();
    }
}

// ─── PKCE ─────────────────────────────────────────────────────────────────────

/// Verify PKCE S256: `BASE64URL(SHA-256(code_verifier)) == code_challenge`.
fn verify_pkce(code_verifier: &str, code_challenge: &str) -> bool {
    let hash = Sha256::digest(code_verifier.as_bytes());
    let computed = URL_SAFE_NO_PAD.encode(hash);
    computed == code_challenge
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn sha256_hex(input: &str) -> String {
    let hash = Sha256::digest(input.as_bytes());
    format!("{hash:x}")
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn tmp_store() -> (TempDir, TokenStore) {
        let dir = TempDir::new().expect("tempdir");
        let store = TokenStore::open(dir.path()).expect("open");
        (dir, store)
    }

    fn pkce_pair() -> (String, String) {
        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk".to_owned();
        let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
        (verifier, challenge)
    }

    // ─── Client registry ──────────────────────────────────────────────────────

    #[test]
    fn test_register_and_get_client() {
        let (_dir, mut store) = tmp_store();

        let client = store.register_client("Test App", vec!["https://example.com/cb".to_owned()]);

        assert!(!client.client_id.is_empty());
        assert_eq!(client.client_name, "Test App");
        assert_eq!(client.redirect_uris, ["https://example.com/cb"]);

        let fetched = store.get_client(&client.client_id).expect("get");
        assert_eq!(fetched.client_id, client.client_id);
        assert_eq!(fetched.client_name, "Test App");
    }

    // ─── Authorization codes ──────────────────────────────────────────────────

    #[test]
    fn test_create_and_consume_code() {
        let (_dir, mut store) = tmp_store();
        let client = store.register_client("App", vec![]);
        let (verifier, challenge) = pkce_pair();

        let code = store.create_code(
            &client.client_id,
            "did:key:zAlice",
            Capability::Read,
            "https://example.com/cb",
            &challenge,
        );

        let consumed = store.consume_code(&code, &verifier).expect("consume");
        assert_eq!(consumed.code, code);
        assert_eq!(consumed.did, "did:key:zAlice");
        assert_eq!(consumed.capability, Capability::Read);
        assert!(consumed.used);
    }

    #[test]
    fn test_code_reuse_rejected() {
        let (_dir, mut store) = tmp_store();
        let client = store.register_client("App", vec![]);
        let (verifier, challenge) = pkce_pair();

        let code = store.create_code(
            &client.client_id,
            "did:key:zAlice",
            Capability::Read,
            "https://example.com/cb",
            &challenge,
        );

        store.consume_code(&code, &verifier).expect("first consume");
        let err = store.consume_code(&code, &verifier).unwrap_err();
        assert!(
            err.to_string().contains("already used"),
            "expected 'already used' error, got: {err}"
        );
    }

    #[test]
    fn test_code_expired_rejected() {
        let (_dir, mut store) = tmp_store();
        let client = store.register_client("App", vec![]);
        let (verifier, challenge) = pkce_pair();

        let code = store.create_code(
            &client.client_id,
            "did:key:zAlice",
            Capability::Read,
            "https://example.com/cb",
            &challenge,
        );

        // Force-expire the code.
        let record = store.codes.get_mut(&code).unwrap();
        record.expires_at = 0;

        let err = store.consume_code(&code, &verifier).unwrap_err();
        assert!(
            err.to_string().contains("expired"),
            "expected 'expired' error, got: {err}"
        );
    }

    #[test]
    fn test_pkce_wrong_verifier_rejected() {
        let (_dir, mut store) = tmp_store();
        let client = store.register_client("App", vec![]);
        let (_verifier, challenge) = pkce_pair();

        let code = store.create_code(
            &client.client_id,
            "did:key:zAlice",
            Capability::Read,
            "https://example.com/cb",
            &challenge,
        );

        // 43 chars (minimum RFC 7636 length) but wrong content.
        let wrong_verifier = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let err = store.consume_code(&code, wrong_verifier).unwrap_err();
        assert!(
            err.to_string().contains("PKCE"),
            "expected PKCE error, got: {err}"
        );
    }

    // ─── Refresh tokens ───────────────────────────────────────────────────────

    #[test]
    fn test_refresh_store_validate_revoke() {
        let (_dir, mut store) = tmp_store();

        let raw_token = "super-secret-refresh-token";
        let expires_at = unix_now() + 86_400;

        store.store_refresh(
            raw_token,
            "client-abc",
            "did:key:zAlice",
            Capability::Write,
            expires_at,
        );

        // Validate succeeds.
        let record = store.validate_refresh(raw_token).expect("valid");
        assert_eq!(record.did, "did:key:zAlice");
        assert_eq!(record.capability, Capability::Write);
        assert!(!record.revoked);
        // Raw token must not be stored.
        assert_ne!(record.token_hash, raw_token);

        // Revoke and re-validate.
        store.revoke_refresh(raw_token);
        let err = store.validate_refresh(raw_token).unwrap_err();
        assert!(
            err.to_string().contains("revoked"),
            "expected 'revoked' error, got: {err}"
        );
    }

    // ─── validate_and_revoke_refresh (atomic) ─────────────────────────────────

    #[test]
    fn test_validate_and_revoke_refresh_atomic() {
        let (_dir, mut store) = tmp_store();

        let raw_token = "atomic-refresh-token";
        let expires_at = unix_now() + 86_400;
        store.store_refresh(
            raw_token,
            "client-1",
            "did:key:zAlice",
            Capability::Write,
            expires_at,
        );

        // Atomic validate + revoke should return the record data.
        let (client_id, did, cap) = store
            .validate_and_revoke_refresh(raw_token)
            .expect("should succeed");
        assert_eq!(client_id, "client-1");
        assert_eq!(did, "did:key:zAlice");
        assert_eq!(cap, Capability::Write);

        // Second call should fail because the token is now revoked.
        let err = store.validate_and_revoke_refresh(raw_token).unwrap_err();
        assert!(
            err.to_string().contains("revoked"),
            "expected 'revoked', got: {err}"
        );
    }

    // ─── Purge expired ────────────────────────────────────────────────────────

    #[test]
    fn test_purge_expired() {
        let (_dir, mut store) = tmp_store();
        let client = store.register_client("App", vec![]);
        let (_verifier, challenge) = pkce_pair();

        // Create one live code and one used code.
        let live_code = store.create_code(
            &client.client_id,
            "did:key:zA",
            Capability::Read,
            "",
            &challenge,
        );
        let dead_code = store.create_code(
            &client.client_id,
            "did:key:zB",
            Capability::Read,
            "",
            &challenge,
        );
        store.codes.get_mut(&dead_code).unwrap().used = true;

        // Create one live refresh token and one expired one.
        let live_token = "live-token";
        let dead_token = "dead-token";
        store.store_refresh(live_token, "c", "d", Capability::Read, unix_now() + 3600);
        store.store_refresh(dead_token, "c", "d", Capability::Read, 1); // epoch

        store.purge_expired();

        assert!(store.codes.contains_key(&live_code), "live code removed");
        assert!(
            !store.codes.contains_key(&dead_code),
            "used code not removed"
        );

        let live_hash = sha256_hex(live_token);
        let dead_hash = sha256_hex(dead_token);
        assert!(
            store.refresh_tokens.contains_key(&live_hash),
            "live token removed"
        );
        assert!(
            !store.refresh_tokens.contains_key(&dead_hash),
            "expired token not removed"
        );
    }

    // ─── Persistence ─────────────────────────────────────────────────────────

    #[test]
    fn test_persistence() {
        let dir = TempDir::new().expect("tempdir");

        let client_id;
        let code;
        let (verifier, challenge) = pkce_pair();

        // Write data in a first store instance.
        {
            let mut store = TokenStore::open(dir.path()).expect("open 1");
            let client = store.register_client("Persistent App", vec!["https://cb".to_owned()]);
            client_id = client.client_id.clone();

            code = store.create_code(
                &client_id,
                "did:key:zAlice",
                Capability::Admin,
                "https://cb",
                &challenge,
            );

            store.store_refresh(
                "my-refresh-token",
                &client_id,
                "did:key:zAlice",
                Capability::Write,
                unix_now() + 3600,
            );
        }

        // Reopen from disk.
        let mut store2 = TokenStore::open(dir.path()).expect("open 2");

        let client = store2.get_client(&client_id).expect("client persisted");
        assert_eq!(client.client_name, "Persistent App");

        let consumed = store2
            .consume_code(&code, &verifier)
            .expect("code persisted");
        assert_eq!(consumed.capability, Capability::Admin);

        let record = store2
            .validate_refresh("my-refresh-token")
            .expect("refresh persisted");
        assert_eq!(record.capability, Capability::Write);

        // Confirm the store file exists with restricted permissions.
        let path = dir.path().join("oauth_store.json");
        assert!(path.exists(), "store file missing");

        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let meta = std::fs::metadata(&path).unwrap();
            let mode = meta.mode() & 0o777;
            assert_eq!(mode, 0o600, "expected 0o600, got 0o{mode:o}");
        }
    }
}
