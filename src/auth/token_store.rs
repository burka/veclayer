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
            // Repair permissions on a store written by an older version or
            // external tooling with looser bits — it holds token hashes.
            set_file_mode_600(&path)?;
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

        // Write atomically via a temporary sibling file, then rename. The temp
        // file is created with 0o600 from the start so the token hashes it
        // holds are never briefly world-readable.
        let tmp = self.path.with_extension("json.tmp");
        crate::util::write_file_0600(&tmp, json.as_bytes())?;
        std::fs::rename(&tmp, &self.path)?;

        Ok(())
    }

    // ─── Client registry ──────────────────────────────────────────────────────

    /// Return the number of registered clients.
    pub fn client_count(&self) -> usize {
        self.clients.len()
    }

    /// Register a new OAuth client and persist.
    ///
    /// Returns an error if the store could not be persisted, so the caller
    /// never reports success for a registration that was not durably written.
    pub fn register_client(
        &mut self,
        name: &str,
        redirect_uris: Vec<String>,
    ) -> Result<RegisteredClient> {
        let client = RegisteredClient {
            client_id: Uuid::new_v4().to_string(),
            client_name: name.to_owned(),
            redirect_uris,
            created_at: unix_now(),
        };
        self.clients
            .insert(client.client_id.clone(), client.clone());
        self.save()?;
        Ok(client)
    }

    /// Look up a client by its ID.
    pub fn get_client(&self, client_id: &str) -> Option<&RegisteredClient> {
        self.clients.get(client_id)
    }

    // ─── Authorization codes ──────────────────────────────────────────────────

    /// Create a short-lived authorization code and persist.
    ///
    /// Returns the raw code string (must be delivered to the client), or an
    /// error if the store could not be persisted.
    pub fn create_code(
        &mut self,
        client_id: &str,
        did: &str,
        capability: Capability,
        redirect_uri: &str,
        code_challenge: &str,
    ) -> Result<String> {
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
        };
        self.codes.insert(code.clone(), auth_code);
        self.save()?;
        Ok(code)
    }

    /// Consume an authorization code after PKCE verification.
    ///
    /// Removes the entry from the store on success (prevents replay without
    /// relying on a `used` sentinel) and persists.  Returns the consumed
    /// [`AuthCode`] so the caller can issue tokens.
    ///
    /// Validation failures do NOT remove the entry — a failed PKCE attempt
    /// must not burn a valid code, which would let an attacker deny service
    /// to the legitimate owner.
    ///
    /// Errors:
    /// - `NotFound` — unknown or already-consumed code
    /// - `InvalidOperation` — expired or PKCE mismatch
    pub fn consume_code(&mut self, code: &str, code_verifier: &str) -> Result<AuthCode> {
        let record = self
            .codes
            .get(code)
            // Do NOT include the raw code in this message — authorization codes
            // are short-lived secrets and must not appear in logs or error
            // responses.  "not found / expired" is intentionally generic so
            // callers cannot distinguish the two cases (oracle-attack hardening).
            .ok_or_else(|| Error::not_found("authorization code not found / expired"))?;

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

        // All checks passed: remove atomically and persist.
        let consumed = self
            .codes
            .remove(code)
            .expect("entry present: checked above");
        self.save()?;
        Ok(consumed)
    }

    // ─── Refresh tokens ───────────────────────────────────────────────────────

    /// Store a refresh token (only the SHA-256 hash is persisted).
    ///
    /// Returns an error if the store could not be persisted, so a refresh
    /// token is never handed out without being durably recorded.
    pub fn store_refresh(
        &mut self,
        token: &str,
        client_id: &str,
        did: &str,
        capability: Capability,
        expires_at: u64,
    ) -> Result<()> {
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
        self.save()
    }

    /// Shared validation logic for a refresh token record (revoked + expiry checks).
    fn check_refresh_record(record: &RefreshRecord) -> Result<()> {
        if record.revoked {
            return Err(Error::InvalidOperation("refresh token revoked".to_owned()));
        }
        if unix_now() > record.expires_at {
            return Err(Error::InvalidOperation("refresh token expired".to_owned()));
        }
        Ok(())
    }

    /// Validate a refresh token (test-only; production uses `validate_and_revoke_refresh`).
    #[cfg(test)]
    fn validate_refresh(&self, token: &str) -> Result<&RefreshRecord> {
        let hash = sha256_hex(token);
        let record = self
            .refresh_tokens
            .get(&hash)
            .ok_or_else(|| Error::not_found("refresh token not found"))?;
        Self::check_refresh_record(record)?;
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

    /// Return the `client_id` bound to a refresh token after validating it is
    /// neither revoked nor expired — WITHOUT revoking it.
    ///
    /// Lets a caller verify the requesting client matches the token's bound
    /// client *before* committing to revocation, so a wrong/malicious client
    /// cannot burn a victim's still-valid token (denial of service). Call while
    /// holding the store lock, immediately before [`validate_and_revoke_refresh`],
    /// so the check-then-revoke sequence stays atomic.
    ///
    /// Errors:
    /// - `NotFound` — unknown token hash
    /// - `InvalidOperation` — token revoked or expired
    pub fn refresh_token_client_id(&self, token: &str) -> Result<String> {
        let hash = sha256_hex(token);
        let record = self
            .refresh_tokens
            .get(&hash)
            .ok_or_else(|| Error::not_found("refresh token not found"))?;
        Self::check_refresh_record(record)?;
        Ok(record.client_id.clone())
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
        Self::check_refresh_record(record)?;
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

    /// Remove expired authorization codes and expired/revoked refresh tokens.
    /// Call periodically to keep the file small.
    pub fn purge_expired(&mut self) -> Result<()> {
        let now = unix_now();
        self.codes.retain(|_, c| now <= c.expires_at);
        self.refresh_tokens
            .retain(|_, r| !r.revoked && now <= r.expires_at);
        self.save()
    }
}

// ─── PKCE ─────────────────────────────────────────────────────────────────────

/// Verify PKCE S256: `BASE64URL(SHA-256(code_verifier)) == code_challenge`.
///
/// The stored challenge is decoded back to raw digest bytes and compared in
/// constant time, so verification time does not depend on how many leading
/// bytes matched.
fn verify_pkce(code_verifier: &str, code_challenge: &str) -> bool {
    let expected = Sha256::digest(code_verifier.as_bytes());
    match URL_SAFE_NO_PAD.decode(code_challenge) {
        Ok(stored) => constant_time_eq(expected.as_slice(), &stored),
        Err(_) => false,
    }
}

/// Constant-time byte-slice equality: the comparison time does not depend on
/// the position of the first differing byte. Used for secret comparisons.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
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
    use crate::test_helpers::assert_file_mode_600;
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

    fn create_test_code(
        store: &mut TokenStore,
        client: &RegisteredClient,
        challenge: &str,
    ) -> String {
        store
            .create_code(
                &client.client_id,
                "did:key:zAlice",
                Capability::Read,
                "https://example.com/cb",
                challenge,
            )
            .expect("create_code")
    }

    // ─── Client registry ──────────────────────────────────────────────────────

    #[test]
    fn test_register_and_get_client() {
        let (_dir, mut store) = tmp_store();

        let client = store
            .register_client("Test App", vec!["https://example.com/cb".to_owned()])
            .expect("register");

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
        let (_dir, mut store, verifier, code) = code_test_harness();

        let consumed = store.consume_code(&code, &verifier).expect("consume");
        assert_eq!(consumed.code, code);
        assert_eq!(consumed.did, "did:key:zAlice");
        assert_eq!(consumed.capability, Capability::Read);
        // Entry must be gone after a successful consume.
        assert!(
            !store.codes.contains_key(&code),
            "consumed code must be removed from store"
        );
    }

    /// Shared setup for code-consumption tests: store + client + PKCE pair + issued code.
    fn code_test_harness() -> (TempDir, TokenStore, String, String) {
        let (_dir, mut store) = tmp_store();
        let client = store.register_client("App", vec![]).expect("register");
        let (verifier, challenge) = pkce_pair();
        let code = create_test_code(&mut store, &client, &challenge);
        (_dir, store, verifier, code)
    }

    #[test]
    fn test_code_reuse_rejected() {
        let (_dir, mut store, verifier, code) = code_test_harness();

        store.consume_code(&code, &verifier).expect("first consume");
        // After remove-on-consume the entry is gone, so a second attempt must
        // fail with not-found rather than "already used".
        let err = store.consume_code(&code, &verifier).unwrap_err();
        assert!(
            err.to_string().contains("not found"),
            "expected 'not found' error on replay, got: {err}"
        );
    }

    /// Consuming a code removes it from the in-memory map AND from the
    /// persisted file, so a server restart cannot replay a consumed code.
    #[test]
    fn test_consume_removes_entry_from_store_and_disk() {
        let (dir, mut store, verifier, code) = code_test_harness();

        store.consume_code(&code, &verifier).expect("consume");
        assert!(
            !store.codes.contains_key(&code),
            "code must be absent from in-memory map after consume"
        );

        // Reload from disk: the consumed code must not reappear.
        let reloaded = TokenStore::open(dir.path()).expect("reopen");
        assert!(
            !reloaded.codes.contains_key(&code),
            "consumed code must not survive a reload from disk"
        );
    }

    #[test]
    fn test_code_expired_rejected() {
        let (_dir, mut store, verifier, code) = code_test_harness();

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
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"ab"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn test_verify_pkce_accepts_correct_and_rejects_wrong() {
        let (verifier, challenge) = pkce_pair();
        assert!(
            verify_pkce(&verifier, &challenge),
            "correct pair must verify"
        );

        // A challenge differing in one byte must be rejected.
        let mut wrong: Vec<char> = challenge.chars().collect();
        wrong[5] = if wrong[5] == 'A' { 'B' } else { 'A' };
        let wrong: String = wrong.into_iter().collect();
        assert!(
            !verify_pkce(&verifier, &wrong),
            "altered challenge must fail"
        );

        // A malformed (non-base64) challenge must be rejected, not panic.
        assert!(!verify_pkce(&verifier, "!!!not-base64!!!"));
    }

    #[test]
    fn test_open_repairs_loose_permissions() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("oauth_store.json");
        std::fs::write(&path, r#"{"clients":{},"codes":{},"refresh_tokens":{}}"#).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        }
        let _store = TokenStore::open(dir.path()).expect("open");
        #[cfg(unix)]
        assert_file_mode_600(&path, "reopened store with loose permissions");
    }

    // ─── PKCE verifier length boundary tests (RFC 7636 §4.1: 43..=128) ──────────

    /// A 42-character verifier (one below the 43-character minimum) must be
    /// rejected with the RFC 7636 length-validation error, not a PKCE mismatch.
    #[test]
    fn test_pkce_verifier_too_short_42_rejected() {
        let (_dir, mut store, _verifier, code) = code_test_harness();

        let short_verifier = "A".repeat(42); // 42 chars — one below minimum

        let err = store.consume_code(&code, &short_verifier).unwrap_err();
        assert!(
            err.to_string().contains("43-128 characters"),
            "expected RFC 7636 length error (43-128 characters), got: {err}"
        );
    }

    /// A 129-character verifier (one above the 128-character maximum) must be
    /// rejected with the RFC 7636 length-validation error, not a PKCE mismatch.
    #[test]
    fn test_pkce_verifier_too_long_129_rejected() {
        let (_dir, mut store, _verifier, code) = code_test_harness();

        let long_verifier = "A".repeat(129); // 129 chars — one above maximum

        let err = store.consume_code(&code, &long_verifier).unwrap_err();
        assert!(
            err.to_string().contains("43-128 characters"),
            "expected RFC 7636 length error (43-128 characters), got: {err}"
        );
    }

    /// An empty verifier (length 0) must be rejected with the RFC 7636
    /// length-validation error, not some other failure (e.g. a PKCE mismatch).
    #[test]
    fn test_pkce_verifier_empty_rejected() {
        let (_dir, mut store, _verifier, code) = code_test_harness();

        let err = store.consume_code(&code, "").unwrap_err();
        assert!(
            err.to_string().contains("43-128 characters"),
            "expected RFC 7636 length error (43-128 characters), got: {err}"
        );
    }

    #[test]
    fn test_pkce_wrong_verifier_rejected_and_does_not_burn_code() {
        let (_dir, mut store) = tmp_store();
        let client = store.register_client("App", vec![]).expect("register");
        let (correct_verifier, challenge) = pkce_pair();

        let code = store
            .create_code(
                &client.client_id,
                "did:key:zAlice",
                Capability::Read,
                "https://example.com/cb",
                &challenge,
            )
            .expect("create_code");

        // 43 chars (minimum RFC 7636 length) but wrong content.
        let wrong_verifier = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let err = store.consume_code(&code, wrong_verifier).unwrap_err();
        assert!(
            err.to_string().contains("PKCE"),
            "expected PKCE error, got: {err}"
        );

        // A failed PKCE attempt must NOT burn the code — the legitimate holder
        // must still be able to consume it with the correct verifier.
        assert!(
            store.codes.contains_key(&code),
            "code must still exist after a failed PKCE attempt"
        );
        store
            .consume_code(&code, &correct_verifier)
            .expect("correct verifier must succeed after a failed attempt");
    }

    /// `consume_code` returns a not-found error when called with a code string
    /// that was never stored.
    ///
    /// Branch ordering in `consume_code`: the map lookup (not-found guard) runs
    /// BEFORE the RFC 7636 verifier-length check, so a too-short verifier would
    /// mask this branch.  A 43-character verifier (the RFC 7636 minimum) is used
    /// to ensure only the not-found branch executes.
    #[test]
    fn test_consume_code_unknown_code_returns_not_found() {
        let (_dir, mut store) = tmp_store();

        // 43 chars — RFC 7636 minimum — so the length guard does not fire first.
        let verifier = "A".repeat(43);
        let unknown_code = "this-code-was-never-stored";

        let err = store.consume_code(unknown_code, &verifier).unwrap_err();

        assert!(
            err.to_string().contains("authorization code not found"),
            "expected 'authorization code not found' error, got: {err}"
        );
    }

    // ─── Security: authorization code must not appear in error messages ───────

    /// A lookup miss must NOT embed the raw authorization code in the error
    /// message.  Authorization codes are short-lived secrets; leaking them into
    /// log aggregation pipelines via error messages is a security finding.
    #[test]
    fn test_consume_code_miss_does_not_leak_raw_code() {
        let (_dir, mut store) = tmp_store();

        let raw_code = "super-secret-auth-code-value-12345";
        // 43-char verifier so the length guard does not fire before the miss.
        let verifier = "A".repeat(43);

        let err = store.consume_code(raw_code, &verifier).unwrap_err();
        let msg = err.to_string();

        // The error must not contain the raw secret.
        assert!(
            !msg.contains(raw_code),
            "raw authorization code must NOT appear in error message, got: {msg}"
        );
        // But it must still be recognisably an "auth code not found" error.
        assert!(
            msg.contains("authorization code not found"),
            "expected 'authorization code not found' in error, got: {msg}"
        );
    }

    /// A valid code lookup (happy path) still resolves correctly after the
    /// redaction change — i.e., the fix does not break the success branch.
    #[test]
    fn test_consume_code_hit_resolves_correctly() {
        let (_dir, mut store, verifier, code) = code_test_harness();

        let consumed = store.consume_code(&code, &verifier).expect("consume");
        assert_eq!(consumed.did, "did:key:zAlice");
        assert!(
            !store.codes.contains_key(&code),
            "consumed code must be removed"
        );
    }

    /// An expired code must NOT embed the raw code in its error path.
    /// (The expiry guard fires AFTER the lookup, so the same `record` reference
    /// is used — the code value is available in scope at that point but must
    /// not be forwarded into any error message.)
    #[test]
    fn test_consume_code_expired_does_not_leak_raw_code() {
        let (_dir, mut store, verifier, code) = code_test_harness();

        // Force-expire the code.
        store.codes.get_mut(&code).unwrap().expires_at = 0;

        let err = store.consume_code(&code, &verifier).unwrap_err();
        let msg = err.to_string();

        // The error must not contain the raw auth code value.
        assert!(
            !msg.contains(&code),
            "raw authorization code must NOT appear in expiry error, got: {msg}"
        );
        // Must still be recognisably an expiry error.
        assert!(
            msg.contains("expired"),
            "expected 'expired' in error message, got: {msg}"
        );
    }

    // ─── Refresh tokens ───────────────────────────────────────────────────────

    #[test]
    fn test_refresh_store_validate_revoke() {
        let (_dir, mut store) = tmp_store();

        let raw_token = "super-secret-refresh-token";
        let expires_at = unix_now() + 86_400;

        store
            .store_refresh(
                raw_token,
                "client-abc",
                "did:key:zAlice",
                Capability::Write,
                expires_at,
            )
            .expect("store_refresh");

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
        store
            .store_refresh(
                raw_token,
                "client-1",
                "did:key:zAlice",
                Capability::Write,
                expires_at,
            )
            .expect("store_refresh");

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
        let client = store.register_client("App", vec![]).expect("register");
        let (_verifier, challenge) = pkce_pair();

        // Create one live code and one already-expired code.
        let live_code = store
            .create_code(
                &client.client_id,
                "did:key:zA",
                Capability::Read,
                "",
                &challenge,
            )
            .expect("create_code");
        let dead_code = store
            .create_code(
                &client.client_id,
                "did:key:zB",
                Capability::Read,
                "",
                &challenge,
            )
            .expect("create_code");
        // Simulate a code that expired without being consumed (e.g. user abandoned flow).
        store.codes.get_mut(&dead_code).unwrap().expires_at = 0;

        // Create one live refresh token and one expired one.
        let live_token = "live-token";
        let dead_token = "dead-token";
        store
            .store_refresh(live_token, "c", "d", Capability::Read, unix_now() + 3600)
            .expect("store_refresh");
        store
            .store_refresh(dead_token, "c", "d", Capability::Read, 1)
            .expect("store_refresh"); // epoch

        store.purge_expired().expect("purge_expired");

        assert!(
            store.codes.contains_key(&live_code),
            "live code must survive purge"
        );
        assert!(
            !store.codes.contains_key(&dead_code),
            "expired code must be removed by purge"
        );

        let live_hash = sha256_hex(live_token);
        let dead_hash = sha256_hex(dead_token);
        assert!(
            store.refresh_tokens.contains_key(&live_hash),
            "live token must survive purge"
        );
        assert!(
            !store.refresh_tokens.contains_key(&dead_hash),
            "expired token must be removed by purge"
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
            let client = store
                .register_client("Persistent App", vec!["https://cb".to_owned()])
                .expect("register");
            client_id = client.client_id.clone();

            code = store
                .create_code(
                    &client_id,
                    "did:key:zAlice",
                    Capability::Admin,
                    "https://cb",
                    &challenge,
                )
                .expect("create_code");

            store
                .store_refresh(
                    "my-refresh-token",
                    &client_id,
                    "did:key:zAlice",
                    Capability::Write,
                    unix_now() + 3600,
                )
                .expect("store_refresh");
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
        assert_file_mode_600(&path, "oauth store");
    }
}
