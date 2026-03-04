//! Encrypted keystore — persists an Ed25519 signing key on disk.
//!
//! # File format
//!
//! The keystore is a JSON file at `{data_dir}/identity.key`:
//!
//! ```json
//! {
//!   "version": 1,
//!   "did": "did:key:z6Mk...",
//!   "salt": "<base64>",
//!   "nonce": "<base64>",
//!   "ciphertext": "<base64>"
//! }
//! ```
//!
//! # Encryption scheme
//!
//! Passphrase → Argon2id (salt, 3 iterations, 64 MiB, 1 parallelism) → 256-bit key
//! → AES-256-GCM (nonce, signing_key_bytes)

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use argon2::{Algorithm, Argon2, Params, Version};
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine as _;
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use zeroize::Zeroizing;

use super::keypair;
use super::CryptoError;

/// Keystore file name within the data directory.
const KEYSTORE_FILENAME: &str = "identity.key";

/// AES-256-GCM nonce length in bytes.
const NONCE_LEN: usize = 12;

/// Argon2id salt length in bytes.
const SALT_LEN: usize = 16;

/// Argon2id parameters: 3 iterations, 64 MiB memory, 1 parallelism thread.
const ARGON2_T_COST: u32 = 3;
const ARGON2_M_COST: u32 = 64 * 1024; // 64 MiB in KiB
const ARGON2_P_COST: u32 = 1;

/// On-disk JSON envelope for the encrypted signing key.
#[derive(Serialize, Deserialize)]
struct KeystoreEnvelope {
    version: u32,
    did: String,
    salt: String,
    nonce: String,
    ciphertext: String,
}

/// Returns the canonical path for the keystore file.
pub fn keystore_path(data_dir: &Path) -> PathBuf {
    data_dir.join(KEYSTORE_FILENAME)
}

/// Returns `true` if a keystore file exists at the given path.
pub fn exists(path: &Path) -> bool {
    path.exists()
}

/// Persist an encrypted signing key to disk.
///
/// The file is created with mode `0o600` on Unix (owner read/write only),
/// set atomically on open so there is no window of world-readable exposure.
///
/// # Errors
///
/// Returns [`CryptoError`] on I/O failure, Argon2 parameter errors, or
/// AES-GCM encryption failure.
pub fn save(signing_key: &SigningKey, passphrase: &str, path: &Path) -> Result<(), CryptoError> {
    let mut salt = [0u8; SALT_LEN];
    let mut nonce_bytes = [0u8; NONCE_LEN];
    OsRng.fill_bytes(&mut salt);
    OsRng.fill_bytes(&mut nonce_bytes);

    let derived_key = Zeroizing::new(derive_key(passphrase, &salt)?);
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&*derived_key));
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher
        .encrypt(nonce, signing_key.as_bytes().as_ref())
        .map_err(|_| CryptoError::Keystore("AES-GCM encryption failed".to_string()))?;

    let verifying_key = signing_key.verifying_key();
    let did = keypair::to_did(&verifying_key);

    let envelope = KeystoreEnvelope {
        version: 1,
        did,
        salt: B64.encode(salt),
        nonce: B64.encode(nonce_bytes),
        ciphertext: B64.encode(ciphertext),
    };

    let json = serde_json::to_string_pretty(&envelope)
        .map_err(|e| CryptoError::Keystore(format!("JSON serialization failed: {e}")))?;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)?;
        file.write_all(json.as_bytes())?;
    }
    #[cfg(not(unix))]
    {
        fs::write(path, json)?;
    }

    Ok(())
}

/// Load and decrypt a signing key from the keystore file.
///
/// Verifies that the DID embedded in the keystore matches the recovered key
/// as an integrity check.
///
/// # Errors
///
/// Returns [`CryptoError::DecryptionFailed`] for wrong passphrase,
/// [`CryptoError::DidMismatch`] if the integrity check fails, and
/// [`CryptoError::Keystore`] or [`CryptoError::Io`] for other failures.
pub fn load(passphrase: &str, path: &Path) -> Result<SigningKey, CryptoError> {
    let json = fs::read_to_string(path)?;

    let envelope: KeystoreEnvelope = serde_json::from_str(&json)
        .map_err(|e| CryptoError::Keystore(format!("JSON parse failed: {e}")))?;

    if envelope.version != 1 {
        return Err(CryptoError::Keystore(format!(
            "unsupported keystore version {} (expected 1)",
            envelope.version
        )));
    }

    let salt = B64
        .decode(&envelope.salt)
        .map_err(|e| CryptoError::Keystore(format!("base64 decode (salt): {e}")))?;

    let nonce_bytes = B64
        .decode(&envelope.nonce)
        .map_err(|e| CryptoError::Keystore(format!("base64 decode (nonce): {e}")))?;

    let ciphertext = B64
        .decode(&envelope.ciphertext)
        .map_err(|e| CryptoError::Keystore(format!("base64 decode (ciphertext): {e}")))?;

    let derived_key = Zeroizing::new(derive_key(passphrase, &salt)?);
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&*derived_key));
    let nonce = Nonce::from_slice(&nonce_bytes);

    let plaintext = Zeroizing::new(
        cipher
            .decrypt(nonce, ciphertext.as_ref())
            .map_err(|_| CryptoError::DecryptionFailed)?,
    );

    let key_bytes = Zeroizing::new(
        <[u8; 32]>::try_from(plaintext.as_slice())
            .map_err(|_| CryptoError::Keystore("invalid key length".into()))?,
    );

    let signing_key = SigningKey::from_bytes(&key_bytes);

    let expected_did = keypair::to_did(&signing_key.verifying_key());
    if expected_did != envelope.did {
        return Err(CryptoError::DidMismatch);
    }

    Ok(signing_key)
}

/// Derive a 256-bit key from a passphrase and salt using Argon2id.
fn derive_key(passphrase: &str, salt: &[u8]) -> Result<[u8; 32], CryptoError> {
    let params = Params::new(ARGON2_M_COST, ARGON2_T_COST, ARGON2_P_COST, Some(32))
        .map_err(|e| CryptoError::Keystore(format!("Argon2 params error: {e}")))?;

    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);

    let mut output = [0u8; 32];
    argon2
        .hash_password_into(passphrase.as_bytes(), salt, &mut output)
        .map_err(|e| CryptoError::Keystore(format!("Argon2 KDF failed: {e}")))?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup() -> (TempDir, PathBuf) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("identity.key");
        (dir, path)
    }

    fn generate_signing_key() -> SigningKey {
        keypair::generate()
    }

    #[test]
    fn test_save_load_roundtrip() {
        let (_dir, path) = setup();
        let signing_key = generate_signing_key();
        let passphrase = "hunter2";

        save(&signing_key, passphrase, &path).expect("save must succeed");
        let loaded = load(passphrase, &path).expect("load must succeed");

        assert_eq!(signing_key.to_bytes(), loaded.to_bytes());
    }

    #[test]
    fn test_wrong_passphrase_fails() {
        let (_dir, path) = setup();
        let signing_key = generate_signing_key();

        save(&signing_key, "correct-passphrase", &path).unwrap();
        let err = load("wrong-passphrase", &path).unwrap_err();

        assert!(
            matches!(err, CryptoError::DecryptionFailed),
            "expected DecryptionFailed, got: {err:?}"
        );
    }

    #[test]
    fn test_keystore_file_format() {
        let (_dir, path) = setup();
        let signing_key = generate_signing_key();

        save(&signing_key, "passphrase", &path).unwrap();

        let raw = fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&raw).unwrap();

        assert_eq!(value["version"], 1);
        assert!(
            value["did"].as_str().unwrap().starts_with("did:key:z"),
            "DID must start with 'did:key:z'"
        );
        assert!(value["salt"].is_string(), "salt must be a string");
        assert!(value["nonce"].is_string(), "nonce must be a string");
        assert!(
            value["ciphertext"].is_string(),
            "ciphertext must be a string"
        );

        // Verify base64 fields are non-empty and decodable.
        B64.decode(value["salt"].as_str().unwrap()).unwrap();
        B64.decode(value["nonce"].as_str().unwrap()).unwrap();
        B64.decode(value["ciphertext"].as_str().unwrap()).unwrap();
    }

    #[test]
    fn test_exists_false_when_missing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nonexistent.key");
        assert!(!exists(&path));
    }

    #[test]
    fn test_exists_true_after_save() {
        let (_dir, path) = setup();
        let signing_key = generate_signing_key();
        save(&signing_key, "pass", &path).unwrap();
        assert!(exists(&path));
    }

    #[test]
    fn test_did_integrity_check() {
        let (_dir, path) = setup();
        let signing_key = generate_signing_key();

        save(&signing_key, "pass", &path).unwrap();

        // Tamper with the DID field.
        let raw = fs::read_to_string(&path).unwrap();
        let mut value: serde_json::Value = serde_json::from_str(&raw).unwrap();
        value["did"] = serde_json::Value::String("did:key:z6MkTampered".to_string());
        fs::write(&path, serde_json::to_string_pretty(&value).unwrap()).unwrap();

        let err = load("pass", &path).unwrap_err();
        assert!(
            matches!(err, CryptoError::DidMismatch),
            "expected DidMismatch, got: {err:?}"
        );
    }

    #[test]
    fn test_keystore_path_returns_correct_filename() {
        let dir = TempDir::new().unwrap();
        let path = keystore_path(dir.path());
        assert_eq!(path.file_name().unwrap(), "identity.key");
        assert_eq!(path.parent().unwrap(), dir.path());
    }

    #[test]
    fn test_unsupported_version_rejected() {
        let (_dir, path) = setup();
        let signing_key = generate_signing_key();

        save(&signing_key, "pass", &path).unwrap();

        // Bump the version field to a value we don't support.
        let raw = fs::read_to_string(&path).unwrap();
        let mut value: serde_json::Value = serde_json::from_str(&raw).unwrap();
        value["version"] = serde_json::Value::Number(99.into());
        fs::write(&path, serde_json::to_string_pretty(&value).unwrap()).unwrap();

        let err = load("pass", &path).unwrap_err();
        assert!(
            matches!(err, CryptoError::Keystore(_)),
            "expected Keystore error for unknown version, got: {err:?}"
        );
    }
}
