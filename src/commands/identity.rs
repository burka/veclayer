//! Identity management commands — init and show.

use std::io::{self, IsTerminal};
use std::path::Path;

use owo_colors::{OwoColorize, Stream};
use zeroize::Zeroizing;

use crate::commands::auth::prompt_passphrase;
use crate::crypto::{keypair, keystore};
use crate::Result;

// ──────────────────────────────────────────────────────────────────────────────
// Public command functions
// ──────────────────────────────────────────────────────────────────────────────

/// Initialize a new identity: generate an Ed25519 keypair and save it to the
/// encrypted keystore.
///
/// # Errors
///
/// Returns an error if the keystore already exists and `force` is `false`,
/// if passphrase confirmation fails, or if the keystore cannot be written.
pub async fn identity_init(data_dir: &Path, force: bool) -> Result<()> {
    identity_init_with_passphrase(data_dir, force, None).await
}

/// Show the current identity: DID, public key, and keystore path.
///
/// # Errors
///
/// Returns an error if no keystore exists or if decryption fails.
pub async fn identity_show(data_dir: &Path) -> Result<()> {
    identity_show_with_passphrase(data_dir, None).await
}

// ──────────────────────────────────────────────────────────────────────────────
// Passphrase-injectable variants (used directly in tests to avoid env-var races)
// ──────────────────────────────────────────────────────────────────────────────

pub(crate) async fn identity_init_with_passphrase(
    data_dir: &Path,
    force: bool,
    passphrase: Option<&str>,
) -> Result<()> {
    let path = keystore::keystore_path(data_dir);

    if keystore::exists(&path) && !force {
        return Err(crate::Error::InvalidOperation(format!(
            "Identity already exists at {}. Use --force to overwrite.",
            path.display()
        )));
    }

    let passphrase: Zeroizing<String> = match passphrase {
        Some(p) => Zeroizing::new(p.to_string()),
        None => resolve_passphrase_for_write()?,
    };
    let signing_key = keypair::generate();
    keystore::save(&signing_key, &passphrase, &path)?;

    let did = keypair::to_did(&signing_key.verifying_key());
    println!(
        "Identity initialized.\n  DID:      {}\n  Keystore: {}",
        did.if_supports_color(Stream::Stdout, |s| s.green()),
        path.display()
            .if_supports_color(Stream::Stdout, |s| s.dimmed()),
    );

    Ok(())
}

pub(crate) async fn identity_show_with_passphrase(
    data_dir: &Path,
    passphrase: Option<&str>,
) -> Result<()> {
    let signing_key =
        crate::commands::auth::load_signing_key_with_passphrase(data_dir, passphrase)?;
    let verifying_key = signing_key.verifying_key();
    let did = keypair::to_did(&verifying_key);
    let path = keystore::keystore_path(data_dir);
    let pubkey_hex: String = verifying_key
        .as_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();

    println!(
        "DID:        {}\nPublic key: {}\nKeystore:   {}",
        did.if_supports_color(Stream::Stdout, |s| s.green()),
        pubkey_hex,
        path.display()
            .if_supports_color(Stream::Stdout, |s| s.dimmed()),
    );

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Passphrase resolution
// ──────────────────────────────────────────────────────────────────────────────

/// Resolve passphrase for writing (init): confirms twice when prompting.
///
/// Returns a `Zeroizing<String>` so the passphrase bytes are wiped from memory
/// when the value is dropped.
fn resolve_passphrase_for_write() -> Result<Zeroizing<String>> {
    if let Ok(pass) = std::env::var("VECLAYER_PASSPHRASE") {
        return Ok(Zeroizing::new(pass));
    }

    if io::stdin().is_terminal() {
        let first = prompt_passphrase("Enter passphrase: ")?;
        let second = prompt_passphrase("Confirm passphrase: ")?;
        if first != second {
            return Err(crate::Error::InvalidOperation(
                "Passphrases do not match.".to_string(),
            ));
        }
        return Ok(first);
    }

    eprintln!("Warning: stdin is not a terminal — using empty passphrase for identity keystore.");
    Ok(Zeroizing::new(String::new()))
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;

    fn temp_data_dir() -> (TempDir, PathBuf) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_path_buf();
        (dir, path)
    }

    #[tokio::test]
    async fn test_identity_init_creates_keystore() {
        let (_dir, data_dir) = temp_data_dir();

        identity_init_with_passphrase(&data_dir, false, Some("test-passphrase"))
            .await
            .unwrap();

        let keystore_path = keystore::keystore_path(&data_dir);
        assert!(
            keystore::exists(&keystore_path),
            "keystore file should exist after init"
        );
    }

    #[tokio::test]
    async fn test_identity_init_refuses_overwrite_without_force() {
        let (_dir, data_dir) = temp_data_dir();

        identity_init_with_passphrase(&data_dir, false, Some("test-passphrase"))
            .await
            .unwrap();
        let err = identity_init_with_passphrase(&data_dir, false, Some("test-passphrase"))
            .await
            .unwrap_err();

        assert!(
            err.to_string().contains("already exists"),
            "expected 'already exists' error, got: {err}"
        );
    }

    #[tokio::test]
    async fn test_identity_init_force_overwrites() {
        let (_dir, data_dir) = temp_data_dir();

        // First init.
        identity_init_with_passphrase(&data_dir, false, Some("pass1"))
            .await
            .unwrap();
        let path = keystore::keystore_path(&data_dir);
        let key_first = keystore::load("pass1", &path).unwrap();
        let did_first = keypair::to_did(&key_first.verifying_key());

        // Force-overwrite with a different passphrase.
        identity_init_with_passphrase(&data_dir, true, Some("pass2"))
            .await
            .unwrap();
        let key_second = keystore::load("pass2", &path).unwrap();
        let did_second = keypair::to_did(&key_second.verifying_key());

        assert_ne!(
            did_first, did_second,
            "force-overwrite should generate a new keypair with a different DID"
        );
    }

    #[tokio::test]
    async fn test_identity_show_requires_existing_keystore() {
        let (_dir, data_dir) = temp_data_dir();

        // No keystore — should get a clear "not found" error without prompting.
        let err = identity_show_with_passphrase(&data_dir, Some(""))
            .await
            .unwrap_err();

        assert!(
            err.to_string().contains("No identity found"),
            "expected 'No identity found' error, got: {err}"
        );
    }

    #[tokio::test]
    async fn test_identity_show_succeeds_after_init() {
        let (_dir, data_dir) = temp_data_dir();

        identity_init_with_passphrase(&data_dir, false, Some("show-pass"))
            .await
            .unwrap();
        identity_show_with_passphrase(&data_dir, Some("show-pass"))
            .await
            .unwrap();
    }

    // ── wrong passphrase ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_identity_show_wrong_passphrase_returns_error() {
        let (_dir, data_dir) = temp_data_dir();

        identity_init_with_passphrase(&data_dir, false, Some("correct-pass"))
            .await
            .unwrap();

        let err = identity_show_with_passphrase(&data_dir, Some("wrong-pass"))
            .await
            .unwrap_err();

        // Should fail decryption — error message varies but must not be empty
        assert!(
            !err.to_string().is_empty(),
            "expected decryption failure, got empty error"
        );
    }

    // ── DID is deterministic for same key ─────────────────────────────────────

    #[tokio::test]
    async fn test_identity_did_is_stable_across_loads() {
        let (_dir, data_dir) = temp_data_dir();

        identity_init_with_passphrase(&data_dir, false, Some("stable-pass"))
            .await
            .unwrap();

        let path = keystore::keystore_path(&data_dir);

        let key1 = keystore::load("stable-pass", &path).unwrap();
        let key2 = keystore::load("stable-pass", &path).unwrap();

        let did1 = keypair::to_did(&key1.verifying_key());
        let did2 = keypair::to_did(&key2.verifying_key());

        assert_eq!(did1, did2, "DID must be stable across loads");
    }

    // ── DID format ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_identity_did_has_did_key_prefix() {
        let (_dir, data_dir) = temp_data_dir();

        identity_init_with_passphrase(&data_dir, false, Some("did-pass"))
            .await
            .unwrap();

        let path = keystore::keystore_path(&data_dir);
        let key = keystore::load("did-pass", &path).unwrap();
        let did = keypair::to_did(&key.verifying_key());

        assert!(
            did.starts_with("did:key:"),
            "DID must start with 'did:key:', got: {did}"
        );
    }

    // ── force overwrite generates new keypair ─────────────────────────────────

    #[tokio::test]
    async fn test_identity_force_generates_fresh_keypair_each_time() {
        let (_dir, data_dir) = temp_data_dir();
        let path = keystore::keystore_path(&data_dir);

        let mut dids = std::collections::HashSet::new();
        for i in 0..3 {
            identity_init_with_passphrase(&data_dir, true, Some(&format!("pass-{i}")))
                .await
                .unwrap();
            let key = keystore::load(&format!("pass-{i}"), &path).unwrap();
            dids.insert(keypair::to_did(&key.verifying_key()));
        }

        // All three force-generated keypairs should be unique
        assert_eq!(dids.len(), 3, "each force-init should produce a unique DID");
    }
}
