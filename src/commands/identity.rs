//! Identity management commands — init and show.

use std::io::{self, IsTerminal, Write};
use std::path::Path;

use owo_colors::{OwoColorize, Stream};

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

    let passphrase = match passphrase {
        Some(p) => p.to_string(),
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
    let path = keystore::keystore_path(data_dir);

    if !keystore::exists(&path) {
        return Err(crate::Error::NotFound(
            "No identity found. Run `veclayer identity init` first.".to_string(),
        ));
    }

    let passphrase = match passphrase {
        Some(p) => p.to_string(),
        None => resolve_passphrase_for_read()?,
    };
    let signing_key = keystore::load(&passphrase, &path)?;
    let verifying_key = signing_key.verifying_key();
    let did = keypair::to_did(&verifying_key);
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
fn resolve_passphrase_for_write() -> Result<String> {
    if let Ok(pass) = std::env::var("VECLAYER_PASSPHRASE") {
        return Ok(pass);
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
    Ok(String::new())
}

/// Resolve passphrase for reading (show/load): prompts once when interactive.
fn resolve_passphrase_for_read() -> Result<String> {
    if let Ok(pass) = std::env::var("VECLAYER_PASSPHRASE") {
        return Ok(pass);
    }

    if io::stdin().is_terminal() {
        return prompt_passphrase("Enter passphrase: ");
    }

    Ok(String::new())
}

/// Print `prompt` to stderr and read a passphrase without echoing input.
fn prompt_passphrase(prompt: &str) -> Result<String> {
    eprint!("{prompt}");
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
}
