//! Ed25519 key generation and DID key formatting/parsing.
//!
//! DID format: `did:key:z6Mk...`
//!
//! The identifier is constructed by prepending the multicodec prefix `[0xed, 0x01]`
//! to the 32-byte raw public key and encoding the result as base58-btc with a `z` prefix.

use ed25519_dalek::{SigningKey, VerifyingKey};
use rand::rngs::OsRng;

use super::CryptoError;

/// Multicodec prefix for Ed25519 public keys.
const ED25519_MULTICODEC: [u8; 2] = [0xed, 0x01];

/// Length of an Ed25519 raw public key in bytes.
const PUBLIC_KEY_LEN: usize = 32;

/// Generate a new Ed25519 signing key using the OS random number generator.
pub fn generate() -> SigningKey {
    SigningKey::generate(&mut OsRng)
}

/// Format a verifying key as a `did:key` string.
///
/// The DID is formed by base58-btc-encoding the concatenation of the Ed25519
/// multicodec prefix (`[0xed, 0x01]`) and the 32-byte public key, then
/// prepending `did:key:z`.
pub fn to_did(verifying_key: &VerifyingKey) -> String {
    let mut payload = Vec::with_capacity(ED25519_MULTICODEC.len() + PUBLIC_KEY_LEN);
    payload.extend_from_slice(&ED25519_MULTICODEC);
    payload.extend_from_slice(verifying_key.as_bytes());
    format!("did:key:z{}", bs58::encode(payload).into_string())
}

/// Parse a `did:key` string back into a [`VerifyingKey`].
///
/// # Errors
///
/// Returns [`CryptoError::InvalidDid`] if the input is not a valid `did:key:z`
/// string, does not decode as base58, lacks the Ed25519 multicodec prefix,
/// or does not contain a valid 32-byte public key.
pub fn from_did(did: &str) -> Result<VerifyingKey, CryptoError> {
    let encoded = did
        .strip_prefix("did:key:z")
        .ok_or_else(|| CryptoError::InvalidDid(format!("expected 'did:key:z' prefix: {did}")))?;

    let bytes = bs58::decode(encoded)
        .into_vec()
        .map_err(|e| CryptoError::InvalidDid(format!("base58 decode failed: {e}")))?;

    if bytes.len() < ED25519_MULTICODEC.len() + PUBLIC_KEY_LEN {
        return Err(CryptoError::InvalidDid(format!(
            "decoded bytes too short: {} bytes",
            bytes.len()
        )));
    }

    if bytes[..2] != ED25519_MULTICODEC {
        return Err(CryptoError::InvalidDid(format!(
            "expected Ed25519 multicodec prefix [0xed, 0x01], got [{:#04x}, {:#04x}]",
            bytes[0], bytes[1]
        )));
    }

    let key_bytes: [u8; PUBLIC_KEY_LEN] = bytes[2..2 + PUBLIC_KEY_LEN]
        .try_into()
        .map_err(|_| CryptoError::InvalidDid("internal: unexpected key length".into()))?;

    VerifyingKey::from_bytes(&key_bytes)
        .map_err(|e| CryptoError::InvalidDid(format!("invalid Ed25519 public key: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_produces_valid_key() {
        let signing_key = generate();
        // We can derive a verifying key from it without panicking.
        let verifying_key = signing_key.verifying_key();
        // The raw bytes are 32 bytes long.
        assert_eq!(verifying_key.as_bytes().len(), 32);
    }

    #[test]
    fn test_did_roundtrip() {
        let signing_key = generate();
        let verifying_key = signing_key.verifying_key();
        let did = to_did(&verifying_key);
        let recovered = from_did(&did).expect("roundtrip must succeed");
        assert_eq!(verifying_key.as_bytes(), recovered.as_bytes());
    }

    #[test]
    fn test_did_format_starts_with_did_key_z() {
        let signing_key = generate();
        let did = to_did(&signing_key.verifying_key());
        assert!(did.starts_with("did:key:z"), "got: {did}");
    }

    #[test]
    fn test_from_did_rejects_invalid_prefix() {
        let err = from_did("did:web:example.com").unwrap_err();
        assert!(matches!(err, CryptoError::InvalidDid(_)));
    }

    #[test]
    fn test_from_did_rejects_wrong_multicodec() {
        // Build a DID with a wrong multicodec prefix (0x01, 0x01 instead of 0xed, 0x01).
        let mut payload = vec![0x01u8, 0x01];
        payload.extend_from_slice(&[0u8; 32]);
        let did = format!("did:key:z{}", bs58::encode(payload).into_string());
        let err = from_did(&did).unwrap_err();
        assert!(matches!(err, CryptoError::InvalidDid(_)));
    }

    #[test]
    fn test_from_did_rejects_too_short() {
        // "did:key:z" followed by base58-encoded single byte — not enough data.
        let did = format!("did:key:z{}", bs58::encode([0xed]).into_string());
        let err = from_did(&did).unwrap_err();
        assert!(matches!(err, CryptoError::InvalidDid(_)));
    }

    #[test]
    fn test_from_did_rejects_not_did_key() {
        let err = from_did("not-a-did").unwrap_err();
        assert!(matches!(err, CryptoError::InvalidDid(_)));
    }
}
