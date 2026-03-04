//! Cryptographic identity — Ed25519 keypairs, DIDs, encrypted keystore,
//! and OAuth 2.0 device authorization flow.

pub mod device_flow;
pub mod keypair;
pub mod keystore;

use thiserror::Error;

/// Errors specific to cryptographic operations.
#[derive(Debug, Error)]
pub enum CryptoError {
    #[error("invalid DID format: {0}")]
    InvalidDid(String),

    #[error("keystore error: {0}")]
    Keystore(String),

    #[error("decryption failed — wrong passphrase?")]
    DecryptionFailed,

    #[error("DID mismatch — keystore may be corrupted")]
    DidMismatch,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<CryptoError> for crate::Error {
    fn from(e: CryptoError) -> Self {
        crate::Error::Crypto(e.to_string())
    }
}
