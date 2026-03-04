//! JWT token minting and verification using Ed25519 (EdDSA).

use ed25519_dalek::pkcs8::EncodePrivateKey;
use ed25519_dalek::{SigningKey, VerifyingKey};
use jsonwebtoken::{Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::capability::Capability;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors produced by the auth token subsystem.
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("token expired")]
    TokenExpired,

    #[error("invalid token: {0}")]
    InvalidToken(String),

    #[error("audience mismatch: expected {expected}, got {actual}")]
    AudienceMismatch { expected: String, actual: String },

    #[error("insufficient capability: need {required}, have {actual}")]
    InsufficientCapability {
        required: Capability,
        actual: Capability,
    },

    #[error("signing error: {0}")]
    Signing(String),
}

// ─── Claims ───────────────────────────────────────────────────────────────────

/// JWT claims carried by a VecLayer auth token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject — the DID of the token holder.
    pub sub: String,
    /// Audience — the DID of the server (prevents token replay across servers).
    pub aud: String,
    /// Capability level granted by this token.
    pub cap: Capability,
    /// Issued at (Unix timestamp).
    pub iat: u64,
    /// Expires at (Unix timestamp).
    pub exp: u64,
    /// Unique token ID.
    pub jti: String,
}

impl Claims {
    /// Create a new Claims value with a generated JTI.
    pub fn new(sub: String, aud: String, cap: Capability, iat: u64, exp: u64) -> Self {
        Self {
            sub,
            aud,
            cap,
            iat,
            exp,
            jti: Uuid::new_v4().to_string(),
        }
    }
}

// ─── Mint ─────────────────────────────────────────────────────────────────────

/// Mint a new JWT token signed with the given Ed25519 key.
pub fn mint(signing_key: &SigningKey, claims: &Claims) -> Result<String, AuthError> {
    let der = signing_key
        .to_pkcs8_der()
        .map_err(|e: ed25519_dalek::pkcs8::Error| AuthError::Signing(e.to_string()))?;

    let encoding_key = EncodingKey::from_ed_der(der.as_bytes());
    let header = Header::new(Algorithm::EdDSA);

    jsonwebtoken::encode(&header, claims, &encoding_key)
        .map_err(|e| AuthError::Signing(e.to_string()))
}

// ─── Verify ───────────────────────────────────────────────────────────────────

/// Verify a JWT token and extract claims.
///
/// Checks: signature, expiry, and audience (if `expected_audience` is `Some`).
pub fn verify(
    token: &str,
    verifying_key: &VerifyingKey,
    expected_audience: Option<&str>,
) -> Result<Claims, AuthError> {
    // jsonwebtoken's rust_crypto EdDSA verifier reads the first 32 bytes as raw
    // public key bytes — it does not parse DER/SPKI structure despite the method name.
    let decoding_key = DecodingKey::from_ed_der(verifying_key.as_bytes());

    let mut validation = Validation::new(Algorithm::EdDSA);
    // Audience is validated manually after decoding so we can return a precise error.
    validation.validate_aud = false;

    let data = jsonwebtoken::decode::<Claims>(token, &decoding_key, &validation).map_err(|e| {
        if e.kind() == &jsonwebtoken::errors::ErrorKind::ExpiredSignature {
            AuthError::TokenExpired
        } else {
            AuthError::InvalidToken(e.to_string())
        }
    })?;

    let claims = data.claims;

    if let Some(expected) = expected_audience {
        if claims.aud != expected {
            return Err(AuthError::AudienceMismatch {
                expected: expected.to_owned(),
                actual: claims.aud,
            });
        }
    }

    Ok(claims)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    use super::*;

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    fn generate_key() -> SigningKey {
        SigningKey::generate(&mut OsRng)
    }

    fn make_claims(cap: Capability, iat: u64, exp: u64) -> Claims {
        Claims::new(
            "did:key:zAlice".to_owned(),
            "did:key:zServer".to_owned(),
            cap,
            iat,
            exp,
        )
    }

    #[test]
    fn test_mint_verify_roundtrip() {
        let key = generate_key();
        let t = now();
        let claims = make_claims(Capability::Write, t, t + 3600);

        let token = mint(&key, &claims).expect("mint");
        let recovered =
            verify(&token, &key.verifying_key(), Some("did:key:zServer")).expect("verify");

        assert_eq!(recovered.sub, claims.sub);
        assert_eq!(recovered.aud, claims.aud);
        assert_eq!(recovered.cap, claims.cap);
        assert_eq!(recovered.iat, claims.iat);
        assert_eq!(recovered.exp, claims.exp);
        assert_eq!(recovered.jti, claims.jti);
    }

    #[test]
    fn test_expired_token_rejected() {
        let key = generate_key();
        // exp in the past
        let t = now();
        let claims = make_claims(Capability::Read, t - 7200, t - 3600);

        let token = mint(&key, &claims).expect("mint");
        let err = verify(&token, &key.verifying_key(), None).unwrap_err();

        assert!(
            matches!(err, AuthError::TokenExpired),
            "expected TokenExpired, got: {err}"
        );
    }

    #[test]
    fn test_wrong_key_rejected() {
        let key_a = generate_key();
        let key_b = generate_key();
        let t = now();
        let claims = make_claims(Capability::Read, t, t + 3600);

        let token = mint(&key_a, &claims).expect("mint");
        let err = verify(&token, &key_b.verifying_key(), None).unwrap_err();

        assert!(
            matches!(err, AuthError::InvalidToken(_)),
            "expected InvalidToken, got: {err}"
        );
    }

    #[test]
    fn test_audience_check() {
        let key = generate_key();
        let t = now();
        let claims = make_claims(Capability::Admin, t, t + 3600);

        let token = mint(&key, &claims).expect("mint");
        let err = verify(&token, &key.verifying_key(), Some("did:key:zOtherServer")).unwrap_err();

        assert!(
            matches!(
                err,
                AuthError::AudienceMismatch { ref expected, ref actual }
                    if expected == "did:key:zOtherServer" && actual == "did:key:zServer"
            ),
            "expected AudienceMismatch, got: {err}"
        );
    }

    #[test]
    fn test_audience_none_skips_check() {
        let key = generate_key();
        let t = now();
        let claims = make_claims(Capability::Read, t, t + 3600);

        let token = mint(&key, &claims).expect("mint");
        // passing None should succeed regardless of aud
        let recovered = verify(&token, &key.verifying_key(), None).expect("verify with None aud");
        assert_eq!(recovered.sub, claims.sub);
    }

    #[test]
    fn test_claims_serde() {
        let t = now();
        let claims = Claims {
            sub: "did:key:zAlice".to_owned(),
            aud: "did:key:zServer".to_owned(),
            cap: Capability::Write,
            iat: t,
            exp: t + 3600,
            jti: "test-jti-uuid".to_owned(),
        };

        let json = serde_json::to_string(&claims).expect("serialize");
        assert!(json.contains("\"sub\":\"did:key:zAlice\""));
        assert!(json.contains("\"cap\":\"write\""));
        assert!(json.contains("\"jti\":\"test-jti-uuid\""));

        let recovered: Claims = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(recovered.sub, claims.sub);
        assert_eq!(recovered.cap, claims.cap);
        assert_eq!(recovered.jti, claims.jti);
    }
}
