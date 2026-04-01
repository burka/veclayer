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
    /// Issuer — the DID of the server that minted this token.
    pub iss: String,
    /// Subject — the DID of the token holder.
    pub sub: String,
    /// Audience — the DID of the server (prevents token replay across servers).
    pub aud: String,
    /// Capability level granted by this token.
    pub cap: Capability,
    /// Not before (Unix timestamp) — same as `iat`; prevents premature use.
    pub nbf: u64,
    /// Issued at (Unix timestamp).
    pub iat: u64,
    /// Expires at (Unix timestamp).
    pub exp: u64,
    /// Unique token ID.
    pub jti: String,
}

impl Claims {
    /// Create a new Claims value with a generated JTI.
    ///
    /// `iss` should be the server's DID. `nbf` is set equal to `iat`.
    pub fn new(iss: String, sub: String, aud: String, cap: Capability, iat: u64, exp: u64) -> Self {
        Self {
            iss,
            sub,
            aud,
            cap,
            nbf: iat,
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
/// Checks: signature, expiry, `nbf`, and audience (if `expected_audience` is `Some`).
/// Use [`verify_with_issuer`] when `iss` validation is also required.
pub fn verify(
    token: &str,
    verifying_key: &VerifyingKey,
    expected_audience: Option<&str>,
) -> Result<Claims, AuthError> {
    verify_with_issuer(token, verifying_key, expected_audience, None)
}

/// Verify a JWT token, additionally requiring a specific issuer DID.
///
/// Checks: signature, expiry, `nbf`, `iss`, and audience.
pub fn verify_with_issuer(
    token: &str,
    verifying_key: &VerifyingKey,
    expected_audience: Option<&str>,
    expected_issuer: Option<&str>,
) -> Result<Claims, AuthError> {
    // jsonwebtoken's rust_crypto EdDSA verifier reads the first 32 bytes as raw
    // public key bytes — it does not parse DER/SPKI structure despite the method name.
    let decoding_key = DecodingKey::from_ed_der(verifying_key.as_bytes());

    let mut validation = Validation::new(Algorithm::EdDSA);
    // Audience is validated manually after decoding so we can return a precise error.
    validation.validate_aud = false;

    if let Some(iss) = expected_issuer {
        validation.set_issuer(&[iss]);
    }
    // When no expected issuer is given, leave `validation.iss` as `None` so that
    // any `iss` value in the token (or its absence) is accepted.

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
    use rand_core::OsRng;

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
            "did:key:zServer".to_owned(),
            "did:key:zAlice".to_owned(),
            "did:key:zServer".to_owned(),
            cap,
            iat,
            exp,
        )
    }

    /// Mint a test token with a Read capability and 1-hour expiry.
    /// Returns (token, signing_key, issued_at).
    fn mint_test_token() -> (String, ed25519_dalek::SigningKey, u64) {
        let key = generate_key();
        let t = now();
        let claims = make_claims(Capability::Read, t, t + 3600);
        let token = mint(&key, &claims).expect("mint");
        (token, key, t)
    }

    #[test]
    fn test_mint_verify_roundtrip() {
        let key = generate_key();
        let t = now();
        let claims = make_claims(Capability::Write, t, t + 3600);

        let token = mint(&key, &claims).expect("mint");
        let recovered =
            verify(&token, &key.verifying_key(), Some("did:key:zServer")).expect("verify");

        assert_eq!(recovered.iss, claims.iss);
        assert_eq!(recovered.sub, claims.sub);
        assert_eq!(recovered.aud, claims.aud);
        assert_eq!(recovered.cap, claims.cap);
        assert_eq!(recovered.nbf, claims.nbf);
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
        let (token, key, _t) = mint_test_token();
        // passing None should succeed regardless of aud
        let recovered = verify(&token, &key.verifying_key(), None).expect("verify with None aud");
        assert_eq!(recovered.sub, "did:key:zAlice");
    }

    #[test]
    fn test_claims_serde() {
        let t = now();
        let claims = Claims {
            iss: "did:key:zServer".to_owned(),
            sub: "did:key:zAlice".to_owned(),
            aud: "did:key:zServer".to_owned(),
            cap: Capability::Write,
            nbf: t,
            iat: t,
            exp: t + 3600,
            jti: "test-jti-uuid".to_owned(),
        };

        let json = serde_json::to_string(&claims).expect("serialize");
        assert!(json.contains("\"iss\":\"did:key:zServer\""));
        assert!(json.contains("\"sub\":\"did:key:zAlice\""));
        assert!(json.contains("\"cap\":\"write\""));
        assert!(json.contains("\"jti\":\"test-jti-uuid\""));
        assert!(json.contains("\"nbf\""));

        let recovered: Claims = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(recovered.iss, claims.iss);
        assert_eq!(recovered.sub, claims.sub);
        assert_eq!(recovered.cap, claims.cap);
        assert_eq!(recovered.jti, claims.jti);
        assert_eq!(recovered.nbf, claims.nbf);
    }

    #[test]
    fn test_issuer_validation_accepted() {
        let (token, key, _t) = mint_test_token();
        let recovered = verify_with_issuer(
            &token,
            &key.verifying_key(),
            Some("did:key:zServer"),
            Some("did:key:zServer"),
        )
        .expect("valid issuer");
        assert_eq!(recovered.iss, "did:key:zServer");
    }

    #[test]
    fn test_issuer_validation_rejected() {
        let (token, key, _t) = mint_test_token();
        let err = verify_with_issuer(
            &token,
            &key.verifying_key(),
            None,
            Some("did:key:zWrongServer"),
        )
        .unwrap_err();
        assert!(
            matches!(err, AuthError::InvalidToken(_)),
            "expected InvalidToken for wrong issuer, got: {err}"
        );
    }

    #[test]
    fn test_nbf_present_in_minted_token() {
        let (token, key, t) = mint_test_token();
        let recovered = verify(&token, &key.verifying_key(), None).expect("verify");
        assert_eq!(recovered.nbf, t, "nbf must equal iat");
    }
}
