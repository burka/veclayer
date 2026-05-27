//! Axum Bearer-token middleware and capability enforcement helpers.

use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use ed25519_dalek::VerifyingKey;
use tracing::warn;

use super::capability::Capability;
use super::token;

// ─── Auth state ───────────────────────────────────────────────────────────────

/// Auth configuration shared with the middleware via `axum::middleware::from_fn_with_state`.
#[derive(Clone)]
pub struct AuthState {
    pub verifying_key: VerifyingKey,
    pub server_did: String,
    pub auth_required: bool,
}

// ─── Middleware ────────────────────────────────────────────────────────────────

/// Axum middleware: extract a Bearer token, verify it, and inject the resulting
/// [`Capability`] (and full [`Claims`]) into the request extensions.
///
/// # Behaviour
///
/// | Situation                           | Response                  |
/// |-------------------------------------|---------------------------|
/// | Valid token                         | request forwarded         |
/// | Invalid / expired token             | 401 Unauthorized          |
/// | No token, `auth_required = false`   | forwarded with Admin cap  |
/// | No token, `auth_required = true`    | 401 with WWW-Authenticate |
///
/// Wire it up with:
/// ```ignore
/// axum::middleware::from_fn_with_state(auth_state, auth_middleware)
/// ```
pub async fn auth_middleware(
    axum::extract::State(auth): axum::extract::State<AuthState>,
    mut request: Request,
    next: Next,
) -> Response {
    let token = extract_bearer(&request);

    match token {
        Some(token_str) => {
            match token::verify(token_str, &auth.verifying_key, Some(&auth.server_did)) {
                Ok(claims) => {
                    request.extensions_mut().insert(claims.cap);
                    request.extensions_mut().insert(claims);
                    next.run(request).await
                }
                Err(e) => {
                    warn!("Auth rejected: {e}");
                    (StatusCode::UNAUTHORIZED, "invalid_token").into_response()
                }
            }
        }
        None if !auth.auth_required => {
            request.extensions_mut().insert(Capability::Admin);
            next.run(request).await
        }
        None => {
            warn!("Auth rejected: no bearer token");
            (
                StatusCode::UNAUTHORIZED,
                [("www-authenticate", "Bearer")],
                "Authentication required",
            )
                .into_response()
        }
    }
}

/// Extract the raw token string from the `Authorization: Bearer <token>` header.
fn extract_bearer(request: &Request) -> Option<&str> {
    request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
}

// ─── Capability enforcement helpers ───────────────────────────────────────────

/// Require at least [`Capability::Read`] — 403 otherwise.
pub async fn require_read(request: Request, next: Next) -> Response {
    enforce_capability(request, next, Capability::Read).await
}

/// Require at least [`Capability::Write`] — 403 otherwise.
pub async fn require_write(request: Request, next: Next) -> Response {
    enforce_capability(request, next, Capability::Write).await
}

/// Require at least [`Capability::Admin`] — 403 otherwise.
pub async fn require_admin(request: Request, next: Next) -> Response {
    enforce_capability(request, next, Capability::Admin).await
}

async fn enforce_capability(request: Request, next: Next, required: Capability) -> Response {
    match request.extensions().get::<Capability>() {
        Some(cap) if cap.permits(required) => next.run(request).await,
        Some(cap) => (
            StatusCode::FORBIDDEN,
            format!("Insufficient capability: need {required}, have {cap}"),
        )
            .into_response(),
        None => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "No capability in request extensions",
        )
            .into_response(),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use axum::{
        body::Body,
        http::{Request, StatusCode},
        middleware,
        routing::get,
        Router,
    };
    use ed25519_dalek::SigningKey;
    use rand_core::OsRng;
    use tower::ServiceExt;

    use super::*;
    use crate::auth::token::{self, Claims};

    // ── Test helpers ──────────────────────────────────────────────────────────

    const SERVER_DID: &str = "did:key:zServer";
    const CLIENT_DID: &str = "did:key:zClient";

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    fn generate_key() -> SigningKey {
        SigningKey::generate(&mut OsRng)
    }

    fn mint_token(key: &SigningKey, cap: Capability, iat: u64, exp: u64) -> String {
        let claims = Claims::new(
            SERVER_DID.to_owned(),
            CLIENT_DID.to_owned(),
            SERVER_DID.to_owned(),
            cap,
            iat,
            exp,
        );
        token::mint(key, &claims).expect("mint")
    }

    fn auth_state(key: &SigningKey, auth_required: bool) -> AuthState {
        AuthState {
            verifying_key: key.verifying_key(),
            server_did: SERVER_DID.to_owned(),
            auth_required,
        }
    }

    /// Build a minimal app with only the auth middleware layer.
    ///
    /// The single GET "/" handler echoes the capability from request extensions.
    fn build_app(state: AuthState) -> Router {
        Router::new()
            .route("/", get(cap_handler_fn))
            .layer(middleware::from_fn_with_state(state, auth_middleware))
    }

    /// Build an app that also enforces a minimum capability level.
    fn build_app_with_guard(
        state: AuthState,
        guard: impl Fn(
                Request<Body>,
                Next,
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Response> + Send>>
            + Clone
            + Send
            + Sync
            + 'static,
    ) -> Router {
        Router::new()
            .route("/", get(cap_handler_fn).layer(middleware::from_fn(guard)))
            .layer(middleware::from_fn_with_state(state, auth_middleware))
    }

    /// GET "/" handler that echoes the capability string from request extensions.
    async fn cap_handler_fn(request: Request<Body>) -> String {
        let cap = request
            .extensions()
            .get::<Capability>()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "none".to_owned());
        cap
    }

    async fn send(app: Router, bearer: Option<&str>) -> (StatusCode, String) {
        let mut builder = Request::builder().uri("/").method("GET");
        if let Some(token) = bearer {
            builder = builder.header("authorization", format!("Bearer {token}"));
        }
        let request = builder.body(Body::empty()).unwrap();
        let response = app.oneshot(request).await.unwrap();
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        (status, String::from_utf8_lossy(&body).into_owned())
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_valid_bearer_passes_through() {
        let key = generate_key();
        let t = now();
        let jwt = mint_token(&key, Capability::Write, t, t + 3600);
        let app = build_app(auth_state(&key, true));

        let (status, body) = send(app, Some(&jwt)).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, "write");
    }

    #[tokio::test]
    async fn test_missing_bearer_when_required() {
        let key = generate_key();
        let app = build_app(auth_state(&key, true));

        let (status, _body) = send(app, None).await;

        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_missing_bearer_www_authenticate_header() {
        let key = generate_key();
        let app = build_app(auth_state(&key, true));

        let request = Request::builder()
            .uri("/")
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            response.headers().get("www-authenticate").unwrap(),
            "Bearer"
        );
    }

    #[tokio::test]
    async fn test_missing_bearer_when_not_required() {
        let key = generate_key();
        let app = build_app(auth_state(&key, false));

        let (status, body) = send(app, None).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, "admin");
    }

    #[tokio::test]
    async fn test_invalid_bearer_rejected() {
        let key = generate_key();
        let app = build_app(auth_state(&key, true));

        let (status, body) = send(app, Some("garbage.token.here")).await;

        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert!(body.contains("invalid_token"));
    }

    #[tokio::test]
    async fn test_expired_bearer_rejected() {
        let key = generate_key();
        let t = now();
        // exp in the past
        let jwt = mint_token(&key, Capability::Read, t - 7200, t - 3600);
        let app = build_app(auth_state(&key, true));

        let (status, body) = send(app, Some(&jwt)).await;

        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert!(body.contains("invalid_token"));
    }

    /// The client-facing error body must be a generic opaque token and must not
    /// contain internal error detail (e.g. jsonwebtoken error messages or
    /// header/signature strings from the token itself).
    #[tokio::test]
    async fn test_invalid_bearer_body_does_not_leak_internal_detail() {
        let key = generate_key();

        // Use a structurally plausible but wrong-key-signed token so that the
        // underlying verify() produces a detailed internal error message (e.g.
        // "InvalidSignature" or similar from jsonwebtoken).
        let wrong_key = generate_key();
        let t = now();
        let jwt = mint_token(&wrong_key, Capability::Write, t, t + 3600);
        let app = build_app(auth_state(&key, true)); // verifier uses `key`, not `wrong_key`

        let (status, body) = send(app, Some(&jwt)).await;

        assert_eq!(status, StatusCode::UNAUTHORIZED);

        // Body must be the generic token — no internal error strings.
        assert_eq!(
            body, "invalid_token",
            "client-facing body must be exactly 'invalid_token', got: {body:?}"
        );

        // Specifically must not contain internal error detail substrings.
        assert!(
            !body.to_lowercase().contains("signature"),
            "body must not leak signature detail, got: {body:?}"
        );
        assert!(
            !body.to_lowercase().contains("invalid token:"),
            "body must not contain 'invalid token:', got: {body:?}"
        );
    }

    #[tokio::test]
    async fn test_capability_enforcement_read_token_blocks_write_route() {
        let key = generate_key();
        check_require_write(
            &key,
            Capability::Read,
            StatusCode::FORBIDDEN,
            &["need write", "have read"],
        )
        .await;
    }

    #[tokio::test]
    async fn test_capability_enforcement_write_token_passes_write_route() {
        let key = generate_key();
        check_require_write(&key, Capability::Write, StatusCode::OK, &["write"]).await;
    }

    /// Sends a request with the given JWT through the require_write guard and
    /// asserts the expected status and that the body contains each expected substring.
    async fn check_require_write(
        key: &SigningKey,
        cap: Capability,
        expected_status: StatusCode,
        body_checks: &[&str],
    ) {
        let t = now();
        let jwt = mint_token(key, cap, t, t + 3600);
        let app = build_app_with_guard(auth_state(key, true), |req, next| {
            Box::pin(require_write(req, next))
        });
        let (status, body) = send(app, Some(&jwt)).await;
        assert_eq!(status, expected_status);
        for check in body_checks {
            assert!(body.contains(check), "body should contain '{check}'");
        }
    }

    // ── require_admin tests ───────────────────────────────────────────────────

    /// A `Write`-capability token must be rejected (403) by an admin-guarded route.
    /// This pins the security invariant: Write does NOT imply Admin.
    #[tokio::test]
    async fn test_require_admin_rejects_write_capability() {
        let key = generate_key();
        let t = now();
        let jwt = mint_token(&key, Capability::Write, t, t + 3600);
        let app = build_app_with_guard(auth_state(&key, true), |req, next| {
            Box::pin(require_admin(req, next))
        });

        let (status, body) = send(app, Some(&jwt)).await;

        assert_eq!(status, StatusCode::FORBIDDEN);
        assert!(
            body.contains("need admin"),
            "body should say 'need admin', got: {body:?}"
        );
        assert!(
            body.contains("have write"),
            "body should say 'have write', got: {body:?}"
        );
    }

    /// A `Read`-capability token must also be rejected (403) by an admin-guarded route.
    #[tokio::test]
    async fn test_require_admin_rejects_read_capability() {
        let key = generate_key();
        let t = now();
        let jwt = mint_token(&key, Capability::Read, t, t + 3600);
        let app = build_app_with_guard(auth_state(&key, true), |req, next| {
            Box::pin(require_admin(req, next))
        });

        let (status, body) = send(app, Some(&jwt)).await;

        assert_eq!(status, StatusCode::FORBIDDEN);
        assert!(
            body.contains("need admin"),
            "body should say 'need admin', got: {body:?}"
        );
        assert!(
            body.contains("have read"),
            "body should say 'have read', got: {body:?}"
        );
    }

    /// An `Admin`-capability token must pass an admin-guarded route (200 OK).
    #[tokio::test]
    async fn test_require_admin_allows_admin_capability() {
        let key = generate_key();
        let t = now();
        let jwt = mint_token(&key, Capability::Admin, t, t + 3600);
        let app = build_app_with_guard(auth_state(&key, true), |req, next| {
            Box::pin(require_admin(req, next))
        });

        let (status, body) = send(app, Some(&jwt)).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, "admin");
    }

    // ── enforce_capability None-branch test ───────────────────────────────────

    /// When no `Capability` extension is present (i.e. `auth_middleware` was not
    /// wired up), `enforce_capability` must return 500 with the sentinel message.
    /// This ensures the server fails loudly rather than silently permitting access.
    #[tokio::test]
    async fn test_enforce_capability_returns_500_when_no_capability_extension() {
        // Deliberately omit auth_middleware so no Capability is injected.
        let app = Router::new().route(
            "/",
            get(cap_handler_fn).layer(middleware::from_fn(|req, next| {
                Box::pin(require_admin(req, next))
            })),
        );

        let request = Request::builder()
            .uri("/")
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        let status = response.status();
        let body_bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let body = String::from_utf8_lossy(&body_bytes);

        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(
            body.contains("No capability in request extensions"),
            "body should contain the sentinel message, got: {body:?}"
        );
    }

    #[tokio::test]
    async fn test_claims_injected_into_extensions() {
        let key = generate_key();
        let t = now();
        let jwt = mint_token(&key, Capability::Admin, t, t + 3600);

        let handler = get(|request: Request<Body>| async move {
            let claims = request.extensions().get::<Claims>().cloned();
            match claims {
                Some(c) => format!("sub={},cap={}", c.sub, c.cap),
                None => "no-claims".to_owned(),
            }
        });
        let app = Router::new()
            .route("/", handler)
            .layer(middleware::from_fn_with_state(
                auth_state(&key, true),
                auth_middleware,
            ));

        let (status, body) = send(app, Some(&jwt)).await;

        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("sub=did:key:zClient"));
        assert!(body.contains("cap=admin"));
    }
}
