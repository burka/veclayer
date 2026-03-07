//! Authentication and authorization — JWT tokens, capabilities, OAuth.
//!
//! The `capability` submodule is always available (used by both transports).
//! All other submodules require the `auth` feature (crypto, JWT, keystore).
//! The `middleware` and `oauth` submodules additionally require the `http` feature.

pub mod capability;
#[cfg(feature = "auth")]
pub mod token;
#[cfg(feature = "auth")]
pub mod token_store;
#[cfg(feature = "http")]
pub mod middleware;
#[cfg(feature = "http")]
pub mod oauth;
