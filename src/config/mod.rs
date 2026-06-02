//! Configuration with 12-factor layered resolution: ENV > TOML file > Defaults.
//!
//! Config file lookup order:
//! 1. `$VECLAYER_CONFIG` (explicit path)
//! 2. `<data_dir>/veclayer.toml`
//! 3. `./veclayer.toml`
//!
//! User config lookup order (for match-based overrides):
//! 1. `$VECLAYER_USER_CONFIG` (explicit path)
//! 2. `$XDG_CONFIG_HOME/veclayer/config.toml`
//! 3. `$HOME/.config/veclayer/config.toml`
//! 4. `$HOME/.veclayer/config.toml`
//!
//! Any field set in the environment always wins over the config file.

mod types;

#[cfg(feature = "config")]
mod discovery;

#[cfg(feature = "config")]
mod user_file;

#[cfg(all(test, feature = "config"))]
mod tests;

// --- Public surface re-exports (preserves crate::config::* paths) ---

// Always-available types (no feature gate)
pub use types::{AuthConfig, Config, EmbedderConfig, LlmConfig, ProjectConfig, ScopeConfig};

// Feature-gated types and functions
#[cfg(feature = "config")]
pub use types::{
    parse_push_mode, MatchOverride, ResolvedConfig, ResolvedScope, GLOB_MATCH_OPTIONS,
};

#[cfg(feature = "config")]
pub use discovery::{discover_project, user_config_path, UserConfig};

#[cfg(feature = "config")]
pub use user_file::append_match_to_user_config;
