//! Shared utility functions and constants used across modules.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

// ─── Well-known URLs ─────────────────────────────────────────────────────────

/// Default base URL for the Ollama API server.
pub const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

// ─── Ollama defaults ─────────────────────────────────────────────────────────

/// Default Ollama embedding model.
pub const DEFAULT_OLLAMA_EMBED_MODEL: &str = "nomic-embed-text";
/// Default Ollama embedding dimension for the default model (768 for nomic-embed-text).
pub const DEFAULT_OLLAMA_DIMENSION: usize = 768;

// ─── Time constants ───────────────────────────────────────────────────────────

/// Default JWT access token expiry (1 hour in seconds).
pub const TOKEN_EXPIRY_SECS: u64 = 3_600;
/// Default refresh token expiry (1 day in seconds).
pub const REFRESH_EXPIRY_SECS: u64 = 86_400;
/// Default refresh token max lifetime (30 days in seconds).
pub const REFRESH_MAX_LIFETIME_SECS: u64 = 2_592_000;
/// Seconds per hour (conversion factor for duration parsing).
pub const SECS_PER_HOUR: u64 = 3_600;
/// Seconds per day (conversion factor for duration parsing).
pub const SECS_PER_DAY: u64 = 86_400;

/// Truncate `s` to at most `max` bytes, replacing newlines with spaces.
///
/// Uses `floor_char_boundary` so multi-byte codepoints are never split.
/// Appends `"..."` when truncation occurs.
pub fn preview(s: &str, max: usize) -> String {
    let clean = s.replace('\n', " ");
    if clean.len() <= max {
        clean
    } else {
        let end = clean.floor_char_boundary(max);
        format!("{}...", &clean[..end])
    }
}

/// Returns the current Unix timestamp in seconds.
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before UNIX epoch")
        .as_secs()
}

/// Set file permissions to 0o600 on Unix; no-op on other platforms.
pub fn set_file_mode_600(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o600);
        std::fs::set_permissions(path, perms)?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_unix_now_reasonable() {
        let now = unix_now();
        // Should be after 2024-01-01
        assert!(
            now > 1_704_067_200,
            "unix_now returned suspiciously low value: {now}"
        );
    }

    #[test]
    fn test_set_file_mode_600() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test_file");
        std::fs::write(&path, "secret").unwrap();
        set_file_mode_600(&path).unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let mode = std::fs::metadata(&path).unwrap().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
    }
}
