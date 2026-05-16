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
/// Seconds per hour.
pub const SECS_PER_HOUR: u64 = 3_600;
/// Seconds per day.
pub const SECS_PER_DAY: u64 = 86_400;
/// Seconds per week.
pub const SECS_PER_WEEK: u64 = 604_800;
/// Seconds per 30-day month.
pub const SECS_PER_MONTH: u64 = 2_592_000;
/// Seconds per 365-day year.
pub const SECS_PER_YEAR: u64 = 31_536_000;

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

/// Format a byte count as a human-readable string with ~4 significant digits.
///
/// Uses binary units (1024-based) labelled as `B` / `KB` / `MB` / `GB`, matching
/// the conventional `du -h` style. Decimal precision shrinks as the value grows
/// so the result stays around four significant digits (e.g. `1.234 GB`,
/// `18.05 MB`, `123.4 KB`).
pub fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;

    let b = bytes as f64;
    let (val, unit) = if b >= GB {
        (b / GB, "GB")
    } else if b >= MB {
        (b / MB, "MB")
    } else if b >= KB {
        (b / KB, "KB")
    } else {
        return format!("{bytes} B");
    };

    let precision = if val >= 100.0 {
        1
    } else if val >= 10.0 {
        2
    } else {
        3
    };
    format!("{val:.precision$} {unit}")
}

/// Write `contents` to `path`, creating the file with 0o600 permissions
/// *before* any data is written so the file is never briefly world-readable.
///
/// Use this instead of `std::fs::write` followed by a chmod when the file
/// holds secrets (token hashes, keys): the write-then-chmod sequence leaves a
/// window during which the file is readable with the process umask.
pub fn write_file_0600(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)?;
        // `mode()` only applies when the file is created; repair the mode in
        // case a stale temp file pre-existed with looser permissions.
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        file.write_all(contents)?;
    }
    #[cfg(not(unix))]
    {
        std::fs::write(path, contents)?;
        set_file_mode_600(path)?;
    }
    Ok(())
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
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.000 KB");
        assert_eq!(format_bytes(1536), "1.500 KB");
        assert_eq!(format_bytes(123 * 1024), "123.0 KB");
        // 18928031 ≈ 18.05 MiB — the original report from the user.
        assert_eq!(format_bytes(18_928_031), "18.05 MB");
        assert_eq!(format_bytes(1024 * 1024), "1.000 MB");
        // 4 GiB → "4.000 GB" (3-decimal precision when val < 10).
        assert_eq!(format_bytes(4u64 * 1024 * 1024 * 1024), "4.000 GB");
        // 250 GiB → "250.0 GB" (1-decimal precision when val ≥ 100).
        assert_eq!(format_bytes(250u64 * 1024 * 1024 * 1024), "250.0 GB");
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
