//! Shared utility functions used across modules.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

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
