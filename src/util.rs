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

/// Run a short-lived async service-discovery probe to completion from a
/// synchronous context (e.g. `Config::new()`), returning `None` when blocking
/// is impossible.
///
/// Dispatch mirrors the constraints `block_in_place` imposes:
/// - Multi-threaded runtime present → `block_in_place` + `block_on`.
/// - Current-thread runtime (e.g. default `#[tokio::test]`) → `None`, because
///   `block_in_place` would panic there.
/// - No runtime → spin up a temporary current-thread runtime for the probe.
///
/// Shared by the Ollama and OpenAI-compatible discovery paths.
#[cfg(feature = "llm")]
pub fn block_on_probe<F, T>(fut: F) -> Option<T>
where
    F: std::future::Future<Output = Option<T>>,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::CurrentThread {
                tracing::debug!("Service auto-discovery skipped: single-threaded runtime");
                return None;
            }
            tokio::task::block_in_place(|| handle.block_on(fut))
        }
        Err(_) => match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt.block_on(fut),
            Err(e) => {
                tracing::debug!("Service discovery: could not build runtime: {e}");
                None
            }
        },
    }
}

// ─── HTTP client hardening ────────────────────────────────────────────────────

/// Maximum number of bytes accepted from any outbound HTTP response body.
///
/// Prevents OOM when talking to configurable (and potentially hostile) endpoints.
/// 16 MiB is generous for all embedding / LLM JSON payloads in practice.
#[cfg(feature = "llm")]
pub const MAX_HTTP_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Build a hardened `reqwest` client: redirects disabled, with explicit timeouts.
///
/// **Redirect policy**: `none()` — any 3xx from a user-configured `base_url`
/// must not silently pivot to a second endpoint (SSRF / pivot guard).
///
/// **Timeouts**: callers provide connect + overall timeout; there is no fallback
/// default so every call site is forced to be explicit.
///
/// Returns `None` only when the TLS backend cannot be initialised (rare).
#[cfg(feature = "llm")]
pub fn build_hardened_client(
    connect_timeout: std::time::Duration,
    timeout: std::time::Duration,
) -> Option<reqwest::Client> {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(connect_timeout)
        .timeout(timeout)
        .build()
        .ok()
}

/// Read a `reqwest::Response` body by streaming, aborting if `cap` bytes are
/// exceeded before the body is fully consumed.
///
/// This is the safe alternative to `response.bytes()` (which buffers without
/// limit) for all configurable / external endpoints.
///
/// The cap is checked **on streamed byte count**, not on `Content-Length` alone:
/// a lying or absent header still triggers the limit.
///
/// Returns the accumulated bytes on success, or an `io::Error` with kind
/// `InvalidData` when the cap is exceeded.
#[cfg(feature = "llm")]
pub async fn read_capped_body(
    mut response: reqwest::Response,
    cap: usize,
) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| std::io::Error::other(e.to_string()))?
    {
        if buf.len() + chunk.len() > cap {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("response body too large: would exceed {} byte cap", cap),
            ));
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf)
}

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
