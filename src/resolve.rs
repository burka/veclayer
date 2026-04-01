//! ID resolution and temporal parsing shared by CLI commands and MCP tools.
//!
//! Extracted from `mcp/tools.rs` so both surfaces use the same logic.

use crate::store::StoreBackend;
use crate::{Result, VectorStore};
use std::sync::Arc;

/// Resolve a short or full entry ID to its canonical full ID using prefix matching.
///
/// Accepts either a full content-hash ID or a short prefix (like git short hashes).
/// Returns an error if no entry matches the prefix.
pub async fn resolve_id(store: &Arc<StoreBackend>, id: &str) -> Result<String> {
    store
        .get_by_id_prefix(id)
        .await?
        .map(|chunk| chunk.id)
        .ok_or_else(|| crate::Error::not_found(format!("Entry '{}' not found", id)))
}

/// Resolve a short or full entry ID and return the full chunk.
///
/// Like `resolve_id` but returns the complete `HierarchicalChunk`.
pub async fn resolve_entry(store: &impl VectorStore, id: &str) -> Result<crate::HierarchicalChunk> {
    store
        .get_by_id_prefix(id)
        .await?
        .ok_or_else(|| crate::Error::not_found(format!("Entry '{}' not found", id)))
}

/// Parse a temporal string to a Unix epoch seconds timestamp.
///
/// Supported formats:
/// - Epoch seconds: `"1740000000"`
/// - ISO 8601 date: `"2026-02-20"`
/// - ISO 8601 datetime: `"2026-02-20T14:30:00Z"`, `"2026-02-20T14:30:00+01:00"`
/// - Relative duration (ago from now): `"15min"`, `"3h"`, `"7d"`, `"2w"`, `"1m"`
///   - `min` = minutes, `h` = hours, `d` = days, `w` = weeks, `m` = months (approx 30 days)
///
/// Returns `None` if the string cannot be parsed in any format.
pub fn parse_temporal(s: &str) -> Option<i64> {
    // Try epoch seconds first
    if let Ok(epoch) = s.parse::<i64>() {
        return Some(epoch);
    }
    // Try ISO 8601 date or datetime (must start with YYYY-MM-DD)
    if s.len() >= 10 && s.as_bytes()[4] == b'-' && s.as_bytes()[7] == b'-' {
        let year: i32 = s[0..4].parse().ok()?;
        let month: u32 = s[5..7].parse().ok()?;
        let day: u32 = s[8..10].parse().ok()?;
        let days = days_since_epoch(year, month, day)?;
        let base = days * 86400;
        // Date-only: "2026-02-20"
        if s.len() == 10 {
            return Some(base);
        }
        // Datetime: expect 'T' separator at position 10
        if s.as_bytes().get(10) != Some(&b'T') {
            return None;
        }
        let time_part = &s[11..];
        // Extract HH:MM:SS (ignore sub-second precision)
        if time_part.len() < 8 {
            return None;
        }
        let hour: i64 = time_part[0..2].parse().ok()?;
        let min: i64 = time_part[3..5].parse().ok()?;
        let sec: i64 = time_part[6..8].parse().ok()?;
        let time_secs = hour * 3600 + min * 60 + sec;
        // Parse timezone offset
        let tz_offset = if time_part.len() == 8 {
            // No timezone specified — assume UTC
            0
        } else {
            let tz = &time_part[8..];
            if tz == "Z" {
                0
            } else if tz.len() >= 5 && (tz.starts_with('+') || tz.starts_with('-')) {
                let sign: i64 = if tz.starts_with('+') { 1 } else { -1 };
                let tz_h: i64 = tz[1..3].parse().ok()?;
                let tz_m: i64 = tz[4..6].parse().ok()?;
                sign * (tz_h * 3600 + tz_m * 60)
            } else {
                return None;
            }
        };
        return Some(base + time_secs - tz_offset);
    }
    // Try relative duration (e.g. "7d", "2w", "1m", "3h")
    parse_relative_duration(s)
}

/// Parse a relative duration string like "7d", "2w", "1m", "3h", "15min" to an epoch
/// timestamp representing that duration ago from now.
///
/// Suffixes: `min` = minutes, `h` = hours, `d` = days, `w` = weeks, `m` = months (~30 days)
fn parse_relative_duration(s: &str) -> Option<i64> {
    if s.is_empty() {
        return None;
    }
    // Try multi-char suffix first ("min")
    let (count, seconds_per_unit) = if let Some(num_part) = s.strip_suffix("min") {
        (num_part.parse::<i64>().ok()?, 60)
    } else {
        let (num_part, unit) = s.split_at(s.len() - 1);
        let c = num_part.parse::<i64>().ok()?;
        let spu = match unit {
            "h" => 3600,
            "d" => 86400,
            "w" => 7 * 86400,
            "m" => 30 * 86400,
            _ => return None,
        };
        (c, spu)
    };
    if count < 0 {
        return None;
    }
    let now = crate::chunk::now_epoch_secs();
    Some(now - count * seconds_per_unit)
}

/// Convert a calendar date to days since Unix epoch (1970-01-01).
/// Uses the algorithm from http://howardhinnant.github.io/date_algorithms.html
fn days_since_epoch(year: i32, month: u32, day: u32) -> Option<i64> {
    let y = if month <= 2 { year - 1 } else { year } as i64;
    let m = if month <= 2 { month + 9 } else { month - 3 } as i64;
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * m + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146_097 + doe - 719_468)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VectorStore;

    use crate::test_helpers::make_test_chunk;

    #[test]
    fn test_parse_temporal_epoch() {
        assert_eq!(parse_temporal("1740000000"), Some(1740000000));
        assert_eq!(parse_temporal("0"), Some(0));
    }

    #[test]
    fn test_parse_temporal_iso_date() {
        // 1970-01-01 = epoch 0
        assert_eq!(parse_temporal("1970-01-01"), Some(0));
        // 2026-02-20 should produce a reasonable epoch
        let result = parse_temporal("2026-02-20");
        assert!(result.is_some());
        let epoch = result.unwrap();
        // Should be around 2026 (> 2025-01-01 = ~1735689600)
        assert!(epoch > 1_735_689_600);
    }

    #[test]
    fn test_parse_temporal_iso_datetime() {
        // UTC with Z suffix: 1970-01-01T01:00:00Z = 3600
        assert_eq!(parse_temporal("1970-01-01T01:00:00Z"), Some(3600));
        // With positive timezone offset: 1970-01-01T02:00:00+01:00 = 3600 (2h local - 1h offset)
        assert_eq!(parse_temporal("1970-01-01T02:00:00+01:00"), Some(3600));
        // With negative timezone offset: 1970-01-01T00:00:00-01:00 = 3600 (0h local + 1h offset)
        assert_eq!(parse_temporal("1970-01-01T00:00:00-01:00"), Some(3600));
        // No timezone = assumed UTC
        assert_eq!(parse_temporal("1970-01-01T00:30:00"), Some(1800));
        // 2026 datetime should produce reasonable epoch
        let result = parse_temporal("2026-02-20T14:30:00Z").unwrap();
        assert!(result > 1_735_689_600);
        // Invalid: T but too short time part
        assert_eq!(parse_temporal("2026-02-20T14"), None);
    }

    #[test]
    fn test_parse_temporal_invalid() {
        assert_eq!(parse_temporal("not-a-date"), None);
        // Malformed date formats (not YYYY-MM-DD and not a valid integer) return None
        assert_eq!(parse_temporal("2026/02/20"), None);
        assert_eq!(parse_temporal("Feb 20 2026"), None);
        assert_eq!(parse_temporal(""), None);
        // "20260220" is a valid integer epoch, not invalid
        assert!(parse_temporal("20260220").is_some());
    }

    #[test]
    fn test_parse_temporal_relative_duration() {
        let now = crate::chunk::now_epoch_secs();
        let day = parse_temporal("1d").unwrap();
        assert!((day - (now - 86400)).abs() <= 2, "1d should be ~1 day ago");

        let week = parse_temporal("7d").unwrap();
        assert!((week - (now - 7 * 86400)).abs() <= 2);

        let month = parse_temporal("1m").unwrap();
        assert!((month - (now - 30 * 86400)).abs() <= 2);

        let hour = parse_temporal("3h").unwrap();
        assert!((hour - (now - 3 * 3600)).abs() <= 2);

        let minutes = parse_temporal("15min").unwrap();
        assert!(
            (minutes - (now - 15 * 60)).abs() <= 2,
            "15min should be ~15 minutes ago"
        );

        // Unknown suffix → None
        assert_eq!(parse_temporal("5x"), None);
        // Negative → None
        assert_eq!(parse_temporal("-1d"), None);
    }

    #[tokio::test]
    async fn test_resolve_id_exact_match() {
        let dir = tempfile::tempdir().unwrap();
        let store = crate::store::StoreBackend::open(dir.path(), 384, false)
            .await
            .unwrap();
        let store = Arc::new(store);

        store
            .insert_chunks(vec![make_test_chunk("abcdef1234567890", "content")])
            .await
            .unwrap();

        let result = resolve_id(&store, "abcdef1234567890").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "abcdef1234567890");
    }

    #[tokio::test]
    async fn test_resolve_id_prefix_match() {
        let dir = tempfile::tempdir().unwrap();
        let store = crate::store::StoreBackend::open(dir.path(), 384, false)
            .await
            .unwrap();
        let store = Arc::new(store);

        store
            .insert_chunks(vec![make_test_chunk("abcdef1234567890", "content")])
            .await
            .unwrap();

        let result = resolve_id(&store, "abcdef1").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "abcdef1234567890");
    }

    #[tokio::test]
    async fn test_resolve_id_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let store = crate::store::StoreBackend::open(dir.path(), 384, false)
            .await
            .unwrap();
        let store = Arc::new(store);

        let result = resolve_id(&store, "nonexistent").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("nonexistent"));
    }

    #[tokio::test]
    async fn test_resolve_entry_returns_full_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let store = crate::store::StoreBackend::open(dir.path(), 384, false)
            .await
            .unwrap();

        let chunk = crate::HierarchicalChunk {
            heading: Some("Test Heading".to_string()),
            ..make_test_chunk("abcdef1234567890", "test content")
        };
        store.insert_chunks(vec![chunk]).await.unwrap();

        let result = resolve_entry(&store, "abcdef1").await;
        assert!(result.is_ok());
        let entry = result.unwrap();
        assert_eq!(entry.id, "abcdef1234567890");
        assert_eq!(entry.content, "test content");
        assert_eq!(entry.heading.as_deref(), Some("Test Heading"));
    }
}
