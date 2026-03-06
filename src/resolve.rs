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
/// - Relative duration (ago from now): `"7d"`, `"2w"`, `"1m"`, `"3h"`
///   - `d` = days, `w` = weeks, `m` = months (approx 30 days), `h` = hours
///
/// Returns `None` if the string cannot be parsed in any format.
pub fn parse_temporal(s: &str) -> Option<i64> {
    // Try epoch seconds first
    if let Ok(epoch) = s.parse::<i64>() {
        return Some(epoch);
    }
    // Try ISO 8601 date (YYYY-MM-DD)
    if s.len() == 10 && s.as_bytes()[4] == b'-' && s.as_bytes()[7] == b'-' {
        let year: i32 = s[0..4].parse().ok()?;
        let month: u32 = s[5..7].parse().ok()?;
        let day: u32 = s[8..10].parse().ok()?;
        let days = days_since_epoch(year, month, day)?;
        return Some(days * 86400);
    }
    // Try relative duration (e.g. "7d", "2w", "1m", "3h")
    parse_relative_duration(s)
}

/// Parse a relative duration string like "7d", "2w", "1m", "3h" to an epoch
/// timestamp representing that duration ago from now.
///
/// Suffixes: `d` = days, `w` = weeks, `m` = months (~30 days), `h` = hours
fn parse_relative_duration(s: &str) -> Option<i64> {
    if s.is_empty() {
        return None;
    }
    let (num_part, unit) = s.split_at(s.len() - 1);
    let count: i64 = num_part.parse().ok()?;
    if count < 0 {
        return None;
    }
    let seconds_ago = match unit {
        "h" => count * 3600,
        "d" => count * 86400,
        "w" => count * 7 * 86400,
        "m" => count * 30 * 86400,
        _ => return None,
    };
    let now = crate::chunk::now_epoch_secs();
    Some(now - seconds_ago)
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
    Some(era * 146097 + doe - 719468)
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
