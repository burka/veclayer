//! RRD-style access tracking for memory aging.
//!
//! Fixed-size time-window buckets (hour/day/week/month/year/total) track access
//! patterns. Finer buckets roll into coarser ones automatically. Recency-weighted
//! relevancy scoring supports temporal search boosting.

use serde::{Deserialize, Serialize};

// Time constants (i64 for timestamp arithmetic). Values are identical to
// util::SECS_PER_* — kept here to avoid forcing AccessProfile fields to u64.
mod time {
    pub const SECS_PER_HOUR: i64 = 3_600;
    pub const SECS_PER_DAY: i64 = 86_400;
    pub const SECS_PER_WEEK: i64 = 604_800;
    pub const SECS_PER_MONTH: i64 = 2_592_000;
    #[cfg(test)]
    pub const SECS_PER_YEAR: i64 = 31_536_000;
}
#[cfg(test)]
use time::SECS_PER_YEAR;
use time::{SECS_PER_DAY, SECS_PER_HOUR, SECS_PER_MONTH, SECS_PER_WEEK};

/// RRD-style access tracking with fixed time-window buckets.
///
/// Inspired by RRDtool: finer buckets roll into coarser ones on a schedule.
/// Each bucket tracks the number of accesses within its time window.
///
/// Buckets: `hour` | `day` | `week` | `month` | `year` | `total`
///
/// Layout (30 bytes, padded to 32):
///   `created_at`:  `i64`  (8 bytes)
///   `last_rolled`: `i64`  (8 bytes)
///   `hour`:        `u16`  (2 bytes)
///   `day`:         `u16`  (2 bytes)
///   `week`:        `u16`  (2 bytes)
///   `month`:       `u16`  (2 bytes)
///   `year`:        `u16`  (2 bytes)
///   `total`:       `u32`  (4 bytes)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccessProfile {
    pub created_at: i64,
    pub last_rolled: i64,
    pub hour: u16,
    pub day: u16,
    pub week: u16,
    pub month: u16,
    pub year: u16,
    pub total: u32,
}

/// Time window for recency-weighted search.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecencyWindow {
    Day,
    Week,
    Month,
}

impl RecencyWindow {
    /// Parse from string (e.g. "24h", "7d", "30d").
    #[must_use]
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "24h" | "day" => Some(Self::Day),
            "7d" | "week" => Some(Self::Week),
            "30d" | "month" => Some(Self::Month),
            _ => None,
        }
    }
}

impl std::str::FromStr for RecencyWindow {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str_opt(s).ok_or_else(|| {
            format!(
                "Unknown recency window '{s}'. Valid values: 24h (or day), 7d (or week), 30d (or month)"
            )
        })
    }
}

/// Internal weight configuration for relevancy scoring.
struct RecencyWeights {
    w_hour: f32,
    w_day: f32,
    w_week: f32,
    w_month: f32,
    w_year: f32,
    w_total: f32,
    scale: f32,
}

impl RecencyWeights {
    fn balanced() -> Self {
        Self {
            w_hour: 8.0,
            w_day: 4.0,
            w_week: 2.0,
            w_month: 1.0,
            w_year: 0.3,
            w_total: 0.1,
            scale: 10.0,
        }
    }

    fn day() -> Self {
        Self {
            w_hour: 10.0,
            w_day: 8.0,
            w_week: 0.5,
            w_month: 0.1,
            w_year: 0.0,
            w_total: 0.0,
            scale: 10.0,
        }
    }

    fn week() -> Self {
        Self {
            w_hour: 6.0,
            w_day: 5.0,
            w_week: 4.0,
            w_month: 0.5,
            w_year: 0.1,
            w_total: 0.05,
            scale: 10.0,
        }
    }

    fn month() -> Self {
        Self {
            w_hour: 4.0,
            w_day: 3.0,
            w_week: 2.5,
            w_month: 2.0,
            w_year: 0.5,
            w_total: 0.1,
            scale: 10.0,
        }
    }
}

impl AccessProfile {
    pub fn new() -> Self {
        let now = now_epoch_secs();
        Self {
            created_at: now,
            last_rolled: now,
            hour: 0,
            day: 0,
            week: 0,
            month: 0,
            year: 0,
            total: 0,
        }
    }

    /// Create with a specific creation time (for testing/migration).
    pub fn with_created_at(created_at: i64) -> Self {
        Self {
            created_at,
            last_rolled: created_at,
            hour: 0,
            day: 0,
            week: 0,
            month: 0,
            year: 0,
            total: 0,
        }
    }

    /// Roll stale values from finer buckets into coarser ones.
    /// Idempotent: calling multiple times with the same `now` is safe.
    pub fn roll_up(&mut self, now: i64) {
        let elapsed = now - self.last_rolled;
        if elapsed <= 0 {
            return;
        }

        // Also handles year+ elapsed: there is no decaying bucket coarser than
        // `year`, so everything finer accumulates into it (the year-zeroing
        // branch was removed — it dropped data).
        if elapsed >= SECS_PER_MONTH {
            self.year = self
                .year
                .saturating_add(self.month)
                .saturating_add(self.week)
                .saturating_add(self.day)
                .saturating_add(self.hour);
            self.month = 0;
            self.week = 0;
            self.day = 0;
            self.hour = 0;
            self.last_rolled = now;
        } else if elapsed >= SECS_PER_WEEK {
            self.month = self
                .month
                .saturating_add(self.week)
                .saturating_add(self.day)
                .saturating_add(self.hour);
            self.week = 0;
            self.day = 0;
            self.hour = 0;
            self.last_rolled = now;
        } else if elapsed >= SECS_PER_DAY {
            self.week = self.week.saturating_add(self.day).saturating_add(self.hour);
            self.day = 0;
            self.hour = 0;
            self.last_rolled = now;
        } else if elapsed >= SECS_PER_HOUR {
            self.day = self.day.saturating_add(self.hour);
            self.hour = 0;
            self.last_rolled = now;
        }
    }

    /// Record a single access at the given time.
    pub fn record_access_at(&mut self, now: i64) {
        self.roll_up(now);
        self.hour = self.hour.saturating_add(1);
        self.total = self.total.saturating_add(1);
    }

    /// Record a single access using the current wall clock.
    pub fn record_access(&mut self) {
        self.record_access_at(now_epoch_secs());
    }

    /// Temporal relevancy score in [0.0, 1.0].
    #[must_use]
    pub fn relevancy_score(&self, recency_window: Option<RecencyWindow>) -> f32 {
        let weights = match recency_window {
            None => RecencyWeights::balanced(),
            Some(RecencyWindow::Day) => RecencyWeights::day(),
            Some(RecencyWindow::Week) => RecencyWeights::week(),
            Some(RecencyWindow::Month) => RecencyWeights::month(),
        };

        let raw = f32::from(self.hour) * weights.w_hour
            + f32::from(self.day) * weights.w_day
            + f32::from(self.week) * weights.w_week
            + f32::from(self.month) * weights.w_month
            + f32::from(self.year) * weights.w_year
            + (self.total as f32) * weights.w_total;

        (raw / weights.scale).tanh()
    }

    /// Seconds since creation.
    #[must_use]
    pub fn age_seconds(&self) -> i64 {
        now_epoch_secs() - self.created_at
    }
}

impl Default for AccessProfile {
    fn default() -> Self {
        Self::new()
    }
}

pub fn now_epoch_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .try_into()
        .unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a profile with 3 recorded accesses at base+10, base+20, base+30.
    fn profile_with_three_accesses(base: i64) -> AccessProfile {
        let mut profile = AccessProfile::with_created_at(base);
        profile.record_access_at(base + 10);
        profile.record_access_at(base + 20);
        profile.record_access_at(base + 30);
        profile
    }

    #[test]
    fn test_access_profile_new() {
        let profile = AccessProfile::new();
        assert_eq!(profile.total, 0);
        assert_eq!(profile.hour, 0);
        assert!(profile.created_at > 0);
    }

    #[test]
    fn test_access_profile_record_access() {
        let mut profile = AccessProfile::new();
        profile.record_access();
        assert_eq!(profile.total, 1);
        assert_eq!(profile.hour, 1);

        profile.record_access();
        assert_eq!(profile.total, 2);
        assert_eq!(profile.hour, 2);
    }

    #[test]
    fn test_roll_up_within_hour() {
        let profile = profile_with_three_accesses(1_000_000);
        assert_eq!(profile.hour, 3);
        assert_eq!(profile.day, 0);
        assert_eq!(profile.total, 3);
    }

    #[test]
    fn test_roll_up_hour_to_day() {
        let mut profile = profile_with_three_accesses(1_000_000);
        profile.record_access_at(1_000_000 + SECS_PER_HOUR + 100);
        assert_eq!(profile.hour, 1);
        assert_eq!(profile.day, 3);
        assert_eq!(profile.total, 4);
    }

    #[test]
    fn test_roll_up_day_to_week() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.record_access_at(base + 10);
        profile.record_access_at(base + 20);

        profile.record_access_at(base + SECS_PER_DAY + 100);
        assert_eq!(profile.hour, 1);
        assert_eq!(profile.day, 0);
        assert_eq!(profile.week, 2);
        assert_eq!(profile.total, 3);
    }

    #[test]
    fn test_roll_up_year_preserves_accumulated_count() {
        // RED→GREEN: year bucket must accumulate finer buckets, not be zeroed.
        // Profile: year=10, hour=5, total=15; all other finer buckets zero.
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.year = 10;
        profile.hour = 5;
        profile.total = 15;

        let now = base + SECS_PER_YEAR + 1;
        profile.roll_up(now);

        // year must accumulate hour (and all finer buckets: day/week/month are 0)
        assert_eq!(
            profile.year, 15,
            "year must absorb finer buckets, not be zeroed"
        );
        assert_eq!(profile.hour, 0);
        assert_eq!(profile.day, 0);
        assert_eq!(profile.week, 0);
        assert_eq!(profile.month, 0);
        assert_eq!(profile.total, 15, "total must never be touched by roll_up");

        // Idempotency: a second roll_up with the same `now` leaves everything unchanged.
        let snapshot = profile.clone();
        profile.roll_up(now);
        assert_eq!(profile, snapshot, "roll_up must be idempotent");
    }

    #[test]
    fn test_roll_up_beyond_year() {
        // Record two accesses, then trigger a year-level roll-up.
        // The two old accesses must be preserved in `year`; the new one lands in `hour`.
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.record_access_at(base + 10);
        profile.record_access_at(base + 20);

        profile.record_access_at(base + SECS_PER_YEAR + 100);
        assert_eq!(profile.hour, 1);
        assert_eq!(profile.day, 0);
        assert_eq!(profile.week, 0);
        assert_eq!(profile.month, 0);
        assert_eq!(profile.year, 2, "the 2 prior accesses must roll into year");
        assert_eq!(profile.total, 3);

        // Idempotency: second roll_up at the same timestamp changes nothing.
        let now = base + SECS_PER_YEAR + 100;
        let snapshot = profile.clone();
        profile.roll_up(now);
        assert_eq!(profile, snapshot, "roll_up must be idempotent");
    }

    #[test]
    fn test_roll_up_idempotent() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.record_access_at(base + 10);

        let now = base + SECS_PER_HOUR + 100;
        profile.roll_up(now);
        let snapshot = profile.clone();
        profile.roll_up(now);
        assert_eq!(profile, snapshot);
    }

    #[test]
    fn test_relevancy_score_no_accesses() {
        let profile = AccessProfile::with_created_at(1_000_000);
        assert_eq!(profile.relevancy_score(None), 0.0);
    }

    #[test]
    fn test_relevancy_score_recent_beats_old() {
        let base = 1_000_000;

        let mut recent = AccessProfile::with_created_at(base);
        recent.record_access_at(base + 10);
        recent.record_access_at(base + 20);

        let mut old = AccessProfile::with_created_at(base);
        old.year = 2;
        old.total = 2;

        assert!(recent.relevancy_score(None) > old.relevancy_score(None));
    }

    #[test]
    fn test_recency_window_from_str() {
        assert_eq!(RecencyWindow::from_str_opt("24h"), Some(RecencyWindow::Day));
        assert_eq!(RecencyWindow::from_str_opt("7d"), Some(RecencyWindow::Week));
        assert_eq!(
            RecencyWindow::from_str_opt("30d"),
            Some(RecencyWindow::Month)
        );
        assert_eq!(RecencyWindow::from_str_opt("invalid"), None);
    }

    #[test]
    fn test_saturating_add_prevents_overflow() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.hour = u16::MAX;
        profile.total = u32::MAX;

        profile.record_access_at(base + 10);
        assert_eq!(profile.hour, u16::MAX);
        assert_eq!(profile.total, u32::MAX);
    }

    // --- with_created_at ---

    #[test]
    fn test_with_created_at_sets_both_timestamps() {
        let ts = 9_999_999_i64;
        let profile = AccessProfile::with_created_at(ts);
        assert_eq!(profile.created_at, ts);
        assert_eq!(profile.last_rolled, ts);
        assert_eq!(profile.total, 0);
        assert_eq!(profile.hour, 0);
    }

    #[test]
    fn test_with_created_at_all_buckets_zero() {
        let profile = AccessProfile::with_created_at(1_000_000);
        assert_eq!(profile.day, 0);
        assert_eq!(profile.week, 0);
        assert_eq!(profile.month, 0);
        assert_eq!(profile.year, 0);
    }

    // --- age_seconds ---

    #[test]
    fn test_age_seconds_non_negative_for_recent_creation() {
        let profile = AccessProfile::new();
        assert!(profile.age_seconds() >= 0);
    }

    #[test]
    fn test_age_seconds_old_entry() {
        // created far in the past
        let past = 1_000_000_i64;
        let profile = AccessProfile::with_created_at(past);
        // age should be at least (now - past) seconds; rough lower-bound check
        assert!(profile.age_seconds() > 1_000_000);
    }

    // --- RecencyWindow aliases ---

    #[test]
    fn test_recency_window_alias_day() {
        assert_eq!(RecencyWindow::from_str_opt("day"), Some(RecencyWindow::Day));
    }

    #[test]
    fn test_recency_window_alias_week() {
        assert_eq!(
            RecencyWindow::from_str_opt("week"),
            Some(RecencyWindow::Week)
        );
    }

    #[test]
    fn test_recency_window_alias_month() {
        assert_eq!(
            RecencyWindow::from_str_opt("month"),
            Some(RecencyWindow::Month)
        );
    }

    #[test]
    fn test_recency_window_empty_string_returns_none() {
        assert_eq!(RecencyWindow::from_str_opt(""), None);
    }

    #[test]
    fn test_recency_window_case_sensitive() {
        // Aliases are case-sensitive — "Day" should not match
        assert_eq!(RecencyWindow::from_str_opt("Day"), None);
        assert_eq!(RecencyWindow::from_str_opt("WEEK"), None);
    }

    // --- saturation of buckets via roll_up ---

    #[test]
    fn test_roll_up_week_to_month() {
        // After a week-level roll-up, prior accesses (hour/day/week) accumulate into month.
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.record_access_at(base + 10);
        profile.record_access_at(base + 20);
        // elapsed >= SECS_PER_WEEK → month accumulates week+day+hour; those buckets zeroed
        profile.record_access_at(base + SECS_PER_WEEK + 100);
        // The 2 old accesses now in month; the new one is in hour
        assert_eq!(profile.month, 2);
        assert_eq!(profile.week, 0);
        assert_eq!(profile.hour, 1);
        // Advance a full month past the current last_rolled to trigger the month branch.
        // last_rolled = base + SECS_PER_WEEK + 100; add SECS_PER_MONTH to exceed it.
        let t2 = base + SECS_PER_WEEK + 100 + SECS_PER_MONTH + 1;
        profile.record_access_at(t2);
        // month branch: year += month+week+day+hour; then month/week/day/hour zeroed
        assert_eq!(profile.month, 0);
        assert_eq!(profile.week, 0);
        assert!(profile.year > 0); // 2 + 1 = 3 rolled into year
    }

    #[test]
    fn test_roll_up_saturation_at_u16_max() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        // Fill hour and day near their max
        profile.hour = u16::MAX;
        profile.day = u16::MAX;

        // Trigger a day-level roll-up: week = week.saturating_add(day).saturating_add(hour)
        profile.roll_up(base + SECS_PER_DAY + 100);
        assert_eq!(profile.week, u16::MAX); // saturated
        assert_eq!(profile.hour, 0);
        assert_eq!(profile.day, 0);
    }

    // --- rollup semantics ---

    #[test]
    fn test_roll_up_does_not_change_last_rolled_if_elapsed_zero() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.roll_up(base); // elapsed = 0
        assert_eq!(profile.last_rolled, base);
    }

    #[test]
    fn test_roll_up_updates_last_rolled() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        let later = base + SECS_PER_HOUR + 1;
        profile.roll_up(later);
        assert_eq!(profile.last_rolled, later);
    }

    #[test]
    fn test_roll_up_negative_elapsed_is_noop() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.hour = 5;
        profile.roll_up(base - 100); // elapsed is negative
        assert_eq!(profile.hour, 5); // unchanged
        assert_eq!(profile.last_rolled, base);
    }

    // --- relevancy scoring per window ---

    #[test]
    fn test_relevancy_score_day_window_favours_hour_bucket() {
        let base = 1_000_000;
        // A profile with only hour accesses should score higher for Day window
        // vs a profile with only year accesses
        let mut recent = AccessProfile::with_created_at(base);
        recent.hour = 5;
        recent.total = 5;

        let mut old = AccessProfile::with_created_at(base);
        old.year = 5;
        old.total = 5;

        assert!(
            recent.relevancy_score(Some(RecencyWindow::Day))
                > old.relevancy_score(Some(RecencyWindow::Day))
        );
    }

    #[test]
    fn test_relevancy_score_week_window() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.week = 3;
        profile.total = 3;
        let score = profile.relevancy_score(Some(RecencyWindow::Week));
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_relevancy_score_month_window() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.month = 3;
        profile.total = 3;
        let score = profile.relevancy_score(Some(RecencyWindow::Month));
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_relevancy_score_in_zero_to_one_range() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.hour = u16::MAX;
        profile.day = u16::MAX;
        profile.week = u16::MAX;
        profile.month = u16::MAX;
        profile.year = u16::MAX;
        profile.total = u32::MAX;
        // tanh(very_large) approaches 1.0 but never exceeds it
        let score = profile.relevancy_score(None);
        assert!(score >= 0.0);
        assert!(score <= 1.0);
    }

    // --- serde ---

    #[test]
    fn test_access_profile_serde_roundtrip() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.hour = 10;
        profile.day = 20;
        profile.week = 30;
        profile.month = 40;
        profile.year = 50;
        profile.total = 150;

        let json = serde_json::to_string(&profile).unwrap();
        let restored: AccessProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(profile, restored);
    }

    #[test]
    fn test_access_profile_default_equals_new_structurally() {
        // Default delegates to new(); both should have 0-valued buckets.
        let via_default = AccessProfile::default();
        assert_eq!(via_default.hour, 0);
        assert_eq!(via_default.day, 0);
        assert_eq!(via_default.week, 0);
        assert_eq!(via_default.month, 0);
        assert_eq!(via_default.year, 0);
        assert_eq!(via_default.total, 0);
    }

    #[test]
    fn test_recency_window_parse_valid() {
        for s in ["24h", "day", "7d", "week", "30d", "month"] {
            assert!(
                s.parse::<RecencyWindow>().is_ok(),
                "expected '{s}' to parse successfully"
            );
        }
    }

    #[test]
    fn test_recency_window_parse_invalid() {
        for s in ["invalid", "48h", ""] {
            let err = s.parse::<RecencyWindow>().unwrap_err();
            assert!(
                err.contains("Unknown recency window"),
                "expected 'Unknown recency window' in error for '{s}', got: {err}"
            );
        }
    }
}
