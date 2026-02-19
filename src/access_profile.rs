//! RRD-style access tracking for memory aging.
//!
//! Fixed-size time-window buckets (hour/day/week/month/year/total) track access
//! patterns. Finer buckets roll into coarser ones automatically. Recency-weighted
//! relevancy scoring supports temporal search boosting.

use serde::{Deserialize, Serialize};

// --- Time constants for RRD bucket boundaries ---

const SECS_PER_HOUR: i64 = 3_600;
const SECS_PER_DAY: i64 = 86_400;
const SECS_PER_WEEK: i64 = 604_800;
const SECS_PER_MONTH: i64 = 2_592_000; // 30 days
const SECS_PER_YEAR: i64 = 31_536_000; // 365 days

/// RRD-style access tracking with fixed time-window buckets.
///
/// Inspired by RRDtool: finer buckets roll into coarser ones on a schedule.
/// Each bucket tracks the number of accesses within its time window.
///
/// Buckets: hour | day | week | month | year | total
///
/// Layout (30 bytes, padded to 32):
///   created_at:  i64  (8 bytes)
///   last_rolled: i64  (8 bytes)
///   hour:        u16  (2 bytes)
///   day:         u16  (2 bytes)
///   week:        u16  (2 bytes)
///   month:       u16  (2 bytes)
///   year:        u16  (2 bytes)
///   total:       u32  (4 bytes)
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecencyWindow {
    Day,
    Week,
    Month,
}

impl RecencyWindow {
    /// Parse from string (e.g. "24h", "7d", "30d").
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "24h" | "day" => Some(Self::Day),
            "7d" | "week" => Some(Self::Week),
            "30d" | "month" => Some(Self::Month),
            _ => None,
        }
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

        if elapsed >= SECS_PER_YEAR {
            self.year = 0;
            self.month = 0;
            self.week = 0;
            self.day = 0;
            self.hour = 0;
            self.last_rolled = now;
        } else if elapsed >= SECS_PER_MONTH {
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
            self.week = self
                .week
                .saturating_add(self.day)
                .saturating_add(self.hour);
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
    pub fn relevancy_score(&self, recency_window: Option<RecencyWindow>) -> f32 {
        let weights = match recency_window {
            None => RecencyWeights::balanced(),
            Some(RecencyWindow::Day) => RecencyWeights::day(),
            Some(RecencyWindow::Week) => RecencyWeights::week(),
            Some(RecencyWindow::Month) => RecencyWeights::month(),
        };

        let raw = (self.hour as f32) * weights.w_hour
            + (self.day as f32) * weights.w_day
            + (self.week as f32) * weights.w_week
            + (self.month as f32) * weights.w_month
            + (self.year as f32) * weights.w_year
            + (self.total as f32) * weights.w_total;

        (raw / weights.scale).tanh()
    }

    /// Seconds since creation.
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
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.record_access_at(base + 10);
        profile.record_access_at(base + 20);
        profile.record_access_at(base + 30);
        assert_eq!(profile.hour, 3);
        assert_eq!(profile.day, 0);
        assert_eq!(profile.total, 3);
    }

    #[test]
    fn test_roll_up_hour_to_day() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.record_access_at(base + 10);
        profile.record_access_at(base + 20);
        profile.record_access_at(base + 30);

        profile.record_access_at(base + SECS_PER_HOUR + 100);
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
    fn test_roll_up_beyond_year() {
        let base = 1_000_000;
        let mut profile = AccessProfile::with_created_at(base);
        profile.record_access_at(base + 10);
        profile.record_access_at(base + 20);

        profile.record_access_at(base + SECS_PER_YEAR + 100);
        assert_eq!(profile.hour, 1);
        assert_eq!(profile.day, 0);
        assert_eq!(profile.week, 0);
        assert_eq!(profile.month, 0);
        assert_eq!(profile.year, 0);
        assert_eq!(profile.total, 3);
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
}
