//! Salience scoring: how "important" an entry is beyond access frequency.
//!
//! Salience combines multiple signals:
//! - **Interaction density**: Access-Profile relevancy (how actively used)
//! - **Perspective spread**: How many perspectives reference this entry
//! - **Revision activity**: How many relations (superseded, versioned, summarized)
//!
//! The final salience score is in [0.0, 1.0] and can be used in ranking
//! alongside semantic similarity and recency.

use crate::HierarchicalChunk;

/// Weights for the salience sub-components.
#[derive(Debug, Clone)]
pub struct SalienceWeights {
    /// Weight for interaction density (access-profile relevancy).
    pub w_interaction: f32,
    /// Weight for perspective spread (how many perspectives).
    pub w_perspective: f32,
    /// Weight for revision activity (relation count).
    pub w_revision: f32,
}

impl Default for SalienceWeights {
    fn default() -> Self {
        Self {
            w_interaction: 0.5,
            w_perspective: 0.25,
            w_revision: 0.25,
        }
    }
}

/// Computed salience for a single entry.
#[derive(Debug, Clone)]
pub struct SalienceScore {
    /// Interaction density component [0, 1].
    pub interaction: f32,
    /// Perspective spread component [0, 1].
    pub perspective: f32,
    /// Revision activity component [0, 1].
    pub revision: f32,
    /// Weighted composite score [0, 1].
    pub composite: f32,
}

/// Maximum perspectives we normalize against (7 defaults + headroom).
const MAX_PERSPECTIVES: f32 = 8.0;

/// Relation count where the revision signal saturates.
const REVISION_SATURATION: f32 = 5.0;

/// Maximum number of stale chunks fetched per `apply_aging` pass.
///
/// Bounds the per-run work to a predictable batch so that a very large store
/// with many stale entries does not cause an unbounded memory or latency spike.
/// Aging is idempotent — subsequent runs will catch whatever this pass misses.
pub const MAX_AGING_BATCH_SIZE: usize = 500;

/// Compute the salience score for a chunk.
pub fn compute(chunk: &HierarchicalChunk, weights: &SalienceWeights) -> SalienceScore {
    let interaction = chunk.access_profile.relevancy_score(None);

    let perspective = (chunk.perspectives.len() as f32 / MAX_PERSPECTIVES).min(1.0);

    let revision = (chunk.relations.len() as f32 / REVISION_SATURATION).tanh();

    let mut composite = interaction * weights.w_interaction
        + perspective * weights.w_perspective
        + revision * weights.w_revision;

    // Consolidated entries (those that have been summarized) are inherently important.
    let has_summary = chunk
        .relations
        .iter()
        .any(|r| r.kind == crate::chunk::relation::SUMMARIZED_BY);
    if has_summary {
        composite += 0.1;
    }

    // Impressions are modulated by their strength (default 1.0 = full weight).
    // Clamp to the documented [0.0, 1.0] range so a caller supplying a negative
    // value does not invert the composite (negative boost would be nonsensical).
    if chunk.entry_type == crate::chunk::EntryType::Impression {
        composite *= chunk.impression_strength.clamp(0.0, 1.0);
    }

    // The composite is contractually in [0, 1]; the summary bonus and the
    // impression-strength multiplier can otherwise push it outside that range.
    let composite = composite.clamp(0.0, 1.0);

    SalienceScore {
        interaction,
        perspective,
        revision,
        composite,
    }
}

/// Compute salience for a batch of chunks.
pub fn compute_batch(
    chunks: &[HierarchicalChunk],
    weights: &SalienceWeights,
) -> Vec<SalienceScore> {
    chunks.iter().map(|c| compute(c, weights)).collect()
}

/// Find the top N most salient chunks from a slice.
/// Returns (chunk_index, SalienceScore) pairs, sorted by composite descending.
pub fn top_salient(
    chunks: &[HierarchicalChunk],
    weights: &SalienceWeights,
    limit: usize,
) -> Vec<(usize, SalienceScore)> {
    let mut scored: Vec<(usize, SalienceScore)> = chunks
        .iter()
        .enumerate()
        .map(|(i, c)| (i, compute(c, weights)))
        .collect();

    crate::chunk::sort_f32_desc(&mut scored, |s| s.1.composite);

    scored.truncate(limit);
    scored
}

/// Determine if a chunk is an archive candidate based on salience.
/// A chunk is an archive candidate if:
/// - Its salience is below the threshold
/// - Its visibility is in the degradable set
pub fn is_archive_candidate(
    chunk: &HierarchicalChunk,
    weights: &SalienceWeights,
    salience_threshold: f32,
    degradable_visibilities: &[String],
) -> bool {
    if !degradable_visibilities.contains(&chunk.visibility) {
        return false;
    }
    let score = compute(chunk, weights);
    score.composite < salience_threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChunkLevel, ChunkRelation};

    fn test_chunk(content: &str) -> HierarchicalChunk {
        HierarchicalChunk::new(
            content.to_string(),
            ChunkLevel::CONTENT,
            None,
            String::new(),
            "test.md".to_string(),
        )
    }

    #[test]
    fn test_salience_empty_chunk() {
        let chunk = test_chunk("empty");
        let score = compute(&chunk, &SalienceWeights::default());
        assert_eq!(score.interaction, 0.0);
        assert_eq!(score.perspective, 0.0);
        assert_eq!(score.revision, 0.0);
        assert_eq!(score.composite, 0.0);
    }

    #[test]
    fn test_salience_with_accesses() {
        let mut chunk = test_chunk("accessed");
        chunk.access_profile.record_access();
        chunk.access_profile.record_access();
        chunk.access_profile.record_access();
        let score = compute(&chunk, &SalienceWeights::default());
        assert!(score.interaction > 0.0);
        assert!(score.composite > 0.0);
    }

    #[test]
    fn test_salience_with_perspectives() {
        let mut chunk = test_chunk("perspectives");
        chunk.perspectives = vec!["decisions".to_string(), "learnings".to_string()];
        let score = compute(&chunk, &SalienceWeights::default());
        assert!((score.perspective - 2.0 / MAX_PERSPECTIVES).abs() < 0.001);
        assert!(score.composite > 0.0);
    }

    #[test]
    fn test_salience_perspective_capped_at_one() {
        let mut chunk = test_chunk("many perspectives");
        chunk.perspectives = (0..20).map(|i| format!("p{}", i)).collect();
        let score = compute(&chunk, &SalienceWeights::default());
        assert_eq!(score.perspective, 1.0);
    }

    #[test]
    fn test_salience_with_relations() {
        let chunk = test_chunk("related")
            .with_relation(ChunkRelation::superseded_by("newer"))
            .with_relation(ChunkRelation::summarized_by("summary"));
        let score = compute(&chunk, &SalienceWeights::default());
        assert!(score.revision > 0.0);
        assert!(score.composite > 0.0);
    }

    #[test]
    fn test_salience_revision_saturates() {
        let mut chunk = test_chunk("many relations");
        for i in 0..20 {
            chunk
                .relations
                .push(ChunkRelation::related_to(format!("r{}", i)));
        }
        let score = compute(&chunk, &SalienceWeights::default());
        // tanh(20/5) ≈ 1.0
        assert!(score.revision > 0.99);
    }

    #[test]
    fn test_salience_composite_clamped_to_one() {
        // impression_strength amplifies the composite; a strength well above 1.0
        // must not push the composite past the documented [0, 1] range.
        let mut chunk = test_chunk("over-strength impression");
        chunk.entry_type = crate::chunk::EntryType::Impression;
        chunk.impression_strength = 20.0;
        chunk.perspectives = (0..20).map(|i| format!("p{i}")).collect();
        for i in 0..20 {
            chunk
                .relations
                .push(ChunkRelation::related_to(format!("r{i}")));
        }
        let score = compute(&chunk, &SalienceWeights::default());
        assert!(
            score.composite <= 1.0,
            "composite must stay within [0, 1], got {}",
            score.composite
        );
        assert!(score.composite >= 0.0);
    }

    #[test]
    fn test_salience_composite_weighted() {
        let weights = SalienceWeights {
            w_interaction: 1.0,
            w_perspective: 0.0,
            w_revision: 0.0,
        };
        let mut chunk = test_chunk("interaction only");
        chunk.access_profile.record_access();
        let score = compute(&chunk, &weights);
        assert_eq!(score.composite, score.interaction);
    }

    #[test]
    fn test_top_salient() {
        let chunks = vec![
            {
                let mut c = test_chunk("low");
                c.access_profile.total = 0;
                c
            },
            {
                let mut c = test_chunk("medium");
                c.perspectives = vec!["decisions".to_string()];
                c
            },
            {
                let mut c = test_chunk("high");
                c.access_profile.record_access();
                c.access_profile.record_access();
                c.perspectives = vec!["decisions".to_string(), "learnings".to_string()];
                c.relations.push(ChunkRelation::superseded_by("newer"));
                c
            },
        ];

        let top = top_salient(&chunks, &SalienceWeights::default(), 2);
        assert_eq!(top.len(), 2);
        // Highest salience first
        assert!(top[0].1.composite >= top[1].1.composite);
        // The "high" chunk (index 2) should be first
        assert_eq!(top[0].0, 2);
    }

    #[test]
    fn test_compute_batch() {
        let chunks = vec![test_chunk("a"), test_chunk("b")];
        let scores = compute_batch(&chunks, &SalienceWeights::default());
        assert_eq!(scores.len(), 2);
    }

    #[test]
    fn test_archive_candidate_low_salience() {
        let chunk = test_chunk("low salience");
        let degradable = vec!["normal".to_string()];
        assert!(is_archive_candidate(
            &chunk,
            &SalienceWeights::default(),
            0.1,
            &degradable
        ));
    }

    #[test]
    fn test_archive_candidate_high_salience() {
        let mut chunk = test_chunk("high salience");
        chunk.access_profile.record_access();
        chunk.access_profile.record_access();
        chunk.access_profile.record_access();
        chunk.perspectives = vec!["decisions".to_string()];
        let degradable = vec!["normal".to_string()];
        assert!(!is_archive_candidate(
            &chunk,
            &SalienceWeights::default(),
            0.05,
            &degradable
        ));
    }

    #[test]
    fn test_archive_candidate_wrong_visibility() {
        let chunk = test_chunk("always visible");
        let degradable = vec!["normal".to_string()];
        // chunk.visibility is "normal" by default, so it IS degradable
        assert!(is_archive_candidate(
            &chunk,
            &SalienceWeights::default(),
            0.1,
            &degradable
        ));

        // But "always" visibility is not in the degradable list
        let mut always_chunk = test_chunk("always");
        always_chunk.visibility = "always".to_string();
        assert!(!is_archive_candidate(
            &always_chunk,
            &SalienceWeights::default(),
            0.1,
            &degradable
        ));
    }

    #[test]
    fn test_default_weights_sum_to_one() {
        let w = SalienceWeights::default();
        let sum = w.w_interaction + w.w_perspective + w.w_revision;
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_impression_strength_modulates_composite() {
        let mut full = test_chunk("full strength impression");
        full.entry_type = crate::chunk::EntryType::Impression;
        full.impression_strength = 1.0;
        full.access_profile.record_access();
        full.perspectives = vec!["decisions".to_string()];

        let mut half = test_chunk("half strength impression");
        half.entry_type = crate::chunk::EntryType::Impression;
        half.impression_strength = 0.5;
        half.access_profile.record_access();
        half.perspectives = vec!["decisions".to_string()];

        let w = SalienceWeights::default();
        let full_score = compute(&full, &w);
        let half_score = compute(&half, &w);

        assert!(full_score.composite > 0.0);
        assert!((half_score.composite - full_score.composite * 0.5).abs() < 0.001);
    }

    #[test]
    fn test_non_impression_ignores_strength() {
        let mut chunk = test_chunk("raw entry");
        chunk.impression_strength = 0.5; // should be ignored for non-impression
        chunk.access_profile.record_access();

        let mut same = test_chunk("raw entry same");
        same.impression_strength = 1.0;
        same.access_profile.record_access();

        let w = SalienceWeights::default();
        let score_half = compute(&chunk, &w);
        let score_full = compute(&same, &w);

        // Non-impression entries ignore impression_strength
        assert!((score_half.composite - score_full.composite).abs() < 0.001);
    }

    #[test]
    fn test_impression_zero_strength_zeroes_composite() {
        let mut chunk = test_chunk("zero impression");
        chunk.entry_type = crate::chunk::EntryType::Impression;
        chunk.impression_strength = 0.0;
        chunk.access_profile.record_access();
        chunk.perspectives = vec!["decisions".to_string()];

        let score = compute(&chunk, &SalienceWeights::default());
        assert_eq!(score.composite, 0.0);
    }

    // ── Finding #1: negative impression_strength must not corrupt salience ──

    #[test]
    fn test_negative_impression_strength_clamped_to_zero() {
        // A negative strength must produce the same result as strength=0.0,
        // i.e. the composite is zeroed rather than negated.
        let mut neg = test_chunk("negative impression");
        neg.entry_type = crate::chunk::EntryType::Impression;
        neg.impression_strength = -5.0;
        neg.access_profile.record_access();
        neg.perspectives = vec!["decisions".to_string()];

        let mut zero = test_chunk("zero impression");
        zero.entry_type = crate::chunk::EntryType::Impression;
        zero.impression_strength = 0.0;
        zero.access_profile.record_access();
        zero.perspectives = vec!["decisions".to_string()];

        let w = SalienceWeights::default();
        let neg_score = compute(&neg, &w);
        let zero_score = compute(&zero, &w);

        // Negative strength must not produce a negative composite.
        assert!(
            neg_score.composite >= 0.0,
            "composite must not go negative with negative strength, got {}",
            neg_score.composite
        );
        // It should match the zero-strength case exactly (both clamp to 0.0).
        assert!(
            (neg_score.composite - zero_score.composite).abs() < 0.001,
            "negative strength should behave like zero strength: {} vs {}",
            neg_score.composite,
            zero_score.composite
        );
    }

    #[test]
    fn test_positive_impression_strength_still_modulates() {
        // A normal positive strength in [0,1] still scales the composite correctly.
        let mut chunk = test_chunk("half strength");
        chunk.entry_type = crate::chunk::EntryType::Impression;
        chunk.impression_strength = 0.5;
        chunk.access_profile.record_access();
        chunk.perspectives = vec!["decisions".to_string()];

        let mut full = test_chunk("full strength");
        full.entry_type = crate::chunk::EntryType::Impression;
        full.impression_strength = 1.0;
        full.access_profile.record_access();
        full.perspectives = vec!["decisions".to_string()];

        let w = SalienceWeights::default();
        let half_score = compute(&chunk, &w);
        let full_score = compute(&full, &w);

        // Half strength should yield roughly half the composite of full strength.
        assert!(
            half_score.composite > 0.0,
            "positive strength must yield positive composite"
        );
        assert!(
            (half_score.composite - full_score.composite * 0.5).abs() < 0.001,
            "0.5 strength should halve the composite: {} vs {}*0.5={}",
            half_score.composite,
            full_score.composite,
            full_score.composite * 0.5
        );
    }

    // ── Finding #2: MAX_AGING_BATCH_SIZE const existence and value ──────────

    #[test]
    fn test_max_aging_batch_size_constant() {
        // Pin the cap so any accidental change is caught by tests.
        // Values below 500 are not capped; 500 and above all resolve to 500.
        // (This mirrors the behavior of .min(MAX_AGING_BATCH_SIZE) in apply_aging.)
        assert_eq!(MAX_AGING_BATCH_SIZE, 500);

        // Simulate boundary behaviour: a batch request of 499 fits within the cap.
        let requested_499 = 499_usize.min(MAX_AGING_BATCH_SIZE);
        assert_eq!(requested_499, 499);

        // A request of exactly 500 is at the cap.
        let requested_500 = 500_usize.min(MAX_AGING_BATCH_SIZE);
        assert_eq!(requested_500, 500);

        // A request of 501 is capped at 500.
        let requested_501 = 501_usize.min(MAX_AGING_BATCH_SIZE);
        assert_eq!(requested_501, 500);
    }
}
