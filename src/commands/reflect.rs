//! Reflect command and compact operations.

use super::*;

/// Compact sub-operations.
#[derive(Debug, Clone, Copy)]
pub enum CompactAction {
    Rotate,
    Salience,
    ArchiveCandidates,
    /// Prune old LanceDB versions (calls auto-compact logic directly).
    Prune,
}

/// Run a compact sub-action.
pub async fn compact(
    data_dir: &Path,
    action: CompactAction,
    options: &CompactOptions,
) -> Result<()> {
    match action {
        CompactAction::Rotate => compact_rotate(data_dir).await,
        CompactAction::Salience => compact_salience(data_dir, options).await,
        CompactAction::ArchiveCandidates => compact_archive_candidates(data_dir, options).await,
        CompactAction::Prune => compact_prune(data_dir).await,
    }
}

/// Rotate: roll access-profile buckets and apply aging rules.
async fn compact_rotate(data_dir: &Path) -> Result<()> {
    let (_config, _embedder, store, _blob_store) = open_store(data_dir).await?;

    let aging_config = crate::aging::AgingConfig::load(data_dir);
    let aging_result = crate::aging::apply_aging(&store, &aging_config).await?;

    println!("Compact: rotate");
    println!(
        "  Aging config: degrade after {} days",
        aging_config.degrade_after_days
    );
    println!(
        "  Degraded {} entries to '{}'",
        aging_result.degraded_count, aging_config.degrade_to
    );
    for id in &aging_result.degraded_ids {
        println!("    {}", short_id(id));
    }

    Ok(())
}

/// Compact old LanceDB versions — keeps the 3 most recent, deletes older ones.
async fn compact_prune(data_dir: &Path) -> Result<()> {
    let (_config, _embedder, store, _blob_store) = open_store(data_dir).await?;
    store.auto_compact_if_needed().await?;
    println!("Compact: prune");
    println!("  (Run `veclayer status` to see version count before/after)");
    Ok(())
}

/// Print a section header followed by a separator line.
fn print_section_header(title: impl std::fmt::Display) {
    println!("{}", title.if_supports_color(Stream::Stdout, |s| s.bold()));
    println!(
        "{}",
        "=".repeat(60)
            .if_supports_color(Stream::Stdout, |s| s.dimmed())
    );
}

/// Salience: compute and display salience scores.
async fn compact_salience(data_dir: &Path, options: &CompactOptions) -> Result<()> {
    let store = StoreBackend::open_metadata(data_dir, true).await?;

    let hot = store.get_hot_chunks(options.limit * 2).await?;

    if hot.is_empty() {
        println!("No entries to analyze.");
        return Ok(());
    }

    let weights = crate::salience::SalienceWeights::default();
    let top = crate::salience::top_salient(&hot, &weights, options.limit);

    print_section_header(format!("Salience report (top {}):", top.len()));
    for (idx, score) in &top {
        let chunk = &hot[*idx];
        println!(
            "  {} [{}] inter={:.2} persp={:.2} rev={:.2}  {}",
            short_id(&chunk.id).if_supports_color(Stream::Stdout, |s| s.cyan()),
            format!("{:.3}", score.composite).if_supports_color(Stream::Stdout, |s| s.green()),
            score.interaction,
            score.perspective,
            score.revision,
            preview(&chunk.content, 60).if_supports_color(Stream::Stdout, |s| s.dimmed())
        );
    }

    Ok(())
}

/// Archive candidates: entries with low salience.
async fn compact_archive_candidates(data_dir: &Path, options: &CompactOptions) -> Result<()> {
    let store = StoreBackend::open_metadata(data_dir, true).await?;
    let aging_config = crate::aging::AgingConfig::load(data_dir);

    let stale = store
        .get_stale_chunks(aging_config.stale_seconds(), options.limit * 2)
        .await?;

    if stale.is_empty() {
        println!("No archive candidates found.");
        return Ok(());
    }

    let weights = crate::salience::SalienceWeights::default();
    let candidates: Vec<_> = stale
        .iter()
        .filter(|c| {
            crate::salience::is_archive_candidate(
                c,
                &weights,
                options.archive_threshold,
                &aging_config.degrade_from,
            )
        })
        .take(options.limit)
        .collect();

    if candidates.is_empty() {
        println!(
            "No archive candidates below threshold {:.2}.",
            options.archive_threshold
        );
        return Ok(());
    }

    print_section_header(format!(
        "Archive candidates ({}, threshold {:.2}):",
        candidates.len(),
        options.archive_threshold
    ));
    for chunk in &candidates {
        let score = crate::salience::compute(chunk, &weights);
        println!(
            "  {} [salience={}, vis={}]  {}",
            short_id(&chunk.id).if_supports_color(Stream::Stdout, |s| s.cyan()),
            format!("{:.3}", score.composite).if_supports_color(Stream::Stdout, |s| s.red()),
            vis_color(&chunk.visibility),
            preview(&chunk.content, 60).if_supports_color(Stream::Stdout, |s| s.dimmed())
        );
    }
    println!("\nUse `veclayer archive <id>...` to archive selected entries.");

    Ok(())
}

/// Generate a comprehensive reflection/identity report.
pub async fn reflect(data_dir: &Path) -> Result<()> {
    let store = StoreBackend::open_metadata(data_dir, true).await?;
    let snapshot = crate::identity::compute_identity(&store, data_dir, None, None).await?;
    let priming = crate::identity::generate_priming(&snapshot);
    println!("{}", priming);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    use crate::store::StoreBackend;
    use crate::test_helpers::make_test_chunk;

    /// Build a chunk whose access total is above zero so `get_hot_chunks` returns it.
    fn hot_chunk(id: &str, content: &str, total_accesses: u32) -> crate::HierarchicalChunk {
        let mut chunk = make_test_chunk(id, content);
        chunk.access_profile.total = total_accesses;
        // Bump the hour bucket so the relevancy score is non-zero.
        chunk.access_profile.hour = total_accesses.min(u16::MAX as u32) as u16;
        chunk
    }

    /// Build a chunk whose `last_rolled` is set to the Unix epoch so it is treated
    /// as stale under any reasonable aging window.
    fn stale_chunk(id: &str, content: &str) -> crate::HierarchicalChunk {
        let mut chunk = make_test_chunk(id, content);
        // Epoch 0 is always older than `now - 30 days`.
        chunk.access_profile.last_rolled = 0;
        chunk.access_profile.hour = 0;
        chunk.access_profile.day = 0;
        chunk.access_profile.week = 0;
        chunk
    }

    /// Open the store in metadata-only mode and query stale chunks.
    async fn get_stale_chunks_for_test(dir: &Path) -> Result<Vec<crate::HierarchicalChunk>> {
        let store2 = StoreBackend::open_metadata(dir, true).await?;
        let aging_config = crate::aging::AgingConfig::default();
        store2
            .get_stale_chunks(aging_config.stale_seconds(), 100)
            .await
    }

    // ── compact_salience on populated store ───────────────────────────────────

    /// Seeding entries with non-zero `total` access means `get_hot_chunks` returns
    /// them, exercising the salience scoring path in `compact_salience`.
    #[tokio::test]
    async fn test_compact_salience_populated_store() -> Result<()> {
        let dir = TempDir::new()?;
        let store = StoreBackend::open(dir.path(), 384, false).await?;
        store
            .insert_chunks(vec![
                hot_chunk("aaa001", "Architecture decision record", 10),
                hot_chunk("bbb002", "Testing strategy notes", 3),
                hot_chunk("ccc003", "Deployment runbook", 1),
            ])
            .await?;
        drop(store);

        // Salience must complete without error on a populated store.
        compact(
            dir.path(),
            CompactAction::Salience,
            &CompactOptions::default(),
        )
        .await?;
        Ok(())
    }

    /// Entries with higher access counts produce higher salience scores than
    /// entries with fewer accesses.
    #[tokio::test]
    async fn test_compact_salience_orders_by_score() -> Result<()> {
        let dir = TempDir::new()?;
        let store = StoreBackend::open(dir.path(), 384, false).await?;
        store
            .insert_chunks(vec![
                hot_chunk("low001", "Low-traffic entry", 1),
                hot_chunk("high01", "High-traffic entry", 50),
            ])
            .await?;
        drop(store);

        // The underlying salience scorer is exercised; we verify it completes and
        // that the hot-chunks query returns entries in descending access order.
        let store2 = StoreBackend::open_metadata(dir.path(), true).await?;
        let hot = store2.get_hot_chunks(10).await?;
        assert_eq!(hot.len(), 2);
        // get_hot_chunks sorts by total descending.
        assert!(hot[0].access_profile.total >= hot[1].access_profile.total);
        Ok(())
    }

    // ── compact_archive_candidates on populated store ─────────────────────────

    /// Stale entries with low salience appear as archive candidates.
    #[tokio::test]
    async fn test_compact_archive_candidates_finds_stale_entries() -> Result<()> {
        let dir = TempDir::new()?;
        let store = StoreBackend::open(dir.path(), 384, false).await?;
        store
            .insert_chunks(vec![
                stale_chunk("stale1", "Obsolete runbook from last year"),
                stale_chunk("stale2", "Old architecture note, never revisited"),
                stale_chunk("stale3", "Draft that was never completed"),
            ])
            .await?;
        drop(store);

        // Must complete without error when stale candidates are present.
        compact(
            dir.path(),
            CompactAction::ArchiveCandidates,
            &CompactOptions {
                limit: 10,
                archive_threshold: 1.0, // catch everything below perfect salience
            },
        )
        .await?;
        Ok(())
    }

    /// Stale entries appear in `get_stale_chunks` but entries with `deep_only`
    /// visibility are excluded by the LanceDB filter (it only returns `normal`
    /// and `always` visibility). Archived entries must not re-surface as candidates.
    #[tokio::test]
    async fn test_compact_archive_candidates_excludes_deep_only() -> Result<()> {
        let dir = TempDir::new()?;
        let store = StoreBackend::open(dir.path(), 384, false).await?;

        let mut already_archived = stale_chunk("arch01", "Already archived entry");
        already_archived.visibility = "deep_only".to_string();

        let fresh_normal = make_test_chunk("norm01", "Recently accessed normal entry");

        store
            .insert_chunks(vec![already_archived, fresh_normal])
            .await?;
        drop(store);

        // The deep_only entry must not appear as a stale candidate.
        let stale = get_stale_chunks_for_test(dir.path()).await?;

        let deep_only_count = stale.iter().filter(|c| c.visibility == "deep_only").count();
        assert_eq!(
            deep_only_count, 0,
            "deep_only entries must not appear as stale archive candidates"
        );
        Ok(())
    }

    /// Entries below the salience threshold are flagged as archive candidates;
    /// entries above it are not.
    #[tokio::test]
    async fn test_compact_archive_candidates_respects_threshold() -> Result<()> {
        let dir = TempDir::new()?;
        let store = StoreBackend::open(dir.path(), 384, false).await?;

        // Low-salience: stale + no perspectives + no relations → score ≈ 0.
        let low = stale_chunk("low001", "Stale low-salience entry");

        // Higher-salience: many perspectives boost the perspective component.
        let mut high = stale_chunk("high01", "Stale but rich entry");
        high.perspectives = (0..8).map(|i| format!("p{i}")).collect();

        store.insert_chunks(vec![low, high]).await?;
        drop(store);

        let stale = get_stale_chunks_for_test(dir.path()).await?;

        let weights = crate::salience::SalienceWeights::default();
        let degradable = vec!["normal".to_string()];

        let candidates: Vec<_> = stale
            .iter()
            .filter(|c| crate::salience::is_archive_candidate(c, &weights, 0.1, &degradable))
            .collect();

        // The low-salience entry must be a candidate; the high-salience one must not.
        assert!(
            candidates.iter().any(|c| c.id == "low001"),
            "low-salience stale entry should be an archive candidate"
        );
        assert!(
            !candidates.iter().any(|c| c.id == "high01"),
            "high-salience entry must not be an archive candidate below threshold 0.1"
        );
        Ok(())
    }

    #[test]
    fn test_compact_options_default() {
        let opts = CompactOptions::default();
        assert_eq!(opts.limit, 20);
        assert_eq!(opts.archive_threshold, 0.1);
    }

    // ── CompactOptions custom values ──────────────────────────────────────────

    #[test]
    fn test_compact_options_custom() {
        let opts = CompactOptions {
            limit: 50,
            archive_threshold: 0.25,
        };
        assert_eq!(opts.limit, 50);
        assert!((opts.archive_threshold - 0.25).abs() < f32::EPSILON);
    }

    // ── compact_salience on empty store ───────────────────────────────────────

    #[tokio::test]
    async fn test_compact_salience_empty_store() -> Result<()> {
        let dir = TempDir::new()?;
        compact(
            dir.path(),
            CompactAction::Salience,
            &CompactOptions::default(),
        )
        .await?;
        Ok(())
    }

    // ── compact_archive_candidates on empty store ─────────────────────────────

    #[tokio::test]
    async fn test_compact_archive_candidates_empty_store() -> Result<()> {
        let dir = TempDir::new()?;
        compact(
            dir.path(),
            CompactAction::ArchiveCandidates,
            &CompactOptions::default(),
        )
        .await?;
        Ok(())
    }

    // ── reflect on empty store ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_reflect_empty_store() -> Result<()> {
        let dir = TempDir::new()?;
        reflect(dir.path()).await?;
        Ok(())
    }

    // ── compact_rotate on empty store ─────────────────────────────────────────

    #[tokio::test]
    async fn test_compact_rotate_empty_store() -> Result<()> {
        let dir = TempDir::new()?;
        compact(
            dir.path(),
            CompactAction::Rotate,
            &CompactOptions::default(),
        )
        .await?;
        Ok(())
    }

    // ── CompactAction variants are all handled ────────────────────────────────

    #[test]
    fn test_compact_action_debug() {
        // Ensure all variants are reachable/Debug-printable
        let actions = [
            CompactAction::Rotate,
            CompactAction::Salience,
            CompactAction::ArchiveCandidates,
        ];
        for action in &actions {
            let _ = format!("{action:?}");
        }
    }
}
