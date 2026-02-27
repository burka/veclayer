//! Reflect command and compact operations.

use super::*;

/// Compact sub-operations.
#[derive(Debug, Clone, Copy)]
pub enum CompactAction {
    Rotate,
    Salience,
    ArchiveCandidates,
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
    }
}

/// Rotate: roll access-profile buckets and apply aging rules.
async fn compact_rotate(data_dir: &Path) -> Result<()> {
    let (_embedder, store, _blob_store) = open_store(data_dir).await?;

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

    println!(
        "{}",
        format!("Salience report (top {}):", top.len())
            .if_supports_color(Stream::Stdout, |s| s.bold())
    );
    println!(
        "{}",
        "=".repeat(60)
            .if_supports_color(Stream::Stdout, |s| s.dimmed())
    );
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

    println!(
        "{}",
        format!(
            "Archive candidates ({}, threshold {:.2}):",
            candidates.len(),
            options.archive_threshold
        )
        .if_supports_color(Stream::Stdout, |s| s.bold())
    );
    println!(
        "{}",
        "=".repeat(60)
            .if_supports_color(Stream::Stdout, |s| s.dimmed())
    );
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

    #[test]
    fn test_compact_options_default() {
        let opts = CompactOptions::default();
        assert_eq!(opts.limit, 20);
        assert_eq!(opts.archive_threshold, 0.1);
    }
}
