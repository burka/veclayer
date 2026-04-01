//! Knowledge curation and aging operations.

use super::*;

/// Run one think cycle: reflect → LLM → add → compact.
#[cfg(feature = "llm")]
pub async fn think(data_dir: &Path) -> Result<()> {
    let (config, embedder, store, blob_store) = open_store(data_dir).await?;

    let llm = crate::llm::LlmBackend::from_config(&config.llm);

    println!(
        "Think: starting sleep cycle (LLM: {} via {})",
        config.llm.model, config.llm.provider
    );

    let result = match crate::think::execute(
        &store,
        &embedder,
        &llm,
        data_dir,
        Some(&blob_store),
        None,
    )
    .await
    {
        Ok(result) => result,
        Err(crate::Error::Llm(msg)) => {
            eprintln!(
                    "Think: LLM unavailable ({}/{} not reachable).\n\
                     Error: {}\n\n\
                     You can consolidate memory yourself:\n\
                     1. Run `veclayer think prepare` to get the reflection data\n\
                     2. Reason about consolidations and learnings\n\
                     3. Run `veclayer store` for each result with --entry-type meta or summary\n\n\
                     Or configure an LLM: VECLAYER_LLM_PROVIDER=openai VECLAYER_LLM_API_KEY=... veclayer think",
                    config.llm.provider, config.llm.model, msg
                );
            return Ok(());
        }
        Err(e) => return Err(e),
    };

    if result.entries_created.is_empty() {
        println!("\nNothing to consolidate. Memory is either empty or already well-organized.");
        return Ok(());
    }

    println!("\nThink cycle complete:");

    if let Some(ref id) = result.narrative_id {
        println!("  Narrative: {}", short_id(id));
    }

    if result.consolidations_added > 0 {
        println!(
            "  Consolidations: {} summaries created",
            result.consolidations_added
        );
    }

    if result.learnings_added > 0 {
        println!(
            "  Learnings: {} meta-entries extracted",
            result.learnings_added
        );
    }

    println!("\nEntries created:");
    for entry in &result.entries_created {
        let persp = if entry.perspectives.is_empty() {
            String::new()
        } else {
            format!(" [{}]", entry.perspectives.join(", "))
        };
        println!(
            "  {} ({}{}) {}",
            short_id(&entry.id),
            entry.entry_type,
            persp,
            entry.content_preview
        );
    }

    println!("\nAging applied. Run `veclayer reflect` to see updated identity.");

    Ok(())
}

/// Gather reflection data for manual consolidation (no LLM needed).
pub async fn think_prepare(data_dir: &Path) -> Result<()> {
    let store = StoreBackend::open_metadata(data_dir, true).await?;
    let prep = crate::think::prepare(&store, data_dir).await?;

    let Some(prep) = prep else {
        println!("Nothing to prepare. Memory is empty.");
        return Ok(());
    };

    println!("## Think Preparation\n");

    println!("### Task\n");
    println!("{}\n", prep.system_prompt);

    println!("### Memory State\n");
    println!("{}\n", prep.user_prompt);

    println!("### How to Apply\n");
    println!("Reason about the memory state above, then run:\n");
    println!("- veclayer store --heading \"[think:narrative]\" --entry-type meta \"...\"");
    println!(
        "- veclayer store --heading \"[think:consolidation]\" --entry-type summary --rel-summarizes <id> \"...\""
    );
    println!(
        "- veclayer store --heading \"[think:learning]\" --entry-type meta -P learnings \"...\""
    );

    Ok(())
}

/// Apply aging rules.
pub async fn think_aging_apply(data_dir: &Path) -> Result<()> {
    let store = StoreBackend::open_metadata(data_dir, false).await?;
    let config = crate::aging::AgingConfig::load(data_dir);
    let result = crate::aging::apply_aging(&store, &config).await?;

    if result.degraded_count == 0 {
        println!("No entries needed aging. All knowledge is fresh.");
    } else {
        println!(
            "Aged {} entries (degraded to '{}'):",
            result.degraded_count, config.degrade_to
        );
        for id in &result.degraded_ids {
            println!("  {}", short_id(id));
        }
    }
    Ok(())
}

/// Configure aging parameters.
pub async fn think_aging_configure(
    data_dir: &Path,
    days: Option<u32>,
    to: Option<&str>,
) -> Result<()> {
    let mut config = crate::aging::AgingConfig::load(data_dir);
    if let Some(days) = days {
        config.degrade_after_days = days;
    }
    if let Some(to) = to {
        config.degrade_to = to.to_string();
    }
    config.save(data_dir)?;
    println!(
        "Aging configured: degrade {} -> '{}' after {} days without access",
        config.degrade_from.join(", "),
        config.degrade_to,
        config.degrade_after_days
    );
    Ok(())
}

/// Discover similar-but-unlinked entries.
pub async fn think_discover(data_dir: &Path, limit: usize) -> Result<()> {
    let store = StoreBackend::open_metadata(data_dir, true).await?;
    let store = std::sync::Arc::new(store);
    let report = crate::think::discover_unlinked_pairs(&store, limit).await?;
    println!("{}", report);
    Ok(())
}

/// Promote an entry's visibility.
pub async fn think_promote(data_dir: &Path, id: &str, visibility: &str) -> Result<()> {
    set_visibility(data_dir, id, visibility, "Promoted").await
}

/// Demote an entry's visibility.
pub async fn think_demote(data_dir: &Path, id: &str, visibility: &str) -> Result<()> {
    set_visibility(data_dir, id, visibility, "Demoted").await
}

/// Add a relation between two entries.
pub async fn think_relate(data_dir: &Path, source: &str, target: &str, kind: &str) -> Result<()> {
    let store = StoreBackend::open_metadata(data_dir, false).await?;
    let store = std::sync::Arc::new(store);
    let source_id = crate::resolve::resolve_id(&store, source).await?;
    let target_id = crate::resolve::resolve_id(&store, target).await?;

    let relation = crate::ChunkRelation::new(kind, &target_id);
    store.add_relation(&source_id, relation).await?;

    if kind == "related_to" {
        let backward = crate::ChunkRelation::new("related_to", &source_id);
        store.add_relation(&target_id, backward).await?;
    }

    println!(
        "Added relation '{}' from {} to {}",
        kind,
        short_id(&source_id),
        short_id(&target_id)
    );
    Ok(())
}

/// Set an entry's visibility and print a labeled confirmation.
async fn set_visibility(data_dir: &Path, id: &str, visibility: &str, label: &str) -> Result<()> {
    let store = StoreBackend::open_metadata(data_dir, false).await?;
    let store = std::sync::Arc::new(store);
    let chunk_id = crate::resolve::resolve_id(&store, id).await?;
    store.update_visibility(&chunk_id, visibility).await?;
    println!(
        "{} {} to visibility '{}'",
        label,
        short_id(&chunk_id),
        visibility
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    use crate::test_helpers::make_test_chunk;

    async fn seed_store(dir: &Path) -> StoreBackend {
        let store = StoreBackend::open(dir, 384, false).await.unwrap();
        store
            .insert_chunks(vec![
                make_test_chunk("aaa111", "First entry about architecture"),
                make_test_chunk("bbb222", "Second entry about testing"),
            ])
            .await
            .unwrap();
        store
    }

    /// Opens the metadata store and returns the entry with id "aaa111".
    async fn get_seeded_entry(dir: &Path) -> Result<crate::HierarchicalChunk> {
        let store = StoreBackend::open_metadata(dir, false).await?;
        let entry = store.get_by_id("aaa111").await?.unwrap();
        Ok(entry)
    }

    #[tokio::test]
    async fn test_think_promote_changes_visibility() -> Result<()> {
        let dir = TempDir::new()?;
        let _store = seed_store(dir.path()).await;

        think_promote(dir.path(), "aaa111", "always").await?;
        let entry = get_seeded_entry(dir.path()).await?;
        assert_eq!(entry.visibility, "always");
        Ok(())
    }

    #[tokio::test]
    async fn test_think_promote_resolves_prefix() -> Result<()> {
        let dir = TempDir::new()?;
        let _store = seed_store(dir.path()).await;

        think_promote(dir.path(), "aaa", "always").await?;
        let entry = get_seeded_entry(dir.path()).await?;
        assert_eq!(entry.visibility, "always");
        Ok(())
    }

    #[tokio::test]
    async fn test_think_promote_not_found() {
        let dir = TempDir::new().unwrap();
        seed_store(dir.path()).await;

        let result = think_promote(dir.path(), "zzz999", "always").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_think_demote_changes_visibility() -> Result<()> {
        let dir = TempDir::new()?;
        let _store = seed_store(dir.path()).await;

        think_demote(dir.path(), "aaa111", "deep_only").await?;

        let store = StoreBackend::open_metadata(dir.path(), false).await?;
        let entry = store.get_by_id("aaa111").await?.unwrap();
        assert_eq!(entry.visibility, "deep_only");
        Ok(())
    }

    #[tokio::test]
    async fn test_think_relate_adds_forward_relation() -> Result<()> {
        let dir = TempDir::new()?;
        let _store = seed_store(dir.path()).await;

        think_relate(dir.path(), "aaa111", "bbb222", "derived_from").await?;

        let store = StoreBackend::open_metadata(dir.path(), false).await?;
        let source = store.get_by_id("aaa111").await?.unwrap();
        assert_eq!(source.relations.len(), 1);
        assert_eq!(source.relations[0].kind, "derived_from");
        assert_eq!(source.relations[0].target_id, "bbb222");
        Ok(())
    }

    #[tokio::test]
    async fn test_think_relate_bidirectional_for_related_to() -> Result<()> {
        let dir = TempDir::new()?;
        let _store = seed_store(dir.path()).await;

        think_relate(dir.path(), "aaa111", "bbb222", "related_to").await?;

        let store = StoreBackend::open_metadata(dir.path(), false).await?;
        let source = store.get_by_id("aaa111").await?.unwrap();
        assert_eq!(source.relations.len(), 1);
        assert_eq!(source.relations[0].kind, "related_to");
        assert_eq!(source.relations[0].target_id, "bbb222");

        let target = store.get_by_id("bbb222").await?.unwrap();
        assert_eq!(target.relations.len(), 1);
        assert_eq!(target.relations[0].kind, "related_to");
        assert_eq!(target.relations[0].target_id, "aaa111");
        Ok(())
    }

    #[tokio::test]
    async fn test_think_relate_no_backward_for_derived_from() -> Result<()> {
        let dir = TempDir::new()?;
        let _store = seed_store(dir.path()).await;

        think_relate(dir.path(), "aaa111", "bbb222", "derived_from").await?;

        let store = StoreBackend::open_metadata(dir.path(), false).await?;
        let target = store.get_by_id("bbb222").await?.unwrap();
        assert!(
            target.relations.is_empty(),
            "non-related_to should not add backward link"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_think_aging_apply_empty_store() -> Result<()> {
        let dir = TempDir::new()?;
        StoreBackend::open_metadata(dir.path(), false).await?;
        think_aging_apply(dir.path()).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_think_aging_configure_saves() -> Result<()> {
        let dir = TempDir::new()?;
        std::fs::create_dir_all(dir.path())?;

        think_aging_configure(dir.path(), Some(7), Some("archived")).await?;

        let config = crate::aging::AgingConfig::load(dir.path());
        assert_eq!(config.degrade_after_days, 7);
        assert_eq!(config.degrade_to, "archived");
        Ok(())
    }

    #[tokio::test]
    async fn test_think_aging_configure_partial_update() -> Result<()> {
        let dir = TempDir::new()?;
        std::fs::create_dir_all(dir.path())?;

        think_aging_configure(dir.path(), Some(14), Some("deep_only")).await?;
        think_aging_configure(dir.path(), Some(3), None).await?;

        let config = crate::aging::AgingConfig::load(dir.path());
        assert_eq!(config.degrade_after_days, 3);
        assert_eq!(config.degrade_to, "deep_only");
        Ok(())
    }
}
