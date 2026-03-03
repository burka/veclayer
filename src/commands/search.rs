//! Semantic search, browse, and focus operations.

use super::*;

/// Semantic search with hierarchical results (prints output).
pub async fn search(data_dir: &Path, query_str: &str, options: &SearchOptions) -> Result<()> {
    if options.similar_to.is_some() && !query_str.is_empty() {
        return Err(crate::Error::InvalidOperation(
            "--similar-to and a text query are mutually exclusive. Use one or the other."
                .to_string(),
        ));
    }

    let results = search_results(data_dir, query_str, options).await?;

    if results.is_empty() {
        if let Some(ref target_id) = options.similar_to {
            println!("No similar entries found for: {}", short_id(target_id));
        } else {
            println!("No results found.");
        }
        return Ok(());
    }

    if let Some(ref target_id) = options.similar_to {
        println!("\nSimilar entries to: {}\n", short_id(target_id));
    } else {
        println!("\nSearch results for: \"{}\"\n", query_str);
    }
    println!("{}", "=".repeat(60));

    for (i, result) in results.iter().enumerate() {
        if i > 0 {
            println!();
        }

        let heading = result
            .chunk
            .heading
            .as_deref()
            .unwrap_or_else(|| result.chunk.content.lines().next().unwrap_or("(untitled)"));
        let tier = crate::mcp::types::relevance_tier(result.score);
        println!(
            "{}  {} {:.2}",
            format!("{}. {}", i + 1, heading).if_supports_color(Stream::Stdout, |s| s.bold()),
            tier.if_supports_color(Stream::Stdout, |s| s.dimmed()),
            result
                .score
                .if_supports_color(Stream::Stdout, |s| s.dimmed()),
        );

        let mut meta = vec![short_id(&result.chunk.id)
            .if_supports_color(Stream::Stdout, |s| s.cyan())
            .to_string()];
        if result.chunk.entry_type != EntryType::Raw {
            meta.push(
                result
                    .chunk
                    .entry_type
                    .to_string()
                    .if_supports_color(Stream::Stdout, |s| s.yellow())
                    .to_string(),
            );
        }
        if !result.chunk.perspectives.is_empty() {
            meta.push(
                result
                    .chunk
                    .perspectives
                    .join(", ")
                    .if_supports_color(Stream::Stdout, |s| s.magenta())
                    .to_string(),
            );
        }
        if result.chunk.visibility != "normal" {
            meta.push(
                result
                    .chunk
                    .visibility
                    .if_supports_color(Stream::Stdout, |s| s.red())
                    .to_string(),
            );
        }
        if options.show_path && !result.hierarchy_path.is_empty() {
            let path: Vec<&str> = result
                .hierarchy_path
                .iter()
                .filter_map(|c| c.heading.as_deref())
                .collect();
            meta.push(path.join(" > "));
        }
        println!("   {}", meta.join(" | "));

        println!(
            "   {}",
            preview(&result.chunk.content, 200).if_supports_color(Stream::Stdout, |s| s.dimmed())
        );

        if !result.relevant_children.is_empty() {
            for child in &result.relevant_children {
                let child_heading = child
                    .chunk
                    .heading
                    .as_deref()
                    .unwrap_or_else(|| child.chunk.content.lines().next().unwrap_or("..."));
                println!(
                    "     {} {} [{}]",
                    ">".if_supports_color(Stream::Stdout, |s| s.dimmed()),
                    preview(child_heading, 60),
                    short_id(&child.chunk.id).if_supports_color(Stream::Stdout, |s| s.cyan())
                );
            }
        }
    }

    println!(
        "\n{} `veclayer focus <id>` to drill in.",
        format!("{} result(s).", results.len()).if_supports_color(Stream::Stdout, |s| s.bold())
    );

    Ok(())
}

/// Run search and return structured results (for programmatic use).
pub async fn search_results(
    data_dir: &Path,
    query_str: &str,
    options: &SearchOptions,
) -> Result<Vec<SearchResult>> {
    let since_epoch = options
        .since
        .as_deref()
        .and_then(crate::resolve::parse_temporal);
    let until_epoch = options
        .until
        .as_deref()
        .and_then(crate::resolve::parse_temporal);

    let (_config, embedder, store, _blob_store) = open_store(data_dir).await?;

    let open_thread_ids = crate::identity::resolve_ongoing_filter(&store, options.ongoing).await?;

    let fetch_limit = if since_epoch.is_some() || until_epoch.is_some() {
        options.top_k * TEMPORAL_PREFETCH_FACTOR
    } else {
        options.top_k
    };

    let config = SearchConfig::for_query(fetch_limit, options.deep, options.recent.as_deref())
        .with_perspective(options.perspective.clone())
        .with_min_salience(options.min_salience)
        .with_min_score(options.min_score);

    let search_engine = HierarchicalSearch::new(store, embedder).with_config(config);

    let results = if let Some(ref target_id) = options.similar_to {
        search_engine
            .search_by_embedding(target_id, fetch_limit)
            .await?
    } else if let Some(ref parent_id) = options.subtree {
        search_engine.search_subtree(query_str, parent_id).await?
    } else {
        search_engine.search(query_str).await?
    };

    let filtered = results
        .into_iter()
        .filter(|r| {
            let created = r.chunk.access_profile.created_at;
            since_epoch.is_none_or(|s| created >= s)
                && until_epoch.is_none_or(|u| created <= u)
                && crate::identity::passes_ongoing_filter(&open_thread_ids, &r.chunk.id)
        })
        .take(options.top_k);

    Ok(filtered
        .map(|r| SearchResult {
            chunk: r.chunk,
            score: r.score,
            hierarchy_path: r.hierarchy_path,
            relevant_children: r
                .relevant_children
                .into_iter()
                .map(|c| SearchResult {
                    chunk: c.chunk,
                    score: c.score,
                    hierarchy_path: vec![],
                    relevant_children: vec![],
                })
                .collect(),
        })
        .collect())
}

/// Focus on an entry: show details and children.
pub async fn focus(data_dir: &Path, id: &str, options: &FocusOptions) -> Result<()> {
    let (_config, embedder, store, _blob_store) = open_store(data_dir).await?;

    let entry = store
        .get_by_id_prefix(id)
        .await?
        .ok_or_else(|| crate::Error::not_found(format!("Entry {} not found", id)))?;

    println!(
        "{}",
        format!("Entry {}", short_id(&entry.id)).if_supports_color(Stream::Stdout, |s| s.bold())
    );
    println!(
        "{}",
        "=".repeat(50)
            .if_supports_color(Stream::Stdout, |s| s.dimmed())
    );
    println!(
        "  {}  {}",
        "Type:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
        entry
            .entry_type
            .to_string()
            .if_supports_color(Stream::Stdout, |s| s.yellow())
    );
    println!(
        "  {}  {}",
        "Level:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
        entry.level
    );
    println!(
        "  {}  {}",
        "Vis:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
        vis_color(&entry.visibility)
    );
    println!(
        "  {}  {}",
        "Source:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
        entry.source_file
    );
    if let Some(ref heading) = entry.heading {
        println!(
            "  {}  {}",
            "Heading:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
            heading
        );
    }
    if !entry.perspectives.is_empty() {
        println!(
            "  {}  {}",
            "Perspectives:".if_supports_color(Stream::Stdout, |s| s.dimmed()),
            entry
                .perspectives
                .join(", ")
                .if_supports_color(Stream::Stdout, |s| s.magenta())
        );
    }
    if !entry.relations.is_empty() {
        println!(
            "  {}",
            "Relations:".if_supports_color(Stream::Stdout, |s| s.dimmed())
        );
        for rel in &entry.relations {
            println!(
                "    {} {} {}",
                rel.kind.if_supports_color(Stream::Stdout, |s| s.yellow()),
                "->".if_supports_color(Stream::Stdout, |s| s.dimmed()),
                short_id(&rel.target_id).if_supports_color(Stream::Stdout, |s| s.cyan())
            );
        }
    }
    println!("\n{}\n", entry.content);

    let mut children = store.get_children(&entry.id).await?;

    if !children.is_empty() {
        if let Some(ref question) = options.question {
            let query_emb = embedder.embed(&[question.as_str()])?;
            if let Some(query_vec) = query_emb.into_iter().next() {
                children.sort_by(|a, b| {
                    let score_a = a
                        .embedding
                        .as_ref()
                        .map(|e| crate::search::cosine_similarity(&query_vec, e))
                        .unwrap_or(0.0);
                    let score_b = b
                        .embedding
                        .as_ref()
                        .map(|e| crate::search::cosine_similarity(&query_vec, e))
                        .unwrap_or(0.0);
                    score_b
                        .partial_cmp(&score_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        let shown = children.iter().take(options.limit).count();
        println!(
            "{}",
            format!("Children ({}/{}):", shown, children.len())
                .if_supports_color(Stream::Stdout, |s| s.bold())
        );
        for child in children.iter().take(options.limit) {
            println!(
                "  {} [{}] {}",
                short_id(&child.id).if_supports_color(Stream::Stdout, |s| s.cyan()),
                child
                    .entry_type
                    .to_string()
                    .if_supports_color(Stream::Stdout, |s| s.yellow()),
                preview(&child.content, 100).if_supports_color(Stream::Stdout, |s| s.dimmed())
            );
        }

        println!("\nUse `veclayer focus <child-id>` to drill deeper.");
    } else {
        println!(
            "{}",
            "(no children)".if_supports_color(Stream::Stdout, |s| s.dimmed())
        );
    }

    Ok(())
}

/// Browse entries without vector search (list by perspective/recency).
pub async fn browse(data_dir: &Path, options: &SearchOptions) -> Result<()> {
    let since_epoch = options
        .since
        .as_deref()
        .and_then(crate::resolve::parse_temporal);
    let until_epoch = options
        .until
        .as_deref()
        .and_then(crate::resolve::parse_temporal);

    let store = StoreBackend::open_metadata(data_dir, true).await?;

    let open_thread_ids = crate::identity::resolve_ongoing_filter(&store, options.ongoing).await?;

    let fetch_limit = if open_thread_ids.is_some() {
        usize::MAX
    } else {
        options.top_k
    };

    let all_entries = store
        .list_entries(
            options.perspective.as_deref(),
            since_epoch,
            until_epoch,
            fetch_limit,
        )
        .await?;

    let entries: Vec<_> = all_entries
        .into_iter()
        .filter(|chunk| crate::identity::passes_ongoing_filter(&open_thread_ids, &chunk.id))
        .take(options.top_k)
        .collect();

    if entries.is_empty() {
        println!("No entries found.");
        return Ok(());
    }

    for (i, chunk) in entries.iter().enumerate() {
        if i > 0 {
            println!();
        }
        print_entry_line(i, chunk);
    }

    println!(
        "\n{} `veclayer focus <id>` to drill in.",
        format!("{} entry(ies).", entries.len()).if_supports_color(Stream::Stdout, |s| s.bold())
    );
    Ok(())
}

fn print_entry_line(index: usize, chunk: &crate::HierarchicalChunk) {
    let heading = chunk
        .heading
        .as_deref()
        .unwrap_or_else(|| chunk.content.lines().next().unwrap_or("(untitled)"));
    println!(
        "{}",
        format!("{}. {}", index + 1, heading).if_supports_color(Stream::Stdout, |s| s.bold())
    );

    let mut meta = vec![short_id(&chunk.id)
        .if_supports_color(Stream::Stdout, |s| s.cyan())
        .to_string()];
    if chunk.entry_type != EntryType::Raw {
        meta.push(
            chunk
                .entry_type
                .to_string()
                .if_supports_color(Stream::Stdout, |s| s.yellow())
                .to_string(),
        );
    }
    if !chunk.perspectives.is_empty() {
        meta.push(
            chunk
                .perspectives
                .join(", ")
                .if_supports_color(Stream::Stdout, |s| s.magenta())
                .to_string(),
        );
    }
    if chunk.visibility != "normal" {
        meta.push(vis_color(&chunk.visibility));
    }
    println!("   {}", meta.join(" | "));
    println!(
        "   {}",
        preview(&chunk.content, 200).if_supports_color(Stream::Stdout, |s| s.dimmed())
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_search_options_default() {
        let opts = SearchOptions::default();
        assert_eq!(opts.top_k, 5);
        assert!(!opts.show_path);
        assert!(opts.subtree.is_none());
        assert!(opts.since.is_none());
        assert!(opts.until.is_none());
    }

    #[test]
    fn test_focus_options_default() {
        let opts = FocusOptions::default();
        assert!(opts.question.is_none());
        assert_eq!(opts.limit, 10);
    }

    #[tokio::test]
    async fn test_browse_empty_store() -> Result<()> {
        let dir = TempDir::new()?;
        StoreBackend::open_metadata(dir.path(), false).await?;
        browse(dir.path(), &SearchOptions::default()).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_browse_returns_entries() -> Result<()> {
        let dir = TempDir::new()?;
        crate::test_helpers::make_test_chunk("aaa111", "First entry");
        crate::test_helpers::make_test_chunk("bbb222", "Second entry");
        browse(dir.path(), &SearchOptions::default()).await?;
        Ok(())
    }
}
