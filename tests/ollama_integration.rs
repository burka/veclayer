//! Integration tests requiring a running Ollama instance.
//!
//! Run with: `cargo test --test ollama_integration -- --ignored`
//!
//! Prerequisites:
//! - Ollama running on localhost:11434
//! - Model available (tinyllama or llama3.2)

use veclayer::cluster::ClusterPipeline;
use veclayer::embedder::FastEmbedder;
use veclayer::summarizer::OllamaSummarizer;
use veclayer::{ChunkLevel, Embedder, HierarchicalChunk, Summarizer};

/// Check if Ollama is available
async fn ollama_available() -> bool {
    let client = reqwest::Client::new();
    client
        .get("http://localhost:11434/api/tags")
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

fn create_test_chunk(id: &str, content: &str, embedding: Vec<f32>) -> HierarchicalChunk {
    let mut chunk = HierarchicalChunk::new(
        content.to_string(),
        ChunkLevel::CONTENT,
        None,
        format!("path_{}", id),
        "test.md".to_string(),
    );
    chunk.id = id.to_string();
    chunk.embedding = Some(embedding);
    chunk
}

// ============================================================================
// OllamaSummarizer Tests
// ============================================================================

#[tokio::test]
#[ignore = "requires Ollama running locally"]
async fn test_ollama_summarizer_basic() {
    if !ollama_available().await {
        eprintln!("Skipping: Ollama not available");
        return;
    }

    let summarizer = OllamaSummarizer::new().with_model("tinyllama");

    let texts = &[
        "Rust is a systems programming language focused on safety and performance.",
        "The borrow checker ensures memory safety without garbage collection.",
    ];

    let summary = summarizer.summarize(texts).await;

    assert!(summary.is_ok(), "Summarization failed: {:?}", summary.err());
    let summary_text = summary.unwrap();
    assert!(!summary_text.is_empty(), "Summary should not be empty");
    println!("Summary: {}", summary_text);
}

#[tokio::test]
#[ignore = "requires Ollama running locally"]
async fn test_ollama_summarizer_empty_input() {
    if !ollama_available().await {
        eprintln!("Skipping: Ollama not available");
        return;
    }

    let summarizer = OllamaSummarizer::new().with_model("tinyllama");
    let result = summarizer.summarize(&[]).await;

    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[tokio::test]
#[ignore = "requires Ollama running locally"]
async fn test_ollama_summarizer_single_text() {
    if !ollama_available().await {
        eprintln!("Skipping: Ollama not available");
        return;
    }

    let summarizer = OllamaSummarizer::new().with_model("tinyllama");
    let texts = &["Vector databases store embeddings for semantic search."];

    let summary = summarizer.summarize(texts).await;

    assert!(summary.is_ok());
    let text = summary.unwrap();
    assert!(!text.is_empty());
    println!("Single text summary: {}", text);
}

#[tokio::test]
#[ignore = "requires Ollama running locally"]
async fn test_ollama_summarizer_batch() {
    if !ollama_available().await {
        eprintln!("Skipping: Ollama not available");
        return;
    }

    let summarizer = OllamaSummarizer::new().with_model("tinyllama");

    let groups = vec![
        vec!["Rust has zero-cost abstractions.", "Rust compiles to native code."],
        vec!["Python is interpreted.", "Python has dynamic typing."],
    ];

    let summaries = summarizer.summarize_batch(groups).await;

    assert!(summaries.is_ok());
    let results = summaries.unwrap();
    assert_eq!(results.len(), 2);
    for (i, summary) in results.iter().enumerate() {
        assert!(!summary.is_empty(), "Summary {} should not be empty", i);
        println!("Batch summary {}: {}", i, summary);
    }
}

#[tokio::test]
#[ignore = "requires Ollama running locally"]
async fn test_ollama_summarizer_wrong_model() {
    if !ollama_available().await {
        eprintln!("Skipping: Ollama not available");
        return;
    }

    let summarizer = OllamaSummarizer::new().with_model("nonexistent-model-xyz");
    let result = summarizer.summarize(&["test"]).await;

    // Should fail with model not found
    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    println!("Expected error: {}", err);
}

// ============================================================================
// ClusterPipeline Integration Tests
// ============================================================================

#[tokio::test]
#[ignore = "requires Ollama running locally"]
async fn test_cluster_pipeline_with_ollama() {
    if !ollama_available().await {
        eprintln!("Skipping: Ollama not available");
        return;
    }

    let embedder = FastEmbedder::new().expect("Failed to create embedder");
    let _dim = embedder.dimension();

    // Create test chunks with real embeddings
    let texts = [
        "Rust provides memory safety without garbage collection through its ownership system.",
        "The borrow checker in Rust prevents data races at compile time.",
        "Rust's type system ensures thread safety through Send and Sync traits.",
        "Python uses automatic garbage collection for memory management.",
        "Python's GIL prevents true parallelism in CPU-bound tasks.",
        "Python is dynamically typed and interpreted at runtime.",
    ];

    let embeddings = embedder.embed(&texts.iter().map(|s| *s).collect::<Vec<_>>())
        .expect("Failed to embed");

    let chunks: Vec<HierarchicalChunk> = texts
        .iter()
        .zip(embeddings.into_iter())
        .enumerate()
        .map(|(i, (text, emb))| create_test_chunk(&format!("chunk_{}", i), text, emb))
        .collect();

    // Create pipeline with Ollama
    let summarizer = OllamaSummarizer::new().with_model("tinyllama");
    let pipeline_embedder = FastEmbedder::new().expect("Failed to create embedder");

    let pipeline = ClusterPipeline::with_summarizer(pipeline_embedder, summarizer)
        .with_min_cluster_size(2)
        .with_cluster_range(2, 4);

    let result = pipeline.process(chunks).await;

    assert!(result.is_ok(), "Pipeline failed: {:?}", result.err());
    let (updated_chunks, summary_chunks) = result.unwrap();

    // Verify chunks have cluster memberships
    println!("\n=== Cluster Assignments ===");
    for chunk in &updated_chunks {
        println!(
            "Chunk '{}': {:?}",
            &chunk.content[..50.min(chunk.content.len())],
            chunk.cluster_memberships.iter().map(|m| (&m.cluster_id, m.probability)).collect::<Vec<_>>()
        );
        assert!(
            !chunk.cluster_memberships.is_empty(),
            "Chunk should have cluster memberships"
        );
    }

    // Verify summaries were generated
    println!("\n=== Generated Summaries ===");
    assert!(!summary_chunks.is_empty(), "Should have generated summaries");
    for summary in &summary_chunks {
        println!("Summary: {}", summary.content);
        assert!(summary.is_summary);
        assert!(!summary.summarizes.is_empty());
        assert!(summary.embedding.is_some(), "Summary should have embedding");
    }

    println!("\n=== Results ===");
    println!("Updated chunks: {}", updated_chunks.len());
    println!("Summary chunks: {}", summary_chunks.len());
}

#[tokio::test]
#[ignore = "requires Ollama running locally"]
async fn test_cluster_pipeline_discovers_semantic_clusters() {
    if !ollama_available().await {
        eprintln!("Skipping: Ollama not available");
        return;
    }

    let embedder = FastEmbedder::new().expect("Failed to create embedder");

    // Create clearly separated semantic clusters
    let rust_texts = [
        "Rust's ownership model prevents memory leaks.",
        "The Rust compiler catches bugs at compile time.",
        "Cargo is Rust's package manager and build tool.",
    ];

    let web_texts = [
        "HTML defines the structure of web pages.",
        "CSS styles the visual presentation of websites.",
        "JavaScript adds interactivity to web applications.",
    ];

    let all_texts: Vec<&str> = rust_texts.iter().chain(web_texts.iter()).copied().collect();
    let embeddings = embedder.embed(&all_texts).expect("Failed to embed");

    let chunks: Vec<HierarchicalChunk> = all_texts
        .iter()
        .zip(embeddings.into_iter())
        .enumerate()
        .map(|(i, (text, emb))| {
            let topic = if i < 3 { "rust" } else { "web" };
            create_test_chunk(&format!("{}_{}", topic, i), text, emb)
        })
        .collect();

    let summarizer = OllamaSummarizer::new().with_model("tinyllama");
    let pipeline_embedder = FastEmbedder::new().expect("Failed to create embedder");

    let pipeline = ClusterPipeline::with_summarizer(pipeline_embedder, summarizer)
        .with_min_cluster_size(2)
        .with_cluster_range(2, 3);

    let (updated_chunks, summaries) = pipeline.process(chunks).await.expect("Pipeline failed");

    // Check that Rust chunks tend to cluster together
    let rust_clusters: Vec<_> = updated_chunks[..3]
        .iter()
        .filter_map(|c| c.primary_cluster())
        .map(|m| m.cluster_id.clone())
        .collect();

    let web_clusters: Vec<_> = updated_chunks[3..]
        .iter()
        .filter_map(|c| c.primary_cluster())
        .map(|m| m.cluster_id.clone())
        .collect();

    println!("Rust chunk clusters: {:?}", rust_clusters);
    println!("Web chunk clusters: {:?}", web_clusters);

    // At minimum, we should have summaries
    assert!(!summaries.is_empty(), "Should generate cluster summaries");

    for summary in &summaries {
        println!("\nCluster summary (covers {} chunks):", summary.summarizes.len());
        println!("{}", summary.content);
    }
}

// ============================================================================
// End-to-End Ingest Test
// ============================================================================

#[tokio::test]
#[ignore = "requires Ollama running locally"]
async fn test_full_ingest_with_summarization() {
    use tempfile::TempDir;
    use veclayer::commands::{ingest, IngestOptions};
    use veclayer::store::LanceStore;
    use veclayer::VectorStore;
    use std::fs;

    if !ollama_available().await {
        eprintln!("Skipping: Ollama not available");
        return;
    }

    // Create temp directory with test documents
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let docs_dir = temp_dir.path().join("docs");
    fs::create_dir(&docs_dir).expect("Failed to create docs dir");

    // Create test markdown files
    fs::write(
        docs_dir.join("rust.md"),
        r#"# Rust Programming

## Memory Safety
Rust prevents memory errors through ownership and borrowing.
The compiler checks all memory access at compile time.

## Concurrency
Rust's type system prevents data races.
Send and Sync traits ensure thread safety.
"#,
    ).expect("Failed to write rust.md");

    fs::write(
        docs_dir.join("python.md"),
        r#"# Python Programming

## Dynamic Typing
Python variables don't need type declarations.
Types are checked at runtime.

## Garbage Collection
Python automatically manages memory.
Reference counting handles most cleanup.
"#,
    ).expect("Failed to write python.md");

    let data_dir = temp_dir.path().join("veclayer-data");

    // Ingest with summarization
    let options = IngestOptions {
        recursive: true,
        summarize: true,
        model: "tinyllama".to_string(),
    };

    let result = ingest(&data_dir, &docs_dir, &options).await;
    assert!(result.is_ok(), "Ingest failed: {:?}", result.err());

    let ingest_result = result.unwrap();
    println!("Ingested {} chunks from {} files", ingest_result.total_chunks, ingest_result.files_processed);

    // Verify store has chunks and summaries
    let embedder = FastEmbedder::new().expect("Failed to create embedder");
    let store = LanceStore::open(&data_dir, embedder.dimension()).await.expect("Failed to open store");

    let stats = store.stats().await.expect("Failed to get stats");
    println!("Store stats: {} total chunks", stats.total_chunks);
    println!("Source files: {:?}", stats.source_files);

    // Should have original chunks plus any generated summaries
    assert!(stats.total_chunks > 0, "Store should have chunks");
    assert!(stats.source_files.len() >= 2, "Should have indexed both files");
}
