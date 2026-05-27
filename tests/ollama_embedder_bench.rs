//! OllamaEmbedder timing benchmark — locks in the "external GPU embedder is the
//! fast path" claim. Counterpart to `embedder_bench.rs` (which times the local
//! CPU FastEmbed backend).
//!
//! Run with: `cargo test --test ollama_embedder_bench --features full -- --ignored --nocapture`
//!
//! Requires a running Ollama at `http://localhost:11434` with the
//! `nomic-embed-text` model pulled (`ollama pull nomic-embed-text`). If the
//! model is missing the tests skip with a clear message rather than failing.

use std::time::{Duration, Instant};

use veclayer::embedder::OllamaEmbedder;
use veclayer::Embedder;

const SAMPLE_TEXT: &str = "The quick brown fox jumps over the lazy dog near the riverbank.";
const MODEL: &str = "nomic-embed-text";
const BASE_URL: &str = "http://localhost:11434";
const DIMENSION: usize = 768;

/// Build an embedder and confirm Ollama answers. Returns `None` (skip) when the
/// service is unreachable or the model is not pulled.
async fn try_warm_embedder() -> Option<OllamaEmbedder> {
    let embedder = OllamaEmbedder::new(MODEL, BASE_URL, DIMENSION).expect("create OllamaEmbedder");
    match embedder.embed(&[SAMPLE_TEXT]).await {
        Ok(_) => Some(embedder),
        Err(e) => {
            eprintln!(
                "SKIP: Ollama unavailable or model '{MODEL}' not pulled ({e}).\n      \
                 Start Ollama and run `ollama pull {MODEL}` to exercise this bench."
            );
            None
        }
    }
}

async fn time_embed(embedder: &OllamaEmbedder, texts: &[&str]) -> Duration {
    let start = Instant::now();
    let out = embedder.embed(texts).await.expect("embed");
    let elapsed = start.elapsed();
    assert_eq!(out.len(), texts.len());
    elapsed
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires a running Ollama with nomic-embed-text; run with --ignored"]
async fn ollama_single_embed_under_50ms() {
    let Some(embedder) = try_warm_embedder().await else {
        return;
    };

    let runs = 5;
    let mut total = Duration::ZERO;
    for _ in 0..runs {
        total += time_embed(&embedder, &[SAMPLE_TEXT]).await;
    }
    let avg = total / runs;
    eprintln!("ollama single embed avg over {runs} runs: {avg:?}");

    assert!(
        avg < Duration::from_millis(50),
        "single embed averaged {avg:?}, expected <50ms on GPU Ollama"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires a running Ollama with nomic-embed-text; run with --ignored"]
async fn ollama_batch_of_32_under_300ms() {
    let Some(embedder) = try_warm_embedder().await else {
        return;
    };

    let batch: Vec<&str> = (0..32).map(|_| SAMPLE_TEXT).collect();

    let runs = 3;
    let mut total = Duration::ZERO;
    for _ in 0..runs {
        total += time_embed(&embedder, &batch).await;
    }
    let avg = total / runs;
    eprintln!(
        "ollama batch-32 avg over {runs} runs: {avg:?} ({:?} per text amortized)",
        avg / 32
    );

    assert!(
        avg < Duration::from_millis(300),
        "batch-of-32 averaged {avg:?}, expected <300ms on GPU Ollama"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires a running Ollama with nomic-embed-text; run with --ignored"]
async fn ollama_cold_call_under_500ms() {
    // First call pays format-probe (try /api/embed) + any model load. Measure it
    // separately from the warm steady-state above.
    let embedder = OllamaEmbedder::new(MODEL, BASE_URL, DIMENSION).expect("create OllamaEmbedder");
    let start = Instant::now();
    let result = embedder.embed(&[SAMPLE_TEXT]).await;
    let cold = start.elapsed();

    if let Err(e) = result {
        eprintln!(
            "SKIP: Ollama unavailable or model '{MODEL}' not pulled ({e}).\n      \
             Start Ollama and run `ollama pull {MODEL}` to exercise this bench."
        );
        return;
    }
    eprintln!("ollama cold call (includes format probe): {cold:?}");

    assert!(
        cold < Duration::from_millis(500),
        "cold call took {cold:?}, expected <500ms"
    );
}
