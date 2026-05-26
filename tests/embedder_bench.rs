//! Embedder timing benchmark — TDD-style assertion of the latency budget
//! encoded in `src/mcp/embed_worker.rs::EMBED_TIME_SECS` (= 2s per batch of
//! up to 32 entries).
//!
//! Run with: `cargo test --test embedder_bench -- --ignored --nocapture`
//!
//! The first invocation downloads the BGE-small ONNX model into the VecLayer
//! cache dir (~90 MB). Subsequent runs reuse it.
//!
//! Requires the `embedding-local` feature (provides `FastEmbedder`); the whole
//! file compiles to nothing without it.
#![cfg(feature = "embedding-local")]

use std::time::Instant;

use veclayer::embedder::FastEmbedder;
use veclayer::Embedder;

const SAMPLE_TEXT: &str = "The quick brown fox jumps over the lazy dog near the riverbank.";

fn warm_embedder() -> FastEmbedder {
    let embedder = FastEmbedder::new().expect("create FastEmbedder");
    // Force model init + warm CPU caches off the timing path.
    let _ = embedder.embed(&[SAMPLE_TEXT]).expect("warmup embed");
    embedder
}

fn time_embed(embedder: &FastEmbedder, texts: &[&str]) -> std::time::Duration {
    let start = Instant::now();
    let out = embedder.embed(texts).expect("embed");
    let elapsed = start.elapsed();
    assert_eq!(out.len(), texts.len());
    elapsed
}

#[test]
#[ignore = "downloads ~90MB model on first run; run explicitly with --ignored"]
fn embed_single_text_under_100ms() {
    let embedder = warm_embedder();

    let runs = 5;
    let mut total = std::time::Duration::ZERO;
    for _ in 0..runs {
        total += time_embed(&embedder, &[SAMPLE_TEXT]);
    }
    let avg = total / runs;
    eprintln!("single embed avg over {runs} runs: {avg:?}");

    // Code's implicit claim: per-text amortized should be tens of ms, not hundreds.
    assert!(
        avg < std::time::Duration::from_millis(100),
        "single embed averaged {avg:?}, expected <100ms"
    );
}

#[test]
#[ignore = "downloads ~90MB model on first run; run explicitly with --ignored"]
fn embed_batch_of_32_under_500ms() {
    let embedder = warm_embedder();

    let batch: Vec<&str> = (0..32).map(|_| SAMPLE_TEXT).collect();

    let runs = 3;
    let mut total = std::time::Duration::ZERO;
    for _ in 0..runs {
        total += time_embed(&embedder, &batch);
    }
    let avg = total / runs;
    eprintln!(
        "batch-32 avg over {runs} runs: {avg:?} ({:?} per text amortized)",
        avg / 32
    );

    // Backs the EMBED_TIME_SECS = 1 budget in embed_worker.rs:19. Measured
    // ~129ms on CPU; 500ms keeps ~4x headroom for slower machines.
    assert!(
        avg < std::time::Duration::from_millis(500),
        "batch-of-32 averaged {avg:?}, expected <500ms (the EMBED_TIME_SECS budget)"
    );
}

#[test]
#[ignore = "downloads ~90MB model on first run; run explicitly with --ignored"]
fn embed_cold_init_under_one_second() {
    // Measure the first-call cost (model load + first inference) separately,
    // since OnceLock init happens inside the first embed() call.
    let embedder = FastEmbedder::new().expect("create FastEmbedder");
    let start = Instant::now();
    let _ = embedder.embed(&[SAMPLE_TEXT]).expect("cold embed");
    let cold = start.elapsed();
    eprintln!("cold embed (includes model init): {cold:?}");

    assert!(
        cold < std::time::Duration::from_secs(1),
        "cold init+embed took {cold:?}, expected <1s"
    );
}
