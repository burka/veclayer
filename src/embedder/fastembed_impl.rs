use std::sync::{Arc, Mutex};

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use super::Embedder;
use crate::{Error, Result};

/// Returns the fastembed model cache directory inside VecLayer's cache path.
fn fastembed_cache_dir() -> std::path::PathBuf {
    crate::default_cache_dir().join("fastembed")
}

/// Lazily initialized cache that, unlike `OnceLock`, does **not** cache a
/// failed initialization. A successful value is cached forever; a failed
/// attempt leaves the cache empty so the next caller retries. This matters for
/// model init, where the first attempt can fail transiently (e.g. a network
/// blip while downloading the ONNX weights) and must not brick the embedder
/// for the rest of the process lifetime.
struct RetryCache<T> {
    cell: Mutex<Option<Arc<T>>>,
}

impl<T> RetryCache<T> {
    fn new() -> Self {
        Self {
            cell: Mutex::new(None),
        }
    }

    /// Return the cached value, or run `init` to produce one. On success the
    /// value is cached and returned; on error nothing is cached and the error
    /// is propagated so a later call can retry.
    fn get_or_try_init<E>(
        &self,
        init: impl FnOnce() -> std::result::Result<T, E>,
    ) -> std::result::Result<Arc<T>, E> {
        // Recover from a poisoned lock instead of propagating the panic. If a
        // prior `init` panicked while holding the guard, the cache is still
        // empty (the value is only stored after `init` returns Ok), so the
        // recovered state is consistent and this caller can simply retry —
        // panicking here would permanently brick the embedder, the very
        // failure mode this cache exists to prevent.
        let mut guard = self.cell.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(value) = guard.as_ref() {
            return Ok(Arc::clone(value));
        }
        let value = Arc::new(init()?);
        *guard = Some(Arc::clone(&value));
        Ok(value)
    }
}

/// FastEmbed-based embedder using local ONNX models.
/// Runs entirely on CPU, no external API required.
///
/// The underlying ONNX session is initialized lazily on first real embed call.
/// This keeps MCP stdio startup cheap for short-lived sessions that only need
/// initialization metadata and never actually run semantic operations.
pub struct FastEmbedder {
    model: RetryCache<TextEmbedding>,
    model_type: EmbeddingModel,
    cache_dir: std::path::PathBuf,
    dimension: usize,
    model_name: String,
}

impl FastEmbedder {
    /// Create a new FastEmbedder with the default model (BGE-small-en-v1.5)
    pub fn new() -> Result<Self> {
        Self::with_model(EmbeddingModel::BGESmallENV15)
    }

    /// Create a FastEmbedder with a specific model
    pub fn with_model(model_type: EmbeddingModel) -> Result<Self> {
        let model_name = format!("{:?}", model_type);
        let dimension = Self::model_dimension(&model_type)?;

        Ok(Self {
            model: RetryCache::new(),
            model_type,
            cache_dir: fastembed_cache_dir(),
            dimension,
            model_name,
        })
    }

    fn get_or_init_model(&self) -> Result<Arc<TextEmbedding>> {
        self.model.get_or_try_init(|| {
            tracing::debug!("Initializing FastEmbed model {}", self.model_name);
            let options =
                InitOptions::new(self.model_type.clone()).with_cache_dir(self.cache_dir.clone());
            TextEmbedding::try_new(options)
                .map_err(|e| Error::embedding(format!("Failed to initialize FastEmbed: {e}")))
        })
    }

    /// Resolve a model's embedding dimension from fastembed's own metadata.
    ///
    /// This is the single source of truth for every supported model, so adding
    /// or upgrading an upstream model needs no change here. Fails fast for a
    /// model fastembed itself does not recognize rather than silently assuming
    /// a dimension that would later corrupt the vector store.
    fn model_dimension(model: &EmbeddingModel) -> Result<usize> {
        TextEmbedding::get_model_info(model)
            .map(|info| info.dim)
            .map_err(|e| Error::embedding(format!("Unsupported embedding model {model:?}: {e}")))
    }
}

impl Embedder for FastEmbedder {
    fn embed<'a>(
        &'a self,
        texts: &'a [&'a str],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + 'a>>
    {
        let owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        Box::pin(async move {
            if owned.is_empty() {
                return Ok(vec![]);
            }
            let model = self.get_or_init_model()?;
            tokio::task::spawn_blocking(move || {
                model
                    .embed(owned, None)
                    .map_err(|e| Error::embedding(format!("Embedding failed: {}", e)))
            })
            .await
            .map_err(|e| Error::embedding(format!("Embedding task panicked: {}", e)))?
        })
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        &self.model_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn retry_cache_caches_success_and_runs_init_once() {
        let cache: RetryCache<u32> = RetryCache::new();
        let calls = Cell::new(0);
        let init = || -> std::result::Result<u32, &'static str> {
            calls.set(calls.get() + 1);
            Ok(42)
        };

        let first = cache.get_or_try_init(init).unwrap();
        let second = cache.get_or_try_init(init).unwrap();

        assert_eq!(*first, 42);
        assert_eq!(*second, 42);
        assert!(
            Arc::ptr_eq(&first, &second),
            "must return the same cached Arc"
        );
        assert_eq!(calls.get(), 1, "init must run exactly once after success");
    }

    #[test]
    fn retry_cache_does_not_cache_failure_and_retries() {
        let cache: RetryCache<u32> = RetryCache::new();
        let calls = Cell::new(0);

        // First attempt fails: nothing must be cached.
        let first = cache.get_or_try_init(|| -> std::result::Result<u32, &'static str> {
            calls.set(calls.get() + 1);
            Err("transient failure")
        });
        assert_eq!(first, Err("transient failure"));
        assert_eq!(calls.get(), 1);

        // Second attempt succeeds: the earlier failure must not be cached, so
        // init runs again and the value is now stored.
        let second = cache
            .get_or_try_init(|| -> std::result::Result<u32, &'static str> {
                calls.set(calls.get() + 1);
                Ok(7)
            })
            .unwrap();
        assert_eq!(*second, 7);
        assert_eq!(
            calls.get(),
            2,
            "failed init must not be cached; retry runs init again"
        );

        // Third attempt must hit the cache without re-running init.
        let third = cache
            .get_or_try_init(|| -> std::result::Result<u32, &'static str> {
                calls.set(calls.get() + 1);
                Ok(999)
            })
            .unwrap();
        assert_eq!(*third, 7, "success must be cached after the retry");
        assert_eq!(calls.get(), 2, "init must not run once a value is cached");
    }

    #[test]
    fn retry_cache_serializes_concurrent_first_callers() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc as StdArc;

        // The mutex serializes callers, so with init failing exactly once the
        // first thread to acquire the lock fails (nothing cached) and the next
        // succeeds and caches; all remaining threads hit the cache. Hence init
        // runs exactly twice and exactly one caller observes the failure,
        // regardless of thread scheduling.
        let cache: StdArc<RetryCache<u32>> = StdArc::new(RetryCache::new());
        let calls = StdArc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let cache = StdArc::clone(&cache);
            let calls = StdArc::clone(&calls);
            handles.push(std::thread::spawn(move || {
                cache.get_or_try_init(|| -> std::result::Result<u32, &'static str> {
                    if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                        Err("first attempt fails")
                    } else {
                        Ok(123)
                    }
                })
            }));
        }

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let failures = results.iter().filter(|r| r.is_err()).count();
        let successes = results.iter().filter(|r| r.is_ok()).count();

        assert_eq!(failures, 1, "exactly one caller must observe the failure");
        assert_eq!(successes, 7, "every other caller must get the cached value");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "init must run once (fail) then once more (succeed), never again"
        );
        for r in results.into_iter().flatten() {
            assert_eq!(*r, 123, "all successful callers see the same cached value");
        }
    }

    #[test]
    #[ignore = "requires ONNX model file (download via fastembed)"]
    fn test_fastembed_creation() {
        let embedder = FastEmbedder::new();
        assert!(embedder.is_ok());
    }

    #[tokio::test]
    #[ignore = "requires ONNX model file (download via fastembed)"]
    async fn test_fastembed_embed() {
        let embedder = FastEmbedder::new().unwrap();
        let texts = vec!["Hello world", "This is a test"];
        let embeddings = embedder.embed(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), embedder.dimension());
        assert_eq!(embeddings[1].len(), embedder.dimension());
    }

    #[tokio::test]
    #[ignore = "requires ONNX model file (download via fastembed)"]
    async fn test_fastembed_empty() {
        let embedder = FastEmbedder::new().unwrap();
        let texts: Vec<&str> = vec![];
        let embeddings = embedder.embed(&texts).await.unwrap();
        assert!(embeddings.is_empty());
    }

    #[test]
    fn test_with_model_resolves_known_dimensions() {
        // Dimensions come from fastembed's own metadata, so these assert the
        // real upstream contract rather than a hand-maintained table. Building
        // the embedder is cheap: the ONNX session is initialized lazily, so no
        // model is downloaded here. NomicEmbedTextV15 is included specifically
        // because it is a fully supported model (dim 768) that an earlier
        // hand-rolled match wrongly rejected.
        let cases = [
            (EmbeddingModel::BGESmallENV15, 384),
            (EmbeddingModel::BGEBaseENV15, 768),
            (EmbeddingModel::BGELargeENV15, 1024),
            (EmbeddingModel::AllMiniLML6V2, 384),
            (EmbeddingModel::AllMiniLML12V2, 384),
            (EmbeddingModel::NomicEmbedTextV15, 768),
        ];
        for (model, expected) in cases {
            let embedder = FastEmbedder::with_model(model.clone())
                .unwrap_or_else(|e| panic!("with_model({model:?}) should succeed: {e}"));
            assert_eq!(
                embedder.dimension(),
                expected,
                "unexpected dimension for {model:?}"
            );
            assert_eq!(embedder.name(), format!("{model:?}"));
        }
    }

    // NOTE: the error path of `model_dimension` (fastembed not recognizing a
    // model) is unreachable from a test today: every `EmbeddingModel` enum
    // variant has metadata in fastembed 4.9.1, and the enum is the only way to
    // construct a model. The `Result` is retained as defensive handling for a
    // future fastembed that ships an enum variant without metadata, so we fail
    // fast instead of unwrapping upstream's `Result`.
}
