mod ollama;

pub use ollama::OllamaSummarizer;

use std::future::Future;

use crate::Result;

/// Trait for generating text summaries.
/// Used to summarize clusters of related chunks.
pub trait Summarizer: Send + Sync {
    /// Generate a summary for the given texts.
    fn summarize(&self, texts: &[&str]) -> impl Future<Output = Result<String>> + Send;

    /// Generate summaries for multiple groups of texts.
    /// Default implementation calls summarize for each group sequentially.
    fn summarize_batch(
        &self,
        text_groups: Vec<Vec<&str>>,
    ) -> impl Future<Output = Result<Vec<String>>> + Send {
        async move {
            let mut summaries = Vec::with_capacity(text_groups.len());
            for group in text_groups {
                let summary = self.summarize(&group).await?;
                summaries.push(summary);
            }
            Ok(summaries)
        }
    }

    /// Get the name/model identifier of this summarizer
    fn name(&self) -> &str;
}

// Arc implementation for trait objects
crate::arc_impl!(Summarizer {
    fn summarize(&self, texts: &[&str]) -> impl Future<Output = Result<String>> + Send;
    fn summarize_batch(&self, text_groups: Vec<Vec<&str>>) -> impl Future<Output = Result<Vec<String>>> + Send;
    fn name(&self) -> &str;
});

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use super::Summarizer;
    use crate::{Error, Result};

    // ── Mock ─────────────────────────────────────────────────────────────────

    /// A minimal `Summarizer` implementation for hermetic unit tests.
    ///
    /// * `fail_on_call` – if `Some(n)`, the n-th call to `summarize` (1-based)
    ///   returns `Err`; all others succeed.
    /// * `call_count` – incremented once per `summarize` invocation so tests can
    ///   assert that processing stopped at the failing group.
    struct MockSummarizer {
        fail_on_call: Option<usize>,
        call_count: AtomicUsize,
    }

    impl MockSummarizer {
        fn always_ok() -> Self {
            Self {
                fail_on_call: None,
                call_count: AtomicUsize::new(0),
            }
        }

        fn fail_on(n: usize) -> Self {
            Self {
                fail_on_call: Some(n),
                call_count: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    impl Summarizer for MockSummarizer {
        async fn summarize(&self, texts: &[&str]) -> Result<String> {
            let call = self.call_count.fetch_add(1, Ordering::SeqCst) + 1;
            if self.fail_on_call == Some(call) {
                return Err(Error::summarization(format!("mock failure on call {call}")));
            }
            Ok(texts.join("|"))
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    // ── Happy path ────────────────────────────────────────────────────────────

    /// `summarize_batch` over N groups returns N summaries in the correct order.
    #[tokio::test]
    async fn batch_returns_summaries_in_order() {
        let s = MockSummarizer::always_ok();
        let groups = vec![
            vec!["alpha", "beta"],
            vec!["gamma"],
            vec!["delta", "epsilon", "zeta"],
        ];
        let result = s.summarize_batch(groups).await.unwrap();
        assert_eq!(result, vec!["alpha|beta", "gamma", "delta|epsilon|zeta"]);
        assert_eq!(s.calls(), 3);
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    /// An empty list of groups returns `Ok(vec![])` without calling `summarize`.
    #[tokio::test]
    async fn batch_empty_groups_returns_empty_vec() {
        let s = MockSummarizer::always_ok();
        let result = s.summarize_batch(vec![]).await.unwrap();
        assert!(result.is_empty());
        assert_eq!(s.calls(), 0, "summarize must not be called for zero groups");
    }

    /// A group containing empty-string texts is forwarded to `summarize` as-is.
    /// The default impl does not filter out blank strings — that is the
    /// responsibility of the concrete summarizer.
    #[tokio::test]
    async fn batch_group_with_empty_strings_is_forwarded() {
        let s = MockSummarizer::always_ok();
        let groups = vec![vec!["", "hello", ""]];
        let result = s.summarize_batch(groups).await.unwrap();
        // MockSummarizer joins with '|', so blanks appear as leading/trailing pipes.
        assert_eq!(result, vec!["|hello|"]);
        assert_eq!(s.calls(), 1);
    }

    /// A single group whose only text is an empty string is still forwarded.
    #[tokio::test]
    async fn batch_single_group_all_empty_strings() {
        let s = MockSummarizer::always_ok();
        let groups = vec![vec![""]];
        let result = s.summarize_batch(groups).await.unwrap();
        assert_eq!(result, vec![""]);
        assert_eq!(s.calls(), 1);
    }

    // ── Error propagation ─────────────────────────────────────────────────────

    /// When `summarize` fails on group k, `summarize_batch` short-circuits:
    /// it returns `Err` immediately and does NOT call `summarize` for any later group.
    #[tokio::test]
    async fn batch_short_circuits_on_error() {
        // Fail on the 2nd call; there are 4 groups total.
        let s = MockSummarizer::fail_on(2);
        let groups = vec![
            vec!["first"],
            vec!["second"], // this call fails
            vec!["third"],
            vec!["fourth"],
        ];
        let result = s.summarize_batch(groups).await;
        assert!(result.is_err(), "expected Err from failing summarize");
        // Exactly 2 calls: one success, one failure — then stopped.
        assert_eq!(
            s.calls(),
            2,
            "summarize must stop after the failing group (2nd call)"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("mock failure on call 2"),
            "error message should identify failing call; got: {msg}"
        );
    }

    /// When the very first group fails, the result vec is empty (no partial results).
    #[tokio::test]
    async fn batch_error_on_first_group_produces_no_partial_results() {
        let s = MockSummarizer::fail_on(1);
        let groups = vec![vec!["only"], vec!["never reached"]];
        let result = s.summarize_batch(groups).await;
        assert!(result.is_err());
        assert_eq!(s.calls(), 1, "only the failing first call should occur");
    }

    // ── Arc forwarding ────────────────────────────────────────────────────────

    /// The arc_impl!-generated impl forwards all three methods through Arc<T>.
    ///
    /// Because the Summarizer trait uses RPITIT (return-position `impl Trait`),
    /// it is not object-safe and cannot be used as `dyn Summarizer`.  The
    /// arc_impl! macro generates `impl<T: Summarizer + ?Sized> Summarizer for
    /// Arc<T>` which works for concrete Arc<MockSummarizer>.
    #[tokio::test]
    async fn arc_forwarding_calls_inner_impl() {
        let s: Arc<MockSummarizer> = Arc::new(MockSummarizer::always_ok());

        // name()
        assert_eq!(s.name(), "mock");

        // summarize()
        let summary = s.summarize(&["x", "y"]).await.unwrap();
        assert_eq!(summary, "x|y");

        // summarize_batch() — exercises the default impl through the Arc layer
        let batch = s
            .summarize_batch(vec![vec!["a", "b"], vec!["c"]])
            .await
            .unwrap();
        assert_eq!(batch, vec!["a|b", "c"]);
    }
}
