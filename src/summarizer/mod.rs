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

/// Arc implementation for trait objects
impl<T: Summarizer> Summarizer for std::sync::Arc<T> {
    fn summarize(&self, texts: &[&str]) -> impl Future<Output = Result<String>> + Send {
        (**self).summarize(texts)
    }

    fn summarize_batch(
        &self,
        text_groups: Vec<Vec<&str>>,
    ) -> impl Future<Output = Result<Vec<String>>> + Send {
        (**self).summarize_batch(text_groups)
    }

    fn name(&self) -> &str {
        (**self).name()
    }
}
