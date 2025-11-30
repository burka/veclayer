mod pipeline;
mod soft_cluster;

pub use pipeline::ClusterPipeline;
pub use soft_cluster::SoftClusterer;

use crate::{ClusterMembership, Result};

/// Result of clustering embeddings
#[derive(Debug, Clone)]
pub struct ClusterAssignment {
    /// Index of the original item
    pub index: usize,
    /// Cluster memberships with probabilities
    pub memberships: Vec<ClusterMembership>,
}

/// Trait for clustering embeddings with soft assignments
pub trait Clusterer: Send + Sync {
    /// Cluster the given embeddings and return soft assignments.
    /// Each embedding can belong to multiple clusters with different probabilities.
    fn cluster(&self, embeddings: &[Vec<f32>]) -> Result<Vec<ClusterAssignment>>;

    /// Get the number of clusters that were found
    fn num_clusters(&self) -> usize;
}
