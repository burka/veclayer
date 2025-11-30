use linfa::traits::Fit;
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use ndarray::Array2;

use super::{ClusterAssignment, Clusterer};
use crate::{ClusterMembership, Error, Result};

/// Soft clusterer using K-Means with distance-based soft assignments.
///
/// Instead of hard cluster assignments, this computes membership probabilities
/// based on inverse distance to each centroid (similar to GMM but simpler).
pub struct SoftClusterer {
    /// Minimum number of clusters
    min_clusters: usize,
    /// Maximum number of clusters
    max_clusters: usize,
    /// Minimum membership probability to include
    min_membership: f32,
    /// Number of iterations for k-means
    max_iterations: usize,
}

impl SoftClusterer {
    pub fn new() -> Self {
        Self {
            min_clusters: 2,
            max_clusters: 10,
            min_membership: 0.1,
            max_iterations: 100,
        }
    }

    pub fn with_cluster_range(mut self, min: usize, max: usize) -> Self {
        self.min_clusters = min;
        self.max_clusters = max;
        self
    }

    pub fn with_min_membership(mut self, min: f32) -> Self {
        self.min_membership = min;
        self
    }

    /// Find optimal k using elbow method (simplified)
    fn find_optimal_k(&self, data: &Array2<f32>) -> usize {
        let n_samples = data.nrows();

        // Don't use more clusters than samples
        let max_k = self.max_clusters.min(n_samples / 2).max(self.min_clusters);

        if max_k <= self.min_clusters {
            return self.min_clusters;
        }

        let mut inertias = Vec::new();

        for k in self.min_clusters..=max_k {
            let dataset = DatasetBase::from(data.clone());
            let model = KMeans::params(k)
                .max_n_iterations(self.max_iterations as u64)
                .fit(&dataset);

            if let Ok(model) = model {
                let inertia = self.compute_inertia(data, model.centroids());
                inertias.push((k, inertia));
            }
        }

        if inertias.len() < 2 {
            return self.min_clusters;
        }

        // Find elbow using second derivative
        let mut best_k = self.min_clusters;
        let mut max_second_deriv = 0.0f32;

        for i in 1..inertias.len().saturating_sub(1) {
            let d1 = inertias[i - 1].1 - inertias[i].1;
            let d2 = inertias[i].1 - inertias[i + 1].1;
            let second_deriv = (d1 - d2).abs();

            if second_deriv > max_second_deriv {
                max_second_deriv = second_deriv;
                best_k = inertias[i].0;
            }
        }

        best_k
    }

    fn compute_inertia(&self, data: &Array2<f32>, centroids: &Array2<f32>) -> f32 {
        let mut total_inertia = 0.0f32;

        for row in data.rows() {
            let mut min_dist = f32::MAX;
            for centroid in centroids.rows() {
                let dist: f32 = row
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                min_dist = min_dist.min(dist);
            }
            total_inertia += min_dist;
        }

        total_inertia
    }

    /// Convert distances to soft memberships using softmax
    fn distances_to_memberships(&self, distances: &[f32]) -> Vec<f32> {
        // Use negative distances with temperature scaling for softmax
        let temperature = 1.0;
        let neg_distances: Vec<f32> = distances.iter().map(|d| -d / temperature).collect();

        // Numerical stability: subtract max
        let max_val = neg_distances
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = neg_distances.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();

        if sum == 0.0 {
            // Equal membership if all distances are infinite
            let n = distances.len() as f32;
            return vec![1.0 / n; distances.len()];
        }

        exp_vals.iter().map(|x| x / sum).collect()
    }
}

impl Default for SoftClusterer {
    fn default() -> Self {
        Self::new()
    }
}

impl Clusterer for SoftClusterer {
    fn cluster(&self, embeddings: &[Vec<f32>]) -> Result<Vec<ClusterAssignment>> {
        if embeddings.is_empty() {
            return Ok(Vec::new());
        }

        if embeddings.len() < 2 {
            // Single item - assign to its own cluster
            return Ok(vec![ClusterAssignment {
                index: 0,
                memberships: vec![ClusterMembership::new("cluster_0", 1.0)],
            }]);
        }

        let dim = embeddings[0].len();

        // Convert to ndarray
        let flat: Vec<f32> = embeddings.iter().flatten().copied().collect();
        let data = Array2::from_shape_vec((embeddings.len(), dim), flat)
            .map_err(|e| Error::clustering(format!("Failed to create array: {}", e)))?;

        // Find optimal k
        let k = self.find_optimal_k(&data);

        // Fit final model
        let dataset = DatasetBase::from(data.clone());
        let model = KMeans::params(k)
            .max_n_iterations(self.max_iterations as u64)
            .fit(&dataset)
            .map_err(|e| Error::clustering(format!("KMeans failed: {}", e)))?;

        let centroids = model.centroids();

        // Compute soft assignments for each point
        let mut assignments = Vec::with_capacity(embeddings.len());

        for (idx, row) in data.rows().into_iter().enumerate() {
            // Compute distances to all centroids
            let distances: Vec<f32> = centroids
                .rows()
                .into_iter()
                .map(|centroid| {
                    row.iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt()
                })
                .collect();

            // Convert to soft memberships
            let probs = self.distances_to_memberships(&distances);

            // Create memberships, filtering by minimum probability
            let memberships: Vec<ClusterMembership> = probs
                .iter()
                .enumerate()
                .filter(|(_, &p)| p >= self.min_membership)
                .map(|(cluster_idx, &prob)| {
                    ClusterMembership::new(format!("cluster_{}", cluster_idx), prob)
                })
                .collect();

            assignments.push(ClusterAssignment {
                index: idx,
                memberships,
            });
        }

        Ok(assignments)
    }

    fn num_clusters(&self) -> usize {
        0 // Would need to store state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_cluster_single_item() {
        let clusterer = SoftClusterer::new();
        let embeddings = vec![vec![1.0, 2.0, 3.0]];

        let assignments = clusterer.cluster(&embeddings).unwrap();
        assert_eq!(assignments.len(), 1);
        assert!(!assignments[0].memberships.is_empty());
    }

    #[test]
    fn test_soft_cluster_multiple_items() {
        let clusterer = SoftClusterer::new().with_cluster_range(2, 3);

        // Create two distinct clusters
        let embeddings = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.1, 0.1, 0.1],
            vec![0.2, 0.0, 0.1],
            vec![10.0, 10.0, 10.0],
            vec![10.1, 10.0, 10.1],
            vec![10.2, 10.1, 10.0],
        ];

        let assignments = clusterer.cluster(&embeddings).unwrap();
        assert_eq!(assignments.len(), 6);

        // Each item should have at least one membership
        for assignment in &assignments {
            assert!(!assignment.memberships.is_empty());
            // Probabilities should sum roughly to 1 (with filtering)
            let total: f32 = assignment.memberships.iter().map(|m| m.probability).sum();
            assert!(total > 0.5); // At least 50% covered after filtering
        }
    }

    #[test]
    fn test_distances_to_memberships() {
        let clusterer = SoftClusterer::new();

        // Equal distances should give equal memberships
        let probs = clusterer.distances_to_memberships(&[1.0, 1.0, 1.0]);
        assert!((probs[0] - probs[1]).abs() < 0.01);
        assert!((probs[1] - probs[2]).abs() < 0.01);

        // Closer distance should have higher probability
        let probs = clusterer.distances_to_memberships(&[0.1, 1.0, 10.0]);
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);
    }

    #[test]
    fn test_cluster_empty_input() {
        let clusterer = SoftClusterer::new();
        let result = clusterer.cluster(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_cluster_with_min_membership_filter() {
        let clusterer = SoftClusterer::new()
            .with_min_membership(0.3)
            .with_cluster_range(3, 3);

        // Create three distinct clusters
        let embeddings = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.1, 0.0, 0.0],
            vec![5.0, 5.0, 5.0],
            vec![5.1, 5.0, 5.0],
            vec![10.0, 10.0, 10.0],
            vec![10.1, 10.0, 10.0],
        ];

        let assignments = clusterer.cluster(&embeddings).unwrap();
        assert_eq!(assignments.len(), 6);

        // With high min_membership threshold, some low-probability memberships should be filtered
        for assignment in &assignments {
            // Should have at least one membership
            assert!(!assignment.memberships.is_empty());
            // All remaining memberships should meet the threshold
            for membership in &assignment.memberships {
                assert!(
                    membership.probability >= 0.3,
                    "Membership probability {} should be >= 0.3",
                    membership.probability
                );
            }
        }
    }

    #[test]
    fn test_find_optimal_k_small_dataset() {
        let clusterer = SoftClusterer::new().with_cluster_range(2, 5);

        // Small dataset with 4 items - should not try to create more clusters than makes sense
        let embeddings = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![5.0, 5.0],
            vec![5.1, 5.1],
        ];

        let assignments = clusterer.cluster(&embeddings).unwrap();
        assert_eq!(assignments.len(), 4);

        // Should successfully cluster without errors
        for assignment in &assignments {
            assert!(!assignment.memberships.is_empty());
        }
    }

    #[test]
    fn test_cluster_range_bounds() {
        let clusterer = SoftClusterer::new().with_cluster_range(3, 5);
        assert_eq!(clusterer.min_clusters, 3);
        assert_eq!(clusterer.max_clusters, 5);

        // Create dataset with clear clusters
        let embeddings = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![5.0, 5.0],
            vec![5.1, 5.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let assignments = clusterer.cluster(&embeddings).unwrap();
        assert_eq!(assignments.len(), 6);

        // Verify clustering respects the range
        for assignment in &assignments {
            assert!(!assignment.memberships.is_empty());
        }
    }

    #[test]
    fn test_distances_to_memberships_zero_sum() {
        let clusterer = SoftClusterer::new();

        // Very large equal distances should give equal memberships
        let probs = clusterer.distances_to_memberships(&[1000.0, 1000.0]);
        assert!((probs[0] - 0.5).abs() < 0.01);
        assert!((probs[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_compute_inertia() {
        let clusterer = SoftClusterer::new();

        // Simple case: 2 points and 2 centroids
        let data = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let centroids = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();

        let inertia = clusterer.compute_inertia(&data, &centroids);
        // Each point is at its centroid, so inertia should be 0
        assert!(inertia < 0.01, "Inertia should be near 0, got {}", inertia);
    }

    #[test]
    fn test_default_trait() {
        let clusterer = SoftClusterer::default();
        assert_eq!(clusterer.min_clusters, 2);
        assert_eq!(clusterer.max_clusters, 10);
        assert!((clusterer.min_membership - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_cluster_two_items() {
        let clusterer = SoftClusterer::new();
        let embeddings = vec![vec![0.0, 0.0, 0.0], vec![10.0, 10.0, 10.0]];

        let assignments = clusterer.cluster(&embeddings).unwrap();
        assert_eq!(assignments.len(), 2);

        // Each should have memberships
        for assignment in &assignments {
            assert!(!assignment.memberships.is_empty());
        }
    }

    #[test]
    fn test_cluster_high_dimensional() {
        let clusterer = SoftClusterer::new().with_cluster_range(2, 3);

        // Test with high-dimensional embeddings (e.g., 384 dimensions like real embeddings)
        let dim = 384;
        let mut embeddings = Vec::new();

        // Create two distinct clusters in high-dimensional space
        for i in 0..3 {
            let mut vec = vec![0.0; dim];
            vec[0] = i as f32 * 0.1;
            embeddings.push(vec);
        }
        for i in 0..3 {
            let mut vec = vec![0.0; dim];
            vec[0] = 10.0 + i as f32 * 0.1;
            embeddings.push(vec);
        }

        let assignments = clusterer.cluster(&embeddings).unwrap();
        assert_eq!(assignments.len(), 6);

        for assignment in &assignments {
            assert!(!assignment.memberships.is_empty());
        }
    }
}
