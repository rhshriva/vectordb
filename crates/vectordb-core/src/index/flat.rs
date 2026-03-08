/// Flat (brute-force) exact index.
///
/// Stores all vectors in a `Vec` and performs an exhaustive linear scan on every
/// query.  100% recall by definition.  O(N·D) per query.
///
/// Use this for:
/// - Datasets < ~100 K vectors where latency is acceptable
/// - Ground-truth / recall evaluation of approximate indexes
/// - Unit-testing pipelines without ANN noise
use std::collections::HashMap;

use crate::{
    distance::Metric,
    error::VectorDbError,
    index::{IndexConfig, SearchResult, VectorIndex},
};

pub struct FlatIndex {
    config: IndexConfig,
    /// Map from caller-supplied ID → raw vector data.
    vectors: HashMap<u64, Vec<f32>>,
}

impl FlatIndex {
    pub fn new(dimensions: usize, metric: Metric) -> Self {
        Self {
            config: IndexConfig { dimensions, metric },
            vectors: HashMap::new(),
        }
    }

    /// Pre-allocate capacity for `n` vectors.
    pub fn with_capacity(dimensions: usize, metric: Metric, n: usize) -> Self {
        Self {
            config: IndexConfig { dimensions, metric },
            vectors: HashMap::with_capacity(n),
        }
    }
}

impl VectorIndex for FlatIndex {
    fn add(&mut self, id: u64, vector: &[f32]) -> Result<(), VectorDbError> {
        if vector.len() != self.config.dimensions {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }
        if self.vectors.contains_key(&id) {
            return Err(VectorDbError::DuplicateId(id));
        }
        self.vectors.insert(id, vector.to_vec());
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, VectorDbError> {
        if query.len() != self.config.dimensions {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len(),
            });
        }
        if k == 0 {
            return Ok(vec![]);
        }

        let metric = self.config.metric;

        // Compute distances to all stored vectors.
        let mut distances: Vec<SearchResult> = self
            .vectors
            .iter()
            .map(|(&id, vec)| SearchResult {
                id,
                distance: metric.distance(query, vec),
            })
            .collect();

        // Partial sort: only need the top-k smallest distances.
        let k = k.min(distances.len());
        distances.select_nth_unstable_by(k - 1, |a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        });
        distances.truncate(k);
        distances.sort_unstable_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(distances)
    }

    fn delete(&mut self, id: u64) -> bool {
        self.vectors.remove(&id).is_some()
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn config(&self) -> &IndexConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_index() -> FlatIndex {
        let mut idx = FlatIndex::new(3, Metric::L2);
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0]).unwrap();
        idx.add(3, &[0.0, 0.0, 1.0]).unwrap();
        idx.add(4, &[1.0, 1.0, 0.0]).unwrap();
        idx
    }

    #[test]
    fn nearest_to_itself() {
        let idx = make_index();
        let results = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, 1);
        assert!(results[0].distance < 1e-5);
    }

    #[test]
    fn top_k_ordered() {
        let idx = make_index();
        let results = idx.search(&[1.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(results.len(), 3);
        // distances must be non-decreasing
        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }

    #[test]
    fn k_larger_than_dataset() {
        let idx = make_index();
        let results = idx.search(&[0.5, 0.5, 0.0], 100).unwrap();
        assert_eq!(results.len(), 4); // only 4 vectors in index
    }

    #[test]
    fn dimension_mismatch_add() {
        let mut idx = FlatIndex::new(3, Metric::L2);
        assert!(matches!(
            idx.add(1, &[1.0, 2.0]),
            Err(VectorDbError::DimensionMismatch { expected: 3, got: 2 })
        ));
    }

    #[test]
    fn dimension_mismatch_search() {
        let idx = make_index();
        assert!(idx.search(&[1.0, 0.0], 1).is_err());
    }

    #[test]
    fn duplicate_id_rejected() {
        let mut idx = make_index();
        assert!(matches!(
            idx.add(1, &[0.0, 0.0, 0.0]),
            Err(VectorDbError::DuplicateId(1))
        ));
    }

    #[test]
    fn delete_removes_vector() {
        let mut idx = make_index();
        assert!(idx.delete(1));
        assert_eq!(idx.len(), 3);
        assert!(!idx.delete(99)); // non-existent
    }

    #[test]
    fn cosine_metric() {
        let mut idx = FlatIndex::new(2, Metric::Cosine);
        idx.add(1, &[1.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0]).unwrap();
        idx.add(3, &[1.0, 1.0]).unwrap();
        let results = idx.search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, 1); // exact match has cosine distance ≈ 0
    }
}
