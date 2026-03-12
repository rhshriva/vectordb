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
use std::io::{BufReader, BufWriter};

use crate::{
    distance::Metric,
    error::VectorDbError,
    index::{IndexConfig, SearchResult, VectorIndex},
};

#[derive(serde::Serialize, serde::Deserialize)]
struct FlatIndexSnapshot {
    dimensions: usize,
    metric: Metric,
    vectors: HashMap<u64, Vec<f32>>,
}

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

    /// Save the index to a binary file at `path` (bincode format).
    pub fn save(&self, path: &str) -> Result<(), VectorDbError> {
        let file = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(file);
        let snapshot = FlatIndexSnapshot {
            dimensions: self.config.dimensions,
            metric: self.config.metric,
            vectors: self.vectors.clone(),
        };
        bincode::serialize_into(&mut writer, &snapshot)
            .map_err(|e| VectorDbError::Serialization(e.to_string()))?;
        Ok(())
    }

    /// Load an index from a binary file previously written by [`save`].
    pub fn load(path: &str) -> Result<Self, VectorDbError> {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);
        let snapshot: FlatIndexSnapshot = bincode::deserialize_from(reader)
            .map_err(|e| VectorDbError::Serialization(e.to_string()))?;
        Ok(Self {
            config: IndexConfig {
                dimensions: snapshot.dimensions,
                metric: snapshot.metric,
            },
            vectors: snapshot.vectors,
        })
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
        if k == 0 {
            return Ok(vec![]);
        }
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

    fn iter_vectors(&self) -> Box<dyn Iterator<Item = (u64, Vec<f32>)> + '_> {
        Box::new(self.vectors.iter().map(|(&id, v)| (id, v.clone())))
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
    fn save_and_load_round_trip() {
        let idx = make_index();
        let path = "/tmp/flat_index_test.json";
        idx.save(path).unwrap();
        let loaded = FlatIndex::load(path).unwrap();
        assert_eq!(loaded.len(), idx.len());
        assert_eq!(loaded.config().dimensions, idx.config().dimensions);
        assert_eq!(loaded.config().metric, idx.config().metric);
        // Search results must match
        let orig = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        let from_disk = loaded.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(orig[0].id, from_disk[0].id);
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

    #[test]
    fn dot_product_metric() {
        let mut idx = FlatIndex::new(2, Metric::DotProduct);
        idx.add(1, &[1.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0]).unwrap();
        idx.add(3, &[3.0, 0.0]).unwrap(); // highest dot with [1,0]
        let results = idx.search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, 3); // -dot(3,0) = -3, lowest = most similar
    }

    #[test]
    fn empty_index_search_returns_empty() {
        let idx = FlatIndex::new(3, Metric::L2);
        let results = idx.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn k_zero_returns_empty() {
        let idx = make_index();
        let results = idx.search(&[1.0, 0.0, 0.0], 0).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn single_vector_is_its_own_nearest() {
        let mut idx = FlatIndex::new(2, Metric::L2);
        idx.add(42, &[0.5, 0.5]).unwrap();
        let results = idx.search(&[0.5, 0.5], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 42);
    }

    #[test]
    fn add_batch_inserts_all() {
        let mut idx = FlatIndex::new(2, Metric::L2);
        let entries = vec![(1u64, vec![1.0_f32, 0.0]), (2, vec![0.0, 1.0]), (3, vec![1.0, 1.0])];
        idx.add_batch(&entries).unwrap();
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn add_batch_stops_on_duplicate() {
        let mut idx = FlatIndex::new(2, Metric::L2);
        idx.add(1, &[1.0, 0.0]).unwrap();
        let entries = vec![(2u64, vec![0.0_f32, 1.0]), (1, vec![0.5, 0.5])]; // ID 1 is duplicate
        assert!(idx.add_batch(&entries).is_err());
    }

    #[test]
    fn upsert_pattern_delete_then_readd() {
        let mut idx = make_index();
        assert!(idx.delete(1));
        idx.add(1, &[9.0, 0.0, 0.0]).unwrap(); // re-add with new vector
        let results = idx.search(&[9.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn save_to_bad_path_returns_io_error() {
        let idx = make_index();
        let result = idx.save("/nonexistent/directory/index.json");
        assert!(matches!(result, Err(VectorDbError::Io(_))));
    }

    #[test]
    fn load_nonexistent_file_returns_io_error() {
        let result = FlatIndex::load("/nonexistent/no_such_file.json");
        assert!(matches!(result, Err(VectorDbError::Io(_))));
    }

    #[test]
    fn load_malformed_data_returns_serialization_error() {
        let path = "/tmp/flat_bad.bin";
        std::fs::write(path, b"not valid bincode data").unwrap();
        let result = FlatIndex::load(path);
        assert!(matches!(result, Err(VectorDbError::Serialization(_))));
    }
}
