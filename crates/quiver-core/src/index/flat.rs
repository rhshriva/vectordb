/// Flat (brute-force) exact index.
///
/// Stores all vectors in a contiguous `Vec<f32>` buffer and performs an
/// exhaustive linear scan on every query.  100% recall by definition.
/// O(N·D) per query.
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
    /// Contiguous vector data in row-major order.
    data: Vec<f32>,
    /// Caller-supplied IDs, one per slot.
    ids: Vec<u64>,
    /// Soft-delete markers, one per slot.
    alive: Vec<bool>,
}

pub struct FlatIndex {
    config: IndexConfig,
    /// Contiguous vector storage: all vectors packed end-to-end.
    data: Vec<f32>,
    /// Caller-supplied ID at each slot.
    ids: Vec<u64>,
    /// ID → slot index (for delete/duplicate checks).
    id_to_slot: HashMap<u64, u32>,
    /// Soft-delete markers, one per slot.
    alive: Vec<bool>,
    /// Number of alive vectors.
    count: usize,
}

impl FlatIndex {
    pub fn new(dimensions: usize, metric: Metric) -> Self {
        Self {
            config: IndexConfig { dimensions, metric },
            data: Vec::new(),
            ids: Vec::new(),
            id_to_slot: HashMap::new(),
            alive: Vec::new(),
            count: 0,
        }
    }

    /// Save the index to a binary file at `path` (bincode format).
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), VectorDbError> {
        let file = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(file);
        let snapshot = FlatIndexSnapshot {
            dimensions: self.config.dimensions,
            metric: self.config.metric,
            data: self.data.clone(),
            ids: self.ids.clone(),
            alive: self.alive.clone(),
        };
        bincode::serialize_into(&mut writer, &snapshot)
            .map_err(|e| VectorDbError::Serialization(e.to_string()))?;
        Ok(())
    }

    /// Load an index from a binary file previously written by [`save`].
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, VectorDbError> {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);
        let snapshot: FlatIndexSnapshot = bincode::deserialize_from(reader)
            .map_err(|e| VectorDbError::Serialization(e.to_string()))?;

        let mut id_to_slot = HashMap::with_capacity(snapshot.ids.len());
        let mut count = 0;
        for (slot, (&id, &is_alive)) in snapshot.ids.iter().zip(snapshot.alive.iter()).enumerate() {
            if is_alive {
                id_to_slot.insert(id, slot as u32);
                count += 1;
            }
        }

        Ok(Self {
            config: IndexConfig {
                dimensions: snapshot.dimensions,
                metric: snapshot.metric,
            },
            data: snapshot.data,
            ids: snapshot.ids,
            id_to_slot,
            alive: snapshot.alive,
            count,
        })
    }

    /// Pre-allocate capacity for `n` vectors.
    pub fn with_capacity(dimensions: usize, metric: Metric, n: usize) -> Self {
        Self {
            config: IndexConfig { dimensions, metric },
            data: Vec::with_capacity(n * dimensions),
            ids: Vec::with_capacity(n),
            id_to_slot: HashMap::with_capacity(n),
            alive: Vec::with_capacity(n),
            count: 0,
        }
    }

    /// Bulk insert from a contiguous f32 buffer (row-major, N × dim).
    /// IDs are assigned sequentially: start_id, start_id+1, ...
    /// No per-vector allocation or Python interaction.
    pub fn add_batch_raw(
        &mut self,
        raw_data: &[f32],
        dim: usize,
        start_id: u64,
    ) -> Result<(), VectorDbError> {
        if dim != self.config.dimensions {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.config.dimensions,
                got: dim,
            });
        }
        if raw_data.len() % dim != 0 {
            return Err(VectorDbError::InvalidConfig(format!(
                "data length {} is not divisible by dimension {}",
                raw_data.len(),
                dim
            )));
        }
        let n = raw_data.len() / dim;

        // Pre-allocate in one shot
        self.data.reserve(n * dim);
        self.ids.reserve(n);
        self.alive.reserve(n);
        self.id_to_slot.reserve(n);

        // Bulk append — single memcpy for vector data
        let base_slot = self.ids.len() as u32;
        self.data.extend_from_slice(raw_data);

        for i in 0..n {
            let id = start_id + i as u64;
            let slot = base_slot + i as u32;
            self.ids.push(id);
            self.alive.push(true);
            self.id_to_slot.insert(id, slot);
            self.count += 1;
        }

        Ok(())
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
        if self.id_to_slot.contains_key(&id) {
            return Err(VectorDbError::DuplicateId(id));
        }
        let slot = self.ids.len() as u32;
        self.data.extend_from_slice(vector);
        self.ids.push(id);
        self.alive.push(true);
        self.id_to_slot.insert(id, slot);
        self.count += 1;
        Ok(())
    }

    fn add_batch(&mut self, entries: &[(u64, Vec<f32>)]) -> Result<(), VectorDbError> {
        let dim = self.config.dimensions;

        // Validate all entries first
        for (id, vec) in entries {
            if vec.len() != dim {
                return Err(VectorDbError::DimensionMismatch {
                    expected: dim,
                    got: vec.len(),
                });
            }
            if self.id_to_slot.contains_key(id) {
                return Err(VectorDbError::DuplicateId(*id));
            }
        }

        // Pre-allocate
        let n = entries.len();
        self.data.reserve(n * dim);
        self.ids.reserve(n);
        self.alive.reserve(n);
        self.id_to_slot.reserve(n);

        // Bulk insert
        for (id, vec) in entries {
            let slot = self.ids.len() as u32;
            self.data.extend_from_slice(vec);
            self.ids.push(*id);
            self.alive.push(true);
            self.id_to_slot.insert(*id, slot);
            self.count += 1;
        }

        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, VectorDbError> {
        if query.len() != self.config.dimensions {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len(),
            });
        }
        if k == 0 || self.count == 0 {
            return Ok(vec![]);
        }

        let dim = self.config.dimensions;
        let metric = self.config.metric;

        // Iterate contiguous buffer in cache-friendly order
        let mut distances: Vec<SearchResult> = Vec::with_capacity(self.count);
        for (slot, &is_alive) in self.alive.iter().enumerate() {
            if !is_alive {
                continue;
            }
            let start = slot * dim;
            let vec = &self.data[start..start + dim];
            distances.push(SearchResult {
                id: self.ids[slot],
                distance: metric.distance(query, vec),
            });
        }

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
        if let Some(&slot) = self.id_to_slot.get(&id) {
            if self.alive[slot as usize] {
                self.alive[slot as usize] = false;
                self.id_to_slot.remove(&id);
                self.count -= 1;
                return true;
            }
        }
        false
    }

    fn len(&self) -> usize {
        self.count
    }

    fn config(&self) -> &IndexConfig {
        &self.config
    }

    fn iter_vectors(&self) -> Box<dyn Iterator<Item = (u64, Vec<f32>)> + '_> {
        let dim = self.config.dimensions;
        Box::new(
            self.alive
                .iter()
                .enumerate()
                .filter(|(_, &alive)| alive)
                .map(move |(slot, _)| {
                    let start = slot * dim;
                    let vec = self.data[start..start + dim].to_vec();
                    (self.ids[slot], vec)
                }),
        )
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
        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }

    #[test]
    fn k_larger_than_dataset() {
        let idx = make_index();
        let results = idx.search(&[0.5, 0.5, 0.0], 100).unwrap();
        assert_eq!(results.len(), 4);
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
        assert!(!idx.delete(99));
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
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn dot_product_metric() {
        let mut idx = FlatIndex::new(2, Metric::DotProduct);
        idx.add(1, &[1.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0]).unwrap();
        idx.add(3, &[3.0, 0.0]).unwrap();
        let results = idx.search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, 3);
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
        let entries = vec![(2u64, vec![0.0_f32, 1.0]), (1, vec![0.5, 0.5])];
        assert!(idx.add_batch(&entries).is_err());
    }

    #[test]
    fn upsert_pattern_delete_then_readd() {
        let mut idx = make_index();
        assert!(idx.delete(1));
        idx.add(1, &[9.0, 0.0, 0.0]).unwrap();
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

    #[test]
    fn add_batch_raw_inserts_all() {
        let mut idx = FlatIndex::new(2, Metric::L2);
        let data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        idx.add_batch_raw(&data, 2, 0).unwrap();
        assert_eq!(idx.len(), 3);
        let results = idx.search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, 0);
    }

    #[test]
    fn add_batch_raw_with_start_id() {
        let mut idx = FlatIndex::new(2, Metric::L2);
        let data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        idx.add_batch_raw(&data, 2, 100).unwrap();
        assert_eq!(idx.len(), 2);
        let results = idx.search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, 100);
    }

    #[test]
    fn add_batch_raw_dimension_mismatch() {
        let mut idx = FlatIndex::new(3, Metric::L2);
        let data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        assert!(idx.add_batch_raw(&data, 2, 0).is_err());
    }

    #[test]
    fn delete_and_search_skips_dead() {
        let mut idx = FlatIndex::new(2, Metric::L2);
        idx.add(0, &[1.0, 0.0]).unwrap();
        idx.add(1, &[0.0, 1.0]).unwrap();
        idx.add(2, &[1.0, 1.0]).unwrap();
        assert!(idx.delete(0));
        assert_eq!(idx.len(), 2);
        let results = idx.search(&[1.0, 0.0], 3).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.id != 0));
    }

    #[test]
    fn iter_vectors_skips_deleted() {
        let mut idx = make_index();
        idx.delete(2);
        let vecs: Vec<(u64, Vec<f32>)> = idx.iter_vectors().collect();
        assert_eq!(vecs.len(), 3);
        assert!(vecs.iter().all(|(id, _)| *id != 2));
    }

    #[test]
    fn save_load_with_deletes() {
        let mut idx = make_index();
        idx.delete(2);
        let path = "/tmp/flat_index_del_test.bin";
        idx.save(path).unwrap();
        let loaded = FlatIndex::load(path).unwrap();
        assert_eq!(loaded.len(), 3);
        let results = loaded.search(&[0.0, 1.0, 0.0], 1).unwrap();
        assert_ne!(results[0].id, 2);
    }
}
