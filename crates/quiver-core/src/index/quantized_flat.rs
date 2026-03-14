//! Scalar-quantized flat index (int8).
//!
//! Stores vectors as `i8` with a per-vector scale factor, achieving ~4×
//! memory reduction compared to `f32` with negligible recall loss for typical
//! unit-norm embeddings.
//!
//! # Quantization scheme
//!
//! For each vector `v` we compute a symmetric per-vector scale:
//! ```text
//! scale  = 127.0 / max(|x| for x in v)
//! q[i]   = round(v[i] * scale).clamp(-127, 127) as i8
//! ```
//! Dequantization: `v̂[i] = q[i] as f32 / scale`
//!
//! The worst-case absolute error per element is `0.5 / scale`.
//! For a unit-norm 1 536-dim vector (max element ≈ 0.07) the error is < 3 × 10⁻⁴.
//!
//! # Memory
//!
//! | Index                | 1 K × 1 536-dim | 1 M × 1 536-dim |
//! |----------------------|-----------------|-----------------|
//! | `FlatIndex` (f32)    | 6.0 MB          | 6.0 GB          |
//! | `QuantizedFlatIndex` | ~1.5 MB         | ~1.5 GB         |

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    distance::Metric,
    error::VectorDbError,
    index::{IndexConfig, SearchResult, VectorIndex},
};

// ── Quantized vector ───────────────────────────────────────────────────────────

/// A single vector stored as quantized `i8` coefficients plus its scale factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuantizedVec {
    /// `q[i] = round(v[i] * scale)`, clamped to [-127, 127].
    data: Vec<i8>,
    /// Encode scale: `scale = 127.0 / max_abs_component`.
    /// Dequantize: `v̂[i] = q[i] as f32 / scale`.
    scale: f32,
}

impl QuantizedVec {
    fn encode(v: &[f32]) -> Self {
        let max_abs = v.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        // If the vector is all-zero use scale = 1 (any value works; result is all zeros).
        let scale = if max_abs > 0.0 { 127.0 / max_abs } else { 1.0 };
        let data = v
            .iter()
            .map(|&x| (x * scale).round().clamp(-127.0, 127.0) as i8)
            .collect();
        QuantizedVec { data, scale }
    }

    /// Decode into `out`, reusing its allocation.
    #[inline]
    fn decode_into(&self, out: &mut Vec<f32>) {
        out.clear();
        out.extend(self.data.iter().map(|&q| q as f32 / self.scale));
    }
}

// ── Index ──────────────────────────────────────────────────────────────────────

/// Brute-force flat index that stores vectors as `i8` (scalar quantization).
///
/// Performs an exhaustive linear scan on search, giving 100% theoretical recall
/// limited only by quantization error (< 1% recall loss in practice).
///
/// Use this index when memory is the primary constraint and exact search is
/// acceptable; for larger datasets prefer [`HnswIndex`][crate::index::hnsw::HnswIndex].
#[derive(Debug, Serialize, Deserialize)]
pub struct QuantizedFlatIndex {
    config: IndexConfig,
    vectors: HashMap<u64, QuantizedVec>,
}

impl QuantizedFlatIndex {
    /// Create an empty index with the given dimensionality and distance metric.
    pub fn new(dimensions: usize, metric: Metric) -> Self {
        Self {
            config: IndexConfig { dimensions, metric },
            vectors: HashMap::new(),
        }
    }

    /// Serialize this index to a binary file (bincode format).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), VectorDbError> {
        let bytes = bincode::serialize(self)
            .map_err(|e| VectorDbError::Serialization(e.to_string()))?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Deserialize a previously saved index.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, VectorDbError> {
        let bytes = std::fs::read(path)?;
        bincode::deserialize(&bytes)
            .map_err(|e| VectorDbError::Serialization(e.to_string()))
    }
}

impl QuantizedFlatIndex {
    /// Bulk-insert from a contiguous f32 buffer (row-major, `n_rows × dim`).
    /// Assigns sequential IDs starting from `start_id`.
    pub fn add_batch_raw(&mut self, raw_data: &[f32], dim: usize, start_id: u64) -> Result<(), VectorDbError> {
        if dim != self.config.dimensions {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.config.dimensions,
                got: dim,
            });
        }
        if raw_data.len() % dim != 0 {
            return Err(VectorDbError::InvalidConfig(
                format!("raw_data length {} is not a multiple of dim {}", raw_data.len(), dim),
            ));
        }
        let n = raw_data.len() / dim;
        self.vectors.reserve(n);
        for i in 0..n {
            let slice = &raw_data[i * dim..(i + 1) * dim];
            let id = start_id + i as u64;
            self.vectors.insert(id, QuantizedVec::encode(slice));
        }
        Ok(())
    }
}

impl VectorIndex for QuantizedFlatIndex {
    fn add(&mut self, id: u64, vector: &[f32]) -> Result<(), VectorDbError> {
        if vector.len() != self.config.dimensions {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }
        // insert overwrites if the id already exists (upsert semantics)
        self.vectors.insert(id, QuantizedVec::encode(vector));
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, VectorDbError> {
        if self.vectors.is_empty() {
            return Ok(vec![]);
        }
        if query.len() != self.config.dimensions {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len(),
            });
        }

        // Reuse a single buffer across all dequantizations to avoid per-vector allocation.
        let mut buf = Vec::with_capacity(self.config.dimensions);

        let mut results: Vec<SearchResult> = self
            .vectors
            .iter()
            .map(|(&id, qv)| {
                qv.decode_into(&mut buf);
                let distance = self.config.metric.distance(query, &buf);
                SearchResult { id, distance }
            })
            .collect();

        results.sort_unstable_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        Ok(results)
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

    /// Iterate over (id, dequantized_vector) pairs.
    ///
    /// The returned vectors are approximate reconstructions (quantization
    /// error < 1%) — suitable for WAL compaction and index migration.
    fn iter_vectors(&self) -> Box<dyn Iterator<Item = (u64, Vec<f32>)> + '_> {
        Box::new(self.vectors.iter().map(|(&id, qv)| {
            let decoded: Vec<f32> = qv.data.iter().map(|&q| q as f32 / qv.scale).collect();
            (id, decoded)
        }))
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(n: usize, hot: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        v[hot] = 1.0;
        v
    }

    #[test]
    fn encode_decode_roundtrip() {
        let v: Vec<f32> = (1..=8).map(|i| i as f32 * 0.1).collect();
        let qv = QuantizedVec::encode(&v);
        let mut out = Vec::new();
        qv.decode_into(&mut out);
        let max_abs = v.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let tolerance = 0.5 * max_abs / 127.0 + 1e-6;
        for (orig, decoded) in v.iter().zip(out.iter()) {
            assert!(
                (orig - decoded).abs() <= tolerance,
                "orig={orig:.4} decoded={decoded:.4} tol={tolerance:.6}"
            );
        }
    }

    #[test]
    fn zero_vector_does_not_panic() {
        let v = vec![0.0_f32; 4];
        let qv = QuantizedVec::encode(&v);
        let mut out = Vec::new();
        qv.decode_into(&mut out);
        assert!(out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn add_and_search_nearest() {
        let mut idx = QuantizedFlatIndex::new(3, Metric::L2);
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0]).unwrap();
        idx.add(3, &[0.0, 0.0, 1.0]).unwrap();

        let results = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, 1);
        // Quantization error for unit vectors should be < 0.05
        assert!(results[0].distance < 0.05, "dist={}", results[0].distance);
    }

    #[test]
    fn search_returns_at_most_k() {
        let mut idx = QuantizedFlatIndex::new(2, Metric::Cosine);
        idx.add(1, &[1.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0]).unwrap();
        let results = idx.search(&[1.0, 0.0], 5).unwrap();
        assert!(results.len() <= 2);
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn delete_removes_vector() {
        let mut idx = QuantizedFlatIndex::new(2, Metric::L2);
        idx.add(1, &[1.0, 0.0]).unwrap();
        assert_eq!(idx.len(), 1);
        assert!(idx.delete(1));
        assert_eq!(idx.len(), 0);
        assert!(!idx.delete(1)); // second delete is a no-op
    }

    #[test]
    fn empty_search_returns_empty() {
        let idx = QuantizedFlatIndex::new(3, Metric::Cosine);
        let results = idx.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("qidx.bin");

        let mut idx = QuantizedFlatIndex::new(3, Metric::Cosine);
        idx.add(10, &[0.5, 0.5, 0.0]).unwrap();
        idx.add(20, &[0.0, 0.5, 0.5]).unwrap();
        idx.save(&path).unwrap();

        let loaded = QuantizedFlatIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        let results = loaded.search(&[0.5, 0.5, 0.0], 1).unwrap();
        assert_eq!(results[0].id, 10);
    }

    #[test]
    fn upsert_overwrites_existing_id() {
        let mut idx = QuantizedFlatIndex::new(2, Metric::L2);
        idx.add(1, &[1.0, 0.0]).unwrap();
        idx.add(1, &[0.0, 1.0]).unwrap(); // should silently overwrite
        assert_eq!(idx.len(), 1);
        let results = idx.search(&[0.0, 1.0], 1).unwrap();
        assert_eq!(results[0].id, 1);
        assert!(results[0].distance < 0.05);
    }

    #[test]
    fn dimension_mismatch_on_add_errors() {
        let mut idx = QuantizedFlatIndex::new(3, Metric::L2);
        let err = idx.add(1, &[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, VectorDbError::DimensionMismatch { expected: 3, got: 2 }));
    }

    #[test]
    fn dimension_mismatch_on_search_errors() {
        let mut idx = QuantizedFlatIndex::new(3, Metric::L2);
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        let err = idx.search(&[1.0, 0.0], 1).unwrap_err();
        assert!(matches!(err, VectorDbError::DimensionMismatch { expected: 3, got: 2 }));
    }

    #[test]
    fn bulk_insert_and_search() {
        let dims = 128;
        let mut idx = QuantizedFlatIndex::new(dims, Metric::Cosine);
        for i in 0..500u64 {
            let v: Vec<f32> = unit_vec(dims, (i as usize) % dims);
            idx.add(i, &v).unwrap();
        }
        assert_eq!(idx.len(), 500);
        let q = unit_vec(dims, 0);
        let results = idx.search(&q, 5).unwrap();
        assert_eq!(results.len(), 5);
        // All top-5 for unit vector [1,0,...] should have id % 128 == 0 (same direction)
        assert!(results[0].distance < 0.05, "dist={}", results[0].distance);
    }

    #[test]
    fn iter_vectors_covers_all_ids() {
        let mut idx = QuantizedFlatIndex::new(2, Metric::L2);
        idx.add(10, &[1.0, 0.0]).unwrap();
        idx.add(20, &[0.0, 1.0]).unwrap();
        let mut ids: Vec<u64> = idx.iter_vectors().map(|(id, _)| id).collect();
        ids.sort();
        assert_eq!(ids, vec![10, 20]);
    }
}
