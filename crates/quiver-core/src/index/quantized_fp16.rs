//! Half-precision (FP16) quantized flat index.
//!
//! Stores vectors as IEEE 754 half-precision floats (`f16`), achieving ~2×
//! memory reduction compared to `f32` with minimal recall loss.
//!
//! # Quantization scheme
//!
//! Each `f32` component is converted to `f16` via `half::f16::from_f32`.
//! The worst-case relative error for typical embedding values (range [-1, 1])
//! is approximately 0.05% — much smaller than int8 quantization.
//!
//! # Memory
//!
//! | Index                | 1 K × 1 536-dim | 1 M × 1 536-dim |
//! |----------------------|-----------------|-----------------|
//! | `FlatIndex` (f32)    | 6.0 MB          | 6.0 GB          |
//! | `Fp16FlatIndex`      | ~3.0 MB         | ~3.0 GB         |
//! | `QuantizedFlatIndex` | ~1.5 MB         | ~1.5 GB         |
//!
//! FP16 offers a middle ground: better precision than int8 at 2× compression
//! instead of 4×.

use std::collections::HashMap;
use std::path::Path;

use half::f16;
use serde::{Deserialize, Serialize};

use crate::{
    distance::Metric,
    error::VectorDbError,
    index::{IndexConfig, SearchResult, VectorIndex},
};

// ── FP16 vector ──────────────────────────────────────────────────────────────

/// A single vector stored as `f16` (half-precision) values.
///
/// Internally stored as `Vec<u16>` (the raw bit representation of `f16`)
/// for efficient serde serialization with bincode.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Fp16Vec {
    /// Raw `f16` bits stored as `u16` for bincode compatibility.
    data: Vec<u16>,
}

impl Fp16Vec {
    /// Encode an `f32` vector into `f16` representation.
    fn encode(v: &[f32]) -> Self {
        let data = v.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
        Fp16Vec { data }
    }

    /// Decode into `out`, reusing its allocation.
    #[inline]
    fn decode_into(&self, out: &mut Vec<f32>) {
        out.clear();
        out.extend(self.data.iter().map(|&bits| f16::from_bits(bits).to_f32()));
    }
}

// ── Index ────────────────────────────────────────────────────────────────────

/// Brute-force flat index that stores vectors as `f16` (half-precision).
///
/// Performs an exhaustive linear scan on search, giving near-100% recall
/// with only ~0.05% precision loss from FP16 rounding.
///
/// Use this index when memory is a constraint but you need higher precision
/// than [`QuantizedFlatIndex`][crate::index::quantized_flat::QuantizedFlatIndex] (int8).
#[derive(Debug, Serialize, Deserialize)]
pub struct Fp16FlatIndex {
    config: IndexConfig,
    vectors: HashMap<u64, Fp16Vec>,
}

impl Fp16FlatIndex {
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

impl Fp16FlatIndex {
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
            self.vectors.insert(id, Fp16Vec::encode(slice));
        }
        Ok(())
    }
}

impl VectorIndex for Fp16FlatIndex {
    fn add(&mut self, id: u64, vector: &[f32]) -> Result<(), VectorDbError> {
        if vector.len() != self.config.dimensions {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }
        // insert overwrites if the id already exists (upsert semantics)
        self.vectors.insert(id, Fp16Vec::encode(vector));
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
            .map(|(&id, fp16v)| {
                fp16v.decode_into(&mut buf);
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
    /// The returned vectors are approximate reconstructions — suitable for
    /// WAL compaction and index migration. FP16 precision loss is < 0.1%.
    fn iter_vectors(&self) -> Box<dyn Iterator<Item = (u64, Vec<f32>)> + '_> {
        Box::new(self.vectors.iter().map(|(&id, fp16v)| {
            let decoded: Vec<f32> = fp16v.data.iter().map(|&bits| f16::from_bits(bits).to_f32()).collect();
            (id, decoded)
        }))
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        let v: Vec<f32> = (1..=8).map(|i| i as f32 * 0.1).collect();
        let fp16v = Fp16Vec::encode(&v);
        let mut out = Vec::new();
        fp16v.decode_into(&mut out);
        // FP16 has ~3 decimal digits of precision for values in [0, 1]
        for (orig, decoded) in v.iter().zip(out.iter()) {
            let err = (orig - decoded).abs();
            assert!(
                err < 0.001,
                "orig={orig:.4} decoded={decoded:.4} err={err:.6}"
            );
        }
    }

    #[test]
    fn zero_vector_does_not_panic() {
        let v = vec![0.0_f32; 4];
        let fp16v = Fp16Vec::encode(&v);
        let mut out = Vec::new();
        fp16v.decode_into(&mut out);
        assert!(out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn add_and_search_nearest() {
        let mut idx = Fp16FlatIndex::new(3, Metric::L2);
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0]).unwrap();
        idx.add(3, &[0.0, 0.0, 1.0]).unwrap();

        let results = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, 1);
        // FP16 error for unit vectors should be negligible
        assert!(results[0].distance < 0.001, "dist={}", results[0].distance);
    }

    #[test]
    fn search_returns_at_most_k() {
        let mut idx = Fp16FlatIndex::new(2, Metric::Cosine);
        idx.add(1, &[1.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0]).unwrap();
        let results = idx.search(&[1.0, 0.0], 5).unwrap();
        assert!(results.len() <= 2);
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn delete_removes_vector() {
        let mut idx = Fp16FlatIndex::new(2, Metric::L2);
        idx.add(1, &[1.0, 0.0]).unwrap();
        assert_eq!(idx.len(), 1);
        assert!(idx.delete(1));
        assert_eq!(idx.len(), 0);
        assert!(!idx.delete(1)); // second delete is a no-op
    }

    #[test]
    fn empty_search_returns_empty() {
        let idx = Fp16FlatIndex::new(3, Metric::Cosine);
        let results = idx.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fp16idx.bin");

        let mut idx = Fp16FlatIndex::new(3, Metric::Cosine);
        idx.add(10, &[0.5, 0.5, 0.0]).unwrap();
        idx.add(20, &[0.0, 0.5, 0.5]).unwrap();
        idx.save(&path).unwrap();

        let loaded = Fp16FlatIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        let results = loaded.search(&[0.5, 0.5, 0.0], 1).unwrap();
        assert_eq!(results[0].id, 10);
    }

    #[test]
    fn upsert_overwrites_existing_id() {
        let mut idx = Fp16FlatIndex::new(2, Metric::L2);
        idx.add(1, &[1.0, 0.0]).unwrap();
        idx.add(1, &[0.0, 1.0]).unwrap(); // should silently overwrite
        assert_eq!(idx.len(), 1);
        let results = idx.search(&[0.0, 1.0], 1).unwrap();
        assert_eq!(results[0].id, 1);
        assert!(results[0].distance < 0.001);
    }

    #[test]
    fn dimension_mismatch_on_add_errors() {
        let mut idx = Fp16FlatIndex::new(3, Metric::L2);
        let err = idx.add(1, &[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, VectorDbError::DimensionMismatch { expected: 3, got: 2 }));
    }

    #[test]
    fn dimension_mismatch_on_search_errors() {
        let mut idx = Fp16FlatIndex::new(3, Metric::L2);
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        let err = idx.search(&[1.0, 0.0], 1).unwrap_err();
        assert!(matches!(err, VectorDbError::DimensionMismatch { expected: 3, got: 2 }));
    }

    #[test]
    fn iter_vectors_covers_all_ids() {
        let mut idx = Fp16FlatIndex::new(2, Metric::L2);
        idx.add(10, &[1.0, 0.0]).unwrap();
        idx.add(20, &[0.0, 1.0]).unwrap();
        let mut ids: Vec<u64> = idx.iter_vectors().map(|(id, _)| id).collect();
        ids.sort();
        assert_eq!(ids, vec![10, 20]);
    }

    #[test]
    fn fp16_precision_better_than_int8() {
        // FP16 should have much lower error than int8 for typical embedding values
        let v: Vec<f32> = (0..128).map(|i| (i as f32 / 128.0) - 0.5).collect();
        let fp16v = Fp16Vec::encode(&v);
        let mut decoded = Vec::new();
        fp16v.decode_into(&mut decoded);

        let max_err: f32 = v.iter().zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // FP16 max error should be < 0.001 for values in [-0.5, 0.5]
        assert!(max_err < 0.001, "max FP16 error = {max_err}");
    }

    #[test]
    fn bulk_insert_and_search() {
        let dims = 128;
        let mut idx = Fp16FlatIndex::new(dims, Metric::Cosine);
        for i in 0..500u64 {
            let mut v = vec![0.0f32; dims];
            v[(i as usize) % dims] = 1.0;
            idx.add(i, &v).unwrap();
        }
        assert_eq!(idx.len(), 500);
        let mut q = vec![0.0f32; dims];
        q[0] = 1.0;
        let results = idx.search(&q, 5).unwrap();
        assert_eq!(results.len(), 5);
        assert!(results[0].distance < 0.001, "dist={}", results[0].distance);
    }

    // ── add_batch_raw tests ─────────────────────────────────────────────

    #[test]
    fn add_batch_raw_basic() {
        let mut idx = Fp16FlatIndex::new(3, Metric::L2);
        let data = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        idx.add_batch_raw(&data, 3, 0).unwrap();
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn add_batch_raw_search_after() {
        let mut idx = Fp16FlatIndex::new(3, Metric::L2);
        let data = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ];
        idx.add_batch_raw(&data, 3, 10).unwrap();
        let results = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, 10);
        assert!(results[0].distance < 0.001);
    }

    #[test]
    fn add_batch_raw_empty_buffer() {
        let mut idx = Fp16FlatIndex::new(3, Metric::L2);
        idx.add_batch_raw(&[], 3, 0).unwrap();
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn add_batch_raw_dim_mismatch() {
        let mut idx = Fp16FlatIndex::new(3, Metric::L2);
        let err = idx.add_batch_raw(&[1.0; 4], 2, 0).unwrap_err();
        assert!(matches!(err, VectorDbError::DimensionMismatch { expected: 3, got: 2 }));
    }

    #[test]
    fn add_batch_raw_not_multiple_of_dim() {
        let mut idx = Fp16FlatIndex::new(3, Metric::L2);
        let err = idx.add_batch_raw(&[1.0; 5], 3, 0).unwrap_err();
        assert!(matches!(err, VectorDbError::InvalidConfig(_)));
    }

    // ── DotProduct metric ───────────────────────────────────────────────

    #[test]
    fn dot_product_metric_ordering() {
        let mut idx = Fp16FlatIndex::new(3, Metric::DotProduct);
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0]).unwrap();
        idx.add(3, &[0.5, 0.5, 0.0]).unwrap();

        let results = idx.search(&[1.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(results[0].id, 1);
    }

    // ── k > total vectors ───────────────────────────────────────────────

    #[test]
    fn k_greater_than_total_vectors() {
        let mut idx = Fp16FlatIndex::new(3, Metric::L2);
        for i in 0..5u64 {
            let mut v = vec![0.0f32; 3];
            v[(i as usize) % 3] = 1.0;
            idx.add(i, &v).unwrap();
        }
        let results = idx.search(&[1.0, 0.0, 0.0], 100).unwrap();
        assert_eq!(results.len(), 5);
    }

    // ── Mixed add() and add_batch_raw() ─────────────────────────────────

    #[test]
    fn mixed_add_and_batch() {
        let mut idx = Fp16FlatIndex::new(3, Metric::L2);
        idx.add(0, &[1.0, 0.0, 0.0]).unwrap();
        idx.add(1, &[0.0, 1.0, 0.0]).unwrap();
        let batch = vec![
            0.0, 0.0, 1.0,  // id 100
            0.5, 0.5, 0.0,  // id 101
        ];
        idx.add_batch_raw(&batch, 3, 100).unwrap();
        assert_eq!(idx.len(), 4);

        let results = idx.search(&[1.0, 0.0, 0.0], 4).unwrap();
        assert_eq!(results.len(), 4);
        assert_eq!(results[0].id, 0);
    }

    // ── Delete after batch insert ───────────────────────────────────────

    #[test]
    fn delete_after_batch_insert() {
        let mut idx = Fp16FlatIndex::new(3, Metric::L2);
        let data = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        idx.add_batch_raw(&data, 3, 0).unwrap();
        assert_eq!(idx.len(), 3);

        assert!(idx.delete(1));
        assert_eq!(idx.len(), 2);

        let results = idx.search(&[0.0, 1.0, 0.0], 3).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.id != 1));
    }

    // ── Negative value precision ────────────────────────────────────────

    #[test]
    fn negative_value_precision() {
        let v: Vec<f32> = vec![-0.1, -0.25, -0.5, -0.75, -1.0];
        let fp16v = Fp16Vec::encode(&v);
        let mut out = Vec::new();
        fp16v.decode_into(&mut out);
        for (orig, decoded) in v.iter().zip(out.iter()) {
            let err = (orig - decoded).abs();
            assert!(err < 0.001, "orig={orig} decoded={decoded} err={err}");
        }
    }

    #[test]
    fn negative_value_search_ordering() {
        let mut idx = Fp16FlatIndex::new(3, Metric::L2);
        idx.add(1, &[-1.0, -0.5, -0.2]).unwrap();
        idx.add(2, &[1.0, 0.5, 0.2]).unwrap();
        let results = idx.search(&[-1.0, -0.5, -0.2], 2).unwrap();
        assert_eq!(results[0].id, 1);
        assert!(results[0].distance < 0.001);
    }

    // ── Sequential batch calls ──────────────────────────────────────────

    #[test]
    fn sequential_batch_calls() {
        let mut idx = Fp16FlatIndex::new(3, Metric::L2);
        let batch1 = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ];
        idx.add_batch_raw(&batch1, 3, 0).unwrap();
        assert_eq!(idx.len(), 2);

        let batch2 = vec![
            0.0, 0.0, 1.0,
            0.5, 0.5, 0.0,
        ];
        idx.add_batch_raw(&batch2, 3, 10).unwrap();
        assert_eq!(idx.len(), 4);

        let batch3 = vec![
            0.3, 0.3, 0.3,
        ];
        idx.add_batch_raw(&batch3, 3, 20).unwrap();
        assert_eq!(idx.len(), 5);

        // All vectors from all batches should be searchable
        let results = idx.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].id, 0); // exact match from batch1
    }

    #[test]
    fn sequential_batches_with_overlapping_ids() {
        let mut idx = Fp16FlatIndex::new(3, Metric::L2);
        let batch1 = vec![1.0, 0.0, 0.0];
        idx.add_batch_raw(&batch1, 3, 0).unwrap();

        // Second batch starts at id=0 again — should overwrite
        let batch2 = vec![0.0, 1.0, 0.0];
        idx.add_batch_raw(&batch2, 3, 0).unwrap();
        assert_eq!(idx.len(), 1);

        let results = idx.search(&[0.0, 1.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, 0);
        assert!(results[0].distance < 0.001);
    }
}
