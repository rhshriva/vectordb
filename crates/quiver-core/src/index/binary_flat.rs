//! Binary (1-bit) quantized flat index.
//!
//! Each f32 dimension is reduced to a single bit: `1` if `value >= 0`, `0`
//! otherwise.  Bits are packed into `u64` words (64 dimensions per word).
//! Search uses Hamming distance via hardware `popcount`.
//!
//! This is the most aggressive quantization available — 32× memory reduction
//! compared to `FlatIndex`.  It is best suited for first-pass candidate
//! filtering in re-ranking workflows or when memory is the primary constraint.
//!
//! # Quantization scheme
//!
//! ```text
//! bit[i] = vector[i] >= 0.0 ? 1 : 0
//! ```
//!
//! The original L2 norm is preserved for optional distance correction.
//!
//! # Memory
//!
//! | Index                | 1 K × 1 536-dim | 1 M × 1 536-dim |
//! |----------------------|-----------------|-----------------|
//! | `FlatIndex` (f32)    | 6.0 MB          | 6.0 GB          |
//! | `BinaryFlatIndex`    | ~0.19 MB        | ~0.19 GB        |

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    distance::Metric,
    error::VectorDbError,
    index::{IndexConfig, SearchResult, VectorIndex},
};

// ── Binary vector ────────────────────────────────────────────────────────────

/// A single binary-quantized vector.
///
/// Each f32 dimension is reduced to 1 bit: `bit[i] = vector[i] >= 0.0 ? 1 : 0`.
/// Bits are packed into `u64` words (64 dimensions per word).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BinaryVec {
    /// Packed bits: `bits[i/64] & (1 << (i%64))` gives bit for dimension `i`.
    bits: Vec<u64>,
    /// L2 norm of the original vector, kept for approximate distance correction.
    norm: f32,
}

impl BinaryVec {
    fn encode(v: &[f32]) -> Self {
        let n_words = (v.len() + 63) / 64;
        let mut bits = vec![0u64; n_words];
        for (i, &x) in v.iter().enumerate() {
            if x >= 0.0 {
                bits[i / 64] |= 1u64 << (i % 64);
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        BinaryVec { bits, norm }
    }

    /// Hamming distance: number of differing bits.
    #[inline]
    fn hamming_distance(&self, other: &BinaryVec) -> u32 {
        self.bits
            .iter()
            .zip(other.bits.iter())
            .map(|(&a, &b)| (a ^ b).count_ones())
            .sum()
    }

    /// Approximate reconstruction: +scale / -scale per bit.
    ///
    /// The reconstructed vector has the same sign pattern and is scaled so that
    /// its L2 norm matches the original. Good enough for WAL compaction (the
    /// re-quantized binary representation is identical).
    fn decode_approx(&self, dimensions: usize) -> Vec<f32> {
        let scale = if dimensions > 0 {
            self.norm / (dimensions as f32).sqrt()
        } else {
            1.0
        };
        (0..dimensions)
            .map(|i| {
                let bit = (self.bits[i / 64] >> (i % 64)) & 1;
                if bit == 1 { scale } else { -scale }
            })
            .collect()
    }
}

// ── Index ────────────────────────────────────────────────────────────────────

/// Brute-force flat index with 1-bit (binary) quantization.
///
/// Achieves 32× memory reduction compared to [`FlatIndex`][crate::index::flat::FlatIndex].
/// Uses Hamming distance via hardware `popcount` — extremely fast for
/// first-pass candidate filtering in re-ranking workflows.
#[derive(Debug, Serialize, Deserialize)]
pub struct BinaryFlatIndex {
    config: IndexConfig,
    vectors: HashMap<u64, BinaryVec>,
}

impl BinaryFlatIndex {
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

impl BinaryFlatIndex {
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
            self.vectors.insert(id, BinaryVec::encode(slice));
        }
        Ok(())
    }
}

impl VectorIndex for BinaryFlatIndex {
    fn add(&mut self, id: u64, vector: &[f32]) -> Result<(), VectorDbError> {
        if vector.len() != self.config.dimensions {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }
        self.vectors.insert(id, BinaryVec::encode(vector));
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

        let query_bin = BinaryVec::encode(query);
        let dims = self.config.dimensions as f32;

        let mut results: Vec<SearchResult> = self
            .vectors
            .iter()
            .map(|(&id, bv)| {
                let hamming = query_bin.hamming_distance(bv);
                // Convert Hamming distance to a float in [0, 1].
                // For all metrics, lower = more similar.
                let distance = match self.config.metric {
                    Metric::L2 | Metric::Cosine => hamming as f32 / dims,
                    Metric::DotProduct => {
                        // Matching bits ≈ agreement in sign.  Convert to a
                        // negative score so lower = more similar.
                        let agreement = dims - 2.0 * hamming as f32;
                        -(agreement * query_bin.norm * bv.norm / dims)
                    }
                };
                SearchResult { id, distance }
            })
            .collect();

        results.sort_unstable_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
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

    /// Iterate over (id, approximate_vector) pairs.
    ///
    /// The returned vectors are coarse reconstructions from the binary encoding.
    /// Re-quantizing them produces the same binary representation.
    fn iter_vectors(&self) -> Box<dyn Iterator<Item = (u64, Vec<f32>)> + '_> {
        let dims = self.config.dimensions;
        Box::new(
            self.vectors
                .iter()
                .map(move |(&id, bv)| (id, bv.decode_approx(dims))),
        )
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_basic() {
        let v = vec![1.0, -1.0, 0.5, -0.3];
        let bv = BinaryVec::encode(&v);
        // bit pattern: 1, 0, 1, 0 = 0b0101 = 5
        assert_eq!(bv.bits[0] & 0xF, 0b0101);
    }

    #[test]
    fn hamming_distance_identical() {
        let v = vec![1.0, -1.0, 0.5, -0.3];
        let bv = BinaryVec::encode(&v);
        assert_eq!(bv.hamming_distance(&bv), 0);
    }

    #[test]
    fn hamming_distance_opposite() {
        let a = vec![1.0, 1.0, 1.0, 1.0];
        let b = vec![-1.0, -1.0, -1.0, -1.0];
        let ba = BinaryVec::encode(&a);
        let bb = BinaryVec::encode(&b);
        assert_eq!(ba.hamming_distance(&bb), 4);
    }

    #[test]
    fn add_and_search_nearest() {
        let mut idx = BinaryFlatIndex::new(4, Metric::L2);
        idx.add(1, &[1.0, 1.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[-1.0, -1.0, 0.0, 0.0]).unwrap();
        idx.add(3, &[-1.0, -1.0, -1.0, -1.0]).unwrap();

        let results = idx.search(&[1.0, 0.5, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn search_returns_at_most_k() {
        let mut idx = BinaryFlatIndex::new(4, Metric::Cosine);
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert!(results.len() <= 2);
    }

    #[test]
    fn delete_removes_vector() {
        let mut idx = BinaryFlatIndex::new(4, Metric::L2);
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(idx.len(), 1);
        assert!(idx.delete(1));
        assert_eq!(idx.len(), 0);
        assert!(!idx.delete(1));
    }

    #[test]
    fn empty_search_returns_empty() {
        let idx = BinaryFlatIndex::new(4, Metric::Cosine);
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("binary.bin");
        let mut idx = BinaryFlatIndex::new(4, Metric::Cosine);
        idx.add(10, &[0.5, 0.5, -0.5, -0.5]).unwrap();
        idx.add(20, &[-0.5, 0.5, 0.5, -0.5]).unwrap();
        idx.save(&path).unwrap();

        let loaded = BinaryFlatIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn dimension_mismatch_on_add_errors() {
        let mut idx = BinaryFlatIndex::new(4, Metric::L2);
        let err = idx.add(1, &[1.0, 2.0]).unwrap_err();
        assert!(matches!(
            err,
            VectorDbError::DimensionMismatch {
                expected: 4,
                got: 2
            }
        ));
    }

    #[test]
    fn dimension_mismatch_on_search_errors() {
        let mut idx = BinaryFlatIndex::new(4, Metric::L2);
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let err = idx.search(&[1.0, 0.0], 1).unwrap_err();
        assert!(matches!(
            err,
            VectorDbError::DimensionMismatch {
                expected: 4,
                got: 2
            }
        ));
    }

    #[test]
    fn upsert_overwrites_existing_id() {
        let mut idx = BinaryFlatIndex::new(4, Metric::L2);
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(1, &[-1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn iter_vectors_covers_all_ids() {
        let mut idx = BinaryFlatIndex::new(4, Metric::L2);
        idx.add(10, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(20, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        let mut ids: Vec<u64> = idx.iter_vectors().map(|(id, _)| id).collect();
        ids.sort();
        assert_eq!(ids, vec![10, 20]);
    }

    #[test]
    fn compression_ratio_32x() {
        let dims = 1536;
        let v: Vec<f32> = (0..dims)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.3 })
            .collect();
        let bv = BinaryVec::encode(&v);
        let binary_bytes = bv.bits.len() * 8;
        let f32_bytes = dims * 4;
        assert_eq!(binary_bytes, 192);
        assert_eq!(f32_bytes, 6144);
        assert_eq!(f32_bytes / binary_bytes, 32);
    }

    #[test]
    fn high_dimensional_search() {
        let dims = 1536;
        let mut idx = BinaryFlatIndex::new(dims, Metric::Cosine);
        for i in 0..100u64 {
            let v: Vec<f32> = (0..dims)
                .map(|d| if (d as u64 + i) % 3 == 0 { 1.0 } else { -1.0 })
                .collect();
            idx.add(i, &v).unwrap();
        }
        assert_eq!(idx.len(), 100);
        let q: Vec<f32> = (0..dims)
            .map(|d| if d % 3 == 0 { 1.0 } else { -1.0 })
            .collect();
        let results = idx.search(&q, 5).unwrap();
        assert_eq!(results.len(), 5);
        // id=0 has the exact same sign pattern as query (as do all multiples of 3)
        // so verify the top result has zero Hamming distance
        assert!(results[0].distance < 1e-6);
    }

    #[test]
    fn decode_approx_preserves_sign_pattern() {
        let v = vec![0.3, -0.7, 0.0, -0.1, 0.9, -0.2];
        let bv = BinaryVec::encode(&v);
        let decoded = bv.decode_approx(v.len());
        // Sign pattern: [+, -, +, -, +, -]  (0.0 maps to +)
        assert!(decoded[0] > 0.0);
        assert!(decoded[1] < 0.0);
        assert!(decoded[2] > 0.0);
        assert!(decoded[3] < 0.0);
        assert!(decoded[4] > 0.0);
        assert!(decoded[5] < 0.0);
    }
}
