//! FAISS-backed vector index.
//!
//! Enabled via the `faiss` Cargo feature.  Requires `libfaiss_c` to be
//! installed, or compile with `features = ["static"]` to build FAISS from
//! source (needs a C++ toolchain and BLAS).
//!
//! # Factory strings
//!
//! The index is created through FAISS's factory notation.  Common examples:
//!
//! | String | Index type | Notes |
//! |--------|-----------|-------|
//! | `"Flat"` | Brute-force exact | SIMD-accelerated, 100 % recall |
//! | `"IVF1024,Flat"` | IVF coarse quantiser | Needs ≥ 39 × nlist training vectors |
//! | `"IVF256,PQ64"` | IVF + product quantisation | Compressed, GPU-friendly |
//! | `"HNSW32"` | Hierarchical NSW graph | Fast ANN, no training needed |
//! | `"IVF4096,SQ8"` | IVF + scalar quantiser | Good memory/recall trade-off |
//!
//! See the [FAISS wiki](https://github.com/facebookresearch/faiss/wiki/The-index-factory)
//! for the full grammar.

use std::collections::HashMap;
use std::sync::Mutex;

use faiss::{index_factory, Index, MetricType};
use tracing::warn;

use crate::{distance::Metric, error::VectorDbError};

use super::{IndexConfig, SearchResult, VectorIndex};

// ── Internal state held inside the Mutex ─────────────────────────────────────

struct FaissState {
    /// The live FAISS index; `None` when a rebuild is required.
    index: Option<faiss::IndexImpl>,
    /// Maps FAISS internal 0-based offset → caller-supplied u64 ID.
    ///
    /// Position i in this Vec corresponds to the i-th vector stored in the
    /// FAISS index (in insertion order at the time of the last rebuild).
    id_map: Vec<u64>,
}

// Safety: `faiss::IndexImpl` wraps a C++ pointer (`faiss_Index_t *`).
// Raw pointers are `!Send` by default in Rust.  FAISS indexes do not use
// thread-local state; we guarantee exclusive access via the surrounding
// `Mutex`, so moving the value to another thread is safe.
unsafe impl Send for FaissState {}

// ── Public struct ─────────────────────────────────────────────────────────────

/// A vector index backed by FAISS.
///
/// Vectors are kept in a plain `HashMap` (the source of truth for WAL
/// compaction and rebuilds).  A FAISS index is built lazily from that map
/// on the first `search()` call after any `add()` or `delete()`, or eagerly
/// by calling `flush()`.
pub struct FaissIndex {
    config: IndexConfig,
    /// FAISS factory string, e.g. `"Flat"` or `"IVF1024,Flat"`.
    factory_string: String,
    /// All live vectors keyed by caller-supplied ID.
    vectors: HashMap<u64, Vec<f32>>,
    /// FAISS search state protected by a mutex to allow lazy rebuild from
    /// within `search(&self, …)`.
    state: Mutex<FaissState>,
}

// `FaissIndex` is `Sync` because all interior mutation goes through `Mutex`.
// `FaissIndex` is `Send` because `Mutex<FaissState>` is `Send` (see the
// `unsafe impl Send for FaissState` above).
unsafe impl Sync for FaissIndex {}

// ── Constructor ───────────────────────────────────────────────────────────────

impl FaissIndex {
    /// Create a new, empty FAISS index.
    ///
    /// Returns an error if `factory_string` is not a valid FAISS factory
    /// description for the requested number of dimensions.
    pub fn new(
        dimensions: usize,
        metric: Metric,
        factory_string: impl Into<String>,
    ) -> Result<Self, VectorDbError> {
        let factory_string = factory_string.into();
        // Validate the factory string early by building a throw-away index.
        index_factory(dimensions as u32, &factory_string, faiss_metric(metric))
            .map_err(|e| VectorDbError::InvalidConfig(format!("FAISS factory '{factory_string}': {e}")))?;

        Ok(Self {
            config: IndexConfig { dimensions, metric },
            factory_string,
            vectors: HashMap::new(),
            state: Mutex::new(FaissState {
                index: None,
                id_map: Vec::new(),
            }),
        })
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// (Re)build a fresh FAISS index from `vectors` and return the new state.
    fn rebuild(
        config: &IndexConfig,
        factory_string: &str,
        vectors: &HashMap<u64, Vec<f32>>,
    ) -> Result<FaissState, VectorDbError> {
        let mut faiss_idx =
            index_factory(config.dimensions as u32, factory_string, faiss_metric(config.metric))
                .map_err(|e| VectorDbError::InvalidConfig(format!("FAISS factory: {e}")))?;

        let n = vectors.len();
        let d = config.dimensions;
        let mut id_map: Vec<u64> = Vec::with_capacity(n);
        let mut flat: Vec<f32> = Vec::with_capacity(n * d);

        for (&id, vec) in vectors {
            id_map.push(id);
            if config.metric == Metric::Cosine {
                // FAISS InnerProduct on unit vectors == cosine similarity.
                let norm = l2_norm(vec);
                if norm > 0.0 {
                    flat.extend(vec.iter().map(|&v| v / norm));
                } else {
                    flat.extend_from_slice(vec);
                }
            } else {
                flat.extend_from_slice(vec);
            }
        }

        if !flat.is_empty() {
            // IVF / PQ indexes need a training pass before vectors can be added.
            if !faiss_idx.is_trained() {
                faiss_idx
                    .train(&flat)
                    .map_err(|e| VectorDbError::InvalidConfig(format!("FAISS train: {e}")))?;
            }
            faiss_idx
                .add(&flat)
                .map_err(|e| VectorDbError::InvalidConfig(format!("FAISS add: {e}")))?;
        }

        Ok(FaissState {
            index: Some(faiss_idx),
            id_map,
        })
    }
}

// ── VectorIndex implementation ────────────────────────────────────────────────

impl VectorIndex for FaissIndex {
    fn add(&mut self, id: u64, vector: &[f32]) -> Result<(), VectorDbError> {
        if vector.len() != self.config.dimensions {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }
        self.vectors.insert(id, vector.to_vec());
        // Invalidate the FAISS index — it will be rebuilt on the next search.
        self.state.get_mut().unwrap().index = None;
        Ok(())
    }

    fn delete(&mut self, id: u64) -> bool {
        let removed = self.vectors.remove(&id).is_some();
        if removed {
            self.state.get_mut().unwrap().index = None;
        }
        removed
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, VectorDbError> {
        if self.vectors.is_empty() {
            return Ok(vec![]);
        }
        let k = k.min(self.vectors.len());

        // Optionally normalise the query for cosine searches.
        let query_vec: Vec<f32> = if self.config.metric == Metric::Cosine {
            let norm = l2_norm(query);
            if norm > 0.0 {
                query.iter().map(|&v| v / norm).collect()
            } else {
                query.to_vec()
            }
        } else {
            query.to_vec()
        };

        // Acquire the lock.  Lazy rebuild if the index was invalidated.
        let mut guard = self.state.lock().unwrap();
        if guard.index.is_none() {
            *guard = Self::rebuild(&self.config, &self.factory_string, &self.vectors)?;
        }

        // Run the FAISS search, then immediately drop the borrow on the index
        // so we can borrow `guard.id_map` in the conversion step below.
        let (labels, distances) = {
            let index = guard.index.as_mut().unwrap();
            let result = index
                .search(&query_vec, k)
                .map_err(|e| VectorDbError::InvalidConfig(format!("FAISS search: {e}")))?;
            (result.labels, result.distances)
        };

        // Convert FAISS internal offsets back to caller-supplied IDs.
        // FAISS pads with label = -1 when the index has fewer than k vectors.
        let results = labels
            .iter()
            .zip(distances.iter())
            .filter(|(&label, _)| label >= 0)
            .map(|(&label, &dist)| {
                let id = guard.id_map[label as usize];
                SearchResult {
                    id,
                    distance: convert_distance(dist, self.config.metric),
                }
            })
            .collect();

        Ok(results)
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

    /// Eagerly rebuild the FAISS index from the current vector set.
    ///
    /// Called automatically by `Collection::load()` after WAL replay, so the
    /// first search after a server restart does not pay the rebuild cost.
    fn flush(&mut self) {
        match Self::rebuild(&self.config, &self.factory_string, &self.vectors) {
            Ok(state) => *self.state.get_mut().unwrap() = state,
            Err(e) => warn!("FaissIndex::flush failed, index will rebuild on next search: {e}"),
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Map our `Metric` to the FAISS `MetricType`.
///
/// For `Cosine` we use `InnerProduct` because we pre-normalise vectors to
/// unit length, making inner product equivalent to cosine similarity.
fn faiss_metric(metric: Metric) -> MetricType {
    match metric {
        Metric::L2 => MetricType::L2,
        Metric::Cosine | Metric::DotProduct => MetricType::InnerProduct,
    }
}

/// Convert a raw FAISS distance to the convention used by this codebase
/// (lower value = more similar).
fn convert_distance(faiss_dist: f32, metric: Metric) -> f32 {
    match metric {
        // FAISS returns *squared* L2; take sqrt to match FlatIndex behaviour.
        Metric::L2 => faiss_dist.sqrt(),
        // FAISS InnerProduct on unit vectors = cosine similarity ∈ [−1, 1].
        // Our convention: distance = 1 − similarity (lower is better).
        Metric::Cosine => 1.0 - faiss_dist,
        // FAISS InnerProduct = raw dot product.
        // Our convention: negate so "lower = better" ordering holds.
        Metric::DotProduct => -faiss_dist,
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_index(metric: Metric, factory: &str) -> FaissIndex {
        FaissIndex::new(3, metric, factory).expect("FaissIndex::new")
    }

    #[test]
    fn flat_l2_basic_search() {
        let mut idx = make_index(Metric::L2, "Flat");
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0]).unwrap();
        idx.add(3, &[0.0, 0.0, 1.0]).unwrap();

        let results = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1);
        assert!(results[0].distance < 1e-5, "expected distance ≈ 0");
    }

    #[test]
    fn flat_l2_top_k() {
        let mut idx = make_index(Metric::L2, "Flat");
        for i in 0..10u64 {
            idx.add(i, &[i as f32, 0.0, 0.0]).unwrap();
        }
        let results = idx.search(&[0.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(results.len(), 3);
        // Closest to [0,0,0] should be id=0
        assert_eq!(results[0].id, 0);
    }

    #[test]
    fn delete_removes_vector() {
        let mut idx = make_index(Metric::L2, "Flat");
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0]).unwrap();
        idx.delete(1);

        assert_eq!(idx.len(), 1);
        let results = idx.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert!(results.iter().all(|r| r.id != 1));
    }

    #[test]
    fn flush_then_search() {
        let mut idx = make_index(Metric::L2, "Flat");
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.flush(); // eagerly build
        let results = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn cosine_metric() {
        let mut idx = make_index(Metric::Cosine, "Flat");
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0]).unwrap();
        let results = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, 1);
        // cosine distance to itself should be ≈ 0
        assert!(results[0].distance < 1e-5);
    }

    #[test]
    fn empty_index_returns_empty() {
        let idx = make_index(Metric::L2, "Flat");
        let results = idx.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn iter_vectors_roundtrip() {
        let mut idx = make_index(Metric::L2, "Flat");
        idx.add(10, &[1.0, 2.0, 3.0]).unwrap();
        idx.add(20, &[4.0, 5.0, 6.0]).unwrap();
        let mut ids: Vec<u64> = idx.iter_vectors().map(|(id, _)| id).collect();
        ids.sort();
        assert_eq!(ids, vec![10, 20]);
    }

    #[test]
    fn invalid_factory_string_returns_error() {
        let result = FaissIndex::new(4, Metric::L2, "NOT_A_VALID_FACTORY");
        assert!(result.is_err());
    }

    #[test]
    fn dimension_mismatch_returns_error() {
        let mut idx = make_index(Metric::L2, "Flat");
        let err = idx.add(1, &[1.0, 2.0]).unwrap_err(); // 2 dims, index is 3
        assert!(matches!(err, VectorDbError::DimensionMismatch { .. }));
    }
}
