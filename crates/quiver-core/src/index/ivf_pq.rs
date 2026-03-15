//! IVF-PQ index — coarse IVF quantizer with product-quantized residuals.
//!
//! Combines the IVF partitioning (k-means centroids → posting lists) with
//! product quantization of the *residual* vectors for dramatic memory savings.
//!
//! # How it works
//!
//! 1. **Training** — when `train_size` vectors have been collected:
//!    a. Run k-means to produce `n_lists` coarse centroids.
//!    b. Compute residuals: `residual[i] = vector[i] - nearest_centroid`.
//!    c. Train a PQ codebook on the residuals.
//!    d. Encode all residuals with PQ and store in posting lists.
//!
//! 2. **Insert (post-training)** — find nearest centroid, compute residual,
//!    encode with PQ, append to posting list.
//!
//! 3. **Search** — find `nprobe` nearest centroids, for each:
//!    a. Compute residual query: `q_residual = query - centroid`.
//!    b. Precompute PQ distance table from `q_residual`.
//!    c. Scan posting list with ADC (asymmetric distance computation).
//!    d. Merge candidates across all probed lists, return top-k.
//!
//! # Memory
//!
//! With PQ m=64 on 1536-dim vectors:
//! - Original f32: 6144 bytes/vector
//! - PQ-encoded:   64 bytes/vector (+ 8 bytes ID = 72 bytes)
//! - Compression:  ~85×

use std::collections::HashMap;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use crate::{
    distance::Metric,
    error::VectorDbError,
    index::{IndexConfig, SearchResult, VectorIndex},
};

use super::pq::{PqCodebook, PqCode, PqConfig};

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for an [`IvfPqIndex`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfPqConfig {
    /// Number of IVF lists (coarse centroids). Rule of thumb: `sqrt(N)`.
    pub n_lists: usize,
    /// Number of posting lists to probe at search time.
    pub nprobe: usize,
    /// Minimum vectors before k-means training is triggered.
    pub train_size: usize,
    /// Maximum Lloyd's k-means iterations for IVF centroids.
    pub max_iter: usize,
    /// PQ configuration for residual encoding.
    pub pq: PqConfig,
}

impl Default for IvfPqConfig {
    fn default() -> Self {
        Self {
            n_lists: 256,
            nprobe: 16,
            train_size: 4096,
            max_iter: 25,
            pq: PqConfig::default(),
        }
    }
}

// ── Snapshot (persistence) ───────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct IvfPqSnapshot {
    dimensions: usize,
    metric: Metric,
    config: IvfPqConfig,
    staging: Vec<(u64, Vec<f32>)>,
    centroids: Vec<Vec<f32>>,
    codebook: Option<PqCodebook>,
    posting_lists: Vec<Vec<(u64, PqCode)>>,
    originals: Vec<(u64, Vec<f32>)>,
    id_to_list: Vec<(u64, usize)>,
    trained: bool,
}

// ── Index ────────────────────────────────────────────────────────────────────

/// IVF-PQ composite index: coarse IVF quantizer + PQ-encoded residuals.
///
/// Memory-efficient approximate nearest-neighbour search with sub-linear
/// query complexity and dramatic compression of stored vectors.
pub struct IvfPqIndex {
    config: IndexConfig,
    ivf_pq_config: IvfPqConfig,

    /// Pre-training staging buffer.
    staging: HashMap<u64, Vec<f32>>,

    /// Coarse centroids (trained via k-means).
    centroids: Vec<Vec<f32>>,

    /// Trained PQ codebook for residual encoding.
    codebook: Option<PqCodebook>,

    /// Posting lists: PQ-encoded residuals per centroid.
    posting_lists: Vec<Vec<(u64, PqCode)>>,

    /// Original vectors for iter_vectors() and accurate reconstruction.
    originals: HashMap<u64, Vec<f32>>,

    /// Maps vector ID to centroid index for O(1) deletion.
    id_to_list: HashMap<u64, usize>,

    /// Whether training has been performed.
    trained: bool,
}

impl IvfPqIndex {
    /// Create an empty, untrained IVF-PQ index.
    pub fn new(dimensions: usize, metric: Metric, config: IvfPqConfig) -> Self {
        Self {
            config: IndexConfig { dimensions, metric },
            ivf_pq_config: config,
            staging: HashMap::new(),
            centroids: Vec::new(),
            codebook: None,
            posting_lists: Vec::new(),
            originals: HashMap::new(),
            id_to_list: HashMap::new(),
            trained: false,
        }
    }

    /// Serialize to a binary file (bincode format).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), VectorDbError> {
        let snapshot = IvfPqSnapshot {
            dimensions: self.config.dimensions,
            metric: self.config.metric,
            config: self.ivf_pq_config.clone(),
            staging: self.staging.iter().map(|(&id, v)| (id, v.clone())).collect(),
            centroids: self.centroids.clone(),
            codebook: self.codebook.clone(),
            posting_lists: self.posting_lists.clone(),
            originals: self.originals.iter().map(|(&id, v)| (id, v.clone())).collect(),
            id_to_list: self.id_to_list.iter().map(|(&k, &v)| (k, v)).collect(),
            trained: self.trained,
        };
        let file = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(file);
        bincode::serialize_into(&mut writer, &snapshot)
            .map_err(|e| VectorDbError::Serialization(e.to_string()))
    }

    /// Deserialize from a binary file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, VectorDbError> {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);
        let snap: IvfPqSnapshot = bincode::deserialize_from(reader)
            .map_err(|e| VectorDbError::Serialization(e.to_string()))?;
        Ok(Self {
            config: IndexConfig {
                dimensions: snap.dimensions,
                metric: snap.metric,
            },
            ivf_pq_config: snap.config,
            staging: snap.staging.into_iter().collect(),
            centroids: snap.centroids,
            codebook: snap.codebook,
            posting_lists: snap.posting_lists,
            originals: snap.originals.into_iter().collect(),
            id_to_list: snap.id_to_list.into_iter().collect(),
            trained: snap.trained,
        })
    }

    // ── Training ─────────────────────────────────────────────────────────────

    /// Run IVF k-means + PQ codebook training on the staging buffer.
    fn train(&mut self) {
        if self.staging.is_empty() {
            return;
        }

        let vectors: Vec<(u64, Vec<f32>)> = self.staging.drain().collect();
        let k = self.ivf_pq_config.n_lists.min(vectors.len());

        // Step 1: Train IVF coarse centroids.
        self.centroids = kmeans_vectors(&vectors, k, self.config.metric, self.ivf_pq_config.max_iter);
        self.posting_lists = vec![Vec::new(); self.centroids.len()];

        // Step 2: Compute residuals for PQ training.
        let mut assignments: Vec<(usize, Vec<f32>)> = Vec::with_capacity(vectors.len());
        for (_, vec) in &vectors {
            let c = nearest_centroid(&self.centroids, vec, self.config.metric);
            let residual: Vec<f32> = vec
                .iter()
                .zip(self.centroids[c].iter())
                .map(|(v, c)| v - c)
                .collect();
            assignments.push((c, residual));
        }

        // Step 3: Train PQ codebook on residuals.
        let residual_refs: Vec<&[f32]> = assignments.iter().map(|(_, r)| r.as_slice()).collect();
        let codebook = PqCodebook::train(&residual_refs, &self.ivf_pq_config.pq);

        // Step 4: Encode all vectors and assign to posting lists.
        for (idx, (id, vec)) in vectors.iter().enumerate() {
            let (c, residual) = &assignments[idx];
            let code = codebook.encode(residual);
            self.id_to_list.insert(*id, *c);
            self.posting_lists[*c].push((*id, code));
            self.originals.insert(*id, vec.clone());
        }

        self.codebook = Some(codebook);
        self.trained = true;
    }

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
        for i in 0..n {
            let slice = &raw_data[i * dim..(i + 1) * dim];
            let id = start_id + i as u64;
            self.add(id, slice)?;
        }
        Ok(())
    }

    /// Assign a single vector to its nearest centroid with PQ encoding.
    fn assign_to_centroid(&mut self, id: u64, vec: Vec<f32>) {
        let c = nearest_centroid(&self.centroids, &vec, self.config.metric);
        let residual: Vec<f32> = vec
            .iter()
            .zip(self.centroids[c].iter())
            .map(|(v, cent)| v - cent)
            .collect();
        let code = self.codebook.as_ref().unwrap().encode(&residual);
        self.id_to_list.insert(id, c);
        self.posting_lists[c].push((id, code));
        self.originals.insert(id, vec);
    }
}

impl VectorIndex for IvfPqIndex {
    fn add(&mut self, id: u64, vector: &[f32]) -> Result<(), VectorDbError> {
        if vector.len() != self.config.dimensions {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }

        if self.trained {
            self.assign_to_centroid(id, vector.to_vec());
        } else {
            self.staging.insert(id, vector.to_vec());
            if self.staging.len() >= self.ivf_pq_config.train_size {
                self.train();
            }
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
        if k == 0 {
            return Ok(vec![]);
        }

        let metric = self.config.metric;
        let mut candidates: Vec<SearchResult> = Vec::new();

        if !self.trained || self.centroids.is_empty() {
            // Fall back to brute-force over staging.
            candidates.extend(self.staging.iter().map(|(&id, v)| SearchResult {
                id,
                distance: metric.distance(query, v),
            }));
        } else {
            let codebook = self.codebook.as_ref().unwrap();
            let nprobe = self.ivf_pq_config.nprobe.min(self.centroids.len());

            // Score all centroids and pick top-nprobe.
            let mut centroid_scores: Vec<(usize, f32)> = self
                .centroids
                .iter()
                .enumerate()
                .map(|(i, c)| (i, metric.distance(query, c)))
                .collect();
            centroid_scores.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            // For each probed centroid, compute residual query + ADC scan.
            for &(c, _) in centroid_scores.iter().take(nprobe) {
                let residual_query: Vec<f32> = query
                    .iter()
                    .zip(self.centroids[c].iter())
                    .map(|(q, cent)| q - cent)
                    .collect();
                let table = codebook.compute_distance_table(&residual_query, metric);

                for (id, code) in &self.posting_lists[c] {
                    let dist = codebook.asymmetric_distance(&table, code);
                    candidates.push(SearchResult {
                        id: *id,
                        distance: dist,
                    });
                }
            }

            // Also scan staging buffer (vectors not yet trained into index).
            candidates.extend(self.staging.iter().map(|(&id, v)| SearchResult {
                id,
                distance: metric.distance(query, v),
            }));
        }

        candidates.sort_unstable_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(k);
        Ok(candidates)
    }

    fn delete(&mut self, id: u64) -> bool {
        // Check staging first.
        if self.staging.remove(&id).is_some() {
            self.originals.remove(&id);
            return true;
        }
        // Look up posting list.
        if let Some(list_idx) = self.id_to_list.remove(&id) {
            if let Some(list) = self.posting_lists.get_mut(list_idx) {
                if let Some(pos) = list.iter().position(|(vid, _)| *vid == id) {
                    list.swap_remove(pos);
                    self.originals.remove(&id);
                    return true;
                }
            }
        }
        false
    }

    fn len(&self) -> usize {
        self.staging.len() + self.posting_lists.iter().map(|l| l.len()).sum::<usize>()
    }

    fn config(&self) -> &IndexConfig {
        &self.config
    }

    fn iter_vectors(&self) -> Box<dyn Iterator<Item = (u64, Vec<f32>)> + '_> {
        let staging_iter = self.staging.iter().map(|(&id, v)| (id, v.clone()));
        let originals_iter = self
            .posting_lists
            .iter()
            .flat_map(|list| list.iter().map(|(id, _)| *id))
            .filter_map(move |id| self.originals.get(&id).map(|v| (id, v.clone())));
        Box::new(staging_iter.chain(originals_iter))
    }

    fn flush(&mut self) {
        if !self.trained && !self.staging.is_empty() {
            self.train();
        }
    }
}

// ── K-means (reused from ivf.rs pattern) ─────────────────────────────────────

/// Lloyd's algorithm on (id, vector) pairs. Returns `k` centroid vectors.
fn kmeans_vectors(
    vectors: &[(u64, Vec<f32>)],
    k: usize,
    metric: Metric,
    max_iter: usize,
) -> Vec<Vec<f32>> {
    let n = vectors.len();
    let dims = vectors[0].1.len();
    let k = k.min(n);

    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    let mut centroids: Vec<Vec<f32>> = indices[..k].iter().map(|&i| vectors[i].1.clone()).collect();

    let mut assignments = vec![0usize; n];

    for _ in 0..max_iter {
        let mut changed = false;
        for (i, (_, v)) in vectors.iter().enumerate() {
            let best = nearest_centroid(&centroids, v, metric);
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        let mut sums = vec![vec![0.0f32; dims]; k];
        let mut counts = vec![0usize; k];
        for (i, (_, v)) in vectors.iter().enumerate() {
            let c = assignments[i];
            for (d, &x) in v.iter().enumerate() {
                sums[c][d] += x;
            }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                let cnt = counts[c] as f32;
                for d in 0..dims {
                    centroids[c][d] = sums[c][d] / cnt;
                }
            }
        }
    }

    centroids
}

/// Return the index of the centroid nearest to `vec`.
#[inline]
fn nearest_centroid(centroids: &[Vec<f32>], vec: &[f32], metric: Metric) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, metric.distance(vec, c)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> IvfPqConfig {
        IvfPqConfig {
            n_lists: 4,
            nprobe: 4,
            train_size: 8,
            max_iter: 10,
            pq: PqConfig {
                m: 2,
                k_sub: 4,
                max_iter: 10,
            },
        }
    }

    fn make_trained(n: usize) -> IvfPqIndex {
        // Use 4-dim vectors so m=2 divides evenly.
        // Set train_size = n so ALL vectors are part of training, ensuring the
        // PQ codebook covers the full data distribution.
        let mut config = make_config();
        config.train_size = n;
        let mut idx = IvfPqIndex::new(4, Metric::L2, config);
        for i in 0..n as u64 {
            idx.add(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
        }
        idx
    }

    #[test]
    fn trains_after_threshold() {
        let idx = make_trained(16);
        assert!(idx.trained, "should be trained after reaching train_size");
        assert!(!idx.centroids.is_empty());
        assert!(idx.codebook.is_some());
    }

    #[test]
    fn search_before_training_brute_force() {
        let mut idx = IvfPqIndex::new(4, Metric::L2, IvfPqConfig {
            train_size: 100,
            ..make_config()
        });
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        let r = idx.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 1);
    }

    #[test]
    fn search_after_training_returns_nearest() {
        let idx = make_trained(20);
        // IVF-PQ is approximate — check that the true nearest is in the top-3.
        let r = idx.search(&[10.0, 0.0, 0.0, 0.0], 3).unwrap();
        assert!(
            r.iter().any(|s| s.id == 10),
            "expected id=10 in top-3, got {:?}",
            r.iter().map(|s| s.id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn top_k_ordered() {
        let idx = make_trained(20);
        let r = idx.search(&[10.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(r.len(), 5);
        for w in r.windows(2) {
            assert!(w[0].distance <= w[1].distance + 1e-3);
        }
    }

    #[test]
    fn delete_from_staging() {
        let mut idx = IvfPqIndex::new(4, Metric::L2, IvfPqConfig {
            train_size: 100,
            ..make_config()
        });
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(idx.delete(1));
        assert_eq!(idx.len(), 0);
        assert!(!idx.delete(1));
    }

    #[test]
    fn delete_from_posting_list() {
        let mut idx = make_trained(16);
        let n_before = idx.len();
        assert!(idx.delete(5));
        assert_eq!(idx.len(), n_before - 1);
        let r = idx.search(&[5.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_ne!(r[0].id, 5);
    }

    #[test]
    fn flush_trains_small_staging() {
        let mut idx = IvfPqIndex::new(4, Metric::L2, IvfPqConfig {
            train_size: 100,
            ..make_config()
        });
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        assert!(!idx.trained);
        idx.flush();
        assert!(idx.trained);
    }

    #[test]
    fn save_and_load_preserves_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ivfpq.bin");
        let idx = make_trained(20);
        idx.save(&path).unwrap();
        let loaded = IvfPqIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), 20);
        // IVF-PQ is approximate — check that the true nearest is in the top-3.
        let r = loaded.search(&[10.0, 0.0, 0.0, 0.0], 3).unwrap();
        assert!(
            r.iter().any(|s| s.id == 10),
            "expected id=10 in top-3 after load, got {:?}",
            r.iter().map(|s| s.id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn iter_vectors_yields_all() {
        let idx = make_trained(16);
        let mut ids: Vec<u64> = idx.iter_vectors().map(|(id, _)| id).collect();
        ids.sort();
        assert_eq!(ids, (0u64..16).collect::<Vec<_>>());
    }

    #[test]
    fn post_training_inserts_go_to_lists() {
        let mut idx = make_trained(16);
        let n_before = idx.len();
        // Use a value within the training range so the PQ codebook encodes it well.
        idx.add(100, &[8.5, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(idx.len(), n_before + 1);
        // Post-training insert should be searchable. Use k=10 because PQ
        // quantization error with small codebooks can push the exact match
        // outside top-3.
        let r = idx.search(&[8.5, 0.0, 0.0, 0.0], 10).unwrap();
        assert!(
            r.iter().any(|s| s.id == 100),
            "post-training vector should be findable in top-10, got {:?}",
            r.iter().map(|s| s.id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn dimension_mismatch_errors() {
        let mut idx = IvfPqIndex::new(4, Metric::L2, make_config());
        let err = idx.add(1, &[1.0, 2.0]).unwrap_err();
        assert!(matches!(
            err,
            VectorDbError::DimensionMismatch {
                expected: 4,
                got: 2
            }
        ));
    }

    // ── Additional coverage tests ─────────────────────────────────────────────

    #[test]
    fn search_dimension_mismatch() {
        let idx = IvfPqIndex::new(4, Metric::L2, make_config());
        let err = idx.search(&[1.0, 2.0], 1).unwrap_err();
        assert!(matches!(err, VectorDbError::DimensionMismatch { expected: 4, got: 2 }));
    }

    #[test]
    fn search_k_zero_returns_empty() {
        let mut idx = IvfPqIndex::new(4, Metric::L2, IvfPqConfig {
            train_size: 100,
            ..make_config()
        });
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let r = idx.search(&[1.0, 0.0, 0.0, 0.0], 0).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn search_k_greater_than_len() {
        let mut idx = IvfPqIndex::new(4, Metric::L2, IvfPqConfig {
            train_size: 100,
            ..make_config()
        });
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        let r = idx.search(&[1.0, 0.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn search_empty_index() {
        let idx = IvfPqIndex::new(4, Metric::L2, make_config());
        let r = idx.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn add_batch_raw_happy_path() {
        let mut idx = IvfPqIndex::new(4, Metric::L2, IvfPqConfig {
            train_size: 100,
            ..make_config()
        });
        let data = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            2.0, 0.0, 0.0, 0.0,
        ];
        idx.add_batch_raw(&data, 4, 10).unwrap();
        assert_eq!(idx.len(), 3);
        let r = idx.search(&[2.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 12);
    }

    #[test]
    fn add_batch_raw_dimension_mismatch() {
        let mut idx = IvfPqIndex::new(4, Metric::L2, make_config());
        let data = vec![1.0, 0.0];
        let err = idx.add_batch_raw(&data, 2, 0).unwrap_err();
        assert!(matches!(err, VectorDbError::DimensionMismatch { expected: 4, got: 2 }));
    }

    #[test]
    fn add_batch_raw_not_multiple_of_dim() {
        let mut idx = IvfPqIndex::new(4, Metric::L2, make_config());
        let data = vec![1.0, 0.0, 0.0]; // 3 elements, not multiple of 4
        let err = idx.add_batch_raw(&data, 4, 0).unwrap_err();
        assert!(matches!(err, VectorDbError::InvalidConfig(_)));
    }

    #[test]
    fn add_batch_raw_triggers_training() {
        let cfg = IvfPqConfig {
            n_lists: 2,
            nprobe: 2,
            train_size: 4,
            max_iter: 10,
            pq: PqConfig { m: 2, k_sub: 4, max_iter: 10 },
        };
        let mut idx = IvfPqIndex::new(4, Metric::L2, cfg);
        let data: Vec<f32> = (0..16).map(|i| {
            if i % 4 == 0 { (i / 4) as f32 } else { 0.0 }
        }).collect();
        idx.add_batch_raw(&data, 4, 0).unwrap();
        assert!(idx.trained);
        assert_eq!(idx.len(), 4);
    }

    #[test]
    fn add_batch_raw_empty_buffer() {
        let mut idx = IvfPqIndex::new(4, Metric::L2, make_config());
        idx.add_batch_raw(&[], 4, 0).unwrap();
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn delete_nonexistent_from_trained_index() {
        let mut idx = make_trained(16);
        assert!(!idx.delete(999));
    }

    #[test]
    fn cosine_metric_search() {
        let cfg = IvfPqConfig {
            n_lists: 2,
            nprobe: 2,
            train_size: 8,
            max_iter: 10,
            pq: PqConfig { m: 2, k_sub: 4, max_iter: 10 },
        };
        let mut idx = IvfPqIndex::new(4, Metric::Cosine, cfg);
        for i in 0..8u64 {
            let v = i as f32 + 1.0;
            idx.add(i, &[v, v, v, v]).unwrap();
        }
        // All vectors point in the same direction; PQ adds quantization error
        // so we just verify a result is returned and results are ordered
        let r = idx.search(&[1.0, 1.0, 1.0, 1.0], 3).unwrap();
        assert!(!r.is_empty());
        for w in r.windows(2) {
            assert!(w[0].distance <= w[1].distance + 1e-3);
        }
    }

    #[test]
    fn dot_product_metric_search() {
        let cfg = IvfPqConfig {
            n_lists: 2,
            nprobe: 2,
            train_size: 8,
            max_iter: 10,
            pq: PqConfig { m: 2, k_sub: 4, max_iter: 10 },
        };
        let mut idx = IvfPqIndex::new(4, Metric::DotProduct, cfg);
        for i in 0..8u64 {
            idx.add(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
        }
        let r = idx.search(&[1.0, 0.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(r.len(), 3);
        for w in r.windows(2) {
            assert!(w[0].distance <= w[1].distance + 1e-3);
        }
    }

    #[test]
    fn flush_on_already_trained_is_noop() {
        let mut idx = make_trained(16);
        assert!(idx.trained);
        let len_before = idx.len();
        idx.flush();
        assert_eq!(idx.len(), len_before);
    }

    #[test]
    fn flush_on_empty_staging_is_noop() {
        let idx_cfg = IvfPqConfig { train_size: 100, ..make_config() };
        let mut idx = IvfPqIndex::new(4, Metric::L2, idx_cfg);
        assert!(!idx.trained);
        idx.flush();
        assert!(!idx.trained);
    }

    #[test]
    fn iter_vectors_untrained() {
        let mut idx = IvfPqIndex::new(4, Metric::L2, IvfPqConfig {
            train_size: 100,
            ..make_config()
        });
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        let mut ids: Vec<u64> = idx.iter_vectors().map(|(id, _)| id).collect();
        ids.sort();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn save_load_untrained() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ivfpq_untrained.bin");
        let mut idx = IvfPqIndex::new(4, Metric::L2, IvfPqConfig {
            train_size: 100,
            ..make_config()
        });
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        idx.save(&path).unwrap();
        let loaded = IvfPqIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert!(!loaded.trained);
        let r = loaded.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 1);
    }

    #[test]
    fn load_nonexistent_file_errors() {
        let result = IvfPqIndex::load("/tmp/nonexistent_ivfpq_test_file.bin");
        assert!(matches!(result, Err(VectorDbError::Io(_))));
    }

    #[test]
    fn nprobe_clamped_to_centroids_len() {
        let cfg = IvfPqConfig {
            n_lists: 2,
            nprobe: 100, // much larger than n_lists
            train_size: 8,
            max_iter: 10,
            pq: PqConfig { m: 2, k_sub: 4, max_iter: 10 },
        };
        let mut idx = IvfPqIndex::new(4, Metric::L2, cfg);
        for i in 0..8u64 {
            idx.add(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
        }
        let r = idx.search(&[3.0, 0.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn config_returns_correct_values() {
        let idx = IvfPqIndex::new(8, Metric::Cosine, make_config());
        assert_eq!(idx.config().dimensions, 8);
        assert_eq!(idx.config().metric, Metric::Cosine);
    }

    #[test]
    fn len_counts_staging_and_posting_lists() {
        let cfg = IvfPqConfig {
            n_lists: 2,
            nprobe: 2,
            train_size: 4,
            max_iter: 10,
            pq: PqConfig { m: 2, k_sub: 4, max_iter: 10 },
        };
        let mut idx = IvfPqIndex::new(4, Metric::L2, cfg);
        assert_eq!(idx.len(), 0);
        idx.add(0, &[0.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(idx.len(), 1);
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(idx.len(), 2);
        idx.add(2, &[2.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(idx.len(), 3);
        idx.add(3, &[3.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(idx.trained);
        assert_eq!(idx.len(), 4);
    }

    #[test]
    fn kmeans_with_fewer_vectors_than_lists() {
        let cfg = IvfPqConfig {
            n_lists: 4,
            nprobe: 4,
            train_size: 2,
            max_iter: 5,
            pq: PqConfig { m: 2, k_sub: 2, max_iter: 5 },
        };
        let mut idx = IvfPqIndex::new(4, Metric::L2, cfg);
        idx.add(0, &[0.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(1, &[10.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(idx.trained);
        assert!(idx.centroids.len() <= 2);
    }

    #[test]
    fn delete_all_from_posting_list() {
        let mut idx = make_trained(16);
        for i in 0..16u64 {
            assert!(idx.delete(i));
        }
        assert_eq!(idx.len(), 0);
        let r = idx.search(&[5.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn delete_from_staging_removes_original() {
        let mut idx = IvfPqIndex::new(4, Metric::L2, IvfPqConfig {
            train_size: 100,
            ..make_config()
        });
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(idx.delete(1));
        // Verify iter_vectors is also empty (originals cleaned up)
        assert_eq!(idx.iter_vectors().count(), 0);
    }

    #[test]
    fn iter_vectors_after_delete() {
        let mut idx = make_trained(16);
        idx.delete(5);
        idx.delete(10);
        let mut ids: Vec<u64> = idx.iter_vectors().map(|(id, _)| id).collect();
        ids.sort();
        let expected: Vec<u64> = (0..16).filter(|&i| i != 5 && i != 10).collect();
        assert_eq!(ids, expected);
    }

    #[test]
    fn save_load_preserves_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ivfpq_cfg.bin");
        let cfg = IvfPqConfig {
            n_lists: 3,
            nprobe: 2,
            train_size: 8,
            max_iter: 7,
            pq: PqConfig { m: 2, k_sub: 4, max_iter: 10 },
        };
        let mut idx = IvfPqIndex::new(4, Metric::Cosine, cfg);
        for i in 0..8u64 {
            let v = i as f32 + 1.0;
            idx.add(i, &[v, v, 0.0, 0.0]).unwrap();
        }
        idx.save(&path).unwrap();
        let loaded = IvfPqIndex::load(&path).unwrap();
        assert_eq!(loaded.config().dimensions, 4);
        assert_eq!(loaded.config().metric, Metric::Cosine);
        assert_eq!(loaded.len(), 8);
        assert!(loaded.trained);
    }
}
