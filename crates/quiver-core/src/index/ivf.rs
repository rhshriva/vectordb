//! Inverted File (IVF) index for sub-linear approximate nearest-neighbour search.
//!
//! # How it works
//!
//! 1. **Training** — when `train_size` vectors have been inserted, Lloyd's k-means
//!    runs on the buffer to produce `n_lists` centroid vectors.
//! 2. **Assignment** — every subsequent insert finds its nearest centroid and is
//!    appended to that centroid's *posting list*.
//! 3. **Search** — the query is compared to all centroids; the `nprobe` closest
//!    centroids are selected and their posting lists are scanned with brute-force
//!    to return the top-k results.
//!
//! # Complexity
//!
//! | Operation | Complexity              |
//! |-----------|-------------------------|
//! | Insert    | O(n_lists) after train  |
//! | Search    | O(n_lists + nprobe·N/n_lists) |
//! | Memory    | O(N·D) — same as Flat   |
//!
//! # Rule of thumb for `n_lists`
//!
//! `n_lists = sqrt(N)` gives the optimal lists-vs-list-length tradeoff.
//! E.g. 1 M vectors → `n_lists = 1000`, `nprobe = 50`.

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

// ── Configuration ──────────────────────────────────────────────────────────────

/// Configuration for an [`IvfIndex`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfConfig {
    /// Number of centroids (clusters).  Rule of thumb: `sqrt(expected_N)`.
    pub n_lists: usize,
    /// Number of posting lists to probe at search time.
    /// Higher = better recall, higher latency.
    pub nprobe: usize,
    /// Minimum number of vectors collected in staging before k-means training
    /// is triggered.
    pub train_size: usize,
    /// Maximum Lloyd's k-means iterations.
    pub max_iter: usize,
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            n_lists: 256,
            nprobe: 16,
            train_size: 4096,
            max_iter: 25,
        }
    }
}

// ── Snapshot (persistence) ─────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct IvfSnapshot {
    dimensions: usize,
    metric: Metric,
    ivf_config: IvfConfig,
    // Pre-training buffer: (id, vector)
    staging: Vec<(u64, Vec<f32>)>,
    // Post-training state
    centroids: Vec<Vec<f32>>,
    posting_lists: Vec<Vec<(u64, Vec<f32>)>>,
    id_to_list: HashMap<u64, usize>,
    trained: bool,
}

// ── Index ──────────────────────────────────────────────────────────────────────

/// Inverted-file approximate nearest-neighbour index.
///
/// Efficient for 100 K – 100 M vectors with sub-linear query latency.
/// Recall is tunable via `nprobe` (higher = better recall, slower queries).
pub struct IvfIndex {
    config: IndexConfig,
    ivf_config: IvfConfig,

    /// Pre-training staging buffer (id → vector).
    staging: HashMap<u64, Vec<f32>>,

    /// Centroid vectors (trained via k-means).
    centroids: Vec<Vec<f32>>,

    /// Posting lists indexed by centroid id: `posting_lists[c]` holds all
    /// vectors assigned to centroid `c`.
    posting_lists: Vec<Vec<(u64, Vec<f32>)>>,

    /// Maps each vector id to its centroid index for O(1) deletion.
    id_to_list: HashMap<u64, usize>,

    /// Whether k-means training has been performed.
    trained: bool,
}

impl IvfIndex {
    /// Create an empty, untrained index.
    pub fn new(dimensions: usize, metric: Metric, config: IvfConfig) -> Self {
        Self {
            config: IndexConfig { dimensions, metric },
            ivf_config: config,
            staging: HashMap::new(),
            centroids: Vec::new(),
            posting_lists: Vec::new(),
            id_to_list: HashMap::new(),
            trained: false,
        }
    }

    /// Serialize to a binary file at `path` (bincode format).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), VectorDbError> {
        let snapshot = IvfSnapshot {
            dimensions: self.config.dimensions,
            metric: self.config.metric,
            ivf_config: self.ivf_config.clone(),
            staging: self.staging.iter().map(|(&id, v)| (id, v.clone())).collect(),
            centroids: self.centroids.clone(),
            posting_lists: self.posting_lists.clone(),
            id_to_list: self.id_to_list.clone(),
            trained: self.trained,
        };
        let file = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(file);
        bincode::serialize_into(&mut writer, &snapshot)
            .map_err(|e| VectorDbError::Serialization(e.to_string()))
    }

    /// Deserialize from a binary file previously written by [`save`].
    pub fn load(path: impl AsRef<Path>) -> Result<Self, VectorDbError> {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);
        let snap: IvfSnapshot = bincode::deserialize_from(reader)
            .map_err(|e| VectorDbError::Serialization(e.to_string()))?;
        Ok(Self {
            config: IndexConfig { dimensions: snap.dimensions, metric: snap.metric },
            ivf_config: snap.ivf_config,
            staging: snap.staging.into_iter().collect(),
            centroids: snap.centroids,
            posting_lists: snap.posting_lists,
            id_to_list: snap.id_to_list,
            trained: snap.trained,
        })
    }

    // ── Training ───────────────────────────────────────────────────────────────

    /// Run Lloyd's k-means on the staging buffer and assign all staged vectors
    /// to posting lists.
    fn train(&mut self) {
        if self.staging.is_empty() {
            return;
        }

        let vectors: Vec<(u64, Vec<f32>)> = self.staging.drain().collect();
        let k = self.ivf_config.n_lists.min(vectors.len());

        self.centroids = kmeans(&vectors, k, self.config.metric, self.ivf_config.max_iter);
        self.posting_lists = vec![Vec::new(); self.centroids.len()];

        // Assign each staged vector to its nearest centroid.
        for (id, vec) in vectors {
            let c = nearest_centroid(&self.centroids, &vec, self.config.metric);
            self.id_to_list.insert(id, c);
            self.posting_lists[c].push((id, vec));
        }

        self.trained = true;
    }

    fn assign_to_centroid(&mut self, id: u64, vec: Vec<f32>) {
        let c = nearest_centroid(&self.centroids, &vec, self.config.metric);
        self.id_to_list.insert(id, c);
        self.posting_lists[c].push((id, vec));
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
}

impl VectorIndex for IvfIndex {
    fn add(&mut self, id: u64, vector: &[f32]) -> Result<(), VectorDbError> {
        if vector.len() != self.config.dimensions {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }

        if self.trained {
            // Post-training: assign directly to nearest centroid posting list.
            self.assign_to_centroid(id, vector.to_vec());
        } else {
            // Pre-training: buffer in staging.
            self.staging.insert(id, vector.to_vec());
            // Trigger training when we have enough vectors.
            if self.staging.len() >= self.ivf_config.train_size {
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
            // Score all centroids and pick top-nprobe.
            let nprobe = self.ivf_config.nprobe.min(self.centroids.len());
            let mut centroid_scores: Vec<(usize, f32)> = self
                .centroids
                .iter()
                .enumerate()
                .map(|(i, c)| (i, metric.distance(query, c)))
                .collect();
            centroid_scores.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Scan posting lists of the nprobe nearest centroids.
            for (c, _) in centroid_scores.iter().take(nprobe) {
                for (id, vec) in &self.posting_lists[*c] {
                    candidates.push(SearchResult {
                        id: *id,
                        distance: metric.distance(query, vec),
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
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(k);
        Ok(candidates)
    }

    fn delete(&mut self, id: u64) -> bool {
        // Check staging first.
        if self.staging.remove(&id).is_some() {
            return true;
        }
        // Look up posting list via id_to_list for O(1) list selection.
        if let Some(list_idx) = self.id_to_list.remove(&id) {
            if let Some(list) = self.posting_lists.get_mut(list_idx) {
                if let Some(pos) = list.iter().position(|(vid, _)| *vid == id) {
                    list.swap_remove(pos);
                    return true;
                }
            }
        }
        false
    }

    fn len(&self) -> usize {
        self.staging.len()
            + self.posting_lists.iter().map(|l| l.len()).sum::<usize>()
    }

    fn config(&self) -> &IndexConfig {
        &self.config
    }

    fn iter_vectors(&self) -> Box<dyn Iterator<Item = (u64, Vec<f32>)> + '_> {
        let staging_iter = self.staging.iter().map(|(&id, v)| (id, v.clone()));
        let lists_iter = self
            .posting_lists
            .iter()
            .flat_map(|list| list.iter().map(|(id, v)| (*id, v.clone())));
        Box::new(staging_iter.chain(lists_iter))
    }

    /// If the index was never trained (not enough vectors reached `train_size`),
    /// force training on whatever is in staging.
    fn flush(&mut self) {
        if !self.trained && !self.staging.is_empty() {
            self.train();
        }
    }
}

// ── K-means ────────────────────────────────────────────────────────────────────

/// Lloyd's algorithm: returns `k` centroid vectors.
fn kmeans(vectors: &[(u64, Vec<f32>)], k: usize, metric: Metric, max_iter: usize) -> Vec<Vec<f32>> {
    let n = vectors.len();
    let dims = vectors[0].1.len();
    let k = k.min(n);

    // Initialise centroids by random sampling (without replacement).
    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    let mut centroids: Vec<Vec<f32>> =
        indices[..k].iter().map(|&i| vectors[i].1.clone()).collect();

    let mut assignments = vec![0usize; n];

    for _ in 0..max_iter {
        let mut changed = false;

        // Assignment step.
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

        // Update step: recompute centroids as mean of assigned vectors.
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

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trained(n: usize) -> IvfIndex {
        let cfg = IvfConfig { n_lists: 4, nprobe: 4, train_size: 8, max_iter: 10 };
        let mut idx = IvfIndex::new(2, Metric::L2, cfg);
        for i in 0..n as u64 {
            idx.add(i, &[i as f32, 0.0]).unwrap();
        }
        idx
    }

    #[test]
    fn trains_after_threshold() {
        let idx = make_trained(16);
        assert!(idx.trained, "should be trained after reaching train_size");
        assert!(!idx.centroids.is_empty());
    }

    #[test]
    fn search_before_training_brute_force() {
        let cfg = IvfConfig { n_lists: 4, nprobe: 2, train_size: 100, max_iter: 10 };
        let mut idx = IvfIndex::new(2, Metric::L2, cfg);
        idx.add(1, &[1.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0]).unwrap();
        idx.add(3, &[2.0, 0.0]).unwrap();
        // Not yet trained; should brute-force over staging
        let r = idx.search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 1);
    }

    #[test]
    fn search_after_training_returns_nearest() {
        let idx = make_trained(20);
        let r = idx.search(&[10.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 10);
        assert!(r[0].distance < 1e-4);
    }

    #[test]
    fn top_k_ordered() {
        let idx = make_trained(20);
        let r = idx.search(&[10.0, 0.0], 5).unwrap();
        assert_eq!(r.len(), 5);
        for w in r.windows(2) {
            assert!(w[0].distance <= w[1].distance + 1e-5);
        }
    }

    #[test]
    fn delete_from_staging() {
        let cfg = IvfConfig { n_lists: 4, nprobe: 2, train_size: 100, max_iter: 10 };
        let mut idx = IvfIndex::new(2, Metric::L2, cfg);
        idx.add(1, &[1.0, 0.0]).unwrap();
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
        let r = idx.search(&[5.0, 0.0], 1).unwrap();
        assert_ne!(r[0].id, 5);
    }

    #[test]
    fn flush_trains_small_staging() {
        let cfg = IvfConfig { n_lists: 2, nprobe: 2, train_size: 100, max_iter: 5 };
        let mut idx = IvfIndex::new(2, Metric::L2, cfg);
        idx.add(1, &[1.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0]).unwrap();
        assert!(!idx.trained);
        idx.flush();
        assert!(idx.trained);
    }

    #[test]
    fn save_and_load_preserves_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ivf.bin");
        let idx = make_trained(20);
        idx.save(&path).unwrap();
        let loaded = IvfIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), 20);
        let r = loaded.search(&[10.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 10);
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
        idx.add(100, &[100.0, 0.0]).unwrap();
        assert_eq!(idx.len(), n_before + 1);
        // The new vector must be reachable by search
        let r = idx.search(&[100.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 100);
    }

    #[test]
    fn dimension_mismatch_errors() {
        let cfg = IvfConfig::default();
        let mut idx = IvfIndex::new(3, Metric::L2, cfg);
        let err = idx.add(1, &[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, VectorDbError::DimensionMismatch { expected: 3, got: 2 }));
    }
}
