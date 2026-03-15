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
    distance::{self, Metric},
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
            max_iter: 10,
        }
    }
}

// ── Contiguous posting list ──────────────────────────────────────────────────

/// A posting list storing vectors contiguously for cache-friendly access.
#[derive(Clone, Serialize, Deserialize)]
struct PostingList {
    ids: Vec<u64>,
    /// Contiguous vector data: `[vec0_f32s | vec1_f32s | ...]`
    data: Vec<f32>,
    dim: usize,
}

impl PostingList {
    fn new(dim: usize) -> Self {
        Self {
            ids: Vec::new(),
            data: Vec::new(),
            dim,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.ids.len()
    }

    #[inline]
    fn push(&mut self, id: u64, vector: &[f32]) {
        self.ids.push(id);
        self.data.extend_from_slice(vector);
    }

    #[inline]
    fn get_vector(&self, idx: usize) -> &[f32] {
        &self.data[idx * self.dim..(idx + 1) * self.dim]
    }

    /// Remove by position using swap-remove semantics.
    fn swap_remove(&mut self, pos: usize) {
        let last = self.len() - 1;
        if pos != last {
            self.ids.swap(pos, last);
            // Swap vector data
            let dim = self.dim;
            for d in 0..dim {
                self.data.swap(pos * dim + d, last * dim + d);
            }
        }
        self.ids.pop();
        self.data.truncate(last * self.dim);
    }

    fn position(&self, id: u64) -> Option<usize> {
        self.ids.iter().position(|&vid| vid == id)
    }

    fn reserve(&mut self, additional: usize) {
        self.ids.reserve(additional);
        self.data.reserve(additional * self.dim);
    }
}

// ── Snapshot (persistence) ─────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct IvfSnapshot {
    dimensions: usize,
    metric: Metric,
    ivf_config: IvfConfig,
    // Pre-training buffer — contiguous storage
    staging_ids: Vec<u64>,
    staging_data: Vec<f32>,
    // Post-training state — contiguous centroids
    centroid_data: Vec<f32>,
    nlist: usize,
    posting_lists: Vec<PostingList>,
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

    /// Pre-training staging buffer — contiguous storage for cache-friendly k-means.
    staging_ids: Vec<u64>,
    staging_data: Vec<f32>,

    /// Contiguous centroid vectors: `[c0_f32s | c1_f32s | ...]`, length = nlist * dim.
    centroid_data: Vec<f32>,

    /// Number of centroids.
    nlist: usize,

    /// Posting lists indexed by centroid id.
    posting_lists: Vec<PostingList>,

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
            staging_ids: Vec::new(),
            staging_data: Vec::new(),
            centroid_data: Vec::new(),
            nlist: 0,
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
            staging_ids: self.staging_ids.clone(),
            staging_data: self.staging_data.clone(),
            centroid_data: self.centroid_data.clone(),
            nlist: self.nlist,
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
            staging_ids: snap.staging_ids,
            staging_data: snap.staging_data,
            centroid_data: snap.centroid_data,
            nlist: snap.nlist,
            posting_lists: snap.posting_lists,
            id_to_list: snap.id_to_list,
            trained: snap.trained,
        })
    }

    // ── Centroid access ────────────────────────────────────────────────────────

    #[inline]
    fn centroid(&self, idx: usize) -> &[f32] {
        let dim = self.config.dimensions;
        &self.centroid_data[idx * dim..(idx + 1) * dim]
    }

    // ── Nearest centroid (batch4 SIMD) ─────────────────────────────────────────

    /// Find the nearest centroid to `vector` using batch4 SIMD acceleration.
    #[inline]
    fn nearest_centroid(&self, vector: &[f32]) -> usize {
        let nlist = self.nlist;
        let dim = self.config.dimensions;
        let metric = self.config.metric;

        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        // Process 4 centroids at a time using SIMD batch4
        let chunks = nlist / 4;
        for chunk in 0..chunks {
            let base = chunk * 4;
            let dists = metric.distance_ord_batch4(
                vector,
                [
                    &self.centroid_data[base * dim..(base + 1) * dim],
                    &self.centroid_data[(base + 1) * dim..(base + 2) * dim],
                    &self.centroid_data[(base + 2) * dim..(base + 3) * dim],
                    &self.centroid_data[(base + 3) * dim..(base + 4) * dim],
                ],
            );
            for (j, &d) in dists.iter().enumerate() {
                if d < best_dist {
                    best_dist = d;
                    best_idx = base + j;
                }
            }
        }

        // Scalar tail for remaining centroids
        for i in (chunks * 4)..nlist {
            let d = metric.distance_ord(vector, self.centroid(i));
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }

        best_idx
    }

    // ── Training ───────────────────────────────────────────────────────────────

    /// Run Lloyd's k-means on the staging buffer and assign all staged vectors
    /// to posting lists.
    fn train(&mut self) {
        let n = self.staging_ids.len();
        if n == 0 {
            return;
        }

        let k = self.ivf_config.n_lists.min(n);
        let dim = self.config.dimensions;

        // Run k-means on contiguous staging data — returns centroids + final assignments
        let (centroid_data, assignments) = kmeans_contiguous(
            &self.staging_data, n, k, dim, self.config.metric, self.ivf_config.max_iter,
        );
        self.centroid_data = centroid_data;
        self.nlist = k;
        self.posting_lists = (0..k).map(|_| PostingList::new(dim)).collect();

        // Pre-allocate id_to_list
        self.id_to_list.reserve(n);

        // Use assignments directly from k-means — no redundant centroid search
        for i in 0..n {
            let c = assignments[i];
            self.id_to_list.insert(self.staging_ids[i], c);
            self.posting_lists[c].push(
                self.staging_ids[i],
                &self.staging_data[i * dim..(i + 1) * dim],
            );
        }

        // Clear staging
        self.staging_ids.clear();
        self.staging_data.clear();
        self.trained = true;
    }

    fn assign_to_centroid(&mut self, id: u64, vec: &[f32]) {
        let c = self.nearest_centroid(vec);
        self.id_to_list.insert(id, c);
        self.posting_lists[c].push(id, vec);
    }

    /// Bulk-insert from a contiguous f32 buffer (row-major, `n_rows × dim`).
    /// Assigns sequential IDs starting from `start_id`.
    ///
    /// Uses a 3-phase pipeline for throughput:
    /// 1. Assign all vectors to centroids (batch4 SIMD)
    /// 2. Count per-list and reserve capacity
    /// 3. Scatter-append vectors to posting lists
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
        if n == 0 {
            return Ok(());
        }

        if !self.trained {
            // Pre-training: add to staging one by one (training may trigger mid-batch)
            for i in 0..n {
                let slice = &raw_data[i * dim..(i + 1) * dim];
                let id = start_id + i as u64;
                self.add(id, slice)?;
            }
            return Ok(());
        }

        // Post-training: true batch insert with 3-phase pipeline
        self.id_to_list.reserve(n);

        // Phase 1: Assign all vectors to centroids (GEMM for L2, batch4 for others)
        let assignments: Vec<usize> = if self.config.metric == Metric::L2 {
            batch_assign_l2_argmin(raw_data, &self.centroid_data, n, self.nlist, dim)
        } else {
            (0..n)
                .map(|i| self.nearest_centroid(&raw_data[i * dim..(i + 1) * dim]))
                .collect()
        };

        // Phase 2: Count per-list and reserve capacity
        let mut counts = vec![0usize; self.nlist];
        for &a in &assignments {
            counts[a] += 1;
        }
        for (list, &c) in self.posting_lists.iter_mut().zip(counts.iter()) {
            if c > 0 {
                list.reserve(c);
            }
        }

        // Phase 3: Scatter-append
        for i in 0..n {
            let id = start_id + i as u64;
            let vec = &raw_data[i * dim..(i + 1) * dim];
            let c = assignments[i];
            self.id_to_list.insert(id, c);
            self.posting_lists[c].push(id, vec);
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
            self.assign_to_centroid(id, vector);
        } else {
            // Pre-training: buffer in contiguous staging.
            self.staging_ids.push(id);
            self.staging_data.extend_from_slice(vector);
            // Trigger training when we have enough vectors.
            if self.staging_ids.len() >= self.ivf_config.train_size {
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

        if !self.trained || self.nlist == 0 {
            // Fall back to brute-force over staging.
            let dim = self.config.dimensions;
            candidates.extend(self.staging_ids.iter().enumerate().map(|(i, &id)| SearchResult {
                id,
                distance: metric.distance(query, &self.staging_data[i * dim..(i + 1) * dim]),
            }));
        } else {
            // Score all centroids using batch4 and pick top-nprobe.
            let nprobe = self.ivf_config.nprobe.min(self.nlist);
            let nlist = self.nlist;

            let mut centroid_scores: Vec<(usize, f32)> = Vec::with_capacity(nlist);

            // Batch4 centroid scoring
            let chunks = nlist / 4;
            let dim = self.config.dimensions;
            for chunk in 0..chunks {
                let base = chunk * 4;
                let dists = metric.distance_ord_batch4(
                    query,
                    [
                        &self.centroid_data[base * dim..(base + 1) * dim],
                        &self.centroid_data[(base + 1) * dim..(base + 2) * dim],
                        &self.centroid_data[(base + 2) * dim..(base + 3) * dim],
                        &self.centroid_data[(base + 3) * dim..(base + 4) * dim],
                    ],
                );
                for (j, &d) in dists.iter().enumerate() {
                    centroid_scores.push((base + j, d));
                }
            }
            // Scalar tail
            for i in (chunks * 4)..nlist {
                centroid_scores.push((i, metric.distance_ord(query, self.centroid(i))));
            }

            centroid_scores.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Scan posting lists of the nprobe nearest centroids.
            for &(c, _) in centroid_scores.iter().take(nprobe) {
                let list = &self.posting_lists[c];
                for j in 0..list.len() {
                    candidates.push(SearchResult {
                        id: list.ids[j],
                        distance: metric.distance(query, list.get_vector(j)),
                    });
                }
            }

            // Also scan staging buffer (vectors not yet trained into index).
            let dim = self.config.dimensions;
            candidates.extend(self.staging_ids.iter().enumerate().map(|(i, &id)| SearchResult {
                id,
                distance: metric.distance(query, &self.staging_data[i * dim..(i + 1) * dim]),
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
        if let Some(pos) = self.staging_ids.iter().position(|&sid| sid == id) {
            let dim = self.config.dimensions;
            let last = self.staging_ids.len() - 1;
            if pos != last {
                self.staging_ids.swap(pos, last);
                for d in 0..dim {
                    self.staging_data.swap(pos * dim + d, last * dim + d);
                }
            }
            self.staging_ids.pop();
            self.staging_data.truncate(last * dim);
            return true;
        }
        // Look up posting list via id_to_list for O(1) list selection.
        if let Some(list_idx) = self.id_to_list.remove(&id) {
            if let Some(list) = self.posting_lists.get_mut(list_idx) {
                if let Some(pos) = list.position(id) {
                    list.swap_remove(pos);
                    return true;
                }
            }
        }
        false
    }

    fn len(&self) -> usize {
        self.staging_ids.len()
            + self.posting_lists.iter().map(|l| l.len()).sum::<usize>()
    }

    fn config(&self) -> &IndexConfig {
        &self.config
    }

    fn iter_vectors(&self) -> Box<dyn Iterator<Item = (u64, Vec<f32>)> + '_> {
        let dim = self.config.dimensions;
        let staging_iter = self.staging_ids.iter().enumerate().map(move |(i, &id)| {
            (id, self.staging_data[i * dim..(i + 1) * dim].to_vec())
        });
        let lists_iter = self
            .posting_lists
            .iter()
            .flat_map(|list| {
                (0..list.len()).map(move |j| (list.ids[j], list.get_vector(j).to_vec()))
            });
        Box::new(staging_iter.chain(lists_iter))
    }

    /// If the index was never trained (not enough vectors reached `train_size`),
    /// force training on whatever is in staging.
    fn flush(&mut self) {
        if !self.trained && !self.staging_ids.is_empty() {
            self.train();
        }
    }
}

// ── K-means (contiguous) ────────────────────────────────────────────────────────

/// Lloyd's algorithm on contiguous data, returning `(centroids, assignments)`.
///
/// `data` is `n * dim` contiguous f32 values (row-major).
/// Returns contiguous centroid data (`k * dim`) and per-vector cluster assignments.
fn kmeans_contiguous(
    data: &[f32],
    n: usize,
    k: usize,
    dim: usize,
    metric: Metric,
    max_iter: usize,
) -> (Vec<f32>, Vec<usize>) {
    let k = k.min(n);

    // Initialise centroids by random sampling (without replacement).
    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);

    // Contiguous centroid storage: k * dim
    let mut centroids = vec![0.0f32; k * dim];
    for (ci, &src_idx) in indices[..k].iter().enumerate() {
        centroids[ci * dim..(ci + 1) * dim]
            .copy_from_slice(&data[src_idx * dim..(src_idx + 1) * dim]);
    }

    let mut assignments = vec![0usize; n];

    for _ in 0..max_iter {
        let mut changed = false;

        // Assignment step — use GEMM decomposition for L2, batch4 for others
        if metric == Metric::L2 {
            batch_assign_l2(data, &centroids, n, k, dim, &mut assignments, &mut changed);
        } else {
            for i in 0..n {
                let vec = &data[i * dim..(i + 1) * dim];
                let best = nearest_centroid_contiguous(&centroids, k, vec, metric);
                if assignments[i] != best {
                    assignments[i] = best;
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }

        // Update step: recompute centroids as mean of assigned vectors.
        let mut sums = vec![0.0f32; k * dim];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            let sums_slice = &mut sums[c * dim..(c + 1) * dim];
            let vec_slice = &data[i * dim..(i + 1) * dim];
            for d in 0..dim {
                sums_slice[d] += vec_slice[d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let cnt = counts[c] as f32;
                let base = c * dim;
                for d in 0..dim {
                    centroids[base + d] = sums[base + d] / cnt;
                }
            }
        }
    }

    (centroids, assignments)
}

/// Batch L2 assignment using true GEMM (matrix multiply):
/// `||x_i - c_j||² = ||x_i||² + ||c_j||² - 2·dot(x_i, c_j)`
///
/// The cross-term `-2·X·C^T` is a single (N×D)×(D×K) matrix multiply,
/// delegated to `matrixmultiply::sgemm` which uses BLIS-style cache-blocked
/// micro-kernels with AVX/FMA on x86-64 and NEON on AArch64.
///
/// This replaces N*K individual dot products with one highly-optimised matmul,
/// giving ~3-5× speedup on the assignment step.
fn batch_assign_l2(
    data: &[f32],
    centroids: &[f32],
    n: usize,
    k: usize,
    dim: usize,
    assignments: &mut [usize],
    changed: &mut bool,
) {
    // Precompute ||x_i||² for all N vectors
    let vec_norms: Vec<f32> = (0..n)
        .map(|i| {
            let v = &data[i * dim..(i + 1) * dim];
            distance::dot(v, v)
        })
        .collect();

    // Precompute ||c_j||² for all K centroids
    let cent_norms: Vec<f32> = (0..k)
        .map(|j| {
            let c = &centroids[j * dim..(j + 1) * dim];
            distance::dot(c, c)
        })
        .collect();

    // Compute dot-product matrix: ip[i*k + j] = dot(x_i, c_j)
    // This is the (N × D) × (D × K) matmul, the expensive part.
    //
    // data is row-major (N × D): row stride = dim, col stride = 1
    // centroids is row-major (K × D), but we need C^T (D × K):
    //   treating centroids as (D × K) column-major = row stride 1, col stride dim
    // output ip is row-major (N × K): row stride = k, col stride = 1
    let mut ip = vec![0.0f32; n * k];
    unsafe {
        matrixmultiply::sgemm(
            n,          // m = number of vectors
            dim,        // k = dimension
            k,          // n = number of centroids
            1.0,        // alpha (we'll apply -2 when assembling distances)
            data.as_ptr(),
            dim as isize,   // rsa: row stride of A (row-major N×D)
            1,              // csa: col stride of A
            centroids.as_ptr(),
            1,              // rsb: row stride of B = C^T, i.e. col stride of C (K×D row-major)
            dim as isize,   // csb: col stride of B = row stride of C
            0.0,        // beta
            ip.as_mut_ptr(),
            k as isize,     // rsc: row stride of output (N×K row-major)
            1,              // csc: col stride of output
        );
    }

    // Assemble distances and find argmin for each vector
    for i in 0..n {
        let xn = vec_norms[i];
        let row = &ip[i * k..(i + 1) * k];
        let mut best_dist = f32::MAX;
        let mut best_idx = 0;

        for j in 0..k {
            let dist = xn + cent_norms[j] - 2.0 * row[j];
            if dist < best_dist {
                best_dist = dist;
                best_idx = j;
            }
        }

        if assignments[i] != best_idx {
            assignments[i] = best_idx;
            *changed = true;
        }
    }
}

/// Batch L2 nearest-centroid assignment using GEMM — returns argmin assignments only.
///
/// Used by `add_batch_raw` for post-training batch inserts.
fn batch_assign_l2_argmin(
    data: &[f32],
    centroids: &[f32],
    n: usize,
    k: usize,
    dim: usize,
) -> Vec<usize> {
    // Precompute norms
    let vec_norms: Vec<f32> = (0..n)
        .map(|i| {
            let v = &data[i * dim..(i + 1) * dim];
            distance::dot(v, v)
        })
        .collect();
    let cent_norms: Vec<f32> = (0..k)
        .map(|j| {
            let c = &centroids[j * dim..(j + 1) * dim];
            distance::dot(c, c)
        })
        .collect();

    // GEMM: ip = data × centroids^T  (N×K)
    let mut ip = vec![0.0f32; n * k];
    unsafe {
        matrixmultiply::sgemm(
            n, dim, k,
            1.0,
            data.as_ptr(), dim as isize, 1,
            centroids.as_ptr(), 1, dim as isize,
            0.0,
            ip.as_mut_ptr(), k as isize, 1,
        );
    }

    // Argmin for each vector
    (0..n)
        .map(|i| {
            let xn = vec_norms[i];
            let row = &ip[i * k..(i + 1) * k];
            let mut best_idx = 0;
            let mut best_dist = f32::MAX;
            for j in 0..k {
                let dist = xn + cent_norms[j] - 2.0 * row[j];
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = j;
                }
            }
            best_idx
        })
        .collect()
}

/// Find nearest centroid from contiguous centroid data using batch4 SIMD.
#[inline]
fn nearest_centroid_contiguous(
    centroids: &[f32],
    k: usize,
    vec: &[f32],
    metric: Metric,
) -> usize {
    let dim = vec.len();
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;

    let chunks = k / 4;
    for chunk in 0..chunks {
        let base = chunk * 4;
        let dists = metric.distance_ord_batch4(
            vec,
            [
                &centroids[base * dim..(base + 1) * dim],
                &centroids[(base + 1) * dim..(base + 2) * dim],
                &centroids[(base + 2) * dim..(base + 3) * dim],
                &centroids[(base + 3) * dim..(base + 4) * dim],
            ],
        );
        for (j, &d) in dists.iter().enumerate() {
            if d < best_dist {
                best_dist = d;
                best_idx = base + j;
            }
        }
    }

    for i in (chunks * 4)..k {
        let d = metric.distance_ord(vec, &centroids[i * dim..(i + 1) * dim]);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }

    best_idx
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
        assert!(!idx.centroid_data.is_empty());
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

    // ── Additional coverage tests ─────────────────────────────────────────────

    #[test]
    fn search_dimension_mismatch() {
        let cfg = IvfConfig { n_lists: 2, nprobe: 2, train_size: 100, max_iter: 5 };
        let idx = IvfIndex::new(3, Metric::L2, cfg);
        let err = idx.search(&[1.0, 2.0], 1).unwrap_err();
        assert!(matches!(err, VectorDbError::DimensionMismatch { expected: 3, got: 2 }));
    }

    #[test]
    fn search_k_zero_returns_empty() {
        let cfg = IvfConfig { n_lists: 2, nprobe: 2, train_size: 100, max_iter: 5 };
        let mut idx = IvfIndex::new(2, Metric::L2, cfg);
        idx.add(1, &[1.0, 0.0]).unwrap();
        let r = idx.search(&[1.0, 0.0], 0).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn search_k_greater_than_len() {
        let cfg = IvfConfig { n_lists: 2, nprobe: 2, train_size: 100, max_iter: 5 };
        let mut idx = IvfIndex::new(2, Metric::L2, cfg);
        idx.add(1, &[1.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0]).unwrap();
        let r = idx.search(&[1.0, 0.0], 10).unwrap();
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn search_empty_index() {
        let cfg = IvfConfig { n_lists: 2, nprobe: 2, train_size: 100, max_iter: 5 };
        let idx = IvfIndex::new(2, Metric::L2, cfg);
        let r = idx.search(&[1.0, 0.0], 5).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn add_batch_raw_happy_path() {
        let cfg = IvfConfig { n_lists: 2, nprobe: 2, train_size: 100, max_iter: 5 };
        let mut idx = IvfIndex::new(2, Metric::L2, cfg);
        let data = vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0];
        idx.add_batch_raw(&data, 2, 10).unwrap();
        assert_eq!(idx.len(), 3);
        let r = idx.search(&[2.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 12);
    }

    #[test]
    fn add_batch_raw_dimension_mismatch() {
        let cfg = IvfConfig { n_lists: 2, nprobe: 2, train_size: 100, max_iter: 5 };
        let mut idx = IvfIndex::new(3, Metric::L2, cfg);
        let data = vec![1.0, 0.0, 0.0, 1.0];
        let err = idx.add_batch_raw(&data, 2, 0).unwrap_err();
        assert!(matches!(err, VectorDbError::DimensionMismatch { expected: 3, got: 2 }));
    }

    #[test]
    fn add_batch_raw_not_multiple_of_dim() {
        let cfg = IvfConfig { n_lists: 2, nprobe: 2, train_size: 100, max_iter: 5 };
        let mut idx = IvfIndex::new(2, Metric::L2, cfg);
        let data = vec![1.0, 0.0, 0.5]; // 3 elements, not multiple of 2
        let err = idx.add_batch_raw(&data, 2, 0).unwrap_err();
        assert!(matches!(err, VectorDbError::InvalidConfig(_)));
    }

    #[test]
    fn add_batch_raw_triggers_training() {
        let cfg = IvfConfig { n_lists: 2, nprobe: 2, train_size: 4, max_iter: 5 };
        let mut idx = IvfIndex::new(2, Metric::L2, cfg);
        let data: Vec<f32> = (0..8).map(|i| if i % 2 == 0 { (i / 2) as f32 } else { 0.0 }).collect();
        idx.add_batch_raw(&data, 2, 0).unwrap();
        assert!(idx.trained);
        assert_eq!(idx.len(), 4);
    }

    #[test]
    fn delete_nonexistent_from_trained_index() {
        let mut idx = make_trained(16);
        assert!(!idx.delete(999));
    }

    #[test]
    fn cosine_metric_search() {
        let cfg = IvfConfig { n_lists: 2, nprobe: 2, train_size: 4, max_iter: 10 };
        let mut idx = IvfIndex::new(2, Metric::Cosine, cfg);
        for i in 0..8u64 {
            idx.add(i, &[(i as f32 + 1.0), (i as f32 + 1.0)]).unwrap();
        }
        let r = idx.search(&[1.0, 1.0], 1).unwrap();
        // All vectors point in the same direction (x, x), so cosine distance should be ~0
        assert!(r[0].distance < 0.01);
    }

    #[test]
    fn dot_product_metric_search() {
        let cfg = IvfConfig { n_lists: 2, nprobe: 2, train_size: 4, max_iter: 10 };
        let mut idx = IvfIndex::new(2, Metric::DotProduct, cfg);
        for i in 0..8u64 {
            idx.add(i, &[i as f32, 0.0]).unwrap();
        }
        let r = idx.search(&[1.0, 0.0], 3).unwrap();
        assert_eq!(r.len(), 3);
        // Results should be ordered by distance (ascending)
        for w in r.windows(2) {
            assert!(w[0].distance <= w[1].distance + 1e-5);
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
        let cfg = IvfConfig { n_lists: 2, nprobe: 2, train_size: 100, max_iter: 5 };
        let mut idx = IvfIndex::new(2, Metric::L2, cfg);
        assert!(!idx.trained);
        idx.flush();
        // train() returns early when staging is empty, so trained stays false
        assert!(!idx.trained);
    }

    #[test]
    fn iter_vectors_untrained() {
        let cfg = IvfConfig { n_lists: 2, nprobe: 2, train_size: 100, max_iter: 5 };
        let mut idx = IvfIndex::new(2, Metric::L2, cfg);
        idx.add(1, &[1.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0]).unwrap();
        let mut ids: Vec<u64> = idx.iter_vectors().map(|(id, _)| id).collect();
        ids.sort();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn save_load_untrained() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ivf_untrained.bin");
        let cfg = IvfConfig { n_lists: 2, nprobe: 2, train_size: 100, max_iter: 5 };
        let mut idx = IvfIndex::new(2, Metric::L2, cfg);
        idx.add(1, &[1.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0]).unwrap();
        idx.save(&path).unwrap();
        let loaded = IvfIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert!(!loaded.trained);
        let r = loaded.search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 1);
    }

    #[test]
    fn load_nonexistent_file_errors() {
        let result = IvfIndex::load("/tmp/nonexistent_ivf_test_file.bin");
        assert!(matches!(result, Err(VectorDbError::Io(_))));
    }

    #[test]
    fn nprobe_clamped_to_centroids_len() {
        // nprobe > n_lists should be clamped, not panic
        let cfg = IvfConfig { n_lists: 2, nprobe: 100, train_size: 4, max_iter: 10 };
        let mut idx = IvfIndex::new(2, Metric::L2, cfg);
        for i in 0..8u64 {
            idx.add(i, &[i as f32, 0.0]).unwrap();
        }
        let r = idx.search(&[3.0, 0.0], 3).unwrap();
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].id, 3);
    }

    #[test]
    fn search_trained_with_staging_vectors() {
        // After training, add more vectors (below next train threshold) and search
        // should scan both posting lists AND staging
        let cfg = IvfConfig { n_lists: 2, nprobe: 2, train_size: 8, max_iter: 10 };
        let mut idx = IvfIndex::new(2, Metric::L2, cfg);
        for i in 0..8u64 {
            idx.add(i, &[i as f32, 0.0]).unwrap();
        }
        assert!(idx.trained);
        // Post-training insert goes to posting list, not staging
        idx.add(100, &[100.0, 0.0]).unwrap();
        let r = idx.search(&[100.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 100);
    }

    #[test]
    fn config_returns_correct_values() {
        let cfg = IvfConfig { n_lists: 2, nprobe: 2, train_size: 100, max_iter: 5 };
        let idx = IvfIndex::new(5, Metric::Cosine, cfg);
        assert_eq!(idx.config().dimensions, 5);
        assert_eq!(idx.config().metric, Metric::Cosine);
    }

    #[test]
    fn len_counts_staging_and_posting_lists() {
        let cfg = IvfConfig { n_lists: 2, nprobe: 2, train_size: 4, max_iter: 10 };
        let mut idx = IvfIndex::new(2, Metric::L2, cfg);
        assert_eq!(idx.len(), 0);
        idx.add(0, &[0.0, 0.0]).unwrap();
        assert_eq!(idx.len(), 1);
        idx.add(1, &[1.0, 0.0]).unwrap();
        assert_eq!(idx.len(), 2);
        idx.add(2, &[2.0, 0.0]).unwrap();
        assert_eq!(idx.len(), 3);
        // Adding 4th triggers training (train_size=4)
        idx.add(3, &[3.0, 0.0]).unwrap();
        assert!(idx.trained);
        assert_eq!(idx.len(), 4);
    }

    #[test]
    fn kmeans_with_fewer_vectors_than_lists() {
        // n_lists=4 but only 2 vectors — k should be clamped to 2
        let cfg = IvfConfig { n_lists: 4, nprobe: 4, train_size: 2, max_iter: 5 };
        let mut idx = IvfIndex::new(2, Metric::L2, cfg);
        idx.add(0, &[0.0, 0.0]).unwrap();
        idx.add(1, &[10.0, 0.0]).unwrap();
        assert!(idx.trained);
        assert!(idx.nlist <= 2);
        let r = idx.search(&[10.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 1);
    }

    #[test]
    fn delete_all_from_posting_list() {
        let mut idx = make_trained(16);
        for i in 0..16u64 {
            assert!(idx.delete(i));
        }
        assert_eq!(idx.len(), 0);
        let r = idx.search(&[5.0, 0.0], 5).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn add_batch_raw_empty_buffer() {
        let cfg = IvfConfig { n_lists: 2, nprobe: 2, train_size: 100, max_iter: 5 };
        let mut idx = IvfIndex::new(2, Metric::L2, cfg);
        idx.add_batch_raw(&[], 2, 0).unwrap();
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn add_batch_raw_post_training_uses_pipeline() {
        // First train the index, then use batch insert which should use the 3-phase pipeline
        let cfg = IvfConfig { n_lists: 2, nprobe: 2, train_size: 4, max_iter: 10 };
        let mut idx = IvfIndex::new(2, Metric::L2, cfg);
        // Train with 4 vectors
        for i in 0..4u64 {
            idx.add(i, &[i as f32 * 10.0, 0.0]).unwrap();
        }
        assert!(idx.trained);
        assert_eq!(idx.len(), 4);

        // Now batch insert post-training (should use 3-phase pipeline)
        let batch_data: Vec<f32> = (0..100)
            .flat_map(|i| vec![i as f32, (i as f32) * 0.5])
            .collect();
        idx.add_batch_raw(&batch_data, 2, 100).unwrap();
        assert_eq!(idx.len(), 104);

        // Verify search still works
        let r = idx.search(&[50.0, 25.0], 1).unwrap();
        assert_eq!(r[0].id, 150);
    }

    #[test]
    fn posting_list_swap_remove() {
        let mut pl = PostingList::new(2);
        pl.push(1, &[1.0, 2.0]);
        pl.push(2, &[3.0, 4.0]);
        pl.push(3, &[5.0, 6.0]);
        assert_eq!(pl.len(), 3);

        // Remove middle element
        pl.swap_remove(1);
        assert_eq!(pl.len(), 2);
        assert_eq!(pl.ids[0], 1);
        assert_eq!(pl.ids[1], 3); // swapped from last
        assert_eq!(pl.get_vector(0), &[1.0, 2.0]);
        assert_eq!(pl.get_vector(1), &[5.0, 6.0]);
    }

    #[test]
    fn posting_list_swap_remove_last() {
        let mut pl = PostingList::new(2);
        pl.push(1, &[1.0, 2.0]);
        pl.push(2, &[3.0, 4.0]);

        // Remove last element (no swap needed)
        pl.swap_remove(1);
        assert_eq!(pl.len(), 1);
        assert_eq!(pl.ids[0], 1);
        assert_eq!(pl.get_vector(0), &[1.0, 2.0]);
    }
}
