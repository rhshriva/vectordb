/// HNSW (Hierarchical Navigable Small World) approximate nearest-neighbour index.
///
/// Optimized incremental implementation based on Malkov & Yashunin 2016.
///
/// Performance optimizations over the previous HashMap-based version:
/// - **Flat Vec storage**: O(1) node access by index, no hash overhead
/// - **Contiguous vector buffer**: all vectors in one Vec<f32> for cache-friendly distance calc
/// - **Generation-counter visited**: O(1) reset between searches, no per-search allocation
/// - **u32 internal IDs**: 8-byte Candidate (was 16), denser neighbor lists
///
/// ## Key parameters
/// | Parameter         | Default | Effect |
/// |-------------------|---------|--------|
/// | `ef_construction` | 200     | Beam width during index build. Higher → better recall, slower build. |
/// | `ef_search`       | 50      | Beam width during query. Higher → better recall, slower query. |
/// | `m`               | 12      | Edges per node per layer. Higher → better recall, more memory. |
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::Mutex;

use rand::Rng;
use rayon::prelude::*;

use crate::{
    distance::Metric,
    error::VectorDbError,
    index::{IndexConfig, SearchResult, VectorIndex},
};

// ── Serialization snapshots (backward compatibility) ─────────────────────

/// Legacy format (pre-incremental HNSW, with flush_threshold + flat vectors).
#[derive(serde::Serialize, serde::Deserialize)]
struct LegacyHnswIndexSnapshot {
    dimensions: usize,
    metric: Metric,
    hnsw_config: HnswConfig,
    flush_threshold: usize,
    vectors: HashMap<u64, Vec<f32>>,
}

/// Node in the serializable graph snapshot.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct HnswNodeSnapshot {
    vector: Vec<f32>,
    neighbors: Vec<Vec<u64>>,
    level: usize,
}

/// Serializable graph snapshot (HashMap-based, serde-compatible).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct HnswGraphSnapshot {
    nodes: HashMap<u64, HnswNodeSnapshot>,
    entry_point: Option<u64>,
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    ml: f64,
}

/// Top-level index snapshot for save/load.
#[derive(serde::Serialize, serde::Deserialize)]
struct HnswIndexSnapshot {
    dimensions: usize,
    metric: Metric,
    hnsw_config: HnswConfig,
    graph: HnswGraphSnapshot,
}

/// Parameters for building and querying the HNSW graph.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HnswConfig {
    /// Beam width during graph construction. Typical range: 100–400.
    pub ef_construction: usize,
    /// Beam width during search. Can be changed after build. Typical range: 10–500.
    pub ef_search: usize,
    /// Number of bi-directional links per node. Typical range: 8–64.
    pub m: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            ef_construction: 200,
            ef_search: 50,
            m: 12,
        }
    }
}

// ── Instrumentation for profiling ───────────────────────────────────────────

/// Accumulated timing stats from instrumented insert.
#[derive(Debug, Default)]
pub struct InsertStats {
    pub random_level: std::time::Duration,
    pub clone_vec: std::time::Duration,
    pub node_insert: std::time::Duration,
    pub greedy_descent: std::time::Duration,
    pub search_layer: std::time::Duration,
    pub sl_distance: std::time::Duration,
    pub sl_hash_lookup: std::time::Duration,
    pub sl_heap_ops: std::time::Duration,
    pub set_neighbors: std::time::Duration,
    pub back_edges: std::time::Duration,
    pub pruning: std::time::Duration,
}

// ── Priority queue element ──────────────────────────────────────────────

/// Candidate for priority queue. 8 bytes (f32 distance + u32 id).
#[derive(Clone, Copy, PartialEq)]
struct Candidate {
    distance: f32,
    id: u32,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ── Visited tracker (generation-based, O(1) reset) ──────────────────────

/// Tracks which nodes have been visited during a search, plus reusable
/// scratch heaps to avoid per-search allocation.
struct VisitedTracker {
    marks: Vec<u32>,
    gen: u32,
    /// Reusable min-heap of candidates (cleared between searches).
    candidates: BinaryHeap<std::cmp::Reverse<Candidate>>,
    /// Reusable max-heap of results (cleared between searches).
    results: BinaryHeap<Candidate>,
}

impl VisitedTracker {
    fn new() -> Self {
        Self {
            marks: Vec::new(),
            gen: 1,
            candidates: BinaryHeap::new(),
            results: BinaryHeap::new(),
        }
    }

    fn ensure_capacity(&mut self, n: usize) {
        if self.marks.len() < n {
            self.marks.resize(n, 0);
        }
    }

    /// Start a new visit epoch. O(1) — just increments a counter.
    #[inline]
    fn reset(&mut self) {
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            // Overflow (every ~4 billion searches): clear everything
            self.marks.fill(0);
            self.gen = 1;
        }
    }

    /// Mark id as visited. Returns true if newly visited this epoch.
    #[inline]
    fn visit(&mut self, id: u32) -> bool {
        let idx = id as usize;
        if self.marks[idx] == self.gen {
            false
        } else {
            self.marks[idx] = self.gen;
            true
        }
    }
}

// ── HNSW Graph (flat Vec storage) ───────────────────────────────────────

/// The HNSW graph with flat Vec-based storage.
///
/// All node data is stored in parallel Vecs indexed by node ID (u32).
/// Vectors are stored contiguously in a single buffer for cache locality.
#[derive(Clone, Debug)]
struct HnswGraph {
    /// Flat vector storage: vector for node i at `[i*dim .. (i+1)*dim]`.
    vectors: Vec<f32>,
    /// Level per node (max 255 layers).
    levels: Vec<u8>,

    // ── Layer-0 neighbors: flat contiguous buffer (cache-friendly) ──
    // Node i's layer-0 neighbors at: layer0[i * m_max0 .. i * m_max0 + layer0_counts[i]]
    // This eliminates 2 levels of pointer indirection vs Vec<Vec<Vec<u32>>>.
    layer0: Vec<u32>,
    layer0_counts: Vec<u16>,

    // ── Upper-layer neighbors (level ≥ 1): dynamic, sparse ──
    // upper_neighbors[node_id][layer_idx] where layer_idx = layer - 1
    // Only nodes with level > 0 have non-empty entries. Rarely accessed.
    upper_neighbors: Vec<Vec<Vec<u32>>>,

    /// Whether each slot is occupied (alive vs deleted).
    alive: Vec<bool>,
    /// Number of alive nodes.
    count: usize,

    /// Entry point node ID.
    entry_point: Option<u32>,
    /// Vector dimensionality.
    dim: usize,
    /// Max connections per upper layer (M).
    m: usize,
    /// Max connections at layer 0 (2*M).
    m_max0: usize,
    /// Beam width during graph construction.
    ef_construction: usize,
    /// Level normalization: 1/ln(M).
    ml: f64,
}

/// Search results for one vector, computed during parallel search phase.
/// Stores pre-computed nearest neighbors at each layer so linking can happen sequentially.
struct ParallelSearchResult {
    id: u32,
    level: usize,
    /// Search results per layer, index 0 = layer 0, index L = layer L.
    /// Each entry is a sorted Vec of Candidates from search_layer.
    layer_results: Vec<Vec<Candidate>>,
}

impl HnswGraph {
    fn new(dim: usize, m: usize, ef_construction: usize) -> Self {
        Self {
            vectors: Vec::new(),
            levels: Vec::new(),
            layer0: Vec::new(),
            layer0_counts: Vec::new(),
            upper_neighbors: Vec::new(),
            alive: Vec::new(),
            count: 0,
            entry_point: None,
            dim,
            m,
            m_max0: 2 * m,
            ef_construction,
            ml: 1.0 / (m.max(2) as f64).ln(),
        }
    }

    /// Stride per node in layer0 buffer: m_max0 + 1 to allow temporary overflow
    /// during pruning (push one over capacity, then swap-remove the worst).
    #[inline]
    fn layer0_stride(&self) -> usize {
        self.m_max0 + 1
    }

    /// Ensure storage capacity for node `id`.
    fn ensure_capacity(&mut self, id: u32) {
        let needed = id as usize + 1;
        if needed > self.alive.len() {
            let stride = self.layer0_stride();
            self.vectors.resize(needed * self.dim, 0.0);
            self.levels.resize(needed, 0);
            self.layer0.resize(needed * stride, 0);
            self.layer0_counts.resize(needed, 0);
            self.upper_neighbors.resize_with(needed, Vec::new);
            self.alive.resize(needed, false);
        }
    }

    /// Pre-allocate for `additional` more nodes.
    fn reserve(&mut self, additional: usize) {
        let stride = self.layer0_stride();
        self.vectors.reserve(additional * self.dim);
        self.levels.reserve(additional);
        self.layer0.reserve(additional * stride);
        self.layer0_counts.reserve(additional);
        self.upper_neighbors.reserve(additional);
        self.alive.reserve(additional);
    }

    /// Get the vector slice for a node. O(1), no hash.
    #[inline]
    fn vector(&self, id: u32) -> &[f32] {
        let start = id as usize * self.dim;
        unsafe { self.vectors.get_unchecked(start..start + self.dim) }
    }

    /// Write a vector into the flat buffer.
    #[inline]
    fn set_vector(&mut self, id: u32, vec: &[f32]) {
        let start = id as usize * self.dim;
        self.vectors[start..start + self.dim].copy_from_slice(vec);
    }

    #[inline]
    fn is_alive(&self, id: u32) -> bool {
        let idx = id as usize;
        idx < self.alive.len() && unsafe { *self.alive.get_unchecked(idx) }
    }

    #[inline]
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen::<f64>().max(1e-15);
        (-r.ln() * self.ml).floor() as usize
    }

    #[inline]
    fn max_level(&self) -> usize {
        self.entry_point
            .map(|ep| self.levels[ep as usize] as usize)
            .unwrap_or(0)
    }

    #[inline]
    fn max_neighbors(&self, layer: usize) -> usize {
        if layer == 0 { self.m_max0 } else { self.m }
    }

    /// Get neighbor IDs for a node at a given layer.
    /// Layer 0 reads from flat contiguous buffer; upper layers from dynamic Vecs.
    #[inline]
    fn get_neighbors(&self, node_id: u32, layer: usize) -> &[u32] {
        if layer == 0 {
            let stride = self.layer0_stride();
            let start = node_id as usize * stride;
            let count = self.layer0_counts[node_id as usize] as usize;
            &self.layer0[start..start + count]
        } else {
            let upper = &self.upper_neighbors[node_id as usize];
            let layer_idx = layer - 1;
            if layer_idx < upper.len() {
                &upper[layer_idx]
            } else {
                &[]
            }
        }
    }

    /// Beam search within a single layer. Returns up to `ef` nearest candidates.
    fn search_layer(
        &self,
        query: &[f32],
        entry_ids: &[u32],
        ef: usize,
        layer: usize,
        metric: Metric,
        vt: &mut VisitedTracker,
    ) -> Vec<Candidate> {
        vt.reset();
        // Reuse scratch heaps (clear retains allocated capacity)
        vt.candidates.clear();
        vt.results.clear();
        let mut farthest_dist = f32::MAX;

        for &ep_id in entry_ids {
            if !vt.visit(ep_id) || !self.alive[ep_id as usize] { continue; }
            let d = metric.distance_ord(query, self.vector(ep_id));
            let c = Candidate { distance: d, id: ep_id };
            vt.candidates.push(std::cmp::Reverse(c));
            vt.results.push(c);
            farthest_dist = d; // single entry point typical
        }

        while let Some(std::cmp::Reverse(closest)) = vt.candidates.pop() {
            if closest.distance > farthest_dist && vt.results.len() >= ef {
                break;
            }

            let nbs = self.get_neighbors(closest.id, layer);

            // Collect valid (newly visited + alive) neighbors into a stack buffer
            let mut valid = [0u32; 64];
            let mut valid_count = 0usize;
            for &nb_id in nbs {
                if vt.visit(nb_id) && self.alive[nb_id as usize] {
                    valid[valid_count] = nb_id;
                    valid_count += 1;
                }
            }

            // Process in batches of 4 for instruction-level parallelism
            let mut i = 0;
            while i + 4 <= valid_count {
                let ids = [valid[i], valid[i + 1], valid[i + 2], valid[i + 3]];
                let dists = metric.distance_ord_batch4(query, [
                    self.vector(ids[0]), self.vector(ids[1]),
                    self.vector(ids[2]), self.vector(ids[3]),
                ]);
                for j in 0..4 {
                    if vt.results.len() < ef || dists[j] < farthest_dist {
                        let c = Candidate { distance: dists[j], id: ids[j] };
                        vt.candidates.push(std::cmp::Reverse(c));
                        vt.results.push(c);
                        if vt.results.len() > ef {
                            vt.results.pop();
                        }
                        if let Some(f) = vt.results.peek() {
                            farthest_dist = f.distance;
                        }
                    }
                }
                i += 4;
            }
            // Remaining 0-3 neighbors
            while i < valid_count {
                let nb_id = valid[i];
                let d = metric.distance_ord(query, self.vector(nb_id));
                if vt.results.len() < ef || d < farthest_dist {
                    let c = Candidate { distance: d, id: nb_id };
                    vt.candidates.push(std::cmp::Reverse(c));
                    vt.results.push(c);
                    if vt.results.len() > ef {
                        vt.results.pop();
                    }
                    if let Some(f) = vt.results.peek() {
                        farthest_dist = f.distance;
                    }
                }
                i += 1;
            }
        }

        // Drain results into a sorted Vec (drain retains heap capacity for reuse)
        let mut result_vec: Vec<Candidate> = vt.results.drain().collect();
        result_vec.sort_unstable();
        result_vec
    }

    /// Greedy search for the single closest node at a given layer.
    #[inline]
    fn greedy_closest(
        &self,
        query: &[f32],
        mut current: u32,
        layer: usize,
        metric: Metric,
    ) -> u32 {
        let mut current_dist = metric.distance_ord(query, self.vector(current));

        loop {
            let mut changed = false;
            let nbs = self.get_neighbors(current, layer);
            for &nb_id in nbs {
                let d = metric.distance_ord(query, self.vector(nb_id));
                if d < current_dist {
                    current = nb_id;
                    current_dist = d;
                    changed = true;
                }
            }
            if !changed { break; }
        }
        current
    }

    // ── Layer-0 flat buffer helpers ─────────────────────────────────────

    /// Set layer-0 neighbors for a node (overwrites existing).
    #[inline]
    fn set_layer0_neighbors(&mut self, node_id: u32, neighbors: &[u32]) {
        let stride = self.layer0_stride();
        let start = node_id as usize * stride;
        let count = neighbors.len().min(self.m_max0);
        self.layer0[start..start + count].copy_from_slice(&neighbors[..count]);
        self.layer0_counts[node_id as usize] = count as u16;
    }

    /// Push a neighbor onto node's layer-0 list. Returns new count.
    /// Buffer has m_max0+1 slots to allow temporary overflow before pruning.
    #[inline]
    fn push_layer0_neighbor(&mut self, node_id: u32, neighbor: u32) -> usize {
        let stride = self.layer0_stride();
        let start = node_id as usize * stride;
        let count = self.layer0_counts[node_id as usize] as usize;
        // stride = m_max0 + 1, so we can always push one over m_max0
        self.layer0[start + count] = neighbor;
        self.layer0_counts[node_id as usize] = (count + 1) as u16;
        count + 1
    }

    /// Remove neighbor at index from node's layer-0 list (swap-remove).
    #[inline]
    fn swap_remove_layer0(&mut self, node_id: u32, idx: usize) {
        let stride = self.layer0_stride();
        let start = node_id as usize * stride;
        let count = self.layer0_counts[node_id as usize] as usize;
        if idx < count {
            self.layer0[start + idx] = self.layer0[start + count - 1];
            self.layer0_counts[node_id as usize] = (count - 1) as u16;
        }
    }

    // ── Insert ──────────────────────────────────────────────────────────

    /// Insert a vector into the graph incrementally.
    fn insert(&mut self, id: u32, vector: &[f32], metric: Metric, vt: &mut VisitedTracker) {
        let new_level = self.random_level();
        let dim = self.dim;

        // Allocate slot and store data
        self.ensure_capacity(id);
        vt.ensure_capacity(id as usize + 1);
        self.set_vector(id, vector);
        self.levels[id as usize] = new_level as u8;
        // Layer-0: already zeroed by ensure_capacity
        self.layer0_counts[id as usize] = 0;
        // Upper layers: pre-allocate if node has level > 0
        if new_level > 0 {
            let mut upper_vecs = Vec::with_capacity(new_level);
            for _ in 0..new_level {
                upper_vecs.push(Vec::with_capacity(self.m));
            }
            self.upper_neighbors[id as usize] = upper_vecs;
        } else {
            self.upper_neighbors[id as usize] = Vec::new();
        }
        self.alive[id as usize] = true;
        self.count += 1;

        // First node → set as entry point
        let ep_id = match self.entry_point {
            None => {
                self.entry_point = Some(id);
                return;
            }
            Some(ep) => ep,
        };

        let current_max_level = self.max_level();

        // Phase 1: Greedy descent from top layers
        let mut current_ep = ep_id;
        if current_max_level > new_level {
            for layer in ((new_level + 1)..=current_max_level).rev() {
                current_ep = self.greedy_closest(vector, current_ep, layer, metric);
            }
        }

        // Phase 2: Insert at layers min(new_level, current_max_level) down to 0
        let insert_top = new_level.min(current_max_level);
        for layer in (0..=insert_top).rev() {
            let results = self.search_layer(
                vector, &[current_ep], self.ef_construction, layer, metric, vt,
            );

            let max_conn = self.max_neighbors(layer);
            let m_to_select = max_conn.min(results.len());

            // Set forward edges for this node (stack array avoids heap allocation)
            if layer == 0 {
                let mut ids = [0u32; 64]; // m_max0 is typically 24
                for (i, c) in results[..m_to_select].iter().enumerate() {
                    ids[i] = c.id;
                }
                self.set_layer0_neighbors(id, &ids[..m_to_select]);
            } else {
                let layer_idx = layer - 1;
                let upper = &mut self.upper_neighbors[id as usize];
                while upper.len() <= layer_idx {
                    upper.push(Vec::new());
                }
                upper[layer_idx].clear();
                upper[layer_idx].extend(results[..m_to_select].iter().map(|c| c.id));
            }

            // Add back-edges — iterate results slice directly
            for i in 0..m_to_select {
                let nb_id = results[i].id;
                if nb_id == id { continue; }

                // Add back-edge + prune if over capacity
                if layer == 0 {
                    let new_count = self.push_layer0_neighbor(nb_id, id);
                    if new_count > max_conn {
                        // Find worst neighbor and swap-remove
                        let nb_start = nb_id as usize * dim;
                        let l0_stride = self.layer0_stride();
                        let l0_start = nb_id as usize * l0_stride;
                        let l0_count = self.layer0_counts[nb_id as usize] as usize;
                        let mut worst_idx = 0;
                        let mut worst_dist = f32::NEG_INFINITY;
                        for j in 0..l0_count {
                            let nid = self.layer0[l0_start + j];
                            if !self.alive[nid as usize] {
                                worst_idx = j;
                                break;
                            }
                            let nid_start = nid as usize * dim;
                            let d = metric.distance_ord(
                                &self.vectors[nb_start..nb_start + dim],
                                &self.vectors[nid_start..nid_start + dim],
                            );
                            if d > worst_dist {
                                worst_dist = d;
                                worst_idx = j;
                            }
                        }
                        self.swap_remove_layer0(nb_id, worst_idx);
                    }
                } else {
                    let layer_idx = layer - 1;
                    let nb_upper = &mut self.upper_neighbors[nb_id as usize];
                    while nb_upper.len() <= layer_idx {
                        nb_upper.push(Vec::new());
                    }
                    nb_upper[layer_idx].push(id);

                    // Prune if over capacity
                    if self.upper_neighbors[nb_id as usize][layer_idx].len() > max_conn {
                        let nb_start = nb_id as usize * dim;
                        let nbs_layer = &self.upper_neighbors[nb_id as usize][layer_idx];
                        let mut worst_idx = 0;
                        let mut worst_dist = f32::NEG_INFINITY;
                        for (j, &nid) in nbs_layer.iter().enumerate() {
                            if !self.alive[nid as usize] {
                                worst_idx = j;
                                break;
                            }
                            let nid_start = nid as usize * dim;
                            let d = metric.distance_ord(
                                &self.vectors[nb_start..nb_start + dim],
                                &self.vectors[nid_start..nid_start + dim],
                            );
                            if d > worst_dist {
                                worst_dist = d;
                                worst_idx = j;
                            }
                        }
                        self.upper_neighbors[nb_id as usize][layer_idx].swap_remove(worst_idx);
                    }
                }
            }

            if let Some(c) = results.first() {
                current_ep = c.id;
            }
        }

        // Update entry point if new node has a higher level
        if new_level > current_max_level {
            self.entry_point = Some(id);
        }
    }

    /// Search for k nearest neighbors.
    fn search_knn(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        metric: Metric,
        vt: &mut VisitedTracker,
    ) -> Vec<(u32, f32)> {
        let ep_id = match self.entry_point {
            None => return Vec::new(),
            Some(ep) => ep,
        };

        let top = self.max_level();
        let ef = ef_search.max(k);

        let mut current_ep = ep_id;
        if top > 0 {
            for layer in (1..=top).rev() {
                current_ep = self.greedy_closest(query, current_ep, layer, metric);
            }
        }

        let results = self.search_layer(query, &[current_ep], ef, 0, metric, vt);
        results.into_iter()
            .take(k)
            .map(|c| (c.id, metric.fixup_distance(c.distance)))
            .collect()
    }

    /// Delete a node from the graph.
    fn delete(&mut self, id: u32) -> bool {
        if !self.is_alive(id) { return false; }

        self.alive[id as usize] = false;
        self.count -= 1;

        let level = self.levels[id as usize] as usize;

        // Remove back-edges from layer-0 neighbors
        {
            let stride = self.layer0_stride();
            let start = id as usize * stride;
            let count = self.layer0_counts[id as usize] as usize;
            let l0_nbs: Vec<u32> = self.layer0[start..start + count].to_vec();
            for nb_id in l0_nbs {
                // Remove `id` from nb_id's layer-0 neighbor list
                let nb_start = nb_id as usize * stride;
                let nb_count = self.layer0_counts[nb_id as usize] as usize;
                for j in 0..nb_count {
                    if self.layer0[nb_start + j] == id {
                        // swap-remove
                        self.layer0[nb_start + j] = self.layer0[nb_start + nb_count - 1];
                        self.layer0_counts[nb_id as usize] = (nb_count - 1) as u16;
                        break;
                    }
                }
            }
            self.layer0_counts[id as usize] = 0;
        }

        // Remove back-edges from upper-layer neighbors
        for layer in 1..=level {
            let layer_idx = layer - 1;
            if layer_idx < self.upper_neighbors[id as usize].len() {
                let nbs: Vec<u32> = self.upper_neighbors[id as usize][layer_idx].clone();
                for nb_id in nbs {
                    let nb_idx = nb_id as usize;
                    if nb_idx < self.upper_neighbors.len() && layer_idx < self.upper_neighbors[nb_idx].len() {
                        self.upper_neighbors[nb_idx][layer_idx].retain(|&nid| nid != id);
                    }
                }
            }
        }
        self.upper_neighbors[id as usize].clear();

        // Update entry point if deleted node was the entry point
        if self.entry_point == Some(id) {
            self.entry_point = self.alive.iter()
                .enumerate()
                .filter(|(_, &a)| a)
                .max_by_key(|(i, _)| self.levels[*i])
                .map(|(i, _)| i as u32);
        }

        true
    }

    /// Convert to serializable snapshot (for save/load).
    fn to_snapshot(&self) -> HnswGraphSnapshot {
        let mut nodes = HashMap::new();
        for (i, &a) in self.alive.iter().enumerate() {
            if a {
                let level = self.levels[i] as usize;
                // Reconstruct neighbors list: layer 0 from flat buffer, upper from dynamic
                let mut all_neighbors = Vec::with_capacity(level + 1);

                // Layer 0
                let stride = self.layer0_stride();
                let l0_start = i * stride;
                let l0_count = self.layer0_counts[i] as usize;
                all_neighbors.push(
                    self.layer0[l0_start..l0_start + l0_count]
                        .iter().map(|&n| n as u64).collect()
                );

                // Upper layers
                for layer_idx in 0..level {
                    if layer_idx < self.upper_neighbors[i].len() {
                        all_neighbors.push(
                            self.upper_neighbors[i][layer_idx]
                                .iter().map(|&n| n as u64).collect()
                        );
                    } else {
                        all_neighbors.push(Vec::new());
                    }
                }

                nodes.insert(i as u64, HnswNodeSnapshot {
                    vector: self.vector(i as u32).to_vec(),
                    neighbors: all_neighbors,
                    level,
                });
            }
        }
        HnswGraphSnapshot {
            nodes,
            entry_point: self.entry_point.map(|e| e as u64),
            m: self.m,
            m_max0: self.m_max0,
            ef_construction: self.ef_construction,
            ml: self.ml,
        }
    }

    /// Restore from serializable snapshot.
    fn from_snapshot(snap: HnswGraphSnapshot, dim: usize) -> Self {
        if snap.nodes.is_empty() {
            return Self {
                m: snap.m,
                m_max0: snap.m_max0,
                ef_construction: snap.ef_construction,
                ml: snap.ml,
                ..Self::new(dim, snap.m, snap.ef_construction)
            };
        }

        let max_id = snap.nodes.keys().copied().max().unwrap_or(0) as usize;
        let capacity = max_id + 1;
        let m_max0 = snap.m_max0;

        let mut graph = Self::new(dim, snap.m, snap.ef_construction);
        graph.m_max0 = m_max0;
        graph.ml = snap.ml;
        graph.entry_point = snap.entry_point.map(|e| e as u32);

        let stride = graph.layer0_stride();
        graph.vectors.resize(capacity * dim, 0.0);
        graph.levels.resize(capacity, 0);
        graph.layer0.resize(capacity * stride, 0);
        graph.layer0_counts.resize(capacity, 0);
        graph.upper_neighbors.resize_with(capacity, Vec::new);
        graph.alive.resize(capacity, false);

        for (id, node) in snap.nodes {
            let idx = id as usize;
            graph.set_vector(id as u32, &node.vector);
            graph.levels[idx] = node.level as u8;

            // Layer 0 → flat buffer
            if !node.neighbors.is_empty() {
                let l0_nbs: Vec<u32> = node.neighbors[0].iter().map(|&n| n as u32).collect();
                let count = l0_nbs.len().min(m_max0);
                let start = idx * stride;
                graph.layer0[start..start + count].copy_from_slice(&l0_nbs[..count]);
                graph.layer0_counts[idx] = count as u16;
            }

            // Upper layers → dynamic vecs
            if node.neighbors.len() > 1 {
                graph.upper_neighbors[idx] = node.neighbors[1..].iter()
                    .map(|layer| layer.iter().map(|&n| n as u32).collect())
                    .collect();
            }

            graph.alive[idx] = true;
            graph.count += 1;
        }

        graph
    }

    /// Instrumented search_layer for profiling.
    fn search_layer_instrumented(
        &self,
        query: &[f32],
        entry_ids: &[u32],
        ef: usize,
        layer: usize,
        metric: Metric,
        vt: &mut VisitedTracker,
        stats: &mut InsertStats,
    ) -> Vec<Candidate> {
        vt.reset();
        vt.candidates.clear();
        vt.results.clear();
        let mut farthest_dist = f32::MAX;

        for &ep_id in entry_ids {
            if !vt.visit(ep_id) { continue; }
            let t = std::time::Instant::now();
            let vec = self.vector(ep_id);
            stats.sl_hash_lookup += t.elapsed();
            let t = std::time::Instant::now();
            let d = metric.distance_ord(query, vec);
            stats.sl_distance += t.elapsed();
            let c = Candidate { distance: d, id: ep_id };
            let t = std::time::Instant::now();
            vt.candidates.push(std::cmp::Reverse(c));
            vt.results.push(c);
            farthest_dist = d;
            stats.sl_heap_ops += t.elapsed();
        }

        while let Some(std::cmp::Reverse(closest)) = vt.candidates.pop() {
            if closest.distance > farthest_dist && vt.results.len() >= ef {
                break;
            }

            let nbs = self.get_neighbors(closest.id, layer);
            for &nb_id in nbs {
                if !vt.visit(nb_id) { continue; }

                let t = std::time::Instant::now();
                let vec = self.vector(nb_id);
                stats.sl_hash_lookup += t.elapsed();
                let t = std::time::Instant::now();
                let d = metric.distance_ord(query, vec);
                stats.sl_distance += t.elapsed();

                if vt.results.len() < ef || d < farthest_dist {
                    let c = Candidate { distance: d, id: nb_id };
                    let t = std::time::Instant::now();
                    vt.candidates.push(std::cmp::Reverse(c));
                    vt.results.push(c);
                    if vt.results.len() > ef {
                        vt.results.pop();
                    }
                    if let Some(f) = vt.results.peek() {
                        farthest_dist = f.distance;
                    }
                    stats.sl_heap_ops += t.elapsed();
                }
            }
        }

        let mut result_vec: Vec<Candidate> = vt.results.drain().collect();
        result_vec.sort_unstable();
        result_vec
    }

    /// Instrumented insert for profiling.
    fn insert_instrumented(
        &mut self, id: u32, vector: &[f32], metric: Metric,
        vt: &mut VisitedTracker, stats: &mut InsertStats,
    ) {
        let t = std::time::Instant::now();
        let new_level = self.random_level();
        stats.random_level += t.elapsed();
        let dim = self.dim;

        let t = std::time::Instant::now();
        self.ensure_capacity(id);
        vt.ensure_capacity(id as usize + 1);
        self.set_vector(id, vector);
        self.levels[id as usize] = new_level as u8;
        self.layer0_counts[id as usize] = 0;
        if new_level > 0 {
            let mut upper_vecs = Vec::with_capacity(new_level);
            for _ in 0..new_level {
                upper_vecs.push(Vec::with_capacity(self.m));
            }
            self.upper_neighbors[id as usize] = upper_vecs;
        } else {
            self.upper_neighbors[id as usize] = Vec::new();
        }
        self.alive[id as usize] = true;
        self.count += 1;
        stats.node_insert += t.elapsed();

        let ep_id = match self.entry_point {
            None => {
                self.entry_point = Some(id);
                return;
            }
            Some(ep) => ep,
        };

        let current_max_level = self.max_level();

        let mut current_ep = ep_id;
        if current_max_level > new_level {
            let t = std::time::Instant::now();
            for layer in ((new_level + 1)..=current_max_level).rev() {
                current_ep = self.greedy_closest(vector, current_ep, layer, metric);
            }
            stats.greedy_descent += t.elapsed();
        }

        let insert_top = new_level.min(current_max_level);
        for layer in (0..=insert_top).rev() {
            let t = std::time::Instant::now();
            let results = self.search_layer_instrumented(
                vector, &[current_ep], self.ef_construction, layer, metric, vt, stats,
            );
            stats.search_layer += t.elapsed();

            let t = std::time::Instant::now();
            let max_conn = self.max_neighbors(layer);
            let m_to_select = max_conn.min(results.len());

            // Set forward edges (stack array avoids heap allocation)
            if layer == 0 {
                let mut ids = [0u32; 64];
                for (i, c) in results[..m_to_select].iter().enumerate() {
                    ids[i] = c.id;
                }
                self.set_layer0_neighbors(id, &ids[..m_to_select]);
            } else {
                let layer_idx = layer - 1;
                let upper = &mut self.upper_neighbors[id as usize];
                while upper.len() <= layer_idx {
                    upper.push(Vec::new());
                }
                upper[layer_idx].clear();
                upper[layer_idx].extend(results[..m_to_select].iter().map(|c| c.id));
            }
            stats.set_neighbors += t.elapsed();

            for i in 0..m_to_select {
                let nb_id = results[i].id;
                if nb_id == id { continue; }

                if layer == 0 {
                    let t = std::time::Instant::now();
                    let new_count = self.push_layer0_neighbor(nb_id, id);
                    stats.back_edges += t.elapsed();

                    if new_count > max_conn {
                        let t = std::time::Instant::now();
                        let nb_start = nb_id as usize * dim;
                        let l0_start = nb_id as usize * self.m_max0;
                        let l0_count = self.layer0_counts[nb_id as usize] as usize;
                        let mut worst_idx = 0;
                        let mut worst_dist = f32::NEG_INFINITY;
                        for j in 0..l0_count {
                            let nid = self.layer0[l0_start + j];
                            if !self.alive[nid as usize] {
                                worst_idx = j;
                                break;
                            }
                            let nid_start = nid as usize * dim;
                            let d = metric.distance_ord(
                                &self.vectors[nb_start..nb_start + dim],
                                &self.vectors[nid_start..nid_start + dim],
                            );
                            if d > worst_dist {
                                worst_dist = d;
                                worst_idx = j;
                            }
                        }
                        self.swap_remove_layer0(nb_id, worst_idx);
                        stats.pruning += t.elapsed();
                    }
                } else {
                    let t = std::time::Instant::now();
                    let layer_idx = layer - 1;
                    let nb_upper = &mut self.upper_neighbors[nb_id as usize];
                    while nb_upper.len() <= layer_idx {
                        nb_upper.push(Vec::new());
                    }
                    nb_upper[layer_idx].push(id);
                    stats.back_edges += t.elapsed();

                    if self.upper_neighbors[nb_id as usize][layer_idx].len() > max_conn {
                        let t = std::time::Instant::now();
                        let nb_start = nb_id as usize * dim;
                        let nbs_layer = &self.upper_neighbors[nb_id as usize][layer_idx];
                        let mut worst_idx = 0;
                        let mut worst_dist = f32::NEG_INFINITY;
                        for (j, &nid) in nbs_layer.iter().enumerate() {
                            if !self.alive[nid as usize] {
                                worst_idx = j;
                                break;
                            }
                            let nid_start = nid as usize * dim;
                            let d = metric.distance_ord(
                                &self.vectors[nb_start..nb_start + dim],
                                &self.vectors[nid_start..nid_start + dim],
                            );
                            if d > worst_dist {
                                worst_dist = d;
                                worst_idx = j;
                            }
                        }
                        self.upper_neighbors[nb_id as usize][layer_idx].swap_remove(worst_idx);
                        stats.pruning += t.elapsed();
                    }
                }
            }

            if let Some(c) = results.first() {
                current_ep = c.id;
            }
        }

        if new_level > current_max_level {
            self.entry_point = Some(id);
        }
    }

    // ── Parallel micro-batch insert ────────────────────────────────────────

    /// Insert a batch of vectors using micro-batching for parallelism.
    ///
    /// Strategy:
    /// 1. **Bootstrap**: Insert the first `micro_batch_size` vectors sequentially to build
    ///    a well-connected graph for subsequent parallel searches.
    /// 2. **Parallel micro-batches**: For remaining vectors, process in micro-batches:
    ///    a. **Store** vector data (sequential, fast memcpy)
    ///    b. **Search** for neighbors in parallel (rayon, read-only graph access)
    ///    c. **Link** edges sequentially (forward + back-edges + pruning)
    ///
    /// This gives ~N× speedup on N cores for the search phase (which is 90%+ of insert time).
    /// No unsafe code or per-node locks — just phase-separated parallelism.
    fn insert_batch_parallel(
        &mut self,
        ids: &[u32],
        vectors: &[f32], // flat buffer, row-major (N × dim)
        metric: Metric,
        micro_batch_size: usize,
        num_threads: usize,
    ) {
        let n = ids.len();
        if n == 0 { return; }
        let dim = self.dim;

        // Pre-allocate capacity for all nodes upfront (avoids Vec reallocation during parallel access)
        if let Some(&max_id) = ids.iter().max() {
            self.ensure_capacity(max_id);
        }
        self.reserve(n);

        // ── Bootstrap phase: insert first micro_batch_size vectors sequentially ──
        // This builds a well-connected graph so parallel search has real structure to traverse.
        let bootstrap_count = micro_batch_size.min(n);
        let mut vt = VisitedTracker::new();
        vt.ensure_capacity(self.alive.len());
        for i in 0..bootstrap_count {
            let id = ids[i];
            let vec_data = &vectors[i * dim..(i + 1) * dim];
            self.insert(id, vec_data, metric, &mut vt);
        }

        // If all vectors fit in the bootstrap, we're done
        if bootstrap_count >= n {
            return;
        }

        // ── Parallel phase: process remaining vectors in micro-batches ──
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());

        for chunk_start in (bootstrap_count..n).step_by(micro_batch_size) {
            let chunk_end = (chunk_start + micro_batch_size).min(n);
            let chunk_ids = &ids[chunk_start..chunk_end];

            // ── Phase A: Store vector data (sequential) ──
            // Write vectors into flat buffer, set levels, mark alive.
            // Neighbor counts stay at 0 — linking happens in Phase C.
            let chunk_levels: Vec<usize> = (0..chunk_ids.len()).map(|_| self.random_level()).collect();
            for (i, &id) in chunk_ids.iter().enumerate() {
                let global_i = chunk_start + i;
                let vec_data = &vectors[global_i * dim..(global_i + 1) * dim];
                self.set_vector(id, vec_data);
                self.levels[id as usize] = chunk_levels[i] as u8;
                self.layer0_counts[id as usize] = 0;
                if chunk_levels[i] > 0 {
                    let mut upper_vecs = Vec::with_capacity(chunk_levels[i]);
                    for _ in 0..chunk_levels[i] {
                        upper_vecs.push(Vec::with_capacity(self.m));
                    }
                    self.upper_neighbors[id as usize] = upper_vecs;
                } else {
                    self.upper_neighbors[id as usize] = Vec::new();
                }
                self.alive[id as usize] = true;
                self.count += 1;
            }

            // ── Phase B: Parallel search (read-only graph access) ──
            // Reborrow graph as &self for concurrent search.
            // Each rayon thread gets its own VisitedTracker.
            let graph: &HnswGraph = &*self;
            let capacity = graph.alive.len();

            let search_results: Vec<ParallelSearchResult> = pool.install(|| {
                chunk_ids.par_iter().enumerate().map(|(i, &id)| {
                    let level = chunk_levels[i];

                    // Per-thread visited tracker (no shared state)
                    let mut vt = VisitedTracker::new();
                    vt.ensure_capacity(capacity);

                    let ep_id = match graph.entry_point {
                        None => {
                            return ParallelSearchResult {
                                id,
                                level,
                                layer_results: Vec::new(),
                            };
                        }
                        Some(ep) => ep,
                    };

                    let vec_data = graph.vector(id);
                    let current_max_level = graph.max_level();

                    // Greedy descent from top layers
                    let mut current_ep = ep_id;
                    if current_max_level > level {
                        for layer in ((level + 1)..=current_max_level).rev() {
                            current_ep = graph.greedy_closest(vec_data, current_ep, layer, metric);
                        }
                    }

                    // Search at each layer from insert_top down to 0
                    let insert_top = level.min(current_max_level);
                    let mut layer_results = vec![Vec::new(); insert_top + 1];
                    for layer in (0..=insert_top).rev() {
                        let results = graph.search_layer(
                            vec_data, &[current_ep], graph.ef_construction, layer, metric, &mut vt,
                        );
                        if let Some(c) = results.first() {
                            current_ep = c.id;
                        }
                        layer_results[layer] = results;
                    }

                    ParallelSearchResult { id, level, layer_results }
                }).collect()
            });

            // ── Phase C: Sequential linking (mutable graph access) ──
            // Apply forward + back-edges + pruning for all vectors in this micro-batch.
            for result in search_results {
                let id = result.id;
                let new_level = result.level;

                if result.layer_results.is_empty() {
                    continue;
                }

                for (layer, results) in result.layer_results.iter().enumerate() {
                    let max_conn = self.max_neighbors(layer);
                    let m_to_select = max_conn.min(results.len());

                    // Set forward edges
                    if layer == 0 {
                        let mut fwd_ids = [0u32; 64];
                        for (j, c) in results[..m_to_select].iter().enumerate() {
                            fwd_ids[j] = c.id;
                        }
                        self.set_layer0_neighbors(id, &fwd_ids[..m_to_select]);
                    } else {
                        let layer_idx = layer - 1;
                        let upper = &mut self.upper_neighbors[id as usize];
                        while upper.len() <= layer_idx {
                            upper.push(Vec::new());
                        }
                        upper[layer_idx].clear();
                        upper[layer_idx].extend(results[..m_to_select].iter().map(|c| c.id));
                    }

                    // Add back-edges + prune
                    for j in 0..m_to_select {
                        let nb_id = results[j].id;
                        if nb_id == id { continue; }

                        if layer == 0 {
                            let new_count = self.push_layer0_neighbor(nb_id, id);
                            if new_count > max_conn {
                                let nb_start = nb_id as usize * dim;
                                let l0_stride = self.layer0_stride();
                                let l0_start = nb_id as usize * l0_stride;
                                let l0_count = self.layer0_counts[nb_id as usize] as usize;
                                let mut worst_idx = 0;
                                let mut worst_dist = f32::NEG_INFINITY;
                                for k in 0..l0_count {
                                    let nid = self.layer0[l0_start + k];
                                    if !self.alive[nid as usize] {
                                        worst_idx = k;
                                        break;
                                    }
                                    let nid_start = nid as usize * dim;
                                    let d = metric.distance_ord(
                                        &self.vectors[nb_start..nb_start + dim],
                                        &self.vectors[nid_start..nid_start + dim],
                                    );
                                    if d > worst_dist {
                                        worst_dist = d;
                                        worst_idx = k;
                                    }
                                }
                                self.swap_remove_layer0(nb_id, worst_idx);
                            }
                        } else {
                            let layer_idx = layer - 1;
                            let nb_upper = &mut self.upper_neighbors[nb_id as usize];
                            while nb_upper.len() <= layer_idx {
                                nb_upper.push(Vec::new());
                            }
                            nb_upper[layer_idx].push(id);

                            if self.upper_neighbors[nb_id as usize][layer_idx].len() > max_conn {
                                let nb_start = nb_id as usize * dim;
                                let nbs_layer = &self.upper_neighbors[nb_id as usize][layer_idx];
                                let mut worst_idx = 0;
                                let mut worst_dist = f32::NEG_INFINITY;
                                for (k, &nid) in nbs_layer.iter().enumerate() {
                                    if !self.alive[nid as usize] {
                                        worst_idx = k;
                                        break;
                                    }
                                    let nid_start = nid as usize * dim;
                                    let d = metric.distance_ord(
                                        &self.vectors[nb_start..nb_start + dim],
                                        &self.vectors[nid_start..nid_start + dim],
                                    );
                                    if d > worst_dist {
                                        worst_dist = d;
                                        worst_idx = k;
                                    }
                                }
                                self.upper_neighbors[nb_id as usize][layer_idx].swap_remove(worst_idx);
                            }
                        }
                    }
                }

                // Update entry point if new node has higher level
                if new_level > self.max_level() {
                    self.entry_point = Some(id);
                }
            }
        }
    }
}

// ── HnswIndex ───────────────────────────────────────────────────────────────

/// The HNSW index with incremental insertion support.
///
/// Each vector is inserted directly into the HNSW graph in O(M·log N) time.
/// No batch rebuild is required — `flush()` is a no-op.
pub struct HnswIndex {
    config: IndexConfig,
    hnsw_config: HnswConfig,
    graph: HnswGraph,
    /// Mutex allows `search` (&self) to mutate visited tracker.
    /// Lock overhead (~100ns) is negligible vs search time (~50µs).
    visited: Mutex<VisitedTracker>,
}

impl HnswIndex {
    pub fn new(dimensions: usize, metric: Metric, hnsw_config: HnswConfig) -> Self {
        let graph = HnswGraph::new(dimensions, hnsw_config.m, hnsw_config.ef_construction);
        Self {
            config: IndexConfig { dimensions, metric },
            hnsw_config,
            graph,
            visited: Mutex::new(VisitedTracker::new()),
        }
    }

    /// Kept for API compatibility; no longer has any effect.
    pub fn with_flush_threshold(self, _n: usize) -> Self {
        self
    }

    /// Save the index to a binary file (bincode format).
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), VectorDbError> {
        let file = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(file);
        let snapshot = HnswIndexSnapshot {
            dimensions: self.config.dimensions,
            metric: self.config.metric,
            hnsw_config: self.hnsw_config.clone(),
            graph: self.graph.to_snapshot(),
        };
        bincode::serialize_into(&mut writer, &snapshot)
            .map_err(|e| VectorDbError::Serialization(e.to_string()))?;
        Ok(())
    }

    /// Load an index from a binary file.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, VectorDbError> {
        let file = std::fs::File::open(&path)?;
        let reader = BufReader::new(file);

        // Try new format first
        let result: Result<HnswIndexSnapshot, _> = bincode::deserialize_from(reader);
        if let Ok(snapshot) = result {
            let dim = snapshot.dimensions;
            let graph = HnswGraph::from_snapshot(snapshot.graph, dim);
            let mut vt = VisitedTracker::new();
            vt.ensure_capacity(graph.alive.len());
            return Ok(Self {
                config: IndexConfig {
                    dimensions: snapshot.dimensions,
                    metric: snapshot.metric,
                },
                hnsw_config: snapshot.hnsw_config,
                graph,
                visited: Mutex::new(vt),
            });
        }

        // Fall back to legacy format (vectors only, needs rebuild)
        let file = std::fs::File::open(&path)?;
        let reader = BufReader::new(file);
        let legacy: LegacyHnswIndexSnapshot = bincode::deserialize_from(reader)
            .map_err(|e| VectorDbError::Serialization(e.to_string()))?;

        let hnsw_config = legacy.hnsw_config;
        let metric = legacy.metric;
        let mut index = Self::new(legacy.dimensions, metric, hnsw_config);
        let vt = index.visited.get_mut().unwrap();
        for (id, vec) in legacy.vectors {
            index.graph.insert(id as u32, &vec, metric, vt);
        }
        Ok(index)
    }

    /// Persist the HNSW graph to `path`.
    pub fn save_graph(&self, path: &Path) -> Result<(), VectorDbError> {
        if self.graph.count == 0 {
            return Ok(());
        }
        let file = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(file);
        let snapshot = self.graph.to_snapshot();
        bincode::serialize_into(&mut writer, &snapshot)
            .map_err(|e| VectorDbError::Serialization(e.to_string()))?;
        Ok(())
    }

    /// Load a previously saved HNSW graph from `path`.
    pub fn load_graph_mmap(&mut self, path: &Path) -> Result<(), VectorDbError> {
        if !path.exists() {
            return Ok(());
        }
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let snapshot: HnswGraphSnapshot = bincode::deserialize(&mmap[..])
            .map_err(|e| VectorDbError::Serialization(e.to_string()))?;
        self.graph = HnswGraph::from_snapshot(snapshot, self.config.dimensions);
        self.visited.get_mut().unwrap().ensure_capacity(self.graph.alive.len());
        Ok(())
    }

    /// Instrumented batch insert that returns timing stats.
    pub fn add_batch_instrumented(&mut self, entries: &[(u64, Vec<f32>)]) -> InsertStats {
        let mut stats = InsertStats::default();
        let metric = self.config.metric;
        let vt = self.visited.get_mut().unwrap();
        if let Some(max_id) = entries.iter().map(|(id, _)| *id).max() {
            self.graph.ensure_capacity(max_id as u32);
            vt.ensure_capacity(max_id as usize + 1);
        }
        self.graph.reserve(entries.len());
        for (id, vec) in entries {
            self.graph.insert_instrumented(*id as u32, vec, metric, vt, &mut stats);
        }
        stats
    }

    /// Batch insert from a contiguous flat f32 buffer (row-major, N × dim).
    /// IDs are assigned sequentially: start_id, start_id+1, ...
    /// No per-vector allocation — reads directly from the flat slice.
    /// This is the fastest insert path for bulk loading.
    pub fn add_batch_raw(&mut self, data: &[f32], dim: usize, start_id: u64) -> Result<(), VectorDbError> {
        if dim != self.config.dimensions {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.config.dimensions,
                got: dim,
            });
        }
        if data.len() % dim != 0 {
            return Err(VectorDbError::InvalidConfig(
                format!("data length {} is not divisible by dimension {}", data.len(), dim),
            ));
        }
        let n = data.len() / dim;
        let max_id = start_id + n as u64 - 1;
        let metric = self.config.metric;
        let vt = self.visited.get_mut().unwrap();
        self.graph.ensure_capacity(max_id as u32);
        vt.ensure_capacity(max_id as usize + 1);
        self.graph.reserve(n);
        for i in 0..n {
            let vec_slice = &data[i * dim..(i + 1) * dim];
            let id = (start_id + i as u64) as u32;
            self.graph.insert(id, vec_slice, metric, vt);
        }
        Ok(())
    }

    /// Parallel batch insert from a contiguous flat f32 buffer using micro-batching.
    ///
    /// Splits the batch into micro-batches of `micro_batch_size` vectors (default 256).
    /// For each micro-batch:
    /// 1. Store vector data (sequential)
    /// 2. Search for neighbors in parallel using `num_threads` threads (rayon)
    /// 3. Link edges sequentially
    ///
    /// This gives ~N× speedup on N cores for the search phase (which is 90%+ of insert time).
    /// No unsafe code or per-node locks — just phase-separated parallelism.
    pub fn add_batch_parallel(
        &mut self,
        data: &[f32],
        dim: usize,
        start_id: u64,
        num_threads: usize,
        micro_batch_size: usize,
    ) -> Result<(), VectorDbError> {
        if dim != self.config.dimensions {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.config.dimensions,
                got: dim,
            });
        }
        if data.len() % dim != 0 {
            return Err(VectorDbError::InvalidConfig(
                format!("data length {} is not divisible by dimension {}", data.len(), dim),
            ));
        }
        let n = data.len() / dim;
        if n == 0 { return Ok(()); }

        let ids: Vec<u32> = (0..n).map(|i| (start_id + i as u64) as u32).collect();
        let metric = self.config.metric;
        let threads = if num_threads == 0 {
            rayon::current_num_threads()
        } else {
            num_threads
        };
        let batch_size = if micro_batch_size == 0 { 256 } else { micro_batch_size };

        self.graph.insert_batch_parallel(&ids, data, metric, batch_size, threads);
        // Ensure the index-level visited tracker has capacity for subsequent searches
        let vt = self.visited.get_mut().unwrap();
        vt.ensure_capacity(self.graph.alive.len());
        Ok(())
    }

    /// Set the `ef_search` parameter at runtime (no rebuild needed).
    pub fn set_ef_search(&mut self, ef: usize) {
        self.hnsw_config.ef_search = ef;
    }
}

impl VectorIndex for HnswIndex {
    fn add(&mut self, id: u64, vector: &[f32]) -> Result<(), VectorDbError> {
        if vector.len() != self.config.dimensions {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }
        let id32 = id as u32;
        if self.graph.is_alive(id32) {
            return Err(VectorDbError::DuplicateId(id));
        }
        let metric = self.config.metric;
        let vt = self.visited.get_mut().unwrap();
        self.graph.insert(id32, vector, metric, vt);
        Ok(())
    }

    fn add_batch(&mut self, entries: &[(u64, Vec<f32>)]) -> Result<(), VectorDbError> {
        for (id, vec) in entries {
            if vec.len() != self.config.dimensions {
                return Err(VectorDbError::DimensionMismatch {
                    expected: self.config.dimensions,
                    got: vec.len(),
                });
            }
            if self.graph.is_alive(*id as u32) {
                return Err(VectorDbError::DuplicateId(*id));
            }
        }
        let metric = self.config.metric;
        let vt = self.visited.get_mut().unwrap();
        if let Some(max_id) = entries.iter().map(|(id, _)| *id).max() {
            self.graph.ensure_capacity(max_id as u32);
            vt.ensure_capacity(max_id as usize + 1);
        }
        self.graph.reserve(entries.len());
        for (id, vec) in entries {
            self.graph.insert(*id as u32, vec, metric, vt);
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
        if k == 0 || self.graph.count == 0 {
            return Ok(vec![]);
        }

        let mut vt = self.visited.lock().unwrap();
        let results = self.graph.search_knn(
            query, k, self.hnsw_config.ef_search, self.config.metric, &mut vt,
        );

        Ok(results
            .into_iter()
            .map(|(id, distance)| SearchResult { id: id as u64, distance })
            .collect())
    }

    fn delete(&mut self, id: u64) -> bool {
        self.graph.delete(id as u32)
    }

    fn len(&self) -> usize {
        self.graph.count
    }

    fn config(&self) -> &IndexConfig {
        &self.config
    }

    fn iter_vectors(&self) -> Box<dyn Iterator<Item = (u64, Vec<f32>)> + '_> {
        Box::new(
            self.graph.alive.iter()
                .enumerate()
                .filter(|(_, &a)| a)
                .map(|(i, _)| (i as u64, self.graph.vector(i as u32).to_vec()))
        )
    }

    fn flush(&mut self) {
        // No-op: graph is always up-to-date with incremental insertion.
    }

    fn save_graph(&self, path: &std::path::Path) -> Result<(), VectorDbError> {
        HnswIndex::save_graph(self, path)
    }

    fn load_graph_mmap(&mut self, path: &std::path::Path) -> Result<(), VectorDbError> {
        HnswIndex::load_graph_mmap(self, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_index() -> HnswIndex {
        let cfg = HnswConfig {
            ef_construction: 100,
            ef_search: 20,
            m: 8,
        };
        let mut idx = HnswIndex::new(3, Metric::L2, cfg);
        for i in 0..20u64 {
            let v = vec![i as f32, 0.0, 0.0];
            idx.add(i, &v).unwrap();
        }
        idx.flush();
        idx
    }

    #[test]
    fn finds_nearest() {
        let idx = make_index();
        let results = idx.search(&[5.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, 5);
        assert!(results[0].distance < 1e-4);
    }

    #[test]
    fn top_k_ordered() {
        let idx = make_index();
        let results = idx.search(&[10.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(results.len(), 5);
        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance + 1e-5);
        }
    }

    #[test]
    fn dimension_error() {
        let idx = make_index();
        assert!(idx.search(&[1.0, 0.0], 1).is_err());
    }

    #[test]
    fn delete_then_not_found() {
        let mut idx = make_index();
        assert!(idx.delete(5));
        assert_eq!(idx.len(), 19);
    }

    #[test]
    fn save_and_load_round_trip() {
        let idx = make_index();
        let path = "/tmp/hnsw_index_test_v3.bin";
        idx.save(path).unwrap();
        let loaded = HnswIndex::load(path).unwrap();
        assert_eq!(loaded.len(), idx.len());
        assert_eq!(loaded.config().dimensions, idx.config().dimensions);
        assert_eq!(loaded.config().metric, idx.config().metric);
        let orig = idx.search(&[5.0, 0.0, 0.0], 1).unwrap();
        let from_disk = loaded.search(&[5.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(orig[0].id, from_disk[0].id);
    }

    #[test]
    fn fallback_before_flush() {
        let cfg = HnswConfig::default();
        let mut idx = HnswIndex::new(2, Metric::L2, cfg);
        idx.add(1, &[1.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0]).unwrap();
        let r = idx.search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 1);
    }

    #[test]
    fn cosine_metric() {
        let cfg = HnswConfig { ef_construction: 100, ef_search: 20, m: 8 };
        let mut idx = HnswIndex::new(2, Metric::Cosine, cfg);
        for i in 1u64..=20 {
            idx.add(i, &[i as f32, 0.0]).unwrap();
        }
        idx.flush();
        let r = idx.search(&[1.0, 0.0], 1).unwrap();
        assert!(r[0].distance < 1e-4);
    }

    #[test]
    fn dot_product_metric() {
        let cfg = HnswConfig { ef_construction: 100, ef_search: 20, m: 8 };
        let mut idx = HnswIndex::new(2, Metric::DotProduct, cfg);
        for i in 1u64..=20 {
            idx.add(i, &[i as f32, 0.0]).unwrap();
        }
        idx.flush();
        let r = idx.search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 20);
    }

    #[test]
    fn empty_index_search_returns_empty() {
        let idx = HnswIndex::new(3, Metric::L2, HnswConfig::default());
        let r = idx.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn k_zero_returns_empty() {
        let idx = make_index();
        let r = idx.search(&[1.0, 0.0, 0.0], 0).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn auto_flush_triggers_at_threshold() {
        let cfg = HnswConfig::default();
        let mut idx = HnswIndex::new(1, Metric::L2, cfg);
        for i in 0..5u64 {
            idx.add(i, &[i as f32]).unwrap();
        }
        assert_eq!(idx.len(), 5);
        let r = idx.search(&[2.0], 1).unwrap();
        assert_eq!(r[0].id, 2);
    }

    #[test]
    fn delete_then_flush_restores_search() {
        let mut idx = make_index();
        assert!(idx.delete(10));
        idx.flush();
        let r = idx.search(&[10.0, 0.0, 0.0], 1).unwrap();
        assert_ne!(r[0].id, 10);
    }

    #[test]
    fn add_batch_then_explicit_flush() {
        let cfg = HnswConfig { ef_construction: 100, ef_search: 20, m: 8 };
        let mut idx = HnswIndex::new(1, Metric::L2, cfg);
        let entries: Vec<(u64, Vec<f32>)> = (0..50u64).map(|i| (i, vec![i as f32])).collect();
        idx.add_batch(&entries).unwrap();
        idx.flush();
        assert_eq!(idx.len(), 50);
        let r = idx.search(&[25.0], 1).unwrap();
        assert_eq!(r[0].id, 25);
    }

    #[test]
    fn save_with_staging_persists_all_vectors() {
        let cfg = HnswConfig::default();
        let mut idx = HnswIndex::new(1, Metric::L2, cfg);
        for i in 0..10u64 {
            idx.add(i, &[i as f32]).unwrap();
        }
        let path = "/tmp/hnsw_staging_test_v3.bin";
        idx.save(path).unwrap();
        let loaded = HnswIndex::load(path).unwrap();
        assert_eq!(loaded.len(), 10);
        let r = loaded.search(&[5.0], 1).unwrap();
        assert_eq!(r[0].id, 5);
    }

    #[test]
    fn parallel_insert_correctness() {
        // Insert 500 vectors in parallel and verify search finds nearest neighbor
        let dim = 32;
        let n = 500;
        let cfg = HnswConfig {
            ef_construction: 200,
            ef_search: 50,
            m: 16,
        };
        let mut idx = HnswIndex::new(dim, Metric::L2, cfg);
        let mut rng = rand::thread_rng();

        // Generate random vectors
        let mut data = vec![0.0f32; n * dim];
        for v in data.iter_mut() {
            *v = rng.gen::<f32>() * 100.0;
        }

        // Insert with parallel micro-batching (4 threads, batch size 64)
        idx.add_batch_parallel(&data, dim, 0, 4, 64).unwrap();
        assert_eq!(idx.len(), n);

        // Verify: search for each vector should find itself as nearest
        let mut found_self = 0;
        for i in 0..n {
            let query = &data[i * dim..(i + 1) * dim];
            let results = idx.search(query, 1).unwrap();
            if !results.is_empty() && results[0].id == i as u64 {
                found_self += 1;
            }
        }
        // Micro-batching trades slight quality for parallelism (vectors in the same
        // micro-batch can't see each other's edges during search), so 90% is acceptable
        let recall = found_self as f64 / n as f64;
        assert!(recall > 0.90, "self-recall too low: {recall:.2} ({found_self}/{n})");
    }

    #[test]
    fn parallel_insert_matches_sequential() {
        // Both parallel and sequential insert should produce similar recall
        let dim = 16;
        let n = 200;
        let cfg = HnswConfig {
            ef_construction: 200,
            ef_search: 50,
            m: 12,
        };
        let mut rng = rand::thread_rng();
        let mut data = vec![0.0f32; n * dim];
        for v in data.iter_mut() {
            *v = rng.gen::<f32>() * 100.0;
        }

        // Sequential insert
        let mut seq_idx = HnswIndex::new(dim, Metric::L2, cfg.clone());
        seq_idx.add_batch_raw(&data, dim, 0).unwrap();

        // Parallel insert
        let mut par_idx = HnswIndex::new(dim, Metric::L2, cfg);
        par_idx.add_batch_parallel(&data, dim, 0, 4, 32).unwrap();

        assert_eq!(seq_idx.len(), par_idx.len());

        // Compare recall: both should return results for all queries
        let n_queries = 50;
        let k = 10;
        let mut seq_total_dist = 0.0f64;
        let mut par_total_dist = 0.0f64;
        for _ in 0..n_queries {
            let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 100.0).collect();
            let seq_results = seq_idx.search(&query, k).unwrap();
            let par_results = par_idx.search(&query, k).unwrap();
            // Both should return k results
            assert_eq!(seq_results.len(), k);
            assert_eq!(par_results.len(), k);
            seq_total_dist += seq_results[0].distance as f64;
            par_total_dist += par_results[0].distance as f64;
        }
        // Parallel quality should be within 2x of sequential (both are approximate)
        let ratio = par_total_dist / seq_total_dist.max(1e-10);
        assert!(ratio < 2.0, "parallel quality too degraded: ratio={ratio:.2}");
    }

    #[test]
    fn parallel_insert_single_thread_matches_sequential() {
        // With 1 thread, parallel insert should work identically
        let dim = 8;
        let n = 100;
        let cfg = HnswConfig {
            ef_construction: 100,
            ef_search: 30,
            m: 8,
        };
        let mut idx = HnswIndex::new(dim, Metric::L2, cfg);
        let mut data = vec![0.0f32; n * dim];
        let mut rng = rand::thread_rng();
        for v in data.iter_mut() {
            *v = rng.gen::<f32>();
        }

        idx.add_batch_parallel(&data, dim, 0, 1, 256).unwrap();
        assert_eq!(idx.len(), n);

        // Self-recall should be high (micro-batching may slightly reduce quality)
        let mut found = 0;
        for i in 0..n {
            let q = &data[i * dim..(i + 1) * dim];
            let r = idx.search(q, 1).unwrap();
            if !r.is_empty() && r[0].id == i as u64 {
                found += 1;
            }
        }
        assert!(found as f64 / n as f64 > 0.90,
            "self-recall too low: {}/{}", found, n);
    }

    // ── iter_vectors tests ──────────────────────────────────────────────

    #[test]
    fn iter_vectors_returns_all_alive() {
        let idx = make_index();
        let all: Vec<(u64, Vec<f32>)> = idx.iter_vectors().collect();
        assert_eq!(all.len(), 20);
        // Each vector should match what was inserted
        for (id, vec) in &all {
            assert_eq!(vec.len(), 3);
            assert!((vec[0] - *id as f32).abs() < 1e-6);
            assert!((vec[1]).abs() < 1e-6);
            assert!((vec[2]).abs() < 1e-6);
        }
    }

    #[test]
    fn iter_vectors_excludes_deleted() {
        let mut idx = make_index();
        idx.delete(3);
        idx.delete(7);
        idx.delete(15);
        let all: Vec<(u64, Vec<f32>)> = idx.iter_vectors().collect();
        assert_eq!(all.len(), 17);
        let ids: Vec<u64> = all.iter().map(|(id, _)| *id).collect();
        assert!(!ids.contains(&3));
        assert!(!ids.contains(&7));
        assert!(!ids.contains(&15));
    }

    #[test]
    fn iter_vectors_empty_index() {
        let idx = HnswIndex::new(3, Metric::L2, HnswConfig::default());
        let all: Vec<(u64, Vec<f32>)> = idx.iter_vectors().collect();
        assert!(all.is_empty());
    }

    // ── set_ef_search tests ─────────────────────────────────────────────

    #[test]
    fn set_ef_search_changes_parameter() {
        let mut idx = make_index();
        assert_eq!(idx.hnsw_config.ef_search, 20);
        idx.set_ef_search(100);
        assert_eq!(idx.hnsw_config.ef_search, 100);
        // Search should still work after changing ef_search
        let r = idx.search(&[5.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 5);
    }

    #[test]
    fn set_ef_search_to_1() {
        let mut idx = make_index();
        idx.set_ef_search(1);
        // With ef=1, search is greedy — may not find exact best but should return something
        let r = idx.search(&[5.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn set_ef_search_high_improves_results() {
        // Build a larger index where ef_search matters
        let dim = 16;
        let n = 200;
        let cfg = HnswConfig { ef_construction: 100, ef_search: 1, m: 8 };
        let mut idx = HnswIndex::new(dim, Metric::L2, cfg);
        let mut rng = rand::thread_rng();
        let mut data = vec![0.0f32; n * dim];
        for v in data.iter_mut() {
            *v = rng.gen::<f32>() * 100.0;
        }
        for i in 0..n {
            idx.add(i as u64, &data[i * dim..(i + 1) * dim]).unwrap();
        }

        // Count self-recall with ef=1
        let mut found_low = 0;
        for i in 0..n {
            let q = &data[i * dim..(i + 1) * dim];
            let r = idx.search(q, 1).unwrap();
            if !r.is_empty() && r[0].id == i as u64 { found_low += 1; }
        }

        // Now increase ef_search and re-test
        idx.set_ef_search(200);
        let mut found_high = 0;
        for i in 0..n {
            let q = &data[i * dim..(i + 1) * dim];
            let r = idx.search(q, 1).unwrap();
            if !r.is_empty() && r[0].id == i as u64 { found_high += 1; }
        }
        assert!(found_high >= found_low,
            "higher ef_search should not degrade recall: low={found_low} high={found_high}");
    }

    // ── VisitedTracker tests (via public API) ───────────────────────────

    #[test]
    fn visited_tracker_generation_overflow() {
        // Test that VisitedTracker handles generation overflow gracefully
        let mut vt = VisitedTracker::new();
        vt.ensure_capacity(10);
        // Set generation close to overflow
        vt.gen = u32::MAX;
        vt.visit(0);
        assert_eq!(vt.marks[0], u32::MAX);

        // This reset should trigger overflow handling: clears marks, sets gen=1
        vt.reset();
        assert_eq!(vt.gen, 1);
        // After overflow reset, all marks should be cleared
        for &m in vt.marks.iter() {
            assert_eq!(m, 0);
        }
        // visit should work after overflow
        assert!(vt.visit(0)); // newly visited
        assert!(!vt.visit(0)); // already visited
    }

    #[test]
    fn visited_tracker_ensure_capacity() {
        let mut vt = VisitedTracker::new();
        assert_eq!(vt.marks.len(), 0);
        vt.ensure_capacity(100);
        assert!(vt.marks.len() >= 100);
        // Calling with smaller value is a no-op
        vt.ensure_capacity(50);
        assert!(vt.marks.len() >= 100);
        // Calling with larger value grows
        vt.ensure_capacity(200);
        assert!(vt.marks.len() >= 200);
    }

    #[test]
    fn visited_tracker_reset_increments_generation() {
        let mut vt = VisitedTracker::new();
        vt.ensure_capacity(5);
        assert_eq!(vt.gen, 1);
        assert!(vt.visit(0));
        assert!(!vt.visit(0));
        vt.reset();
        assert_eq!(vt.gen, 2);
        // After reset, same node should be "newly visited"
        assert!(vt.visit(0));
    }

    // ── Delete entry point tests ────────────────────────────────────────

    #[test]
    fn delete_entry_point_selects_new_one() {
        let cfg = HnswConfig { ef_construction: 100, ef_search: 20, m: 8 };
        let mut idx = HnswIndex::new(3, Metric::L2, cfg);
        // Insert just a few vectors
        for i in 0..5u64 {
            idx.add(i, &[i as f32, 0.0, 0.0]).unwrap();
        }
        // The entry point is one of 0..5
        let ep = idx.graph.entry_point.unwrap();
        // Delete the entry point
        assert!(idx.delete(ep as u64));
        // A new entry point should be selected
        assert!(idx.graph.entry_point.is_some());
        assert_ne!(idx.graph.entry_point.unwrap(), ep);
        // Search should still work
        let r = idx.search(&[2.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn delete_all_clears_entry_point() {
        let cfg = HnswConfig { ef_construction: 100, ef_search: 20, m: 8 };
        let mut idx = HnswIndex::new(2, Metric::L2, cfg);
        idx.add(0, &[1.0, 0.0]).unwrap();
        idx.add(1, &[0.0, 1.0]).unwrap();
        idx.delete(0);
        idx.delete(1);
        assert_eq!(idx.len(), 0);
        assert!(idx.graph.entry_point.is_none());
        // Search on empty index should return empty
        let r = idx.search(&[1.0, 0.0], 1).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn delete_non_existent_returns_false() {
        let mut idx = make_index();
        assert!(!idx.delete(999));
        assert_eq!(idx.len(), 20);
    }

    #[test]
    fn delete_twice_returns_false_second_time() {
        let mut idx = make_index();
        assert!(idx.delete(5));
        assert!(!idx.delete(5));
        assert_eq!(idx.len(), 19);
    }

    // ── Search edge cases ───────────────────────────────────────────────

    #[test]
    fn search_ef_1_returns_result() {
        let cfg = HnswConfig { ef_construction: 100, ef_search: 1, m: 8 };
        let mut idx = HnswIndex::new(3, Metric::L2, cfg);
        for i in 0..10u64 {
            idx.add(i, &[i as f32, 0.0, 0.0]).unwrap();
        }
        let r = idx.search(&[5.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn search_k_greater_than_len() {
        let cfg = HnswConfig { ef_construction: 100, ef_search: 20, m: 8 };
        let mut idx = HnswIndex::new(2, Metric::L2, cfg);
        idx.add(0, &[1.0, 0.0]).unwrap();
        idx.add(1, &[0.0, 1.0]).unwrap();
        idx.add(2, &[1.0, 1.0]).unwrap();
        // k=10 but only 3 vectors in index
        let r = idx.search(&[0.5, 0.5], 10).unwrap();
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn search_single_vector_index() {
        let cfg = HnswConfig { ef_construction: 100, ef_search: 20, m: 8 };
        let mut idx = HnswIndex::new(3, Metric::L2, cfg);
        idx.add(42, &[1.0, 2.0, 3.0]).unwrap();
        let r = idx.search(&[1.0, 2.0, 3.0], 5).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].id, 42);
        assert!(r[0].distance < 1e-6);
    }

    #[test]
    fn search_after_all_deleted_returns_empty() {
        let cfg = HnswConfig { ef_construction: 100, ef_search: 20, m: 8 };
        let mut idx = HnswIndex::new(2, Metric::L2, cfg);
        idx.add(0, &[1.0, 0.0]).unwrap();
        idx.add(1, &[0.0, 1.0]).unwrap();
        idx.delete(0);
        idx.delete(1);
        let r = idx.search(&[0.5, 0.5], 1).unwrap();
        assert!(r.is_empty());
    }

    // ── Parallel insert edge cases ──────────────────────────────────────

    #[test]
    fn parallel_insert_empty_batch() {
        let cfg = HnswConfig { ef_construction: 100, ef_search: 20, m: 8 };
        let mut idx = HnswIndex::new(4, Metric::L2, cfg);
        let data: Vec<f32> = vec![];
        idx.add_batch_parallel(&data, 4, 0, 4, 64).unwrap();
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn parallel_insert_bootstrap_larger_than_n() {
        // micro_batch_size > n means all vectors go through bootstrap (sequential)
        let dim = 4;
        let n = 10;
        let cfg = HnswConfig { ef_construction: 100, ef_search: 20, m: 8 };
        let mut idx = HnswIndex::new(dim, Metric::L2, cfg);
        let mut data = vec![0.0f32; n * dim];
        let mut rng = rand::thread_rng();
        for v in data.iter_mut() {
            *v = rng.gen::<f32>();
        }
        // micro_batch_size = 1000 >> n = 10
        idx.add_batch_parallel(&data, dim, 0, 2, 1000).unwrap();
        assert_eq!(idx.len(), n);
        // Should still find vectors
        let r = idx.search(&data[0..dim], 1).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].id, 0);
    }

    #[test]
    fn parallel_insert_dimension_mismatch() {
        let cfg = HnswConfig { ef_construction: 100, ef_search: 20, m: 8 };
        let mut idx = HnswIndex::new(4, Metric::L2, cfg);
        let data = vec![1.0f32; 15]; // 15 is not divisible by 4... actually it is not
        // Wrong dimension
        let result = idx.add_batch_parallel(&data, 3, 0, 2, 64);
        assert!(result.is_err());
    }

    // ── Snapshot edge cases ─────────────────────────────────────────────

    #[test]
    fn snapshot_empty_graph_round_trip() {
        let cfg = HnswConfig { ef_construction: 100, ef_search: 20, m: 8 };
        let idx = HnswIndex::new(4, Metric::L2, cfg);
        let path = "/tmp/hnsw_empty_snapshot_test.bin";
        // save_graph returns Ok(()) for empty graph (no-op)
        idx.save_graph(std::path::Path::new(path)).unwrap();
        // save (full index) should work for empty
        idx.save(path).unwrap();
        let loaded = HnswIndex::load(path).unwrap();
        assert_eq!(loaded.len(), 0);
        let r = loaded.search(&[1.0, 2.0, 3.0, 4.0], 1).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn snapshot_sparse_ids_round_trip() {
        let cfg = HnswConfig { ef_construction: 100, ef_search: 20, m: 8 };
        let mut idx = HnswIndex::new(2, Metric::L2, cfg);
        // Use sparse IDs: 0, 10, 100, 500
        idx.add(0, &[0.0, 0.0]).unwrap();
        idx.add(10, &[1.0, 0.0]).unwrap();
        idx.add(100, &[2.0, 0.0]).unwrap();
        idx.add(500, &[3.0, 0.0]).unwrap();
        let path = "/tmp/hnsw_sparse_ids_test.bin";
        idx.save(path).unwrap();
        let loaded = HnswIndex::load(path).unwrap();
        assert_eq!(loaded.len(), 4);
        // Verify each vector is retrievable
        let r = loaded.search(&[0.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 0);
        let r = loaded.search(&[3.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 500);
    }

    #[test]
    fn snapshot_after_deletes_round_trip() {
        let mut idx = make_index();
        idx.delete(5);
        idx.delete(10);
        idx.delete(15);
        let path = "/tmp/hnsw_deleted_snapshot_test.bin";
        idx.save(path).unwrap();
        let loaded = HnswIndex::load(path).unwrap();
        assert_eq!(loaded.len(), 17);
        // Deleted IDs should not be found
        let all: Vec<(u64, Vec<f32>)> = loaded.iter_vectors().collect();
        let ids: Vec<u64> = all.iter().map(|(id, _)| *id).collect();
        assert!(!ids.contains(&5));
        assert!(!ids.contains(&10));
        assert!(!ids.contains(&15));
    }

    // ── Multi-metric tests ──────────────────────────────────────────────

    #[test]
    fn l2_metric_insert_and_search() {
        let cfg = HnswConfig { ef_construction: 100, ef_search: 50, m: 8 };
        let mut idx = HnswIndex::new(3, Metric::L2, cfg);
        idx.add(0, &[0.0, 0.0, 0.0]).unwrap();
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[10.0, 0.0, 0.0]).unwrap();
        let r = idx.search(&[0.5, 0.0, 0.0], 1).unwrap();
        // Closest to [0.5, 0, 0] under L2 should be id=0 or id=1
        assert!(r[0].id == 0 || r[0].id == 1);
    }

    #[test]
    fn cosine_metric_direction_matters() {
        let cfg = HnswConfig { ef_construction: 100, ef_search: 50, m: 8 };
        let mut idx = HnswIndex::new(2, Metric::Cosine, cfg);
        idx.add(0, &[1.0, 0.0]).unwrap();
        idx.add(1, &[0.0, 1.0]).unwrap();
        idx.add(2, &[0.707, 0.707]).unwrap();
        // Query in x-direction; closest cosine should be id=0
        let r = idx.search(&[10.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 0);
        // Query in y-direction; closest cosine should be id=1
        let r = idx.search(&[0.0, 10.0], 1).unwrap();
        assert_eq!(r[0].id, 1);
    }

    #[test]
    fn dot_product_favors_largest_projection() {
        let cfg = HnswConfig { ef_construction: 100, ef_search: 50, m: 8 };
        let mut idx = HnswIndex::new(2, Metric::DotProduct, cfg);
        idx.add(0, &[1.0, 0.0]).unwrap();
        idx.add(1, &[5.0, 0.0]).unwrap();
        idx.add(2, &[10.0, 0.0]).unwrap();
        // DotProduct: query [1,0] should favor largest x component (id=2)
        let r = idx.search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 2);
    }

    #[test]
    fn all_metrics_search_returns_results() {
        for metric in &[Metric::L2, Metric::Cosine, Metric::DotProduct] {
            let cfg = HnswConfig { ef_construction: 100, ef_search: 20, m: 8 };
            let mut idx = HnswIndex::new(4, *metric, cfg);
            for i in 1..=10u64 {
                idx.add(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
            }
            let r = idx.search(&[5.0, 0.0, 0.0, 0.0], 3).unwrap();
            assert_eq!(r.len(), 3, "metric {:?} should return 3 results", metric);
        }
    }

    // ── Layer-0 buffer helpers ──────────────────────────────────────────

    #[test]
    fn layer0_set_and_get_neighbors() {
        let mut graph = HnswGraph::new(2, 4, 100); // m=4, m_max0=8
        graph.ensure_capacity(5);
        graph.alive[0] = true;
        // Set neighbors for node 0
        graph.set_layer0_neighbors(0, &[1, 2, 3]);
        let nbs = graph.get_neighbors(0, 0);
        assert_eq!(nbs, &[1, 2, 3]);
    }

    #[test]
    fn layer0_push_neighbor() {
        let mut graph = HnswGraph::new(2, 4, 100);
        graph.ensure_capacity(5);
        graph.alive[0] = true;
        graph.set_layer0_neighbors(0, &[1, 2]);
        let new_count = graph.push_layer0_neighbor(0, 3);
        assert_eq!(new_count, 3);
        let nbs = graph.get_neighbors(0, 0);
        assert_eq!(nbs, &[1, 2, 3]);
    }

    #[test]
    fn layer0_swap_remove() {
        let mut graph = HnswGraph::new(2, 4, 100);
        graph.ensure_capacity(5);
        graph.alive[0] = true;
        graph.set_layer0_neighbors(0, &[10, 20, 30, 40]);
        // Remove index 1 (value 20), should swap in last element (40)
        graph.swap_remove_layer0(0, 1);
        let nbs = graph.get_neighbors(0, 0);
        assert_eq!(nbs.len(), 3);
        assert_eq!(nbs[0], 10);
        assert_eq!(nbs[1], 40); // swapped from end
        assert_eq!(nbs[2], 30);
    }

    #[test]
    fn layer0_swap_remove_last() {
        let mut graph = HnswGraph::new(2, 4, 100);
        graph.ensure_capacity(5);
        graph.alive[0] = true;
        graph.set_layer0_neighbors(0, &[10, 20, 30]);
        // Remove last element
        graph.swap_remove_layer0(0, 2);
        let nbs = graph.get_neighbors(0, 0);
        assert_eq!(nbs, &[10, 20]);
    }

    #[test]
    fn layer0_overwrite_neighbors() {
        let mut graph = HnswGraph::new(2, 4, 100);
        graph.ensure_capacity(5);
        graph.alive[0] = true;
        graph.set_layer0_neighbors(0, &[1, 2, 3, 4, 5]);
        assert_eq!(graph.get_neighbors(0, 0).len(), 5);
        // Overwrite with fewer neighbors
        graph.set_layer0_neighbors(0, &[10, 20]);
        assert_eq!(graph.get_neighbors(0, 0), &[10, 20]);
    }

    // ── Duplicate ID error ──────────────────────────────────────────────

    #[test]
    fn add_duplicate_id_returns_error() {
        let mut idx = make_index();
        let result = idx.add(5, &[99.0, 99.0, 99.0]);
        assert!(result.is_err());
        assert_eq!(idx.len(), 20); // no change
    }

    // ── add_batch_raw tests ─────────────────────────────────────────────

    #[test]
    fn add_batch_raw_dimension_mismatch() {
        let cfg = HnswConfig::default();
        let mut idx = HnswIndex::new(3, Metric::L2, cfg);
        let data = vec![1.0f32; 8]; // 8 / 4 = 2 vectors of dim 4, but index is dim 3
        let result = idx.add_batch_raw(&data, 4, 0);
        assert!(result.is_err());
    }

    #[test]
    fn add_batch_raw_not_divisible() {
        let cfg = HnswConfig::default();
        let mut idx = HnswIndex::new(3, Metric::L2, cfg);
        let data = vec![1.0f32; 7]; // 7 not divisible by 3
        let result = idx.add_batch_raw(&data, 3, 0);
        assert!(result.is_err());
    }

    // ── HnswGraph snapshot from_snapshot empty ──────────────────────────

    #[test]
    fn graph_from_snapshot_empty() {
        let snap = HnswGraphSnapshot {
            nodes: HashMap::new(),
            entry_point: None,
            m: 8,
            m_max0: 16,
            ef_construction: 100,
            ml: 1.0 / (8.0f64).ln(),
        };
        let graph = HnswGraph::from_snapshot(snap, 4);
        assert_eq!(graph.count, 0);
        assert!(graph.entry_point.is_none());
        assert_eq!(graph.dim, 4);
        assert_eq!(graph.m, 8);
    }

    // ── with_flush_threshold (no-op) ────────────────────────────────────

    #[test]
    fn with_flush_threshold_is_noop() {
        let cfg = HnswConfig::default();
        let idx = HnswIndex::new(3, Metric::L2, cfg).with_flush_threshold(42);
        assert_eq!(idx.config().dimensions, 3);
    }

    // ── config accessor ─────────────────────────────────────────────────

    #[test]
    fn config_returns_correct_values() {
        let cfg = HnswConfig { ef_construction: 50, ef_search: 10, m: 4 };
        let idx = HnswIndex::new(8, Metric::Cosine, cfg);
        assert_eq!(idx.config().dimensions, 8);
        assert_eq!(idx.config().metric, Metric::Cosine);
    }

    // ── Graph internal consistency ──────────────────────────────────────

    #[test]
    fn graph_count_matches_alive() {
        let mut idx = make_index();
        assert_eq!(idx.graph.count, 20);
        idx.delete(0);
        idx.delete(19);
        assert_eq!(idx.graph.count, 18);
        let alive_count = idx.graph.alive.iter().filter(|&&a| a).count();
        assert_eq!(alive_count, 18);
    }

    #[test]
    fn graph_max_neighbors_layer0_vs_upper() {
        let graph = HnswGraph::new(4, 8, 100); // m=8
        assert_eq!(graph.max_neighbors(0), 16); // m_max0 = 2*m = 16
        assert_eq!(graph.max_neighbors(1), 8);  // upper layers = m
        assert_eq!(graph.max_neighbors(5), 8);
    }

    // ── Batch add error on duplicate ────────────────────────────────────

    #[test]
    fn add_batch_duplicate_id_error() {
        let mut idx = make_index();
        let entries: Vec<(u64, Vec<f32>)> = vec![
            (5, vec![99.0, 99.0, 99.0]), // id=5 already exists
        ];
        let result = idx.add_batch(&entries);
        assert!(result.is_err());
    }

    // ── Large batch with delete and re-search ───────────────────────────

    #[test]
    fn bulk_insert_delete_half_search_remaining() {
        let cfg = HnswConfig { ef_construction: 100, ef_search: 50, m: 8 };
        let mut idx = HnswIndex::new(2, Metric::L2, cfg);
        let n = 100u64;
        for i in 0..n {
            idx.add(i, &[i as f32, 0.0]).unwrap();
        }
        assert_eq!(idx.len(), 100);
        // Delete even IDs
        for i in (0..n).step_by(2) {
            idx.delete(i);
        }
        assert_eq!(idx.len(), 50);
        // iter_vectors should only show odd IDs
        let remaining: Vec<u64> = idx.iter_vectors().map(|(id, _)| id).collect();
        assert_eq!(remaining.len(), 50);
        for id in &remaining {
            assert!(id % 2 == 1, "iter_vectors should only yield odd IDs, got {}", id);
        }
        // Search should return results from remaining vectors
        let r = idx.search(&[1.0, 0.0], 3).unwrap();
        assert!(!r.is_empty(), "search should return results after partial delete");
    }

    // ── Stress: many searches without reset issues ──────────────────────

    #[test]
    fn many_searches_no_tracker_issue() {
        let idx = make_index();
        // Run many searches to stress VisitedTracker reset
        for i in 0..500 {
            let q = vec![(i % 20) as f32, 0.0, 0.0];
            let r = idx.search(&q, 3).unwrap();
            assert!(!r.is_empty());
        }
    }
}
