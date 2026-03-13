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

/// Tracks which nodes have been visited during a search.
/// Uses a generation counter to avoid clearing the Vec between searches.
struct VisitedTracker {
    marks: Vec<u32>,
    gen: u32,
}

impl VisitedTracker {
    fn new() -> Self {
        Self { marks: Vec::new(), gen: 1 }
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
    /// `neighbors[node_id][layer]` = Vec of neighbor u32 IDs.
    neighbors: Vec<Vec<Vec<u32>>>,
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

impl HnswGraph {
    fn new(dim: usize, m: usize, ef_construction: usize) -> Self {
        Self {
            vectors: Vec::new(),
            levels: Vec::new(),
            neighbors: Vec::new(),
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

    /// Ensure storage capacity for node `id`.
    fn ensure_capacity(&mut self, id: u32) {
        let needed = id as usize + 1;
        if needed > self.alive.len() {
            self.vectors.resize(needed * self.dim, 0.0);
            self.levels.resize(needed, 0);
            self.neighbors.resize_with(needed, Vec::new);
            self.alive.resize(needed, false);
        }
    }

    /// Pre-allocate for `additional` more nodes.
    fn reserve(&mut self, additional: usize) {
        self.vectors.reserve(additional * self.dim);
        self.levels.reserve(additional);
        self.neighbors.reserve(additional);
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
        let mut candidates = BinaryHeap::<std::cmp::Reverse<Candidate>>::new();
        let mut results = BinaryHeap::<Candidate>::new();
        let mut farthest_dist = f32::MAX;

        for &ep_id in entry_ids {
            if !vt.visit(ep_id) || !self.alive[ep_id as usize] { continue; }
            let d = metric.distance(query, self.vector(ep_id));
            let c = Candidate { distance: d, id: ep_id };
            candidates.push(std::cmp::Reverse(c));
            results.push(c);
            farthest_dist = d; // single entry point typical
        }

        while let Some(std::cmp::Reverse(closest)) = candidates.pop() {
            if closest.distance > farthest_dist && results.len() >= ef {
                break;
            }

            let nbs = &self.neighbors[closest.id as usize];
            if layer < nbs.len() {
                for &nb_id in &nbs[layer] {
                    if !vt.visit(nb_id) || !self.alive[nb_id as usize] { continue; }

                    let d = metric.distance(query, self.vector(nb_id));

                    if results.len() < ef || d < farthest_dist {
                        let c = Candidate { distance: d, id: nb_id };
                        candidates.push(std::cmp::Reverse(c));
                        results.push(c);
                        if results.len() > ef {
                            results.pop();
                        }
                        // Update cached farthest distance
                        if let Some(f) = results.peek() {
                            farthest_dist = f.distance;
                        }
                    }
                }
            }
        }

        let mut result_vec: Vec<Candidate> = results.into_vec();
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
        let mut current_dist = metric.distance(query, self.vector(current));

        loop {
            let mut changed = false;
            let nbs = &self.neighbors[current as usize];
            if layer < nbs.len() {
                for &nb_id in &nbs[layer] {
                    let d = metric.distance(query, self.vector(nb_id));
                    if d < current_dist {
                        current = nb_id;
                        current_dist = d;
                        changed = true;
                    }
                }
            }
            if !changed { break; }
        }
        current
    }

    /// Insert a vector into the graph incrementally.
    fn insert(&mut self, id: u32, vector: &[f32], metric: Metric, vt: &mut VisitedTracker) {
        let new_level = self.random_level();
        let dim = self.dim;

        // Allocate slot and store data
        self.ensure_capacity(id);
        vt.ensure_capacity(id as usize + 1);
        self.set_vector(id, vector);
        self.levels[id as usize] = new_level as u8;
        // Pre-allocate neighbor vecs with expected capacity
        let mut layer_vecs = Vec::with_capacity(new_level + 1);
        for l in 0..=new_level {
            layer_vecs.push(Vec::with_capacity(if l == 0 { self.m_max0 } else { self.m }));
        }
        self.neighbors[id as usize] = layer_vecs;
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
        // Use the original `vector` parameter — same data, no clone needed.
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

            // Set forward edges directly from results (no intermediate Vec + clone)
            self.neighbors[id as usize][layer].clear();
            self.neighbors[id as usize][layer].extend(
                results[..m_to_select].iter().map(|c| c.id)
            );

            // Add back-edges — iterate results slice directly
            for i in 0..m_to_select {
                let nb_id = results[i].id;
                if nb_id == id { continue; }

                // Add back-edge
                let nb_nbs = &mut self.neighbors[nb_id as usize];
                while nb_nbs.len() <= layer {
                    nb_nbs.push(Vec::new());
                }
                nb_nbs[layer].push(id);

                // Prune if over capacity: only 1 excess, so just find & remove the worst
                if self.neighbors[nb_id as usize][layer].len() > max_conn {
                    let nb_start = nb_id as usize * dim;
                    let nbs_layer = &self.neighbors[nb_id as usize][layer];
                    let mut worst_idx = 0;
                    let mut worst_dist = f32::NEG_INFINITY;
                    for (j, &nid) in nbs_layer.iter().enumerate() {
                        if !self.alive[nid as usize] {
                            // Dead nodes are worst — remove them first
                            worst_idx = j;
                            break;
                        }
                        let nid_start = nid as usize * dim;
                        let d = metric.distance(
                            &self.vectors[nb_start..nb_start + dim],
                            &self.vectors[nid_start..nid_start + dim],
                        );
                        if d > worst_dist {
                            worst_dist = d;
                            worst_idx = j;
                        }
                    }
                    self.neighbors[nb_id as usize][layer].swap_remove(worst_idx);
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
            .map(|c| (c.id, c.distance))
            .collect()
    }

    /// Delete a node from the graph.
    fn delete(&mut self, id: u32) -> bool {
        if !self.is_alive(id) { return false; }

        self.alive[id as usize] = false;
        self.count -= 1;

        // Remove back-edges from all neighbors
        let level = self.levels[id as usize] as usize;
        for layer in 0..=level {
            if layer < self.neighbors[id as usize].len() {
                let nbs: Vec<u32> = self.neighbors[id as usize][layer].clone();
                for nb_id in nbs {
                    let nb_idx = nb_id as usize;
                    if nb_idx < self.neighbors.len() && layer < self.neighbors[nb_idx].len() {
                        self.neighbors[nb_idx][layer].retain(|&nid| nid != id);
                    }
                }
            }
        }
        self.neighbors[id as usize].clear();

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
                nodes.insert(i as u64, HnswNodeSnapshot {
                    vector: self.vector(i as u32).to_vec(),
                    neighbors: self.neighbors[i].iter()
                        .map(|layer_nbs| layer_nbs.iter().map(|&n| n as u64).collect())
                        .collect(),
                    level: self.levels[i] as usize,
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

        let mut graph = Self::new(dim, snap.m, snap.ef_construction);
        graph.m_max0 = snap.m_max0;
        graph.ml = snap.ml;
        graph.entry_point = snap.entry_point.map(|e| e as u32);

        graph.vectors.resize(capacity * dim, 0.0);
        graph.levels.resize(capacity, 0);
        graph.neighbors.resize_with(capacity, Vec::new);
        graph.alive.resize(capacity, false);

        for (id, node) in snap.nodes {
            let idx = id as usize;
            graph.set_vector(id as u32, &node.vector);
            graph.levels[idx] = node.level as u8;
            graph.neighbors[idx] = node.neighbors.iter()
                .map(|layer| layer.iter().map(|&n| n as u32).collect())
                .collect();
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
        let mut candidates = BinaryHeap::<std::cmp::Reverse<Candidate>>::new();
        let mut results = BinaryHeap::<Candidate>::new();
        let mut farthest_dist = f32::MAX;

        for &ep_id in entry_ids {
            if !vt.visit(ep_id) { continue; }
            let t = std::time::Instant::now();
            let vec = self.vector(ep_id);
            stats.sl_hash_lookup += t.elapsed();
            let t = std::time::Instant::now();
            let d = metric.distance(query, vec);
            stats.sl_distance += t.elapsed();
            let c = Candidate { distance: d, id: ep_id };
            let t = std::time::Instant::now();
            candidates.push(std::cmp::Reverse(c));
            results.push(c);
            farthest_dist = d;
            stats.sl_heap_ops += t.elapsed();
        }

        while let Some(std::cmp::Reverse(closest)) = candidates.pop() {
            if closest.distance > farthest_dist && results.len() >= ef {
                break;
            }

            let nbs = &self.neighbors[closest.id as usize];
            if layer < nbs.len() {
                for &nb_id in &nbs[layer] {
                    if !vt.visit(nb_id) { continue; }

                    let t = std::time::Instant::now();
                    let vec = self.vector(nb_id);
                    stats.sl_hash_lookup += t.elapsed();
                    let t = std::time::Instant::now();
                    let d = metric.distance(query, vec);
                    stats.sl_distance += t.elapsed();

                    if results.len() < ef || d < farthest_dist {
                        let c = Candidate { distance: d, id: nb_id };
                        let t = std::time::Instant::now();
                        candidates.push(std::cmp::Reverse(c));
                        results.push(c);
                        if results.len() > ef {
                            results.pop();
                        }
                        if let Some(f) = results.peek() {
                            farthest_dist = f.distance;
                        }
                        stats.sl_heap_ops += t.elapsed();
                    }
                }
            }
        }

        let mut result_vec: Vec<Candidate> = results.into_vec();
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
        // Pre-allocate neighbor vecs with expected capacity
        let mut layer_vecs = Vec::with_capacity(new_level + 1);
        for l in 0..=new_level {
            layer_vecs.push(Vec::with_capacity(if l == 0 { self.m_max0 } else { self.m }));
        }
        self.neighbors[id as usize] = layer_vecs;
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

        // Use original vector parameter directly — no clone needed
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
            // Set forward edges directly
            self.neighbors[id as usize][layer].clear();
            self.neighbors[id as usize][layer].extend(
                results[..m_to_select].iter().map(|c| c.id)
            );
            stats.set_neighbors += t.elapsed();

            for i in 0..m_to_select {
                let nb_id = results[i].id;
                if nb_id == id { continue; }

                let t = std::time::Instant::now();
                let nb_nbs = &mut self.neighbors[nb_id as usize];
                while nb_nbs.len() <= layer {
                    nb_nbs.push(Vec::new());
                }
                nb_nbs[layer].push(id);
                stats.back_edges += t.elapsed();

                // Prune if over capacity: swap-remove the worst
                if self.neighbors[nb_id as usize][layer].len() > max_conn {
                    let t = std::time::Instant::now();
                    let nb_start = nb_id as usize * dim;
                    let nbs_layer = &self.neighbors[nb_id as usize][layer];
                    let mut worst_idx = 0;
                    let mut worst_dist = f32::NEG_INFINITY;
                    for (j, &nid) in nbs_layer.iter().enumerate() {
                        if !self.alive[nid as usize] {
                            worst_idx = j;
                            break;
                        }
                        let nid_start = nid as usize * dim;
                        let d = metric.distance(
                            &self.vectors[nb_start..nb_start + dim],
                            &self.vectors[nid_start..nid_start + dim],
                        );
                        if d > worst_dist {
                            worst_dist = d;
                            worst_idx = j;
                        }
                    }
                    self.neighbors[nb_id as usize][layer].swap_remove(worst_idx);
                    stats.pruning += t.elapsed();
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
}
