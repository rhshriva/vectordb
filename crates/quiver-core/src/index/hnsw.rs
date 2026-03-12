/// HNSW (Hierarchical Navigable Small World) approximate nearest-neighbour index.
///
/// Custom incremental implementation based on the Malkov & Yashunin 2016 paper.
/// Supports true online insertion — each vector is inserted directly into the
/// graph in O(M·log N) time, with no batch rebuild required.
///
/// ## Key parameters
/// | Parameter         | Default | Effect |
/// |-------------------|---------|--------|
/// | `ef_construction` | 200     | Beam width during index build. Higher → better recall, slower build. |
/// | `ef_search`       | 50      | Beam width during query. Higher → better recall, slower query. |
/// | `m`               | 12      | Edges per node per layer. Higher → better recall, more memory. |
///
/// ## Tradeoffs vs FlatIndex
/// - FlatIndex: 100% recall, O(N·D) query, trivial updates
/// - HnswIndex: ~95–99% recall (tunable), O(log N · ef) query, incremental insert
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{BufReader, BufWriter};
use std::path::Path;

use rand::Rng;

use crate::{
    distance::Metric,
    error::VectorDbError,
    index::{IndexConfig, SearchResult, VectorIndex},
};

// ── Fast hasher for u64 keys ────────────────────────────────────────────────
// Replaces SipHash (designed for DOS resistance) with identity hash for u64.
// Since our IDs are sequential integers, they distribute well without mixing.

#[derive(Default)]
struct IdHasher(u64);

impl Hasher for IdHasher {
    #[inline]
    fn finish(&self) -> u64 { self.0 }
    #[inline]
    fn write_u64(&mut self, i: u64) { self.0 = i; }
    #[inline]
    fn write(&mut self, _bytes: &[u8]) { unreachable!("IdHasher only supports u64") }
}

type FastHashMap<V> = HashMap<u64, V, BuildHasherDefault<IdHasher>>;
type FastHashSet = HashSet<u64, BuildHasherDefault<IdHasher>>;

// ── Serialization snapshot (legacy format for migration) ─────────────────────

#[derive(serde::Serialize, serde::Deserialize)]
struct LegacyHnswIndexSnapshot {
    dimensions: usize,
    metric: Metric,
    hnsw_config: HnswConfig,
    flush_threshold: usize,
    vectors: HashMap<u64, Vec<f32>>,  // uses default hasher for serde compat
}

// ── New serialization snapshot ───────────────────────────────────────────────

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

// ── Priority queue helpers ──────────────────────────────────────────────────

/// Element for priority queue: (distance, id). Sorted by distance ascending.
#[derive(Clone, Copy, PartialEq)]
struct Candidate {
    distance: f32,
    id: u64,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Natural ordering: smallest distance first (for min-heap via reverse)
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ── HNSW Graph (core data structures) ───────────────────────────────────────

/// A single node in the HNSW graph.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct HnswNode {
    /// The vector data.
    vector: Vec<f32>,
    /// Neighbors at each layer. neighbors[0] = base layer.
    /// Base layer allows up to 2*M connections; upper layers allow up to M.
    neighbors: Vec<Vec<u64>>,
    /// The maximum layer this node appears in (0-indexed).
    level: usize,
}

/// Serializable snapshot of the HNSW graph (uses default HashMap hasher for serde).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct HnswGraphSnapshot {
    nodes: HashMap<u64, HnswNode>,
    entry_point: Option<u64>,
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    ml: f64,
}

/// The HNSW graph supporting incremental insertion.
/// Uses a fast identity hasher for u64 keys (avoids SipHash overhead).
#[derive(Clone, Debug)]
struct HnswGraph {
    /// All nodes indexed by their user-provided u64 ID.
    nodes: FastHashMap<HnswNode>,
    /// The entry point node ID (the node at the highest layer).
    entry_point: Option<u64>,
    /// Maximum number of connections per upper layer (M).
    m: usize,
    /// Maximum number of connections at layer 0 (2*M).
    m_max0: usize,
    /// Beam width during graph construction.
    ef_construction: usize,
    /// Level normalization factor: 1/ln(M).
    ml: f64,
}

impl HnswGraph {
    fn new(m: usize, ef_construction: usize) -> Self {
        Self {
            nodes: FastHashMap::default(),
            entry_point: None,
            m,
            m_max0: 2 * m,
            ef_construction,
            ml: 1.0 / (m.max(2) as f64).ln(),
        }
    }

    /// Convert to serializable snapshot.
    fn to_snapshot(&self) -> HnswGraphSnapshot {
        HnswGraphSnapshot {
            nodes: self.nodes.iter().map(|(&k, v)| (k, v.clone())).collect(),
            entry_point: self.entry_point,
            m: self.m,
            m_max0: self.m_max0,
            ef_construction: self.ef_construction,
            ml: self.ml,
        }
    }

    /// Restore from serializable snapshot.
    fn from_snapshot(snap: HnswGraphSnapshot) -> Self {
        Self {
            nodes: snap.nodes.into_iter().collect(),
            entry_point: snap.entry_point,
            m: snap.m,
            m_max0: snap.m_max0,
            ef_construction: snap.ef_construction,
            ml: snap.ml,
        }
    }

    /// Generate a random level for a new node.
    #[inline]
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen::<f64>().max(1e-15);
        (-r.ln() * self.ml).floor() as usize
    }

    /// Get the max level of the current entry point (or 0 if empty).
    #[inline]
    fn max_level(&self) -> usize {
        self.entry_point
            .and_then(|ep| self.nodes.get(&ep))
            .map(|node| node.level)
            .unwrap_or(0)
    }

    /// Get the maximum number of neighbors for the given layer.
    #[inline]
    fn max_neighbors(&self, layer: usize) -> usize {
        if layer == 0 { self.m_max0 } else { self.m }
    }

    /// Search within a single layer using beam search.
    /// Returns up to `ef` nearest neighbors found at this layer, sorted closest-first.
    fn search_layer(
        &self,
        query: &[f32],
        entry_ids: &[u64],
        ef: usize,
        layer: usize,
        metric: Metric,
    ) -> Vec<Candidate> {
        let mut visited = FastHashSet::with_capacity_and_hasher(ef * 2, Default::default());
        // Min-heap: pop closest candidate to explore
        let mut candidates = BinaryHeap::<std::cmp::Reverse<Candidate>>::new();
        // Max-heap: pop farthest result to maintain bounded result set
        let mut results = BinaryHeap::<Candidate>::new();

        // Initialize with entry points
        for &ep_id in entry_ids {
            if !visited.insert(ep_id) {
                continue;
            }
            if let Some(node) = self.nodes.get(&ep_id) {
                let d = metric.distance(query, &node.vector);
                let c = Candidate { distance: d, id: ep_id };
                candidates.push(std::cmp::Reverse(c));
                results.push(c);
            }
        }

        while let Some(std::cmp::Reverse(closest)) = candidates.pop() {
            // Stop if closest candidate is farther than the farthest result
            if let Some(farthest) = results.peek() {
                if closest.distance > farthest.distance && results.len() >= ef {
                    break;
                }
            }

            // Explore neighbors of closest candidate at this layer
            if let Some(node) = self.nodes.get(&closest.id) {
                if layer < node.neighbors.len() {
                    for &nb_id in &node.neighbors[layer] {
                        if !visited.insert(nb_id) {
                            continue;
                        }

                        if let Some(nb_node) = self.nodes.get(&nb_id) {
                            let d = metric.distance(query, &nb_node.vector);

                            let should_add = if results.len() < ef {
                                true
                            } else if let Some(farthest) = results.peek() {
                                d < farthest.distance
                            } else {
                                true
                            };

                            if should_add {
                                let c = Candidate { distance: d, id: nb_id };
                                candidates.push(std::cmp::Reverse(c));
                                results.push(c);
                                if results.len() > ef {
                                    results.pop(); // remove farthest
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert max-heap to sorted vec (closest first)
        let mut result_vec: Vec<Candidate> = results.into_vec();
        result_vec.sort_unstable();
        result_vec
    }

    /// Greedy search for the single closest node, descending from top layers.
    /// More efficient than search_layer with ef=1 for the descent phase.
    #[inline]
    fn greedy_closest(
        &self,
        query: &[f32],
        mut current: u64,
        layer: usize,
        metric: Metric,
    ) -> u64 {
        let mut current_dist = self.nodes.get(&current)
            .map(|n| metric.distance(query, &n.vector))
            .unwrap_or(f32::MAX);

        loop {
            let mut changed = false;
            if let Some(node) = self.nodes.get(&current) {
                if layer < node.neighbors.len() {
                    for &nb_id in &node.neighbors[layer] {
                        if let Some(nb_node) = self.nodes.get(&nb_id) {
                            let d = metric.distance(query, &nb_node.vector);
                            if d < current_dist {
                                current = nb_id;
                                current_dist = d;
                                changed = true;
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    /// Insert a vector into the graph incrementally.
    fn insert(&mut self, id: u64, vector: Vec<f32>, metric: Metric) {
        let new_level = self.random_level();

        // Create the node with empty neighbor lists
        let node = HnswNode {
            vector,
            neighbors: vec![Vec::new(); new_level + 1],
            level: new_level,
        };
        self.nodes.insert(id, node);

        // If this is the first node, set as entry point and return
        let ep_id = match self.entry_point {
            None => {
                self.entry_point = Some(id);
                return;
            }
            Some(ep) => ep,
        };

        let current_max_level = self.max_level();

        // Get the query vector reference
        let query = self.nodes[&id].vector.clone();

        // Phase 1: Greedy descent from top layer down to new_level + 1
        let mut current_ep = ep_id;
        if current_max_level > new_level {
            for layer in ((new_level + 1)..=current_max_level).rev() {
                current_ep = self.greedy_closest(&query, current_ep, layer, metric);
            }
        }

        // Phase 2: Insert at layers min(new_level, current_max_level) down to 0
        let insert_top = new_level.min(current_max_level);
        for layer in (0..=insert_top).rev() {
            let results = self.search_layer(
                &query,
                &[current_ep],
                self.ef_construction,
                layer,
                metric,
            );

            // Select neighbors for this layer
            let max_nb = self.max_neighbors(layer);
            let m_to_select = max_nb.min(results.len());
            let selected: Vec<u64> = results[..m_to_select].iter().map(|c| c.id).collect();

            // Set neighbors for the new node at this layer
            if let Some(node) = self.nodes.get_mut(&id) {
                if layer < node.neighbors.len() {
                    node.neighbors[layer] = selected.clone();
                }
            }

            // Add back-edges from selected neighbors to the new node
            let max_conn = self.max_neighbors(layer);
            for &nb_id in &selected {
                if nb_id == id { continue; }

                // Add the back-edge
                if let Some(nb_node) = self.nodes.get_mut(&nb_id) {
                    while nb_node.neighbors.len() <= layer {
                        nb_node.neighbors.push(Vec::new());
                    }
                    nb_node.neighbors[layer].push(id);
                }

                // Prune if too many connections (separate borrow scope)
                let needs_pruning = self.nodes.get(&nb_id)
                    .map(|n| layer < n.neighbors.len() && n.neighbors[layer].len() > max_conn)
                    .unwrap_or(false);

                if needs_pruning {
                    let (nb_vec, nb_neighbor_ids) = {
                        let nb_node = &self.nodes[&nb_id];
                        (nb_node.vector.as_slice() as *const [f32], nb_node.neighbors[layer].clone())
                    };

                    // SAFETY: nb_vec points to stable data in HashMap; we only mutate
                    // neighbors, not the vector itself, so the pointer remains valid.
                    let nb_vec_ref = unsafe { &*nb_vec };

                    let mut nb_with_dist: Vec<Candidate> = nb_neighbor_ids
                        .iter()
                        .filter_map(|&nid| {
                            self.nodes.get(&nid).map(|n| {
                                Candidate { id: nid, distance: metric.distance(nb_vec_ref, &n.vector) }
                            })
                        })
                        .collect();
                    nb_with_dist.sort_unstable_by(|a, b| a.cmp(b));
                    nb_with_dist.truncate(max_conn);

                    if let Some(nb_node) = self.nodes.get_mut(&nb_id) {
                        nb_node.neighbors[layer] = nb_with_dist.iter().map(|c| c.id).collect();
                    }
                }
            }

            // Update entry point for the next lower layer's search
            if let Some(c) = results.first() {
                current_ep = c.id;
            }
        }

        // Update entry point if new node has a higher level
        if new_level > current_max_level {
            self.entry_point = Some(id);
        }
    }

    /// Search for k nearest neighbors across all layers.
    fn search_knn(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        metric: Metric,
    ) -> Vec<(u64, f32)> {
        let ep_id = match self.entry_point {
            None => return Vec::new(),
            Some(ep) => ep,
        };

        let top = self.max_level();
        let ef = ef_search.max(k);

        // Phase 1: Greedy descent from top layer to layer 1
        let mut current_ep = ep_id;
        if top > 0 {
            for layer in (1..=top).rev() {
                current_ep = self.greedy_closest(query, current_ep, layer, metric);
            }
        }

        // Phase 2: Beam search at layer 0
        let results = self.search_layer(query, &[current_ep], ef, 0, metric);
        results.into_iter()
            .take(k)
            .map(|c| (c.id, c.distance))
            .collect()
    }

    /// Delete a node from the graph.
    fn delete(&mut self, id: u64, _metric: Metric) -> bool {
        let node = match self.nodes.remove(&id) {
            Some(n) => n,
            None => return false,
        };

        // Remove back-edges from all neighbors
        for (layer, neighbors) in node.neighbors.iter().enumerate() {
            for &nb_id in neighbors {
                if let Some(nb_node) = self.nodes.get_mut(&nb_id) {
                    if layer < nb_node.neighbors.len() {
                        nb_node.neighbors[layer].retain(|&nid| nid != id);
                    }
                }
            }
        }

        // If deleted node was the entry point, find a new one
        if self.entry_point == Some(id) {
            self.entry_point = self.nodes.keys()
                .max_by_key(|&&nid| self.nodes.get(&nid).map(|n| n.level).unwrap_or(0))
                .copied();
        }

        true
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
}

impl HnswIndex {
    pub fn new(dimensions: usize, metric: Metric, hnsw_config: HnswConfig) -> Self {
        let graph = HnswGraph::new(hnsw_config.m, hnsw_config.ef_construction);
        Self {
            config: IndexConfig { dimensions, metric },
            hnsw_config,
            graph,
        }
    }

    /// Kept for API compatibility; no longer has any effect since inserts are incremental.
    pub fn with_flush_threshold(self, _n: usize) -> Self {
        self
    }

    /// Save the index to a binary file at `path` (bincode format).
    ///
    /// The full graph (vectors + connectivity) is persisted.
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

    /// Load an index from a binary file previously written by [`save`].
    ///
    /// The full graph is restored — no rebuild needed.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, VectorDbError> {
        let file = std::fs::File::open(&path)?;
        let reader = BufReader::new(file);

        // Try new format first
        let result: Result<HnswIndexSnapshot, _> = bincode::deserialize_from(reader);
        if let Ok(snapshot) = result {
            return Ok(Self {
                config: IndexConfig {
                    dimensions: snapshot.dimensions,
                    metric: snapshot.metric,
                },
                hnsw_config: snapshot.hnsw_config,
                graph: HnswGraph::from_snapshot(snapshot.graph),
            });
        }

        // Fall back to legacy format (vectors only, needs incremental rebuild)
        let file = std::fs::File::open(&path)?;
        let reader = BufReader::new(file);
        let legacy: LegacyHnswIndexSnapshot = bincode::deserialize_from(reader)
            .map_err(|e| VectorDbError::Serialization(e.to_string()))?;

        let hnsw_config = legacy.hnsw_config;
        let metric = legacy.metric;
        let mut index = Self::new(legacy.dimensions, metric, hnsw_config);
        for (id, vec) in legacy.vectors {
            index.graph.insert(id, vec, metric);
        }
        Ok(index)
    }

    /// Persist the HNSW graph to `path` so that future loads can skip rebuild.
    /// With incremental HNSW, the graph is always up-to-date.
    pub fn save_graph(&self, path: &Path) -> Result<(), VectorDbError> {
        if self.graph.nodes.is_empty() {
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
        self.graph = HnswGraph::from_snapshot(snapshot);
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
        if self.graph.nodes.contains_key(&id) {
            return Err(VectorDbError::DuplicateId(id));
        }
        let metric = self.config.metric;
        self.graph.insert(id, vector.to_vec(), metric);
        Ok(())
    }

    fn add_batch(&mut self, entries: &[(u64, Vec<f32>)]) -> Result<(), VectorDbError> {
        // Validate all entries first
        for (id, vec) in entries {
            if vec.len() != self.config.dimensions {
                return Err(VectorDbError::DimensionMismatch {
                    expected: self.config.dimensions,
                    got: vec.len(),
                });
            }
            if self.graph.nodes.contains_key(id) {
                return Err(VectorDbError::DuplicateId(*id));
            }
        }
        let metric = self.config.metric;
        // Pre-allocate nodes capacity
        self.graph.nodes.reserve(entries.len());
        for (id, vec) in entries {
            self.graph.insert(*id, vec.clone(), metric);
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
        if k == 0 || self.graph.nodes.is_empty() {
            return Ok(vec![]);
        }

        let results = self.graph.search_knn(
            query,
            k,
            self.hnsw_config.ef_search,
            self.config.metric,
        );

        Ok(results
            .into_iter()
            .map(|(id, distance)| SearchResult { id, distance })
            .collect())
    }

    fn delete(&mut self, id: u64) -> bool {
        self.graph.delete(id, self.config.metric)
    }

    fn len(&self) -> usize {
        self.graph.nodes.len()
    }

    fn config(&self) -> &IndexConfig {
        &self.config
    }

    fn iter_vectors(&self) -> Box<dyn Iterator<Item = (u64, Vec<f32>)> + '_> {
        Box::new(
            self.graph.nodes.iter().map(|(&id, node)| (id, node.vector.clone()))
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
        // flush is a no-op now, but called for API compatibility
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
        let path = "/tmp/hnsw_index_test.json";
        idx.save(path).unwrap();
        let loaded = HnswIndex::load(path).unwrap();
        assert_eq!(loaded.len(), idx.len());
        assert_eq!(loaded.config().dimensions, idx.config().dimensions);
        assert_eq!(loaded.config().metric, idx.config().metric);
        // Nearest neighbour must still be correct after load
        let orig = idx.search(&[5.0, 0.0, 0.0], 1).unwrap();
        let from_disk = loaded.search(&[5.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(orig[0].id, from_disk[0].id);
    }

    #[test]
    fn fallback_before_flush() {
        // With incremental HNSW, search works immediately after insert (no flush needed).
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
        // All vectors point in the same direction — cosine distance ≈ 0 for any of them
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
        // Largest dot product with [1,0] → id=20 (vector [20,0], distance = -20)
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
        // With incremental HNSW, inserts build the graph immediately.
        // No flush threshold needed — search works after any number of inserts.
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
        let mut idx = make_index(); // 20 vectors
        assert!(idx.delete(10));
        idx.flush(); // no-op
        let r = idx.search(&[10.0, 0.0, 0.0], 1).unwrap();
        assert_ne!(r[0].id, 10); // deleted vector must not appear
    }

    #[test]
    fn add_batch_then_explicit_flush() {
        let cfg = HnswConfig { ef_construction: 100, ef_search: 20, m: 8 };
        let mut idx = HnswIndex::new(1, Metric::L2, cfg);
        let entries: Vec<(u64, Vec<f32>)> = (0..50u64).map(|i| (i, vec![i as f32])).collect();
        idx.add_batch(&entries).unwrap();
        idx.flush(); // no-op
        assert_eq!(idx.len(), 50);
        let r = idx.search(&[25.0], 1).unwrap();
        assert_eq!(r[0].id, 25);
    }

    #[test]
    fn save_with_staging_persists_all_vectors() {
        // Insert vectors and verify they persist through save/load.
        let cfg = HnswConfig::default();
        let mut idx = HnswIndex::new(1, Metric::L2, cfg);
        for i in 0..10u64 {
            idx.add(i, &[i as f32]).unwrap();
        }
        let path = "/tmp/hnsw_staging_test.json";
        idx.save(path).unwrap();
        let loaded = HnswIndex::load(path).unwrap();
        assert_eq!(loaded.len(), 10);
        let r = loaded.search(&[5.0], 1).unwrap();
        assert_eq!(r[0].id, 5);
    }
}
