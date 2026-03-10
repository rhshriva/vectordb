/// HNSW (Hierarchical Navigable Small World) approximate nearest-neighbour index.
///
/// Wraps the `instant-distance` crate which provides a pure-Rust, production-grade
/// HNSW implementation with auto-vectorized distance computation.
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
/// - HnswIndex: ~95–99% recall (tunable), O(log N · ef) query, no single-delete support
use std::collections::HashMap;
use std::io::{BufReader, BufWriter};

use crate::{
    distance::Metric,
    error::VectorDbError,
    index::{IndexConfig, SearchResult, VectorIndex},
};

#[derive(serde::Serialize, serde::Deserialize)]
struct HnswIndexSnapshot {
    dimensions: usize,
    metric: Metric,
    hnsw_config: HnswConfig,
    flush_threshold: usize,
    vectors: HashMap<u64, Vec<f32>>,
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

// ── instant-distance integration ─────────────────────────────────────────────
//
// instant-distance requires implementing the `instant_distance::Point` trait.
// We wrap our f32 vectors in a newtype so we can implement the foreign trait.

#[derive(Clone, Debug)]
struct HnswPoint(Vec<f32>, Metric);

impl instant_distance::Point for HnswPoint {
    fn distance(&self, other: &Self) -> f32 {
        self.1.distance(&self.0, &other.0)
    }
}

// ── HnswIndex ─────────────────────────────────────────────────────────────────

/// The HNSW index.
///
/// Note: `instant-distance` builds the graph in a single `build` call.
/// To support incremental inserts we maintain a staging buffer and rebuild
/// the graph when the buffer crosses a threshold or when `flush()` is called
/// explicitly.
pub struct HnswIndex {
    config: IndexConfig,
    hnsw_config: HnswConfig,

    /// Staging buffer for vectors not yet incorporated into the HNSW graph.
    /// New vectors are added here until `flush()` is called.
    staging: Vec<(u64, Vec<f32>)>,

    /// All vectors (staging + committed), kept for rebuilds and fallback search.
    all_vectors: HashMap<u64, Vec<f32>>,

    /// Auto-flush threshold: rebuild graph when staging reaches this size.
    flush_threshold: usize,

    /// Built HNSW graph + id mapping (rebuilt on flush).
    built: Option<BuiltHnsw>,
}

struct BuiltHnsw {
    hnsw: instant_distance::HnswMap<HnswPoint, u64>,
}

impl HnswIndex {
    pub fn new(dimensions: usize, metric: Metric, hnsw_config: HnswConfig) -> Self {
        Self {
            config: IndexConfig { dimensions, metric },
            hnsw_config,
            staging: Vec::new(),
            all_vectors: HashMap::new(),
            flush_threshold: 1000,
            built: None,
        }
    }

    pub fn with_flush_threshold(mut self, n: usize) -> Self {
        self.flush_threshold = n;
        self
    }

    /// Force rebuild of the HNSW graph from all stored vectors.
    ///
    /// Call this after a large batch of inserts to get ANN search quality.
    /// Complexity: O(N · M · log N).
    pub fn flush(&mut self) {
        self.staging.clear();
        if self.all_vectors.is_empty() {
            self.built = None;
            return;
        }

        let metric = self.config.metric;
        let points: Vec<HnswPoint> = self
            .all_vectors
            .values()
            .map(|v| HnswPoint(v.clone(), metric))
            .collect();
        let values: Vec<u64> = self.all_vectors.keys().copied().collect();

        let builder = instant_distance::Builder::default()
            .ef_construction(self.hnsw_config.ef_construction)
            .ef_search(self.hnsw_config.ef_search)
            .ml(1.0 / (self.hnsw_config.m as f32).ln());

        let hnsw = builder.build(points, values);
        self.built = Some(BuiltHnsw { hnsw });
    }

    /// Save the index to a JSON file at `path`.
    ///
    /// The HNSW graph itself is not persisted — it is rebuilt from the stored
    /// vectors when [`load`] calls [`flush`] after deserializing.
    pub fn save(&self, path: &str) -> Result<(), VectorDbError> {
        let file = std::fs::File::create(path)?;
        let writer = BufWriter::new(file);
        let snapshot = HnswIndexSnapshot {
            dimensions: self.config.dimensions,
            metric: self.config.metric,
            hnsw_config: self.hnsw_config.clone(),
            flush_threshold: self.flush_threshold,
            vectors: self.all_vectors.clone(),
        };
        serde_json::to_writer(writer, &snapshot)?;
        Ok(())
    }

    /// Load an index from a JSON file previously written by [`save`].
    ///
    /// The HNSW graph is rebuilt immediately via [`flush`], so the returned
    /// index is ready for ANN search.
    pub fn load(path: &str) -> Result<Self, VectorDbError> {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);
        let snapshot: HnswIndexSnapshot = serde_json::from_reader(reader)?;
        let mut index = Self {
            config: IndexConfig {
                dimensions: snapshot.dimensions,
                metric: snapshot.metric,
            },
            hnsw_config: snapshot.hnsw_config,
            staging: Vec::new(),
            all_vectors: snapshot.vectors,
            flush_threshold: snapshot.flush_threshold,
            built: None,
        };
        index.flush();
        Ok(index)
    }

    /// Set the `ef_search` parameter at runtime (no rebuild needed).
    pub fn set_ef_search(&mut self, ef: usize) {
        self.hnsw_config.ef_search = ef;
    }

    fn maybe_flush(&mut self) {
        if self.staging.len() >= self.flush_threshold {
            self.flush();
        }
    }

    /// Search the staging buffer with brute force (used as fallback when graph
    /// hasn't been built yet or to supplement ANN results).
    fn search_staging(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let metric = self.config.metric;
        let mut results: Vec<SearchResult> = self
            .staging
            .iter()
            .map(|(id, vec)| SearchResult {
                id: *id,
                distance: metric.distance(query, vec),
            })
            .collect();
        results.sort_unstable_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        results
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
        if self.all_vectors.contains_key(&id) {
            return Err(VectorDbError::DuplicateId(id));
        }
        self.all_vectors.insert(id, vector.to_vec());
        self.staging.push((id, vector.to_vec()));
        self.maybe_flush();
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

        // If graph isn't built yet, fall back to brute-force over all vectors.
        let built = match &self.built {
            Some(b) => b,
            None => {
                // Brute force over all_vectors
                let metric = self.config.metric;
                let mut results: Vec<SearchResult> = self
                    .all_vectors
                    .iter()
                    .map(|(&id, vec)| SearchResult {
                        id,
                        distance: metric.distance(query, vec),
                    })
                    .collect();
                results.sort_unstable_by(|a, b| {
                    a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
                });
                results.truncate(k);
                return Ok(results);
            }
        };

        let query_point = HnswPoint(query.to_vec(), self.config.metric);
        let mut search = instant_distance::Search::default();

        let ann_results: Vec<SearchResult> = built
            .hnsw
            .search(&query_point, &mut search)
            .take(k)
            .map(|item| SearchResult {
                id: *item.value,
                distance: item.distance,
            })
            .collect();

        // Merge with staging buffer results for freshness.
        if self.staging.is_empty() {
            return Ok(ann_results);
        }
        let staging_results = self.search_staging(query, k);
        let mut merged = ann_results;
        for r in staging_results {
            if !merged.iter().any(|x| x.id == r.id) {
                merged.push(r);
            }
        }
        merged.sort_unstable_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(k);
        Ok(merged)
    }

    fn delete(&mut self, id: u64) -> bool {
        let removed = self.all_vectors.remove(&id).is_some();
        if removed {
            self.staging.retain(|(sid, _)| *sid != id);
            // Mark graph as stale — rebuild on next flush/search.
            self.built = None;
        }
        removed
    }

    fn len(&self) -> usize {
        self.all_vectors.len()
    }

    fn config(&self) -> &IndexConfig {
        &self.config
    }

    fn iter_vectors(&self) -> Box<dyn Iterator<Item = (u64, Vec<f32>)> + '_> {
        Box::new(self.all_vectors.iter().map(|(&id, v)| (id, v.clone())))
    }

    fn flush(&mut self) {
        HnswIndex::flush(self);
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
        let path = "/tmp/hnsw_index_test.json";
        idx.save(path).unwrap();
        let loaded = HnswIndex::load(path).unwrap();
        assert_eq!(loaded.len(), idx.len());
        assert_eq!(loaded.config().dimensions, idx.config().dimensions);
        assert_eq!(loaded.config().metric, idx.config().metric);
        // Nearest neighbour must still be correct after load+rebuild
        let orig = idx.search(&[5.0, 0.0, 0.0], 1).unwrap();
        let from_disk = loaded.search(&[5.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(orig[0].id, from_disk[0].id);
    }

    #[test]
    fn fallback_before_flush() {
        // Without flush the index falls back to brute force — still correct.
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
        let cfg = HnswConfig::default();
        let mut idx = HnswIndex::new(1, Metric::L2, cfg).with_flush_threshold(5);
        for i in 0..5u64 {
            idx.add(i, &[i as f32]).unwrap();
        }
        // After 5 inserts the graph should be built automatically
        assert_eq!(idx.len(), 5);
        let r = idx.search(&[2.0], 1).unwrap();
        assert_eq!(r[0].id, 2);
    }

    #[test]
    fn delete_then_flush_restores_search() {
        let mut idx = make_index(); // 20 vectors, already flushed
        assert!(idx.delete(10));
        idx.flush();
        let r = idx.search(&[10.0, 0.0, 0.0], 1).unwrap();
        assert_ne!(r[0].id, 10); // deleted vector must not appear
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
        // Insert fewer than flush_threshold so staging buffer is non-empty at save time.
        let cfg = HnswConfig::default();
        let mut idx = HnswIndex::new(1, Metric::L2, cfg).with_flush_threshold(1000);
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
