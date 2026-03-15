pub mod flat;
pub mod hnsw;
pub mod quantized_flat;
pub mod quantized_fp16;
pub mod pq;
pub mod ivf;
pub mod ivf_pq;
pub mod binary_flat;
pub mod mmap_flat;
pub mod sparse;
#[cfg(feature = "faiss")]
pub mod faiss;

use std::path::Path;

use crate::{distance::Metric, error::VectorDbError};
use serde::{Deserialize, Serialize};

/// A single search result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    /// Caller-supplied ID for the vector.
    pub id: u64,
    /// Distance from the query vector (lower = more similar).
    pub distance: f32,
}

/// Configuration shared by all index types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Number of dimensions every vector must have.
    pub dimensions: usize,
    /// Distance metric used by this index.
    pub metric: Metric,
}

/// The unified interface every index implementation must satisfy.
///
/// All methods are synchronous; callers that need async should wrap
/// the index in a `tokio::task::spawn_blocking` call.
pub trait VectorIndex: Send + Sync {
    /// Add a single vector with the given ID.
    /// Returns an error if the vector has the wrong dimension or the ID already exists.
    fn add(&mut self, id: u64, vector: &[f32]) -> Result<(), VectorDbError>;

    /// Add multiple vectors in one call (may be more efficient for batch builds).
    fn add_batch(&mut self, entries: &[(u64, Vec<f32>)]) -> Result<(), VectorDbError> {
        for (id, vec) in entries {
            self.add(*id, vec)?;
        }
        Ok(())
    }

    /// Search for the `k` nearest neighbours of `query`.
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, VectorDbError>;

    /// Remove a vector by ID. Returns `true` if found and removed, `false` if not found.
    fn delete(&mut self, id: u64) -> bool;

    /// Number of vectors currently stored.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Index configuration (dimensions, metric).
    fn config(&self) -> &IndexConfig;

    /// Iterate over all (id, vector) pairs currently in the index.
    /// Used during WAL compaction to snapshot live state.
    fn iter_vectors(&self) -> Box<dyn Iterator<Item = (u64, Vec<f32>)> + '_>;

    /// Rebuild internal graph/structures from current vectors.
    /// Default is a no-op; HNSW overrides this to rebuild the graph.
    fn flush(&mut self) {}

    /// Persist the built graph structure to `path`.
    /// Default is a no-op; HnswIndex overrides this to write the built graph.
    fn save_graph(&self, _path: &Path) -> Result<(), VectorDbError> {
        Ok(())
    }

    /// Load a previously saved graph from `path`, skipping a rebuild.
    /// Default falls back to `flush()`; HnswIndex overrides to deserialize.
    fn load_graph_mmap(&mut self, path: &Path) -> Result<(), VectorDbError> {
        // Default: check if graph file exists and skip flush if not needed
        let _ = path;
        self.flush();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::flat::FlatIndex;
    use tempfile::tempdir;

    #[test]
    fn search_result_clone_and_eq() {
        let r1 = SearchResult { id: 1, distance: 0.5 };
        let r2 = r1.clone();
        assert_eq!(r1, r2);
        assert_eq!(r1.id, 1);
        assert_eq!(r1.distance, 0.5);
    }

    #[test]
    fn search_result_debug() {
        let r = SearchResult { id: 42, distance: 1.23 };
        let dbg = format!("{:?}", r);
        assert!(dbg.contains("42"));
    }

    #[test]
    fn index_config_debug_and_clone() {
        let cfg = IndexConfig { dimensions: 3, metric: Metric::Cosine };
        let cfg2 = cfg.clone();
        assert_eq!(cfg.dimensions, cfg2.dimensions);
        let _ = format!("{:?}", cfg);
    }

    #[test]
    fn default_add_batch_delegates_to_add() {
        let mut idx = FlatIndex::new(3, Metric::L2);
        let entries = vec![
            (1, vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0]),
        ];
        idx.add_batch(&entries).unwrap();
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn is_empty_when_no_vectors() {
        let idx = FlatIndex::new(3, Metric::L2);
        assert!(idx.is_empty());
    }

    #[test]
    fn is_empty_false_after_add() {
        let mut idx = FlatIndex::new(3, Metric::L2);
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        assert!(!idx.is_empty());
    }

    #[test]
    fn default_flush_is_noop() {
        let mut idx = FlatIndex::new(3, Metric::L2);
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.flush(); // should not panic or change state
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn default_save_graph_is_noop() {
        let idx = FlatIndex::new(3, Metric::L2);
        let dir = tempdir().unwrap();
        let result = idx.save_graph(dir.path());
        assert!(result.is_ok());
    }

    #[test]
    fn default_load_graph_mmap_calls_flush() {
        let mut idx = FlatIndex::new(3, Metric::L2);
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        let dir = tempdir().unwrap();
        let result = idx.load_graph_mmap(dir.path());
        assert!(result.is_ok());
        // Index should still be intact after load_graph_mmap
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn config_returns_correct_values() {
        let idx = FlatIndex::new(4, Metric::DotProduct);
        let cfg = idx.config();
        assert_eq!(cfg.dimensions, 4);
        assert_eq!(cfg.metric, Metric::DotProduct);
    }
}
