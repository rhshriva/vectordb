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
