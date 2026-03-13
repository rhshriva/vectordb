pub mod collection;
pub mod db;
pub mod distance;
pub mod embedder;
pub mod error;
pub mod index;
pub mod manager;
pub mod payload;
pub mod registry;
pub mod wal;

pub use distance::Metric;
pub use error::VectorDbError;
pub use index::{IndexConfig, SearchResult, VectorIndex};
pub use index::{flat::FlatIndex, hnsw::{HnswConfig, HnswIndex}};
pub use index::binary_flat::BinaryFlatIndex;
pub use index::quantized_flat::QuantizedFlatIndex;
pub use index::quantized_fp16::Fp16FlatIndex;
pub use index::pq::{PqCodebook, PqCode, PqConfig};
pub use index::ivf::{IvfConfig, IvfIndex};
pub use index::ivf_pq::{IvfPqConfig, IvfPqIndex};
pub use index::mmap_flat::MmapFlatIndex;
pub use index::sparse::{SparseIndex, SparseVector, SparseSearchResult};

pub use collection::{Collection, CollectionMeta, CollectionSearchResult, HybridSearchResult, IndexType, SnapshotMeta};
pub use manager::CollectionManager;
pub use registry::IndexRegistry;
pub use payload::{FieldFilter, FieldOp, FilterCondition, Payload, matches_filter};
pub use wal::{Wal, WalEntry};
pub use embedder::{EmbedderRef, TextEmbedder};

/// Top-level embedded API — the main entry point for library users.
pub use db::Quiver;
