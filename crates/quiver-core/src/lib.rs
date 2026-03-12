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
pub use index::quantized_flat::QuantizedFlatIndex;
pub use index::ivf::{IvfConfig, IvfIndex};
pub use index::mmap_flat::MmapFlatIndex;

pub use collection::{Collection, CollectionMeta, CollectionSearchResult, IndexType};
pub use manager::CollectionManager;
pub use registry::IndexRegistry;
pub use payload::{FieldFilter, FieldOp, FilterCondition, Payload, matches_filter};
pub use wal::{Wal, WalEntry};
pub use embedder::{EmbedderRef, TextEmbedder};

/// Top-level embedded API — the main entry point for library users.
pub use db::Quiver;
