pub mod collection;
pub mod distance;
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

pub use collection::{Collection, CollectionMeta, CollectionSearchResult, IndexType};
pub use manager::CollectionManager;
pub use registry::IndexRegistry;
pub use payload::{FieldFilter, FieldOp, FilterCondition, Payload, matches_filter};
pub use wal::{Wal, WalEntry};
