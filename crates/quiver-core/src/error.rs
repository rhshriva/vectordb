use thiserror::Error;

#[derive(Debug, Error)]
pub enum VectorDbError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("duplicate vector ID: {0}")]
    DuplicateId(u64),

    #[error("vector ID not found: {0}")]
    NotFound(u64),

    #[error("index is empty")]
    EmptyIndex,

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Serialization/deserialization error (JSON, bincode, etc.)
    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("collection already exists: {0}")]
    CollectionAlreadyExists(String),

    #[error("collection not found: {0}")]
    CollectionNotFound(String),

    #[error("WAL corruption at entry {entry}: {reason}")]
    WalCorruption { entry: usize, reason: String },

    /// Returned when a text embedding call fails (network, API, or model error).
    #[error("embedding error: {0}")]
    EmbeddingError(String),

    /// Returned when `upsert_text` or `search_text` is called but no embedder
    /// has been attached to the collection via `set_embedder`.
    #[error("no embedder configured for collection '{0}'; call set_embedder first")]
    NoEmbedder(String),

    #[error("snapshot already exists: {0}")]
    SnapshotAlreadyExists(String),

    #[error("snapshot not found: {0}")]
    SnapshotNotFound(String),
}

impl From<serde_json::Error> for VectorDbError {
    fn from(e: serde_json::Error) -> Self {
        VectorDbError::Serialization(e.to_string())
    }
}
