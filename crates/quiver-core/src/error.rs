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

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("collection already exists: {0}")]
    CollectionAlreadyExists(String),

    #[error("collection not found: {0}")]
    CollectionNotFound(String),

    #[error("WAL corruption at entry {entry}: {reason}")]
    WalCorruption { entry: usize, reason: String },
}
