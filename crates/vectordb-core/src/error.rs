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
}
