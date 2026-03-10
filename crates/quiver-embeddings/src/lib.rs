//! Embedding model integrations for Quiver.
//!
//! Provides the [`EmbeddingProvider`] trait and concrete implementations for
//! hosted models (OpenAI, Ollama) so that the server can accept raw text and
//! automatically convert it to a vector before inserting or searching.
//!
//! # Example
//! ```no_run
//! use quiver_embeddings::{EmbeddingProvider, OllamaProvider};
//!
//! # async fn run() -> Result<(), quiver_embeddings::EmbeddingError> {
//! let provider = OllamaProvider::new("http://localhost:11434", "nomic-embed-text");
//! let vec = provider.embed("Hello, world!".to_string()).await?;
//! println!("dims: {}", vec.len());
//! # Ok(())
//! # }
//! ```

pub mod openai;
pub mod ollama;
pub mod blocking;

use async_trait::async_trait;

// Re-exports
pub use openai::OpenAiProvider;
pub use ollama::OllamaProvider;
pub use blocking::BlockingEmbedder;

/// Errors produced by embedding providers.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error ({status}): {message}")]
    Api { status: u16, message: String },

    #[error("Unexpected response format: {0}")]
    Format(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

/// A provider that converts text into a dense float vector.
///
/// Implementations are expected to be cheap to clone and shareable across
/// threads (i.e. should be `Send + Sync`).
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// The model identifier, e.g. `"text-embedding-3-small"` or `"nomic-embed-text"`.
    fn model_id(&self) -> &str;

    /// The number of dimensions in the output vectors.
    /// Returns `None` when the dimensionality is not known until after the first
    /// call (e.g. providers that support multiple models).
    fn dimensions(&self) -> Option<usize>;

    /// Embed a single text input. Prefer [`embed_batch`] for bulk work.
    async fn embed(&self, text: String) -> Result<Vec<f32>, EmbeddingError>;

    /// Embed multiple texts in a single API call where supported.
    /// The default implementation calls [`embed`] in sequence.
    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.embed(t).await?);
        }
        Ok(out)
    }
}

/// Lightweight metadata about a registered provider.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProviderInfo {
    pub model_id: String,
    pub provider_type: String,
    pub dimensions: Option<usize>,
}
