//! Synchronous (blocking) wrapper around async [`EmbeddingProvider`] implementations.
//!
//! Bridges the async `EmbeddingProvider` trait into the synchronous
//! `quiver_core::embedder::TextEmbedder` trait expected by `Collection`.
//!
//! # How it works
//!
//! Each call to [`BlockingEmbedder::embed`] spawns a fresh OS thread that owns
//! a single-threaded Tokio runtime, drives the async provider to completion,
//! and returns the result via the thread's join handle.
//!
//! This approach is safe whether or not the caller is already inside a Tokio
//! context (avoiding the "cannot call `block_on` inside an async context"
//! error) and requires no `unsafe` code.
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use quiver_embeddings::{OllamaProvider, BlockingEmbedder};
//! use quiver_core::{Quiver, Metric};
//!
//! let provider = OllamaProvider::new("http://localhost:11434", "nomic-embed-text");
//! let embedder = Arc::new(BlockingEmbedder::new(provider));
//!
//! let mut db = Quiver::open("./data").unwrap();
//! db.create_collection("docs", 768, Metric::Cosine).unwrap();
//! db.set_embedder("docs", embedder).unwrap();
//!
//! db.upsert_text("docs", 1, "Hello, world!", None).unwrap();
//! let hits = db.search_text("docs", "greeting", 5).unwrap();
//! ```

use std::sync::Arc;

use quiver_core::embedder::TextEmbedder;

use crate::EmbeddingProvider;

/// Wraps any [`EmbeddingProvider`] and exposes a synchronous [`TextEmbedder`]
/// interface by running the async call in a dedicated OS thread.
pub struct BlockingEmbedder {
    provider: Arc<dyn EmbeddingProvider>,
}

impl BlockingEmbedder {
    /// Wrap any `EmbeddingProvider` in a blocking adapter.
    pub fn new(provider: impl EmbeddingProvider + 'static) -> Self {
        Self { provider: Arc::new(provider) }
    }
}

impl TextEmbedder for BlockingEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        let provider = Arc::clone(&self.provider);
        let text = text.to_string();

        // Spawn a dedicated OS thread with its own single-threaded Tokio runtime.
        // This is safe whether called from sync or async context.
        std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to build tokio runtime")
                .block_on(provider.embed(text))
        })
        .join()
        .map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
            Box::from("embedding thread panicked")
        })?
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }

    fn dimensions(&self) -> Option<usize> {
        self.provider.dimensions()
    }
}
