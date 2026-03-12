//! Synchronous text-embedding abstraction used by [`Collection`].
//!
//! Keeping this trait in `quiver-core` (rather than `quiver-embeddings`) lets
//! the core crate stay free of async runtimes and HTTP clients.  The async
//! providers in `quiver-embeddings` bridge to this trait via
//! `quiver_embeddings::BlockingEmbedder`.
//!
//! # Attaching an embedder to a collection
//!
//! ```no_run
//! use std::sync::Arc;
//! use quiver_core::{Quiver, Metric, embedder::TextEmbedder};
//!
//! // Any type that implements TextEmbedder can be used.
//! // In practice you'd use quiver_embeddings::BlockingEmbedder.
//! struct MyEmbedder;
//! impl TextEmbedder for MyEmbedder {
//!     fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
//!         // ... call your embedding model ...
//!         Ok(vec![0.0; 384])
//!     }
//!     fn dimensions(&self) -> Option<usize> { Some(384) }
//! }
//!
//! let mut db = Quiver::open("./data").unwrap();
//! db.create_collection("docs", 384, Metric::Cosine).unwrap();
//! db.set_embedder("docs", Arc::new(MyEmbedder)).unwrap();
//!
//! db.upsert_text("docs", 1, "Hello, world!", None).unwrap();
//! let hits = db.search_text("docs", "greeting", 5).unwrap();
//! ```

use std::sync::Arc;

/// Trait for **synchronous** text-to-vector embedding.
///
/// Implementations must be `Send + Sync` so they can be stored in a
/// `Collection` and called from any thread.
///
/// # Implementing
///
/// The simplest adapter wraps an async provider in a `std::thread::spawn` +
/// dedicated Tokio runtime (see `quiver_embeddings::BlockingEmbedder`).
pub trait TextEmbedder: Send + Sync {
    /// Convert `text` into a dense float vector.
    ///
    /// Returns `Err` on network failures, model errors, or configuration issues.
    fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>>;

    /// Number of dimensions produced by this model, or `None` if not known
    /// until the first call.  Used to validate the embedder against the
    /// collection's configured dimensionality.
    fn dimensions(&self) -> Option<usize> {
        None
    }
}

/// A cheaply clonable, heap-allocated reference to a [`TextEmbedder`].
pub type EmbedderRef = Arc<dyn TextEmbedder>;
