//! High-level embedded API for Quiver.
//!
//! This is the primary entry point for using Quiver as an embedded library
//! directly in your Rust application — no server, no HTTP, no async runtime
//! required.
//!
//! # Quick start
//! ```no_run
//! use quiver_core::db::Quiver;
//! use quiver_core::Metric;
//!
//! // Open (or create) a database at a local path.
//! let mut db = Quiver::open("./my_data").unwrap();
//!
//! // Create a collection.
//! db.create_collection("docs", 3, Metric::Cosine).unwrap();
//!
//! // Insert vectors.
//! db.upsert("docs", 1, &[0.1, 0.2, 0.3], None).unwrap();
//! db.upsert("docs", 2, &[0.9, 0.8, 0.7], None).unwrap();
//!
//! // Search for nearest neighbours.
//! let hits = db.search("docs", &[0.1, 0.2, 0.3], 5).unwrap();
//! for hit in hits {
//!     println!("id={} distance={:.4}", hit.id, hit.distance);
//! }
//! ```

use std::path::Path;
use std::sync::Arc;

use crate::{
    collection::{CollectionMeta, CollectionSearchResult, IndexType},
    distance::Metric,
    embedder::TextEmbedder,
    error::VectorDbError,
    index::hnsw::HnswConfig,
    manager::CollectionManager,
    payload::FilterCondition,
};

/// The top-level embedded Quiver database handle.
///
/// Wraps a [`CollectionManager`] and exposes a flat, ergonomic API so you
/// never need to think about the manager/collection split.
pub struct Quiver {
    manager: CollectionManager,
}

impl Quiver {
    /// Open (or create) a Quiver database at `path`.
    ///
    /// If the directory does not exist it is created. Any collections
    /// persisted in previous sessions are automatically loaded.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, VectorDbError> {
        let manager = CollectionManager::open(path)?;
        Ok(Self { manager })
    }

    // ── Collection management ─────────────────────────────────────────────────

    /// Create a new collection with the given name, vector dimensions and
    /// distance metric. Defaults to an HNSW index with auto-promotion disabled.
    ///
    /// Returns an error if a collection with the same name already exists.
    pub fn create_collection(
        &mut self,
        name: &str,
        dimensions: usize,
        metric: Metric,
    ) -> Result<(), VectorDbError> {
        let meta = CollectionMeta {
            name: name.to_string(),
            dimensions,
            metric,
            index_type: IndexType::Hnsw,
            hnsw_config: Some(HnswConfig::default()),
            wal_compact_threshold: 50_000,
            auto_promote_threshold: None,
            promotion_hnsw_config: None,
            embedding_model: None,
            faiss_factory: None,
        };
        self.manager.create_collection(meta)
    }

    /// Create a collection if it does not exist, or return without error if it does.
    pub fn get_or_create_collection(
        &mut self,
        name: &str,
        dimensions: usize,
        metric: Metric,
    ) -> Result<(), VectorDbError> {
        if self.manager.get_collection(name).is_some() {
            return Ok(());
        }
        self.create_collection(name, dimensions, metric)
    }

    /// Delete a collection and all its data. Returns `false` if not found.
    pub fn delete_collection(&mut self, name: &str) -> Result<bool, VectorDbError> {
        self.manager.delete_collection(name)
    }

    /// List the names of all collections.
    pub fn list_collections(&self) -> Vec<String> {
        self.manager.list_collections().iter().map(|m| m.name.clone()).collect()
    }

    /// Return the number of vectors in a collection.
    pub fn count(&self, collection: &str) -> Result<usize, VectorDbError> {
        let col = self.manager.get_collection(collection)
            .ok_or_else(|| VectorDbError::CollectionNotFound(collection.to_string()))?;
        Ok(col.count())
    }

    // ── Vector operations ─────────────────────────────────────────────────────

    /// Insert or update a vector by `id`.
    ///
    /// `payload` is an arbitrary JSON value attached to the vector and returned
    /// with search results. Pass `None` for no payload.
    pub fn upsert(
        &mut self,
        collection: &str,
        id: u64,
        vector: &[f32],
        payload: Option<serde_json::Value>,
    ) -> Result<(), VectorDbError> {
        let col = self.manager.get_collection_mut(collection)
            .ok_or_else(|| VectorDbError::CollectionNotFound(collection.to_string()))?;
        col.upsert(id, vector.to_vec(), payload)
    }

    /// Delete a vector by `id`. Returns `true` if the vector existed.
    pub fn delete(&mut self, collection: &str, id: u64) -> Result<bool, VectorDbError> {
        let col = self.manager.get_collection_mut(collection)
            .ok_or_else(|| VectorDbError::CollectionNotFound(collection.to_string()))?;
        col.delete(id)
    }

    // ── Search ────────────────────────────────────────────────────────────────

    /// Find the `k` nearest neighbours of `query`.
    pub fn search(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<CollectionSearchResult>, VectorDbError> {
        let col = self.manager.get_collection(collection)
            .ok_or_else(|| VectorDbError::CollectionNotFound(collection.to_string()))?;
        col.search(query, k, None)
    }

    /// Find the `k` nearest neighbours of `query`, restricted to vectors whose
    /// payload matches `filter`.
    pub fn search_filtered(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
        filter: &FilterCondition,
    ) -> Result<Vec<CollectionSearchResult>, VectorDbError> {
        let col = self.manager.get_collection(collection)
            .ok_or_else(|| VectorDbError::CollectionNotFound(collection.to_string()))?;
        col.search(query, k, Some(filter))
    }

    // ── Text embedding helpers ─────────────────────────────────────────────────

    /// Attach a [`TextEmbedder`] to `collection`, enabling [`upsert_text`] and
    /// [`search_text`].
    ///
    /// The embedder is **not persisted** — re-attach it after each process
    /// restart.
    ///
    /// [`upsert_text`]: Quiver::upsert_text
    /// [`search_text`]: Quiver::search_text
    pub fn set_embedder(
        &mut self,
        collection: &str,
        embedder: Arc<dyn TextEmbedder>,
    ) -> Result<(), VectorDbError> {
        let col = self.manager.get_collection_mut(collection)
            .ok_or_else(|| VectorDbError::CollectionNotFound(collection.to_string()))?;
        col.set_embedder(embedder)
    }

    /// Embed `text` using the collection's attached embedder, then upsert the
    /// resulting vector under `id`.
    ///
    /// Returns [`VectorDbError::NoEmbedder`] if [`set_embedder`] has not been
    /// called for this collection.
    ///
    /// [`set_embedder`]: Quiver::set_embedder
    pub fn upsert_text(
        &mut self,
        collection: &str,
        id: u64,
        text: &str,
        payload: Option<serde_json::Value>,
    ) -> Result<(), VectorDbError> {
        let col = self.manager.get_collection_mut(collection)
            .ok_or_else(|| VectorDbError::CollectionNotFound(collection.to_string()))?;
        col.upsert_text(id, text, payload)
    }

    /// Embed `text` using the collection's attached embedder, then return the
    /// `k` nearest neighbours.
    ///
    /// Returns [`VectorDbError::NoEmbedder`] if [`set_embedder`] has not been
    /// called for this collection.
    ///
    /// [`set_embedder`]: Quiver::set_embedder
    pub fn search_text(
        &self,
        collection: &str,
        text: &str,
        k: usize,
    ) -> Result<Vec<CollectionSearchResult>, VectorDbError> {
        let col = self.manager.get_collection(collection)
            .ok_or_else(|| VectorDbError::CollectionNotFound(collection.to_string()))?;
        col.search_text(text, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn open_db(dir: &tempfile::TempDir) -> Quiver {
        Quiver::open(dir.path()).unwrap()
    }

    #[test]
    fn create_and_list_collections() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("a", 3, Metric::L2).unwrap();
        db.create_collection("b", 3, Metric::Cosine).unwrap();
        let mut names = db.list_collections();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn upsert_and_search() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("vecs", 3, Metric::L2).unwrap();
        db.upsert("vecs", 1, &[1.0, 0.0, 0.0], None).unwrap();
        db.upsert("vecs", 2, &[0.0, 1.0, 0.0], None).unwrap();
        db.upsert("vecs", 3, &[0.0, 0.0, 1.0], None).unwrap();

        let hits = db.search("vecs", &[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].id, 1);
        assert!(hits[0].distance < 1e-4);
    }

    #[test]
    fn upsert_with_payload_returned_in_search() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("docs", 2, Metric::Cosine).unwrap();
        db.upsert("docs", 42, &[1.0, 0.0], Some(serde_json::json!({"title": "hello"}))).unwrap();

        let hits = db.search("docs", &[1.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].id, 42);
        assert_eq!(hits[0].payload.as_ref().unwrap()["title"], "hello");
    }

    #[test]
    fn delete_vector() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("v", 2, Metric::L2).unwrap();
        db.upsert("v", 1, &[1.0, 0.0], None).unwrap();
        db.upsert("v", 2, &[0.0, 1.0], None).unwrap();
        assert!(db.delete("v", 1).unwrap());
        assert_eq!(db.count("v").unwrap(), 1);
    }

    #[test]
    fn get_or_create_is_idempotent() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.get_or_create_collection("x", 4, Metric::DotProduct).unwrap();
        db.get_or_create_collection("x", 4, Metric::DotProduct).unwrap(); // must not error
        assert_eq!(db.list_collections().len(), 1);
    }

    #[test]
    fn delete_collection() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("tmp", 2, Metric::L2).unwrap();
        assert!(db.delete_collection("tmp").unwrap());
        assert!(db.list_collections().is_empty());
    }

    #[test]
    fn persists_across_reopen() {
        let dir = tempdir().unwrap();
        {
            let mut db = open_db(&dir);
            db.create_collection("p", 2, Metric::L2).unwrap();
            db.upsert("p", 1, &[1.0, 0.0], None).unwrap();
            db.upsert("p", 2, &[0.0, 1.0], None).unwrap();
        }
        // Re-open — data must survive.
        let db = open_db(&dir);
        assert_eq!(db.count("p").unwrap(), 2);
        let hits = db.search("p", &[1.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].id, 1);
    }

    #[test]
    fn search_unknown_collection_errors() {
        let dir = tempdir().unwrap();
        let db = open_db(&dir);
        let result = db.search("ghost", &[0.0, 1.0], 5);
        assert!(matches!(result, Err(VectorDbError::CollectionNotFound(_))));
    }

    #[test]
    fn search_filtered_by_payload() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("f", 2, Metric::L2).unwrap();
        db.upsert("f", 1, &[1.0, 0.0], Some(serde_json::json!({"cat": "a"}))).unwrap();
        db.upsert("f", 2, &[1.0, 0.1], Some(serde_json::json!({"cat": "b"}))).unwrap();

        let filter: FilterCondition = serde_json::from_value(
            serde_json::json!({"cat": {"$eq": "a"}})
        ).unwrap();
        let hits = db.search_filtered("f", &[1.0, 0.0], 5, &filter).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 1);
    }
}
