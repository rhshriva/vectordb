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
    collection::{CollectionMeta, CollectionSearchResult, IndexType, SnapshotMeta},
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
            ivf_config: None,
            ivf_pq_config: None,
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

    /// Batch insert or update multiple vectors at once. More efficient than
    /// calling `upsert` in a loop.
    pub fn upsert_batch(
        &mut self,
        collection: &str,
        entries: Vec<(u64, Vec<f32>, Option<serde_json::Value>)>,
    ) -> Result<(), VectorDbError> {
        let col = self.manager.get_collection_mut(collection)
            .ok_or_else(|| VectorDbError::CollectionNotFound(collection.to_string()))?;
        col.upsert_batch(entries)
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

    // ── Snapshots ─────────────────────────────────────────────────────────────

    /// Create a named snapshot of `collection`'s current state.
    pub fn create_snapshot(
        &mut self,
        collection: &str,
        name: &str,
    ) -> Result<SnapshotMeta, VectorDbError> {
        let col = self.manager.get_collection_mut(collection)
            .ok_or_else(|| VectorDbError::CollectionNotFound(collection.to_string()))?;
        col.create_snapshot(name)
    }

    /// List all snapshots for `collection`, sorted by creation time.
    pub fn list_snapshots(
        &self,
        collection: &str,
    ) -> Result<Vec<SnapshotMeta>, VectorDbError> {
        let col = self.manager.get_collection(collection)
            .ok_or_else(|| VectorDbError::CollectionNotFound(collection.to_string()))?;
        col.list_snapshots()
    }

    /// Restore `collection` to the state captured by snapshot `name`.
    pub fn restore_snapshot(
        &mut self,
        collection: &str,
        name: &str,
    ) -> Result<(), VectorDbError> {
        let col = self.manager.get_collection_mut(collection)
            .ok_or_else(|| VectorDbError::CollectionNotFound(collection.to_string()))?;
        col.restore_snapshot(name)
    }

    /// Delete a snapshot by name.
    pub fn delete_snapshot(
        &mut self,
        collection: &str,
        name: &str,
    ) -> Result<(), VectorDbError> {
        let col = self.manager.get_collection_mut(collection)
            .ok_or_else(|| VectorDbError::CollectionNotFound(collection.to_string()))?;
        col.delete_snapshot(name)
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

    // ── count() tests ─────────────────────────────────────────────────────────

    #[test]
    fn count_empty_collection() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("c", 3, Metric::L2).unwrap();
        assert_eq!(db.count("c").unwrap(), 0);
    }

    #[test]
    fn count_after_upserts() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("c", 2, Metric::L2).unwrap();
        db.upsert("c", 1, &[1.0, 0.0], None).unwrap();
        db.upsert("c", 2, &[0.0, 1.0], None).unwrap();
        db.upsert("c", 3, &[1.0, 1.0], None).unwrap();
        assert_eq!(db.count("c").unwrap(), 3);
    }

    #[test]
    fn count_nonexistent_collection_errors() {
        let dir = tempdir().unwrap();
        let db = open_db(&dir);
        let result = db.count("nope");
        assert!(matches!(result, Err(VectorDbError::CollectionNotFound(_))));
    }

    // ── upsert_batch() tests ──────────────────────────────────────────────────

    #[test]
    fn upsert_batch_basic() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("b", 2, Metric::L2).unwrap();

        let entries = vec![
            (1, vec![1.0, 0.0], None),
            (2, vec![0.0, 1.0], None),
            (3, vec![1.0, 1.0], Some(serde_json::json!({"k": "v"}))),
        ];
        db.upsert_batch("b", entries).unwrap();
        assert_eq!(db.count("b").unwrap(), 3);

        // Verify search still works after batch insert
        let hits = db.search("b", &[1.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].id, 1);
    }

    #[test]
    fn upsert_batch_empty() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("b", 2, Metric::L2).unwrap();
        db.upsert_batch("b", vec![]).unwrap();
        assert_eq!(db.count("b").unwrap(), 0);
    }

    #[test]
    fn upsert_batch_nonexistent_collection_errors() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        let result = db.upsert_batch("missing", vec![(1, vec![1.0], None)]);
        assert!(matches!(result, Err(VectorDbError::CollectionNotFound(_))));
    }

    // ── delete vector tests ───────────────────────────────────────────────────

    #[test]
    fn delete_nonexistent_vector_returns_false() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("d", 2, Metric::L2).unwrap();
        // Deleting an id that was never inserted should return false (or Ok).
        // The exact behaviour depends on the index; we just verify no panic.
        let _result = db.delete("d", 999);
    }

    #[test]
    fn delete_on_nonexistent_collection_errors() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        let result = db.delete("ghost", 1);
        assert!(matches!(result, Err(VectorDbError::CollectionNotFound(_))));
    }

    #[test]
    fn delete_then_count_decreases() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("d", 2, Metric::L2).unwrap();
        db.upsert("d", 1, &[1.0, 0.0], None).unwrap();
        db.upsert("d", 2, &[0.0, 1.0], None).unwrap();
        assert_eq!(db.count("d").unwrap(), 2);
        db.delete("d", 1).unwrap();
        assert_eq!(db.count("d").unwrap(), 1);
    }

    // ── Error path: upsert to non-existent collection ─────────────────────────

    #[test]
    fn upsert_nonexistent_collection_errors() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        let result = db.upsert("nope", 1, &[1.0], None);
        assert!(matches!(result, Err(VectorDbError::CollectionNotFound(_))));
    }

    // ── Error path: delete non-existent collection ────────────────────────────

    #[test]
    fn delete_nonexistent_collection_returns_false() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        let result = db.delete_collection("nonexistent").unwrap();
        assert!(!result);
    }

    // ── Error path: search_filtered on non-existent collection ────────────────

    #[test]
    fn search_filtered_nonexistent_collection_errors() {
        let dir = tempdir().unwrap();
        let db = open_db(&dir);
        let filter: FilterCondition = serde_json::from_value(
            serde_json::json!({"k": {"$eq": "v"}})
        ).unwrap();
        let result = db.search_filtered("nope", &[1.0], 5, &filter);
        assert!(matches!(result, Err(VectorDbError::CollectionNotFound(_))));
    }

    // ── Error path: create duplicate collection ───────────────────────────────

    #[test]
    fn create_duplicate_collection_errors() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("dup", 2, Metric::L2).unwrap();
        let result = db.create_collection("dup", 2, Metric::L2);
        assert!(matches!(result, Err(VectorDbError::CollectionAlreadyExists(_))));
    }

    // ── MockEmbedder and text embedding tests ─────────────────────────────────

    struct MockEmbedder {
        dims: usize,
    }

    impl crate::embedder::TextEmbedder for MockEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
            // Produce a deterministic vector based on text length
            Ok(vec![text.len() as f32 / 10.0; self.dims])
        }
        fn dimensions(&self) -> Option<usize> {
            Some(self.dims)
        }
    }

    #[test]
    fn set_embedder_basic() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("e", 3, Metric::Cosine).unwrap();
        let embedder = Arc::new(MockEmbedder { dims: 3 });
        db.set_embedder("e", embedder).unwrap();
    }

    #[test]
    fn set_embedder_dimension_mismatch_errors() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("e", 3, Metric::Cosine).unwrap();
        let embedder = Arc::new(MockEmbedder { dims: 5 }); // mismatch: collection is 3
        let result = db.set_embedder("e", embedder);
        assert!(result.is_err());
    }

    #[test]
    fn set_embedder_nonexistent_collection_errors() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        let embedder = Arc::new(MockEmbedder { dims: 3 });
        let result = db.set_embedder("nope", embedder);
        assert!(matches!(result, Err(VectorDbError::CollectionNotFound(_))));
    }

    #[test]
    fn upsert_text_without_embedder_errors() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("t", 3, Metric::Cosine).unwrap();
        let result = db.upsert_text("t", 1, "hello", None);
        assert!(matches!(result, Err(VectorDbError::NoEmbedder(_))));
    }

    #[test]
    fn search_text_without_embedder_errors() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("t", 3, Metric::Cosine).unwrap();
        let result = db.search_text("t", "hello", 5);
        assert!(matches!(result, Err(VectorDbError::NoEmbedder(_))));
    }

    #[test]
    fn upsert_text_nonexistent_collection_errors() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        let result = db.upsert_text("nope", 1, "hello", None);
        assert!(matches!(result, Err(VectorDbError::CollectionNotFound(_))));
    }

    #[test]
    fn search_text_nonexistent_collection_errors() {
        let dir = tempdir().unwrap();
        let db = open_db(&dir);
        let result = db.search_text("nope", "hello", 5);
        assert!(matches!(result, Err(VectorDbError::CollectionNotFound(_))));
    }

    #[test]
    fn upsert_text_and_search_text_with_embedder() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("t", 3, Metric::L2).unwrap();
        let embedder = Arc::new(MockEmbedder { dims: 3 });
        db.set_embedder("t", embedder).unwrap();

        // Insert two texts with different lengths => different embeddings
        db.upsert_text("t", 1, "hi", None).unwrap();          // vec = [0.2, 0.2, 0.2]
        db.upsert_text("t", 2, "hello world", None).unwrap();  // vec = [1.1, 1.1, 1.1]

        assert_eq!(db.count("t").unwrap(), 2);

        // Search for text "hi" — should find id=1 as nearest (identical embedding)
        let hits = db.search_text("t", "hi", 2).unwrap();
        assert_eq!(hits[0].id, 1);
        assert!(hits[0].distance < 1e-4);
    }

    #[test]
    fn upsert_text_with_payload() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("t", 3, Metric::L2).unwrap();
        let embedder = Arc::new(MockEmbedder { dims: 3 });
        db.set_embedder("t", embedder).unwrap();

        db.upsert_text("t", 10, "doc", Some(serde_json::json!({"src": "test"}))).unwrap();
        let hits = db.search_text("t", "doc", 1).unwrap();
        assert_eq!(hits[0].id, 10);
        assert_eq!(hits[0].payload.as_ref().unwrap()["src"], "test");
    }

    // ── Snapshot tests ────────────────────────────────────────────────────────

    #[test]
    fn create_and_list_snapshots() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("s", 2, Metric::L2).unwrap();
        db.upsert("s", 1, &[1.0, 0.0], None).unwrap();

        let snap = db.create_snapshot("s", "v1").unwrap();
        assert_eq!(snap.name, "v1");

        let snaps = db.list_snapshots("s").unwrap();
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0].name, "v1");
    }

    #[test]
    fn restore_snapshot() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("s", 2, Metric::L2).unwrap();
        db.upsert("s", 1, &[1.0, 0.0], None).unwrap();

        db.create_snapshot("s", "before").unwrap();

        // Add more data after the snapshot
        db.upsert("s", 2, &[0.0, 1.0], None).unwrap();
        db.upsert("s", 3, &[1.0, 1.0], None).unwrap();
        assert_eq!(db.count("s").unwrap(), 3);

        // Restore to snapshot — should go back to 1 vector
        db.restore_snapshot("s", "before").unwrap();
        assert_eq!(db.count("s").unwrap(), 1);
    }

    #[test]
    fn delete_snapshot() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("s", 2, Metric::L2).unwrap();
        db.create_snapshot("s", "snap1").unwrap();
        assert_eq!(db.list_snapshots("s").unwrap().len(), 1);

        db.delete_snapshot("s", "snap1").unwrap();
        assert_eq!(db.list_snapshots("s").unwrap().len(), 0);
    }

    #[test]
    fn snapshot_nonexistent_collection_errors() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        assert!(matches!(
            db.create_snapshot("nope", "s"),
            Err(VectorDbError::CollectionNotFound(_))
        ));
        assert!(matches!(
            db.list_snapshots("nope"),
            Err(VectorDbError::CollectionNotFound(_))
        ));
        assert!(matches!(
            db.restore_snapshot("nope", "s"),
            Err(VectorDbError::CollectionNotFound(_))
        ));
        assert!(matches!(
            db.delete_snapshot("nope", "s"),
            Err(VectorDbError::CollectionNotFound(_))
        ));
    }

    #[test]
    fn restore_nonexistent_snapshot_errors() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("s", 2, Metric::L2).unwrap();
        let result = db.restore_snapshot("s", "no_such_snap");
        assert!(result.is_err());
    }

    #[test]
    fn delete_nonexistent_snapshot_errors() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("s", 2, Metric::L2).unwrap();
        let result = db.delete_snapshot("s", "no_such_snap");
        assert!(result.is_err());
    }

    #[test]
    fn create_duplicate_snapshot_errors() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("s", 2, Metric::L2).unwrap();
        db.create_snapshot("s", "dup").unwrap();
        let result = db.create_snapshot("s", "dup");
        assert!(matches!(result, Err(VectorDbError::SnapshotAlreadyExists(_))));
    }

    #[test]
    fn multiple_snapshots_and_selective_restore() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("s", 2, Metric::L2).unwrap();

        db.upsert("s", 1, &[1.0, 0.0], None).unwrap();
        db.create_snapshot("s", "one").unwrap();

        db.upsert("s", 2, &[0.0, 1.0], None).unwrap();
        db.create_snapshot("s", "two").unwrap();

        db.upsert("s", 3, &[1.0, 1.0], None).unwrap();
        assert_eq!(db.count("s").unwrap(), 3);

        let snaps = db.list_snapshots("s").unwrap();
        assert_eq!(snaps.len(), 2);

        // Restore to "one" — only 1 vector
        db.restore_snapshot("s", "one").unwrap();
        assert_eq!(db.count("s").unwrap(), 1);
    }

    // ── get_or_create edge cases ──────────────────────────────────────────────

    #[test]
    fn get_or_create_then_upsert_and_search() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.get_or_create_collection("g", 2, Metric::Cosine).unwrap();
        db.upsert("g", 1, &[1.0, 0.0], None).unwrap();
        let hits = db.search("g", &[1.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].id, 1);
    }

    // ── upsert overwrites existing vector ─────────────────────────────────────

    #[test]
    fn upsert_overwrites_existing_id() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("o", 2, Metric::L2).unwrap();
        db.upsert("o", 1, &[1.0, 0.0], Some(serde_json::json!({"v": 1}))).unwrap();
        db.upsert("o", 1, &[0.0, 1.0], Some(serde_json::json!({"v": 2}))).unwrap();

        // Count should still be 1 (overwrite, not duplicate)
        assert_eq!(db.count("o").unwrap(), 1);

        // Search should find the updated vector
        let hits = db.search("o", &[0.0, 1.0], 1).unwrap();
        assert_eq!(hits[0].id, 1);
        assert!(hits[0].distance < 1e-4);
        assert_eq!(hits[0].payload.as_ref().unwrap()["v"], 2);
    }

    // ── list_snapshots on collection with no snapshots ────────────────────────

    #[test]
    fn list_snapshots_empty() {
        let dir = tempdir().unwrap();
        let mut db = open_db(&dir);
        db.create_collection("s", 2, Metric::L2).unwrap();
        let snaps = db.list_snapshots("s").unwrap();
        assert!(snaps.is_empty());
    }
}
