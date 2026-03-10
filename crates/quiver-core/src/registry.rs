//! LRU-evicting collection registry.
//!
//! [`IndexRegistry`] replaces the raw `HashMap<String, Collection>` that
//! `CollectionManager` uses today. It provides the same logical API but adds:
//!
//! - **Lazy loading**: collections that were created in a previous session are
//!   discovered from disk at startup but not loaded into RAM until first
//!   accessed.
//! - **LRU eviction**: when the number of concurrently-loaded collections
//!   reaches `max_loaded`, the least-recently-used one is flushed and dropped
//!   to stay within the budget.
//!
//! # Usage
//! ```no_run
//! use std::path::Path;
//! use quiver_core::registry::IndexRegistry;
//!
//! let mut reg = IndexRegistry::open(Path::new("./data"), 16).unwrap();
//! reg.with_collection_mut("my-coll", |col| {
//!     col.upsert(1, vec![0.1, 0.2, 0.3], None)
//! }).unwrap();
//! ```

use std::{
    collections::HashMap,
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

use lru::LruCache;

use crate::{
    collection::{Collection, CollectionMeta},
    error::VectorDbError,
};

/// Default number of collections held in RAM simultaneously.
pub const DEFAULT_MAX_LOADED: usize = 32;

/// An LRU-evicting registry of named collections.
///
/// Collections are loaded from disk on first access and evicted from RAM when
/// `max_loaded` is exceeded (LRU policy). Metadata for every known collection
/// is always kept in memory (it is tiny).
pub struct IndexRegistry {
    base_path: PathBuf,
    /// Lightweight metadata for all known collections (always resident).
    meta: HashMap<String, CollectionMeta>,
    /// Fully-loaded collections, bounded by `max_loaded`.
    loaded: LruCache<String, Collection>,
}

impl IndexRegistry {
    /// Open (or create) a registry rooted at `base_path`.
    ///
    /// Scans the directory for existing collections and populates `meta` without
    /// loading any vectors. `max_loaded` caps the LRU cache size.
    pub fn open(base_path: &Path, max_loaded: usize) -> Result<Self, VectorDbError> {
        std::fs::create_dir_all(base_path)?;

        let cap = NonZeroUsize::new(max_loaded.max(1)).unwrap();
        let mut meta = HashMap::new();

        if let Ok(entries) = std::fs::read_dir(base_path) {
            for entry in entries.flatten() {
                let meta_path = entry.path().join("meta.json");
                if meta_path.is_file() {
                    match std::fs::read(&meta_path)
                        .ok()
                        .and_then(|b| serde_json::from_slice::<CollectionMeta>(&b).ok())
                    {
                        Some(m) => {
                            meta.insert(m.name.clone(), m);
                        }
                        None => {
                            tracing::warn!("skipping malformed collection at {:?}", entry.path());
                        }
                    }
                }
            }
        }

        Ok(Self {
            base_path: base_path.to_path_buf(),
            meta,
            loaded: LruCache::new(cap),
        })
    }

    // ── Collection lifecycle ──────────────────────────────────────────────────

    /// Create a brand-new collection and add it to the registry.
    ///
    /// Returns `Err(CollectionAlreadyExists)` if a collection with that name
    /// already exists.
    pub fn create_collection(&mut self, m: CollectionMeta) -> Result<(), VectorDbError> {
        if self.meta.contains_key(&m.name) {
            return Err(VectorDbError::CollectionAlreadyExists(m.name.clone()));
        }
        let dir = self.base_path.join(&m.name);
        let col = Collection::create(&dir, m.clone())?;
        self.meta.insert(m.name.clone(), m.clone());
        self.loaded.put(m.name, col);
        Ok(())
    }

    /// Delete a collection and all its on-disk data.
    ///
    /// Returns `false` if no collection with that name exists.
    pub fn delete_collection(&mut self, name: &str) -> Result<bool, VectorDbError> {
        if !self.meta.contains_key(name) {
            return Ok(false);
        }
        self.loaded.pop(name);
        self.meta.remove(name);
        let dir = self.base_path.join(name);
        if dir.is_dir() {
            std::fs::remove_dir_all(&dir)?;
        }
        Ok(true)
    }

    /// Returns `true` if a collection with `name` is known (even if not loaded).
    pub fn exists(&self, name: &str) -> bool {
        self.meta.contains_key(name)
    }

    /// List metadata for all known collections (does not require loading).
    pub fn list_collections(&self) -> Vec<&CollectionMeta> {
        self.meta.values().collect()
    }

    // ── Access helpers ────────────────────────────────────────────────────────

    /// Run a closure with read-only access to a collection.
    ///
    /// Loads the collection from disk if it is not currently in RAM (and may
    /// evict the LRU collection to make room).
    pub fn with_collection<F, T>(&mut self, name: &str, f: F) -> Result<T, VectorDbError>
    where
        F: FnOnce(&Collection) -> Result<T, VectorDbError>,
    {
        self.ensure_loaded(name)?;
        let col = self.loaded.get(name).unwrap(); // guaranteed by ensure_loaded
        f(col)
    }

    /// Run a closure with mutable access to a collection.
    ///
    /// Loads the collection from disk if it is not currently in RAM (and may
    /// evict the LRU collection to make room).
    pub fn with_collection_mut<F, T>(&mut self, name: &str, f: F) -> Result<T, VectorDbError>
    where
        F: FnOnce(&mut Collection) -> Result<T, VectorDbError>,
    {
        self.ensure_loaded(name)?;
        let col = self.loaded.get_mut(name).unwrap(); // guaranteed by ensure_loaded
        let result = f(col)?;
        // Sync metadata (e.g. after an auto-promotion the index_type may have changed)
        if let Some(m) = self.meta.get_mut(name) {
            *m = col.meta().clone();
        }
        Ok(result)
    }

    // ── Internals ─────────────────────────────────────────────────────────────

    /// Ensure a collection is in the LRU cache, loading from disk if needed.
    fn ensure_loaded(&mut self, name: &str) -> Result<(), VectorDbError> {
        if !self.meta.contains_key(name) {
            return Err(VectorDbError::CollectionNotFound(name.to_string()));
        }
        if self.loaded.contains(name) {
            return Ok(());
        }
        let dir = self.base_path.join(name);
        let col = Collection::load(&dir)?;
        self.loaded.put(name.to_string(), col);
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{collection::IndexType, distance::Metric};
    use tempfile::tempdir;

    fn meta(name: &str) -> CollectionMeta {
        CollectionMeta {
            name: name.to_string(),
            dimensions: 3,
            metric: Metric::L2,
            index_type: IndexType::Flat,
            hnsw_config: None,
            wal_compact_threshold: 50_000,
            auto_promote_threshold: None,
            promotion_hnsw_config: None,
            embedding_model: None,
            faiss_factory: None,
        }
    }

    #[test]
    fn registry_create_and_list() {
        let dir = tempdir().unwrap();
        let mut reg = IndexRegistry::open(dir.path(), 8).unwrap();
        reg.create_collection(meta("alpha")).unwrap();
        reg.create_collection(meta("beta")).unwrap();
        let names: Vec<_> = reg.list_collections().iter().map(|m| m.name.as_str()).collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
    }

    #[test]
    fn registry_create_duplicate_fails() {
        let dir = tempdir().unwrap();
        let mut reg = IndexRegistry::open(dir.path(), 8).unwrap();
        reg.create_collection(meta("col")).unwrap();
        assert!(matches!(
            reg.create_collection(meta("col")),
            Err(VectorDbError::CollectionAlreadyExists(_))
        ));
    }

    #[test]
    fn registry_with_collection_mut() {
        let dir = tempdir().unwrap();
        let mut reg = IndexRegistry::open(dir.path(), 8).unwrap();
        reg.create_collection(meta("col")).unwrap();
        reg.with_collection_mut("col", |c| c.upsert(1, vec![1.0, 0.0, 0.0], None)).unwrap();
        let count = reg.with_collection("col", |c| Ok(c.count())).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn registry_lazy_loads_from_disk() {
        let dir = tempdir().unwrap();
        // Create collection in one registry instance
        {
            let mut reg = IndexRegistry::open(dir.path(), 8).unwrap();
            reg.create_collection(meta("col")).unwrap();
            reg.with_collection_mut("col", |c| c.upsert(42, vec![1.0, 0.0, 0.0], None))
                .unwrap();
        }
        // Open a fresh registry — collection should be discoverable and lazy-loadable
        let mut reg2 = IndexRegistry::open(dir.path(), 8).unwrap();
        assert!(reg2.exists("col"));
        let count = reg2.with_collection("col", |c| Ok(c.count())).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn registry_lru_evicts_cold_collections() {
        let dir = tempdir().unwrap();
        // max_loaded = 2 so inserting 3 collections forces an eviction
        let mut reg = IndexRegistry::open(dir.path(), 2).unwrap();
        for name in ["a", "b", "c"] {
            reg.create_collection(meta(name)).unwrap();
        }
        // All should still be accessible (cold ones get lazily reloaded)
        for name in ["a", "b", "c"] {
            assert!(reg.exists(name));
            let count = reg.with_collection(name, |c| Ok(c.count())).unwrap();
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn registry_delete_collection() {
        let dir = tempdir().unwrap();
        let mut reg = IndexRegistry::open(dir.path(), 8).unwrap();
        reg.create_collection(meta("col")).unwrap();
        assert!(reg.delete_collection("col").unwrap());
        assert!(!reg.exists("col"));
        assert!(!reg.delete_collection("col").unwrap());
    }

    #[test]
    fn registry_persists_meta_across_restart() {
        let dir = tempdir().unwrap();
        {
            let mut reg = IndexRegistry::open(dir.path(), 8).unwrap();
            reg.create_collection(meta("col")).unwrap();
            reg.with_collection_mut("col", |c| {
                c.upsert(1, vec![1.0, 0.0, 0.0], None)?;
                c.upsert(2, vec![0.0, 1.0, 0.0], None)
            }).unwrap();
        }
        let mut reg2 = IndexRegistry::open(dir.path(), 8).unwrap();
        let count = reg2.with_collection("col", |c| Ok(c.count())).unwrap();
        assert_eq!(count, 2);
    }
}
