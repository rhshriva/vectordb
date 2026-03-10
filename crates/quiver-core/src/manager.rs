use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::{
    collection::{Collection, CollectionMeta},
    error::VectorDbError,
};

/// Manages multiple named collections on disk.
///
/// Directory layout:
/// ```text
/// {base_path}/
///   {collection_name}/
///     meta.json
///     wal.log
/// ```
pub struct CollectionManager {
    base_path: PathBuf,
    collections: HashMap<String, Collection>,
}

impl CollectionManager {
    /// Open (or create) the manager at `base_path`.
    /// Scans for existing collection subdirectories and loads each.
    pub fn open(base_path: impl AsRef<Path>) -> Result<Self, VectorDbError> {
        let base_path = base_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_path)?;

        let mut collections = HashMap::new();

        for entry in std::fs::read_dir(&base_path)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            // Only load dirs that have a meta.json
            if !path.join("meta.json").exists() {
                continue;
            }
            match Collection::load(&path) {
                Ok(col) => {
                    collections.insert(col.meta().name.clone(), col);
                }
                Err(e) => {
                    tracing::warn!(
                        "failed to load collection at {}: {}",
                        path.display(),
                        e
                    );
                }
            }
        }

        Ok(Self { base_path, collections })
    }

    /// Create a new collection. Returns an error if one already exists.
    pub fn create_collection(&mut self, meta: CollectionMeta) -> Result<(), VectorDbError> {
        if self.collections.contains_key(&meta.name) {
            return Err(VectorDbError::CollectionAlreadyExists(meta.name.clone()));
        }
        let dir = self.base_path.join(&meta.name);
        let col = Collection::create(&dir, meta)?;
        self.collections.insert(col.meta().name.clone(), col);
        Ok(())
    }

    pub fn get_collection(&self, name: &str) -> Option<&Collection> {
        self.collections.get(name)
    }

    pub fn get_collection_mut(&mut self, name: &str) -> Option<&mut Collection> {
        self.collections.get_mut(name)
    }

    /// Delete a collection. Returns `true` if found and deleted.
    pub fn delete_collection(&mut self, name: &str) -> Result<bool, VectorDbError> {
        if self.collections.remove(name).is_none() {
            return Ok(false);
        }
        let dir = self.base_path.join(name);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        Ok(true)
    }

    pub fn list_collections(&self) -> Vec<&CollectionMeta> {
        self.collections.values().map(|c| c.meta()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collection::IndexType;
    use crate::distance::Metric;
    use tempfile::tempdir;

    fn make_meta(name: &str) -> CollectionMeta {
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
            ivf_config: None,
            faiss_factory: None,
        }
    }

    #[test]
    fn manager_create_and_list() {
        let dir = tempdir().unwrap();
        let mut mgr = CollectionManager::open(dir.path()).unwrap();
        mgr.create_collection(make_meta("a")).unwrap();
        mgr.create_collection(make_meta("b")).unwrap();
        mgr.create_collection(make_meta("c")).unwrap();
        let names: Vec<&str> = mgr.list_collections().iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names.len(), 3);
    }

    #[test]
    fn manager_duplicate_collection_errors() {
        let dir = tempdir().unwrap();
        let mut mgr = CollectionManager::open(dir.path()).unwrap();
        mgr.create_collection(make_meta("dup")).unwrap();
        let result = mgr.create_collection(make_meta("dup"));
        assert!(matches!(result, Err(VectorDbError::CollectionAlreadyExists(_))));
    }

    #[test]
    fn manager_load_existing_after_restart() {
        let dir = tempdir().unwrap();
        {
            let mut mgr = CollectionManager::open(dir.path()).unwrap();
            mgr.create_collection(make_meta("persist")).unwrap();
            let col = mgr.get_collection_mut("persist").unwrap();
            col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();
        }

        let mgr = CollectionManager::open(dir.path()).unwrap();
        let col = mgr.get_collection("persist").unwrap();
        assert_eq!(col.count(), 1);
    }

    #[test]
    fn manager_delete_removes_dir() {
        let dir = tempdir().unwrap();
        let mut mgr = CollectionManager::open(dir.path()).unwrap();
        mgr.create_collection(make_meta("todel")).unwrap();
        let col_dir = dir.path().join("todel");
        assert!(col_dir.exists());
        let deleted = mgr.delete_collection("todel").unwrap();
        assert!(deleted);
        assert!(!col_dir.exists());
    }

    #[test]
    fn manager_delete_nonexistent_returns_false() {
        let dir = tempdir().unwrap();
        let mut mgr = CollectionManager::open(dir.path()).unwrap();
        let deleted = mgr.delete_collection("ghost").unwrap();
        assert!(!deleted);
    }
}
