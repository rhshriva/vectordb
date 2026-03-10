use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{
    distance::Metric,
    error::VectorDbError,
    index::{flat::FlatIndex, hnsw::{HnswConfig, HnswIndex}, VectorIndex},
    payload::{FilterCondition, matches_filter},
    wal::{Wal, WalEntry},
};
#[cfg(feature = "faiss")]
use crate::index::faiss::FaissIndex;

/// How many extra candidates to fetch per filter pass (overscan factor).
const FILTER_OVERSCAN: usize = 10;

/// Index type stored in collection metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IndexType {
    Flat,
    Hnsw,
    /// FAISS index created via the factory string (e.g. `"Flat"`, `"IVF1024,Flat"`).
    /// Requires the crate to be compiled with the `faiss` feature.
    Faiss,
}

/// Persistent metadata for a collection (written once to `meta.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMeta {
    pub name: String,
    pub dimensions: usize,
    pub metric: Metric,
    pub index_type: IndexType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hnsw_config: Option<HnswConfig>,
    /// WAL entry count threshold that triggers automatic compaction.
    #[serde(default = "default_wal_compact_threshold")]
    pub wal_compact_threshold: usize,
    /// When the vector count reaches this threshold the index is automatically
    /// promoted from Flat to HNSW. `None` disables auto-promotion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_promote_threshold: Option<usize>,
    /// HNSW config to use when auto-promoting. Falls back to `HnswConfig::default()`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub promotion_hnsw_config: Option<HnswConfig>,
    /// Optional embedding model identifier (e.g. `"openai/text-embedding-3-small"`,
    /// `"ollama/nomic-embed-text"`). When set, the server can accept raw text
    /// inputs for embed-and-upsert / embed-and-search endpoints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
    /// FAISS factory string used when `index_type == Faiss`.
    /// Defaults to `"Flat"` (exact brute-force, SIMD-accelerated).
    /// Examples: `"IVF1024,Flat"`, `"HNSW32"`, `"IVF256,PQ64"`.
    /// Only meaningful when compiled with the `faiss` feature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub faiss_factory: Option<String>,
}

fn default_wal_compact_threshold() -> usize {
    50_000
}

/// An enriched search result that includes payload metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionSearchResult {
    pub id: u64,
    pub distance: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
}

/// A named collection: vector index + payload store + WAL.
pub struct Collection {
    meta: CollectionMeta,
    index: Box<dyn VectorIndex>,
    payloads: HashMap<u64, serde_json::Value>,
    wal: Wal,
    dir: PathBuf,
}

impl Collection {
    /// Create a brand-new collection at `dir`.
    /// Writes `meta.json` and creates an empty WAL.
    pub fn create(
        dir: impl AsRef<Path>,
        meta: CollectionMeta,
    ) -> Result<Self, VectorDbError> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;

        let meta_path = dir.join("meta.json");
        let meta_bytes = serde_json::to_vec_pretty(&meta)?;
        std::fs::write(&meta_path, &meta_bytes)?;

        let index = build_index(&meta);
        let wal = Wal::open(dir.join("wal.log"))?;

        Ok(Self {
            meta,
            index,
            payloads: HashMap::new(),
            wal,
            dir,
        })
    }

    /// Load an existing collection from `dir`, replaying the WAL.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, VectorDbError> {
        let dir = dir.as_ref().to_path_buf();

        // Clean up any leftover .tmp file from a crashed compaction
        let tmp_path = dir.join("wal.log.tmp");
        if tmp_path.exists() {
            std::fs::remove_file(&tmp_path)?;
        }

        let meta_path = dir.join("meta.json");
        let meta_bytes = std::fs::read(&meta_path)?;
        let meta: CollectionMeta = serde_json::from_slice(&meta_bytes)?;

        let mut index = build_index(&meta);
        let mut payloads: HashMap<u64, serde_json::Value> = HashMap::new();

        Wal::replay(dir.join("wal.log"), |entry| match entry {
            WalEntry::Add { id, vector, payload_bytes } => {
                // Idempotent upsert on replay
                index.delete(id);
                let _ = index.add(id, &vector);
                match payload_bytes {
                    Some(b) => {
                        if let Ok(p) = serde_json::from_slice(&b) {
                            payloads.insert(id, p);
                        }
                    }
                    None => { payloads.remove(&id); }
                }
            }
            WalEntry::Delete { id } => {
                index.delete(id);
                payloads.remove(&id);
            }
        })?;

        // For HNSW: rebuild graph after replay
        index.flush();

        let wal = Wal::open(dir.join("wal.log"))?;

        Ok(Self {
            meta,
            index,
            payloads,
            wal,
            dir,
        })
    }

    /// Upsert a vector (add or replace by id). WAL is written first.
    pub fn upsert(
        &mut self,
        id: u64,
        vector: Vec<f32>,
        payload: Option<serde_json::Value>,
    ) -> Result<(), VectorDbError> {
        let entry = WalEntry::Add {
            id,
            vector: vector.clone(),
            payload_bytes: payload.as_ref()
                .map(|p| serde_json::to_vec(p).unwrap_or_default()),
        };
        self.wal.append(&entry)?;

        // Update in-memory state
        self.index.delete(id);
        self.index.add(id, &vector)?;
        match payload {
            Some(p) => { self.payloads.insert(id, p); }
            None => { self.payloads.remove(&id); }
        }

        self.maybe_compact()?;
        self.maybe_promote()?;
        Ok(())
    }

    /// Delete a vector by id. Returns true if found and removed.
    pub fn delete(&mut self, id: u64) -> Result<bool, VectorDbError> {
        self.wal.append(&WalEntry::Delete { id })?;
        let found = self.index.delete(id);
        self.payloads.remove(&id);
        self.maybe_compact()?;
        Ok(found)
    }

    /// Search, with optional payload filter applied post-search.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&FilterCondition>,
    ) -> Result<Vec<CollectionSearchResult>, VectorDbError> {
        if filter.is_none() {
            return self.search_inner(query, k);
        }
        let filter = filter.unwrap();

        // Overscan: fetch more candidates then filter down
        let overscan_k = k.saturating_mul(FILTER_OVERSCAN).max(k);
        let candidates = self.search_inner(query, overscan_k)?;

        let results: Vec<CollectionSearchResult> = candidates
            .into_iter()
            .filter(|r| {
                let payload_ref: &serde_json::Value = self
                    .payloads
                    .get(&r.id)
                    .unwrap_or(&serde_json::Value::Null);
                matches_filter(payload_ref, filter)
            })
            .take(k)
            .collect();

        Ok(results)
    }

    fn search_inner(&self, query: &[f32], k: usize) -> Result<Vec<CollectionSearchResult>, VectorDbError> {
        let raw = self.index.search(query, k)?;
        Ok(raw
            .into_iter()
            .map(|r| CollectionSearchResult {
                id: r.id,
                distance: r.distance,
                payload: self.payloads.get(&r.id).cloned(),
            })
            .collect())
    }

    pub fn count(&self) -> usize {
        self.index.len()
    }

    pub fn meta(&self) -> &CollectionMeta {
        &self.meta
    }

    /// Force WAL compaction now.
    pub fn compact(&mut self) -> Result<(), VectorDbError> {
        let live: Vec<WalEntry> = self
            .index
            .iter_vectors()
            .map(|(id, vector)| WalEntry::Add {
                id,
                vector,
                payload_bytes: self.payloads.get(&id)
                    .map(|p| serde_json::to_vec(p).unwrap_or_default()),
            })
            .collect();
        let count = live.len();
        Wal::compact(self.dir.join("wal.log"), live.into_iter())?;
        // Reopen WAL in append mode after compaction
        self.wal = Wal::open(self.dir.join("wal.log"))?;
        self.wal.reset_entry_count(count);
        Ok(())
    }

    fn maybe_compact(&mut self) -> Result<(), VectorDbError> {
        if self.wal.entry_count() >= self.meta.wal_compact_threshold {
            self.compact()?;
        }
        Ok(())
    }

    /// Promote from Flat → HNSW if `auto_promote_threshold` is set and reached.
    fn maybe_promote(&mut self) -> Result<(), VectorDbError> {
        let threshold = match self.meta.auto_promote_threshold {
            Some(t) => t,
            None => return Ok(()),
        };
        if self.meta.index_type != IndexType::Flat {
            return Ok(()); // already HNSW, FAISS, or other type
        }
        if self.index.len() < threshold {
            return Ok(());
        }
        self.promote_to_hnsw()
    }

    /// Rebuild the in-memory index as HNSW and atomically update `meta.json`.
    pub fn promote_to_hnsw(&mut self) -> Result<(), VectorDbError> {
        let cfg = self
            .meta
            .promotion_hnsw_config
            .clone()
            .unwrap_or_default();
        let mut new_index = HnswIndex::new(self.meta.dimensions, self.meta.metric, cfg.clone());
        for (id, vec) in self.index.iter_vectors() {
            new_index.add(id, &vec)?;
        }
        new_index.flush();

        self.index = Box::new(new_index);
        self.meta.index_type = IndexType::Hnsw;
        self.meta.hnsw_config = Some(cfg);

        // Atomically update meta.json so next restart also loads HNSW
        let meta_bytes = serde_json::to_vec_pretty(&self.meta)?;
        let tmp = self.dir.join("meta.json.tmp");
        std::fs::write(&tmp, &meta_bytes)?;
        std::fs::rename(&tmp, self.dir.join("meta.json"))?;

        Ok(())
    }
}

fn build_index(meta: &CollectionMeta) -> Box<dyn VectorIndex> {
    match meta.index_type {
        IndexType::Flat => Box::new(FlatIndex::new(meta.dimensions, meta.metric)),
        IndexType::Hnsw => {
            let cfg = meta.hnsw_config.clone().unwrap_or_default();
            Box::new(HnswIndex::new(meta.dimensions, meta.metric, cfg))
        }
        IndexType::Faiss => {
            #[cfg(feature = "faiss")]
            {
                let factory = meta
                    .faiss_factory
                    .clone()
                    .unwrap_or_else(|| "Flat".to_string());
                Box::new(
                    FaissIndex::new(meta.dimensions, meta.metric, factory)
                        .expect("failed to build FAISS index"),
                )
            }
            #[cfg(not(feature = "faiss"))]
            panic!(
                "collection '{}' uses IndexType::Faiss but quiver-core was compiled \
                 without the 'faiss' feature; rebuild with `--features faiss`",
                meta.name
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_meta(name: &str, index_type: IndexType) -> CollectionMeta {
        CollectionMeta {
            name: name.to_string(),
            dimensions: 3,
            metric: Metric::L2,
            index_type,
            hnsw_config: None,
            wal_compact_threshold: 50_000,
            auto_promote_threshold: None,
            promotion_hnsw_config: None,
            embedding_model: None,
            faiss_factory: None,
        }
    }

    #[test]
    fn collection_create_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("test", IndexType::Flat);

        let _col = Collection::create(&path, meta.clone()).unwrap();
        drop(_col);

        let loaded = Collection::load(&path).unwrap();
        assert_eq!(loaded.meta().name, "test");
        assert_eq!(loaded.count(), 0);
    }

    #[test]
    fn collection_upsert_persists_across_restart() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("test", IndexType::Flat);

        {
            let mut col = Collection::create(&path, meta).unwrap();
            col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();
            col.upsert(2, vec![0.0, 1.0, 0.0], None).unwrap();
            col.upsert(3, vec![0.0, 0.0, 1.0], None).unwrap();
        }

        let col = Collection::load(&path).unwrap();
        assert_eq!(col.count(), 3);
        let results = col.search(&[1.0, 0.0, 0.0], 1, None).unwrap();
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn collection_delete_persists() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("test", IndexType::Flat);

        {
            let mut col = Collection::create(&path, meta).unwrap();
            col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();
            col.upsert(2, vec![0.0, 1.0, 0.0], None).unwrap();
            col.delete(1).unwrap();
        }

        let col = Collection::load(&path).unwrap();
        assert_eq!(col.count(), 1);
        let results = col.search(&[1.0, 0.0, 0.0], 5, None).unwrap();
        assert!(results.iter().all(|r| r.id != 1));
    }

    #[test]
    fn collection_payload_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("test", IndexType::Flat);

        {
            let mut col = Collection::create(&path, meta).unwrap();
            col.upsert(1, vec![1.0, 0.0, 0.0], Some(serde_json::json!({"tag": "news"}))).unwrap();
        }

        let col = Collection::load(&path).unwrap();
        let results = col.search(&[1.0, 0.0, 0.0], 1, None).unwrap();
        assert_eq!(results[0].id, 1);
        assert_eq!(results[0].payload.as_ref().unwrap()["tag"], "news");
    }

    #[test]
    fn collection_filtered_search() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("test", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        col.upsert(1, vec![1.0, 0.0, 0.0], Some(serde_json::json!({"tag": "a"}))).unwrap();
        col.upsert(2, vec![0.9, 0.1, 0.0], Some(serde_json::json!({"tag": "b"}))).unwrap();
        col.upsert(3, vec![0.8, 0.2, 0.0], Some(serde_json::json!({"tag": "a"}))).unwrap();
        col.upsert(4, vec![0.1, 0.9, 0.0], Some(serde_json::json!({"tag": "b"}))).unwrap();
        col.upsert(5, vec![0.0, 1.0, 0.0], Some(serde_json::json!({"tag": "a"}))).unwrap();

        let filter: FilterCondition = serde_json::from_str(r#"{"tag": {"$eq": "a"}}"#).unwrap();
        let results = col.search(&[1.0, 0.0, 0.0], 5, Some(&filter)).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| {
            r.payload.as_ref()
                .and_then(|p| p["tag"].as_str())
                == Some("a")
        }));
    }

    #[test]
    fn collection_upsert_replaces_existing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("test", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();
        col.upsert(1, vec![0.0, 1.0, 0.0], None).unwrap();
        assert_eq!(col.count(), 1);
    }

    #[test]
    fn collection_compact_triggered_at_threshold() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let mut meta = make_meta("test", IndexType::Flat);
        meta.wal_compact_threshold = 5;
        let mut col = Collection::create(&path, meta).unwrap();

        for i in 0..6u64 {
            col.upsert(i, vec![i as f32, 0.0, 0.0], None).unwrap();
        }

        // After 6 inserts (threshold=5 crossed), WAL should have been compacted.
        // Replay must yield at most 6 frames (compacted), not 5+6=11 (uncompacted).
        let mut frame_count = 0usize;
        Wal::replay(path.join("wal.log"), |_| { frame_count += 1; }).unwrap();
        assert!(frame_count <= 6, "expected at most 6 WAL frames after compaction, got {frame_count}");
    }

    #[test]
    fn collection_auto_promotes_flat_to_hnsw() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let mut meta = make_meta("test", IndexType::Flat);
        meta.auto_promote_threshold = Some(3);
        let mut col = Collection::create(&path, meta).unwrap();

        col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();
        col.upsert(2, vec![0.0, 1.0, 0.0], None).unwrap();
        // Still flat — threshold not reached yet
        assert_eq!(col.meta().index_type, IndexType::Flat);

        col.upsert(3, vec![0.0, 0.0, 1.0], None).unwrap();
        // Should have promoted now
        assert_eq!(col.meta().index_type, IndexType::Hnsw);

        // Persist meta.json should reflect HNSW
        let loaded = Collection::load(&path).unwrap();
        assert_eq!(loaded.meta().index_type, IndexType::Hnsw);
        assert_eq!(loaded.count(), 3);

        // Search should still work
        let results = loaded.search(&[1.0, 0.0, 0.0], 1, None).unwrap();
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn collection_compact_manually() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("test", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        for i in 0..10u64 {
            col.upsert(i, vec![i as f32, 0.0, 0.0], None).unwrap();
        }
        col.compact().unwrap();

        let mut entries = Vec::new();
        crate::wal::Wal::replay(path.join("wal.log"), |e| entries.push(e)).unwrap();
        assert_eq!(entries.len(), 10);
    }
}
