use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use std::sync::Arc;

use crate::{
    distance::Metric,
    embedder::TextEmbedder,
    error::VectorDbError,
    index::{
        binary_flat::BinaryFlatIndex,
        flat::FlatIndex,
        hnsw::{HnswConfig, HnswIndex},
        ivf::{IvfConfig, IvfIndex},
        ivf_pq::{IvfPqConfig, IvfPqIndex},
        mmap_flat::MmapFlatIndex,
        quantized_flat::QuantizedFlatIndex,
        quantized_fp16::Fp16FlatIndex,
        sparse::{SparseIndex, SparseVector},
        VectorIndex,
    },
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
    /// Scalar-quantized (int8) brute-force index.
    /// ~4× lower memory than `Flat` with sub-1% recall loss.
    QuantizedFlat,
    /// Inverted file index (k-means + posting lists).
    /// Sub-linear approximate search; configure via [`IvfConfig`].
    Ivf,
    /// IVF with product quantization for dramatic memory reduction.
    /// Sub-linear search with ~96× compression for high-dimensional vectors.
    IvfPq,
    /// FP16 scalar-quantized brute-force index.
    /// ~2× lower memory than `Flat` with < 0.1% recall loss.
    Fp16Flat,
    /// Memory-mapped flat index — vectors live on disk, accessed via mmap.
    /// Keeps RAM usage near zero for large collections.
    MmapFlat,
    /// Binary (1-bit) quantized brute-force index.
    /// 32× lower memory than `Flat`. Uses Hamming distance for search.
    BinaryFlat,
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
    /// IVF configuration used when `index_type == Ivf`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ivf_config: Option<IvfConfig>,
    /// IVF-PQ configuration used when `index_type == IvfPq`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ivf_pq_config: Option<IvfPqConfig>,
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

/// An enriched hybrid search result that includes both dense and sparse scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchResult {
    pub id: u64,
    /// Fused score (higher = more similar).
    pub score: f32,
    /// Raw dense distance (lower = closer).
    pub dense_distance: f32,
    /// Raw sparse dot-product score (higher = more similar).
    pub sparse_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
}

/// A named collection: vector index + payload store + WAL.
pub struct Collection {
    meta: CollectionMeta,
    index: Box<dyn VectorIndex>,
    payloads: HashMap<u64, serde_json::Value>,
    /// Sparse inverted index for hybrid search (lazily created on first sparse upsert).
    sparse_index: SparseIndex,
    wal: Wal,
    dir: PathBuf,
    /// Optional embedder for `upsert_text` / `search_text`.
    /// Not persisted — must be re-attached after loading via `set_embedder`.
    embedder: Option<Arc<dyn TextEmbedder>>,
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

        let index = build_index(&meta, &dir);
        let wal = Wal::open(dir.join("wal.log"))?;

        Ok(Self {
            meta,
            index,
            payloads: HashMap::new(),
            sparse_index: SparseIndex::new(),
            wal,
            dir,
            embedder: None,
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

        let mut index = build_index(&meta, &dir);
        let mut payloads: HashMap<u64, serde_json::Value> = HashMap::new();

        // Load persisted sparse index if it exists; otherwise start empty.
        let sparse_path = dir.join("sparse.bin");
        let mut sparse_index = if sparse_path.is_file() {
            SparseIndex::load(&sparse_path).unwrap_or_else(|e| {
                tracing::warn!("failed to load sparse index: {e}; starting fresh");
                SparseIndex::new()
            })
        } else {
            SparseIndex::new()
        };

        Wal::replay(dir.join("wal.log"), |entry| match entry {
            WalEntry::Add { id, vector, payload_bytes, sparse_bytes } => {
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
                // Replay sparse data if present
                if let Some(sb) = sparse_bytes {
                    if let Ok(sv) = bincode::deserialize::<SparseVector>(&sb) {
                        sparse_index.upsert(id, sv);
                    }
                }
            }
            WalEntry::Delete { id } => {
                index.delete(id);
                payloads.remove(&id);
                sparse_index.delete(id);
            }
        })?;

        // For HNSW: try to load persisted graph to skip O(N log N) rebuild;
        // falls back to flush() if the graph file is absent or stale.
        index.load_graph_mmap(&dir.join("hnsw.graph"))?;

        let wal = Wal::open(dir.join("wal.log"))?;

        Ok(Self {
            meta,
            index,
            payloads,
            sparse_index,
            wal,
            dir,
            embedder: None,
        })
    }

    /// Upsert a vector (add or replace by id). WAL is written first.
    pub fn upsert(
        &mut self,
        id: u64,
        vector: Vec<f32>,
        payload: Option<serde_json::Value>,
    ) -> Result<(), VectorDbError> {
        // Use borrow-based WAL append — avoids cloning the vector.
        let pb = payload.as_ref()
            .map(|p| serde_json::to_vec(p).unwrap_or_default());
        self.wal.append_add(id, &vector, pb.as_deref(), None)?;

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

    /// Batch upsert multiple vectors at once. More efficient than calling
    /// `upsert` in a loop because WAL compaction and promotion checks happen
    /// only once at the end, and WAL flushes once for the entire batch.
    pub fn upsert_batch(
        &mut self,
        entries: Vec<(u64, Vec<f32>, Option<serde_json::Value>)>,
    ) -> Result<(), VectorDbError> {
        for (id, vector, payload) in entries {
            // Use no-flush variant — single flush after entire batch
            let pb = payload.as_ref()
                .map(|p| serde_json::to_vec(p).unwrap_or_default());
            self.wal.append_add_no_flush(id, &vector, pb.as_deref(), None)?;

            self.index.delete(id);
            self.index.add(id, &vector)?;
            match payload {
                Some(p) => { self.payloads.insert(id, p); }
                None => { self.payloads.remove(&id); }
            }
        }
        // Single flush for the entire batch
        self.wal.flush()?;
        self.maybe_compact()?;
        self.maybe_promote()?;
        Ok(())
    }

    /// Upsert a vector with an optional sparse vector for hybrid search.
    ///
    /// The dense vector is stored in the main index. The sparse vector (if
    /// provided) is stored in a separate inverted index for hybrid search.
    pub fn upsert_hybrid(
        &mut self,
        id: u64,
        vector: Vec<f32>,
        sparse_vector: Option<SparseVector>,
        payload: Option<serde_json::Value>,
    ) -> Result<(), VectorDbError> {
        let sparse_bytes_owned = sparse_vector.as_ref().map(|sv| {
            bincode::serialize(sv).unwrap_or_default()
        });
        let pb = payload.as_ref()
            .map(|p| serde_json::to_vec(p).unwrap_or_default());
        self.wal.append_add(
            id,
            &vector,
            pb.as_deref(),
            sparse_bytes_owned.as_deref(),
        )?;

        // Update dense index
        self.index.delete(id);
        self.index.add(id, &vector)?;

        // Update sparse index
        if let Some(sv) = sparse_vector {
            self.sparse_index.upsert(id, sv);
        }

        // Update payload
        match payload {
            Some(p) => { self.payloads.insert(id, p); }
            None => { self.payloads.remove(&id); }
        }

        self.maybe_compact()?;
        self.maybe_promote()?;
        Ok(())
    }

    /// Hybrid search: weighted fusion of dense ANN + sparse keyword search.
    ///
    /// Returns top-k results ranked by:
    /// `score = dense_weight * (1 - norm_dense_dist) + sparse_weight * norm_sparse_score`
    ///
    /// where `norm_dense_dist` and `norm_sparse_score` are min-max normalized
    /// to [0, 1] across the candidate set.
    ///
    /// # Arguments
    /// - `dense_query`: dense query vector for the ANN index
    /// - `sparse_query`: sparse query vector for the inverted index
    /// - `k`: number of results to return
    /// - `dense_weight`: weight for the dense similarity score (default: 0.7)
    /// - `sparse_weight`: weight for the sparse similarity score (default: 0.3)
    /// - `filter`: optional payload filter applied post-search
    pub fn search_hybrid(
        &self,
        dense_query: &[f32],
        sparse_query: &SparseVector,
        k: usize,
        dense_weight: f32,
        sparse_weight: f32,
        filter: Option<&FilterCondition>,
    ) -> Result<Vec<HybridSearchResult>, VectorDbError> {
        if k == 0 {
            return Ok(vec![]);
        }

        // Overscan for dense and sparse to get enough candidates.
        let overscan = k.saturating_mul(FILTER_OVERSCAN).max(k);

        // Dense ANN search.
        let dense_results = self.index.search(dense_query, overscan)?;

        // Sparse search.
        let sparse_results = self.sparse_index.search(sparse_query, overscan);

        // Collect all candidate IDs and their raw scores.
        let mut dense_scores: HashMap<u64, f32> = HashMap::new();
        let mut sparse_scores: HashMap<u64, f32> = HashMap::new();

        for r in &dense_results {
            dense_scores.insert(r.id, r.distance);
        }
        for r in &sparse_results {
            sparse_scores.insert(r.id, r.score);
        }

        // Union of all candidate IDs.
        let mut all_ids: Vec<u64> = dense_scores.keys()
            .chain(sparse_scores.keys())
            .copied()
            .collect::<std::collections::HashSet<u64>>()
            .into_iter()
            .collect();

        // Min-max normalization for dense distances (lower = better → invert).
        let (dense_min, dense_max) = if dense_scores.is_empty() {
            (0.0, 1.0)
        } else {
            let min = dense_scores.values().cloned().fold(f32::MAX, f32::min);
            let max = dense_scores.values().cloned().fold(f32::MIN, f32::max);
            (min, max)
        };
        let dense_range = (dense_max - dense_min).max(1e-9);

        // Min-max normalization for sparse scores (higher = better).
        let (sparse_min, sparse_max) = if sparse_scores.is_empty() {
            (0.0, 1.0)
        } else {
            let min = sparse_scores.values().cloned().fold(f32::MAX, f32::min);
            let max = sparse_scores.values().cloned().fold(f32::MIN, f32::max);
            (min, max)
        };
        let sparse_range = (sparse_max - sparse_min).max(1e-9);

        // Compute fused scores.
        let mut results: Vec<HybridSearchResult> = all_ids.drain(..).filter_map(|id| {
            // Apply filter if present.
            if let Some(f) = filter {
                let payload_ref = self.payloads.get(&id).unwrap_or(&serde_json::Value::Null);
                if !matches_filter(payload_ref, f) {
                    return None;
                }
            }

            let raw_dense = *dense_scores.get(&id).unwrap_or(&dense_max);
            let raw_sparse = *sparse_scores.get(&id).unwrap_or(&0.0);

            // Normalize: dense distance → similarity (1 = most similar, 0 = least).
            let norm_dense = 1.0 - (raw_dense - dense_min) / dense_range;
            // Normalize: sparse score → [0, 1].
            let norm_sparse = (raw_sparse - sparse_min) / sparse_range;

            let fused_score = dense_weight * norm_dense + sparse_weight * norm_sparse;

            Some(HybridSearchResult {
                id,
                score: fused_score,
                dense_distance: raw_dense,
                sparse_score: raw_sparse,
                payload: self.payloads.get(&id).cloned(),
            })
        }).collect();

        // Sort by fused score descending (higher = better).
        results.sort_unstable_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        Ok(results)
    }

    /// Number of sparse vectors stored.
    pub fn sparse_count(&self) -> usize {
        self.sparse_index.len()
    }

    /// Delete a vector by id. Returns true if found and removed.
    pub fn delete(&mut self, id: u64) -> Result<bool, VectorDbError> {
        self.wal.append(&WalEntry::Delete { id })?;
        let found = self.index.delete(id);
        self.sparse_index.delete(id);
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

    // ── Embedding helpers ──────────────────────────────────────────────────────

    /// Attach a [`TextEmbedder`] to this collection, enabling [`upsert_text`]
    /// and [`search_text`].
    ///
    /// If the embedder reports a known dimensionality that does not match the
    /// collection's configured dimensions, an error is returned.
    ///
    /// The embedder is **not persisted** — it must be re-attached after each
    /// process restart.
    ///
    /// [`upsert_text`]: Collection::upsert_text
    /// [`search_text`]: Collection::search_text
    pub fn set_embedder(&mut self, embedder: Arc<dyn TextEmbedder>) -> Result<(), VectorDbError> {
        if let Some(dims) = embedder.dimensions() {
            if dims != self.meta.dimensions {
                return Err(VectorDbError::InvalidConfig(format!(
                    "embedder produces {dims}-dim vectors but collection '{}' expects {}",
                    self.meta.name, self.meta.dimensions
                )));
            }
        }
        self.embedder = Some(embedder);
        Ok(())
    }

    /// Embed `text` with the attached embedder, then upsert the resulting vector.
    ///
    /// Returns [`VectorDbError::NoEmbedder`] if no embedder has been set.
    pub fn upsert_text(
        &mut self,
        id: u64,
        text: &str,
        payload: Option<serde_json::Value>,
    ) -> Result<(), VectorDbError> {
        let embedder = self
            .embedder
            .clone()
            .ok_or_else(|| VectorDbError::NoEmbedder(self.meta.name.clone()))?;
        let vector = embedder
            .embed(text)
            .map_err(|e| VectorDbError::EmbeddingError(e.to_string()))?;
        self.upsert(id, vector, payload)
    }

    /// Embed `text` with the attached embedder, then search for the `k`
    /// nearest neighbours.
    ///
    /// Returns [`VectorDbError::NoEmbedder`] if no embedder has been set.
    pub fn search_text(
        &self,
        text: &str,
        k: usize,
    ) -> Result<Vec<CollectionSearchResult>, VectorDbError> {
        let embedder = self
            .embedder
            .as_ref()
            .ok_or_else(|| VectorDbError::NoEmbedder(self.meta.name.clone()))?;
        let query = embedder
            .embed(text)
            .map_err(|e| VectorDbError::EmbeddingError(e.to_string()))?;
        self.search(&query, k, None)
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
                sparse_bytes: self.sparse_index.get(id)
                    .map(|sv| bincode::serialize(sv).unwrap_or_default()),
            })
            .collect();
        let count = live.len();
        Wal::compact(self.dir.join("wal.log"), live.into_iter())?;
        // Reopen WAL in append mode after compaction
        self.wal = Wal::open(self.dir.join("wal.log"))?;
        self.wal.reset_entry_count(count);
        // Persist HNSW graph so the next load skips the rebuild
        self.index.save_graph(&self.dir.join("hnsw.graph"))?;
        // Persist sparse index
        if !self.sparse_index.is_empty() {
            self.sparse_index.save(self.dir.join("sparse.bin"))?;
        }
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

    // ── Snapshot methods ──────────────────────────────────────────────────────

    /// Create a named snapshot of the current collection state.
    ///
    /// The snapshot contains a compacted WAL with only live entries, a copy of
    /// the collection metadata, and the sparse index (if non-empty).  The
    /// in-memory index is not snapshotted — it is rebuilt from the WAL on
    /// restore.
    pub fn create_snapshot(&self, name: &str) -> Result<SnapshotMeta, VectorDbError> {
        if name.is_empty()
            || name.contains('/')
            || name.contains('\\')
            || name.contains("..")
        {
            return Err(VectorDbError::InvalidConfig(format!(
                "invalid snapshot name: {name:?}"
            )));
        }

        let snap_dir = self.dir.join("snapshots").join(name);
        if snap_dir.exists() {
            return Err(VectorDbError::SnapshotAlreadyExists(name.to_string()));
        }
        std::fs::create_dir_all(&snap_dir)?;

        // Copy meta.json
        std::fs::copy(self.dir.join("meta.json"), snap_dir.join("meta.json"))?;

        // Write compacted WAL with live entries
        let live: Vec<WalEntry> = self
            .index
            .iter_vectors()
            .map(|(id, vector)| WalEntry::Add {
                id,
                vector,
                payload_bytes: self
                    .payloads
                    .get(&id)
                    .map(|p| serde_json::to_vec(p).unwrap_or_default()),
                sparse_bytes: self
                    .sparse_index
                    .get(id)
                    .map(|sv| bincode::serialize(sv).unwrap_or_default()),
            })
            .collect();

        let vector_count = live.len();
        Wal::compact(snap_dir.join("wal.log"), live.into_iter())?;

        // Copy sparse index if non-empty
        let sparse_count = self.sparse_index.len();
        if !self.sparse_index.is_empty() {
            self.sparse_index.save(snap_dir.join("sparse.bin"))?;
        }

        // Write snapshot metadata
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let meta = SnapshotMeta {
            name: name.to_string(),
            created_at,
            vector_count,
            sparse_count,
        };
        let meta_bytes = serde_json::to_vec_pretty(&meta)?;
        std::fs::write(snap_dir.join("snapshot.json"), &meta_bytes)?;

        Ok(meta)
    }

    /// List all snapshots for this collection, sorted by creation time.
    pub fn list_snapshots(&self) -> Result<Vec<SnapshotMeta>, VectorDbError> {
        let snap_base = self.dir.join("snapshots");
        if !snap_base.exists() {
            return Ok(vec![]);
        }
        let mut snapshots = Vec::new();
        for entry in std::fs::read_dir(&snap_base)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let meta_path = path.join("snapshot.json");
            if !meta_path.exists() {
                continue;
            }
            let bytes = std::fs::read(&meta_path)?;
            if let Ok(meta) = serde_json::from_slice::<SnapshotMeta>(&bytes) {
                snapshots.push(meta);
            }
        }
        snapshots.sort_by_key(|s| s.created_at);
        Ok(snapshots)
    }

    /// Restore this collection to a previously saved snapshot.
    ///
    /// Overwrites the current WAL, metadata, and sparse index with the
    /// snapshot's copies, then reloads the entire collection from disk.
    pub fn restore_snapshot(&mut self, name: &str) -> Result<(), VectorDbError> {
        let snap_dir = self.dir.join("snapshots").join(name);
        if !snap_dir.exists() {
            return Err(VectorDbError::SnapshotNotFound(name.to_string()));
        }

        // Overwrite current files with snapshot files
        std::fs::copy(snap_dir.join("wal.log"), self.dir.join("wal.log"))?;
        std::fs::copy(snap_dir.join("meta.json"), self.dir.join("meta.json"))?;

        // Handle sparse.bin
        let snap_sparse = snap_dir.join("sparse.bin");
        let cur_sparse = self.dir.join("sparse.bin");
        if snap_sparse.exists() {
            std::fs::copy(&snap_sparse, &cur_sparse)?;
        } else if cur_sparse.exists() {
            std::fs::remove_file(&cur_sparse)?;
        }

        // Remove stale HNSW graph (it will be rebuilt from WAL on load)
        let graph_path = self.dir.join("hnsw.graph");
        if graph_path.exists() {
            std::fs::remove_file(&graph_path)?;
        }

        // Reload the collection from the restored files
        let dir = self.dir.clone();
        *self = Collection::load(&dir)?;

        Ok(())
    }

    /// Delete a snapshot by name.
    pub fn delete_snapshot(&self, name: &str) -> Result<(), VectorDbError> {
        let snap_dir = self.dir.join("snapshots").join(name);
        if !snap_dir.exists() {
            return Err(VectorDbError::SnapshotNotFound(name.to_string()));
        }
        std::fs::remove_dir_all(&snap_dir)?;
        Ok(())
    }
}

/// Metadata for a collection snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMeta {
    /// User-provided snapshot name.
    pub name: String,
    /// Unix timestamp (seconds since epoch) when the snapshot was created.
    pub created_at: u64,
    /// Number of vectors in the collection at snapshot time.
    pub vector_count: usize,
    /// Number of sparse vectors at snapshot time.
    pub sparse_count: usize,
}

fn build_index(meta: &CollectionMeta, dir: &Path) -> Box<dyn VectorIndex> {
    match meta.index_type {
        IndexType::Flat => Box::new(FlatIndex::new(meta.dimensions, meta.metric)),
        IndexType::QuantizedFlat => Box::new(QuantizedFlatIndex::new(meta.dimensions, meta.metric)),
        IndexType::Fp16Flat => Box::new(Fp16FlatIndex::new(meta.dimensions, meta.metric)),
        IndexType::BinaryFlat => Box::new(BinaryFlatIndex::new(meta.dimensions, meta.metric)),
        IndexType::Hnsw => {
            let cfg = meta.hnsw_config.clone().unwrap_or_default();
            Box::new(HnswIndex::new(meta.dimensions, meta.metric, cfg))
        }
        IndexType::Ivf => {
            let cfg = meta.ivf_config.clone().unwrap_or_default();
            Box::new(IvfIndex::new(meta.dimensions, meta.metric, cfg))
        }
        IndexType::IvfPq => {
            let cfg = meta.ivf_pq_config.clone().unwrap_or_default();
            Box::new(IvfPqIndex::new(meta.dimensions, meta.metric, cfg))
        }
        IndexType::MmapFlat => {
            let data_path = dir.join("vectors.mmap");
            Box::new(
                MmapFlatIndex::new(meta.dimensions, meta.metric, data_path)
                    .expect("failed to open MmapFlatIndex"),
            )
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
            ivf_config: None,
            ivf_pq_config: None,
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

    // ── Hybrid search tests ─────────────────────────────────────────────────

    #[test]
    fn collection_hybrid_upsert_and_search() {
        use crate::index::sparse::SparseVector;

        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("hybrid", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        // Vector 1: strong dense match, weak sparse match
        col.upsert_hybrid(
            1,
            vec![1.0, 0.0, 0.0],
            Some(SparseVector::new(vec![0, 1], vec![0.1, 0.1])),
            None,
        ).unwrap();

        // Vector 2: moderate dense match, strong sparse match
        col.upsert_hybrid(
            2,
            vec![0.5, 0.5, 0.0],
            Some(SparseVector::new(vec![0, 1], vec![1.0, 1.0])),
            None,
        ).unwrap();

        // Vector 3: weak dense match, no sparse match
        col.upsert_hybrid(
            3,
            vec![0.0, 0.0, 1.0],
            Some(SparseVector::new(vec![5], vec![0.1])),
            None,
        ).unwrap();

        // Dense-only query would prefer id=1.
        // Sparse query [0:1.0, 1:1.0] strongly prefers id=2.
        // With balanced weights, id=2 should rank higher.
        let results = col.search_hybrid(
            &[1.0, 0.0, 0.0],
            &SparseVector::new(vec![0, 1], vec![1.0, 1.0]),
            3,
            0.5,
            0.5,
            None,
        ).unwrap();

        assert_eq!(results.len(), 3);
        // id=2 should be top due to strong sparse match
        assert_eq!(results[0].id, 2, "expected id=2 as top hybrid result");
        assert_eq!(col.sparse_count(), 3);
    }

    #[test]
    fn collection_hybrid_search_with_filter() {
        use crate::index::sparse::SparseVector;

        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("hybrid", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        col.upsert_hybrid(
            1, vec![1.0, 0.0, 0.0],
            Some(SparseVector::new(vec![0], vec![1.0])),
            Some(serde_json::json!({"category": "tech"})),
        ).unwrap();
        col.upsert_hybrid(
            2, vec![0.9, 0.1, 0.0],
            Some(SparseVector::new(vec![0], vec![0.5])),
            Some(serde_json::json!({"category": "news"})),
        ).unwrap();

        let filter: FilterCondition = serde_json::from_str(r#"{"category": {"$eq": "tech"}}"#).unwrap();
        let results = col.search_hybrid(
            &[1.0, 0.0, 0.0],
            &SparseVector::new(vec![0], vec![1.0]),
            10,
            0.5,
            0.5,
            Some(&filter),
        ).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn collection_hybrid_persists_across_restart() {
        use crate::index::sparse::SparseVector;

        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("hybrid", IndexType::Flat);

        {
            let mut col = Collection::create(&path, meta).unwrap();
            col.upsert_hybrid(
                1, vec![1.0, 0.0, 0.0],
                Some(SparseVector::new(vec![0, 1], vec![1.0, 0.5])),
                Some(serde_json::json!({"tag": "a"})),
            ).unwrap();
            col.upsert_hybrid(
                2, vec![0.0, 1.0, 0.0],
                Some(SparseVector::new(vec![1, 2], vec![0.8, 0.3])),
                None,
            ).unwrap();
        }

        // Reload and verify sparse data survives WAL replay
        let col = Collection::load(&path).unwrap();
        assert_eq!(col.count(), 2);
        assert_eq!(col.sparse_count(), 2);

        let results = col.search_hybrid(
            &[1.0, 0.0, 0.0],
            &SparseVector::new(vec![0], vec![1.0]),
            1,
            0.5,
            0.5,
            None,
        ).unwrap();
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn collection_hybrid_delete_removes_sparse() {
        use crate::index::sparse::SparseVector;

        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("hybrid", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        col.upsert_hybrid(1, vec![1.0, 0.0, 0.0], Some(SparseVector::new(vec![0], vec![1.0])), None).unwrap();
        col.upsert_hybrid(2, vec![0.0, 1.0, 0.0], Some(SparseVector::new(vec![1], vec![1.0])), None).unwrap();
        assert_eq!(col.sparse_count(), 2);

        col.delete(1).unwrap();
        assert_eq!(col.sparse_count(), 1);
        assert_eq!(col.count(), 1);
    }

    #[test]
    fn collection_dense_only_upsert_works_with_hybrid_search() {
        use crate::index::sparse::SparseVector;

        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("dense_hybrid", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        // Regular dense-only upsert
        col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();
        col.upsert(2, vec![0.0, 1.0, 0.0], None).unwrap();

        // Hybrid search with empty sparse should still work (all sparse scores = 0).
        let results = col.search_hybrid(
            &[1.0, 0.0, 0.0],
            &SparseVector::new(vec![0], vec![1.0]),
            2,
            1.0,
            0.0,
            None,
        ).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 1); // closest dense
    }

    // ── Snapshot tests ────────────────────────────────────────────────────────

    #[test]
    fn snapshot_create_and_list() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("snap_test", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        col.upsert(1, vec![1.0, 0.0, 0.0], Some(serde_json::json!({"k": "v"}))).unwrap();
        col.upsert(2, vec![0.0, 1.0, 0.0], None).unwrap();

        let snap = col.create_snapshot("v1").unwrap();
        assert_eq!(snap.name, "v1");
        assert_eq!(snap.vector_count, 2);

        let snaps = col.list_snapshots().unwrap();
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0].name, "v1");
    }

    #[test]
    fn snapshot_list_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("snap_test", IndexType::Flat);
        let col = Collection::create(&path, meta).unwrap();
        let snaps = col.list_snapshots().unwrap();
        assert!(snaps.is_empty());
    }

    #[test]
    fn snapshot_restore() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("snap_test", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();
        col.upsert(2, vec![0.0, 1.0, 0.0], None).unwrap();
        col.create_snapshot("v1").unwrap();

        // Mutate after snapshot
        col.upsert(3, vec![0.0, 0.0, 1.0], None).unwrap();
        col.delete(1).unwrap();
        assert_eq!(col.count(), 2); // ids 2, 3

        // Restore
        col.restore_snapshot("v1").unwrap();
        assert_eq!(col.count(), 2); // ids 1, 2 again
        let results = col.search(&[1.0, 0.0, 0.0], 1, None).unwrap();
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn snapshot_delete() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("snap_test", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();
        col.create_snapshot("v1").unwrap();

        col.delete_snapshot("v1").unwrap();
        let snaps = col.list_snapshots().unwrap();
        assert!(snaps.is_empty());
    }

    #[test]
    fn snapshot_duplicate_name_errors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("snap_test", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();
        col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();

        col.create_snapshot("dup").unwrap();
        let err = col.create_snapshot("dup").unwrap_err();
        assert!(matches!(err, VectorDbError::SnapshotAlreadyExists(_)));
    }

    #[test]
    fn snapshot_invalid_name_errors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("snap_test", IndexType::Flat);
        let col = Collection::create(&path, meta).unwrap();

        // Empty name
        assert!(col.create_snapshot("").is_err());
        // Path traversal
        assert!(col.create_snapshot("..").is_err());
        assert!(col.create_snapshot("a/../b").is_err());
        // Slash
        assert!(col.create_snapshot("a/b").is_err());
        // Backslash
        assert!(col.create_snapshot("a\\b").is_err());
    }

    #[test]
    fn snapshot_restore_nonexistent_errors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("snap_test", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        let err = col.restore_snapshot("nope").unwrap_err();
        assert!(matches!(err, VectorDbError::SnapshotNotFound(_)));
    }

    #[test]
    fn snapshot_delete_nonexistent_errors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("snap_test", IndexType::Flat);
        let col = Collection::create(&path, meta).unwrap();

        let err = col.delete_snapshot("nope").unwrap_err();
        assert!(matches!(err, VectorDbError::SnapshotNotFound(_)));
    }

    #[test]
    fn snapshot_multiple_listed() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("snap_test", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();
        col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();

        col.create_snapshot("alpha").unwrap();
        col.create_snapshot("beta").unwrap();

        let snaps = col.list_snapshots().unwrap();
        assert_eq!(snaps.len(), 2);
        let names: Vec<&str> = snaps.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
    }

    // ── Batch upsert tests ────────────────────────────────────────────────────

    #[test]
    fn upsert_batch_basic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("batch", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        let entries = vec![
            (1, vec![1.0, 0.0, 0.0], None),
            (2, vec![0.0, 1.0, 0.0], None),
            (3, vec![0.0, 0.0, 1.0], None),
        ];
        col.upsert_batch(entries).unwrap();
        assert_eq!(col.count(), 3);

        let results = col.search(&[1.0, 0.0, 0.0], 1, None).unwrap();
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn upsert_batch_with_payloads() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("batch", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        let entries = vec![
            (1, vec![1.0, 0.0, 0.0], Some(serde_json::json!({"a": 1}))),
            (2, vec![0.0, 1.0, 0.0], Some(serde_json::json!({"a": 2}))),
        ];
        col.upsert_batch(entries).unwrap();

        let results = col.search(&[1.0, 0.0, 0.0], 2, None).unwrap();
        let r1 = results.iter().find(|r| r.id == 1).unwrap();
        assert_eq!(r1.payload.as_ref().unwrap()["a"], 1);
    }

    #[test]
    fn upsert_batch_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("batch", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        col.upsert_batch(vec![]).unwrap();
        assert_eq!(col.count(), 0);
    }

    #[test]
    fn upsert_batch_duplicate_ids_last_wins() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("batch", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        let entries = vec![
            (1, vec![1.0, 0.0, 0.0], Some(serde_json::json!({"v": "first"}))),
            (1, vec![0.0, 1.0, 0.0], Some(serde_json::json!({"v": "second"}))),
        ];
        col.upsert_batch(entries).unwrap();
        assert_eq!(col.count(), 1);

        let results = col.search(&[0.0, 1.0, 0.0], 1, None).unwrap();
        assert_eq!(results[0].id, 1);
        assert_eq!(results[0].payload.as_ref().unwrap()["v"], "second");
    }

    #[test]
    fn upsert_batch_persists_across_restart() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("batch", IndexType::Flat);

        {
            let mut col = Collection::create(&path, meta).unwrap();
            let entries = vec![
                (1, vec![1.0, 0.0, 0.0], None),
                (2, vec![0.0, 1.0, 0.0], None),
            ];
            col.upsert_batch(entries).unwrap();
        }

        let col = Collection::load(&path).unwrap();
        assert_eq!(col.count(), 2);
    }

    // ── Embedder tests ────────────────────────────────────────────────────────

    struct MockEmbedder {
        dims: usize,
    }

    impl crate::embedder::TextEmbedder for MockEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(vec![text.len() as f32 / 10.0; self.dims])
        }
        fn dimensions(&self) -> Option<usize> {
            Some(self.dims)
        }
    }

    struct FailingEmbedder;

    impl crate::embedder::TextEmbedder for FailingEmbedder {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
            Err("model unavailable".into())
        }
        fn dimensions(&self) -> Option<usize> {
            None
        }
    }

    #[test]
    fn set_embedder_dimension_mismatch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("emb", IndexType::Flat); // dims=3
        let mut col = Collection::create(&path, meta).unwrap();

        let embedder = Arc::new(MockEmbedder { dims: 128 }); // mismatched
        let err = col.set_embedder(embedder).unwrap_err();
        assert!(matches!(err, VectorDbError::InvalidConfig(_)));
    }

    #[test]
    fn set_embedder_matching_dims_succeeds() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("emb", IndexType::Flat); // dims=3
        let mut col = Collection::create(&path, meta).unwrap();

        let embedder = Arc::new(MockEmbedder { dims: 3 });
        col.set_embedder(embedder).unwrap();
    }

    #[test]
    fn upsert_text_without_embedder_errors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("emb", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        let err = col.upsert_text(1, "hello", None).unwrap_err();
        assert!(matches!(err, VectorDbError::NoEmbedder(_)));
    }

    #[test]
    fn search_text_without_embedder_errors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("emb", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();
        col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();

        let err = col.search_text("hello", 5).unwrap_err();
        assert!(matches!(err, VectorDbError::NoEmbedder(_)));
    }

    #[test]
    fn upsert_text_and_search_text_with_embedder() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("emb", IndexType::Flat); // dims=3
        let mut col = Collection::create(&path, meta).unwrap();

        let embedder = Arc::new(MockEmbedder { dims: 3 });
        col.set_embedder(embedder).unwrap();

        col.upsert_text(1, "hi", Some(serde_json::json!({"text": "hi"}))).unwrap();
        col.upsert_text(2, "hello world", None).unwrap();
        assert_eq!(col.count(), 2);

        // Search for text similar to "hi" (len=2 → [0.2, 0.2, 0.2])
        let results = col.search_text("hi", 2).unwrap();
        assert!(!results.is_empty());
        // "hi" (len=2) should be closest to query "hi" (len=2)
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn embedding_error_propagates() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("emb", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        let embedder = Arc::new(FailingEmbedder);
        col.set_embedder(embedder).unwrap(); // unknown dims, so no mismatch check

        let err = col.upsert_text(1, "test", None).unwrap_err();
        assert!(matches!(err, VectorDbError::EmbeddingError(_)));
    }

    // ── Search edge cases ─────────────────────────────────────────────────────

    #[test]
    fn search_k_zero_returns_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("test", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();
        col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();

        let results = col.search(&[1.0, 0.0, 0.0], 0, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_filter_matches_nothing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("test", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        col.upsert(1, vec![1.0, 0.0, 0.0], Some(serde_json::json!({"tag": "a"}))).unwrap();
        col.upsert(2, vec![0.0, 1.0, 0.0], Some(serde_json::json!({"tag": "a"}))).unwrap();

        let filter: FilterCondition = serde_json::from_str(r#"{"tag": {"$eq": "nonexistent"}}"#).unwrap();
        let results = col.search(&[1.0, 0.0, 0.0], 5, Some(&filter)).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_filter_returns_fewer_than_k() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("test", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        col.upsert(1, vec![1.0, 0.0, 0.0], Some(serde_json::json!({"tag": "a"}))).unwrap();
        col.upsert(2, vec![0.0, 1.0, 0.0], Some(serde_json::json!({"tag": "b"}))).unwrap();
        col.upsert(3, vec![0.0, 0.0, 1.0], Some(serde_json::json!({"tag": "b"}))).unwrap();

        let filter: FilterCondition = serde_json::from_str(r#"{"tag": {"$eq": "a"}}"#).unwrap();
        let results = col.search(&[1.0, 0.0, 0.0], 10, Some(&filter)).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn search_empty_collection() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("test", IndexType::Flat);
        let col = Collection::create(&path, meta).unwrap();

        let results = col.search(&[1.0, 0.0, 0.0], 5, None).unwrap();
        assert!(results.is_empty());
    }

    // ── Delete behavior ───────────────────────────────────────────────────────

    #[test]
    fn delete_returns_true_when_found() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("test", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();
        let found = col.delete(1).unwrap();
        assert!(found);
    }

    #[test]
    fn delete_returns_false_when_not_found() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("test", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        let found = col.delete(999).unwrap();
        assert!(!found);
    }

    #[test]
    fn double_delete_returns_false_second_time() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("test", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();
        assert!(col.delete(1).unwrap());
        assert!(!col.delete(1).unwrap());
        assert_eq!(col.count(), 0);
    }

    #[test]
    fn delete_removes_payload() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("test", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        col.upsert(1, vec![1.0, 0.0, 0.0], Some(serde_json::json!({"k": "v"}))).unwrap();
        col.delete(1).unwrap();

        // Re-add without payload — old payload should not reappear
        col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();
        let results = col.search(&[1.0, 0.0, 0.0], 1, None).unwrap();
        assert!(results[0].payload.is_none());
    }

    // ── Hybrid search normalization edge case ─────────────────────────────────

    #[test]
    fn hybrid_search_identical_dense_scores() {
        use crate::index::sparse::SparseVector;

        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("hybrid_norm", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        // Two identical dense vectors — dense distances will be equal (min == max)
        col.upsert_hybrid(
            1, vec![1.0, 0.0, 0.0],
            Some(SparseVector::new(vec![0], vec![1.0])),
            None,
        ).unwrap();
        col.upsert_hybrid(
            2, vec![1.0, 0.0, 0.0],
            Some(SparseVector::new(vec![0], vec![0.5])),
            None,
        ).unwrap();

        // Should not panic or produce NaN despite dense_range → max(eps, 0)
        let results = col.search_hybrid(
            &[1.0, 0.0, 0.0],
            &SparseVector::new(vec![0], vec![1.0]),
            2,
            0.5,
            0.5,
            None,
        ).unwrap();
        assert_eq!(results.len(), 2);
        // All scores should be finite
        for r in &results {
            assert!(r.score.is_finite(), "score should be finite, got {}", r.score);
        }
        // id=1 has higher sparse score, so should rank first
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn hybrid_search_k_zero_returns_empty() {
        use crate::index::sparse::SparseVector;

        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("hybrid", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();
        col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();

        let results = col.search_hybrid(
            &[1.0, 0.0, 0.0],
            &SparseVector::new(vec![0], vec![1.0]),
            0,
            0.5,
            0.5,
            None,
        ).unwrap();
        assert!(results.is_empty());
    }

    // ── Count method ──────────────────────────────────────────────────────────

    #[test]
    fn count_reflects_upserts_and_deletes() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("test", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        assert_eq!(col.count(), 0);
        col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();
        assert_eq!(col.count(), 1);
        col.upsert(2, vec![0.0, 1.0, 0.0], None).unwrap();
        assert_eq!(col.count(), 2);
        // Re-upsert same id should not increase count
        col.upsert(1, vec![0.5, 0.5, 0.0], None).unwrap();
        assert_eq!(col.count(), 2);
        col.delete(1).unwrap();
        assert_eq!(col.count(), 1);
        col.delete(2).unwrap();
        assert_eq!(col.count(), 0);
    }

    // ── Multiple index types ──────────────────────────────────────────────────

    #[test]
    fn create_with_quantized_flat() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("qf", IndexType::QuantizedFlat);
        let mut col = Collection::create(&path, meta).unwrap();

        col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();
        col.upsert(2, vec![0.0, 1.0, 0.0], None).unwrap();
        assert_eq!(col.count(), 2);

        let results = col.search(&[1.0, 0.0, 0.0], 1, None).unwrap();
        assert_eq!(results[0].id, 1);

        // Verify persistence
        drop(col);
        let loaded = Collection::load(&path).unwrap();
        assert_eq!(loaded.count(), 2);
        assert_eq!(loaded.meta().index_type, IndexType::QuantizedFlat);
    }

    #[test]
    fn create_with_fp16_flat() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("fp16", IndexType::Fp16Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();
        col.upsert(2, vec![0.0, 1.0, 0.0], None).unwrap();
        assert_eq!(col.count(), 2);

        let results = col.search(&[1.0, 0.0, 0.0], 1, None).unwrap();
        assert_eq!(results[0].id, 1);

        drop(col);
        let loaded = Collection::load(&path).unwrap();
        assert_eq!(loaded.count(), 2);
        assert_eq!(loaded.meta().index_type, IndexType::Fp16Flat);
    }

    #[test]
    fn create_with_binary_flat() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("bin", IndexType::BinaryFlat);
        let mut col = Collection::create(&path, meta).unwrap();

        col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();
        col.upsert(2, vec![0.0, 1.0, 0.0], None).unwrap();
        assert_eq!(col.count(), 2);

        let results = col.search(&[1.0, 0.0, 0.0], 1, None).unwrap();
        // Binary quantization is lossy; just verify we get a result
        assert!(!results.is_empty());

        drop(col);
        let loaded = Collection::load(&path).unwrap();
        assert_eq!(loaded.count(), 2);
        assert_eq!(loaded.meta().index_type, IndexType::BinaryFlat);
    }

    #[test]
    fn create_with_hnsw() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let mut meta = make_meta("hnsw", IndexType::Hnsw);
        meta.hnsw_config = Some(HnswConfig::default());
        let mut col = Collection::create(&path, meta).unwrap();

        col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();
        col.upsert(2, vec![0.0, 1.0, 0.0], None).unwrap();
        col.upsert(3, vec![0.0, 0.0, 1.0], None).unwrap();
        assert_eq!(col.count(), 3);

        let results = col.search(&[1.0, 0.0, 0.0], 1, None).unwrap();
        assert_eq!(results[0].id, 1);
    }

    // ── Payload upsert-clears-old-payload ─────────────────────────────────────

    #[test]
    fn upsert_with_none_payload_clears_previous() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("test", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        col.upsert(1, vec![1.0, 0.0, 0.0], Some(serde_json::json!({"k": "v"}))).unwrap();
        col.upsert(1, vec![1.0, 0.0, 0.0], None).unwrap();

        let results = col.search(&[1.0, 0.0, 0.0], 1, None).unwrap();
        assert!(results[0].payload.is_none());
    }

    // ── Snapshot with payloads ────────────────────────────────────────────────

    #[test]
    fn snapshot_preserves_payloads() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("snap_pay", IndexType::Flat);
        let mut col = Collection::create(&path, meta).unwrap();

        col.upsert(1, vec![1.0, 0.0, 0.0], Some(serde_json::json!({"tag": "snap"}))).unwrap();
        col.create_snapshot("v1").unwrap();

        // Mutate payload
        col.upsert(1, vec![1.0, 0.0, 0.0], Some(serde_json::json!({"tag": "changed"}))).unwrap();

        // Restore and verify original payload
        col.restore_snapshot("v1").unwrap();
        let results = col.search(&[1.0, 0.0, 0.0], 1, None).unwrap();
        assert_eq!(results[0].payload.as_ref().unwrap()["tag"], "snap");
    }

    // ── Meta accessor ─────────────────────────────────────────────────────────

    #[test]
    fn meta_accessor_returns_correct_values() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col");
        let meta = make_meta("my_col", IndexType::Flat);
        let col = Collection::create(&path, meta).unwrap();

        assert_eq!(col.meta().name, "my_col");
        assert_eq!(col.meta().dimensions, 3);
        assert_eq!(col.meta().metric, Metric::L2);
        assert_eq!(col.meta().index_type, IndexType::Flat);
    }
}
