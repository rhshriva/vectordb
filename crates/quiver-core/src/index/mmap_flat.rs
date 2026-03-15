//! Memory-mapped flat index — brute-force search backed by a memory-mapped file.
//!
//! # Why mmap?
//!
//! A [`FlatIndex`][crate::index::flat::FlatIndex] loads all vectors into a
//! `HashMap<u64, Vec<f32>>` at startup.  For 100 M × 1536-dim vectors that is
//! ~600 GB of heap — impossible on most machines.
//!
//! `MmapFlatIndex` stores vectors in a **dense binary file** and memory-maps it.
//! The OS brings in only the pages that are actually read.  During a sequential
//! scan only a few 4 KB pages are resident at any time, so you can scan 100 M
//! vectors on an 8 GB machine.
//!
//! # File format
//!
//! ```text
//! ┌────────────────────────────────────── Header (32 bytes) ──────────────────┐
//! │ magic   : [0x51, 0x56, 0x45, 0x43]  "QVEC"   (4 bytes)                  │
//! │ version : u32 LE                               (4 bytes)                  │
//! │ dims    : u32 LE                               (4 bytes)                  │
//! │ metric  : u32 LE  0=L2, 1=Cosine, 2=DotProd   (4 bytes)                  │
//! │ count   : u64 LE  number of records             (8 bytes)                  │
//! │ _pad    : u64 LE  reserved                      (8 bytes)                  │
//! └────────────────────────────────────────────────────────────────────────────┘
//! ┌──────────────── Record × count  (8 + 4·dims bytes each) ──────────────────┐
//! │ id     : u64 LE                                                            │
//! │ vector : f32[dims]  (IEEE 754 little-endian)                               │
//! └────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Mutation model
//!
//! - **Inserts** go into an in-memory `staging` buffer (no file I/O).
//! - **Deletes** mark the ID in a `deleted` tombstone set (no file I/O).
//! - **`flush()`** appends staged vectors to the file, remaps, and clears staging.
//!   If there are deletions, the file is fully rewritten first to compact tombstones.
//! - **Search** iterates the mmap region (skipping tombstones) then the staging buffer.
//!
//! This design means that the mmap file only grows on `flush()`.  Between
//! flushes the staging buffer is bounded in size by the WAL compact threshold.

use std::collections::{HashMap, HashSet};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::{
    distance::Metric,
    error::VectorDbError,
    index::{IndexConfig, SearchResult, VectorIndex},
};

// ── File format constants ──────────────────────────────────────────────────────

const MAGIC: &[u8; 4] = b"QVEC";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 32;

fn metric_to_u32(m: Metric) -> u32 {
    match m {
        Metric::L2 => 0,
        Metric::Cosine => 1,
        Metric::DotProduct => 2,
    }
}

fn u32_to_metric(v: u32) -> Option<Metric> {
    match v {
        0 => Some(Metric::L2),
        1 => Some(Metric::Cosine),
        2 => Some(Metric::DotProduct),
        _ => None,
    }
}

/// Byte stride of a single record in the mmap file.
#[inline]
fn record_stride(dims: usize) -> usize {
    8 + 4 * dims // u64 id + f32 × dims
}

// ── MmapFlatIndex ──────────────────────────────────────────────────────────────

/// Brute-force exact index backed by a memory-mapped binary file.
///
/// Supports datasets larger than available RAM by letting the OS page in only
/// the vector data that is currently being scanned.
pub struct MmapFlatIndex {
    config: IndexConfig,
    data_path: PathBuf,

    /// Memory-mapped view of the on-disk vector data.
    mmap: Option<Mmap>,
    /// Number of records in the current mmap file.
    file_count: usize,

    /// Vectors inserted since the last `flush()` — not yet on disk.
    staging: HashMap<u64, Vec<f32>>,
    /// IDs that have been deleted but not yet compacted out of the file.
    deleted: HashSet<u64>,
}

impl MmapFlatIndex {
    /// Open (or create) a mmap-backed flat index at `path`.
    ///
    /// - If `path` exists and is a valid Quiver vector file, it is mapped immediately.
    /// - If `path` does not exist, an empty index is created; the file is written on
    ///   the first `flush()` call.
    pub fn new(dimensions: usize, metric: Metric, path: impl Into<PathBuf>)
        -> Result<Self, VectorDbError>
    {
        let data_path = path.into();
        let (mmap, file_count) = if data_path.exists() {
            let (mmap, count) = open_mmap(&data_path, dimensions, metric)?;
            (Some(mmap), count)
        } else {
            (None, 0)
        };
        Ok(Self {
            config: IndexConfig { dimensions, metric },
            data_path,
            mmap,
            file_count,
            staging: HashMap::new(),
            deleted: HashSet::new(),
        })
    }

    /// Bulk-insert from a contiguous f32 buffer (row-major, `n_rows × dim`).
    /// Assigns sequential IDs starting from `start_id`.
    pub fn add_batch_raw(&mut self, raw_data: &[f32], dim: usize, start_id: u64) -> Result<(), VectorDbError> {
        if dim != self.config.dimensions {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.config.dimensions,
                got: dim,
            });
        }
        if raw_data.len() % dim != 0 {
            return Err(VectorDbError::InvalidConfig(
                format!("raw_data length {} is not a multiple of dim {}", raw_data.len(), dim),
            ));
        }
        let n = raw_data.len() / dim;
        for i in 0..n {
            let slice = &raw_data[i * dim..(i + 1) * dim];
            let id = start_id + i as u64;
            self.add(id, slice)?;
        }
        Ok(())
    }

    /// Total vectors: those in the file (minus deleted) plus staging.
    fn file_live_count(&self) -> usize {
        self.file_count.saturating_sub(self.deleted.len())
    }

    // ── Internal helpers ───────────────────────────────────────────────────────

    /// Iterate over (id, vector_slice) pairs in the mmap file, skipping deleted.
    fn iter_file<'a>(&'a self) -> Box<dyn Iterator<Item = (u64, &'a [f32])> + 'a> {
        let dims = self.config.dimensions;
        let stride = record_stride(dims);
        let mmap = match &self.mmap {
            Some(m) => m,
            None => return Box::new(std::iter::empty()),
        };
        let count = self.file_count;
        let deleted = &self.deleted;

        Box::new((0..count).filter_map(move |i| {
            let offset = HEADER_SIZE + i * stride;
            let id_bytes = &mmap[offset..offset + 8];
            let id = u64::from_le_bytes(id_bytes.try_into().unwrap());
            if deleted.contains(&id) {
                return None;
            }
            // Cast the vector bytes to &[f32] — safe because:
            // 1. The file format writes f32 as IEEE 754 LE, same as Rust's f32.
            // 2. The data is 4-byte aligned (header is 32 bytes, stride = 8 + 4*dims).
            let vec_bytes = &mmap[offset + 8..offset + stride];
            // Safety: vec_bytes.len() == 4*dims, alignment guaranteed by file format.
            let vec_f32 = unsafe {
                std::slice::from_raw_parts(vec_bytes.as_ptr() as *const f32, dims)
            };
            Some((id, vec_f32))
        }))
    }

    /// Rewrite the entire file with only live vectors (compacts tombstones).
    fn compact_file(&mut self) -> Result<(), VectorDbError> {
        let live: Vec<(u64, Vec<f32>)> = self
            .iter_file()
            .map(|(id, v)| (id, v.to_vec()))
            .chain(self.staging.iter().map(|(&id, v)| (id, v.clone())))
            .collect();

        write_file(&self.data_path, self.config.dimensions, self.config.metric, &live)?;
        self.mmap = Some(open_mmap_raw(&self.data_path)?);
        self.file_count = live.len();
        self.staging.clear();
        self.deleted.clear();
        Ok(())
    }

    /// Append staging vectors to the file without rewriting existing records.
    fn append_staging(&mut self) -> Result<(), VectorDbError> {
        if self.staging.is_empty() {
            return Ok(());
        }

        if self.mmap.is_none() {
            // No file yet — write a fresh one.
            let live: Vec<(u64, Vec<f32>)> =
                self.staging.iter().map(|(&id, v)| (id, v.clone())).collect();
            write_file(&self.data_path, self.config.dimensions, self.config.metric, &live)?;
            self.file_count = live.len();
            self.staging.clear();
            self.mmap = Some(open_mmap_raw(&self.data_path)?);
            return Ok(());
        }

        // Append new records to the end of the existing file, then update header count.
        {
            let new_count = self.file_count + self.staging.len();
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .open(&self.data_path)?;
            use std::io::Seek;
            let append_pos = (HEADER_SIZE + self.file_count * record_stride(self.config.dimensions))
                as u64;
            file.seek(std::io::SeekFrom::Start(append_pos))?;

            let dims = self.config.dimensions;
            for (id, vec) in &self.staging {
                file.write_all(&id.to_le_bytes())?;
                for &x in vec {
                    file.write_all(&x.to_le_bytes())?;
                }
            }
            // Update header count field (bytes 16..24).
            file.seek(std::io::SeekFrom::Start(16))?;
            file.write_all(&(new_count as u64).to_le_bytes())?;
            file.flush()?;
            self.file_count = new_count;
            let _ = dims;
        }

        self.staging.clear();
        // Re-map the (now larger) file.
        self.mmap = Some(open_mmap_raw(&self.data_path)?);
        Ok(())
    }
}

impl VectorIndex for MmapFlatIndex {
    fn add(&mut self, id: u64, vector: &[f32]) -> Result<(), VectorDbError> {
        if vector.len() != self.config.dimensions {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }
        // Remove from deleted set if re-inserting a previously deleted id.
        self.deleted.remove(&id);
        self.staging.insert(id, vector.to_vec());
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, VectorDbError> {
        if query.len() != self.config.dimensions {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len(),
            });
        }
        if k == 0 {
            return Ok(vec![]);
        }

        let metric = self.config.metric;

        // Scan mmap file.
        let mut results: Vec<SearchResult> = self
            .iter_file()
            .map(|(id, v)| SearchResult { id, distance: metric.distance(query, v) })
            .collect();

        // Also scan staging (recent inserts not yet flushed to file).
        results.extend(
            self.staging
                .iter()
                .filter(|(&id, _)| !self.deleted.contains(&id))
                .map(|(&id, v)| SearchResult { id, distance: metric.distance(query, v) }),
        );

        let k = k.min(results.len());
        if k == 0 {
            return Ok(vec![]);
        }
        results.select_nth_unstable_by(k - 1, |a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        results.sort_unstable_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }

    fn delete(&mut self, id: u64) -> bool {
        let in_staging = self.staging.remove(&id).is_some();
        // Check if it's in the mmap file (don't iterate, just check id_to_list via
        // a tombstone set — O(1)).
        let in_file = self.file_count > 0 && !self.deleted.contains(&id) && {
            // Scan to verify the id actually exists in the file.
            self.iter_file().any(|(fid, _)| fid == id)
        };
        if in_file {
            self.deleted.insert(id);
        }
        in_staging || in_file
    }

    fn len(&self) -> usize {
        self.file_live_count() + self.staging.len()
    }

    fn config(&self) -> &IndexConfig {
        &self.config
    }

    fn iter_vectors(&self) -> Box<dyn Iterator<Item = (u64, Vec<f32>)> + '_> {
        let file_iter = self.iter_file().map(|(id, v)| (id, v.to_vec()));
        let stage_iter = self
            .staging
            .iter()
            .filter(|(&id, _)| !self.deleted.contains(&id))
            .map(|(&id, v)| (id, v.clone()));
        Box::new(file_iter.chain(stage_iter))
    }

    /// Flush staging buffer to the mmap file.
    ///
    /// - If there are no deletions: appends staging records to the file.
    /// - If there are pending deletions: fully rewrites the file to compact tombstones.
    fn flush(&mut self) {
        if !self.deleted.is_empty() {
            let _ = self.compact_file();
        } else if !self.staging.is_empty() {
            let _ = self.append_staging();
        }
    }
}

// ── File I/O helpers ───────────────────────────────────────────────────────────

/// Write a complete vector file to `path`.
fn write_file(
    path: &Path,
    dims: usize,
    metric: Metric,
    vectors: &[(u64, Vec<f32>)],
) -> Result<(), VectorDbError> {
    let file = std::fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    // Header
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(dims as u32).to_le_bytes())?;
    w.write_all(&metric_to_u32(metric).to_le_bytes())?;
    w.write_all(&(vectors.len() as u64).to_le_bytes())?;
    w.write_all(&0u64.to_le_bytes())?; // padding

    // Records
    for (id, vec) in vectors {
        w.write_all(&id.to_le_bytes())?;
        for &x in vec {
            w.write_all(&x.to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}

/// Open an existing file as a read-only mmap, validating the header.
fn open_mmap(path: &Path, dims: usize, metric: Metric) -> Result<(Mmap, usize), VectorDbError> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| VectorDbError::Io(e))?;

    if mmap.len() < HEADER_SIZE {
        return Err(VectorDbError::Serialization("mmap file too small".into()));
    }
    if &mmap[0..4] != MAGIC {
        return Err(VectorDbError::Serialization("invalid QVEC magic".into()));
    }
    let file_dims = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;
    if file_dims != dims {
        return Err(VectorDbError::DimensionMismatch {
            expected: dims,
            got: file_dims,
        });
    }
    let file_metric_u32 = u32::from_le_bytes(mmap[12..16].try_into().unwrap());
    let file_metric = u32_to_metric(file_metric_u32)
        .ok_or_else(|| VectorDbError::Serialization(format!("unknown metric id {file_metric_u32}")))?;
    if file_metric != metric {
        return Err(VectorDbError::InvalidConfig(
            format!("metric mismatch: file={file_metric:?}, expected={metric:?}")
        ));
    }
    let count = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
    Ok((mmap, count))
}

/// Open a file as read-only mmap without header validation (for internal use after write).
fn open_mmap_raw(path: &Path) -> Result<Mmap, VectorDbError> {
    let file = std::fs::File::open(path)?;
    unsafe { Mmap::map(&file) }.map_err(|e| VectorDbError::Io(e))
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_index(dir: &std::path::Path) -> MmapFlatIndex {
        MmapFlatIndex::new(3, Metric::L2, dir.join("vecs.mmap")).unwrap()
    }

    #[test]
    fn add_and_search_before_flush() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = make_index(dir.path());
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0]).unwrap();
        idx.add(3, &[0.0, 0.0, 1.0]).unwrap();
        // All vectors are in staging — search should still work
        let r = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 1);
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn flush_writes_to_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vecs.mmap");
        {
            let mut idx = MmapFlatIndex::new(3, Metric::L2, &path).unwrap();
            idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
            idx.add(2, &[0.0, 1.0, 0.0]).unwrap();
            idx.flush();
            assert!(path.exists(), "file should exist after flush");
            assert_eq!(idx.file_count, 2);
            assert!(idx.staging.is_empty());
        }
    }

    #[test]
    fn search_after_flush_from_mmap() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vecs.mmap");
        {
            let mut idx = MmapFlatIndex::new(3, Metric::L2, &path).unwrap();
            idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
            idx.add(2, &[0.0, 1.0, 0.0]).unwrap();
            idx.flush();
        }
        // Reopen and search (data comes from mmap file now)
        let idx = MmapFlatIndex::new(3, Metric::L2, &path).unwrap();
        assert_eq!(idx.file_count, 2);
        let r = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 1);
    }

    #[test]
    fn delete_from_staging_removes_vector() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = make_index(dir.path());
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        assert!(idx.delete(1));
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn delete_from_file_adds_tombstone() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vecs.mmap");
        let mut idx = MmapFlatIndex::new(3, Metric::L2, &path).unwrap();
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0]).unwrap();
        idx.flush();

        // Delete from file
        assert!(idx.delete(1));
        assert_eq!(idx.len(), 1);
        // Should not appear in search
        let r = idx.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(!r.iter().any(|x| x.id == 1));
    }

    #[test]
    fn flush_with_deletions_compacts_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vecs.mmap");
        let mut idx = MmapFlatIndex::new(3, Metric::L2, &path).unwrap();
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0]).unwrap();
        idx.flush();

        idx.delete(1);
        idx.flush(); // compact

        assert_eq!(idx.file_count, 1);
        assert!(idx.deleted.is_empty());
        let r = idx.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].id, 2);
    }

    #[test]
    fn incremental_append() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vecs.mmap");
        let mut idx = MmapFlatIndex::new(3, Metric::L2, &path).unwrap();

        for i in 0..10u64 {
            idx.add(i, &[i as f32, 0.0, 0.0]).unwrap();
            idx.flush(); // flush after each insert
        }
        assert_eq!(idx.file_count, 10);
        let r = idx.search(&[5.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 5);
    }

    #[test]
    fn iter_vectors_yields_all_live() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vecs.mmap");
        let mut idx = MmapFlatIndex::new(3, Metric::L2, &path).unwrap();
        for i in 0..5u64 {
            idx.add(i, &[i as f32, 0.0, 0.0]).unwrap();
        }
        idx.flush();
        idx.delete(2);

        let mut ids: Vec<u64> = idx.iter_vectors().map(|(id, _)| id).collect();
        ids.sort();
        assert_eq!(ids, vec![0, 1, 3, 4]);
    }

    #[test]
    fn dimension_mismatch_errors() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = make_index(dir.path());
        let err = idx.add(1, &[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, VectorDbError::DimensionMismatch { expected: 3, got: 2 }));
    }

    #[test]
    fn cosine_and_dot_product_work() {
        let dir = tempfile::tempdir().unwrap();
        for metric in [Metric::Cosine, Metric::DotProduct] {
            let path = dir.path().join(format!("vecs_{metric:?}.mmap"));
            let mut idx = MmapFlatIndex::new(2, metric, &path).unwrap();
            idx.add(1, &[1.0, 0.0]).unwrap();
            idx.add(2, &[0.0, 1.0]).unwrap();
            idx.flush();
            let r = idx.search(&[1.0, 0.0], 1).unwrap();
            assert_eq!(r[0].id, 1);
        }
    }

    // ── Additional coverage tests ─────────────────────────────────────────────

    #[test]
    fn add_batch_raw_basic() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = make_index(dir.path());
        // 4 vectors of dim 3
        let raw = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ];
        idx.add_batch_raw(&raw, 3, 10).unwrap();
        assert_eq!(idx.len(), 4);
        let r = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 10);
    }

    #[test]
    fn add_batch_raw_flush_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("batch.mmap");
        let mut idx = MmapFlatIndex::new(3, Metric::L2, &path).unwrap();
        let raw = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        idx.add_batch_raw(&raw, 3, 0).unwrap();
        idx.flush();
        assert_eq!(idx.file_count, 2);
        // Reopen
        let idx2 = MmapFlatIndex::new(3, Metric::L2, &path).unwrap();
        assert_eq!(idx2.len(), 2);
        let r = idx2.search(&[0.0, 1.0, 0.0], 1).unwrap();
        assert_eq!(r[0].id, 1);
    }

    #[test]
    fn add_batch_raw_dim_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = make_index(dir.path());
        let raw = vec![1.0, 2.0, 3.0, 4.0];
        let result = idx.add_batch_raw(&raw, 2, 0);
        assert!(matches!(result, Err(VectorDbError::DimensionMismatch { expected: 3, got: 2 })));
    }

    #[test]
    fn add_batch_raw_not_multiple_of_dim() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = make_index(dir.path());
        let raw = vec![1.0, 2.0, 3.0, 4.0]; // 4 not divisible by 3
        let result = idx.add_batch_raw(&raw, 3, 0);
        assert!(matches!(result, Err(VectorDbError::InvalidConfig(_))));
    }

    #[test]
    fn search_dimension_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = make_index(dir.path());
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        let result = idx.search(&[1.0, 0.0], 1);
        assert!(matches!(result, Err(VectorDbError::DimensionMismatch { expected: 3, got: 2 })));
    }

    #[test]
    fn search_k_zero_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = make_index(dir.path());
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        let r = idx.search(&[1.0, 0.0, 0.0], 0).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn search_empty_index() {
        let dir = tempfile::tempdir().unwrap();
        let idx = make_index(dir.path());
        let r = idx.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn search_k_greater_than_len() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = make_index(dir.path());
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        let r = idx.search(&[1.0, 0.0, 0.0], 100).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].id, 1);
    }

    #[test]
    fn delete_nonexistent_returns_false() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = make_index(dir.path());
        assert!(!idx.delete(999));
    }

    #[test]
    fn delete_nonexistent_from_file_returns_false() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vecs.mmap");
        let mut idx = MmapFlatIndex::new(3, Metric::L2, &path).unwrap();
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.flush();
        assert!(!idx.delete(999));
    }

    #[test]
    fn config_returns_correct_values() {
        let dir = tempfile::tempdir().unwrap();
        let idx = make_index(dir.path());
        let cfg = idx.config();
        assert_eq!(cfg.dimensions, 3);
        assert_eq!(cfg.metric, Metric::L2);
    }

    #[test]
    fn flush_noop_when_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vecs.mmap");
        let mut idx = MmapFlatIndex::new(3, Metric::L2, &path).unwrap();
        // flush on empty index should be a no-op, no file created
        idx.flush();
        assert!(!path.exists());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn reinsert_deleted_id_via_staging() {
        // When an id is deleted then re-added, the deleted set is cleared so the
        // new staging version is visible in search. This exercises the
        // `self.deleted.remove(&id)` path in `add()`.
        let dir = tempfile::tempdir().unwrap();
        let mut idx = make_index(dir.path());
        // Add to staging, delete, re-add
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        assert!(idx.delete(1));
        assert_eq!(idx.len(), 0);
        idx.add(1, &[0.0, 0.0, 1.0]).unwrap();
        assert_eq!(idx.len(), 1);
        let r = idx.search(&[0.0, 0.0, 1.0], 1).unwrap();
        assert_eq!(r[0].id, 1);
        assert!(r[0].distance < 1e-6);
    }

    #[test]
    fn iter_vectors_file_and_staging_combined() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vecs.mmap");
        let mut idx = MmapFlatIndex::new(3, Metric::L2, &path).unwrap();
        // Add to file
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0]).unwrap();
        idx.flush();
        // Add to staging
        idx.add(3, &[0.0, 0.0, 1.0]).unwrap();
        let mut ids: Vec<u64> = idx.iter_vectors().map(|(id, _)| id).collect();
        ids.sort();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn compact_preserves_staging_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vecs.mmap");
        let mut idx = MmapFlatIndex::new(3, Metric::L2, &path).unwrap();
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0]).unwrap();
        idx.flush();
        // Delete from file and add a new staging vector
        idx.delete(1);
        idx.add(3, &[0.0, 0.0, 1.0]).unwrap();
        idx.flush(); // compact: merges remaining file + staging
        assert_eq!(idx.file_count, 2); // id 2 from file + id 3 from staging
        assert!(idx.staging.is_empty());
        assert!(idx.deleted.is_empty());
        let mut ids: Vec<u64> = idx.iter_vectors().map(|(id, _)| id).collect();
        ids.sort();
        assert_eq!(ids, vec![2, 3]);
    }

    #[test]
    fn persistence_after_compact() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vecs.mmap");
        {
            let mut idx = MmapFlatIndex::new(3, Metric::L2, &path).unwrap();
            idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
            idx.add(2, &[0.0, 1.0, 0.0]).unwrap();
            idx.add(3, &[0.0, 0.0, 1.0]).unwrap();
            idx.flush();
            idx.delete(2);
            idx.flush(); // compact
        }
        // Reopen and verify
        let idx = MmapFlatIndex::new(3, Metric::L2, &path).unwrap();
        assert_eq!(idx.len(), 2);
        let mut ids: Vec<u64> = idx.iter_vectors().map(|(id, _)| id).collect();
        ids.sort();
        assert_eq!(ids, vec![1, 3]);
    }

    #[test]
    fn open_mmap_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_magic.mmap");
        std::fs::write(&path, &[0u8; 32]).unwrap();
        let result = MmapFlatIndex::new(3, Metric::L2, &path);
        assert!(matches!(result, Err(VectorDbError::Serialization(_))));
    }

    #[test]
    fn open_mmap_file_too_small() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("small.mmap");
        std::fs::write(&path, &[0u8; 10]).unwrap();
        let result = MmapFlatIndex::new(3, Metric::L2, &path);
        assert!(matches!(result, Err(VectorDbError::Serialization(_))));
    }

    #[test]
    fn open_mmap_dimension_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vecs.mmap");
        {
            let mut idx = MmapFlatIndex::new(3, Metric::L2, &path).unwrap();
            idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
            idx.flush();
        }
        let result = MmapFlatIndex::new(5, Metric::L2, &path);
        assert!(matches!(result, Err(VectorDbError::DimensionMismatch { expected: 5, got: 3 })));
    }

    #[test]
    fn open_mmap_metric_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vecs.mmap");
        {
            let mut idx = MmapFlatIndex::new(3, Metric::L2, &path).unwrap();
            idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
            idx.flush();
        }
        let result = MmapFlatIndex::new(3, Metric::Cosine, &path);
        assert!(matches!(result, Err(VectorDbError::InvalidConfig(_))));
    }

    #[test]
    fn open_mmap_unknown_metric() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_metric.mmap");
        let mut header = Vec::new();
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&VERSION.to_le_bytes());
        header.extend_from_slice(&3u32.to_le_bytes());
        header.extend_from_slice(&99u32.to_le_bytes()); // invalid metric
        header.extend_from_slice(&0u64.to_le_bytes());
        header.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(&path, &header).unwrap();
        let result = MmapFlatIndex::new(3, Metric::L2, &path);
        assert!(matches!(result, Err(VectorDbError::Serialization(_))));
    }

    #[test]
    fn u32_to_metric_all_variants() {
        assert_eq!(u32_to_metric(0), Some(Metric::L2));
        assert_eq!(u32_to_metric(1), Some(Metric::Cosine));
        assert_eq!(u32_to_metric(2), Some(Metric::DotProduct));
        assert_eq!(u32_to_metric(3), None);
        assert_eq!(u32_to_metric(255), None);
    }

    #[test]
    fn metric_to_u32_roundtrip() {
        for (m, expected) in [(Metric::L2, 0), (Metric::Cosine, 1), (Metric::DotProduct, 2)] {
            assert_eq!(metric_to_u32(m), expected);
            assert_eq!(u32_to_metric(expected), Some(m));
        }
    }

    #[test]
    fn search_after_all_deleted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vecs.mmap");
        let mut idx = MmapFlatIndex::new(3, Metric::L2, &path).unwrap();
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0]).unwrap();
        idx.flush();
        idx.delete(1);
        idx.delete(2);
        assert_eq!(idx.len(), 0);
        let r = idx.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn dot_product_search_after_flush() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dot.mmap");
        let mut idx = MmapFlatIndex::new(3, Metric::DotProduct, &path).unwrap();
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.5, 0.5, 0.0]).unwrap();
        idx.add(3, &[0.0, 0.0, 1.0]).unwrap();
        idx.flush();
        // Reopen and search
        let idx2 = MmapFlatIndex::new(3, Metric::DotProduct, &path).unwrap();
        let r = idx2.search(&[1.0, 0.0, 0.0], 1).unwrap();
        // Dot product: distance = -dot. id=1 has dot=1.0 (smallest negative distance)
        assert_eq!(r[0].id, 1);
    }

    #[test]
    fn multiple_flush_cycles() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vecs.mmap");
        let mut idx = MmapFlatIndex::new(3, Metric::L2, &path).unwrap();

        // Cycle 1: add + flush
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.flush();
        assert_eq!(idx.file_count, 1);

        // Cycle 2: add + flush (append)
        idx.add(2, &[0.0, 1.0, 0.0]).unwrap();
        idx.flush();
        assert_eq!(idx.file_count, 2);

        // Cycle 3: delete + add + flush (compact)
        idx.delete(1);
        idx.add(3, &[0.0, 0.0, 1.0]).unwrap();
        idx.flush();
        assert_eq!(idx.file_count, 2); // id 2 + id 3

        // Cycle 4: add + flush (append again after compact)
        idx.add(4, &[1.0, 1.0, 0.0]).unwrap();
        idx.flush();
        assert_eq!(idx.file_count, 3);

        let mut ids: Vec<u64> = idx.iter_vectors().map(|(id, _)| id).collect();
        ids.sort();
        assert_eq!(ids, vec![2, 3, 4]);
    }

    #[test]
    fn delete_already_deleted_returns_false() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vecs.mmap");
        let mut idx = MmapFlatIndex::new(3, Metric::L2, &path).unwrap();
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.flush();
        assert!(idx.delete(1));
        // Second delete of same id should return false
        assert!(!idx.delete(1));
    }

    #[test]
    fn add_batch_raw_empty() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = make_index(dir.path());
        idx.add_batch_raw(&[], 3, 0).unwrap();
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn record_stride_calculation() {
        assert_eq!(record_stride(3), 8 + 4 * 3); // 20
        assert_eq!(record_stride(128), 8 + 4 * 128); // 520
        assert_eq!(record_stride(1), 12);
    }

    #[test]
    fn file_live_count_accuracy() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vecs.mmap");
        let mut idx = MmapFlatIndex::new(3, Metric::L2, &path).unwrap();
        assert_eq!(idx.file_live_count(), 0);
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0]).unwrap();
        idx.add(3, &[0.0, 0.0, 1.0]).unwrap();
        idx.flush();
        assert_eq!(idx.file_live_count(), 3);
        idx.delete(1);
        assert_eq!(idx.file_live_count(), 2);
        idx.delete(2);
        assert_eq!(idx.file_live_count(), 1);
    }
}
