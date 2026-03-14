/// Write-ahead log using length-prefixed bincode frames.
///
/// ## Wire format
/// Each entry is stored as:
/// ```text
/// [ u32 LE length ][ bincode-encoded WalEntry bytes ]
/// ```
/// This is compact, fast, and crash-safe: a partial write at the end of the
/// file produces an unreadable frame that the replayer skips with a warning —
/// exactly the same semantics as the former NDJSON format.
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::error::VectorDbError;

/// A single WAL entry — either adding or deleting a vector.
///
/// The payload is stored as raw JSON bytes (`Vec<u8>`) rather than
/// `serde_json::Value` because bincode is a non-self-describing format and
/// cannot round-trip types that rely on `Deserializer::deserialize_any`
/// (which `serde_json::Value` uses).  Callers convert to/from
/// `serde_json::Value` at the collection layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalEntry {
    Add {
        id: u64,
        vector: Vec<f32>,
        /// JSON-encoded payload bytes, or `None` when there is no payload.
        payload_bytes: Option<Vec<u8>>,
        /// Bincode-encoded sparse vector bytes, or `None` when not a hybrid insert.
        /// Added in V2 — older WALs will deserialize this as `None` via `#[serde(default)]`.
        #[serde(default)]
        sparse_bytes: Option<Vec<u8>>,
    },
    Delete {
        id: u64,
    },
}

/// Append-only binary WAL writer.
pub struct Wal {
    #[allow(dead_code)]
    path: PathBuf,
    writer: BufWriter<File>,
    entry_count: usize,
}

impl Wal {
    /// Open or create the WAL file at `path` in append mode.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, VectorDbError> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            path,
            writer: BufWriter::new(file),
            entry_count: 0,
        })
    }

    /// Append one entry as a length-prefixed bincode frame. Flushes immediately.
    pub fn append(&mut self, entry: &WalEntry) -> Result<(), VectorDbError> {
        let bytes = bincode::serialize(entry)
            .map_err(|e| VectorDbError::Serialization(e.to_string()))?;
        let len = bytes.len() as u32;
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&bytes)?;
        self.writer.flush()?;
        self.entry_count += 1;
        Ok(())
    }

    /// Append a vector-add entry by borrowing data — avoids the `vector.clone()`
    /// required by [`WalEntry::Add`]. Wire format is identical.
    /// Flushes immediately (use [`append_add_no_flush`] for batch operations).
    pub fn append_add(
        &mut self,
        id: u64,
        vector: &[f32],
        payload_bytes: Option<&[u8]>,
        sparse_bytes: Option<&[u8]>,
    ) -> Result<(), VectorDbError> {
        self.write_add_frame(id, vector, payload_bytes, sparse_bytes)?;
        self.writer.flush()?;
        self.entry_count += 1;
        Ok(())
    }

    /// Like [`append_add`] but does NOT flush — for batch operations.
    /// Call [`flush`] after the entire batch has been written.
    pub fn append_add_no_flush(
        &mut self,
        id: u64,
        vector: &[f32],
        payload_bytes: Option<&[u8]>,
        sparse_bytes: Option<&[u8]>,
    ) -> Result<(), VectorDbError> {
        self.write_add_frame(id, vector, payload_bytes, sparse_bytes)?;
        self.entry_count += 1;
        Ok(())
    }

    /// Explicitly flush the underlying writer. Call after a batch of
    /// `append_add_no_flush` calls.
    pub fn flush(&mut self) -> Result<(), VectorDbError> {
        self.writer.flush()?;
        Ok(())
    }

    /// Internal: serialize an Add entry into a length-prefixed bincode frame
    /// and write it to the BufWriter **without allocating an owned Vec<f32>**.
    ///
    /// Wire format matches `bincode::serialize(&WalEntry::Add { .. })` exactly:
    /// ```text
    /// [ u32 variant_tag=0 ]
    /// [ u64 id ]
    /// [ u64 vec_len ][ f32 × vec_len ]
    /// [ u8 option_tag ][ u64 len ][ u8 × len ]   // payload_bytes
    /// [ u8 option_tag ][ u64 len ][ u8 × len ]   // sparse_bytes
    /// ```
    fn write_add_frame(
        &mut self,
        id: u64,
        vector: &[f32],
        payload_bytes: Option<&[u8]>,
        sparse_bytes: Option<&[u8]>,
    ) -> Result<(), VectorDbError> {
        // Compute frame size upfront to write the length prefix.
        let vec_bytes_len = vector.len() * std::mem::size_of::<f32>();
        let payload_size = match payload_bytes {
            Some(b) => 1 + 8 + b.len(), // option tag(1) + length(8) + data
            None => 1,                   // option tag only
        };
        let sparse_size = match sparse_bytes {
            Some(b) => 1 + 8 + b.len(),
            None => 1,
        };
        // variant_tag(4) + id(8) + vec_len(8) + vec_data + payload + sparse
        let frame_size = 4 + 8 + 8 + vec_bytes_len + payload_size + sparse_size;

        // Write length prefix
        self.writer.write_all(&(frame_size as u32).to_le_bytes())?;
        // Variant tag: Add = 0
        self.writer.write_all(&0u32.to_le_bytes())?;
        // id
        self.writer.write_all(&id.to_le_bytes())?;
        // vector: length-prefixed f32 sequence
        self.writer.write_all(&(vector.len() as u64).to_le_bytes())?;
        // SAFETY: reinterpret &[f32] as &[u8] — f32 is Pod, no padding
        let vec_as_bytes = unsafe {
            std::slice::from_raw_parts(
                vector.as_ptr() as *const u8,
                vec_bytes_len,
            )
        };
        self.writer.write_all(vec_as_bytes)?;

        // payload_bytes: Option<Vec<u8>> in bincode = {0 for None, 1 + len + data for Some}
        Self::write_option_bytes(&mut self.writer, payload_bytes)?;
        // sparse_bytes
        Self::write_option_bytes(&mut self.writer, sparse_bytes)?;

        Ok(())
    }

    /// Write a bincode-encoded `Option<Vec<u8>>` from a borrowed `Option<&[u8]>`.
    fn write_option_bytes(w: &mut BufWriter<File>, opt: Option<&[u8]>) -> Result<(), io::Error> {
        match opt {
            None => {
                w.write_all(&[0u8])?; // None tag
            }
            Some(b) => {
                w.write_all(&[1u8])?; // Some tag
                w.write_all(&(b.len() as u64).to_le_bytes())?;
                w.write_all(b)?;
            }
        }
        Ok(())
    }

    /// Number of entries written since this `Wal` was opened.
    pub fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Reset the entry count (called after compaction).
    pub fn reset_entry_count(&mut self, count: usize) {
        self.entry_count = count;
    }

    /// Replay all valid entries from `path`, calling `visitor` for each.
    /// A partial final frame (power-loss crash) is skipped with a warning.
    pub fn replay(
        path: impl AsRef<Path>,
        mut visitor: impl FnMut(WalEntry),
    ) -> Result<(), VectorDbError> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(());
        }
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut frame_idx: usize = 0;

        loop {
            // Read the 4-byte length prefix.
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break, // clean EOF
                Err(e) => return Err(VectorDbError::Io(e)),
            }
            let len = u32::from_le_bytes(len_buf) as usize;

            // Sanity-check: reject absurdly large frames (>256 MB) as corruption.
            if len > 256 * 1024 * 1024 {
                warn!("WAL: frame {} has unreasonable length {len}; stopping replay", frame_idx);
                break;
            }

            // Read the payload.
            let mut buf = vec![0u8; len];
            match reader.read_exact(&mut buf) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    warn!("WAL: frame {} is truncated (expected {len} bytes); skipping", frame_idx);
                    break;
                }
                Err(e) => return Err(VectorDbError::Io(e)),
            }

            match bincode::deserialize::<WalEntry>(&buf) {
                Ok(entry) => visitor(entry),
                Err(e) => {
                    warn!("WAL: corrupt frame {} — {e}; skipping", frame_idx);
                }
            }
            frame_idx += 1;
        }
        Ok(())
    }

    /// Atomically compact the WAL: write `live_entries` to a temp file,
    /// then rename to the canonical WAL path.
    pub fn compact(
        path: impl AsRef<Path>,
        live_entries: impl Iterator<Item = WalEntry>,
    ) -> Result<usize, VectorDbError> {
        let path = path.as_ref();
        let tmp_path = path.with_extension("log.tmp");

        let tmp_file = File::create(&tmp_path)?;
        let mut writer = BufWriter::new(tmp_file);
        let mut count = 0;
        for entry in live_entries {
            let bytes = bincode::serialize(&entry)
                .map_err(|e| VectorDbError::Serialization(e.to_string()))?;
            let len = bytes.len() as u32;
            writer.write_all(&len.to_le_bytes())?;
            writer.write_all(&bytes)?;
            count += 1;
        }
        writer.flush()?;
        drop(writer);

        // Atomic rename
        std::fs::rename(&tmp_path, path)?;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn add_entry(id: u64, vec: Vec<f32>) -> WalEntry {
        WalEntry::Add { id, vector: vec, payload_bytes: None, sparse_bytes: None }
    }

    fn add_entry_with_payload(id: u64, vec: Vec<f32>, payload: serde_json::Value) -> WalEntry {
        WalEntry::Add {
            id,
            vector: vec,
            payload_bytes: Some(serde_json::to_vec(&payload).unwrap()),
            sparse_bytes: None,
        }
    }

    #[test]
    fn wal_append_and_replay_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.log");

        let mut wal = Wal::open(&path).unwrap();
        wal.append(&add_entry(1, vec![1.0, 0.0])).unwrap();
        wal.append(&add_entry(2, vec![0.0, 1.0])).unwrap();
        wal.append(&add_entry(3, vec![0.5, 0.5])).unwrap();
        wal.append(&WalEntry::Delete { id: 2 }).unwrap();
        assert_eq!(wal.entry_count(), 4);

        let mut entries = Vec::new();
        Wal::replay(&path, |e| entries.push(e)).unwrap();
        assert_eq!(entries.len(), 4);
        assert!(matches!(entries[0], WalEntry::Add { id: 1, .. }));
        assert!(matches!(entries[3], WalEntry::Delete { id: 2, .. }));
    }

    #[test]
    fn wal_handles_partial_last_frame() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.log");

        let mut wal = Wal::open(&path).unwrap();
        wal.append(&add_entry(1, vec![1.0])).unwrap();
        wal.append(&add_entry(2, vec![2.0])).unwrap();
        drop(wal);

        // Corrupt the tail (simulate power-loss truncation)
        let mut content = std::fs::read(&path).unwrap();
        let truncate_to = content.len().saturating_sub(3);
        content.truncate(truncate_to);
        std::fs::write(&path, &content).unwrap();

        let mut entries = Vec::new();
        Wal::replay(&path, |e| entries.push(e)).unwrap();
        // First entry must survive
        assert!(!entries.is_empty());
        assert!(matches!(entries[0], WalEntry::Add { id: 1, .. }));
    }

    #[test]
    fn wal_compact_replaces_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.log");

        let mut wal = Wal::open(&path).unwrap();
        for i in 0..10u64 {
            wal.append(&add_entry(i, vec![i as f32])).unwrap();
        }
        drop(wal);

        let live: Vec<WalEntry> = (0..3u64)
            .map(|i| add_entry(i, vec![i as f32]))
            .collect();
        let count = Wal::compact(&path, live.into_iter()).unwrap();
        assert_eq!(count, 3);

        let mut entries = Vec::new();
        Wal::replay(&path, |e| entries.push(e)).unwrap();
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn wal_payload_survives_replay() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.log");

        let mut wal = Wal::open(&path).unwrap();
        wal.append(&add_entry_with_payload(
            1,
            vec![1.0],
            serde_json::json!({"tag": "news"}),
        ))
        .unwrap();
        drop(wal);

        let mut entries = Vec::new();
        Wal::replay(&path, |e| entries.push(e)).unwrap();
        match &entries[0] {
            WalEntry::Add { payload_bytes: Some(b), .. } => {
                let p: serde_json::Value = serde_json::from_slice(b).unwrap();
                assert_eq!(p["tag"], "news");
            }
            _ => panic!("expected Add with payload"),
        }
    }

    #[test]
    fn wal_missing_file_replay_is_ok() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent.log");
        let mut entries = Vec::new();
        Wal::replay(&path, |e| entries.push(e)).unwrap();
        assert!(entries.is_empty());
    }

    /// Verify that `append_add` produces wire-compatible frames that `replay`
    /// (which uses `bincode::deserialize`) can read back correctly.
    #[test]
    fn wal_append_add_wire_compatible_with_replay() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.log");
        let payload = serde_json::json!({"color": "blue"});
        let payload_bytes = serde_json::to_vec(&payload).unwrap();

        let mut wal = Wal::open(&path).unwrap();
        // Write one entry with standard append, one with append_add
        wal.append(&WalEntry::Add {
            id: 1,
            vector: vec![1.0, 2.0, 3.0],
            payload_bytes: Some(payload_bytes.clone()),
            sparse_bytes: None,
        }).unwrap();
        wal.append_add(2, &[4.0, 5.0, 6.0], Some(&payload_bytes), None).unwrap();
        // Also test with no payload
        wal.append_add(3, &[7.0, 8.0], None, None).unwrap();
        // Test with sparse bytes
        let sparse = vec![0xDE, 0xAD];
        wal.append_add(4, &[9.0], Some(&payload_bytes), Some(&sparse)).unwrap();
        drop(wal);

        let mut entries = Vec::new();
        Wal::replay(&path, |e| entries.push(e)).unwrap();
        assert_eq!(entries.len(), 4);

        // Entry 1 (standard append)
        match &entries[0] {
            WalEntry::Add { id, vector, payload_bytes: Some(pb), sparse_bytes: None } => {
                assert_eq!(*id, 1);
                assert_eq!(vector, &[1.0, 2.0, 3.0]);
                let p: serde_json::Value = serde_json::from_slice(pb).unwrap();
                assert_eq!(p["color"], "blue");
            }
            _ => panic!("entry 0 mismatch"),
        }
        // Entry 2 (append_add with payload)
        match &entries[1] {
            WalEntry::Add { id, vector, payload_bytes: Some(pb), sparse_bytes: None } => {
                assert_eq!(*id, 2);
                assert_eq!(vector, &[4.0, 5.0, 6.0]);
                let p: serde_json::Value = serde_json::from_slice(pb).unwrap();
                assert_eq!(p["color"], "blue");
            }
            _ => panic!("entry 1 mismatch"),
        }
        // Entry 3 (append_add without payload)
        match &entries[2] {
            WalEntry::Add { id, vector, payload_bytes: None, sparse_bytes: None } => {
                assert_eq!(*id, 3);
                assert_eq!(vector, &[7.0, 8.0]);
            }
            _ => panic!("entry 2 mismatch"),
        }
        // Entry 4 (append_add with payload + sparse)
        match &entries[3] {
            WalEntry::Add { id, vector, payload_bytes: Some(_), sparse_bytes: Some(sb) } => {
                assert_eq!(*id, 4);
                assert_eq!(vector, &[9.0]);
                assert_eq!(sb, &[0xDE, 0xAD]);
            }
            _ => panic!("entry 3 mismatch"),
        }
    }

    /// Verify that `append_add_no_flush` + `flush` produces valid entries.
    #[test]
    fn wal_batch_no_flush_then_flush() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.log");

        let mut wal = Wal::open(&path).unwrap();
        for i in 0..100u64 {
            let vec = vec![i as f32; 4];
            wal.append_add_no_flush(i, &vec, None, None).unwrap();
        }
        wal.flush().unwrap();
        assert_eq!(wal.entry_count(), 100);
        drop(wal);

        let mut entries = Vec::new();
        Wal::replay(&path, |e| entries.push(e)).unwrap();
        assert_eq!(entries.len(), 100);
        // Spot-check first and last
        match &entries[0] {
            WalEntry::Add { id: 0, vector, .. } => assert_eq!(vector, &[0.0; 4]),
            _ => panic!("entry 0 mismatch"),
        }
        match &entries[99] {
            WalEntry::Add { id: 99, vector, .. } => assert_eq!(vector, &[99.0; 4]),
            _ => panic!("entry 99 mismatch"),
        }
    }
}
