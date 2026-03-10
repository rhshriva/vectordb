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
        WalEntry::Add { id, vector: vec, payload_bytes: None }
    }

    fn add_entry_with_payload(id: u64, vec: Vec<f32>, payload: serde_json::Value) -> WalEntry {
        WalEntry::Add {
            id,
            vector: vec,
            payload_bytes: Some(serde_json::to_vec(&payload).unwrap()),
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
}
