use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::error::VectorDbError;

/// A single WAL entry — either adding or deleting a vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WalEntry {
    Add {
        id: u64,
        vector: Vec<f32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        payload: Option<serde_json::Value>,
    },
    Delete {
        id: u64,
    },
}

/// Append-only NDJSON WAL writer.
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

    /// Append one entry. Flushes the buffer to ensure kernel receipt.
    pub fn append(&mut self, entry: &WalEntry) -> Result<(), VectorDbError> {
        let line = serde_json::to_string(entry)?;
        self.writer.write_all(line.as_bytes())?;
        self.writer.write_all(b"\n")?;
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
    /// A partial final line (power-loss crash) is skipped with a warning.
    pub fn replay(
        path: impl AsRef<Path>,
        mut visitor: impl FnMut(WalEntry),
    ) -> Result<(), VectorDbError> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(());
        }
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        for (idx, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<WalEntry>(&line) {
                Ok(entry) => visitor(entry),
                Err(e) => {
                    // Partial/corrupt last line — skip with a warning
                    warn!("WAL: skipping corrupt entry at line {}: {}", idx, e);
                }
            }
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
            let line = serde_json::to_string(&entry)?;
            writer.write_all(line.as_bytes())?;
            writer.write_all(b"\n")?;
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
        WalEntry::Add { id, vector: vec, payload: None }
    }

    fn add_entry_with_payload(id: u64, vec: Vec<f32>, payload: serde_json::Value) -> WalEntry {
        WalEntry::Add { id, vector: vec, payload: Some(payload) }
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
        assert!(matches!(entries[3], WalEntry::Delete { id: 2 }));
    }

    #[test]
    fn wal_handles_partial_last_line() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.log");

        let mut wal = Wal::open(&path).unwrap();
        wal.append(&add_entry(1, vec![1.0])).unwrap();
        wal.append(&add_entry(2, vec![2.0])).unwrap();
        drop(wal);

        // Corrupt the last byte (simulate truncation)
        let mut content = std::fs::read(&path).unwrap();
        content.pop(); // remove trailing newline or part of JSON
        content.pop();
        std::fs::write(&path, &content).unwrap();

        let mut entries = Vec::new();
        Wal::replay(&path, |e| entries.push(e)).unwrap();
        // First entry must survive; second may be corrupt
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

        // Compact to only 3 live entries
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
            WalEntry::Add { payload: Some(p), .. } => {
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
