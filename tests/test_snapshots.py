"""Tests for data versioning / snapshots."""

import tempfile

import pytest
import quiver_vector_db as qv

from conftest import random_vectors


class TestSnapshots:
    """Snapshot create / list / restore / delete tests."""

    def _make_collection(self, db, name="test", n=10, dim=32):
        col = db.create_collection(name, dimensions=dim, metric="cosine")
        vecs = random_vectors(n, dim)
        for i, v in enumerate(vecs):
            col.upsert(i, v, payload={"idx": i})
        return col, vecs

    def test_create_and_list(self):
        with tempfile.TemporaryDirectory() as d:
            db = qv.Client(path=d)
            col, _ = self._make_collection(db)
            meta = col.create_snapshot("v1")
            assert meta["name"] == "v1"
            assert meta["vector_count"] == 10
            snaps = col.list_snapshots()
            assert len(snaps) == 1
            assert snaps[0]["name"] == "v1"

    def test_restore_rolls_back(self):
        with tempfile.TemporaryDirectory() as d:
            db = qv.Client(path=d)
            col, vecs = self._make_collection(db, n=5)
            col.create_snapshot("before_insert")
            assert col.count == 5

            # Insert more vectors
            extra = random_vectors(5, 32, seed=99)
            for i, v in enumerate(extra):
                col.upsert(100 + i, v)
            assert col.count == 10

            # Restore
            col.restore_snapshot("before_insert")
            assert col.count == 5

            # Verify original vectors are searchable
            results = col.search(vecs[0], k=1)
            assert results[0]["id"] == 0

    def test_restore_preserves_payloads(self):
        with tempfile.TemporaryDirectory() as d:
            db = qv.Client(path=d)
            col, _ = self._make_collection(db, n=3)
            col.create_snapshot("snap")

            # Mutate payloads
            col.upsert(0, random_vectors(1, 32, seed=77)[0], payload={"idx": 999})
            col.restore_snapshot("snap")

            results = col.search(random_vectors(1, 32)[0], k=3)
            payloads = {r["id"]: r["payload"]["idx"] for r in results}
            for rid, idx in payloads.items():
                assert idx == rid  # original payload preserved

    def test_delete_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            db = qv.Client(path=d)
            col, _ = self._make_collection(db)
            col.create_snapshot("tmp")
            assert len(col.list_snapshots()) == 1
            col.delete_snapshot("tmp")
            assert len(col.list_snapshots()) == 0

    def test_duplicate_snapshot_raises(self):
        with tempfile.TemporaryDirectory() as d:
            db = qv.Client(path=d)
            col, _ = self._make_collection(db)
            col.create_snapshot("dup")
            with pytest.raises(KeyError, match="snapshot already exists"):
                col.create_snapshot("dup")

    def test_restore_nonexistent_raises(self):
        with tempfile.TemporaryDirectory() as d:
            db = qv.Client(path=d)
            col, _ = self._make_collection(db)
            with pytest.raises(KeyError, match="snapshot not found"):
                col.restore_snapshot("ghost")

    def test_delete_nonexistent_raises(self):
        with tempfile.TemporaryDirectory() as d:
            db = qv.Client(path=d)
            col, _ = self._make_collection(db)
            with pytest.raises(KeyError, match="snapshot not found"):
                col.delete_snapshot("ghost")

    def test_multiple_snapshots(self):
        with tempfile.TemporaryDirectory() as d:
            db = qv.Client(path=d)
            col, _ = self._make_collection(db, n=5)
            col.create_snapshot("v1")
            extra = random_vectors(5, 32, seed=99)
            for i, v in enumerate(extra):
                col.upsert(100 + i, v)
            col.create_snapshot("v2")

            snaps = col.list_snapshots()
            assert len(snaps) == 2
            assert snaps[0]["vector_count"] == 5
            assert snaps[1]["vector_count"] == 10

            # Restore to v1
            col.restore_snapshot("v1")
            assert col.count == 5

    def test_snapshot_survives_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            db = qv.Client(path=d)
            col, _ = self._make_collection(db, n=5)
            col.create_snapshot("persist")
            del db, col

            db2 = qv.Client(path=d)
            col2 = db2.get_collection("test")
            snaps = col2.list_snapshots()
            assert len(snaps) == 1
            assert snaps[0]["name"] == "persist"
