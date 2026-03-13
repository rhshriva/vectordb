"""Tests for BinaryFlatIndex (1-bit quantization)."""

import math
import os
import tempfile

import pytest
import quiver_vector_db as qv

from conftest import random_vectors, SAMPLE_DIM


class TestBinaryFlatIndexStandalone:
    """Standalone BinaryFlatIndex API tests."""

    def test_add_and_search(self):
        idx = qv.BinaryFlatIndex(dimensions=128, metric="cosine")
        vecs = random_vectors(100, 128)
        for i, v in enumerate(vecs):
            idx.add(i, v)
        assert len(idx) == 100
        results = idx.search(vecs[0], k=5)
        assert results[0]["id"] == 0  # self is closest

    def test_search_l2(self):
        idx = qv.BinaryFlatIndex(dimensions=64, metric="l2")
        vecs = random_vectors(50, 64)
        for i, v in enumerate(vecs):
            idx.add(i, v)
        results = idx.search(vecs[0], k=3)
        assert len(results) == 3
        assert results[0]["id"] == 0

    def test_delete(self):
        idx = qv.BinaryFlatIndex(dimensions=32, metric="cosine")
        idx.add(1, [float(i) for i in range(32)])
        idx.add(2, [float(i) for i in range(32)])
        assert len(idx) == 2
        assert idx.delete(1) is True
        assert idx.delete(999) is False
        assert len(idx) == 1

    def test_add_batch(self):
        idx = qv.BinaryFlatIndex(dimensions=64, metric="cosine")
        vecs = random_vectors(20, 64)
        entries = [(i, v) for i, v in enumerate(vecs)]
        idx.add_batch(entries)
        assert len(idx) == 20

    def test_dimension_mismatch(self):
        idx = qv.BinaryFlatIndex(dimensions=32, metric="l2")
        with pytest.raises(ValueError, match="dimension mismatch"):
            idx.add(1, [1.0, 2.0])  # wrong dimension

    def test_save_load(self):
        idx = qv.BinaryFlatIndex(dimensions=64, metric="cosine")
        vecs = random_vectors(30, 64)
        for i, v in enumerate(vecs):
            idx.add(i, v)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "binary.bin")
            idx.save(path)
            loaded = qv.BinaryFlatIndex.load(path)
            assert len(loaded) == 30
            r1 = idx.search(vecs[0], k=1)
            r2 = loaded.search(vecs[0], k=1)
            assert r1[0]["id"] == r2[0]["id"]
            assert abs(r1[0]["distance"] - r2[0]["distance"]) < 1e-6

    def test_properties(self):
        idx = qv.BinaryFlatIndex(dimensions=128, metric="dot_product")
        assert idx.dimensions == 128
        assert idx.metric == "dot_product"
        assert "BinaryFlatIndex" in repr(idx)


class TestBinaryFlatViaClient:
    """BinaryFlatIndex via Client.create_collection(index_type='binary_flat')."""

    def test_create_and_search(self):
        with tempfile.TemporaryDirectory() as d:
            db = qv.Client(path=d)
            col = db.create_collection("bin", dimensions=64, metric="cosine", index_type="binary_flat")
            vecs = random_vectors(50, 64)
            for i, v in enumerate(vecs):
                col.upsert(i, v)
            results = col.search(vecs[0], k=5)
            assert len(results) >= 1
            assert results[0]["id"] == 0

    def test_with_payload(self):
        with tempfile.TemporaryDirectory() as d:
            db = qv.Client(path=d)
            col = db.create_collection("bin", dimensions=32, metric="l2", index_type="binary_flat")
            col.upsert(1, [1.0] * 32, payload={"tag": "hello"})
            results = col.search([1.0] * 32, k=1)
            assert results[0]["id"] == 1
            assert results[0]["payload"]["tag"] == "hello"

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            db = qv.Client(path=d)
            col = db.create_collection("bin", dimensions=32, metric="cosine", index_type="binary_flat")
            col.upsert(1, [1.0] * 32)
            col.upsert(2, [-1.0] * 32)
            del db, col

            db2 = qv.Client(path=d)
            col2 = db2.get_collection("bin")
            assert col2.count == 2
