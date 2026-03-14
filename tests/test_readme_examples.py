"""Integration tests for every feature documented in the README and PyPI page.

Each test corresponds to a numbered README section and verifies that the
documented code examples actually work end-to-end.  Run with:

    pytest tests/test_readme_examples.py -v -s

No external services or API keys are required (sentence-transformers tests
use a mock embedder so they run without the optional dependency).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import urllib.request
import urllib.error

import numpy as np
import pytest

import quiver_vector_db as quiver
from quiver_vector_db import BM25, MultiVectorCollection, TextCollection


# ── Helpers ──────────────────────────────────────────────────────────────────

DIM = 128  # keep small for fast tests


def _rand(n: int = 1, dim: int = DIM) -> list[list[float]]:
    """Return n random vectors as lists of floats."""
    return np.random.randn(n, dim).astype(np.float32).tolist()


def _rand1(dim: int = DIM) -> list[float]:
    return _rand(1, dim)[0]


class MockEmbedding:
    """Deterministic mock embedding function for testing without sentence-transformers."""

    def __init__(self, dim: int = DIM):
        self._dim = dim

    def __call__(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for text in texts:
            np.random.seed(hash(text) % (2**31))
            vecs.append(np.random.randn(self._dim).astype(np.float32).tolist())
        return vecs

    @property
    def dimensions(self) -> int:
        return self._dim


@pytest.fixture
def tmp_path_str():
    """Provide a temporary directory that is cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="quiver_readme_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ── 1. Persistent Collection (WAL-backed) ───────────────────────────────────


class TestPersistentCollection:
    """README §1 — Client, create_collection, upsert, search, reopen."""

    def test_create_upsert_search(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        col = db.create_collection("docs", dimensions=DIM, metric="cosine")

        v1, v2 = _rand(2)
        col.upsert(id=1, vector=v1, payload={"title": "Hello world"})
        col.upsert(id=2, vector=v2, payload={"title": "Vector search"})

        assert col.count == 2

        hits = col.search(query=v1, k=5)
        assert len(hits) >= 1
        assert hits[0]["id"] == 1  # closest to itself
        assert "distance" in hits[0]
        assert "payload" in hits[0]
        assert hits[0]["payload"]["title"] == "Hello world"

    def test_reopen_persistence(self, tmp_path_str):
        v1 = _rand1()

        # First session: create and insert
        db1 = quiver.Client(path=tmp_path_str)
        col1 = db1.create_collection("persist", dimensions=DIM, metric="cosine")
        col1.upsert(id=42, vector=v1, payload={"key": "val"})
        assert col1.count == 1
        del col1, db1

        # Second session: reopen and verify
        db2 = quiver.Client(path=tmp_path_str)
        col2 = db2.get_collection("persist")
        assert col2.count == 1
        hits = col2.search(query=v1, k=1)
        assert hits[0]["id"] == 42
        assert hits[0]["payload"]["key"] == "val"

    def test_list_and_delete_collection(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        db.create_collection("a", dimensions=DIM, metric="l2")
        db.create_collection("b", dimensions=DIM, metric="l2")
        names = db.list_collections()
        assert set(names) == {"a", "b"}

        db.delete_collection("a")
        assert "a" not in db.list_collections()

    def test_get_or_create(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        col1 = db.get_or_create_collection("shared", dimensions=DIM, metric="cosine")
        col2 = db.get_or_create_collection("shared", dimensions=DIM, metric="cosine")
        col1.upsert(id=1, vector=_rand1())
        assert col2.count == 1  # same underlying collection


# ── 2. Text Search with Built-in Embeddings ─────────────────────────────────


class TestTextSearch:
    """README §2 — TextCollection with embedding + BM25."""

    def test_add_and_query_hybrid(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        col = db.create_collection("articles", dimensions=DIM, metric="cosine")
        text_col = TextCollection(col, MockEmbedding(DIM))

        text_col.add(
            ids=[1, 2, 3],
            documents=[
                "Introduction to machine learning",
                "Advanced deep learning techniques",
                "Cooking recipes from around the world",
            ],
        )
        assert text_col.count == 3

        # Hybrid mode (default)
        hits = text_col.query("machine learning", k=5)
        assert len(hits) >= 1
        assert any(h["id"] == 1 for h in hits)

    def test_semantic_mode(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        col = db.create_collection("sem", dimensions=DIM, metric="cosine")
        text_col = TextCollection(col, MockEmbedding(DIM))
        text_col.add(ids=[1, 2], documents=["hello world", "goodbye moon"])

        hits = text_col.query("hello", k=5, mode="semantic")
        assert len(hits) >= 1
        # Semantic mode returns distance, not score
        assert "distance" in hits[0] or "score" in hits[0]

    def test_keyword_mode(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        col = db.create_collection("kw", dimensions=DIM, metric="cosine")
        text_col = TextCollection(col, MockEmbedding(DIM))
        text_col.add(ids=[1, 2], documents=["quick brown fox", "lazy dog"])

        hits = text_col.query("fox", k=5, mode="keyword")
        assert len(hits) >= 1
        assert hits[0]["id"] == 1

    def test_delete_documents(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        col = db.create_collection("del", dimensions=DIM, metric="cosine")
        text_col = TextCollection(col, MockEmbedding(DIM))
        text_col.add(ids=[1, 2], documents=["aaa", "bbb"])
        assert text_col.count == 2

        text_col.delete([1])
        assert text_col.count == 1

    def test_add_with_payloads(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        col = db.create_collection("pay", dimensions=DIM, metric="cosine")
        text_col = TextCollection(col, MockEmbedding(DIM))
        text_col.add(
            ids=[1],
            documents=["hello"],
            payloads=[{"source": "test"}],
        )
        hits = text_col.query("hello", k=1, mode="semantic")
        assert hits[0]["payload"]["source"] == "test"


# ── 3. Filtered Search with Metadata ────────────────────────────────────────


class TestFilteredSearch:
    """README §3 — payload filter operators."""

    @pytest.fixture
    def col(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        col = db.create_collection("filter_test", dimensions=DIM, metric="cosine")
        # Insert 3 vectors with metadata
        vecs = _rand(3)
        col.upsert(id=1, vector=vecs[0], payload={"category": "tech", "rating": 4.8})
        col.upsert(id=2, vector=vecs[1], payload={"category": "science", "rating": 3.2})
        col.upsert(id=3, vector=vecs[2], payload={"category": "tech", "rating": 4.1})
        return col, vecs

    def test_eq_filter(self, col):
        c, vecs = col
        hits = c.search(query=vecs[0], k=5, filter={"category": {"$eq": "tech"}})
        assert all(h["payload"]["category"] == "tech" for h in hits)
        assert len(hits) == 2

    def test_gte_filter(self, col):
        c, vecs = col
        hits = c.search(query=vecs[0], k=5, filter={"rating": {"$gte": 4.0}})
        assert all(h["payload"]["rating"] >= 4.0 for h in hits)
        assert len(hits) == 2

    def test_compound_and_filter(self, col):
        c, vecs = col
        hits = c.search(
            query=vecs[0],
            k=5,
            filter={
                "$and": [
                    {"category": {"$in": ["tech", "science"]}},
                    {"rating": {"$gte": 4.0}},
                ]
            },
        )
        for h in hits:
            assert h["payload"]["category"] in ("tech", "science")
            assert h["payload"]["rating"] >= 4.0
        assert len(hits) == 2  # id=1 (tech, 4.8) and id=3 (tech, 4.1)

    def test_ne_filter(self, col):
        c, vecs = col
        hits = c.search(query=vecs[0], k=5, filter={"category": {"$ne": "tech"}})
        assert all(h["payload"]["category"] != "tech" for h in hits)

    def test_lt_filter(self, col):
        c, vecs = col
        hits = c.search(query=vecs[0], k=5, filter={"rating": {"$lt": 4.0}})
        assert all(h["payload"]["rating"] < 4.0 for h in hits)

    def test_or_filter(self, col):
        c, vecs = col
        hits = c.search(
            query=vecs[0],
            k=5,
            filter={
                "$or": [
                    {"rating": {"$gte": 4.8}},
                    {"category": {"$eq": "science"}},
                ]
            },
        )
        ids = {h["id"] for h in hits}
        assert 1 in ids  # rating 4.8
        assert 2 in ids  # science


# ── 4. Hybrid Dense + Sparse Search ─────────────────────────────────────────


class TestHybridSearch:
    """README §4 — upsert_hybrid + search_hybrid."""

    def test_hybrid_upsert_and_search(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        col = db.create_collection("hybrid", dimensions=DIM, metric="cosine")

        v1, v2 = _rand(2)
        col.upsert_hybrid(
            id=1,
            vector=v1,
            sparse_vector={42: 0.8, 100: 0.5, 3001: 0.3},
            payload={"title": "Rust guide"},
        )
        col.upsert_hybrid(
            id=2,
            vector=v2,
            sparse_vector={42: 0.1, 200: 0.9},
            payload={"title": "Python guide"},
        )

        assert col.count == 2
        assert col.sparse_count == 2

        hits = col.search_hybrid(
            dense_query=v1,
            sparse_query={42: 0.7, 100: 0.6},
            k=10,
            dense_weight=0.7,
            sparse_weight=0.3,
        )
        assert len(hits) >= 1
        assert "score" in hits[0]
        assert "dense_distance" in hits[0]
        assert "sparse_score" in hits[0]

    def test_hybrid_search_sparse_only_signal(self, tmp_path_str):
        """Sparse overlap should boost score."""
        db = quiver.Client(path=tmp_path_str)
        col = db.create_collection("hsp", dimensions=DIM, metric="cosine")

        # Use identical dense vectors so only sparse differs
        v = _rand1()
        col.upsert_hybrid(id=1, vector=v, sparse_vector={10: 1.0, 20: 1.0})
        col.upsert_hybrid(id=2, vector=v, sparse_vector={30: 1.0, 40: 1.0})

        hits = col.search_hybrid(
            dense_query=v,
            sparse_query={10: 1.0, 20: 1.0},
            k=2,
            dense_weight=0.0,
            sparse_weight=1.0,
        )
        # Id 1 has matching sparse dims, should rank higher
        assert hits[0]["id"] == 1


# ── 5. Batch Upsert ─────────────────────────────────────────────────────────


class TestBatchUpsert:
    """README §5 — upsert_batch."""

    def test_batch_with_payloads(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        col = db.create_collection("batch", dimensions=DIM, metric="cosine")

        vecs = _rand(3)
        col.upsert_batch([
            (1, vecs[0], {"title": "Doc A"}),
            (2, vecs[1], {"title": "Doc B"}),
            (3, vecs[2]),  # payload is optional
        ])
        assert col.count == 3

        hits = col.search(query=vecs[0], k=1)
        assert hits[0]["id"] == 1
        assert hits[0]["payload"]["title"] == "Doc A"

    def test_batch_overwrite(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        col = db.create_collection("bov", dimensions=DIM, metric="cosine")
        v = _rand1()
        col.upsert(id=1, vector=v, payload={"v": 1})
        col.upsert_batch([(1, v, {"v": 2})])
        hits = col.search(query=v, k=1)
        assert hits[0]["payload"]["v"] == 2

    def test_batch_large(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        col = db.create_collection("blg", dimensions=DIM, metric="l2")
        vecs = _rand(500)
        entries = [(i, vecs[i]) for i in range(500)]
        col.upsert_batch(entries)
        assert col.count == 500


# ── 6. Multi-Vector / Multi-Modal Search ────────────────────────────────────


class TestMultiVector:
    """README §6 — MultiVectorCollection."""

    def test_multi_upsert_and_search(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        multi = MultiVectorCollection(
            client=db,
            name="products",
            vector_spaces={
                "text": {"dimensions": DIM, "metric": "cosine"},
                "image": {"dimensions": DIM, "metric": "cosine"},
            },
        )

        tv1, tv2 = _rand(2)
        iv1, iv2 = _rand(2)
        multi.upsert(id=1, vectors={"text": tv1, "image": iv1}, payload={"title": "Blue T-Shirt"})
        multi.upsert(id=2, vectors={"text": tv2, "image": iv2}, payload={"title": "Red Jacket"})

        assert multi.count == 2

        # Single-space search
        hits = multi.search(vector_space="text", query=tv1, k=5)
        assert len(hits) >= 1
        assert hits[0]["id"] == 1

    def test_multi_fusion_search(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        multi = MultiVectorCollection(
            client=db,
            name="fusion",
            vector_spaces={
                "text": {"dimensions": DIM, "metric": "cosine"},
                "image": {"dimensions": DIM, "metric": "cosine"},
            },
        )

        tv, iv = _rand1(), _rand1()
        multi.upsert(id=1, vectors={"text": tv, "image": iv})
        multi.upsert(id=2, vectors={"text": _rand1(), "image": _rand1()})

        hits = multi.search_multi(
            queries={"text": tv, "image": iv},
            k=5,
            weights={"text": 0.6, "image": 0.4},
        )
        assert len(hits) >= 1
        assert hits[0]["id"] == 1
        assert "score" in hits[0]

    def test_multi_delete(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        multi = MultiVectorCollection(
            client=db,
            name="mdel",
            vector_spaces={
                "a": {"dimensions": DIM, "metric": "l2"},
                "b": {"dimensions": DIM, "metric": "l2"},
            },
        )
        multi.upsert(id=1, vectors={"a": _rand1(), "b": _rand1()})
        multi.upsert(id=2, vectors={"a": _rand1(), "b": _rand1()})
        assert multi.count == 2

        multi.delete(id=1)
        assert multi.count == 1

    def test_multi_batch_upsert(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        multi = MultiVectorCollection(
            client=db,
            name="mbatch",
            vector_spaces={"x": {"dimensions": DIM, "metric": "cosine"}},
        )
        multi.upsert_batch([
            (1, {"x": _rand1()}),
            (2, {"x": _rand1()}, {"tag": "b"}),
        ])
        assert multi.count == 2


# ── 7. Data Versioning / Snapshots ──────────────────────────────────────────


class TestSnapshots:
    """README §7 — create, list, restore, delete snapshots."""

    def test_snapshot_create_and_restore(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        col = db.create_collection("snap", dimensions=DIM, metric="cosine")

        # Insert initial data
        vecs = _rand(100)
        for i in range(100):
            col.upsert(id=i, vector=vecs[i], payload={"version": 1})

        snapshot = col.create_snapshot("v1")
        assert snapshot["name"] == "v1"
        assert snapshot["vector_count"] == 100

        # Mutate data
        for i in range(100, 200):
            col.upsert(id=i, vector=_rand1())
        assert col.count == 200

        # Restore
        col.restore_snapshot("v1")
        assert col.count == 100

    def test_list_snapshots(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        col = db.create_collection("slist", dimensions=DIM, metric="cosine")
        col.upsert(id=1, vector=_rand1())

        col.create_snapshot("a")
        col.create_snapshot("b")
        snaps = col.list_snapshots()
        names = {s["name"] for s in snaps}
        assert names == {"a", "b"}

    def test_delete_snapshot(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        col = db.create_collection("sdel", dimensions=DIM, metric="cosine")
        col.upsert(id=1, vector=_rand1())
        col.create_snapshot("tmp")
        col.delete_snapshot("tmp")
        assert len(col.list_snapshots()) == 0


# ── 8. Standalone Index (No WAL) ────────────────────────────────────────────


class TestStandaloneIndex:
    """README §8 — HnswIndex standalone, add_batch_np, parallel, save/load."""

    def test_hnsw_add_batch_np_and_search(self):
        idx = quiver.HnswIndex(dimensions=DIM, metric="l2", m=16, ef_construction=200)
        vectors = np.random.randn(1000, DIM).astype(np.float32)
        idx.add_batch_np(vectors)
        idx.flush()

        assert len(idx) == 1000
        results = idx.search(query=vectors[0].tolist(), k=10)
        assert len(results) == 10
        assert results[0]["id"] == 0  # nearest to itself
        assert results[0]["distance"] < 1e-6

    def test_hnsw_parallel_insert(self):
        idx = quiver.HnswIndex(dimensions=DIM, metric="l2")
        vectors = np.random.randn(2000, DIM).astype(np.float32)
        idx.add_batch_parallel(vectors, num_threads=4)
        idx.flush()

        assert len(idx) == 2000
        results = idx.search(query=vectors[0].tolist(), k=5)
        assert results[0]["id"] == 0

    def test_hnsw_save_and_load(self, tmp_path_str):
        idx = quiver.HnswIndex(dimensions=DIM, metric="l2")
        vectors = np.random.randn(500, DIM).astype(np.float32)
        idx.add_batch_np(vectors)
        idx.flush()

        path = os.path.join(tmp_path_str, "my_index.qvec")
        idx.save(path)

        loaded = quiver.HnswIndex.load(path)
        assert len(loaded) == 500
        r1 = idx.search(query=vectors[42].tolist(), k=5)
        r2 = loaded.search(query=vectors[42].tolist(), k=5)
        assert [h["id"] for h in r1] == [h["id"] for h in r2]

    def test_flat_index_standalone(self):
        idx = quiver.FlatIndex(dimensions=DIM, metric="cosine")
        vecs = _rand(100)
        for i, v in enumerate(vecs):
            idx.add(id=i, vector=v)

        results = idx.search(query=vecs[0], k=5)
        assert results[0]["id"] == 0

    def test_quantized_flat_index(self):
        idx = quiver.QuantizedFlatIndex(dimensions=DIM, metric="cosine")
        vecs = _rand(100)
        idx.add_batch([(i, vecs[i]) for i in range(100)])
        assert len(idx) == 100
        results = idx.search(query=vecs[0], k=5)
        assert len(results) == 5

    def test_fp16_flat_index(self):
        idx = quiver.Fp16FlatIndex(dimensions=DIM, metric="cosine")
        vecs = _rand(100)
        idx.add_batch([(i, vecs[i]) for i in range(100)])
        assert len(idx) == 100
        results = idx.search(query=vecs[0], k=5)
        assert len(results) == 5

    def test_binary_flat_index(self):
        idx = quiver.BinaryFlatIndex(dimensions=DIM, metric="l2")
        vecs = _rand(100)
        idx.add_batch([(i, vecs[i]) for i in range(100)])
        assert len(idx) == 100
        results = idx.search(query=vecs[0], k=5)
        assert len(results) == 5

    def test_ivf_index(self):
        idx = quiver.IvfIndex(
            dimensions=DIM, metric="l2",
            n_lists=8, nprobe=4, train_size=200,
        )
        vecs = _rand(300)
        idx.add_batch([(i, vecs[i]) for i in range(300)])
        idx.flush()
        assert len(idx) == 300
        results = idx.search(query=vecs[0], k=5)
        assert len(results) >= 1

    def test_ivf_pq_index(self):
        idx = quiver.IvfPqIndex(
            dimensions=DIM, metric="l2",
            n_lists=8, nprobe=4, train_size=200,
            pq_m=8, pq_k_sub=256,
        )
        vecs = _rand(300)
        idx.add_batch([(i, vecs[i]) for i in range(300)])
        idx.flush()
        assert len(idx) == 300
        results = idx.search(query=vecs[0], k=5)
        assert len(results) >= 1

    def test_mmap_flat_index(self, tmp_path_str):
        path = os.path.join(tmp_path_str, "mmap_index.qvec")
        idx = quiver.MmapFlatIndex(dimensions=DIM, metric="cosine", path=path)
        vecs = _rand(100)
        idx.add_batch([(i, vecs[i]) for i in range(100)])
        idx.flush()
        assert len(idx) == 100
        results = idx.search(query=vecs[0], k=5)
        assert len(results) == 5

    def test_index_delete(self):
        idx = quiver.FlatIndex(dimensions=DIM, metric="l2")
        vecs = _rand(10)
        for i, v in enumerate(vecs):
            idx.add(id=i, vector=v)
        assert len(idx) == 10

        assert idx.delete(5) is True
        assert idx.delete(999) is False
        assert len(idx) == 9

    def test_index_save_load_flat(self, tmp_path_str):
        idx = quiver.FlatIndex(dimensions=DIM, metric="l2")
        vecs = _rand(50)
        idx.add_batch([(i, vecs[i]) for i in range(50)])

        path = os.path.join(tmp_path_str, "flat.qvec")
        idx.save(path)
        loaded = quiver.FlatIndex.load(path)
        assert len(loaded) == 50


# ── 9. REST API Server ──────────────────────────────────────────────────────


class TestRESTServer:
    """README §9 — create_server, all endpoints."""

    @pytest.fixture
    def server_url(self, tmp_path_str):
        """Start a server on a random port and return its base URL."""
        from quiver_vector_db.server import create_server

        server = create_server(host="127.0.0.1", port=0, data_path=tmp_path_str)
        addr = server.server_address
        base = f"http://{addr[0]}:{addr[1]}"
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        yield base
        server.shutdown()

    def _request(self, url: str, method: str = "GET", data: dict | None = None) -> dict:
        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(url, data=body, method=method)
        if body:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())

    def test_healthz(self, server_url):
        result = self._request(f"{server_url}/healthz")
        assert result == {"status": "ok"}

    def test_create_list_delete_collection(self, server_url):
        # Create
        result = self._request(
            f"{server_url}/collections",
            method="POST",
            data={"name": "test_col", "dimensions": DIM, "metric": "cosine"},
        )
        assert result["created"] == "test_col"

        # List
        result = self._request(f"{server_url}/collections")
        assert "test_col" in result["collections"]

        # Delete
        req = urllib.request.Request(
            f"{server_url}/collections/test_col", method="DELETE"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
        assert result["deleted"] is True

    def test_upsert_search_delete_vector(self, server_url):
        # Create collection
        self._request(
            f"{server_url}/collections",
            method="POST",
            data={"name": "api_col", "dimensions": DIM, "metric": "l2"},
        )

        v1, v2 = _rand(2)

        # Upsert
        result = self._request(
            f"{server_url}/collections/api_col/upsert",
            method="POST",
            data={"id": 1, "vector": v1, "payload": {"key": "val"}},
        )
        assert result["upserted"] == 1

        self._request(
            f"{server_url}/collections/api_col/upsert",
            method="POST",
            data={"id": 2, "vector": v2},
        )

        # Count
        result = self._request(f"{server_url}/collections/api_col/count")
        assert result["count"] == 2

        # Search
        result = self._request(
            f"{server_url}/collections/api_col/search",
            method="POST",
            data={"query": v1, "k": 5},
        )
        assert len(result["results"]) >= 1
        assert result["results"][0]["id"] == 1

        # Delete vector
        result = self._request(
            f"{server_url}/collections/api_col/delete",
            method="POST",
            data={"id": 1},
        )
        assert result["deleted"] is True

        # Verify count
        result = self._request(f"{server_url}/collections/api_col/count")
        assert result["count"] == 1

    def test_batch_upsert_api(self, server_url):
        self._request(
            f"{server_url}/collections",
            method="POST",
            data={"name": "batch_col", "dimensions": DIM, "metric": "l2"},
        )
        vecs = _rand(3)
        result = self._request(
            f"{server_url}/collections/batch_col/upsert_batch",
            method="POST",
            data={
                "entries": [
                    {"id": 1, "vector": vecs[0], "payload": {"a": 1}},
                    {"id": 2, "vector": vecs[1]},
                    {"id": 3, "vector": vecs[2]},
                ]
            },
        )
        assert result["upserted"] == 3

    def test_snapshot_api(self, server_url):
        self._request(
            f"{server_url}/collections",
            method="POST",
            data={"name": "snap_col", "dimensions": DIM, "metric": "l2"},
        )
        v = _rand1()
        self._request(
            f"{server_url}/collections/snap_col/upsert",
            method="POST",
            data={"id": 1, "vector": v},
        )

        # Create snapshot
        result = self._request(
            f"{server_url}/collections/snap_col/snapshots",
            method="POST",
            data={"name": "s1"},
        )
        assert result["name"] == "s1"

        # List snapshots
        result = self._request(f"{server_url}/collections/snap_col/snapshots")
        assert any(s["name"] == "s1" for s in result["snapshots"])

        # Restore snapshot
        self._request(
            f"{server_url}/collections/snap_col/upsert",
            method="POST",
            data={"id": 2, "vector": _rand1()},
        )
        result = self._request(
            f"{server_url}/collections/snap_col/snapshots/restore",
            method="POST",
            data={"name": "s1"},
        )
        assert result["restored"] == "s1"

        result = self._request(f"{server_url}/collections/snap_col/count")
        assert result["count"] == 1

        # Delete snapshot
        req = urllib.request.Request(
            f"{server_url}/collections/snap_col/snapshots/s1", method="DELETE"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
        assert result["deleted"] is True


# ── 10. BM25 Standalone ─────────────────────────────────────────────────────


class TestBM25Standalone:
    """README §10 — BM25 tokenizer and sparse vector generator."""

    def test_index_and_query(self):
        bm25 = BM25(k1=1.5, b=0.75)

        bm25.index_document(0, "the quick brown fox jumps over the lazy dog")
        bm25.index_document(1, "machine learning with neural networks")
        bm25.index_document(2, "the fox and the hound")

        assert bm25.doc_count == 3
        assert bm25.vocab_size > 0
        assert bm25.avg_dl > 0

        sparse_query = bm25.encode_query("quick fox")
        assert isinstance(sparse_query, dict)
        assert len(sparse_query) > 0
        # Values should be positive IDF weights
        assert all(v > 0 for v in sparse_query.values())

    def test_save_and_load(self, tmp_path_str):
        bm25 = BM25(k1=1.5, b=0.75)
        bm25.index_document(0, "hello world")
        bm25.index_document(1, "goodbye world")

        path = os.path.join(tmp_path_str, "bm25_state.json")
        bm25.save(path)

        loaded = BM25.load(path)
        assert loaded.doc_count == 2
        assert loaded.vocab_size == bm25.vocab_size

        # Queries should produce identical results
        q1 = bm25.encode_query("hello")
        q2 = loaded.encode_query("hello")
        assert q1 == q2

    def test_remove_document(self):
        bm25 = BM25()
        bm25.index_document(0, "foo bar")
        bm25.index_document(1, "baz qux")
        assert bm25.doc_count == 2

        bm25.remove_document(0)
        assert bm25.doc_count == 1


# ── Index Types Reference ───────────────────────────────────────────────────


class TestAllIndexTypesViaClient:
    """Verify all 8 index types work through the Client API (README Index Types table)."""

    INDEX_TYPES = ["hnsw", "flat", "quantized_flat", "fp16_flat", "ivf", "ivf_pq", "mmap_flat", "binary_flat"]

    @pytest.mark.parametrize("index_type", INDEX_TYPES)
    def test_create_upsert_search(self, tmp_path_str, index_type):
        db = quiver.Client(path=tmp_path_str)
        col = db.create_collection(
            f"col_{index_type}",
            dimensions=DIM,
            metric="l2",
            index_type=index_type,
        )

        # Insert enough vectors for IVF training
        n = 300 if index_type.startswith("ivf") else 50
        vecs = _rand(n)
        entries = [(i, vecs[i]) for i in range(n)]
        col.upsert_batch(entries)

        results = col.search(query=vecs[0], k=5)
        assert len(results) >= 1
        assert results[0]["id"] == 0 or index_type in ("binary_flat", "ivf_pq")


# ── Distance Metrics ────────────────────────────────────────────────────────


class TestDistanceMetrics:
    """Verify all 3 distance metrics work (README Distance Metrics table)."""

    @pytest.mark.parametrize("metric", ["cosine", "l2", "dot_product"])
    def test_metric(self, tmp_path_str, metric):
        db = quiver.Client(path=tmp_path_str)
        col = db.create_collection(
            f"metric_{metric}", dimensions=DIM, metric=metric,
        )
        vecs = _rand(20)
        for i, v in enumerate(vecs):
            col.upsert(id=i, vector=v)

        results = col.search(query=vecs[0], k=5)
        assert len(results) == 5
        assert results[0]["id"] == 0

    @pytest.mark.parametrize("metric", ["cosine", "l2", "dot_product"])
    def test_metric_standalone(self, metric):
        idx = quiver.FlatIndex(dimensions=DIM, metric=metric)
        vecs = _rand(20)
        for i, v in enumerate(vecs):
            idx.add(id=i, vector=v)

        results = idx.search(query=vecs[0], k=5)
        assert len(results) == 5
        assert results[0]["id"] == 0


# ── Collection Delete ────────────────────────────────────────────────────────


class TestCollectionDelete:
    """Verify vector deletion via Collection."""

    def test_delete_vector(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        col = db.create_collection("cdel", dimensions=DIM, metric="l2")
        vecs = _rand(5)
        for i, v in enumerate(vecs):
            col.upsert(id=i, vector=v)
        assert col.count == 5

        result = col.delete(id=2)
        assert result is True
        assert col.count == 4

        result = col.delete(id=999)
        assert result is False


# ── Edge Cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Additional edge cases to ensure robustness."""

    def test_upsert_overwrite(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        col = db.create_collection("ow", dimensions=DIM, metric="l2")
        v1, v2 = _rand(2)

        col.upsert(id=1, vector=v1, payload={"v": 1})
        col.upsert(id=1, vector=v2, payload={"v": 2})
        assert col.count == 1

        hits = col.search(query=v2, k=1)
        assert hits[0]["id"] == 1
        assert hits[0]["payload"]["v"] == 2

    def test_search_empty_collection(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        col = db.create_collection("empty", dimensions=DIM, metric="l2")
        hits = col.search(query=_rand1(), k=5)
        assert hits == []

    def test_search_k_larger_than_count(self, tmp_path_str):
        db = quiver.Client(path=tmp_path_str)
        col = db.create_collection("small", dimensions=DIM, metric="l2")
        col.upsert(id=1, vector=_rand1())
        hits = col.search(query=_rand1(), k=100)
        assert len(hits) == 1
