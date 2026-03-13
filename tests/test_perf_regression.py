"""Performance regression tests for Quiver.

These tests enforce minimum performance thresholds to catch regressions.
Quiver metrics are asserted; competitor metrics are printed for comparison.

Covers ALL Quiver features:
- 7 index types: Flat, HNSW, Int8, FP16, IVF, IVF-PQ, Mmap
- WAL-backed Collection: upsert, search, hybrid search, filtered search
- TextCollection: semantic, keyword (BM25), hybrid search
- Save/load persistence

Compares with 3 competitors: hnswlib, faiss-cpu, chromadb

Run:
    pytest tests/test_perf_regression.py -v -s

Install competitors for full comparison:
    pip install hnswlib faiss-cpu chromadb numpy
"""

import hashlib
import os
import time
import statistics
import tempfile
import pytest
import numpy as np
import quiver_vector_db as quiver
from quiver_vector_db.text_collection import TextCollection
from quiver_vector_db.bm25 import BM25


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DIM = 128
N = 10_000
N_QUERIES = 500
K = 10
SEED = 42


# ---------------------------------------------------------------------------
# Minimum performance thresholds (intentionally conservative)
# ---------------------------------------------------------------------------

# Insert throughput (vec/s)
MIN_INSERT_THROUGHPUT = {
    "flat": 20_000,
    "hnsw": 5_000,
    "int8": 15_000,
    "fp16": 15_000,
    "ivf": 3_000,
    "ivf_pq": 3_000,
    "mmap": 10_000,
    "collection": 3_000,
}

# Search latency (avg ms per query)
MAX_SEARCH_LATENCY_MS = {
    "flat": 1.0,
    "hnsw": 0.2,
    "int8": 1.0,
    "fp16": 1.0,
    "ivf": 0.5,
    "ivf_pq": 0.5,
    "mmap": 2.0,
    "collection": 1.0,
    "collection_hybrid": 2.0,
    "collection_filtered": 2.0,
}

# Recall@K minimums
MIN_RECALL = {
    "flat": 1.0,
    "hnsw": 0.60,
    "int8": 0.95,
    "fp16": 0.98,
    "ivf": 0.35,
    "ivf_pq": 0.02,
    "mmap": 1.0,
}

# TextCollection / BM25
MAX_TEXT_SEARCH_LATENCY_MS = 5.0
MIN_BM25_INDEX_THROUGHPUT = 10_000  # docs/s

# Save/load
MAX_SAVE_TIME_S = 2.0
MAX_LOAD_TIME_S = 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gen_vectors_np(n, dim=DIM, seed=SEED):
    rng = np.random.RandomState(seed)
    return rng.randn(n, dim).astype(np.float32)


def ground_truth_ids(vectors_np, queries_np, k):
    """Brute-force exact L2 top-k via numpy."""
    results = []
    for q in queries_np:
        dists = np.sum((vectors_np - q) ** 2, axis=1)
        results.append(np.argsort(dists)[:k])
    return np.array(results)


def compute_recall(predicted_ids, true_ids):
    recalls = []
    for pred, true in zip(predicted_ids, true_ids):
        pred_set = set(int(x) for x in pred)
        true_set = set(int(x) for x in true)
        recalls.append(len(pred_set & true_set) / len(true_set))
    return statistics.mean(recalls)


def fmt_table(headers, rows):
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |")
    print(sep)


def try_import(module_name):
    try:
        return __import__(module_name)
    except ImportError:
        return None


class MockEmbedding:
    """Deterministic mock embedder for TextCollection tests."""
    def __init__(self, dimensions=DIM):
        self._dimensions = dimensions

    def __call__(self, texts):
        results = []
        for text in texts:
            h = int(hashlib.md5(text.encode()).hexdigest(), 16)
            vec = [(h >> (i * 8) & 0xFF) / 255.0 for i in range(self._dimensions)]
            results.append(vec)
        return results

    @property
    def dimensions(self):
        return self._dimensions


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vectors_np():
    return gen_vectors_np(N)


@pytest.fixture(scope="module")
def vectors_list(vectors_np):
    return vectors_np.tolist()


@pytest.fixture(scope="module")
def queries_np():
    return gen_vectors_np(N_QUERIES, seed=999)


@pytest.fixture(scope="module")
def queries_list(queries_np):
    return queries_np.tolist()


@pytest.fixture(scope="module")
def gt(vectors_np, queries_np):
    return ground_truth_ids(vectors_np, queries_np, K)


# ---------------------------------------------------------------------------
# Quiver index builders
# ---------------------------------------------------------------------------

def build_quiver_index(name, vectors_list, tmp_path=None):
    """Build and populate a Quiver index by name."""
    entries = [(i, v) for i, v in enumerate(vectors_list)]
    if name == "flat":
        idx = quiver.FlatIndex(dimensions=DIM, metric="l2")
        idx.add_batch(entries)
        return idx
    elif name == "hnsw":
        idx = quiver.HnswIndex(dimensions=DIM, metric="l2", ef_search=50)
        idx.add_batch(entries)
        idx.flush()
        return idx
    elif name == "int8":
        idx = quiver.QuantizedFlatIndex(dimensions=DIM, metric="l2")
        idx.add_batch(entries)
        return idx
    elif name == "fp16":
        idx = quiver.Fp16FlatIndex(dimensions=DIM, metric="l2")
        idx.add_batch(entries)
        return idx
    elif name == "ivf":
        idx = quiver.IvfIndex(dimensions=DIM, metric="l2",
                              n_lists=32, nprobe=8, train_size=N)
        idx.add_batch(entries)
        return idx
    elif name == "ivf_pq":
        idx = quiver.IvfPqIndex(dimensions=DIM, metric="l2",
                                n_lists=32, nprobe=8, train_size=N,
                                pq_m=8, pq_k_sub=16)
        idx.add_batch(entries)
        return idx
    elif name == "mmap":
        path = str(tmp_path / "mmap_bench.qvec") if tmp_path else "./mmap_bench.qvec"
        idx = quiver.MmapFlatIndex(dimensions=DIM, metric="l2", path=path)
        idx.add_batch(entries)
        idx.flush()
        return idx
    raise ValueError(f"Unknown index: {name}")


# ===================================================================
# 1. INSERT THROUGHPUT — All Quiver index types + competitors
# ===================================================================


class TestInsertThroughput:
    """Ensure insert throughput doesn't regress for ANY index type."""

    @pytest.mark.parametrize("index_name", [
        "flat", "hnsw", "int8", "fp16", "ivf", "ivf_pq",
    ])
    def test_index_insert_throughput(self, vectors_list, index_name):
        entries = [(i, v) for i, v in enumerate(vectors_list)]

        if index_name == "flat":
            idx = quiver.FlatIndex(dimensions=DIM, metric="l2")
        elif index_name == "hnsw":
            idx = quiver.HnswIndex(dimensions=DIM, metric="l2")
        elif index_name == "int8":
            idx = quiver.QuantizedFlatIndex(dimensions=DIM, metric="l2")
        elif index_name == "fp16":
            idx = quiver.Fp16FlatIndex(dimensions=DIM, metric="l2")
        elif index_name == "ivf":
            idx = quiver.IvfIndex(dimensions=DIM, metric="l2",
                                  n_lists=32, nprobe=8, train_size=N)
        elif index_name == "ivf_pq":
            idx = quiver.IvfPqIndex(dimensions=DIM, metric="l2",
                                    n_lists=32, nprobe=8, train_size=N,
                                    pq_m=8, pq_k_sub=16)

        t0 = time.perf_counter()
        idx.add_batch(entries)
        if hasattr(idx, "flush"):
            idx.flush()
        elapsed = time.perf_counter() - t0
        throughput = N / elapsed

        print(f"\n  Quiver {index_name}: {throughput:,.0f} vec/s ({elapsed:.3f}s)")

        min_tp = MIN_INSERT_THROUGHPUT[index_name]
        assert throughput >= min_tp, (
            f"{index_name} insert regression: {throughput:,.0f} < {min_tp:,} vec/s"
        )

    def test_mmap_insert_throughput(self, vectors_list, tmp_path):
        entries = [(i, v) for i, v in enumerate(vectors_list)]
        path = str(tmp_path / "mmap_insert_bench.qvec")
        idx = quiver.MmapFlatIndex(dimensions=DIM, metric="l2", path=path)

        t0 = time.perf_counter()
        idx.add_batch(entries)
        idx.flush()
        elapsed = time.perf_counter() - t0
        throughput = N / elapsed

        print(f"\n  Quiver mmap: {throughput:,.0f} vec/s ({elapsed:.3f}s)")

        assert throughput >= MIN_INSERT_THROUGHPUT["mmap"], (
            f"mmap insert regression: {throughput:,.0f} < {MIN_INSERT_THROUGHPUT['mmap']:,}"
        )

    def test_collection_insert_throughput(self, vectors_list, tmp_path):
        rows = []

        # Quiver Collection (asserted)
        db = quiver.Client(path=str(tmp_path / "quiver_insert"))
        col = db.create_collection("bench", dimensions=DIM, metric="l2")
        t0 = time.perf_counter()
        for i, v in enumerate(vectors_list):
            col.upsert(id=i, vector=v)
        elapsed = time.perf_counter() - t0
        quiver_tp = N / elapsed
        rows.append(("Quiver Collection", f"{elapsed:.3f}s", f"{quiver_tp:,.0f}"))

        # chromadb (comparison only)
        try:
            import chromadb
            client = chromadb.Client()
            chroma_col = client.create_collection("bench_insert")
            str_ids = [str(i) for i in range(N)]
            t0 = time.perf_counter()
            batch = 5000
            for start in range(0, N, batch):
                end = min(start + batch, N)
                chroma_col.add(ids=str_ids[start:end],
                               embeddings=vectors_list[start:end])
            elapsed = time.perf_counter() - t0
            rows.append(("ChromaDB", f"{elapsed:.3f}s", f"{N/elapsed:,.0f}"))
        except ImportError:
            rows.append(("ChromaDB", "SKIP", "not installed"))

        print(f"\n\n=== Collection Insert ({N:,} vectors, {DIM}d) ===")
        fmt_table(["Library", "Time", "Throughput (vec/s)"], rows)

        assert quiver_tp >= MIN_INSERT_THROUGHPUT["collection"], (
            f"Collection insert regression: {quiver_tp:,.0f} < {MIN_INSERT_THROUGHPUT['collection']:,}"
        )

    def test_insert_comparison_table(self, vectors_list):
        """Comparison table of insert throughput vs 3 competitors."""
        rows = []

        # Quiver HNSW
        idx = quiver.HnswIndex(dimensions=DIM, metric="l2")
        entries = [(i, v) for i, v in enumerate(vectors_list)]
        t0 = time.perf_counter()
        idx.add_batch(entries)
        idx.flush()
        elapsed = time.perf_counter() - t0
        rows.append(("Quiver HNSW", f"{elapsed:.3f}s", f"{N/elapsed:,.0f}"))

        # hnswlib
        hnswlib = try_import("hnswlib")
        if hnswlib:
            idx_h = hnswlib.Index(space="l2", dim=DIM)
            idx_h.init_index(max_elements=N, M=16, ef_construction=200)
            vectors_np = np.array(vectors_list, dtype=np.float32)
            t0 = time.perf_counter()
            idx_h.add_items(vectors_np, np.arange(N))
            elapsed = time.perf_counter() - t0
            rows.append(("hnswlib", f"{elapsed:.3f}s", f"{N/elapsed:,.0f}"))
        else:
            rows.append(("hnswlib", "SKIP", "not installed"))

        # faiss HNSW
        faiss = try_import("faiss")
        if faiss:
            idx_f = faiss.IndexHNSWFlat(DIM, 16)
            idx_f.hnsw.efConstruction = 200
            vectors_np = np.array(vectors_list, dtype=np.float32)
            t0 = time.perf_counter()
            idx_f.add(vectors_np)
            elapsed = time.perf_counter() - t0
            rows.append(("faiss HNSW", f"{elapsed:.3f}s", f"{N/elapsed:,.0f}"))
        else:
            rows.append(("faiss HNSW", "SKIP", "not installed"))

        # chromadb
        try:
            import chromadb
            client = chromadb.Client()
            chroma_col = client.create_collection("bench_insert_cmp")
            str_ids = [str(i) for i in range(N)]
            t0 = time.perf_counter()
            batch = 5000
            for start in range(0, N, batch):
                end = min(start + batch, N)
                chroma_col.add(ids=str_ids[start:end],
                               embeddings=vectors_list[start:end])
            elapsed = time.perf_counter() - t0
            rows.append(("ChromaDB", f"{elapsed:.3f}s", f"{N/elapsed:,.0f}"))
        except ImportError:
            rows.append(("ChromaDB", "SKIP", "not installed"))

        print(f"\n\n=== Insert Throughput vs Competitors ({N:,} vectors, {DIM}d) ===")
        fmt_table(["Library", "Time", "Throughput (vec/s)"], rows)


# ===================================================================
# 2. SEARCH LATENCY — All Quiver index types + competitors
# ===================================================================


class TestSearchLatency:
    """Ensure search latency doesn't regress for ANY index type."""

    @pytest.mark.parametrize("index_name", [
        "flat", "hnsw", "int8", "fp16", "ivf", "ivf_pq",
    ])
    def test_index_search_latency(self, vectors_list, queries_list, index_name):
        idx = build_quiver_index(index_name, vectors_list)

        lats = []
        for q in queries_list:
            t0 = time.perf_counter()
            idx.search(query=q, k=K)
            lats.append((time.perf_counter() - t0) * 1000)

        avg = statistics.mean(lats)
        p50 = statistics.median(lats)
        p99 = sorted(lats)[int(0.99 * len(lats))]

        print(f"\n  Quiver {index_name}: avg={avg:.3f}ms p50={p50:.3f}ms p99={p99:.3f}ms")

        max_lat = MAX_SEARCH_LATENCY_MS[index_name]
        assert avg <= max_lat, (
            f"{index_name} search latency regression: {avg:.3f}ms > {max_lat}ms"
        )

    def test_mmap_search_latency(self, vectors_list, queries_list, tmp_path):
        idx = build_quiver_index("mmap", vectors_list, tmp_path)

        lats = []
        for q in queries_list:
            t0 = time.perf_counter()
            idx.search(query=q, k=K)
            lats.append((time.perf_counter() - t0) * 1000)

        avg = statistics.mean(lats)
        print(f"\n  Quiver mmap: avg={avg:.3f}ms")

        assert avg <= MAX_SEARCH_LATENCY_MS["mmap"], (
            f"mmap search latency regression: {avg:.3f}ms > {MAX_SEARCH_LATENCY_MS['mmap']}ms"
        )

    def test_search_latency_comparison(self, vectors_list, queries_list):
        """Comparison table of search latency vs 3 competitors."""
        queries_np_arr = np.array(queries_list, dtype=np.float32)
        rows = []

        # Quiver HNSW
        idx = build_quiver_index("hnsw", vectors_list)
        lats = []
        for q in queries_list:
            t0 = time.perf_counter()
            idx.search(query=q, k=K)
            lats.append((time.perf_counter() - t0) * 1000)
        rows.append(("Quiver HNSW",
                      f"{statistics.mean(lats):.3f}",
                      f"{statistics.median(lats):.3f}",
                      f"{sorted(lats)[int(0.99*len(lats))]:.3f}"))

        # hnswlib
        hnswlib = try_import("hnswlib")
        if hnswlib:
            idx_h = hnswlib.Index(space="l2", dim=DIM)
            idx_h.init_index(max_elements=N, M=16, ef_construction=200)
            vectors_np = np.array(vectors_list, dtype=np.float32)
            idx_h.add_items(vectors_np, np.arange(N))
            idx_h.set_ef(50)
            lats = []
            for q in queries_np_arr:
                t0 = time.perf_counter()
                idx_h.knn_query(q, k=K)
                lats.append((time.perf_counter() - t0) * 1000)
            rows.append(("hnswlib",
                          f"{statistics.mean(lats):.3f}",
                          f"{statistics.median(lats):.3f}",
                          f"{sorted(lats)[int(0.99*len(lats))]:.3f}"))
        else:
            rows.append(("hnswlib", "SKIP", "", ""))

        # faiss HNSW
        faiss = try_import("faiss")
        if faiss:
            idx_f = faiss.IndexHNSWFlat(DIM, 16)
            idx_f.hnsw.efConstruction = 200
            idx_f.hnsw.efSearch = 50
            vectors_np = np.array(vectors_list, dtype=np.float32)
            idx_f.add(vectors_np)
            lats = []
            for q in queries_np_arr:
                t0 = time.perf_counter()
                idx_f.search(q.reshape(1, -1), K)
                lats.append((time.perf_counter() - t0) * 1000)
            rows.append(("faiss HNSW",
                          f"{statistics.mean(lats):.3f}",
                          f"{statistics.median(lats):.3f}",
                          f"{sorted(lats)[int(0.99*len(lats))]:.3f}"))
        else:
            rows.append(("faiss HNSW", "SKIP", "", ""))

        # chromadb
        try:
            import chromadb
            client = chromadb.Client()
            chroma_col = client.create_collection("bench_search_lat")
            str_ids = [str(i) for i in range(N)]
            batch = 5000
            for start in range(0, N, batch):
                end = min(start + batch, N)
                chroma_col.add(ids=str_ids[start:end],
                               embeddings=vectors_list[start:end])
            lats = []
            for q in queries_list[:100]:  # chromadb is slow, use fewer queries
                t0 = time.perf_counter()
                chroma_col.query(query_embeddings=[q], n_results=K)
                lats.append((time.perf_counter() - t0) * 1000)
            rows.append(("ChromaDB",
                          f"{statistics.mean(lats):.3f}",
                          f"{statistics.median(lats):.3f}",
                          f"{sorted(lats)[int(0.99*len(lats))]:.3f}"))
        except ImportError:
            rows.append(("ChromaDB", "SKIP", "", ""))

        print(f"\n\n=== Search Latency vs Competitors ({N:,} vectors, k={K}) ===")
        fmt_table(["Library", "Avg (ms)", "p50 (ms)", "p99 (ms)"], rows)


# ===================================================================
# 3. RECALL — All Quiver index types + competitors
# ===================================================================


class TestRecall:
    """Ensure recall doesn't regress for ANY index type."""

    @pytest.mark.parametrize("index_name", [
        "flat", "hnsw", "int8", "fp16", "ivf", "ivf_pq",
    ])
    def test_index_recall(self, vectors_list, queries_list, gt, index_name):
        idx = build_quiver_index(index_name, vectors_list)

        pred = []
        for q in queries_list:
            r = idx.search(query=q, k=K)
            pred.append([h["id"] for h in r])
        recall = compute_recall(pred, gt)

        print(f"\n  Quiver {index_name} recall@{K}: {recall:.4f}")

        min_rec = MIN_RECALL[index_name]
        assert recall >= min_rec, (
            f"{index_name} recall regression: {recall:.4f} < {min_rec}"
        )

    def test_mmap_recall(self, vectors_list, queries_list, gt, tmp_path):
        idx = build_quiver_index("mmap", vectors_list, tmp_path)

        pred = []
        for q in queries_list:
            r = idx.search(query=q, k=K)
            pred.append([h["id"] for h in r])
        recall = compute_recall(pred, gt)

        print(f"\n  Quiver mmap recall@{K}: {recall:.4f}")

        assert recall >= MIN_RECALL["mmap"], (
            f"mmap recall regression: {recall:.4f} < {MIN_RECALL['mmap']}"
        )

    def test_recall_comparison(self, vectors_list, queries_list, gt):
        """Comparison table of recall vs 3 competitors."""
        rows = []

        # All Quiver index types
        for name in ["flat", "hnsw", "int8", "fp16", "ivf", "ivf_pq"]:
            idx = build_quiver_index(name, vectors_list)
            pred = []
            for q in queries_list:
                r = idx.search(query=q, k=K)
                pred.append([h["id"] for h in r])
            recall = compute_recall(pred, gt)
            rows.append((f"Quiver {name}", f"{recall:.4f}"))

        # hnswlib
        hnswlib = try_import("hnswlib")
        if hnswlib:
            idx_h = hnswlib.Index(space="l2", dim=DIM)
            idx_h.init_index(max_elements=N, M=16, ef_construction=200)
            vectors_np = np.array(vectors_list, dtype=np.float32)
            idx_h.add_items(vectors_np, np.arange(N))
            idx_h.set_ef(50)
            pred = []
            for q in np.array(queries_list, dtype=np.float32):
                labels, _ = idx_h.knn_query(q, k=K)
                pred.append(labels[0].tolist())
            rows.append(("hnswlib", f"{compute_recall(pred, gt):.4f}"))
        else:
            rows.append(("hnswlib", "SKIP"))

        # faiss (Flat + HNSW + SQ8)
        faiss = try_import("faiss")
        if faiss:
            vectors_np = np.array(vectors_list, dtype=np.float32)
            queries_np_arr = np.array(queries_list, dtype=np.float32)

            # faiss Flat (exact)
            idx_f = faiss.IndexFlatL2(DIM)
            idx_f.add(vectors_np)
            pred = []
            for q in queries_np_arr:
                _, l = idx_f.search(q.reshape(1, -1), K)
                pred.append(l[0].tolist())
            rows.append(("faiss Flat", f"{compute_recall(pred, gt):.4f}"))

            # faiss HNSW
            idx_f = faiss.IndexHNSWFlat(DIM, 16)
            idx_f.hnsw.efConstruction = 200
            idx_f.hnsw.efSearch = 50
            idx_f.add(vectors_np)
            pred = []
            for q in queries_np_arr:
                _, l = idx_f.search(q.reshape(1, -1), K)
                pred.append(l[0].tolist())
            rows.append(("faiss HNSW", f"{compute_recall(pred, gt):.4f}"))

            # faiss SQ8
            idx_f = faiss.IndexScalarQuantizer(DIM, faiss.ScalarQuantizer.QT_8bit)
            idx_f.train(vectors_np)
            idx_f.add(vectors_np)
            pred = []
            for q in queries_np_arr:
                _, l = idx_f.search(q.reshape(1, -1), K)
                pred.append(l[0].tolist())
            rows.append(("faiss SQ8", f"{compute_recall(pred, gt):.4f}"))
        else:
            rows.append(("faiss Flat", "SKIP"))
            rows.append(("faiss HNSW", "SKIP"))
            rows.append(("faiss SQ8", "SKIP"))

        # chromadb
        try:
            import chromadb
            client = chromadb.Client()
            chroma_col = client.create_collection("bench_recall")
            str_ids = [str(i) for i in range(N)]
            batch = 5000
            for start in range(0, N, batch):
                end = min(start + batch, N)
                chroma_col.add(ids=str_ids[start:end],
                               embeddings=vectors_list[start:end])
            pred = []
            for q in queries_list[:100]:  # chromadb is slower
                r = chroma_col.query(query_embeddings=[q], n_results=K)
                pred.append([int(x) for x in r["ids"][0]])
            rows.append(("ChromaDB", f"{compute_recall(pred, gt[:100]):.4f}"))
        except ImportError:
            rows.append(("ChromaDB", "SKIP"))

        print(f"\n\n=== Recall@{K} vs Competitors ({N:,} vectors) ===")
        fmt_table(["Library", f"Recall@{K}"], rows)


# ===================================================================
# 4. COLLECTION FEATURES — Hybrid search, filtered search
# ===================================================================


class TestCollectionSearch:
    """Test WAL-backed Collection search features: dense, hybrid, filtered."""

    @pytest.fixture()
    def populated_collection(self, vectors_list, tmp_path):
        db = quiver.Client(path=str(tmp_path / "col_search"))
        col = db.create_collection("bench", dimensions=DIM, metric="l2")
        for i, v in enumerate(vectors_list):
            payload = {"category": "even" if i % 2 == 0 else "odd", "value": i}
            sparse = {i % 50: 1.0, (i * 7) % 100: 0.5}
            col.upsert_hybrid(id=i, vector=v, sparse_vector=sparse, payload=payload)
        return col

    def test_collection_dense_search(self, populated_collection, queries_list):
        col = populated_collection
        lats = []
        for q in queries_list:
            t0 = time.perf_counter()
            col.search(query=q, k=K)
            lats.append((time.perf_counter() - t0) * 1000)
        avg = statistics.mean(lats)

        print(f"\n  Collection dense search: avg={avg:.3f}ms")

        assert avg <= MAX_SEARCH_LATENCY_MS["collection"], (
            f"Collection search latency regression: {avg:.3f}ms > "
            f"{MAX_SEARCH_LATENCY_MS['collection']}ms"
        )

    def test_collection_hybrid_search(self, populated_collection, queries_list):
        col = populated_collection
        sparse_query = {0: 1.0, 5: 0.8, 10: 0.6}
        lats = []
        for q in queries_list:
            t0 = time.perf_counter()
            col.search_hybrid(dense_query=q, sparse_query=sparse_query, k=K)
            lats.append((time.perf_counter() - t0) * 1000)
        avg = statistics.mean(lats)

        print(f"\n  Collection hybrid search: avg={avg:.3f}ms")

        assert avg <= MAX_SEARCH_LATENCY_MS["collection_hybrid"], (
            f"Hybrid search latency regression: {avg:.3f}ms > "
            f"{MAX_SEARCH_LATENCY_MS['collection_hybrid']}ms"
        )

    def test_collection_filtered_search(self, populated_collection, queries_list):
        col = populated_collection
        filt = {"category": {"$eq": "even"}}
        lats = []
        for q in queries_list:
            t0 = time.perf_counter()
            results = col.search(query=q, k=K, filter=filt)
            lats.append((time.perf_counter() - t0) * 1000)
        avg = statistics.mean(lats)

        print(f"\n  Collection filtered search: avg={avg:.3f}ms")

        assert avg <= MAX_SEARCH_LATENCY_MS["collection_filtered"], (
            f"Filtered search latency regression: {avg:.3f}ms > "
            f"{MAX_SEARCH_LATENCY_MS['collection_filtered']}ms"
        )

    def test_collection_delete_throughput(self, vectors_list, tmp_path):
        db = quiver.Client(path=str(tmp_path / "col_del"))
        col = db.create_collection("bench_del", dimensions=DIM, metric="l2")
        n_del = 1000
        for i in range(n_del):
            col.upsert(id=i, vector=vectors_list[i])

        t0 = time.perf_counter()
        for i in range(n_del):
            col.delete(id=i)
        elapsed = time.perf_counter() - t0
        throughput = n_del / elapsed

        print(f"\n  Collection delete: {throughput:,.0f} ops/s ({elapsed:.3f}s)")

        assert throughput >= 1_000, (
            f"Collection delete throughput: {throughput:,.0f} < 1,000 ops/s"
        )


# ===================================================================
# 5. TEXT COLLECTION — Semantic, keyword, hybrid search
# ===================================================================


class TestTextCollectionPerf:
    """Test TextCollection (embedding + BM25) performance."""

    @pytest.fixture()
    def text_col(self, tmp_path):
        db = quiver.Client(path=str(tmp_path / "text_perf"))
        raw_col = db.create_collection("docs", dimensions=DIM, metric="cosine")
        return TextCollection(
            collection=raw_col,
            embedding_function=MockEmbedding(dimensions=DIM),
        )

    def test_text_add_throughput(self, text_col):
        n_docs = 1000
        docs = [f"document number {i} about topic {i % 10} with content" for i in range(n_docs)]
        ids = list(range(n_docs))

        t0 = time.perf_counter()
        text_col.add(ids=ids, documents=docs)
        elapsed = time.perf_counter() - t0
        throughput = n_docs / elapsed

        print(f"\n  TextCollection add: {throughput:,.0f} docs/s ({elapsed:.3f}s)")

        assert throughput >= 500, (
            f"TextCollection add throughput: {throughput:,.0f} < 500 docs/s"
        )

    def test_text_semantic_search(self, text_col):
        n_docs = 500
        docs = [f"document {i} about {'science' if i % 3 == 0 else 'cooking'}" for i in range(n_docs)]
        text_col.add(ids=list(range(n_docs)), documents=docs)

        lats = []
        for _ in range(100):
            t0 = time.perf_counter()
            text_col.query("science research", k=10, mode="semantic")
            lats.append((time.perf_counter() - t0) * 1000)
        avg = statistics.mean(lats)

        print(f"\n  TextCollection semantic search: avg={avg:.3f}ms")

        assert avg <= MAX_TEXT_SEARCH_LATENCY_MS, (
            f"Semantic search latency: {avg:.3f}ms > {MAX_TEXT_SEARCH_LATENCY_MS}ms"
        )

    def test_text_keyword_search(self, text_col):
        n_docs = 500
        docs = [f"document {i} about {'science' if i % 3 == 0 else 'cooking'}" for i in range(n_docs)]
        text_col.add(ids=list(range(n_docs)), documents=docs)

        lats = []
        for _ in range(100):
            t0 = time.perf_counter()
            text_col.query("science document", k=10, mode="keyword")
            lats.append((time.perf_counter() - t0) * 1000)
        avg = statistics.mean(lats)

        print(f"\n  TextCollection keyword search: avg={avg:.3f}ms")

        assert avg <= MAX_TEXT_SEARCH_LATENCY_MS, (
            f"Keyword search latency: {avg:.3f}ms > {MAX_TEXT_SEARCH_LATENCY_MS}ms"
        )

    def test_text_hybrid_search(self, text_col):
        n_docs = 500
        docs = [f"document {i} about {'science' if i % 3 == 0 else 'cooking'}" for i in range(n_docs)]
        text_col.add(ids=list(range(n_docs)), documents=docs)

        lats = []
        for _ in range(100):
            t0 = time.perf_counter()
            text_col.query("science document", k=10, mode="hybrid")
            lats.append((time.perf_counter() - t0) * 1000)
        avg = statistics.mean(lats)

        print(f"\n  TextCollection hybrid search: avg={avg:.3f}ms")

        assert avg <= MAX_TEXT_SEARCH_LATENCY_MS, (
            f"Hybrid search latency: {avg:.3f}ms > {MAX_TEXT_SEARCH_LATENCY_MS}ms"
        )


# ===================================================================
# 6. BM25 INDEXING THROUGHPUT
# ===================================================================


class TestBM25Perf:
    """Test BM25 tokenizer and indexing throughput."""

    def test_bm25_index_throughput(self):
        bm25 = BM25()
        n_docs = 10_000
        docs = [f"the quick brown fox jumps over the lazy dog document {i}" for i in range(n_docs)]

        t0 = time.perf_counter()
        for i, doc in enumerate(docs):
            bm25.index_document(i, doc)
        elapsed = time.perf_counter() - t0
        throughput = n_docs / elapsed

        print(f"\n  BM25 index: {throughput:,.0f} docs/s ({elapsed:.3f}s)")

        assert throughput >= MIN_BM25_INDEX_THROUGHPUT, (
            f"BM25 index throughput: {throughput:,.0f} < {MIN_BM25_INDEX_THROUGHPUT:,} docs/s"
        )

    def test_bm25_query_throughput(self):
        bm25 = BM25()
        n_docs = 5_000
        for i in range(n_docs):
            bm25.index_document(i, f"document {i} about topic {i % 50}")

        n_queries = 1000
        t0 = time.perf_counter()
        for i in range(n_queries):
            bm25.encode_query(f"topic {i % 50} document")
        elapsed = time.perf_counter() - t0
        throughput = n_queries / elapsed

        print(f"\n  BM25 query encode: {throughput:,.0f} queries/s ({elapsed:.3f}s)")

        assert throughput >= 10_000, (
            f"BM25 query throughput: {throughput:,.0f} < 10,000 queries/s"
        )


# ===================================================================
# 7. SAVE / LOAD PERSISTENCE
# ===================================================================


class TestSaveLoadPerf:
    """Test save/load performance for index types that support it."""

    @pytest.mark.parametrize("index_name", ["flat", "hnsw", "int8", "fp16"])
    def test_save_load(self, vectors_list, queries_list, gt, index_name, tmp_path):
        idx = build_quiver_index(index_name, vectors_list)
        save_path = str(tmp_path / f"{index_name}.qvec")

        # Save
        t0 = time.perf_counter()
        idx.save(save_path)
        save_time = time.perf_counter() - t0

        # Load
        loader = {
            "flat": quiver.FlatIndex.load,
            "hnsw": quiver.HnswIndex.load,
            "int8": quiver.QuantizedFlatIndex.load,
            "fp16": quiver.Fp16FlatIndex.load,
        }[index_name]
        t0 = time.perf_counter()
        idx2 = loader(save_path)
        load_time = time.perf_counter() - t0

        # Verify loaded index returns same results
        r1 = idx.search(query=queries_list[0], k=K)
        r2 = idx2.search(query=queries_list[0], k=K)
        ids1 = {h["id"] for h in r1}
        ids2 = {h["id"] for h in r2}

        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)

        print(f"\n  {index_name}: save={save_time:.3f}s load={load_time:.3f}s "
              f"size={file_size_mb:.1f}MB")

        assert save_time <= MAX_SAVE_TIME_S, (
            f"{index_name} save too slow: {save_time:.3f}s > {MAX_SAVE_TIME_S}s"
        )
        assert load_time <= MAX_LOAD_TIME_S, (
            f"{index_name} load too slow: {load_time:.3f}s > {MAX_LOAD_TIME_S}s"
        )
        assert ids1 == ids2, f"{index_name} save/load corrupted results"


# ===================================================================
# 8. FULL SUMMARY TABLE
# ===================================================================


class TestFullSummary:
    """Print comprehensive comparison tables across all Quiver features."""

    def test_full_comparison(self, vectors_list, queries_list, gt, tmp_path):
        rows = []

        def bench(name, idx, queries):
            lats = []
            pred = []
            for q in queries:
                t0 = time.perf_counter()
                r = idx.search(query=q, k=K)
                lats.append((time.perf_counter() - t0) * 1000)
                pred.append([h["id"] for h in r])
            recall = compute_recall(pred, gt)
            return statistics.mean(lats), recall

        # All Quiver index types
        for name in ["flat", "hnsw", "int8", "fp16", "ivf", "ivf_pq"]:
            idx = build_quiver_index(name, vectors_list)
            lat, rec = bench(name, idx, queries_list)
            rows.append((f"Quiver {name}", f"{lat:.3f}", f"{rec:.4f}"))

        # Mmap
        idx = build_quiver_index("mmap", vectors_list, tmp_path)
        lat, rec = bench("mmap", idx, queries_list)
        rows.append(("Quiver mmap", f"{lat:.3f}", f"{rec:.4f}"))

        # hnswlib
        hnswlib = try_import("hnswlib")
        if hnswlib:
            idx_h = hnswlib.Index(space="l2", dim=DIM)
            idx_h.init_index(max_elements=N, M=16, ef_construction=200)
            vectors_np = np.array(vectors_list, dtype=np.float32)
            idx_h.add_items(vectors_np, np.arange(N))
            idx_h.set_ef(50)
            lats, pred = [], []
            for q in np.array(queries_list, dtype=np.float32):
                t0 = time.perf_counter()
                labels, _ = idx_h.knn_query(q, k=K)
                lats.append((time.perf_counter() - t0) * 1000)
                pred.append(labels[0].tolist())
            rows.append(("hnswlib", f"{statistics.mean(lats):.3f}",
                          f"{compute_recall(pred, gt):.4f}"))
        else:
            rows.append(("hnswlib", "SKIP", "SKIP"))

        # faiss
        faiss = try_import("faiss")
        if faiss:
            vectors_np = np.array(vectors_list, dtype=np.float32)
            queries_np_arr = np.array(queries_list, dtype=np.float32)
            for fname, build_fn in [
                ("faiss Flat", lambda: self._faiss_flat(faiss, vectors_np)),
                ("faiss HNSW", lambda: self._faiss_hnsw(faiss, vectors_np)),
                ("faiss SQ8", lambda: self._faiss_sq8(faiss, vectors_np)),
            ]:
                idx_f = build_fn()
                lats, pred = [], []
                for q in queries_np_arr:
                    t0 = time.perf_counter()
                    _, l = idx_f.search(q.reshape(1, -1), K)
                    lats.append((time.perf_counter() - t0) * 1000)
                    pred.append(l[0].tolist())
                rows.append((fname, f"{statistics.mean(lats):.3f}",
                              f"{compute_recall(pred, gt):.4f}"))
        else:
            rows.append(("faiss", "SKIP", "SKIP"))

        # chromadb
        try:
            import chromadb
            client = chromadb.Client()
            chroma_col = client.create_collection("bench_summary")
            str_ids = [str(i) for i in range(N)]
            batch = 5000
            for start in range(0, N, batch):
                end = min(start + batch, N)
                chroma_col.add(ids=str_ids[start:end],
                               embeddings=vectors_list[start:end])
            lats, pred = [], []
            for q in queries_list[:100]:
                t0 = time.perf_counter()
                r = chroma_col.query(query_embeddings=[q], n_results=K)
                lats.append((time.perf_counter() - t0) * 1000)
                pred.append([int(x) for x in r["ids"][0]])
            rows.append(("ChromaDB", f"{statistics.mean(lats):.3f}",
                          f"{compute_recall(pred, gt[:100]):.4f}"))
        except ImportError:
            rows.append(("ChromaDB", "SKIP", "SKIP"))

        print(f"\n\n{'='*70}")
        print(f"  FULL COMPARISON SUMMARY")
        print(f"  {N:,} vectors, {DIM}d, {N_QUERIES} queries, k={K}")
        print(f"{'='*70}")
        fmt_table(["Library", "Avg Latency (ms)", f"Recall@{K}"], rows)

    @staticmethod
    def _faiss_flat(faiss, vectors_np):
        idx = faiss.IndexFlatL2(DIM)
        idx.add(vectors_np)
        return idx

    @staticmethod
    def _faiss_hnsw(faiss, vectors_np):
        idx = faiss.IndexHNSWFlat(DIM, 16)
        idx.hnsw.efConstruction = 200
        idx.hnsw.efSearch = 50
        idx.add(vectors_np)
        return idx

    @staticmethod
    def _faiss_sq8(faiss, vectors_np):
        idx = faiss.IndexScalarQuantizer(DIM, faiss.ScalarQuantizer.QT_8bit)
        idx.train(vectors_np)
        idx.add(vectors_np)
        return idx
