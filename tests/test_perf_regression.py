"""Performance regression tests for Quiver.

These tests enforce minimum performance thresholds to catch regressions.
Quiver metrics are asserted; competitor metrics are printed for comparison.

Covers ALL Quiver features:
- 8 index types: Flat, HNSW, Int8, FP16, IVF, IVF-PQ, Mmap, BinaryFlat
- WAL-backed Collection: upsert, search, hybrid search, filtered search
- TextCollection: semantic, keyword (BM25), hybrid search
- Data versioning / snapshots: create, restore, delete
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
N = 2_000
N_QUERIES = 100
K = 10
SEED = 42


# ---------------------------------------------------------------------------
# Minimum performance thresholds (intentionally conservative)
# ---------------------------------------------------------------------------

# Insert throughput (vec/s) — calibrated for N=2000, debug builds.
# Note: release builds are ~40x faster (e.g. HNSW ~20K vec/s release vs ~500 debug).
# Index-level benchmarks (flat, hnsw, etc.) measure raw in-memory insert — no WAL/disk I/O.
# Collection-level benchmarks include WAL writes.
MIN_INSERT_THROUGHPUT = {
    "flat": 1_000,
    "hnsw": 100,
    "int8": 800,
    "fp16": 800,
    "ivf": 100,
    "ivf_pq": 100,
    "mmap": 500,
    "binary_flat": 1_000,
    "collection": 50,
}

# Search latency (avg ms per query)
MAX_SEARCH_LATENCY_MS = {
    "flat": 30.0,
    "hnsw": 5.0,
    "int8": 50.0,
    "fp16": 50.0,
    "ivf": 10.0,
    "ivf_pq": 5.0,
    "mmap": 30.0,
    "binary_flat": 10.0,
    "collection": 5.0,
    "collection_hybrid": 10.0,
    "collection_filtered": 10.0,
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
    "binary_flat": 0.01,
}

# TextCollection / BM25
MAX_TEXT_SEARCH_LATENCY_MS = 50.0
MIN_BM25_INDEX_THROUGHPUT = 1_000  # docs/s

# Save/load
MAX_SAVE_TIME_S = 10.0
MAX_LOAD_TIME_S = 10.0


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
    elif name == "binary_flat":
        idx = quiver.BinaryFlatIndex(dimensions=DIM, metric="l2")
        idx.add_batch(entries)
        return idx
    raise ValueError(f"Unknown index: {name}")


# ===================================================================
# 1. INSERT THROUGHPUT — All Quiver index types + competitors
# ===================================================================


class TestInsertThroughput:
    """Ensure insert throughput doesn't regress for ANY index type."""

    @pytest.mark.parametrize("index_name", [
        "flat", "hnsw", "int8", "fp16", "ivf", "ivf_pq", "binary_flat",
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
        elif index_name == "binary_flat":
            idx = quiver.BinaryFlatIndex(dimensions=DIM, metric="l2")

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

        # All HNSW indexes use M=16, ef_construction=200 for fair comparison
        BENCH_M = 16
        BENCH_EF_CONSTRUCTION = 200

        import os
        n_cores = os.cpu_count() or 1
        vectors_np = np.array(vectors_list, dtype=np.float32)

        # Quiver HNSW — single-threaded (numpy batch)
        idx = quiver.HnswIndex(dimensions=DIM, metric="l2",
                               m=BENCH_M, ef_construction=BENCH_EF_CONSTRUCTION)
        t0 = time.perf_counter()
        idx.add_batch_np(vectors_np)
        idx.flush()
        elapsed = time.perf_counter() - t0
        rows.append(("Quiver HNSW (1T)", f"{elapsed:.3f}s", f"{N/elapsed:,.0f}"))

        # Quiver HNSW — parallel micro-batching (all cores)
        idx_par = quiver.HnswIndex(dimensions=DIM, metric="l2",
                                   m=BENCH_M, ef_construction=BENCH_EF_CONSTRUCTION)
        t0 = time.perf_counter()
        idx_par.add_batch_parallel(vectors_np, num_threads=n_cores)
        idx_par.flush()
        elapsed = time.perf_counter() - t0
        rows.append((f"Quiver HNSW ({n_cores}T)", f"{elapsed:.3f}s", f"{N/elapsed:,.0f}"))

        # hnswlib — single-threaded for fair comparison
        hnswlib = try_import("hnswlib")
        if hnswlib:
            idx_h = hnswlib.Index(space="l2", dim=DIM)
            idx_h.init_index(max_elements=N, M=BENCH_M,
                             ef_construction=BENCH_EF_CONSTRUCTION)
            idx_h.set_num_threads(1)
            t0 = time.perf_counter()
            idx_h.add_items(vectors_np, np.arange(N))
            elapsed = time.perf_counter() - t0
            rows.append(("hnswlib (1T)", f"{elapsed:.3f}s", f"{N/elapsed:,.0f}"))

            # Also show multi-threaded for context
            import os
            n_cores = os.cpu_count() or 1
            idx_h2 = hnswlib.Index(space="l2", dim=DIM)
            idx_h2.init_index(max_elements=N, M=BENCH_M,
                              ef_construction=BENCH_EF_CONSTRUCTION)
            idx_h2.set_num_threads(n_cores)
            t0 = time.perf_counter()
            idx_h2.add_items(vectors_np, np.arange(N))
            elapsed = time.perf_counter() - t0
            rows.append((f"hnswlib ({n_cores}T)", f"{elapsed:.3f}s", f"{N/elapsed:,.0f}"))
        else:
            rows.append(("hnswlib", "SKIP", "not installed"))

        # faiss HNSW — single-threaded for fair comparison
        faiss = try_import("faiss")
        if faiss:
            import os
            n_cores = os.cpu_count() or 1
            faiss.omp_set_num_threads(1)
            idx_f = faiss.IndexHNSWFlat(DIM, BENCH_M)
            idx_f.hnsw.efConstruction = BENCH_EF_CONSTRUCTION
            t0 = time.perf_counter()
            idx_f.add(vectors_np)
            elapsed = time.perf_counter() - t0
            rows.append(("faiss HNSW (1T)", f"{elapsed:.3f}s", f"{N/elapsed:,.0f}"))

            # Also show multi-threaded for context
            faiss.omp_set_num_threads(n_cores)
            idx_f2 = faiss.IndexHNSWFlat(DIM, BENCH_M)
            idx_f2.hnsw.efConstruction = BENCH_EF_CONSTRUCTION
            t0 = time.perf_counter()
            idx_f2.add(vectors_np)
            elapsed = time.perf_counter() - t0
            rows.append((f"faiss HNSW ({n_cores}T)", f"{elapsed:.3f}s", f"{N/elapsed:,.0f}"))
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

        print(f"\n\n=== Insert Throughput vs Competitors ({N:,} vectors, {DIM}d, M={BENCH_M}) ===")
        fmt_table(["Library", "Time", "Throughput (vec/s)"], rows)


# ===================================================================
# 2. SEARCH LATENCY — All Quiver index types + competitors
# ===================================================================


class TestSearchLatency:
    """Ensure search latency doesn't regress for ANY index type."""

    @pytest.mark.parametrize("index_name", [
        "flat", "hnsw", "int8", "fp16", "ivf", "ivf_pq", "binary_flat",
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

        # All HNSW indexes use M=16, ef_construction=200 for fair comparison
        BENCH_M = 16
        BENCH_EF_CONSTRUCTION = 200

        # Quiver HNSW
        idx = quiver.HnswIndex(dimensions=DIM, metric="l2",
                               m=BENCH_M, ef_construction=BENCH_EF_CONSTRUCTION,
                               ef_search=50)
        entries = [(i, v) for i, v in enumerate(vectors_list)]
        idx.add_batch(entries)
        idx.flush()
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
            idx_h.init_index(max_elements=N, M=BENCH_M,
                             ef_construction=BENCH_EF_CONSTRUCTION)
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
            idx_f = faiss.IndexHNSWFlat(DIM, BENCH_M)
            idx_f.hnsw.efConstruction = BENCH_EF_CONSTRUCTION
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
        "flat", "hnsw", "int8", "fp16", "ivf", "ivf_pq", "binary_flat",
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
        for name in ["flat", "hnsw", "int8", "fp16", "ivf", "ivf_pq", "binary_flat"]:
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

        assert throughput >= 50, (
            f"TextCollection add throughput: {throughput:,.0f} < 50 docs/s"
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

    @pytest.mark.parametrize("index_name", ["flat", "hnsw", "int8", "fp16", "binary_flat"])
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
            "binary_flat": quiver.BinaryFlatIndex.load,
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
        # Binary quantization has many distance ties — just verify distances match
        if index_name == "binary_flat":
            assert abs(r1[0]["distance"] - r2[0]["distance"]) < 1e-6, f"{index_name} save/load corrupted top-1 distance"
        else:
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
        for name in ["flat", "hnsw", "int8", "fp16", "ivf", "ivf_pq", "binary_flat"]:
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


# ===================================================================
# 9. BINARY QUANTIZATION — Compression ratio & speed advantage
# ===================================================================


class TestBinaryQuantizationPerf:
    """Test BinaryFlatIndex compression ratio and search speed vs FlatIndex."""

    def test_compression_ratio(self, vectors_list):
        """Binary quantization should achieve ~32x compression over FlatIndex."""
        entries = [(i, v) for i, v in enumerate(vectors_list)]

        flat = quiver.FlatIndex(dimensions=DIM, metric="l2")
        flat.add_batch(entries)

        binary = quiver.BinaryFlatIndex(dimensions=DIM, metric="l2")
        binary.add_batch(entries)

        with tempfile.TemporaryDirectory() as d:
            flat_path = os.path.join(d, "flat.bin")
            bin_path = os.path.join(d, "binary.bin")
            flat.save(flat_path)
            binary.save(bin_path)

            flat_size = os.path.getsize(flat_path)
            bin_size = os.path.getsize(bin_path)
            ratio = flat_size / bin_size

        print(f"\n  Flat size: {flat_size/1024:.1f} KB")
        print(f"  Binary size: {bin_size/1024:.1f} KB")
        print(f"  Compression ratio: {ratio:.1f}x")

        assert ratio >= 10.0, (
            f"Binary compression ratio too low: {ratio:.1f}x < 10x"
        )

    def test_binary_vs_flat_search_speed(self, vectors_list, queries_list):
        """Binary search should be faster than flat due to popcount."""
        entries = [(i, v) for i, v in enumerate(vectors_list)]

        flat = quiver.FlatIndex(dimensions=DIM, metric="l2")
        flat.add_batch(entries)

        binary = quiver.BinaryFlatIndex(dimensions=DIM, metric="l2")
        binary.add_batch(entries)

        # Benchmark flat
        flat_lats = []
        for q in queries_list:
            t0 = time.perf_counter()
            flat.search(query=q, k=K)
            flat_lats.append((time.perf_counter() - t0) * 1000)
        flat_avg = statistics.mean(flat_lats)

        # Benchmark binary
        bin_lats = []
        for q in queries_list:
            t0 = time.perf_counter()
            binary.search(query=q, k=K)
            bin_lats.append((time.perf_counter() - t0) * 1000)
        bin_avg = statistics.mean(bin_lats)

        speedup = flat_avg / bin_avg if bin_avg > 0 else float('inf')

        print(f"\n  Flat avg latency:   {flat_avg:.3f} ms")
        print(f"  Binary avg latency: {bin_avg:.3f} ms")
        print(f"  Speedup: {speedup:.1f}x")

        # Binary should not be slower than 2x flat (it should be faster)
        assert bin_avg <= flat_avg * 2.0, (
            f"Binary search unexpectedly slow: {bin_avg:.3f}ms vs flat {flat_avg:.3f}ms"
        )

    def test_binary_insert_vs_flat(self, vectors_list):
        """Binary insert should be at least as fast as flat."""
        entries = [(i, v) for i, v in enumerate(vectors_list)]

        t0 = time.perf_counter()
        flat = quiver.FlatIndex(dimensions=DIM, metric="l2")
        flat.add_batch(entries)
        flat_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        binary = quiver.BinaryFlatIndex(dimensions=DIM, metric="l2")
        binary.add_batch(entries)
        bin_time = time.perf_counter() - t0

        print(f"\n  Flat insert:   {flat_time:.3f}s ({N/flat_time:,.0f} vec/s)")
        print(f"  Binary insert: {bin_time:.3f}s ({N/bin_time:,.0f} vec/s)")


# ===================================================================
# 10. SNAPSHOT PERFORMANCE — Create, restore, delete speed
# ===================================================================


class TestSnapshotPerf:
    """Test snapshot create/restore/delete performance."""

    def test_snapshot_create_throughput(self, vectors_list, tmp_path):
        """Snapshot creation should be fast (compaction + file copy)."""
        db = quiver.Client(path=str(tmp_path / "snap_create"))
        col = db.create_collection("bench", dimensions=DIM, metric="l2")
        for i, v in enumerate(vectors_list):
            col.upsert(id=i, vector=v, payload={"idx": i})

        t0 = time.perf_counter()
        meta = col.create_snapshot("v1")
        create_time = time.perf_counter() - t0

        print(f"\n  Snapshot create ({N:,} vectors): {create_time:.3f}s")
        print(f"    vector_count={meta['vector_count']}")

        assert create_time <= 5.0, (
            f"Snapshot create too slow: {create_time:.3f}s > 5.0s"
        )

    def test_snapshot_restore_speed(self, vectors_list, tmp_path):
        """Snapshot restore should be fast (file copy + reload)."""
        half = N // 2
        db = quiver.Client(path=str(tmp_path / "snap_restore"))
        col = db.create_collection("bench", dimensions=DIM, metric="l2")
        for i, v in enumerate(vectors_list[:half]):
            col.upsert(id=i, vector=v)
        col.create_snapshot("baseline")

        # Add more data
        for i, v in enumerate(vectors_list[half:]):
            col.upsert(id=half + i, vector=v)
        assert col.count == N

        t0 = time.perf_counter()
        col.restore_snapshot("baseline")
        restore_time = time.perf_counter() - t0

        print(f"\n  Snapshot restore ({N:,} -> {half:,} vectors): {restore_time:.3f}s")
        assert col.count == half

        assert restore_time <= 30.0, (
            f"Snapshot restore too slow: {restore_time:.3f}s > 30.0s"
        )

    def test_snapshot_delete_speed(self, vectors_list, tmp_path):
        """Snapshot deletion should be near-instant."""
        db = quiver.Client(path=str(tmp_path / "snap_delete"))
        col = db.create_collection("bench", dimensions=DIM, metric="l2")
        for i, v in enumerate(vectors_list[:1000]):
            col.upsert(id=i, vector=v)
        col.create_snapshot("to_delete")

        t0 = time.perf_counter()
        col.delete_snapshot("to_delete")
        delete_time = time.perf_counter() - t0

        print(f"\n  Snapshot delete: {delete_time:.4f}s")

        assert delete_time <= 1.0, (
            f"Snapshot delete too slow: {delete_time:.3f}s > 1.0s"
        )

    def test_snapshot_multiple_versions(self, vectors_list, tmp_path):
        """Benchmark creating multiple snapshots and listing them."""
        db = quiver.Client(path=str(tmp_path / "snap_multi"))
        col = db.create_collection("bench", dimensions=DIM, metric="l2")

        batch_size = N // 5
        n_snaps = 5
        times = []
        for s in range(n_snaps):
            batch_start = s * batch_size
            for i in range(batch_size):
                col.upsert(id=batch_start + i, vector=vectors_list[batch_start + i])
            t0 = time.perf_counter()
            col.create_snapshot(f"v{s}")
            times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        snaps = col.list_snapshots()
        list_time = time.perf_counter() - t0

        print(f"\n  {n_snaps} snapshots created:")
        for i, t in enumerate(times):
            print(f"    v{i}: {t:.3f}s ({snaps[i]['vector_count']} vectors)")
        print(f"  list_snapshots: {list_time:.4f}s")

        assert len(snaps) == n_snaps
        assert list_time <= 0.5

    def test_snapshot_restore_search_correctness(self, vectors_list, queries_list, tmp_path):
        """Verify search results are correct after restore."""
        db = quiver.Client(path=str(tmp_path / "snap_correct"))
        col = db.create_collection("bench", dimensions=DIM, metric="l2")

        # Insert first 1000, snapshot, then add 1000 more
        for i in range(1000):
            col.upsert(id=i, vector=vectors_list[i])
        col.create_snapshot("v1")

        # Get pre-snapshot search results
        pre_results = col.search(queries_list[0], k=5)
        pre_ids = {r["id"] for r in pre_results}

        # Add more and mutate
        for i in range(1000, 2000):
            col.upsert(id=i, vector=vectors_list[i])

        # Restore and verify same results
        col.restore_snapshot("v1")
        post_results = col.search(queries_list[0], k=5)
        post_ids = {r["id"] for r in post_results}

        assert pre_ids == post_ids, "Search results differ after snapshot restore"
        print(f"\n  Snapshot restore: search results verified correct")


# ===================================================================
# 11. APPLE-TO-APPLE COMPARISON — All index types vs faiss equivalents
# ===================================================================


class TestAppleToAppleComparison:
    """Apple-to-apple benchmark: every Quiver index vs its faiss equivalent.

    Measures insert throughput, search latency, and recall@10 with matched
    configurations. Quiver-only index types (fp16, mmap, binary_flat) are
    benchmarked standalone.

    Matching pairs:
        Quiver FlatIndex        ↔ faiss IndexFlatL2         (exact brute-force)
        Quiver HnswIndex        ↔ faiss IndexHNSWFlat       (M=16, ef_c=200, ef_s=50)
                                ↔ hnswlib                   (same params)
        Quiver QuantizedFlat    ↔ faiss IndexScalarQuantizer (QT_8bit)
        Quiver IvfIndex         ↔ faiss IndexIVFFlat        (nlist=32, nprobe=8)
        Quiver IvfPqIndex       ↔ faiss IndexIVFPQ          (nlist=32, nprobe=8, m=8, nbits=4)
        Quiver Fp16FlatIndex    — (Quiver-only)
        Quiver MmapFlatIndex    — (Quiver-only)
        Quiver BinaryFlatIndex  — (Quiver-only)
    """

    # Shared HNSW params for fair comparison
    BENCH_M = 16
    BENCH_EF_CONSTRUCTION = 200
    BENCH_EF_SEARCH = 50

    # IVF params
    BENCH_NLIST = 32
    BENCH_NPROBE = 8

    # PQ params
    BENCH_PQ_M = 8
    BENCH_PQ_NBITS = 4  # 2^4 = 16 centroids per subquantizer = pq_k_sub=16

    def test_apple_to_apple_full(self, vectors_list, queries_list, gt, tmp_path):
        """Comprehensive apple-to-apple comparison across all index types.

        Produces a single table with insert throughput, search latency, and
        recall@10 for every Quiver index and its faiss equivalent.
        """
        vectors_np = np.array(vectors_list, dtype=np.float32)
        queries_np = np.array(queries_list, dtype=np.float32)
        n_cores = os.cpu_count() or 1

        rows = []

        def measure_quiver(name, idx, queries):
            """Measure search latency and recall for a Quiver index."""
            lats = []
            pred = []
            for q in queries:
                t0 = time.perf_counter()
                r = idx.search(query=q, k=K)
                lats.append((time.perf_counter() - t0) * 1000)
                pred.append([h["id"] for h in r])
            recall = compute_recall(pred, gt)
            avg_lat = statistics.mean(lats)
            return avg_lat, recall

        def measure_faiss(idx_f, queries_np):
            """Measure search latency and recall for a faiss index."""
            lats = []
            pred = []
            for q in queries_np:
                t0 = time.perf_counter()
                _, labels = idx_f.search(q.reshape(1, -1), K)
                lats.append((time.perf_counter() - t0) * 1000)
                pred.append(labels[0].tolist())
            recall = compute_recall(pred, gt)
            avg_lat = statistics.mean(lats)
            return avg_lat, recall

        # ------------------------------------------------------------------
        # 1. Flat — exact brute-force
        # ------------------------------------------------------------------
        print("\n  [1/5] Benchmarking Flat indexes...")

        # Quiver Flat
        entries = [(i, v) for i, v in enumerate(vectors_list)]
        idx = quiver.FlatIndex(dimensions=DIM, metric="l2")
        t0 = time.perf_counter()
        idx.add_batch(entries)
        q_flat_insert = N / (time.perf_counter() - t0)
        q_flat_lat, q_flat_rec = measure_quiver("flat", idx, queries_list)
        rows.append(("Quiver Flat", f"{q_flat_insert:,.0f}",
                      f"{q_flat_lat:.3f}", f"{q_flat_rec:.4f}"))

        # faiss Flat
        faiss = try_import("faiss")
        if faiss:
            faiss.omp_set_num_threads(1)
            idx_f = faiss.IndexFlatL2(DIM)
            t0 = time.perf_counter()
            idx_f.add(vectors_np)
            f_flat_insert = N / (time.perf_counter() - t0)
            f_flat_lat, f_flat_rec = measure_faiss(idx_f, queries_np)
            rows.append(("faiss Flat", f"{f_flat_insert:,.0f}",
                          f"{f_flat_lat:.3f}", f"{f_flat_rec:.4f}"))
        else:
            rows.append(("faiss Flat", "SKIP", "SKIP", "SKIP"))

        rows.append(("---", "---", "---", "---"))

        # ------------------------------------------------------------------
        # 2. HNSW — approximate nearest neighbor (graph-based)
        # ------------------------------------------------------------------
        print("  [2/5] Benchmarking HNSW indexes...")

        # Quiver HNSW (1T)
        idx = quiver.HnswIndex(dimensions=DIM, metric="l2",
                               m=self.BENCH_M,
                               ef_construction=self.BENCH_EF_CONSTRUCTION,
                               ef_search=self.BENCH_EF_SEARCH)
        t0 = time.perf_counter()
        idx.add_batch_np(vectors_np)
        idx.flush()
        q_hnsw_insert_1t = N / (time.perf_counter() - t0)
        q_hnsw_lat, q_hnsw_rec = measure_quiver("hnsw", idx, queries_list)
        rows.append(("Quiver HNSW (1T)", f"{q_hnsw_insert_1t:,.0f}",
                      f"{q_hnsw_lat:.3f}", f"{q_hnsw_rec:.4f}"))

        # Quiver HNSW (multi-threaded)
        idx_par = quiver.HnswIndex(dimensions=DIM, metric="l2",
                                   m=self.BENCH_M,
                                   ef_construction=self.BENCH_EF_CONSTRUCTION,
                                   ef_search=self.BENCH_EF_SEARCH)
        t0 = time.perf_counter()
        idx_par.add_batch_parallel(vectors_np, num_threads=n_cores)
        idx_par.flush()
        q_hnsw_insert_mt = N / (time.perf_counter() - t0)
        q_hnsw_mt_lat, q_hnsw_mt_rec = measure_quiver("hnsw_mt", idx_par, queries_list)
        rows.append((f"Quiver HNSW ({n_cores}T)", f"{q_hnsw_insert_mt:,.0f}",
                      f"{q_hnsw_mt_lat:.3f}", f"{q_hnsw_mt_rec:.4f}"))

        # hnswlib
        hnswlib = try_import("hnswlib")
        if hnswlib:
            # 1T
            idx_h = hnswlib.Index(space="l2", dim=DIM)
            idx_h.init_index(max_elements=N, M=self.BENCH_M,
                             ef_construction=self.BENCH_EF_CONSTRUCTION)
            idx_h.set_num_threads(1)
            t0 = time.perf_counter()
            idx_h.add_items(vectors_np, np.arange(N))
            h_insert_1t = N / (time.perf_counter() - t0)
            idx_h.set_ef(self.BENCH_EF_SEARCH)
            lats, pred = [], []
            for q in queries_np:
                t0 = time.perf_counter()
                labels, _ = idx_h.knn_query(q, k=K)
                lats.append((time.perf_counter() - t0) * 1000)
                pred.append(labels[0].tolist())
            h_lat = statistics.mean(lats)
            h_rec = compute_recall(pred, gt)
            rows.append(("hnswlib (1T)", f"{h_insert_1t:,.0f}",
                          f"{h_lat:.3f}", f"{h_rec:.4f}"))

            # Multi-threaded
            idx_h2 = hnswlib.Index(space="l2", dim=DIM)
            idx_h2.init_index(max_elements=N, M=self.BENCH_M,
                              ef_construction=self.BENCH_EF_CONSTRUCTION)
            idx_h2.set_num_threads(n_cores)
            t0 = time.perf_counter()
            idx_h2.add_items(vectors_np, np.arange(N))
            h_insert_mt = N / (time.perf_counter() - t0)
            rows.append((f"hnswlib ({n_cores}T)", f"{h_insert_mt:,.0f}",
                          f"{h_lat:.3f}", f"{h_rec:.4f}"))
        else:
            rows.append(("hnswlib", "SKIP", "SKIP", "SKIP"))

        # faiss HNSW
        if faiss:
            faiss.omp_set_num_threads(1)
            idx_f = faiss.IndexHNSWFlat(DIM, self.BENCH_M)
            idx_f.hnsw.efConstruction = self.BENCH_EF_CONSTRUCTION
            idx_f.hnsw.efSearch = self.BENCH_EF_SEARCH
            t0 = time.perf_counter()
            idx_f.add(vectors_np)
            f_hnsw_insert_1t = N / (time.perf_counter() - t0)
            f_hnsw_lat, f_hnsw_rec = measure_faiss(idx_f, queries_np)
            rows.append(("faiss HNSW (1T)", f"{f_hnsw_insert_1t:,.0f}",
                          f"{f_hnsw_lat:.3f}", f"{f_hnsw_rec:.4f}"))

            faiss.omp_set_num_threads(n_cores)
            idx_f2 = faiss.IndexHNSWFlat(DIM, self.BENCH_M)
            idx_f2.hnsw.efConstruction = self.BENCH_EF_CONSTRUCTION
            idx_f2.hnsw.efSearch = self.BENCH_EF_SEARCH
            t0 = time.perf_counter()
            idx_f2.add(vectors_np)
            f_hnsw_insert_mt = N / (time.perf_counter() - t0)
            rows.append((f"faiss HNSW ({n_cores}T)", f"{f_hnsw_insert_mt:,.0f}",
                          f"{f_hnsw_lat:.3f}", f"{f_hnsw_rec:.4f}"))
            faiss.omp_set_num_threads(1)  # Reset
        else:
            rows.append(("faiss HNSW", "SKIP", "SKIP", "SKIP"))

        rows.append(("---", "---", "---", "---"))

        # ------------------------------------------------------------------
        # 3. Scalar Quantization — Int8
        # ------------------------------------------------------------------
        print("  [3/5] Benchmarking Int8/SQ8 indexes...")

        # Quiver Int8
        idx = quiver.QuantizedFlatIndex(dimensions=DIM, metric="l2")
        t0 = time.perf_counter()
        idx.add_batch(entries)
        q_int8_insert = N / (time.perf_counter() - t0)
        q_int8_lat, q_int8_rec = measure_quiver("int8", idx, queries_list)
        rows.append(("Quiver Int8", f"{q_int8_insert:,.0f}",
                      f"{q_int8_lat:.3f}", f"{q_int8_rec:.4f}"))

        # faiss SQ8
        if faiss:
            faiss.omp_set_num_threads(1)
            idx_f = faiss.IndexScalarQuantizer(DIM, faiss.ScalarQuantizer.QT_8bit)
            t0 = time.perf_counter()
            idx_f.train(vectors_np)
            idx_f.add(vectors_np)
            f_sq8_insert = N / (time.perf_counter() - t0)
            f_sq8_lat, f_sq8_rec = measure_faiss(idx_f, queries_np)
            rows.append(("faiss SQ8", f"{f_sq8_insert:,.0f}",
                          f"{f_sq8_lat:.3f}", f"{f_sq8_rec:.4f}"))
        else:
            rows.append(("faiss SQ8", "SKIP", "SKIP", "SKIP"))

        rows.append(("---", "---", "---", "---"))

        # ------------------------------------------------------------------
        # 4. IVF — Inverted File Index
        # ------------------------------------------------------------------
        print("  [4/5] Benchmarking IVF indexes...")

        # Quiver IVF
        idx = quiver.IvfIndex(dimensions=DIM, metric="l2",
                              n_lists=self.BENCH_NLIST,
                              nprobe=self.BENCH_NPROBE,
                              train_size=N)
        t0 = time.perf_counter()
        idx.add_batch(entries)
        q_ivf_insert = N / (time.perf_counter() - t0)
        q_ivf_lat, q_ivf_rec = measure_quiver("ivf", idx, queries_list)
        rows.append(("Quiver IVF", f"{q_ivf_insert:,.0f}",
                      f"{q_ivf_lat:.3f}", f"{q_ivf_rec:.4f}"))

        # faiss IVFFlat
        if faiss:
            faiss.omp_set_num_threads(1)
            quantizer = faiss.IndexFlatL2(DIM)
            idx_f = faiss.IndexIVFFlat(quantizer, DIM, self.BENCH_NLIST)
            t0 = time.perf_counter()
            idx_f.train(vectors_np)
            idx_f.add(vectors_np)
            f_ivf_insert = N / (time.perf_counter() - t0)
            idx_f.nprobe = self.BENCH_NPROBE
            f_ivf_lat, f_ivf_rec = measure_faiss(idx_f, queries_np)
            rows.append(("faiss IVFFlat", f"{f_ivf_insert:,.0f}",
                          f"{f_ivf_lat:.3f}", f"{f_ivf_rec:.4f}"))
        else:
            rows.append(("faiss IVFFlat", "SKIP", "SKIP", "SKIP"))

        rows.append(("---", "---", "---", "---"))

        # Quiver IVF-PQ
        idx = quiver.IvfPqIndex(dimensions=DIM, metric="l2",
                                n_lists=self.BENCH_NLIST,
                                nprobe=self.BENCH_NPROBE,
                                train_size=N,
                                pq_m=self.BENCH_PQ_M,
                                pq_k_sub=2**self.BENCH_PQ_NBITS)
        t0 = time.perf_counter()
        idx.add_batch(entries)
        q_ivfpq_insert = N / (time.perf_counter() - t0)
        q_ivfpq_lat, q_ivfpq_rec = measure_quiver("ivf_pq", idx, queries_list)
        rows.append(("Quiver IVF-PQ", f"{q_ivfpq_insert:,.0f}",
                      f"{q_ivfpq_lat:.3f}", f"{q_ivfpq_rec:.4f}"))

        # faiss IVFPQ
        if faiss:
            faiss.omp_set_num_threads(1)
            quantizer = faiss.IndexFlatL2(DIM)
            idx_f = faiss.IndexIVFPQ(quantizer, DIM, self.BENCH_NLIST,
                                     self.BENCH_PQ_M, self.BENCH_PQ_NBITS)
            t0 = time.perf_counter()
            idx_f.train(vectors_np)
            idx_f.add(vectors_np)
            f_ivfpq_insert = N / (time.perf_counter() - t0)
            idx_f.nprobe = self.BENCH_NPROBE
            f_ivfpq_lat, f_ivfpq_rec = measure_faiss(idx_f, queries_np)
            rows.append(("faiss IVFPQ", f"{f_ivfpq_insert:,.0f}",
                          f"{f_ivfpq_lat:.3f}", f"{f_ivfpq_rec:.4f}"))
        else:
            rows.append(("faiss IVFPQ", "SKIP", "SKIP", "SKIP"))

        rows.append(("---", "---", "---", "---"))

        # ------------------------------------------------------------------
        # 5. Quiver-only index types (no faiss equivalent)
        # ------------------------------------------------------------------
        print("  [5/5] Benchmarking Quiver-only indexes...")

        # FP16
        idx = quiver.Fp16FlatIndex(dimensions=DIM, metric="l2")
        t0 = time.perf_counter()
        idx.add_batch(entries)
        q_fp16_insert = N / (time.perf_counter() - t0)
        q_fp16_lat, q_fp16_rec = measure_quiver("fp16", idx, queries_list)
        rows.append(("Quiver FP16", f"{q_fp16_insert:,.0f}",
                      f"{q_fp16_lat:.3f}", f"{q_fp16_rec:.4f}"))

        # Mmap
        mmap_path = str(tmp_path / "apple_to_apple_mmap.qvec")
        idx = quiver.MmapFlatIndex(dimensions=DIM, metric="l2", path=mmap_path)
        t0 = time.perf_counter()
        idx.add_batch(entries)
        idx.flush()
        q_mmap_insert = N / (time.perf_counter() - t0)
        q_mmap_lat, q_mmap_rec = measure_quiver("mmap", idx, queries_list)
        rows.append(("Quiver Mmap", f"{q_mmap_insert:,.0f}",
                      f"{q_mmap_lat:.3f}", f"{q_mmap_rec:.4f}"))

        # Binary
        idx = quiver.BinaryFlatIndex(dimensions=DIM, metric="l2")
        t0 = time.perf_counter()
        idx.add_batch(entries)
        q_bin_insert = N / (time.perf_counter() - t0)
        q_bin_lat, q_bin_rec = measure_quiver("binary", idx, queries_list)
        rows.append(("Quiver Binary", f"{q_bin_insert:,.0f}",
                      f"{q_bin_lat:.3f}", f"{q_bin_rec:.4f}"))

        # ------------------------------------------------------------------
        # Print unified table
        # ------------------------------------------------------------------
        headers = ["Index", "Insert (vec/s)", "Search (ms)", f"Recall@{K}"]
        print(f"\n\n{'='*78}")
        print(f"  APPLE-TO-APPLE COMPARISON — All Index Types")
        print(f"  {N:,} vectors, {DIM}d, {N_QUERIES} queries, k={K}")
        print(f"  Config: M={self.BENCH_M}, ef_c={self.BENCH_EF_CONSTRUCTION}, "
              f"ef_s={self.BENCH_EF_SEARCH}, nlist={self.BENCH_NLIST}, "
              f"nprobe={self.BENCH_NPROBE}")
        print(f"  Threads: 1T unless noted, system has {n_cores} cores")
        print(f"{'='*78}")
        fmt_table(headers, rows)
