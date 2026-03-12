"""Competitive benchmarks: Quiver vs other embedded vector databases.

Compares insert throughput, search latency, recall, and quantization
across Quiver, hnswlib, faiss, usearch, and chromadb.

Run with:
    pip install hnswlib faiss-cpu chromadb usearch numpy
    pytest tests/test_benchmark.py -v -s

Missing libraries are skipped gracefully.
"""

import os
import time
import random
import statistics
import tempfile
import pytest
import numpy as np
import quiver_vector_db as quiver


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DIM = 128
N = 10_000
N_QUERIES = 1_000
K = 10
SEED = 42


def gen_vectors_np(n, dim=DIM, seed=SEED):
    rng = np.random.RandomState(seed)
    return rng.randn(n, dim).astype(np.float32)


def gen_vectors_list(n, dim=DIM, seed=SEED):
    rng = random.Random(seed)
    return [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(n)]


def fmt_table(headers, rows):
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |")
    print(sep)


def ground_truth_ids(vectors_np, queries_np, k):
    """Compute exact L2 top-k using faiss or brute-force numpy."""
    try:
        import faiss
        idx = faiss.IndexFlatL2(vectors_np.shape[1])
        idx.add(vectors_np)
        _, ids = idx.search(queries_np, k)
        return ids
    except ImportError:
        # Fallback to numpy brute force
        results = []
        for q in queries_np:
            dists = np.sum((vectors_np - q) ** 2, axis=1)
            results.append(np.argsort(dists)[:k])
        return np.array(results)


def compute_recall(predicted_ids, true_ids):
    """Average recall across queries."""
    recalls = []
    for pred, true in zip(predicted_ids, true_ids):
        pred_set = set(int(x) for x in pred)
        true_set = set(int(x) for x in true)
        recalls.append(len(pred_set & true_set) / len(true_set))
    return statistics.mean(recalls)


# ===================================================================
# 1. HNSW COMPARISON
# ===================================================================


class TestHnswComparison:
    """Compare HNSW implementations across libraries."""

    def test_hnsw_insert_throughput(self):
        vectors_np = gen_vectors_np(N)
        vectors_list = vectors_np.tolist()
        ids = np.arange(N)
        rows = []

        # Quiver HNSW
        idx = quiver.HnswIndex(dimensions=DIM, metric="l2")
        t0 = time.perf_counter()
        idx.add_batch([(int(i), v) for i, v in zip(ids, vectors_list)])
        idx.flush()
        t = time.perf_counter() - t0
        rows.append(("Quiver HNSW", f"{t:.3f}s", f"{N/t:,.0f} vec/s"))

        # hnswlib
        try:
            import hnswlib
            idx_h = hnswlib.Index(space="l2", dim=DIM)
            idx_h.init_index(max_elements=N, M=16, ef_construction=200)
            t0 = time.perf_counter()
            idx_h.add_items(vectors_np, ids)
            t = time.perf_counter() - t0
            rows.append(("hnswlib", f"{t:.3f}s", f"{N/t:,.0f} vec/s"))
        except ImportError:
            rows.append(("hnswlib", "SKIP", "not installed"))

        # faiss HNSW
        try:
            import faiss
            idx_f = faiss.IndexHNSWFlat(DIM, 16)  # M=16
            idx_f.hnsw.efConstruction = 200
            t0 = time.perf_counter()
            idx_f.add(vectors_np)
            t = time.perf_counter() - t0
            rows.append(("faiss HNSW", f"{t:.3f}s", f"{N/t:,.0f} vec/s"))
        except ImportError:
            rows.append(("faiss HNSW", "SKIP", "not installed"))

        # usearch
        try:
            from usearch.index import Index, MetricKind
            idx_u = Index(ndim=DIM, metric=MetricKind.L2sq, connectivity=16, expansion_add=200)
            t0 = time.perf_counter()
            idx_u.add(ids, vectors_np)
            t = time.perf_counter() - t0
            rows.append(("usearch", f"{t:.3f}s", f"{N/t:,.0f} vec/s"))
        except ImportError:
            rows.append(("usearch", "SKIP", "not installed"))

        # chromadb
        try:
            import chromadb
            client = chromadb.Client()
            col = client.create_collection("bench_insert", metadata={"hnsw:M": 16, "hnsw:construction_ef": 200})
            str_ids = [str(i) for i in range(N)]
            embeddings = vectors_list
            t0 = time.perf_counter()
            # chromadb add in batches of 5000 (API limit)
            batch = 5000
            for start in range(0, N, batch):
                end = min(start + batch, N)
                col.add(ids=str_ids[start:end], embeddings=embeddings[start:end])
            t = time.perf_counter() - t0
            rows.append(("chromadb", f"{t:.3f}s", f"{N/t:,.0f} vec/s"))
        except ImportError:
            rows.append(("chromadb", "SKIP", "not installed"))

        print(f"\n\n=== HNSW Insert Throughput ({N} vectors, {DIM}d) ===")
        fmt_table(["Library", "Time", "Throughput"], rows)

    def test_hnsw_search_latency(self):
        vectors_np = gen_vectors_np(N)
        vectors_list = vectors_np.tolist()
        queries_np = gen_vectors_np(N_QUERIES, seed=999)
        queries_list = queries_np.tolist()
        ids = np.arange(N)
        rows = []

        # Quiver HNSW
        idx = quiver.HnswIndex(dimensions=DIM, metric="l2", ef_search=50)
        idx.add_batch([(int(i), v) for i, v in zip(ids, vectors_list)])
        idx.flush()
        lats = []
        for q in queries_list:
            t0 = time.perf_counter()
            idx.search(query=q, k=K)
            lats.append((time.perf_counter() - t0) * 1000)
        rows.append(("Quiver HNSW", f"{statistics.mean(lats):.3f}", f"{statistics.median(lats):.3f}", f"{sorted(lats)[int(0.99*len(lats))]:.3f}"))

        # hnswlib
        try:
            import hnswlib
            idx_h = hnswlib.Index(space="l2", dim=DIM)
            idx_h.init_index(max_elements=N, M=16, ef_construction=200)
            idx_h.add_items(vectors_np, ids)
            idx_h.set_ef(50)
            lats = []
            for q in queries_np:
                t0 = time.perf_counter()
                idx_h.knn_query(q, k=K)
                lats.append((time.perf_counter() - t0) * 1000)
            rows.append(("hnswlib", f"{statistics.mean(lats):.3f}", f"{statistics.median(lats):.3f}", f"{sorted(lats)[int(0.99*len(lats))]:.3f}"))
        except ImportError:
            rows.append(("hnswlib", "SKIP", "", ""))

        # faiss HNSW
        try:
            import faiss
            idx_f = faiss.IndexHNSWFlat(DIM, 16)
            idx_f.hnsw.efConstruction = 200
            idx_f.hnsw.efSearch = 50
            idx_f.add(vectors_np)
            lats = []
            for q in queries_np:
                t0 = time.perf_counter()
                idx_f.search(q.reshape(1, -1), K)
                lats.append((time.perf_counter() - t0) * 1000)
            rows.append(("faiss HNSW", f"{statistics.mean(lats):.3f}", f"{statistics.median(lats):.3f}", f"{sorted(lats)[int(0.99*len(lats))]:.3f}"))
        except ImportError:
            rows.append(("faiss HNSW", "SKIP", "", ""))

        # usearch
        try:
            from usearch.index import Index, MetricKind
            idx_u = Index(ndim=DIM, metric=MetricKind.L2sq, connectivity=16, expansion_add=200, expansion_search=50)
            idx_u.add(ids, vectors_np)
            lats = []
            for q in queries_np:
                t0 = time.perf_counter()
                idx_u.search(q, K)
                lats.append((time.perf_counter() - t0) * 1000)
            rows.append(("usearch", f"{statistics.mean(lats):.3f}", f"{statistics.median(lats):.3f}", f"{sorted(lats)[int(0.99*len(lats))]:.3f}"))
        except ImportError:
            rows.append(("usearch", "SKIP", "", ""))

        print(f"\n\n=== HNSW Search Latency ({N} vectors, {N_QUERIES} queries, k={K}) ===")
        fmt_table(["Library", "Avg (ms)", "p50 (ms)", "p99 (ms)"], rows)

    def test_hnsw_recall(self):
        vectors_np = gen_vectors_np(N)
        vectors_list = vectors_np.tolist()
        queries_np = gen_vectors_np(100, seed=999)
        queries_list = queries_np.tolist()
        ids = np.arange(N)
        gt = ground_truth_ids(vectors_np, queries_np, K)
        rows = []

        # Quiver HNSW
        idx = quiver.HnswIndex(dimensions=DIM, metric="l2", ef_search=50)
        idx.add_batch([(int(i), v) for i, v in zip(ids, vectors_list)])
        idx.flush()
        pred = []
        for q in queries_list:
            r = idx.search(query=q, k=K)
            pred.append([h["id"] for h in r])
        rows.append(("Quiver HNSW", f"{compute_recall(pred, gt):.4f}"))

        # hnswlib
        try:
            import hnswlib
            idx_h = hnswlib.Index(space="l2", dim=DIM)
            idx_h.init_index(max_elements=N, M=16, ef_construction=200)
            idx_h.add_items(vectors_np, ids)
            idx_h.set_ef(50)
            pred = []
            for q in queries_np:
                labels, _ = idx_h.knn_query(q, k=K)
                pred.append(labels[0].tolist())
            rows.append(("hnswlib", f"{compute_recall(pred, gt):.4f}"))
        except ImportError:
            rows.append(("hnswlib", "SKIP"))

        # faiss HNSW
        try:
            import faiss
            idx_f = faiss.IndexHNSWFlat(DIM, 16)
            idx_f.hnsw.efConstruction = 200
            idx_f.hnsw.efSearch = 50
            idx_f.add(vectors_np)
            pred = []
            for q in queries_np:
                _, labels = idx_f.search(q.reshape(1, -1), K)
                pred.append(labels[0].tolist())
            rows.append(("faiss HNSW", f"{compute_recall(pred, gt):.4f}"))
        except ImportError:
            rows.append(("faiss HNSW", "SKIP"))

        # usearch
        try:
            from usearch.index import Index, MetricKind
            idx_u = Index(ndim=DIM, metric=MetricKind.L2sq, connectivity=16, expansion_add=200, expansion_search=50)
            idx_u.add(ids, vectors_np)
            pred = []
            for q in queries_np:
                matches = idx_u.search(q, K)
                pred.append(matches.keys.tolist())
            rows.append(("usearch", f"{compute_recall(pred, gt):.4f}"))
        except ImportError:
            rows.append(("usearch", "SKIP"))

        print(f"\n\n=== HNSW Recall@{K} ({N} vectors, 100 queries) ===")
        fmt_table(["Library", "Recall@10"], rows)


# ===================================================================
# 2. INT8 QUANTIZATION COMPARISON
# ===================================================================


class TestInt8Quantization:
    """Compare int8/scalar quantized indexes."""

    def test_int8_search_latency_and_recall(self):
        vectors_np = gen_vectors_np(N)
        vectors_list = vectors_np.tolist()
        queries_np = gen_vectors_np(100, seed=999)
        queries_list = queries_np.tolist()
        gt = ground_truth_ids(vectors_np, queries_np, K)
        rows = []

        # Quiver QuantizedFlatIndex (int8)
        idx = quiver.QuantizedFlatIndex(dimensions=DIM, metric="l2")
        idx.add_batch([(i, v) for i, v in enumerate(vectors_list)])
        lats = []
        pred = []
        for q in queries_list:
            t0 = time.perf_counter()
            r = idx.search(query=q, k=K)
            lats.append((time.perf_counter() - t0) * 1000)
            pred.append([h["id"] for h in r])
        recall = compute_recall(pred, gt)
        rows.append(("Quiver Int8", f"{statistics.mean(lats):.3f}", f"{recall:.4f}"))

        # faiss ScalarQuantizer (QT_8bit)
        try:
            import faiss
            idx_f = faiss.IndexScalarQuantizer(DIM, faiss.ScalarQuantizer.QT_8bit)
            idx_f.train(vectors_np)
            idx_f.add(vectors_np)
            lats = []
            pred = []
            for q in queries_np:
                t0 = time.perf_counter()
                d, labels = idx_f.search(q.reshape(1, -1), K)
                lats.append((time.perf_counter() - t0) * 1000)
                pred.append(labels[0].tolist())
            recall = compute_recall(pred, gt)
            rows.append(("faiss SQ8", f"{statistics.mean(lats):.3f}", f"{recall:.4f}"))
        except ImportError:
            rows.append(("faiss SQ8", "SKIP", ""))

        # usearch i8
        try:
            from usearch.index import Index, MetricKind, ScalarKind
            idx_u = Index(ndim=DIM, metric=MetricKind.L2sq, dtype=ScalarKind.I8)
            idx_u.add(np.arange(N), vectors_np)
            lats = []
            pred = []
            for q in queries_np:
                t0 = time.perf_counter()
                matches = idx_u.search(q, K)
                lats.append((time.perf_counter() - t0) * 1000)
                pred.append(matches.keys.tolist())
            recall = compute_recall(pred, gt)
            rows.append(("usearch i8", f"{statistics.mean(lats):.3f}", f"{recall:.4f}"))
        except (ImportError, Exception) as e:
            rows.append(("usearch i8", "SKIP", str(e)[:20] if not isinstance(e, ImportError) else ""))

        print(f"\n\n=== Int8 Quantized: Latency & Recall ({N} vectors, 100 queries) ===")
        fmt_table(["Library", "Avg Latency (ms)", "Recall@10"], rows)

    def test_int8_file_size(self, tmp_path):
        vectors_np = gen_vectors_np(N)
        vectors_list = vectors_np.tolist()
        rows = []

        # Quiver
        idx = quiver.QuantizedFlatIndex(dimensions=DIM, metric="l2")
        idx.add_batch([(i, v) for i, v in enumerate(vectors_list)])
        p = str(tmp_path / "quiver_int8.bin")
        idx.save(p)
        rows.append(("Quiver Int8", f"{os.path.getsize(p)/1024/1024:.2f} MB"))

        # faiss
        try:
            import faiss
            idx_f = faiss.IndexScalarQuantizer(DIM, faiss.ScalarQuantizer.QT_8bit)
            idx_f.train(vectors_np)
            idx_f.add(vectors_np)
            p = str(tmp_path / "faiss_sq8.bin")
            faiss.write_index(idx_f, p)
            rows.append(("faiss SQ8", f"{os.path.getsize(p)/1024/1024:.2f} MB"))
        except ImportError:
            rows.append(("faiss SQ8", "SKIP"))

        # Raw f32 for reference
        raw_size = N * DIM * 4 / 1024 / 1024
        rows.append(("Raw f32 (ref)", f"{raw_size:.2f} MB"))

        print(f"\n\n=== Int8 Quantized: File Size ({N} vectors, {DIM}d) ===")
        fmt_table(["Library", "File Size"], rows)


# ===================================================================
# 3. FP16 QUANTIZATION COMPARISON
# ===================================================================


class TestFp16Quantization:
    """Compare float16 quantized indexes."""

    def test_fp16_search_latency_and_recall(self):
        vectors_np = gen_vectors_np(N)
        vectors_list = vectors_np.tolist()
        queries_np = gen_vectors_np(100, seed=999)
        queries_list = queries_np.tolist()
        gt = ground_truth_ids(vectors_np, queries_np, K)
        rows = []

        # Quiver Fp16FlatIndex
        idx = quiver.Fp16FlatIndex(dimensions=DIM, metric="l2")
        idx.add_batch([(i, v) for i, v in enumerate(vectors_list)])
        lats = []
        pred = []
        for q in queries_list:
            t0 = time.perf_counter()
            r = idx.search(query=q, k=K)
            lats.append((time.perf_counter() - t0) * 1000)
            pred.append([h["id"] for h in r])
        recall = compute_recall(pred, gt)
        rows.append(("Quiver FP16", f"{statistics.mean(lats):.3f}", f"{recall:.4f}"))

        # faiss ScalarQuantizer (QT_fp16)
        try:
            import faiss
            idx_f = faiss.IndexScalarQuantizer(DIM, faiss.ScalarQuantizer.QT_fp16)
            idx_f.train(vectors_np)
            idx_f.add(vectors_np)
            lats = []
            pred = []
            for q in queries_np:
                t0 = time.perf_counter()
                d, labels = idx_f.search(q.reshape(1, -1), K)
                lats.append((time.perf_counter() - t0) * 1000)
                pred.append(labels[0].tolist())
            recall = compute_recall(pred, gt)
            rows.append(("faiss FP16", f"{statistics.mean(lats):.3f}", f"{recall:.4f}"))
        except ImportError:
            rows.append(("faiss FP16", "SKIP", ""))

        # usearch f16
        try:
            from usearch.index import Index, MetricKind, ScalarKind
            idx_u = Index(ndim=DIM, metric=MetricKind.L2sq, dtype=ScalarKind.F16)
            idx_u.add(np.arange(N), vectors_np)
            lats = []
            pred = []
            for q in queries_np:
                t0 = time.perf_counter()
                matches = idx_u.search(q, K)
                lats.append((time.perf_counter() - t0) * 1000)
                pred.append(matches.keys.tolist())
            recall = compute_recall(pred, gt)
            rows.append(("usearch f16", f"{statistics.mean(lats):.3f}", f"{recall:.4f}"))
        except (ImportError, Exception) as e:
            rows.append(("usearch f16", "SKIP", str(e)[:20] if not isinstance(e, ImportError) else ""))

        print(f"\n\n=== FP16 Quantized: Latency & Recall ({N} vectors, 100 queries) ===")
        fmt_table(["Library", "Avg Latency (ms)", "Recall@10"], rows)

    def test_fp16_file_size(self, tmp_path):
        vectors_np = gen_vectors_np(N)
        vectors_list = vectors_np.tolist()
        rows = []

        # Quiver
        idx = quiver.Fp16FlatIndex(dimensions=DIM, metric="l2")
        idx.add_batch([(i, v) for i, v in enumerate(vectors_list)])
        p = str(tmp_path / "quiver_fp16.bin")
        idx.save(p)
        rows.append(("Quiver FP16", f"{os.path.getsize(p)/1024/1024:.2f} MB"))

        # faiss
        try:
            import faiss
            idx_f = faiss.IndexScalarQuantizer(DIM, faiss.ScalarQuantizer.QT_fp16)
            idx_f.train(vectors_np)
            idx_f.add(vectors_np)
            p = str(tmp_path / "faiss_fp16.bin")
            faiss.write_index(idx_f, p)
            rows.append(("faiss FP16", f"{os.path.getsize(p)/1024/1024:.2f} MB"))
        except ImportError:
            rows.append(("faiss FP16", "SKIP"))

        raw_size = N * DIM * 4 / 1024 / 1024
        rows.append(("Raw f32 (ref)", f"{raw_size:.2f} MB"))

        print(f"\n\n=== FP16 Quantized: File Size ({N} vectors, {DIM}d) ===")
        fmt_table(["Library", "File Size"], rows)


# ===================================================================
# 4. IVF-PQ COMPARISON
# ===================================================================


class TestIvfPqComparison:
    """Compare IVF-PQ product quantization indexes."""

    def test_ivfpq_search_latency_and_recall(self):
        vectors_np = gen_vectors_np(N)
        vectors_list = vectors_np.tolist()
        queries_np = gen_vectors_np(100, seed=999)
        queries_list = queries_np.tolist()
        gt = ground_truth_ids(vectors_np, queries_np, K)
        rows = []

        # Quiver IvfPqIndex
        idx = quiver.IvfPqIndex(
            dimensions=DIM, metric="l2",
            n_lists=32, nprobe=8, train_size=N,
            pq_m=8, pq_k_sub=16,
        )
        idx.add_batch([(i, v) for i, v in enumerate(vectors_list)])
        lats = []
        pred = []
        for q in queries_list:
            t0 = time.perf_counter()
            r = idx.search(query=q, k=K)
            lats.append((time.perf_counter() - t0) * 1000)
            pred.append([h["id"] for h in r])
        recall = compute_recall(pred, gt)
        rows.append(("Quiver IVF-PQ", f"{statistics.mean(lats):.3f}", f"{recall:.4f}"))

        # faiss IndexIVFPQ
        try:
            import faiss
            quantizer = faiss.IndexFlatL2(DIM)
            idx_f = faiss.IndexIVFPQ(quantizer, DIM, 32, 8, 8)  # n_lists=32, pq_m=8, nbits=8
            idx_f.train(vectors_np)
            idx_f.add(vectors_np)
            idx_f.nprobe = 8
            lats = []
            pred = []
            for q in queries_np:
                t0 = time.perf_counter()
                d, labels = idx_f.search(q.reshape(1, -1), K)
                lats.append((time.perf_counter() - t0) * 1000)
                pred.append(labels[0].tolist())
            recall = compute_recall(pred, gt)
            rows.append(("faiss IVF-PQ", f"{statistics.mean(lats):.3f}", f"{recall:.4f}"))
        except ImportError:
            rows.append(("faiss IVF-PQ", "SKIP", ""))

        print(f"\n\n=== IVF-PQ: Latency & Recall ({N} vectors, 100 queries) ===")
        fmt_table(["Library", "Avg Latency (ms)", "Recall@10"], rows)

    def test_ivfpq_file_size(self, tmp_path):
        vectors_np = gen_vectors_np(N)
        vectors_list = vectors_np.tolist()
        rows = []

        # Quiver
        idx = quiver.IvfPqIndex(
            dimensions=DIM, metric="l2",
            n_lists=32, nprobe=8, train_size=N,
            pq_m=8, pq_k_sub=16,
        )
        idx.add_batch([(i, v) for i, v in enumerate(vectors_list)])
        p = str(tmp_path / "quiver_ivfpq.bin")
        idx.save(p)
        rows.append(("Quiver IVF-PQ", f"{os.path.getsize(p)/1024/1024:.2f} MB"))

        # faiss
        try:
            import faiss
            quantizer = faiss.IndexFlatL2(DIM)
            idx_f = faiss.IndexIVFPQ(quantizer, DIM, 32, 8, 8)
            idx_f.train(vectors_np)
            idx_f.add(vectors_np)
            p = str(tmp_path / "faiss_ivfpq.bin")
            faiss.write_index(idx_f, p)
            rows.append(("faiss IVF-PQ", f"{os.path.getsize(p)/1024/1024:.2f} MB"))
        except ImportError:
            rows.append(("faiss IVF-PQ", "SKIP"))

        raw_size = N * DIM * 4 / 1024 / 1024
        rows.append(("Raw f32 (ref)", f"{raw_size:.2f} MB"))

        print(f"\n\n=== IVF-PQ: File Size ({N} vectors, {DIM}d) ===")
        fmt_table(["Library", "File Size"], rows)


# ===================================================================
# 5. SUMMARY TABLE
# ===================================================================


class TestSummary:
    """End-to-end summary across all index types and libraries."""

    def test_full_comparison(self):
        vectors_np = gen_vectors_np(N)
        vectors_list = vectors_np.tolist()
        queries_np = gen_vectors_np(100, seed=999)
        queries_list = queries_np.tolist()
        ids_np = np.arange(N)
        gt = ground_truth_ids(vectors_np, queries_np, K)
        rows = []

        def bench_quiver(name, idx, queries):
            lats = []
            pred = []
            for q in queries:
                t0 = time.perf_counter()
                r = idx.search(query=q, k=K)
                lats.append((time.perf_counter() - t0) * 1000)
                pred.append([h["id"] for h in r])
            recall = compute_recall(pred, gt)
            return f"{statistics.mean(lats):.3f}", f"{recall:.4f}"

        # Quiver indexes
        idx = quiver.FlatIndex(dimensions=DIM, metric="l2")
        idx.add_batch([(i, v) for i, v in enumerate(vectors_list)])
        lat, rec = bench_quiver("Flat", idx, queries_list)
        rows.append(("Quiver Flat", "exact", lat, rec))

        idx = quiver.HnswIndex(dimensions=DIM, metric="l2", ef_search=50)
        idx.add_batch([(i, v) for i, v in enumerate(vectors_list)])
        idx.flush()
        lat, rec = bench_quiver("HNSW", idx, queries_list)
        rows.append(("Quiver HNSW", "hnsw", lat, rec))

        idx = quiver.QuantizedFlatIndex(dimensions=DIM, metric="l2")
        idx.add_batch([(i, v) for i, v in enumerate(vectors_list)])
        lat, rec = bench_quiver("Int8", idx, queries_list)
        rows.append(("Quiver Int8", "int8", lat, rec))

        idx = quiver.Fp16FlatIndex(dimensions=DIM, metric="l2")
        idx.add_batch([(i, v) for i, v in enumerate(vectors_list)])
        lat, rec = bench_quiver("FP16", idx, queries_list)
        rows.append(("Quiver FP16", "fp16", lat, rec))

        idx = quiver.IvfPqIndex(dimensions=DIM, metric="l2", n_lists=32, nprobe=8, train_size=N, pq_m=8, pq_k_sub=16)
        idx.add_batch([(i, v) for i, v in enumerate(vectors_list)])
        lat, rec = bench_quiver("IVF-PQ", idx, queries_list)
        rows.append(("Quiver IVF-PQ", "ivf-pq", lat, rec))

        # faiss
        try:
            import faiss
            # Flat
            idx_f = faiss.IndexFlatL2(DIM)
            idx_f.add(vectors_np)
            lats = []
            pred = []
            for q in queries_np:
                t0 = time.perf_counter()
                _, l = idx_f.search(q.reshape(1, -1), K)
                lats.append((time.perf_counter() - t0) * 1000)
                pred.append(l[0].tolist())
            rows.append(("faiss Flat", "exact", f"{statistics.mean(lats):.3f}", f"{compute_recall(pred, gt):.4f}"))

            # HNSW
            idx_f = faiss.IndexHNSWFlat(DIM, 16)
            idx_f.hnsw.efConstruction = 200
            idx_f.hnsw.efSearch = 50
            idx_f.add(vectors_np)
            lats = []
            pred = []
            for q in queries_np:
                t0 = time.perf_counter()
                _, l = idx_f.search(q.reshape(1, -1), K)
                lats.append((time.perf_counter() - t0) * 1000)
                pred.append(l[0].tolist())
            rows.append(("faiss HNSW", "hnsw", f"{statistics.mean(lats):.3f}", f"{compute_recall(pred, gt):.4f}"))

            # SQ8
            idx_f = faiss.IndexScalarQuantizer(DIM, faiss.ScalarQuantizer.QT_8bit)
            idx_f.train(vectors_np)
            idx_f.add(vectors_np)
            lats = []
            pred = []
            for q in queries_np:
                t0 = time.perf_counter()
                _, l = idx_f.search(q.reshape(1, -1), K)
                lats.append((time.perf_counter() - t0) * 1000)
                pred.append(l[0].tolist())
            rows.append(("faiss SQ8", "int8", f"{statistics.mean(lats):.3f}", f"{compute_recall(pred, gt):.4f}"))

            # FP16
            idx_f = faiss.IndexScalarQuantizer(DIM, faiss.ScalarQuantizer.QT_fp16)
            idx_f.train(vectors_np)
            idx_f.add(vectors_np)
            lats = []
            pred = []
            for q in queries_np:
                t0 = time.perf_counter()
                _, l = idx_f.search(q.reshape(1, -1), K)
                lats.append((time.perf_counter() - t0) * 1000)
                pred.append(l[0].tolist())
            rows.append(("faiss FP16", "fp16", f"{statistics.mean(lats):.3f}", f"{compute_recall(pred, gt):.4f}"))

            # IVF-PQ
            quantizer = faiss.IndexFlatL2(DIM)
            idx_f = faiss.IndexIVFPQ(quantizer, DIM, 32, 8, 8)
            idx_f.train(vectors_np)
            idx_f.add(vectors_np)
            idx_f.nprobe = 8
            lats = []
            pred = []
            for q in queries_np:
                t0 = time.perf_counter()
                _, l = idx_f.search(q.reshape(1, -1), K)
                lats.append((time.perf_counter() - t0) * 1000)
                pred.append(l[0].tolist())
            rows.append(("faiss IVF-PQ", "ivf-pq", f"{statistics.mean(lats):.3f}", f"{compute_recall(pred, gt):.4f}"))
        except ImportError:
            pass

        # hnswlib
        try:
            import hnswlib
            idx_h = hnswlib.Index(space="l2", dim=DIM)
            idx_h.init_index(max_elements=N, M=16, ef_construction=200)
            idx_h.add_items(vectors_np, ids_np)
            idx_h.set_ef(50)
            lats = []
            pred = []
            for q in queries_np:
                t0 = time.perf_counter()
                labels, _ = idx_h.knn_query(q, k=K)
                lats.append((time.perf_counter() - t0) * 1000)
                pred.append(labels[0].tolist())
            rows.append(("hnswlib", "hnsw", f"{statistics.mean(lats):.3f}", f"{compute_recall(pred, gt):.4f}"))
        except ImportError:
            pass

        # usearch
        try:
            from usearch.index import Index, MetricKind
            idx_u = Index(ndim=DIM, metric=MetricKind.L2sq, connectivity=16, expansion_add=200, expansion_search=50)
            idx_u.add(ids_np, vectors_np)
            lats = []
            pred = []
            for q in queries_np:
                t0 = time.perf_counter()
                matches = idx_u.search(q, K)
                lats.append((time.perf_counter() - t0) * 1000)
                pred.append(matches.keys.tolist())
            rows.append(("usearch", "hnsw", f"{statistics.mean(lats):.3f}", f"{compute_recall(pred, gt):.4f}"))
        except ImportError:
            pass

        print(f"\n\n=== Full Comparison Summary ({N} vectors, {DIM}d, 100 queries, k={K}) ===")
        fmt_table(["Library", "Type", "Avg Latency (ms)", "Recall@10"], rows)
