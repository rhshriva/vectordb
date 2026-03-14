"""Competitive benchmarks: Quiver vs faiss/hnswlib — apple-to-apple comparison.

Measures insert throughput, search latency, and recall@10 for every Quiver
index type against its faiss equivalent (when available).

Uses add_batch_np (numpy buffer protocol) for all Quiver indexes to make the
comparison fair — both Quiver and faiss receive contiguous numpy arrays.

Run:
    pytest tests/test_competitive_benchmarks.py -v -s

Install competitors for full comparison:
    pip install hnswlib faiss-cpu numpy
"""

import os
import time
import statistics
import pytest
import numpy as np
import quiver_vector_db as quiver


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DIM = 128
N = 2_000
N_QUERIES = 100
K = 10
SEED = 42


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
# Competitive Benchmarks
# ---------------------------------------------------------------------------

class TestCompetitiveBenchmarks:
    """Apple-to-apple benchmark: every Quiver index vs its faiss equivalent.

    Uses add_batch_np (numpy buffer protocol) for all Quiver indexes so that
    both Quiver and faiss receive contiguous numpy arrays — a fair comparison.

    Measures insert throughput, search latency, and recall@10 with matched
    configurations. Quiver-only index types (fp16, mmap, binary_flat) are
    benchmarked standalone.

    Matching pairs:
        Quiver FlatIndex        <-> faiss IndexFlatL2         (exact brute-force)
        Quiver HnswIndex        <-> faiss IndexHNSWFlat       (M=16, ef_c=200, ef_s=50)
                                <-> hnswlib                   (same params)
        Quiver QuantizedFlat    <-> faiss IndexScalarQuantizer (QT_8bit)
        Quiver IvfIndex         <-> faiss IndexIVFFlat        (nlist=32, nprobe=8)
        Quiver IvfPqIndex       <-> faiss IndexIVFPQ          (nlist=32, nprobe=8, m=8, nbits=4)
        Quiver Fp16FlatIndex    - (Quiver-only)
        Quiver MmapFlatIndex    - (Quiver-only)
        Quiver BinaryFlatIndex  - (Quiver-only)
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

    def test_competitive_full(self, vectors_np, vectors_list, queries_np, queries_list, gt, tmp_path):
        """Comprehensive apple-to-apple comparison across all index types."""
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

        # Quiver Flat (using add_batch_np)
        idx = quiver.FlatIndex(dimensions=DIM, metric="l2")
        t0 = time.perf_counter()
        idx.add_batch_np(vectors_np)
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

        # Quiver Int8 (using add_batch_np)
        idx = quiver.QuantizedFlatIndex(dimensions=DIM, metric="l2")
        t0 = time.perf_counter()
        idx.add_batch_np(vectors_np)
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

        # Quiver IVF (using add_batch_np)
        idx = quiver.IvfIndex(dimensions=DIM, metric="l2",
                              n_lists=self.BENCH_NLIST,
                              nprobe=self.BENCH_NPROBE,
                              train_size=N)
        t0 = time.perf_counter()
        idx.add_batch_np(vectors_np)
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

        # Quiver IVF-PQ (using add_batch_np)
        idx = quiver.IvfPqIndex(dimensions=DIM, metric="l2",
                                n_lists=self.BENCH_NLIST,
                                nprobe=self.BENCH_NPROBE,
                                train_size=N,
                                pq_m=self.BENCH_PQ_M,
                                pq_k_sub=2**self.BENCH_PQ_NBITS)
        t0 = time.perf_counter()
        idx.add_batch_np(vectors_np)
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

        # FP16 (using add_batch_np)
        idx = quiver.Fp16FlatIndex(dimensions=DIM, metric="l2")
        t0 = time.perf_counter()
        idx.add_batch_np(vectors_np)
        q_fp16_insert = N / (time.perf_counter() - t0)
        q_fp16_lat, q_fp16_rec = measure_quiver("fp16", idx, queries_list)
        rows.append(("Quiver FP16", f"{q_fp16_insert:,.0f}",
                      f"{q_fp16_lat:.3f}", f"{q_fp16_rec:.4f}"))

        # Mmap (using add_batch_np)
        mmap_path = str(tmp_path / "competitive_mmap.qvec")
        idx = quiver.MmapFlatIndex(dimensions=DIM, metric="l2", path=mmap_path)
        t0 = time.perf_counter()
        idx.add_batch_np(vectors_np)
        idx.flush()
        q_mmap_insert = N / (time.perf_counter() - t0)
        q_mmap_lat, q_mmap_rec = measure_quiver("mmap", idx, queries_list)
        rows.append(("Quiver Mmap", f"{q_mmap_insert:,.0f}",
                      f"{q_mmap_lat:.3f}", f"{q_mmap_rec:.4f}"))

        # Binary (using add_batch_np)
        idx = quiver.BinaryFlatIndex(dimensions=DIM, metric="l2")
        t0 = time.perf_counter()
        idx.add_batch_np(vectors_np)
        q_bin_insert = N / (time.perf_counter() - t0)
        q_bin_lat, q_bin_rec = measure_quiver("binary", idx, queries_list)
        rows.append(("Quiver Binary", f"{q_bin_insert:,.0f}",
                      f"{q_bin_lat:.3f}", f"{q_bin_rec:.4f}"))

        # ------------------------------------------------------------------
        # Print unified table
        # ------------------------------------------------------------------
        headers = ["Index", "Insert (vec/s)", "Search (ms)", f"Recall@{K}"]
        print(f"\n\n{'='*78}")
        print(f"  COMPETITIVE BENCHMARKS — All Index Types")
        print(f"  {N:,} vectors, {DIM}d, {N_QUERIES} queries, k={K}")
        print(f"  Config: M={self.BENCH_M}, ef_c={self.BENCH_EF_CONSTRUCTION}, "
              f"ef_s={self.BENCH_EF_SEARCH}, nlist={self.BENCH_NLIST}, "
              f"nprobe={self.BENCH_NPROBE}")
        print(f"  Threads: 1T unless noted, system has {n_cores} cores")
        print(f"{'='*78}")
        fmt_table(headers, rows)
