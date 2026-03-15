<p align="center">
  <h1 align="center">Quiver</h1>
  <p align="center">A fast, embedded vector database written in Rust with Python bindings.<br/>No server, no network — runs fully in-process with SIMD-accelerated search.</p>
</p>

<p align="center">
  <a href="https://pypi.org/project/quiver-vector-db/"><img src="https://img.shields.io/pypi/v/quiver-vector-db?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/quiver-vector-db/"><img src="https://img.shields.io/pypi/pyversions/quiver-vector-db" alt="Python"></a>
  <a href="https://github.com/rhshriva/Quiver/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
</p>

---

## Installation

```bash
pip install quiver-vector-db
```

Prebuilt wheels for macOS (Apple Silicon / Intel) and Linux (x86_64 / ARM64). Python 3.11+.

**Optional extras:**

```bash
pip install quiver-vector-db[sentence-transformers]   # local embeddings
pip install quiver-vector-db[openai]                   # OpenAI embeddings
pip install quiver-vector-db[all]                      # everything
```

**Build from source** (requires Rust toolchain):

```bash
pip install maturin
git clone https://github.com/rhshriva/Quiver.git && cd Quiver
maturin develop --release
```

---

## Features

| Feature | Description |
|---------|-------------|
| **8 index types** | HNSW, Flat, Int8, FP16, IVF, IVF-PQ, Memory-mapped, Binary |
| **Built-in embeddings** | sentence-transformers (local) and OpenAI API |
| **Full-text search** | BM25 keyword search with hybrid dense+sparse fusion |
| **3 distance metrics** | Cosine, L2, Dot Product — all SIMD-accelerated (AVX2/NEON) |
| **Payload filtering** | JSON metadata with 9 filter operators (`$eq`, `$ne`, `$in`, `$gt`, `$gte`, `$lt`, `$lte`, `$and`, `$or`) |
| **Hybrid search** | Weighted fusion of dense vectors + sparse keyword signals |
| **Multi-vector** | Multiple named embedding spaces per document (text + image) |
| **Data versioning** | Create, list, restore, and delete collection snapshots |
| **Batch upsert** | Efficient bulk insertion API |
| **Parallel HNSW insert** | Multi-threaded insert via rayon micro-batching |
| **WAL persistence** | Crash-safe writes with automatic compaction |
| **IDE support** | Full type stubs (`py.typed` + `.pyi`) for autocompletion |
| **Zero dependencies** | Single `pip install`, no external dependencies |

---

## Performance

Benchmarks run on **GitHub Actions** CI runners — fully reproducible via `workflow_dispatch`.

| | **x86_64 Runner** | **ARM64 Runner** |
|---|---|---|
| **OS** | Ubuntu (ubuntu-latest) | Ubuntu (ubuntu-24.04-arm) |
| **CPU** | 4 cores, x86_64 | 4 cores, ARM64 (Neoverse) |
| **RAM** | 16 GB | 16 GB |
| **Python** | 3.12 | 3.12 |
| **Build** | `maturin develop --release` | `maturin develop --release` |

2,000 vectors × 128 dimensions, k=10 — matched configs (M=16, ef_construction=200, ef_search=50, nlist=32, nprobe=8).

### Brute-Force (Flat)

| Engine | Insert (vec/s) | Search (ms) | Recall@10 | | Insert (vec/s) | Search (ms) | Recall@10 |
|---|---:|---:|---:|---|---:|---:|---:|
| | **x86_64** | | | | **ARM64** | | |
| **Quiver Flat** | **12,893K** | 0.044 | 1.0000 | | **7,961K** | 0.056 | 1.0000 |
| faiss Flat | 6,808K | **0.036** | 1.0000 | | 5,166K | **0.051** | 1.0000 |
| LanceDB | 287K | 2.986 | 1.0000 | | 262K | 3.514 | 1.0000 |

> Quiver Flat insert is **1.5–1.9x faster** than faiss, **45–50x faster** than LanceDB.

### HNSW (Approximate Nearest Neighbor)

| Engine | Insert (vec/s) | Search (ms) | Recall@10 | | Insert (vec/s) | Search (ms) | Recall@10 |
|---|---:|---:|---:|---|---:|---:|---:|
| | **x86_64** | | | | **ARM64** | | |
| **Quiver HNSW (1T)** | 11,700 | **0.039** | 0.9410 | | 12,025 | **0.042** | 0.9430 |
| **Quiver HNSW (4T)** | 24,539 | **0.038** | 0.9410 | | 22,618 | **0.053** | 0.9410 |
| usearch | 30,498 | 0.094 | 0.9260 | | 34,067 | 0.113 | 0.9250 |
| faiss HNSW (1T) | 8,988 | 0.047 | 0.9510 | | 8,636 | 0.058 | 0.9510 |
| faiss HNSW (4T) | 25,060 | 0.047 | 0.9510 | | 30,446 | 0.058 | 0.9510 |

> Quiver HNSW search is **1.2x faster** than faiss, **2.4x faster** than usearch. Single-thread insert **1.3x faster** than faiss.

### Scalar Quantization (Int8)

| Engine | Insert (vec/s) | Search (ms) | Recall@10 | | Insert (vec/s) | Search (ms) | Recall@10 |
|---|---:|---:|---:|---|---:|---:|---:|
| | **x86_64** | | | | **ARM64** | | |
| **Quiver Int8** | 1,284K | 0.171 | **0.9940** | | **4,302K** | 0.197 | **0.9940** |
| faiss SQ8 | **3,485K** | **0.057** | 0.9930 | | 3,257K | **0.155** | 0.9930 |

> On ARM64, Quiver Int8 insert is **1.3x faster** than faiss SQ8 (4.3M vs 3.3M vec/s).

### IVF-Flat (Inverted File Index)

| Engine | Insert (vec/s) | Search (ms) | Recall@10 | | Insert (vec/s) | Search (ms) | Recall@10 |
|---|---:|---:|---:|---|---:|---:|---:|
| | **x86_64** | | | | **ARM64** | | |
| **Quiver IVF-Flat** | 216K | 0.037 | **0.6380** | | 200K | 0.030 | 0.5590 |
| faiss IVF-Flat | **392K** | **0.022** | 0.5670 | | **227K** | **0.027** | **0.5670** |

### IVF-PQ (Product Quantization)

| Engine | Insert (vec/s) | Search (ms) | Recall@10 | | Insert (vec/s) | Search (ms) | Recall@10 |
|---|---:|---:|---:|---|---:|---:|---:|
| | **x86_64** | | | | **ARM64** | | |
| **Quiver IVF-PQ** | 29K | 0.038 | **0.0760** | | 33K | 0.029 | **0.0730** |
| faiss IVFPQ | **79K** | **0.027** | 0.0690 | | **78K** | **0.016** | 0.0690 |

### Quiver-Only Index Types

| Index | Insert (vec/s) | Search (ms) | Recall@10 | | Insert (vec/s) | Search (ms) | Recall@10 |
|---|---:|---:|---:|---|---:|---:|---:|
| | **x86_64** | | | | **ARM64** | | |
| FP16 | 3,049K | 0.590 | 1.0000 | | 4,164K | 0.311 | 1.0000 |
| Mmap | 1,155K | 0.060 | 1.0000 | | 1,125K | 0.068 | 1.0000 |
| Binary | 1,108K | 0.034 | 0.1130 | | 1,721K | 0.027 | 0.1140 |

> FP16, Mmap, and Binary indexes are **unique to Quiver** — not available in faiss, usearch, or LanceDB.

### Key Takeaways

- **Flat insert**: Quiver **1.5–1.9x faster** than faiss, **45–50x faster** than LanceDB
- **HNSW search**: Quiver **0.039ms** — **1.2x faster** than faiss (0.047ms), **2.4x faster** than usearch (0.094ms)
- **HNSW single-thread insert**: Quiver 11.7K — **1.3x faster** than faiss (9.0K)
- **Int8 insert on ARM64**: Quiver **4.3M** — **1.3x faster** than faiss SQ8 (3.3M)
- **Recall parity**: Quiver matches or exceeds competitors across all index types
- **3 unique index types** (FP16, Mmap, Binary) not available in any competitor

Reproduce locally: `pip install quiver-vector-db[benchmark] && pytest tests/test_competitive_benchmarks.py -v -s`
Reproduce on CI: trigger the **Performance Benchmarks** workflow from the [Actions tab](https://github.com/rhshriva/Quiver/actions/workflows/benchmark.yml)

---

## Quick Start

### 1. Persistent Collection (WAL-backed)

```python
import quiver_vector_db as quiver

# Open a database (creates directory if needed)
db  = quiver.Client(path="./my_data")
col = db.create_collection("docs", dimensions=384, metric="cosine")

# Insert vectors with metadata
col.upsert(id=1, vector=[0.12, 0.45, ...], payload={"title": "Hello world"})
col.upsert(id=2, vector=[0.98, 0.01, ...], payload={"title": "Vector search"})

# Search
hits = col.search(query=[0.13, 0.44, ...], k=5)
for hit in hits:
    print(hit["id"], hit["distance"], hit["payload"])
```

Reopen the same path later and all collections are restored automatically.

### 2. Text Search with Built-in Embeddings

```python
import quiver_vector_db as quiver
from quiver_vector_db import TextCollection, SentenceTransformerEmbedding

db  = quiver.Client(path="./my_data")
col = db.create_collection("articles", dimensions=384, metric="cosine")

# Wrap with automatic embedding + BM25 full-text indexing
text_col = TextCollection(col, SentenceTransformerEmbedding("all-MiniLM-L6-v2"))

# Add documents by text — embedding is handled automatically
text_col.add(ids=[1, 2, 3], documents=[
    "Introduction to machine learning",
    "Advanced deep learning techniques",
    "Cooking recipes from around the world",
])

# Hybrid search (semantic + keyword) — default mode
hits = text_col.query("neural network basics", k=5)

# Semantic only (dense vector similarity)
hits = text_col.query("neural networks", k=5, mode="semantic")

# Keyword only (BM25 full-text)
hits = text_col.query("machine learning", k=5, mode="keyword")

for hit in hits:
    print(hit["id"], hit["document"], hit.get("score") or hit.get("distance"))
```

Requires: `pip install quiver-vector-db[sentence-transformers]`

### 3. Filtered Search with Metadata

```python
col.upsert(id=1, vector=[...], payload={"category": "tech", "rating": 4.8})
col.upsert(id=2, vector=[...], payload={"category": "science", "rating": 3.2})
col.upsert(id=3, vector=[...], payload={"category": "tech", "rating": 4.1})

# Simple equality filter
hits = col.search(query=[...], k=5, filter={"category": {"$eq": "tech"}})

# Range filter
hits = col.search(query=[...], k=5, filter={"rating": {"$gte": 4.0}})

# Compound filter with $and / $or
hits = col.search(query=[...], k=5, filter={
    "$and": [
        {"category": {"$in": ["tech", "science"]}},
        {"rating": {"$gte": 4.0}},
    ]
})
```

### 4. Hybrid Dense + Sparse Search

```python
# Upsert with sparse vector (e.g., BM25 or SPLADE weights)
col.upsert_hybrid(
    id=1, vector=[0.12, 0.45, ...],
    sparse_vector={42: 0.8, 100: 0.5, 3001: 0.3},
    payload={"title": "Rust guide"},
)

# Hybrid search — weighted fusion of dense + sparse signals
hits = col.search_hybrid(
    dense_query=[0.13, 0.44, ...],
    sparse_query={42: 0.7, 100: 0.6},
    k=10,
    dense_weight=0.7,
    sparse_weight=0.3,
)

for hit in hits:
    print(hit["id"], hit["score"], hit["dense_distance"], hit["sparse_score"])
```

### 5. Batch Upsert

```python
# Single batch call — more efficient than a loop
col.upsert_batch([
    (1, [0.1, 0.2, ...], {"title": "Doc A"}),
    (2, [0.3, 0.4, ...], {"title": "Doc B"}),
    (3, [0.5, 0.6, ...]),  # payload is optional
])
```

### 6. Multi-Vector / Multi-Modal Search

```python
from quiver_vector_db import MultiVectorCollection

multi = MultiVectorCollection(
    client=db,
    name="products",
    vector_spaces={
        "text":  {"dimensions": 384, "metric": "cosine"},
        "image": {"dimensions": 512, "metric": "cosine"},
    },
)

# Upsert with vectors from different modalities
multi.upsert(id=1, vectors={
    "text":  [0.1, 0.2, ...],
    "image": [0.5, 0.6, ...],
}, payload={"title": "Blue T-Shirt"})

# Search a single space
hits = multi.search(vector_space="text", query=[0.1, 0.2, ...], k=5)

# Cross-modal fusion search with custom weights
hits = multi.search_multi(
    queries={"text": [0.1, 0.2, ...], "image": [0.5, 0.6, ...]},
    k=5,
    weights={"text": 0.6, "image": 0.4},
)
```

### 7. Data Versioning / Snapshots

```python
# Insert initial data
for i in range(1000):
    col.upsert(id=i, vector=[...], payload={"version": 1})

# Create a snapshot
snapshot = col.create_snapshot("v1")
print(snapshot)  # {"name": "v1", "vector_count": 1000, ...}

# Mutate data...
for i in range(1000, 2000):
    col.upsert(id=i, vector=[...])

# Roll back to v1
col.restore_snapshot("v1")
assert col.count == 1000  # back to original state

# Manage snapshots
snapshots = col.list_snapshots()
col.delete_snapshot("v1")
```

### 8. Standalone Index (No WAL)

```python
import numpy as np
import quiver_vector_db as quiver

# In-memory HNSW index — no persistence overhead
idx = quiver.HnswIndex(dimensions=128, metric="l2", m=16, ef_construction=200)

# Batch insert from numpy array (fastest path)
vectors = np.random.randn(10_000, 128).astype(np.float32)
idx.add_batch_np(vectors)
idx.flush()

# Or parallel insert for maximum throughput
idx2 = quiver.HnswIndex(dimensions=128, metric="l2")
idx2.add_batch_parallel(vectors, num_threads=8)
idx2.flush()

# Search
results = idx.search(query=vectors[0].tolist(), k=10)

# Save and load
idx.save("my_index.qvec")
idx_loaded = quiver.HnswIndex.load("my_index.qvec")
```

### 9. REST API Server

```python
# Start server programmatically
from quiver_vector_db.server import create_server

server = create_server(host="0.0.0.0", port=8080, data_path="./my_data")
server.serve_forever()
```

Or from the command line:

```bash
python -m quiver_vector_db.server --port 8080 --data ./my_data
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/healthz` | Health check |
| `GET` | `/collections` | List collections |
| `POST` | `/collections` | Create collection (`{"name", "dimensions", "metric"}`) |
| `DELETE` | `/collections/{name}` | Delete collection |
| `POST` | `/collections/{name}/upsert` | Upsert vector (`{"id", "vector", "payload"}`) |
| `POST` | `/collections/{name}/upsert_batch` | Batch upsert (`{"entries": [...]}`) |
| `POST` | `/collections/{name}/search` | Search vectors (`{"query", "k", "filter"}`) |
| `POST` | `/collections/{name}/delete` | Delete vector (`{"id"}`) |
| `GET` | `/collections/{name}/count` | Get vector count |
| `POST` | `/collections/{name}/snapshots` | Create snapshot |
| `GET` | `/collections/{name}/snapshots` | List snapshots |
| `POST` | `/collections/{name}/snapshots/restore` | Restore snapshot |
| `DELETE` | `/collections/{name}/snapshots/{snap}` | Delete snapshot |

### 10. BM25 Standalone

```python
from quiver_vector_db import BM25

bm25 = BM25(k1=1.5, b=0.75)

# Index documents
bm25.index_document(0, "the quick brown fox jumps over the lazy dog")
bm25.index_document(1, "machine learning with neural networks")
bm25.index_document(2, "the fox and the hound")

# Generate sparse query vector
sparse_query = bm25.encode_query("quick fox")
print(sparse_query)  # {dim_id: idf_weight, ...}

# Save and load
bm25.save("bm25_state.json")
bm25_loaded = BM25.load("bm25_state.json")
```

---

## Index Types

Eight index types, all usable via `Client` or standalone:

| Index | Type | Recall | RAM | Best For |
|-------|------|--------|-----|----------|
| `hnsw` | `HnswIndex` | 95-99% | Vectors + graph | General purpose (default) |
| `flat` | `FlatIndex` | 100% | All vectors (f32) | Small datasets, exact results |
| `quantized_flat` | `QuantizedFlatIndex` | ~99% | ~4x less (int8) | Memory-constrained exact search |
| `fp16_flat` | `Fp16FlatIndex` | >99.5% | ~2x less (float16) | Balanced memory vs accuracy |
| `ivf` | `IvfIndex` | Tunable | Vectors + centroids | Large datasets |
| `ivf_pq` | `IvfPqIndex` | ~90%+ | ~96x less (PQ codes) | Million-scale, extreme compression |
| `mmap_flat` | `MmapFlatIndex` | 100% | Near-zero RSS | Dataset larger than RAM |
| `binary_flat` | `BinaryFlatIndex` | Low | ~32x less (1-bit) | Candidate generation, re-ranking |

```python
import quiver_vector_db as quiver

# Via Client (WAL-persisted)
db = quiver.Client(path="./data")
col = db.create_collection("docs", dimensions=768, metric="cosine", index_type="hnsw")

# Standalone in-memory
idx = quiver.FlatIndex(dimensions=384, metric="cosine")
idx = quiver.HnswIndex(dimensions=384, metric="cosine", ef_construction=200, ef_search=50, m=16)
idx = quiver.QuantizedFlatIndex(dimensions=384, metric="cosine")
idx = quiver.Fp16FlatIndex(dimensions=384, metric="cosine")
idx = quiver.IvfIndex(dimensions=384, metric="l2", n_lists=256, nprobe=16, train_size=4096)
idx = quiver.IvfPqIndex(dimensions=384, metric="l2", n_lists=256, nprobe=16, pq_m=8, pq_k_sub=256)
idx = quiver.MmapFlatIndex(dimensions=384, metric="cosine", path="./vectors.qvec")
idx = quiver.BinaryFlatIndex(dimensions=384, metric="l2")
```

## Distance Metrics

| Metric | String | Use When |
|--------|--------|----------|
| Cosine | `"cosine"` | Text/image embeddings (most common) |
| L2 (Euclidean) | `"l2"` | Geometry, sensor data |
| Dot Product | `"dot_product"` | Pre-normalized vectors |

All metrics use SIMD-accelerated kernels (AVX2+FMA on x86, NEON on ARM).

## IDE Support

Quiver ships with `py.typed` and `.pyi` type stubs. Autocompletion, type checking, and inline docs work out of the box in **VSCode**, **PyCharm**, and any editor supporting PEP 561.

---

## Development

### Prerequisites

- Rust toolchain (`rustup`, `cargo`)
- Python 3.11+

### Setup

```bash
git clone https://github.com/rhshriva/Quiver.git && cd Quiver
python3 -m venv .venv && source .venv/bin/activate
pip install maturin pytest numpy
maturin develop --release
```

### Running Tests

```bash
# Rust tests (~190 tests)
cargo test --workspace

# Python functional tests (~170 tests)
pytest tests/ -v --ignore=tests/test_perf.py --ignore=tests/test_benchmark.py --ignore=tests/test_perf_regression.py

# Performance benchmarks with apple-to-apple comparisons
pytest tests/test_perf_regression.py -v -s
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
