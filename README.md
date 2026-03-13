# Quiver

A fast, embedded vector database written in Rust with a Python SDK.
No server, no network — runs fully in-process with SIMD-accelerated search.

## Features

- **7 index types** — HNSW, Flat, Int8/FP16 quantized, IVF, IVF-PQ, memory-mapped
- **3 distance metrics** — Cosine, L2, Dot Product with AVX2/NEON SIMD
- **Hybrid search** — combine dense vectors with sparse keyword signals (BM25/SPLADE)
- **Payload filtering** — JSON metadata with operators: `$eq`, `$ne`, `$in`, `$gt`, `$gte`, `$lt`, `$lte`, `$and`, `$or`
- **WAL persistence** — crash-safe writes, automatic compaction
- **Type stubs included** — full IDE autocompletion in VSCode, PyCharm, etc.
- **Zero dependencies** — single pip install, no servers or runtimes

## Installation

```bash
pip install quiver-vector-db
```

Prebuilt wheels for macOS (Apple Silicon / Intel) and Linux (x86_64 / ARM64). Python 3.8+.

**Build from source** (requires Rust toolchain):

```bash
pip install maturin
maturin develop --release -m crates/quiver-python/Cargo.toml
```

## Quick start

```python
import quiver_vector_db as quiver

db  = quiver.Client(path="./my_data")
col = db.create_collection("docs", dimensions=384, metric="cosine")

col.upsert(id=1, vector=[0.12, 0.45, ...], payload={"title": "Hello world"})
col.upsert(id=2, vector=[0.98, 0.01, ...], payload={"title": "Vector search"})

hits = col.search(query=[0.13, 0.44, ...], k=5)
for hit in hits:
    print(hit["id"], hit["distance"], hit["payload"])
```

Collections are persisted via WAL — reopen the same path and everything is restored.

---

## API Reference

### `Client(path="./data")`

Persistent vector database client. Opens a directory on disk; all writes are WAL-backed.

| Method | Description |
|--------|-------------|
| `create_collection(name, dimensions, metric="cosine", index_type="hnsw")` | Create a new collection |
| `get_collection(name)` | Get an existing collection. Raises `KeyError` if not found |
| `get_or_create_collection(name, dimensions, metric="cosine")` | Get or create (defaults to HNSW) |
| `delete_collection(name)` | Delete collection and all its data from disk |
| `list_collections()` | Returns list of collection name strings |

### `Collection`

Returned by `Client.create_collection()` or `Client.get_collection()`.

| Method | Description |
|--------|-------------|
| `upsert(id, vector, payload=None)` | Insert or update a vector with optional metadata dict |
| `search(query, k, filter=None)` | Search k nearest vectors. Returns `[{"id", "distance", "payload"}]` |
| `upsert_hybrid(id, vector, sparse_vector=None, payload=None)` | Upsert with optional sparse vector (`{dim_index: weight}`) |
| `search_hybrid(dense_query, sparse_query, k, dense_weight=0.7, sparse_weight=0.3, filter=None)` | Hybrid search. Returns `[{"id", "score", "dense_distance", "sparse_score", "payload"}]` |
| `delete(id)` | Delete a vector by ID. Returns `True` if found |
| `count` | Property: number of dense vectors |
| `sparse_count` | Property: number of sparse vectors |
| `name` | Property: collection name |

### Standalone Index Classes

All standalone indexes share a common interface:

| Method | Description |
|--------|-------------|
| `add(id, vector)` | Add a single vector |
| `add_batch(entries)` | Add multiple vectors. `entries`: list of `(id, vector)` tuples |
| `search(query, k)` | Returns `[{"id", "distance"}]` |
| `delete(id)` | Returns `True` if removed |
| `save(path)` / `Index.load(path)` | Persist and restore |
| `dimensions` | Property |
| `metric` | Property |
| `len(idx)` | Number of vectors |

#### `FlatIndex(dimensions, metric="l2")`

Exact brute-force index. 100% recall, O(N*D) per query.

#### `HnswIndex(dimensions, metric="l2", ef_construction=200, ef_search=50, m=12)`

Graph-based approximate nearest-neighbour index. 95-99% recall.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `ef_construction` | 200 | Build beam width. Higher = better recall, slower build |
| `ef_search` | 50 | Query beam width. Tunable at runtime |
| `m` | 12 | Graph edges per node. Higher = better recall, more RAM |

Additional method: `flush()` — build the HNSW graph after bulk inserts.

#### `QuantizedFlatIndex(dimensions, metric="l2")`

Int8 quantized brute-force. ~4x less RAM, ~99% recall.

#### `Fp16FlatIndex(dimensions, metric="l2")`

Float16 quantized brute-force. 2x less RAM, >99.5% recall.

#### `IvfIndex(dimensions, metric="l2", n_lists=256, nprobe=16, train_size=4096)`

Cluster-based ANN using k-means. Auto-trains after `train_size` inserts.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_lists` | 256 | Number of clusters (rule of thumb: `sqrt(N)`) |
| `nprobe` | 16 | Clusters scanned per query |
| `train_size` | 4096 | Vectors buffered before auto-training |

Additional method: `flush()` — trigger training.

#### `IvfPqIndex(dimensions, metric="l2", n_lists=256, nprobe=16, train_size=4096, pq_m=8, pq_k_sub=256)`

IVF with product quantization. ~96x memory reduction for 1536-dim vectors.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `pq_m` | 8 | Sub-quantizers (must divide `dimensions`). Memory per vector = `pq_m` bytes |
| `pq_k_sub` | 256 | Centroids per sub-quantizer |

Additional method: `flush()` — trigger training.

#### `MmapFlatIndex(dimensions, metric="l2", path="./mmap_index.qvec")`

Memory-mapped brute-force. Near-zero RAM; pages loaded on demand by the OS. Additional method: `flush()`.

---

## Index types

Seven index types, all usable via `Client` or standalone:

| Index | Recall | RAM | Best for |
|-------|--------|-----|----------|
| `hnsw` | 95-99% | Vectors + graph | General purpose (default) |
| `flat` | 100% | All vectors (f32) | Small datasets, exact required |
| `quantized_flat` | ~99% | ~4x less (int8) | Memory-constrained exact search |
| `fp16_flat` | >99.5% | ~2x less (float16) | Balanced memory vs accuracy |
| `ivf` | Tunable | Vectors + centroids | Large datasets |
| `ivf_pq` | ~90%+ | ~96x less (PQ codes) | Million-scale, extreme compression |
| `mmap_flat` | 100% | Near-zero RSS | Dataset larger than RAM |

```python
import quiver_vector_db as quiver

# Via Client (WAL-persisted)
db = quiver.Client(path="./data")
col = db.create_collection("name", dimensions=768, metric="cosine", index_type="hnsw")

# Standalone in-memory
idx = quiver.FlatIndex(dimensions=384, metric="cosine")
idx = quiver.HnswIndex(dimensions=384, metric="cosine", ef_construction=200, ef_search=50, m=12)
idx = quiver.QuantizedFlatIndex(dimensions=384, metric="cosine")
idx = quiver.Fp16FlatIndex(dimensions=384, metric="cosine")
idx = quiver.IvfIndex(dimensions=384, metric="l2", n_lists=256, nprobe=16, train_size=4096)
idx = quiver.IvfPqIndex(dimensions=384, metric="l2", n_lists=256, nprobe=16, train_size=4096, pq_m=8, pq_k_sub=256)
idx = quiver.MmapFlatIndex(dimensions=384, metric="cosine", path="./vectors.qvec")
```

## Distance metrics

| Metric | String | Use when |
|--------|--------|----------|
| Cosine | `"cosine"` | Text/image embeddings (most common) |
| L2 | `"l2"` | Geometry, sensor data |
| Dot product | `"dot_product"` | Pre-normalised vectors |

All metrics use SIMD-accelerated kernels (AVX2+FMA on x86, NEON on ARM).

## Payload & filtered search

```python
col.upsert(id=1, vector=[...], payload={"category": "tech", "score": 4.8})

# Filter operators: $eq, $ne, $in, $gt, $gte, $lt, $lte, $and, $or
hits = col.search(query=[...], k=5, filter={"category": {"$eq": "tech"}})
hits = col.search(query=[...], k=5, filter={"score": {"$gte": 4.0}})
hits = col.search(query=[...], k=5, filter={
    "$and": [
        {"category": {"$in": ["tech", "science"]}},
        {"score": {"$gte": 4.0}},
    ]
})
```

## Hybrid dense+sparse search

Combine dense vector similarity with sparse keyword signals (e.g. BM25/SPLADE weights):

```python
col.upsert_hybrid(
    id=1, vector=[...],
    sparse_vector={42: 0.8, 100: 0.5, 3001: 0.3},
    payload={"title": "Rust guide"},
)

hits = col.search_hybrid(
    dense_query=[...],
    sparse_query={42: 0.7, 100: 0.6},
    k=10,
    dense_weight=0.7,
    sparse_weight=0.3,
    filter={"category": {"$eq": "tech"}},  # optional
)

for hit in hits:
    print(hit["id"], hit["score"], hit["dense_distance"], hit["sparse_score"])
```

Regular `upsert()` and `upsert_hybrid()` can be mixed freely in the same collection.

## Persistence

All data written through `Client` is WAL-backed:

- **Crash-safe** — length-prefixed binary frames; partial writes are safely skipped on recovery
- **Automatic compaction** — after 50K WAL entries, live state is rewritten and tombstones discarded
- **Graph snapshots** — HNSW graph structure is saved to skip O(N log N) rebuild on reload
- **Reopen anytime** — point `Client` at the same directory and all collections are restored

Standalone indexes can be saved/loaded manually with `save(path)` and `Index.load(path)`.

## IDE support

Quiver ships with `py.typed` and `.pyi` type stubs. Autocompletion, type checking, and inline docs work out of the box in VSCode, PyCharm, and any editor that supports PEP 561.

## Development

### Prerequisites

- Rust toolchain (`rustup`, `cargo`)
- Python 3.8+

### Environment setup

```bash
git clone https://github.com/rhshriva/Quiver.git && cd Quiver
python3 -m venv .venv && source .venv/bin/activate
pip install maturin pytest numpy
maturin develop --release -m crates/quiver-python/Cargo.toml
```

### Build

```bash
./dev_build.sh               # build + test Rust core
./dev_build.sh --python       # also build Python wheel
./dev_build.sh --faiss --python  # with FAISS support
```

### Running tests

```bash
# Rust unit tests (175 tests)
cargo test --workspace

# Python functional tests
pytest tests/ -v --ignore=tests/test_perf.py

# Python performance benchmarks (insert throughput, search latency, recall)
pytest tests/test_perf.py -v -s

# All Python tests
pytest tests/ -v -s
```

### Publishing

```bash
./publish.sh              # build cross-platform wheels + upload to PyPI
./publish.sh --test       # upload to TestPyPI instead
./publish.sh --build-only # build wheels without uploading
```

## License

MIT
