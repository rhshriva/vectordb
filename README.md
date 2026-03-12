# Quiver

An **embedded** vector database written in Rust with a Python SDK.

Runs fully in-process — no server, no network, no extra processes.

Two storage modes:

| Mode | How | When |
|------|-----|------|
| **Persistent** | `Client(path=...)` | WAL-backed; survives restarts |
| **In-memory** | Low-level index objects | Temporary; nothing written to disk unless you call `.save()` |

---

## Table of Contents

1. [Python SDK](#python-sdk)
   - [Installation](#installation)
   - [Persistent storage — Client](#persistent-storage--client)
   - [In-memory storage — low-level indexes](#in-memory-storage--low-level-indexes)
   - [Payload metadata](#payload-metadata)
   - [Filtered search](#filtered-search)
   - [Index types](#index-types)
   - [Collection management](#collection-management)
   - [Real-world example](#real-world-example)
2. [Rust library](#rust-library)
   - [Persistent storage](#persistent-storage-rust)
   - [In-memory storage](#in-memory-storage-rust)
   - [All index types via CollectionManager](#all-index-types-via-collectionmanager)
3. [Distance metrics](#distance-metrics)
4. [Index type reference](#index-type-reference)
5. [On-disk layout](#on-disk-layout)
6. [Build](#build)

---

## Python SDK

### Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin

# Build and install in development mode:
maturin develop --release -m crates/quiver-python/Cargo.toml

# Verify:
python -c "import quiver; print('Quiver ready')"
```

---

### Persistent storage — Client

`quiver.Client` opens a directory on disk. Every write is appended to a
write-ahead log (WAL) and survives process restarts.

```python
import quiver

# Open (or create) a database directory.
# All previously saved collections are loaded automatically.
db = quiver.Client(path="./my_quiver_data")

# Create a collection — name, vector dimensions, distance metric.
# Supported index types: "hnsw" (default, approximate) or "flat" (exact).
col = db.create_collection("sentences", dimensions=384, metric="cosine")

# Insert vectors. IDs are unsigned integers.
col.upsert(id=1, vector=[0.12, 0.45, ...])   # 384-dim list of floats
col.upsert(id=2, vector=[0.98, 0.01, ...])
col.upsert(id=3, vector=[0.55, 0.33, ...])

# Search for the k nearest neighbours.
hits = col.search(query=[0.13, 0.44, ...], k=5)

for hit in hits:
    print(f"id={hit['id']:4d}  distance={hit['distance']:.6f}")
```

Reopen the same path in a later session — all collections are restored:

```python
db  = quiver.Client(path="./my_quiver_data")
col = db.get_collection("sentences")
hits = col.search(query=[...], k=5)
```

---

### In-memory storage — low-level indexes

`FlatIndex` and `HnswIndex` live entirely in RAM. Nothing is written to disk
unless you call `.save()`. Ideal for ephemeral workloads, unit tests, or
pipelines that build a fresh index each run.

```python
import quiver

# ── FlatIndex — exact brute-force, 100% recall ────────────────────────────────
idx = quiver.FlatIndex(dimensions=3, metric="l2")
idx.add(id=1, vector=[1.0, 0.0, 0.0])
idx.add(id=2, vector=[0.0, 1.0, 0.0])
idx.add_batch([(3, [0.5, 0.5, 0.0]), (4, [0.1, 0.9, 0.0])])

results = idx.search(query=[0.9, 0.1, 0.0], k=2)
for r in results:
    print(f"id={r['id']}  dist={r['distance']:.4f}")

print(len(idx))        # 4
print(idx.dimensions)  # 3
print(idx.metric)      # "l2"

idx.delete(id=3)

# Optional: save to disk and reload later.
idx.save("./flat_index.bin")
loaded = quiver.FlatIndex.load("./flat_index.bin")

# ── HnswIndex — approximate nearest-neighbour, fast queries ──────────────────
hnsw = quiver.HnswIndex(
    dimensions=384,
    metric="cosine",
    ef_construction=200,   # beam width during graph build (higher = better recall, slower build)
    ef_search=50,          # beam width at query time (higher = better recall, slower query)
    m=12,                  # graph edges per node (higher = better recall, more RAM)
)

for vid, vec in my_vectors:
    hnsw.add(id=vid, vector=vec)

hnsw.flush()   # build the HNSW graph after bulk inserts

results = hnsw.search(query=query_vec, k=10)

# Optional: persist and reload.
hnsw.save("./hnsw_index.bin")
loaded = quiver.HnswIndex.load("./hnsw_index.bin")
```

---

### Payload metadata

Each vector can carry an arbitrary JSON-serialisable `dict` as metadata.
Payloads are stored alongside the vector and returned with every search result.

> **Note:** Payloads are only available through `Client` / `Collection`.
> The low-level `FlatIndex` and `HnswIndex` objects store vectors only.

```python
import quiver

db  = quiver.Client(path="./data")
col = db.create_collection("articles", dimensions=1536, metric="cosine")

col.upsert(id=1, vector=embedding_1, payload={
    "title":    "Introduction to Rust",
    "author":   "Jane Smith",
    "category": "programming",
    "score":    4.8,
})
col.upsert(id=2, vector=embedding_2, payload={
    "title":    "Vector Databases Explained",
    "author":   "Bob Jones",
    "category": "databases",
    "score":    4.5,
})

hits = col.search(query=query_embedding, k=3)
for hit in hits:
    print(f"{hit['id']} — {hit['payload']['title']} (dist={hit['distance']:.4f})")
```

---

### Filtered search

Pass a `filter` dict to `search()` to restrict results to vectors whose
payload matches the condition. Filters are applied before ranking.

**Supported operators:** `$eq`, `$ne`, `$in`, `$gt`, `$gte`, `$lt`, `$lte`,
`$and`, `$or`. Dot-notation accesses nested fields (`"meta.author"`).

```python
col = db.get_collection("articles")

# Equality
hits = col.search(query=query_embedding, k=5,
    filter={"category": {"$eq": "programming"}})

# Exclusion
hits = col.search(query=query_embedding, k=5,
    filter={"category": {"$ne": "spam"}})

# Set membership
hits = col.search(query=query_embedding, k=5,
    filter={"category": {"$in": ["programming", "databases"]}})

# Numeric range
hits = col.search(query=query_embedding, k=5,
    filter={"score": {"$gte": 4.5}})

# AND — all conditions must match
hits = col.search(query=query_embedding, k=5,
    filter={
        "$and": [
            {"category": {"$eq": "programming"}},
            {"score":    {"$gte": 4.0}},
        ]
    })

# OR — at least one condition must match
hits = col.search(query=query_embedding, k=5,
    filter={
        "$or": [
            {"category": {"$eq": "programming"}},
            {"category": {"$eq": "databases"}},
        ]
    })

# Nested field via dot notation
hits = col.search(query=query_embedding, k=5,
    filter={"meta.author": {"$eq": "Jane Smith"}})
```

---

### Index types

`Client.create_collection` supports two index types:

| `index_type` | Recall | Best for |
|---|---|---|
| `"hnsw"` (default) | 95–99% | General purpose, any dataset size |
| `"flat"` | 100% exact | Small datasets or when exact recall is required |

HNSW uses sensible defaults (`ef_construction=200`, `ef_search=50`, `m=12`).
To tune HNSW parameters directly, use the in-memory `HnswIndex` and manage
persistence with `.save()` / `.load()`.

```python
db = quiver.Client(path="./data")

# HNSW — approximate, fast (default)
col = db.create_collection("ann", dimensions=768, metric="cosine", index_type="hnsw")

# Flat — exact brute-force
col = db.create_collection("exact", dimensions=768, metric="cosine", index_type="flat")
```

---

### Collection management

```python
db = quiver.Client(path="./data")

# Create only if it doesn't exist yet (idempotent, uses HNSW).
col = db.get_or_create_collection("docs", dimensions=768, metric="cosine")

# Get a handle to an existing collection.
col = db.get_collection("docs")

# List all collection names.
print(db.list_collections())   # ['docs', 'articles', ...]

# Count vectors.
print(col.count)               # 1024

# Upsert = insert or overwrite by ID.
col.upsert(id=42, vector=new_vec, payload={"updated": True})

# Delete a single vector by ID.
col.delete(id=42)

# Drop an entire collection and remove all data from disk.
db.delete_collection("docs")
```

---

### Real-world example

Semantic search over documents using `sentence-transformers`:

```python
import quiver
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim embeddings

docs = [
    {"id": 1, "text": "Rust is a systems programming language focused on safety."},
    {"id": 2, "text": "Python is great for data science and machine learning."},
    {"id": 3, "text": "Vector databases store embeddings for similarity search."},
    {"id": 4, "text": "HNSW is a graph-based approximate nearest neighbour algorithm."},
]

db  = quiver.Client(path="./semantic_data")
col = db.get_or_create_collection("docs", dimensions=384, metric="cosine")

for doc in docs:
    vec = model.encode(doc["text"]).tolist()
    col.upsert(id=doc["id"], vector=vec, payload={"text": doc["text"]})

# Similarity search
query_vec = model.encode("how do embeddings work?").tolist()
hits = col.search(query=query_vec, k=3)
for hit in hits:
    print(f"  [{hit['distance']:.4f}] {hit['payload']['text']}")

# Filtered search
hits = col.search(
    query=query_vec,
    k=3,
    filter={"$or": [
        {"text": {"$eq": "Rust is a systems programming language focused on safety."}},
        {"text": {"$eq": "Python is great for data science and machine learning."}},
    ]}
)
```

---

## Rust library

`quiver-core` is a pure Rust library — no async runtime, no network dependencies.

```toml
[dependencies]
quiver-core = { path = "./crates/quiver-core" }
serde_json   = "1"
```

### Persistent storage (Rust)

```rust
use quiver_core::db::Quiver;
use quiver_core::distance::Metric;
use quiver_core::payload::FilterCondition;
use serde_json::json;

let mut db = Quiver::open("./my_quiver_data")?;

db.create_collection("sentences", 384, Metric::Cosine)?;
db.upsert("sentences", 1, &[0.12, 0.45, /* ... */], None)?;
db.upsert("sentences", 2, &[0.98, 0.01, /* ... */], None)?;

let hits = db.search("sentences", &[0.13, 0.44, /* ... */], 5)?;
for hit in &hits {
    println!("id={:4}  distance={:.6}", hit.id, hit.distance);
}

// Filtered search
let filter: FilterCondition = serde_json::from_value(json!({
    "$and": [
        { "category": { "$eq": "programming" } },
        { "score":    { "$gte": 4.0 } }
    ]
}))?;
let hits = db.search_filtered("articles", &query, 5, &filter)?;

// Collection management
db.get_or_create_collection("docs", 768, Metric::Cosine)?;
let names = db.list_collections();
let n     = db.count("docs")?;
db.delete("docs", 42)?;
db.delete_collection("docs")?;
```

### In-memory storage (Rust)

Use the index types directly. No WAL, no directory — nothing hits disk
unless you call `.save()`.

```rust
use quiver_core::index::flat::FlatIndex;
use quiver_core::index::hnsw::{HnswConfig, HnswIndex};
use quiver_core::index::VectorIndex;
use quiver_core::distance::Metric;

// ── FlatIndex (exact, 100% recall) ────────────────────────────────────────────
let mut flat = FlatIndex::new(3, Metric::L2);
flat.add(1, &[1.0, 0.0, 0.0])?;
flat.add(2, &[0.0, 1.0, 0.0])?;
flat.add_batch(&[(3, vec![0.5, 0.5, 0.0])])?;

let results = flat.search(&[0.9, 0.1, 0.0], 2)?;
flat.save("./flat.bin")?;
let loaded = FlatIndex::load("./flat.bin")?;

// ── HnswIndex (approximate, fast) ─────────────────────────────────────────────
let cfg = HnswConfig { ef_construction: 200, ef_search: 50, m: 12 };
let mut hnsw = HnswIndex::new(384, Metric::Cosine, cfg);

for (id, vec) in &my_vectors {
    hnsw.add(*id, vec)?;
}
hnsw.flush();             // build graph after bulk insert
hnsw.set_ef_search(100);  // tune recall at query time without rebuilding

let results = hnsw.search(&query, 10)?;
hnsw.save("./hnsw.bin")?;
let loaded = HnswIndex::load("./hnsw.bin")?;
```

### All index types via CollectionManager

`CollectionManager` provides direct access to all six index types. All are
persistent (WAL-backed). The Python SDK currently exposes `"flat"` and `"hnsw"`
only; the remaining types are available to Rust users.

```rust
use quiver_core::manager::CollectionManager;
use quiver_core::collection::{CollectionMeta, IndexType};
use quiver_core::distance::Metric;
use quiver_core::index::hnsw::HnswConfig;
use quiver_core::index::ivf::IvfConfig;

let mut mgr = CollectionManager::open("./data")?;

// Flat — exact, O(N·D) per query
mgr.create_collection(CollectionMeta {
    name: "exact".into(), dimensions: 768, metric: Metric::Cosine,
    index_type: IndexType::Flat,
    hnsw_config: None, ivf_config: None, faiss_factory: None,
    wal_compact_threshold: 50_000,
    auto_promote_threshold: None, promotion_hnsw_config: None,
    embedding_model: None,
})?;

// HNSW — ~95–99% recall, O(log N · ef) per query
mgr.create_collection(CollectionMeta {
    name: "ann".into(), dimensions: 768, metric: Metric::Cosine,
    index_type: IndexType::Hnsw,
    hnsw_config: Some(HnswConfig { ef_construction: 200, ef_search: 50, m: 12 }),
    ivf_config: None, faiss_factory: None,
    wal_compact_threshold: 50_000,
    auto_promote_threshold: None, promotion_hnsw_config: None,
    embedding_model: None,
})?;

// QuantizedFlat — int8 brute-force, ~4× less RAM, ~99% recall vs Flat
mgr.create_collection(CollectionMeta {
    name: "quantized".into(), dimensions: 1536, metric: Metric::Cosine,
    index_type: IndexType::QuantizedFlat,
    hnsw_config: None, ivf_config: None, faiss_factory: None,
    wal_compact_threshold: 50_000,
    auto_promote_threshold: None, promotion_hnsw_config: None,
    embedding_model: None,
})?;

// IVF — cluster-based ANN; auto-trains after train_size inserts
// Rule of thumb: n_lists = sqrt(N), nprobe = sqrt(n_lists)
mgr.create_collection(CollectionMeta {
    name: "ivf".into(), dimensions: 768, metric: Metric::L2,
    index_type: IndexType::Ivf,
    ivf_config: Some(IvfConfig { n_lists: 256, nprobe: 16, train_size: 4096, max_iter: 25 }),
    hnsw_config: None, faiss_factory: None,
    wal_compact_threshold: 50_000,
    auto_promote_threshold: None, promotion_hnsw_config: None,
    embedding_model: None,
})?;

// MmapFlat — disk-mapped brute-force; near-zero RAM; OS pages in as needed
mgr.create_collection(CollectionMeta {
    name: "mmap".into(), dimensions: 768, metric: Metric::Cosine,
    index_type: IndexType::MmapFlat,
    hnsw_config: None, ivf_config: None, faiss_factory: None,
    wal_compact_threshold: 50_000,
    auto_promote_threshold: None, promotion_hnsw_config: None,
    embedding_model: None,
})?;

// FAISS — requires `--features faiss` and libfaiss_c
// Factory strings: "Flat", "IVF1024,Flat", "IVF256,PQ64", "HNSW32"
#[cfg(feature = "faiss")]
mgr.create_collection(CollectionMeta {
    name: "faiss_ivf".into(), dimensions: 768, metric: Metric::L2,
    index_type: IndexType::Faiss,
    faiss_factory: Some("IVF1024,Flat".into()),
    hnsw_config: None, ivf_config: None,
    wal_compact_threshold: 50_000,
    auto_promote_threshold: None, promotion_hnsw_config: None,
    embedding_model: None,
})?;

// Auto-promote: start exact, automatically switch to HNSW at threshold
mgr.create_collection(CollectionMeta {
    name: "auto".into(), dimensions: 768, metric: Metric::Cosine,
    index_type: IndexType::Flat,
    auto_promote_threshold: Some(10_000),
    promotion_hnsw_config:  Some(HnswConfig::default()),
    hnsw_config: None, ivf_config: None, faiss_factory: None,
    wal_compact_threshold: 50_000,
    embedding_model: None,
})?;
```

---

## Distance metrics

| Metric | Rust | Python | Use when |
|--------|------|--------|----------|
| Cosine | `Metric::Cosine` | `"cosine"` | Text/image embeddings (most common) |
| Euclidean (L2) | `Metric::L2` | `"l2"` | Geometry, sensor data, unnormalised vectors |
| Dot product | `Metric::DotProduct` | `"dot_product"` | Pre-normalised vectors, recommendation models |

> Most embedding models output cosine-space vectors. Use `"cosine"` unless your
> model documentation says otherwise.

---

## Index type reference

### Python SDK

| API | Index | Storage | Recall | Notes |
|-----|-------|---------|--------|-------|
| `create_collection(..., index_type="hnsw")` | HNSW | Persistent (WAL) | 95–99% | Default; fast approximate search |
| `create_collection(..., index_type="flat")` | Flat | Persistent (WAL) | 100% | Exact; slower on large datasets |
| `FlatIndex(...)` | Flat | **In-memory** | 100% | No WAL; optional `.save()` / `.load()` |
| `HnswIndex(...)` | HNSW | **In-memory** | 95–99% | Configurable; optional `.save()` / `.load()` |

### Rust (all index types)

| Index | Recall | Query complexity | RAM | Best for |
|-------|--------|-----------------|-----|----------|
| Flat | 100% | O(N·D) | All vectors (f32) | <100K vectors, exact required |
| HNSW | 95–99% | O(log N · ef) | All vectors + graph | General purpose |
| QuantizedFlat | ~99% | O(N·D) | **~4× less** (int8) | Memory-constrained exact search |
| IVF | Tunable | O(n\_lists + nprobe·N/n\_lists) | All vectors | Large datasets with training phase |
| MmapFlat | 100% | O(N·D) disk-paged | Staging only | Dataset larger than RAM |
| FAISS | Varies | Varies | Varies | GPU, PQ compression, custom factory |

### HNSW parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `ef_construction` | 200 | Build beam width — higher = better recall, slower build |
| `ef_search` | 50 | Query beam width — tunable at runtime without rebuild |
| `m` | 12 | Edges per node per layer — higher = better recall, more RAM |

### IVF parameters (Rust only)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_lists` | 256 | k-means clusters — rule of thumb: `sqrt(N)` |
| `nprobe` | 16 | Clusters scanned per query — higher = better recall, slower |
| `train_size` | 4096 | Vectors buffered before training triggers automatically |
| `max_iter` | 25 | Lloyd's k-means iterations |

---

## On-disk layout

```
my_quiver_data/
├── sentences/
│   ├── meta.json   ← collection config (dimensions, metric, index type)
│   └── wal.log     ← write-ahead log; replayed on open to rebuild the index
├── mmap_col/
│   ├── meta.json
│   ├── wal.log
│   └── vectors.mmap   ← memory-mapped vector file (MmapFlat only)
└── ...
```

- **`meta.json`** — human-readable collection metadata
- **`wal.log`** — binary append log (bincode frames); the authoritative record of all writes
- **`vectors.mmap`** — flat binary vector file for `MmapFlatIndex`

In-memory indexes (`FlatIndex`, `HnswIndex` used directly) write nothing
unless you explicitly call `.save(path)`.

---

## Build

```bash
# Build and test quiver-core
./dev_build.sh

# Build Python wheel and smoke-test
python3 -m venv .venv && source .venv/bin/activate
pip install maturin
./dev_build.sh --python

# Build with FAISS support (requires libfaiss_c)
./dev_build.sh --faiss --python
```

---

## Crates

| Crate | Purpose |
|-------|---------|
| `quiver-core` | Embedded vector database engine — indexes, WAL, filtering, distance metrics |
| `quiver-python` | Python bindings via PyO3/maturin |

## License

MIT
