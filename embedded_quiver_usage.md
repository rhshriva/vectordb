# Quiver — Embedded Usage Guide

Quiver runs **fully in-process** — no server to start, no network calls, no
extra processes. Open a path and go.

---

## Table of Contents

1. [Rust](#rust)
   - [Installation](#installation-rust)
   - [Basic usage](#basic-usage-rust)
   - [Payload metadata](#payload-metadata-rust)
   - [Filtered search](#filtered-search-rust)
   - [All index types](#all-index-types-rust)
   - [Collection management](#collection-management-rust)
   - [Low-level index API](#low-level-index-api-rust)
2. [Python](#python)
   - [Installation](#installation-python)
   - [Basic usage](#basic-usage-python)
   - [Payload metadata](#payload-metadata-python)
   - [Filtered search](#filtered-search-python)
   - [All index types](#all-index-types-python)
   - [Collection management](#collection-management-python)
   - [Low-level index API](#low-level-index-api-python)
   - [Real-world example](#real-world-example-python)
3. [Distance metrics](#distance-metrics)
4. [Index type reference](#index-type-reference)
5. [Data directory layout](#data-directory-layout)

---

## Rust

### Installation (Rust)

Add `quiver-core` to your `Cargo.toml`:

```toml
[dependencies]
quiver-core = { path = "./crates/quiver-core" }
serde_json   = "1"    # for payload values
```

### Basic usage (Rust)

```rust
use quiver_core::db::Quiver;
use quiver_core::distance::Metric;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open (or create) a database. All existing collections are loaded automatically.
    let mut db = Quiver::open("./my_quiver_data")?;

    // Create a collection: name, vector dimensions, distance metric.
    db.create_collection("sentences", 384, Metric::Cosine)?;

    // Insert vectors with u64 IDs.
    db.upsert("sentences", 1, &[0.12, 0.45, /* ...384 dims */], None)?;
    db.upsert("sentences", 2, &[0.98, 0.01, /* ... */], None)?;
    db.upsert("sentences", 3, &[0.55, 0.33, /* ... */], None)?;

    // Search for the 5 nearest neighbours.
    let query = vec![0.13, 0.44, /* ... */];
    let hits = db.search("sentences", &query, 5)?;

    for hit in &hits {
        println!("id={:4}  distance={:.6}", hit.id, hit.distance);
    }
    Ok(())
}
```

### Payload metadata (Rust)

Each vector carries an arbitrary JSON payload returned with search results.

```rust
use quiver_core::db::Quiver;
use quiver_core::distance::Metric;
use serde_json::json;

let mut db = Quiver::open("./data")?;
db.create_collection("articles", 1536, Metric::Cosine)?;

db.upsert("articles", 1, &embedding_1, Some(json!({
    "title":    "Introduction to Rust",
    "author":   "Jane Smith",
    "category": "programming",
    "score":    4.8
})))?;

db.upsert("articles", 2, &embedding_2, Some(json!({
    "title":    "Vector Databases Explained",
    "author":   "Bob Jones",
    "category": "databases",
    "score":    4.5
})))?;

let hits = db.search("articles", &query_embedding, 3)?;
for hit in &hits {
    if let Some(p) = &hit.payload {
        println!("{} — {} (dist={:.4})", hit.id, p["title"], hit.distance);
    }
}
```

### Filtered search (Rust)

Filter on any payload field. Supported operators: `$eq`, `$ne`, `$in`,
`$gt`, `$gte`, `$lt`, `$lte`. Dot-notation reaches nested fields
(`"meta.author"`).

```rust
use quiver_core::db::Quiver;
use quiver_core::distance::Metric;
use quiver_core::payload::FilterCondition;
use serde_json::json;

let mut db = Quiver::open("./data")?;
// ... upsert vectors with category / score payloads ...

// Equality
let filter: FilterCondition = serde_json::from_value(json!({
    "category": { "$eq": "programming" }
}))?;
let hits = db.search_filtered("articles", &query, 5, &filter)?;

// Range: score >= 4.5
let filter: FilterCondition = serde_json::from_value(json!({
    "score": { "$gte": 4.5 }
}))?;

// $ne — exclude a value
let filter: FilterCondition = serde_json::from_value(json!({
    "category": { "$ne": "spam" }
}))?;

// $in — match any of a set
let filter: FilterCondition = serde_json::from_value(json!({
    "category": { "$in": ["programming", "databases"] }
}))?;

// $and — all conditions must hold
let filter: FilterCondition = serde_json::from_value(json!({
    "$and": [
        { "category": { "$eq": "programming" } },
        { "score":    { "$gte": 4.0 } }
    ]
}))?;

// $or — at least one condition must hold
let filter: FilterCondition = serde_json::from_value(json!({
    "$or": [
        { "category": { "$eq": "programming" } },
        { "category": { "$eq": "databases"   } }
    ]
}))?;

// Nested field via dot notation
let filter: FilterCondition = serde_json::from_value(json!({
    "meta.author": { "$eq": "Jane Smith" }
}))?;
```

### All index types (Rust)

The `Quiver` high-level API defaults to HNSW. To choose a different index type
use `CollectionManager` and `CollectionMeta` directly.

```rust
use quiver_core::manager::CollectionManager;
use quiver_core::collection::{CollectionMeta, IndexType};
use quiver_core::distance::Metric;
use quiver_core::index::hnsw::HnswConfig;
use quiver_core::index::ivf::IvfConfig;

let mut mgr = CollectionManager::open("./data")?;

// ── 1. Flat — exact brute-force, 100% recall ──────────────────────────────────
//    Best for: <100K vectors or when exact recall is required.
//    Query: O(N·D)
mgr.create_collection(CollectionMeta {
    name:        "exact".to_string(),
    dimensions:  768,
    metric:      Metric::Cosine,
    index_type:  IndexType::Flat,
    hnsw_config: None,
    ivf_config:  None,
    faiss_factory: None,
    wal_compact_threshold: 50_000,
    auto_promote_threshold: None,
    promotion_hnsw_config:  None,
    embedding_model: None,
})?;

// ── 2. HNSW — approximate graph index, ~95–99% recall ────────────────────────
//    Best for: general-purpose, 100K–100M vectors.
//    Query: O(log N · ef_search)
mgr.create_collection(CollectionMeta {
    name:       "ann".to_string(),
    dimensions: 768,
    metric:     Metric::Cosine,
    index_type: IndexType::Hnsw,
    hnsw_config: Some(HnswConfig {
        ef_construction: 200,  // higher → better graph, slower build
        ef_search:       50,   // higher → better recall, slower query (tunable at runtime)
        m:               12,   // edges per node per layer
    }),
    ivf_config:   None,
    faiss_factory: None,
    wal_compact_threshold:  50_000,
    auto_promote_threshold: None,
    promotion_hnsw_config:  None,
    embedding_model: None,
})?;

// ── 3. QuantizedFlat — int8 brute-force, ~4× lower memory ────────────────────
//    Best for: memory-constrained machines needing exact search.
//    Recall: ~99% vs FlatIndex. Memory: 4× less than f32.
mgr.create_collection(CollectionMeta {
    name:        "quantized".to_string(),
    dimensions:  1536,
    metric:      Metric::Cosine,
    index_type:  IndexType::QuantizedFlat,
    hnsw_config: None,
    ivf_config:  None,
    faiss_factory: None,
    wal_compact_threshold: 50_000,
    auto_promote_threshold: None,
    promotion_hnsw_config:  None,
    embedding_model: None,
})?;

// ── 4. IVF — cluster-based ANN, sub-linear search ────────────────────────────
//    Best for: 100K–100M vectors; recall tunable via nprobe.
//    Requires a training phase (auto-triggered after train_size inserts).
//    Rule of thumb: n_lists = sqrt(N), nprobe = sqrt(n_lists)
mgr.create_collection(CollectionMeta {
    name:        "ivf".to_string(),
    dimensions:  768,
    metric:      Metric::L2,
    index_type:  IndexType::Ivf,
    hnsw_config: None,
    ivf_config:  Some(IvfConfig {
        n_lists:    256,   // number of k-means clusters
        nprobe:     16,    // clusters to scan at query time
        train_size: 4096,  // vectors to collect before training
        max_iter:   25,    // Lloyd's k-means iterations
    }),
    faiss_factory: None,
    wal_compact_threshold: 50_000,
    auto_promote_threshold: None,
    promotion_hnsw_config:  None,
    embedding_model: None,
})?;

// ── 5. MmapFlat — disk-mapped brute-force, near-zero RAM ─────────────────────
//    Best for: datasets larger than available RAM.
//    The OS pages in only the 4 KB file blocks being scanned.
//    Query: O(N·D) with disk paging.
mgr.create_collection(CollectionMeta {
    name:        "mmap".to_string(),
    dimensions:  768,
    metric:      Metric::Cosine,
    index_type:  IndexType::MmapFlat,
    hnsw_config: None,
    ivf_config:  None,
    faiss_factory: None,
    wal_compact_threshold: 50_000,
    auto_promote_threshold: None,
    promotion_hnsw_config:  None,
    embedding_model: None,
})?;

// ── 6. FAISS — Facebook AI Similarity Search (feature-gated) ─────────────────
//    Requires: compile with `--features faiss` and libfaiss_c installed.
//    Factory strings: "Flat", "IVF1024,Flat", "IVF256,PQ64", "HNSW32", "IVF4096,SQ8"
//    See: https://github.com/facebookresearch/faiss/wiki/The-index-factory
#[cfg(feature = "faiss")]
mgr.create_collection(CollectionMeta {
    name:          "faiss_ivf".to_string(),
    dimensions:    768,
    metric:        Metric::L2,
    index_type:    IndexType::Faiss,
    hnsw_config:   None,
    ivf_config:    None,
    faiss_factory: Some("IVF1024,Flat".to_string()),
    wal_compact_threshold: 50_000,
    auto_promote_threshold: None,
    promotion_hnsw_config:  None,
    embedding_model: None,
})?;
```

#### Auto-promote Flat → HNSW

Start with exact search and automatically switch to HNSW once the collection
grows large enough:

```rust
mgr.create_collection(CollectionMeta {
    name:        "auto".to_string(),
    dimensions:  768,
    metric:      Metric::Cosine,
    index_type:  IndexType::Flat,       // starts exact
    hnsw_config: None,
    ivf_config:  None,
    faiss_factory: None,
    wal_compact_threshold:  50_000,
    auto_promote_threshold: Some(10_000), // promote when 10K vectors inserted
    promotion_hnsw_config:  Some(HnswConfig::default()),
    embedding_model: None,
})?;
```

### Collection management (Rust)

```rust
use quiver_core::db::Quiver;
use quiver_core::distance::Metric;

let mut db = Quiver::open("./data")?;

// Create only if it doesn't exist (idempotent).
db.get_or_create_collection("docs", 768, Metric::Cosine)?;

// List all collection names.
let names = db.list_collections();
println!("{:?}", names);   // ["docs", "articles", ...]

// Count vectors in a collection.
let n = db.count("docs")?;
println!("{n} vectors");

// Replace a vector (upsert = insert or update).
db.upsert("docs", 42, &new_vector, Some(serde_json::json!({"updated": true})))?;

// Delete a single vector.
db.delete("docs", 42)?;

// Drop an entire collection (removes all data from disk).
db.delete_collection("docs")?;
```

### Low-level index API (Rust)

Use index types directly — no collection layer, no WAL, no persistence unless
you call `save` manually.

```rust
use quiver_core::index::flat::FlatIndex;
use quiver_core::index::hnsw::{HnswConfig, HnswIndex};
use quiver_core::index::ivf::{IvfConfig, IvfIndex};
use quiver_core::index::quantized_flat::QuantizedFlatIndex;
use quiver_core::index::mmap_flat::MmapFlatIndex;
use quiver_core::index::VectorIndex;
use quiver_core::distance::Metric;

// ── FlatIndex (exact) ─────────────────────────────────────────────────────────
let mut flat = FlatIndex::new(3, Metric::L2);
flat.add(1, &[1.0, 0.0, 0.0])?;
flat.add(2, &[0.0, 1.0, 0.0])?;
flat.add_batch(&[(3, vec![0.5, 0.5, 0.0]), (4, vec![0.1, 0.9, 0.0])])?;

let results = flat.search(&[0.9, 0.1, 0.0], 2)?;
flat.save("/tmp/flat.bin")?;
let loaded = FlatIndex::load("/tmp/flat.bin")?;

// ── HnswIndex (ANN) ───────────────────────────────────────────────────────────
let cfg = HnswConfig { ef_construction: 200, ef_search: 50, m: 12 };
let mut hnsw = HnswIndex::new(384, Metric::Cosine, cfg);
for (id, vec) in &my_vectors {
    hnsw.add(*id, vec)?;
}
hnsw.flush();   // build graph after bulk insert
hnsw.set_ef_search(100);  // tune recall without rebuilding
let results = hnsw.search(&query, 10)?;
hnsw.save("/tmp/hnsw.bin")?;
let loaded = HnswIndex::load("/tmp/hnsw.bin")?;

// ── QuantizedFlatIndex (int8, ~4× less RAM) ───────────────────────────────────
let mut qflat = QuantizedFlatIndex::new(1536, Metric::Cosine);
qflat.add(1, &embedding)?;   // quantizes to i8 on insert
let results = qflat.search(&query, 5)?;

// ── IvfIndex (cluster ANN) ────────────────────────────────────────────────────
let cfg = IvfConfig { n_lists: 256, nprobe: 16, train_size: 4096, max_iter: 25 };
let mut ivf = IvfIndex::new(768, Metric::L2, cfg);
// Insert >= train_size vectors to trigger k-means training automatically
for (id, vec) in &large_dataset {
    ivf.add(*id, vec)?;
}
let results = ivf.search(&query, 10)?;
ivf.save("/tmp/ivf.bin")?;
let loaded = IvfIndex::load("/tmp/ivf.bin")?;

// ── MmapFlatIndex (disk-backed, near-zero RAM) ────────────────────────────────
let mut mmap = MmapFlatIndex::new(768, Metric::Cosine, "/data/vectors.mmap")?;
mmap.add(1, &embedding)?;
mmap.flush();  // writes staging buffer to disk; re-mmaps the file
let results = mmap.search(&query, 5)?;
// Persist by re-opening same path — the file IS the index
let reloaded = MmapFlatIndex::new(768, Metric::Cosine, "/data/vectors.mmap")?;
```

---

## Python

### Installation (Python)

Build and install the native extension with [maturin](https://maturin.rs):

```bash
pip install maturin

# Development install (rebuild on source changes):
maturin develop --release -m crates/quiver-python/Cargo.toml

# Verify:
python -c "import quiver; print('Quiver ready')"
```

### Basic usage (Python)

```python
import quiver

# Open (or create) a database. All existing collections load automatically.
db = quiver.Client(path="./my_quiver_data")

# Create a collection: name, vector dimensions, distance metric.
col = db.create_collection("sentences", dimensions=384, metric="cosine")

# Insert vectors with integer IDs.
col.upsert(id=1, vector=[0.12, 0.45, ...])   # 384-dim list of floats
col.upsert(id=2, vector=[0.98, 0.01, ...])
col.upsert(id=3, vector=[0.55, 0.33, ...])

# Search for the 5 nearest neighbours.
query = [0.13, 0.44, ...]
hits = col.search(query=query, k=5)

for hit in hits:
    print(f"id={hit['id']:4d}  distance={hit['distance']:.6f}")
```

### Payload metadata (Python)

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

### Filtered search (Python)

Filter on any payload field at query time. Supported operators: `$eq`, `$ne`,
`$in`, `$gt`, `$gte`, `$lt`, `$lte`. Dot-notation reaches nested fields.

```python
import quiver

db  = quiver.Client(path="./data")
col = db.get_collection("articles")

# $eq — equality
hits = col.search(query=query_embedding, k=5,
    filter={"category": {"$eq": "programming"}})

# $ne — exclude
hits = col.search(query=query_embedding, k=5,
    filter={"category": {"$ne": "spam"}})

# $in — match any of a set
hits = col.search(query=query_embedding, k=5,
    filter={"category": {"$in": ["programming", "databases"]}})

# $gt / $gte / $lt / $lte — numeric range
hits = col.search(query=query_embedding, k=5,
    filter={"score": {"$gte": 4.5}})

hits = col.search(query=query_embedding, k=5,
    filter={"score": {"$lt": 3.0}})

# $and — all conditions must hold
hits = col.search(query=query_embedding, k=5,
    filter={
        "$and": [
            {"category": {"$eq": "programming"}},
            {"score":    {"$gte": 4.0}},
        ]
    })

# $or — at least one condition must hold
hits = col.search(query=query_embedding, k=5,
    filter={
        "$or": [
            {"category": {"$eq": "programming"}},
            {"category": {"$eq": "databases"}},
        ]
    })

# Dot notation — nested payload field
hits = col.search(query=query_embedding, k=5,
    filter={"meta.author": {"$eq": "Jane Smith"}})
```

### All index types (Python)

```python
import quiver

db = quiver.Client(path="./data")

# ── 1. HNSW — approximate graph index, ~95–99% recall ────────────────────────
#    Best for: general-purpose, 100K–100M vectors.
#    Query: O(log N · ef_search)
col = db.create_collection(
    "ann",
    dimensions=768,
    metric="cosine",
    index_type="hnsw",
    ef_construction=200,  # higher → better graph, slower build
    ef_search=50,         # higher → better recall, slower query
    m=12,                 # graph edges per node per layer
)

# ── 2. Flat — exact brute-force, 100% recall ──────────────────────────────────
#    Best for: <100K vectors or when exact recall is required.
#    Query: O(N·D)
col = db.create_collection(
    "exact",
    dimensions=768,
    metric="cosine",
    index_type="flat",
)

# ── 3. QuantizedFlat — int8 brute-force, ~4× lower memory ────────────────────
#    Best for: memory-constrained machines. ~99% recall vs FlatIndex.
#    1M × 1536-dim: ~1.5 GB instead of 6 GB.
col = db.create_collection(
    "quantized",
    dimensions=1536,
    metric="cosine",
    index_type="quantized_flat",
)

# ── 4. IVF — cluster-based ANN, sub-linear search ────────────────────────────
#    Best for: 100K–100M vectors; recall tunable via nprobe.
#    Auto-trains on first train_size inserts (Lloyd's k-means).
#    Rule of thumb: n_lists = sqrt(N), nprobe = sqrt(n_lists)
col = db.create_collection(
    "ivf",
    dimensions=768,
    metric="l2",
    index_type="ivf",
    n_lists=256,     # number of k-means clusters
    nprobe=16,       # clusters to scan at query time (higher = better recall)
    train_size=4096, # vectors to buffer before training
)

# ── 5. MmapFlat — disk-mapped brute-force, near-zero RAM ─────────────────────
#    Best for: datasets larger than available RAM.
#    The OS pages in only the 4 KB blocks currently being scanned.
col = db.create_collection(
    "mmap",
    dimensions=768,
    metric="cosine",
    index_type="mmap_flat",
)

# ── 6. FAISS — factory-string index (feature-gated) ──────────────────────────
#    Requires quiver-python compiled with --features faiss.
#    Common factory strings:
#      "Flat"         — exact brute-force (SIMD-accelerated)
#      "IVF1024,Flat" — IVF with 1024 clusters
#      "IVF256,PQ64"  — IVF + product quantisation (compressed)
#      "HNSW32"       — FAISS's own HNSW
#      "IVF4096,SQ8"  — IVF + 8-bit scalar quantiser
col = db.create_collection(
    "faiss_ivf",
    dimensions=768,
    metric="l2",
    index_type="faiss",
    faiss_factory="IVF1024,Flat",
)
```

#### Per-index usage notes

```python
# IVF — insert at least train_size vectors before searching.
# The index buffers vectors and trains automatically.
ivf_col = db.create_collection("ivf_demo", dimensions=128, metric="l2",
                                index_type="ivf", n_lists=64, nprobe=8, train_size=512)
for i, vec in enumerate(dataset):
    ivf_col.upsert(id=i, vector=vec)
# After train_size inserts the k-means training triggers automatically.
hits = ivf_col.search(query=query_vec, k=10)

# MmapFlat — flush() writes the staging buffer to the memory-mapped file.
# Searching also works before flush (staging is scanned in-memory).
mmap_col = db.create_collection("mmap_demo", dimensions=128, metric="cosine",
                                 index_type="mmap_flat")
for i, vec in enumerate(dataset):
    mmap_col.upsert(id=i, vector=vec)
# The Collection layer calls flush() automatically on WAL compaction.
```

### Collection management (Python)

```python
import quiver

db = quiver.Client(path="./data")

# Create only if it doesn't exist (idempotent).
col = db.get_or_create_collection("docs", dimensions=768, metric="cosine")

# Get a handle to an existing collection.
col = db.get_collection("docs")

# List all collection names.
print(db.list_collections())    # ['docs', 'articles', ...]

# Count vectors.
print(col.count)                # 1024

# Replace (upsert = insert or update by ID).
col.upsert(id=42, vector=new_vec, payload={"updated": True})

# Delete a single vector.
col.delete(id=42)

# Drop an entire collection (data removed from disk).
db.delete_collection("docs")
```

### Low-level index API (Python)

Use `FlatIndex` and `HnswIndex` directly — no collection layer, no WAL,
no persistence unless you call `save` explicitly.

```python
import quiver

# ── FlatIndex (exact, 100% recall) ────────────────────────────────────────────
idx = quiver.FlatIndex(dimensions=3, metric="l2")
idx.add(id=1, vector=[1.0, 0.0, 0.0])
idx.add(id=2, vector=[0.0, 1.0, 0.0])
idx.add_batch([(3, [0.5, 0.5, 0.0]), (4, [0.1, 0.9, 0.0])])

results = idx.search(query=[0.9, 0.1, 0.0], k=2)
for r in results:
    print(f"id={r['id']}  dist={r['distance']:.4f}")

print(len(idx))           # 4
print(idx.dimensions)     # 3
print(idx.metric)         # "l2"

idx.save("/tmp/flat.bin")
loaded = quiver.FlatIndex.load("/tmp/flat.bin")

# ── HnswIndex (approximate, fast) ─────────────────────────────────────────────
hnsw = quiver.HnswIndex(
    dimensions=384,
    metric="cosine",
    ef_construction=200,   # beam width during graph build
    ef_search=50,          # beam width at query time (tunable without rebuild)
    m=12,                  # edges per node per layer
)

for id, vec in my_vectors:
    hnsw.add(id=id, vector=vec)

hnsw.flush()    # build HNSW graph after bulk insert

results = hnsw.search(query=query_vec, k=10)
hnsw.save("/tmp/hnsw.bin")
loaded = quiver.HnswIndex.load("/tmp/hnsw.bin")
```

### Real-world example (Python)

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
    {"id": 5, "text": "IVF clusters vectors with k-means for sub-linear search."},
]

db  = quiver.Client(path="./semantic_search_data")
col = db.get_or_create_collection("docs", dimensions=384, metric="cosine")

for doc in docs:
    vec = model.encode(doc["text"]).tolist()
    col.upsert(id=doc["id"], vector=vec, payload={"text": doc["text"]})

# --- Unfiltered search ---
query_vec = model.encode("how do embeddings work?").tolist()
hits = col.search(query=query_vec, k=3)
print("Top 3 results:")
for hit in hits:
    print(f"  [{hit['distance']:.4f}] {hit['payload']['text']}")

# --- Filtered search: only programming content ---
hits = col.search(
    query=query_vec,
    k=3,
    filter={"$or": [
        {"text": {"$eq": "Rust is a systems programming language focused on safety."}},
    ]}
)
```

---

## Distance metrics

| Metric | Rust | Python | Use when |
|--------|------|--------|----------|
| Cosine similarity | `Metric::Cosine` | `"cosine"` | Text/image embeddings (most common) |
| Euclidean (L2) | `Metric::L2` | `"l2"` | Geometry, sensor data, unnormalised vectors |
| Dot product | `Metric::DotProduct` | `"dot_product"` | Pre-normalised vectors, recommendation models |

> **Tip:** Most embedding models (OpenAI, sentence-transformers, Ollama) produce
> vectors intended for cosine similarity. Use `Metric::Cosine` / `"cosine"`
> unless your model documentation says otherwise.

---

## Index type reference

| Index | Rust | Python | Recall | Query complexity | RAM | Best for |
|-------|------|--------|--------|-----------------|-----|----------|
| Flat | `IndexType::Flat` | `"flat"` | 100% | O(N·D) | All vectors | <100K vectors, exact recall required |
| HNSW | `IndexType::Hnsw` | `"hnsw"` | 95–99% | O(log N · ef) | All vectors | General purpose, 100K–100M vectors |
| QuantizedFlat | `IndexType::QuantizedFlat` | `"quantized_flat"` | ~99% | O(N·D) | **4× less** | Memory-constrained, brute-force needed |
| IVF | `IndexType::Ivf` | `"ivf"` | Tunable | O(n_lists + nprobe·N/n_lists) | All vectors | 100K–100M, needs training phase |
| MmapFlat | `IndexType::MmapFlat` | `"mmap_flat"` | 100% | O(N·D) disk-paged | Staging only | Datasets larger than RAM |
| FAISS | `IndexType::Faiss` | `"faiss"` | Varies | Varies | Varies | FAISS factory strings, GPU, PQ compression |

### Config parameters

**HNSW**
| Parameter | Default | Effect |
|-----------|---------|--------|
| `ef_construction` | 200 | Beam width during graph build. Higher → better recall, slower build. |
| `ef_search` | 50 | Beam width at query time. Tunable without rebuilding. |
| `m` | 12 | Edges per node per layer. Higher → better recall, more RAM. |

**IVF**
| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_lists` | 256 | Number of k-means clusters. Rule of thumb: `sqrt(N)`. |
| `nprobe` | 16 | Clusters scanned per query. Higher → better recall, slower query. |
| `train_size` | 4096 | Vectors buffered before k-means training triggers. |
| `max_iter` | 25 | Lloyd's k-means iterations. |

---

## Data directory layout

```
my_quiver_data/
├── sentences/
│   ├── meta.json      ← collection config (dimensions, metric, index type)
│   ├── wal.log        ← write-ahead log (binary, bincode frames)
│   └── hnsw.graph     ← persisted HNSW graph (written on WAL compaction)
├── mmap_collection/
│   ├── meta.json
│   ├── wal.log
│   └── vectors.mmap   ← memory-mapped binary vector file (MmapFlat only)
└── ...
```

- **`meta.json`** — human-readable collection metadata
- **`wal.log`** — binary append log; replayed on startup to reconstruct the index
- **`hnsw.graph`** — serialised HNSW graph; loaded with `mmap` to skip O(N log N) rebuild
- **`vectors.mmap`** — flat binary file used by `MmapFlatIndex`; format: 32-byte header + packed `(u64 id, f32[dims])` records
