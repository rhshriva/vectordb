# Quiver — Embedded Usage Guide

Quiver runs **fully in-process** — no server to start, no network calls, no
extra processes.  Just open a path and go.

---

## Table of Contents

1. [Rust](#rust)
   - [Installation](#installation-rust)
   - [Basic usage](#basic-usage-rust)
   - [Payload metadata](#payload-metadata-rust)
   - [Filtered search](#filtered-search-rust)
   - [Index types](#index-types-rust)
   - [Low-level index API](#low-level-index-api-rust)
2. [Python](#python)
   - [Installation](#installation-python)
   - [Basic usage](#basic-usage-python)
   - [Payload metadata](#payload-metadata-python)
   - [Filtered search](#filtered-search-python)
   - [Index types](#index-types-python)
   - [Low-level index API](#low-level-index-api-python)
3. [Data directory layout](#data-directory-layout)
4. [Distance metrics](#distance-metrics)

---

## Rust

### Installation (Rust)

Add `quiver-core` to your `Cargo.toml`:

```toml
[dependencies]
quiver-core = { path = "./crates/quiver-core" }   # local workspace
serde_json = "1"                                   # for payload values
```

### Basic usage (Rust)

```rust
use quiver_core::{Quiver, Metric};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open (or create) a database at a local directory.
    // All collections inside are loaded automatically.
    let mut db = Quiver::open("./my_quiver_data")?;

    // Create a collection — name, vector dimensions, distance metric.
    db.create_collection("sentences", 384, Metric::Cosine)?;

    // Insert vectors with integer IDs.
    // Vectors can come from any embedding model (OpenAI, Ollama, sentence-transformers, etc.)
    db.upsert("sentences", 1, &[0.12, 0.45, /* ... 384 dims */], None)?;
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

Each vector can carry an arbitrary JSON payload.  Payloads are returned with
search results.

```rust
use quiver_core::{Quiver, Metric};
use serde_json::json;

let mut db = Quiver::open("./data")?;
db.create_collection("articles", 1536, Metric::Cosine)?;

// Attach metadata to every vector.
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

// Search returns payloads alongside distances.
let hits = db.search("articles", &query_embedding, 3)?;
for hit in &hits {
    if let Some(payload) = &hit.payload {
        println!("{} — {}", hit.id, payload["title"]);
    }
}
```

### Filtered search (Rust)

Filter by any payload field at query time.  Only matching vectors are returned
(pre-filtered before ranking).

```rust
use quiver_core::{Quiver, Metric};
use quiver_core::payload::FilterCondition;
use serde_json::json;

let mut db = Quiver::open("./data")?;
// ... upsert vectors with category / score payloads ...

// Equality filter — only return articles in "programming".
let filter: FilterCondition = serde_json::from_value(json!({
    "category": { "$eq": "programming" }
}))?;
let hits = db.search_filtered("articles", &query, 5, &filter)?;

// Range filter — score >= 4.5
let filter: FilterCondition = serde_json::from_value(json!({
    "score": { "$gte": 4.5 }
}))?;

// AND of multiple conditions
let filter: FilterCondition = serde_json::from_value(json!({
    "$and": [
        { "category": { "$eq": "programming" } },
        { "score": { "$gte": 4.0 } }
    ]
}))?;

// OR — one of several values
let filter: FilterCondition = serde_json::from_value(json!({
    "category": { "$in": ["programming", "databases"] }
}))?;
```

**Supported operators:** `$eq`, `$ne`, `$in`, `$gt`, `$gte`, `$lt`, `$lte`

### Index types (Rust)

Quiver selects HNSW by default (best recall/speed tradeoff for most datasets).

| Index | When to use |
|---|---|
| `Metric::Cosine` / `Metric::L2` / `Metric::DotProduct` | Choose the metric that matches your embedding model |
| HNSW (default) | General purpose, ~95–99% recall, sub-linear query time |
| Flat | Small datasets (<100K), 100% recall, exact brute-force |

```rust
use quiver_core::{Quiver, Metric};

let mut db = Quiver::open("./data")?;

// HNSW (default) — best for most use cases
db.create_collection("hnsw_collection", 768, Metric::Cosine)?;

// For exact search on small sets, use the collection manager directly
// with IndexType::Flat (see CollectionManager docs)
```

### Collection management (Rust)

```rust
let mut db = Quiver::open("./data")?;

// Create only if it doesn't exist yet (idempotent).
db.get_or_create_collection("docs", 768, Metric::Cosine)?;

// List all collections.
let names = db.list_collections();
println!("{:?}", names);  // ["docs", "articles", ...]

// Count vectors in a collection.
let n = db.count("docs")?;
println!("{n} vectors");

// Delete a single vector.
db.delete("docs", 42)?;

// Drop an entire collection (removes all data from disk).
db.delete_collection("docs")?;
```

### Low-level index API (Rust)

For maximum control you can use `FlatIndex` or `HnswIndex` directly — no
collection layer, no WAL, no persistence by default.

```rust
use quiver_core::{FlatIndex, HnswIndex, HnswConfig, Metric, VectorIndex};

// --- FlatIndex (exact, 100% recall) ---
let mut idx = FlatIndex::new(3, Metric::L2);
idx.add(1, &[1.0, 0.0, 0.0])?;
idx.add(2, &[0.0, 1.0, 0.0])?;

let results = idx.search(&[0.9, 0.1, 0.0], 2)?;
for r in &results {
    println!("id={} dist={:.4}", r.id, r.distance);
}

// Persist to disk (bincode binary format).
idx.save("/tmp/my_flat_index.bin")?;
let loaded = FlatIndex::load("/tmp/my_flat_index.bin")?;

// --- HnswIndex (approximate, fast) ---
let cfg = HnswConfig {
    ef_construction: 200,   // higher = better recall at build time
    ef_search: 50,          // higher = better recall at query time
    m: 12,                  // graph connectivity
};
let mut hnsw = HnswIndex::new(384, Metric::Cosine, cfg);
for (id, vec) in my_vectors.iter() {
    hnsw.add(*id, vec)?;
}
hnsw.flush(); // build graph after bulk insert

let results = hnsw.search(&query, 10)?;
hnsw.save("/tmp/my_hnsw_index.bin")?;
let loaded = HnswIndex::load("/tmp/my_hnsw_index.bin")?;
```

---

## Python

### Installation (Python)

Build and install the native extension with [maturin](https://maturin.rs):

```bash
# Inside a virtual environment:
pip install maturin

# Build and install in development mode:
maturin develop --release -m crates/quiver-python/Cargo.toml

# Verify:
python -c "import quiver; print('Quiver ready')"
```

### Basic usage (Python)

```python
import quiver

# Open (or create) a database at a local directory.
# All previously saved collections are loaded automatically.
db = quiver.Client(path="./my_quiver_data")

# Create a collection — name, vector dimensions, distance metric.
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

db = quiver.Client(path="./data")
col = db.create_collection("articles", dimensions=1536, metric="cosine")

# Attach any JSON-serializable dict as payload.
col.upsert(id=1, vector=embedding_1, payload={
    "title":    "Introduction to Rust",
    "author":   "Jane Smith",
    "category": "programming",
    "score":    4.8
})

col.upsert(id=2, vector=embedding_2, payload={
    "title":    "Vector Databases Explained",
    "author":   "Bob Jones",
    "category": "databases",
    "score":    4.5
})

# Payloads come back with each search result.
hits = col.search(query=query_embedding, k=3)
for hit in hits:
    print(f"{hit['id']} — {hit['payload']['title']} (dist={hit['distance']:.4f})")
```

### Filtered search (Python)

```python
import quiver

db = quiver.Client(path="./data")
col = db.get_collection("articles")

# Equality filter
hits = col.search(
    query=query_embedding,
    k=5,
    filter={"category": {"$eq": "programming"}}
)

# Range filter — score >= 4.5
hits = col.search(
    query=query_embedding,
    k=5,
    filter={"score": {"$gte": 4.5}}
)

# AND of multiple conditions
hits = col.search(
    query=query_embedding,
    k=5,
    filter={
        "$and": [
            {"category": {"$eq": "programming"}},
            {"score": {"$gte": 4.0}}
        ]
    }
)

# OR — match any of several values
hits = col.search(
    query=query_embedding,
    k=5,
    filter={"category": {"$in": ["programming", "databases"]}}
)
```

**Supported operators:** `$eq`, `$ne`, `$in`, `$gt`, `$gte`, `$lt`, `$lte`

### Index types (Python)

```python
import quiver

db = quiver.Client(path="./data")

# HNSW — default, best for most datasets (approximate, fast)
col = db.create_collection("hnsw_col", dimensions=768, metric="cosine", index_type="hnsw")

# Flat — exact brute-force, 100% recall, best for small datasets (<100K)
col = db.create_collection("flat_col", dimensions=768, metric="cosine", index_type="flat")

# Supported metrics: "cosine", "l2", "dot_product"
```

### Collection management (Python)

```python
import quiver

db = quiver.Client(path="./data")

# Create only if it doesn't exist yet.
col = db.get_or_create_collection("docs", dimensions=768, metric="cosine")

# Get a handle to an existing collection.
col = db.get_collection("docs")

# List all collection names.
print(db.list_collections())   # ['docs', 'articles', ...]

# Number of vectors in a collection.
print(col.count)               # 1024

# Delete a single vector.
col.delete(id=42)

# Drop an entire collection (data removed from disk).
db.delete_collection("docs")
```

### Low-level index API (Python)

For direct index access without the collection/persistence layer:

```python
import quiver

# --- FlatIndex (exact, 100% recall) ---
idx = quiver.FlatIndex(dimensions=3, metric="l2")
idx.add(id=1, vector=[1.0, 0.0, 0.0])
idx.add(id=2, vector=[0.0, 1.0, 0.0])
idx.add_batch([(3, [0.5, 0.5, 0.0]), (4, [0.1, 0.9, 0.0])])

results = idx.search(query=[0.9, 0.1, 0.0], k=2)
for r in results:
    print(f"id={r['id']}  dist={r['distance']:.4f}")

print(len(idx))          # 4
print(idx.dimensions)    # 3
print(idx.metric)        # "l2"

# Save and load (binary format).
idx.save("/tmp/my_flat_index.bin")
loaded = quiver.FlatIndex.load("/tmp/my_flat_index.bin")

# --- HnswIndex (approximate, fast) ---
hnsw = quiver.HnswIndex(
    dimensions=384,
    metric="cosine",
    ef_construction=200,   # higher = better recall at build
    ef_search=50,          # higher = better recall at query
    m=12,                  # graph connectivity
)

for id, vec in my_vectors:
    hnsw.add(id=id, vector=vec)

hnsw.flush()    # build HNSW graph after bulk insert

results = hnsw.search(query=query_vec, k=10)
hnsw.save("/tmp/my_hnsw_index.bin")
loaded = quiver.HnswIndex.load("/tmp/my_hnsw_index.bin")
```

### Real-world example — semantic search over documents (Python)

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

# Build the index
db = quiver.Client(path="./semantic_search_data")
col = db.get_or_create_collection("docs", dimensions=384, metric="cosine")

for doc in docs:
    embedding = model.encode(doc["text"]).tolist()
    col.upsert(id=doc["id"], vector=embedding, payload={"text": doc["text"]})

# Query
query = "how do embeddings work for similarity?"
query_vec = model.encode(query).tolist()
hits = col.search(query=query_vec, k=3)

print(f"Query: {query}\n")
for hit in hits:
    print(f"  [{hit['distance']:.4f}] {hit['payload']['text']}")
```

---

## Data directory layout

```
my_quiver_data/
├── sentences/
│   ├── meta.json       ← collection config (dimensions, metric, index type)
│   └── wal.log         ← write-ahead log (binary, length-prefixed bincode frames)
├── articles/
│   ├── meta.json
│   └── wal.log
└── ...
```

- **`meta.json`** — human-readable collection metadata (name, dimensions, metric, index type)
- **`wal.log`** — binary append log; replayed on startup to reconstruct the in-memory index

---

## Distance metrics

| Metric | API value | Use when |
|---|---|---|
| Cosine similarity | `Metric::Cosine` / `"cosine"` | Text and image embeddings (most common) |
| Euclidean (L2) | `Metric::L2` / `"l2"` | Geometry, sensor data, unnormalized vectors |
| Dot product | `Metric::DotProduct` / `"dot_product"` | Pre-normalized vectors, recommendation models |

> **Tip:** Most embedding models (OpenAI, sentence-transformers, Ollama) produce
> vectors intended for cosine similarity. Use `Metric::Cosine` unless your model
> documentation says otherwise.
