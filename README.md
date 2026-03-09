# vectordb

A high-performance vector database written in Rust — available as an **embedded Python library**, a **standalone HTTP server**, or a **Rust library** you link directly into your application.

---

## Table of Contents

- [Overview](#overview)
- [Two Deployment Modes](#two-deployment-modes)
- [Prerequisites](#prerequisites)
- [Building the Project](#building-the-project)
- [Mode 1 — Client-Server Setup](#mode-1--client-server-setup)
  - [Start the server](#start-the-server)
  - [Call via curl](#call-via-curl)
  - [Call via TypeScript SDK](#call-via-typescript-sdk)
  - [Call via CLI (vdb)](#call-via-cli-vdb)
  - [API key authentication](#api-key-authentication)
- [Mode 2 — Embedded Setup (Python)](#mode-2--embedded-setup-python)
  - [Install](#install-python-library)
  - [Quickstart](#python-quickstart)
  - [Persistence across restarts](#persistence-across-restarts)
  - [Python API reference](#python-api-reference)
- [Payload Metadata & Filtered Search](#payload-metadata--filtered-search)
- [Index Types](#index-types)
  - [HNSW tuning](#hnsw-tuning)
  - [FAISS integration](#faiss-integration)
- [Distance Metrics](#distance-metrics)
- [Persistence & Durability](#persistence--durability)
- [REST API Reference](#rest-api-reference)
- [Architecture](#architecture)
- [Development Build Script](#development-build-script)
- [Building on macOS](#building-on-macos)

---

## Overview

**vectordb** stores, indexes, and searches high-dimensional vectors at scale. It supports:

- **Exact search** via `FlatIndex` (brute-force, 100% recall)
- **Approximate search** via `HnswIndex` (graph-based ANN, ~95–99% recall, sub-linear time)
- **FAISS-accelerated search** via `FaissIndex` (SIMD-optimised flat, IVF, PQ, HNSW — optional feature)
- **Payload metadata** attached to every vector — filter results by arbitrary JSON fields
- **WAL-based persistence** — all writes survive process restarts automatically
- **Two deployment modes** — embedded Python library or standalone HTTP server

---

## Two Deployment Modes

| | Embedded Python | HTTP Server |
|---|---|---|
| **Setup** | `pip install` + one import | Start a binary, call over HTTP |
| **Process boundary** | In-process | Separate service |
| **Use case** | Scripts, notebooks, ML pipelines | Multi-client, microservices, polyglot teams |
| **Networking** | None (zero-copy) | HTTP/JSON on port 8080 |
| **Languages** | Python | Any (curl, TypeScript SDK, vdb CLI, …) |
| **Persistence** | WAL on disk | WAL on disk |
| **FAISS** | Not exposed (use server mode) | Yes, with `--features faiss` |

---

## Prerequisites

### Rust (required for all modes)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustc --version   # should print rustc 1.75 or newer
```

### Python bindings (embedded mode only)

```bash
python3 --version   # 3.8 or newer required
pip install maturin
```

### FAISS (optional — server mode only)

FAISS must be available as a shared library (`libfaiss_c.so` on Linux,
`libfaiss_c.dylib` on macOS) **before** building with the `faiss` feature.

```bash
# Ubuntu / Debian — install pre-built FAISS
sudo apt-get install libfaiss-dev

# macOS
brew install faiss

# Or build from source (needs cmake, a BLAS library, and a C++17 compiler):
# https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
```

> **No FAISS installed?** The crate compiles and runs fine without it. FAISS
> is an opt-in feature — omitting `--features faiss` gives you the same
> `Flat` and `HNSW` indexes as before.

---

## Building the Project

### Quick start (recommended)

Use the included `dev_build.sh` script:

```bash
# Build + run all tests (default, no FAISS)
./dev_build.sh

# Build with FAISS support + run tests
./dev_build.sh --faiss

# Build release binaries (no tests)
./dev_build.sh --release

# Build and start the HTTP server
./dev_build.sh --server

# Build and start the HTTP server with FAISS support
./dev_build.sh --faiss --server

# Build the Python wheel and smoke-test it (activate a venv first)
python3 -m venv .venv && source .venv/bin/activate
./dev_build.sh --python

# See all options
./dev_build.sh --help
```

### Manual cargo commands

```bash
# ── Debug build (fast iteration) ─────────────────────────────────────────────

# All Rust crates, no FAISS
cargo build

# All Rust crates, with FAISS
cargo build --features vectordb-core/faiss

# Run the full test suite
cargo test

# Run tests with FAISS
cargo test --features vectordb-core/faiss

# ── Release build (production) ────────────────────────────────────────────────

cargo build --release
cargo build --release --features vectordb-server/faiss

# Binaries end up in:
#   target/release/vectordb-server
#   target/release/vdb              (CLI)
```

### Project layout

```
vectordb/
├── dev_build.sh                  ← developer helper script
├── Cargo.toml                    ← workspace manifest
├── crates/
│   ├── vectordb-core/            ← index trait, WAL, Collection, filters
│   │   └── src/index/
│   │       ├── flat.rs           ← FlatIndex  (exact, brute-force)
│   │       ├── hnsw.rs           ← HnswIndex  (approximate, graph)
│   │       └── faiss.rs          ← FaissIndex (FAISS, optional feature)
│   ├── vectordb-server/          ← Axum HTTP server (port 8080)
│   ├── vectordb-cli/             ← vdb command-line client
│   ├── vectordb-embeddings/      ← OpenAI / Ollama embedding providers
│   └── vectordb-python/          ← PyO3 Python bindings
└── sdks/
    └── typescript/               ← TypeScript HTTP client SDK
```

---

## Mode 1 — Client-Server Setup

In this mode you run `vectordb-server` as a standalone process and connect to
it from any language over HTTP.

```
Your app (Python / TypeScript / Go / curl / …)
        │  HTTP  POST /collections/:name/search
        ▼
  vectordb-server  :8080
        │
  vectordb-core  (FlatIndex / HnswIndex / FaissIndex)
        │
  ./data/{collection}/  (WAL on disk)
```

### Start the server

```bash
# --- Debug (fastest to compile) ---
cargo run -p vectordb-server

# --- Release (fastest to run, recommended for production) ---
cargo build --release
./target/release/vectordb-server

# --- With FAISS support ---
cargo build --release --features vectordb-server/faiss
./target/release/vectordb-server

# --- Environment variables ---
VECTORDB_DATA_DIR=/var/lib/vectordb \   # where to persist data (default: ./data)
VECTORDB_API_KEY=mysecretkey        \   # enable Bearer-token auth (optional)
RUST_LOG=info                       \   # log level
./target/release/vectordb-server

# Server prints:
# INFO vectordb-server listening on 0.0.0.0:8080
```

### Call via curl

```bash
# ── Collections ──────────────────────────────────────────────────────────────

# Create a flat collection (exact search)
curl -s -X POST http://localhost:8080/collections/articles \
  -H "Content-Type: application/json" \
  -d '{"dimensions": 3, "metric": "cosine", "index_type": "flat"}'

# Create an HNSW collection (approximate search, good for large datasets)
curl -s -X POST http://localhost:8080/collections/large \
  -H "Content-Type: application/json" \
  -d '{
    "dimensions": 1536,
    "metric": "cosine",
    "index_type": "hnsw",
    "hnsw": {"ef_construction": 200, "ef_search": 50, "m": 16}
  }'

# Create a FAISS collection (requires --features faiss at build time)
curl -s -X POST http://localhost:8080/collections/faiss_col \
  -H "Content-Type: application/json" \
  -d '{
    "dimensions": 1536,
    "metric": "cosine",
    "index_type": "faiss",
    "faiss_factory": "IVF1024,Flat"
  }'

# List all collections
curl -s http://localhost:8080/collections

# Get collection info (count, dimensions, metric)
curl -s http://localhost:8080/collections/articles

# Delete a collection
curl -s -X DELETE http://localhost:8080/collections/articles

# ── Vectors ───────────────────────────────────────────────────────────────────

# Upsert vectors (with optional JSON payload)
curl -s -X POST http://localhost:8080/collections/articles/vectors \
  -H "Content-Type: application/json" \
  -d '{"id": 1, "vector": [1.0, 0.0, 0.0], "payload": {"category": "tech", "year": 2024}}'

curl -s -X POST http://localhost:8080/collections/articles/vectors \
  -H "Content-Type: application/json" \
  -d '{"id": 2, "vector": [0.0, 1.0, 0.0], "payload": {"category": "sport", "year": 2023}}'

# Search — top-k nearest neighbours
curl -s -X POST http://localhost:8080/collections/articles/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [1.0, 0.0, 0.0], "k": 2}'
# {"results":[{"id":1,"distance":0.0,"payload":{"category":"tech","year":2024}}, ...]}

# Filtered search — only tech articles
curl -s -X POST http://localhost:8080/collections/articles/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [1.0, 0.0, 0.0],
    "k": 5,
    "filter": {"category": {"$eq": "tech"}}
  }'

# Compound filter (AND)
curl -s -X POST http://localhost:8080/collections/articles/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [1.0, 0.0, 0.0],
    "k": 5,
    "filter": {"$and": [
      {"category": {"$eq": "tech"}},
      {"year": {"$gte": 2024}}
    ]}
  }'

# Delete a vector
curl -s -X DELETE http://localhost:8080/collections/articles/vectors/1

# ── Embedding endpoints (auto embed text → vector) ────────────────────────────

# Create a collection that uses OpenAI embeddings
curl -s -X POST http://localhost:8080/collections/docs \
  -H "Content-Type: application/json" \
  -d '{"dimensions": 1536, "metric": "cosine", "embedding_model": "openai/text-embedding-3-small"}'

# Upsert by raw text (server calls OpenAI automatically)
OPENAI_API_KEY=sk-... curl -s -X POST http://localhost:8080/collections/docs/embed-upsert \
  -H "Content-Type: application/json" \
  -d '{"id": 1, "text": "Rust is a systems programming language", "payload": {"lang": "rust"}}'

# Search by raw text
OPENAI_API_KEY=sk-... curl -s -X POST http://localhost:8080/collections/docs/embed-search \
  -H "Content-Type: application/json" \
  -d '{"text": "memory-safe systems languages", "k": 5}'
```

### Call via TypeScript SDK

```bash
cd sdks/typescript
npm install
npm run build
```

```typescript
import { VectorDbClient } from "vectordb-client";

const client = new VectorDbClient({ baseUrl: "http://localhost:8080" });

// Create a collection
await client.createCollection("articles", {
  dimensions: 3,
  metric: "cosine",
  index_type: "hnsw",
});

// Upsert vectors
await client.upsert("articles", { id: 1, vector: [1.0, 0.0, 0.0], payload: { category: "tech" } });
await client.upsert("articles", { id: 2, vector: [0.0, 1.0, 0.0], payload: { category: "sport" } });

// Search
const results = await client.search("articles", {
  vector: [1.0, 0.0, 0.0],
  k: 2,
  filter: { category: { $eq: "tech" } },
});
console.log(results);

// Create a FAISS collection (server must be built with --features faiss)
await client.createCollection("faiss_col", {
  dimensions: 1536,
  metric: "cosine",
  index_type: "faiss",
  faiss_factory: "IVF1024,Flat",
});
```

### Call via CLI (vdb)

```bash
# Build (included in cargo build --release above)
# Binary: target/release/vdb

# Default server: http://localhost:8080
# Override: --host or VDB_HOST env var

# List collections
vdb list

# Create a collection
vdb create articles --dimensions 3 --metric cosine --index hnsw

# Insert
vdb insert articles --id 1 --vector "1.0,0.0,0.0"

# Search
vdb search articles --vector "1.0,0.0,0.0" --k 5

# Delete a vector
vdb delete articles --id 1

# Drop a collection
vdb drop articles

# Point at a different server
vdb --host http://prod-server:8080 list
```

### API key authentication

```bash
# Start with auth enabled
VECTORDB_API_KEY=mysecretkey ./target/release/vectordb-server

# All requests must include the Bearer token
curl -s http://localhost:8080/collections \
  -H "Authorization: Bearer mysecretkey"

# Missing or wrong key → 401
curl -s http://localhost:8080/collections
# {"error":"invalid or missing API key"}
```

---

## Mode 2 — Embedded Setup (Python)

In this mode the database runs entirely inside your Python process. No server
binary, no network, no serialisation overhead.

```
Your Python process
  │
  ├── vectordb.Client(path="./data")
  │     └── CollectionManager  (Rust, in-process via PyO3)
  │           └── Collection   (FlatIndex / HnswIndex + WAL)
  │
  └── ./data/{collection}/  (WAL on disk)
```

### Install Python library

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate         # Linux / macOS
# .venv\Scripts\activate          # Windows

# 2. Install maturin (the Rust→Python build tool)
pip install maturin

# 3. Build and install vectordb into the active venv
maturin develop --release -m crates/vectordb-python/Cargo.toml

# 4. Verify
python3 -c "import vectordb; print('vectordb', vectordb.__version__)"
```

> **Tip:** `maturin develop` re-compiles on every call. For one-off builds use
> `maturin build --release` to produce a `.whl` wheel you can `pip install`.

### Python quickstart

```python
import vectordb

# ── Create / open a database ──────────────────────────────────────────────────
# All writes are persisted automatically to ./mydata on disk.
db = vectordb.Client(path="./mydata")

# ── Collections ───────────────────────────────────────────────────────────────
# "flat"  → exact search, best for < 100 K vectors
# "hnsw"  → approximate search, best for > 100 K vectors
col = db.create_collection("articles", dimensions=3, metric="cosine", index_type="hnsw")

# ── Upsert vectors with optional metadata ────────────────────────────────────
col.upsert(1, [1.0, 0.0, 0.0], payload={"category": "tech",  "year": 2024})
col.upsert(2, [0.0, 1.0, 0.0], payload={"category": "sport", "year": 2023})
col.upsert(3, [0.9, 0.1, 0.0], payload={"category": "tech",  "year": 2023})
col.upsert(4, [0.1, 0.9, 0.0], payload={"category": "sport", "year": 2024})

# ── Plain search ──────────────────────────────────────────────────────────────
results = col.search([1.0, 0.0, 0.0], k=2)
# [{"id": 1, "distance": 0.0,  "payload": {"category": "tech",  "year": 2024}},
#  {"id": 3, "distance": 0.02, "payload": {"category": "tech",  "year": 2023}}]

# ── Filtered search ───────────────────────────────────────────────────────────
results = col.search(
    [1.0, 0.0, 0.0],
    k=5,
    filter={"category": {"$eq": "tech"}},
)

# Compound filter
results = col.search(
    [1.0, 0.0, 0.0],
    k=5,
    filter={"$and": [
        {"category": {"$eq": "tech"}},
        {"year":     {"$gte": 2024}},
    ]},
)

# ── Other operations ──────────────────────────────────────────────────────────
col.delete(3)                      # remove vector by ID → bool
print(col.count)                   # number of live vectors
print(col.name)                    # "articles"

db.list_collections()              # ["articles"]
db.delete_collection("articles")   # remove everything
```

### Persistence across restarts

```python
import vectordb

# First run — write some vectors
db = vectordb.Client(path="./mydata")
col = db.create_collection("docs", dimensions=768, metric="cosine")
col.upsert(1, my_embedding, payload={"text": "hello world"})

# Later run — data reloaded automatically from the WAL
db = vectordb.Client(path="./mydata")
col = db.get_collection("docs")   # KeyError if it doesn't exist
print(col.count)                  # 1 — survived the restart
```

### Python API reference

**`vectordb.Client(path="./data")`**

| Method | Returns | Description |
|--------|---------|-------------|
| `create_collection(name, dimensions, metric="cosine", index_type="hnsw")` | `Collection` | Create a new collection |
| `get_collection(name)` | `Collection` | Retrieve existing collection (`KeyError` if missing) |
| `get_or_create_collection(name, dimensions, metric="cosine")` | `Collection` | Create if absent |
| `delete_collection(name)` | `bool` | Delete collection and all its data |
| `list_collections()` | `list[str]` | Names of all known collections |

**`vectordb.Collection`**

| Method / property | Description |
|-------------------|-------------|
| `upsert(id, vector, payload=None)` | Add or replace a vector |
| `search(query, k, filter=None)` → `list[dict]` | kNN search with optional filter |
| `delete(id)` → `bool` | Remove a vector |
| `count` | Number of live vectors |
| `name` | Collection name |

---

## Payload Metadata & Filtered Search

Every vector can carry an arbitrary JSON payload, stored alongside the vector
and returned in search results.

### Upsert with payload

```json
POST /collections/docs/vectors
{
  "id": 42,
  "vector": [0.1, 0.2, 0.3],
  "payload": {
    "title":  "Introduction to Rust",
    "author": "Alice",
    "tags":   ["rust", "systems"],
    "score":  0.95,
    "meta":   { "source": "blog" }
  }
}
```

### Filter operators

Filters are applied **after** the ANN search with a 10× overscan.
Syntax uses MongoDB-style operators:

| Operator | Meaning | Example |
|----------|---------|---------|
| `$eq` | Equal | `{"field": {"$eq": "value"}}` |
| `$ne` | Not equal | `{"field": {"$ne": "value"}}` |
| `$in` | One of | `{"field": {"$in": ["a", "b"]}}` |
| `$gt` | Greater than | `{"score": {"$gt": 0.5}}` |
| `$gte` | Greater or equal | `{"year": {"$gte": 2020}}` |
| `$lt` | Less than | `{"score": {"$lt": 0.9}}` |
| `$lte` | Less or equal | `{"year": {"$lte": 2024}}` |
| `$and` | All conditions true | `{"$and": [c1, c2]}` |
| `$or` | Any condition true | `{"$or": [c1, c2]}` |

Dot-notation field paths work: `"meta.source"` matches `{"meta": {"source": "blog"}}`.

Missing fields and type mismatches return `false` (vector excluded from results), never errors.

### Filter examples

```json
{"category": {"$eq": "tech"}}
{"score": {"$gte": 0.8}}
{"status": {"$in": ["published", "featured"]}}
{"$and": [{"category": {"$eq": "tech"}}, {"year": {"$gte": 2024}}]}
{"$or":  [{"category": {"$eq": "tech"}}, {"category": {"$eq": "science"}}]}
{"meta.author": {"$eq": "alice"}}
```

---

## Index Types

| Index | Recall | Query complexity | Best for |
|-------|--------|-----------------|----------|
| `flat` | 100% | O(N · D) | < 100 K vectors, ground-truth eval |
| `hnsw` | ~95–99% (tunable) | O(log N · ef) | > 100 K vectors, latency-sensitive |
| `faiss` | depends on factory | SIMD-optimised | Large scale, GPU, compressed indexes |

### HNSW tuning

```json
{
  "index_type": "hnsw",
  "hnsw": {
    "ef_construction": 200,
    "ef_search": 50,
    "m": 12
  }
}
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `ef_construction` | 200 | Graph quality during build. Higher → better recall, slower build |
| `ef_search` | 50 | Beam width during query. Higher → better recall, slower query |
| `m` | 12 | Edges per node. Higher → better recall, more memory |

### FAISS integration

FAISS is enabled via the `faiss` Cargo feature. The server must be compiled
with that feature for `"index_type": "faiss"` to work — otherwise the server
returns HTTP 400.

```bash
# Build server with FAISS
cargo build --release --features vectordb-server/faiss
```

Create a FAISS collection by specifying a **factory string** that follows the
[FAISS index factory grammar](https://github.com/facebookresearch/faiss/wiki/The-index-factory):

```bash
# Exact flat search (SIMD-accelerated, same recall as "flat" but faster)
curl -X POST http://localhost:8080/collections/exact \
  -H "Content-Type: application/json" \
  -d '{"dimensions": 1536, "metric": "cosine", "index_type": "faiss", "faiss_factory": "Flat"}'

# IVF — faster approximate search at scale (>500 K vectors)
# Needs ≥ 39 × nlist training vectors before useful results
curl -X POST http://localhost:8080/collections/large \
  -H "Content-Type: application/json" \
  -d '{"dimensions": 1536, "metric": "cosine", "index_type": "faiss", "faiss_factory": "IVF4096,Flat"}'

# IVF + Product Quantisation — compressed index (~32× smaller)
curl -X POST http://localhost:8080/collections/compressed \
  -H "Content-Type: application/json" \
  -d '{"dimensions": 1536, "metric": "cosine", "index_type": "faiss", "faiss_factory": "IVF256,PQ64"}'

# FAISS HNSW (different from the built-in hnsw index)
curl -X POST http://localhost:8080/collections/graph \
  -H "Content-Type: application/json" \
  -d '{"dimensions": 1536, "metric": "l2", "index_type": "faiss", "faiss_factory": "HNSW32"}'
```

**Common factory strings:**

| String | Type | Notes |
|--------|------|-------|
| `"Flat"` | Exact | SIMD-accelerated, 100% recall |
| `"IVF1024,Flat"` | IVF | Good for 100 K–10 M vectors |
| `"IVF4096,Flat"` | IVF | Better for >1 M vectors |
| `"IVF256,PQ64"` | IVF + PQ | 64-byte codes, ~32× smaller |
| `"HNSW32"` | Graph | Fast ANN, no training needed |
| `"IVF4096,SQ8"` | IVF + SQ | Scalar quantiser, 8-bit |

**How FAISS fits the architecture:**

```
Collection::upsert() / delete()
    │
    ▼
FaissIndex (HashMap<u64, Vec<f32>> = source of truth)
    │                │
    │         Mutex<FaissState>
    │              │
    │       FAISS IndexImpl  ← rebuilt lazily on first search after
    │                            any mutation, or eagerly on flush()
    │
Collection::load() → WAL replay → index.flush() → FAISS built once
```

---

## Distance Metrics

| Metric | Formula | Best for |
|--------|---------|----------|
| `l2` | `‖a − b‖₂` | Absolute position matters (coordinates, pixel embeddings) |
| `cosine` | `1 − (a·b) / (‖a‖ ‖b‖)` | Direction matters, magnitude doesn't (NLP embeddings) |
| `dot_product` | `−(a · b)` | Pre-normalised vectors, max-inner-product search |

All distance values follow the convention **lower = more similar** (distance 0.0 = identical).

---

## Persistence & Durability

vectordb uses a **write-ahead log (WAL)** — every `upsert` and `delete` is
appended to an NDJSON log file before the in-memory index is updated.

```
./data/
  articles/
    meta.json   ← collection config (dimensions, metric, index type, faiss_factory)
    wal.log     ← append-only NDJSON journal
```

- **Crash-safe** — partial final entries are silently skipped on replay.
- **Automatic** — no `save()` or `flush()` call required.
- **Compaction** — WAL is rewritten to contain only live entries when it exceeds 50 000 lines (atomic rename, crash-safe).

---

## REST API Reference

All endpoints consume and produce `application/json`.

### Collections

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/collections` | List all collection names |
| `POST` | `/collections/:name` | Create a collection |
| `GET` | `/collections/:name` | Get collection info (count, dimensions, metric) |
| `DELETE` | `/collections/:name` | Delete a collection and all its data |

**Create collection body:**

```jsonc
{
  "dimensions": 1536,         // required
  "metric": "cosine",         // "l2" | "cosine" | "dot_product"  (default: "cosine")
  "index_type": "hnsw",       // "flat" | "hnsw" | "faiss"        (default: "hnsw")
  "hnsw": {                   // only for index_type = "hnsw"
    "ef_construction": 200,
    "ef_search": 50,
    "m": 12
  },
  "faiss_factory": "IVF1024,Flat",  // only for index_type = "faiss"
  "auto_promote_threshold": 10000,  // auto-promote flat → hnsw at N vectors
  "embedding_model": "openai/text-embedding-3-small"  // optional
}
```

### Vectors

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/collections/:name/vectors` | Upsert a vector |
| `POST` | `/collections/:name/search` | kNN search |
| `DELETE` | `/collections/:name/vectors/:id` | Delete a vector |

**Upsert body:**
```json
{"id": 42, "vector": [0.1, 0.2, 0.3], "payload": {"key": "value"}}
```

**Search body:**
```json
{"vector": [0.1, 0.2, 0.3], "k": 10, "filter": {"category": {"$eq": "tech"}}}
```

**Search response:**
```json
{"results": [{"id": 1, "distance": 0.012, "payload": {"category": "tech"}}]}
```

### Embedding endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/collections/:name/embed-upsert` | Embed text then upsert |
| `POST` | `/collections/:name/embed-search` | Embed text then search |

These require `embedding_model` to be set on the collection.
Supported prefixes: `"openai/<model>"` (needs `OPENAI_API_KEY`) and
`"ollama/<model>"` (needs `OLLAMA_BASE_URL`, default `http://localhost:11434`).

---

## Architecture

```
                    ┌────────────────────────────────────────────────┐
                    │              vectordb-core                     │
                    │                                                │
                    │  Collection  — index + WAL + payload store     │
                    │  CollectionManager / IndexRegistry             │
                    │  WAL         — NDJSON append-only log          │
                    │  FilterCondition — payload predicates          │
                    │                                                │
                    │  VectorIndex trait                             │
                    │    ├── FlatIndex   (exact, brute-force)        │
                    │    ├── HnswIndex   (ANN, graph)                │
                    │    └── FaissIndex  (FAISS, optional feature)   │
                    └──────────────────┬─────────────────────────────┘
                                       │
               ┌───────────────────────┴──────────────────────────┐
               │                                                  │
  ┌────────────▼────────────┐              ┌─────────────────────▼──────────┐
  │  vectordb-python        │              │  vectordb-server                │
  │  (PyO3 / maturin)       │              │  (Axum + Tokio)                 │
  │                         │              │                                 │
  │  Client(path=...)       │              │  REST API :8080                 │
  │  Collection             │              │  VECTORDB_DATA_DIR              │
  │  FlatIndex (low-level)  │              │  VECTORDB_API_KEY (optional)    │
  │  HnswIndex (low-level)  │              │  embed-upsert / embed-search    │
  └─────────────────────────┘              └─────────────────────────────────┘
      Embedded Python                           Client-Server
      (in-process, zero network)                (any language via HTTP)

                                         ┌──────────────────────────┐
                                         │  sdks/typescript         │
                                         │  VectorDbClient          │
                                         └──────────────────────────┘

                                         ┌──────────────────────────┐
                                         │  vectordb-cli  (vdb)     │
                                         └──────────────────────────┘
```

| Crate / package | Role |
|-----------------|------|
| `vectordb-core` | Index trait, FlatIndex, HnswIndex, FaissIndex, WAL, Collection, CollectionManager, IndexRegistry, payload filtering |
| `vectordb-server` | Axum HTTP server — REST API, auth, embedding endpoints |
| `vectordb-cli` | `vdb` CLI binary |
| `vectordb-embeddings` | OpenAI and Ollama embedding providers |
| `vectordb-python` | PyO3 bindings — `Client`, `Collection`, `FlatIndex`, `HnswIndex` |
| `sdks/typescript` | TypeScript HTTP client SDK |

---

## Development Build Script

`dev_build.sh` is a convenience wrapper around `cargo`:

```
Usage:
  ./dev_build.sh                  # build + test everything (no FAISS)
  ./dev_build.sh --faiss          # build + test with FAISS support
  ./dev_build.sh --server         # build + start the HTTP server
  ./dev_build.sh --server --faiss # build + start server with FAISS
  ./dev_build.sh --python         # build Python wheel + smoke-test
  ./dev_build.sh --release        # release-mode build (no tests)
  ./dev_build.sh --help           # show this message
```

Environment variables honoured by the script (and by the server):

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTORDB_DATA_DIR` | `./data` | Directory for persistent data |
| `VECTORDB_API_KEY` | *(unset)* | Enable Bearer-token auth |
| `RUST_LOG` | `info` | Log filter (e.g. `debug`, `vectordb_server=trace`) |

---

## Building on macOS

### Apple Silicon — universal wheel

```bash
rustup target add x86_64-apple-darwin aarch64-apple-darwin
maturin build --release --target universal2-apple-darwin
```

### Known issues

#### `cargo build` fails with "Undefined symbols for architecture arm64"

PyO3 extension modules leave Python symbols unresolved at link time by design.
Use `maturin develop` instead of `cargo build` for the Python bindings crate.
The repo ships `.cargo/config.toml` with `-undefined dynamic_lookup` for macOS.

#### Python not found / wrong version

```bash
export PYO3_PYTHON=$(which python3)
maturin develop
```

#### pyenv Python missing shared library

```bash
PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.11.9
pyenv global 3.11.9
```

---

## License

Apache 2.0
