# vectordb

A high-performance vector database @ Scale, written in Rust.

---

## Table of Contents

- [Overview](#overview)
- [Use Cases](#use-cases)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [REST API](#rest-api)
- [CLI](#cli)
- [Index Types](#index-types)
- [Distance Metrics](#distance-metrics)
- [Python Bindings](#python-bindings)
- [Development Setup](#development-setup)
- [Building on macOS](#building-on-macos)

---

## Overview

**vectordb** stores, indexes, and searches high-dimensional vectors at scale.
It exposes a simple HTTP/JSON API (backed by [Axum](https://github.com/tokio-rs/axum))
and ships a CLI (`vdb`) for quick interaction. The core engine is a pluggable
`VectorIndex` trait with two built-in implementations — an exact `FlatIndex`
and an approximate HNSW index powered by [instant-distance](https://github.com/InstantDomainSearch/instant-distance).

---

## Use Cases

### 1. Semantic Search
Embed documents, web pages, or knowledge-base articles with a text embedding
model (e.g. `text-embedding-3-small`). Store the embeddings in vectordb and
retrieve the most semantically relevant results for any natural-language query —
no keyword matching required.

```
User query → embedding model → query vector
→ vectordb search (top-k) → relevant documents
```

### 2. Recommendation Systems
Convert user behaviour (clicks, purchases, ratings) or item attributes into
latent vectors and find the nearest neighbours to power "you may also like"
or "users like you also bought" features.

### 3. Image & Multimodal Search
Store CLIP or similar vision embeddings for images, videos, or audio clips.
Search by uploading a new image or text description to find visually or
semantically similar media instantly.

### 4. Retrieval-Augmented Generation (RAG)
Pair vectordb with a large language model to build RAG pipelines. At query
time, retrieve the most relevant context chunks from your corpus and inject
them into the LLM prompt — grounding responses in up-to-date or proprietary
knowledge.

```
User question
 → embed → vectordb kNN search → top-k context chunks
 → LLM prompt + context → grounded answer
```

### 5. Anomaly & Fraud Detection
Embed transactional or behavioural events and flag items whose nearest-neighbour
distance exceeds a threshold as potential anomalies — useful for fraud
detection, intrusion detection, and quality control.

### 6. Duplicate & Near-Duplicate Detection
Hash or embed content (text, code, images) and perform a similarity search to
surface near-duplicates before ingesting them, enabling deduplication pipelines
at scale.

### 7. Clustering & Exploratory Data Analysis
Use vectordb as a fast kNN oracle for clustering algorithms (k-means, DBSCAN,
HDBSCAN) over large embedding datasets without loading everything into memory.

### 8. Drug Discovery & Molecular Search
Store molecular fingerprint vectors and retrieve structurally similar compounds
— accelerating virtual screening and lead optimisation workflows.

---

## Architecture

```
┌────────────────────────────────────────┐
│            vectordb-server             │  HTTP/JSON (port 8080)
│              (Axum + Tokio)            │
└────────────────┬───────────────────────┘
                 │ calls
┌────────────────▼───────────────────────┐
│            vectordb-core               │
│  ┌──────────────┐  ┌────────────────┐  │
│  │  FlatIndex   │  │   HnswIndex    │  │
│  │  (exact L2/  │  │  (ANN, ~95-99% │  │
│  │  cosine/dot) │  │   recall)      │  │
│  └──────────────┘  └────────────────┘  │
│         implements VectorIndex trait   │
└────────────────────────────────────────┘
         ▲
┌────────┴───────┐
│  vectordb-cli  │  `vdb` binary — talks to the server over HTTP
└────────────────┘
```

| Crate              | Role |
|--------------------|------|
| `vectordb-core`    | Index trait, FlatIndex, HnswIndex, distance metrics |
| `vectordb-server`  | REST API server (collections, upsert, search, delete) |
| `vectordb-cli`     | `vdb` command-line client |
| `vectordb-python`  | PyO3 Python bindings (`FlatIndex`, `HnswIndex`) |

---

## Getting Started

### Build

```bash
cargo build --release
```

Binaries land in `target/release/`:
- `vectordb-server` — the HTTP server
- `vdb` — the CLI client

### Run the server

```bash
./target/release/vectordb-server
# INFO vectordb-server listening on 0.0.0.0:8080
```

Set `RUST_LOG=debug` for verbose output.

---

## Python Bindings

The `vectordb-python` crate exposes `FlatIndex` and `HnswIndex` directly to Python via [PyO3](https://pyo3.rs).

### Install

```bash
python3 -m venv .venv && source .venv/bin/activate
maturin develop --release   # builds and installs into the active venv
```

### Usage

```python
import vectordb

# --- FlatIndex (exact search) ---
idx = vectordb.FlatIndex(dimensions=3, metric="cosine")
idx.add(1, [1.0, 0.0, 0.0])
idx.add(2, [0.0, 1.0, 0.0])
idx.add_batch([(3, [0.0, 0.0, 1.0]), (4, [1.0, 1.0, 0.0])])

results = idx.search([1.0, 0.0, 0.0], k=2)
# [{"id": 1, "distance": 0.0}, {"id": 4, "distance": ...}]

idx.delete(4)
print(len(idx))          # 3
print(idx.dimensions)    # 3
print(idx.metric)        # "cosine"

# Persist to disk
idx.save("my_index.json")
idx2 = vectordb.FlatIndex.load("my_index.json")

# --- HnswIndex (approximate search) ---
hnsw = vectordb.HnswIndex(dimensions=128, metric="l2", ef_construction=200, ef_search=50, m=12)
hnsw.add_batch([(i, [float(i)] * 128) for i in range(10_000)])
hnsw.flush()   # build the HNSW graph

results = hnsw.search([0.0] * 128, k=5)
hnsw.save("hnsw.json")
hnsw2 = vectordb.HnswIndex.load("hnsw.json")  # graph rebuilt automatically
```

### API reference

| Class | Method / property | Description |
|-------|-------------------|-------------|
| `FlatIndex(dimensions, metric="l2")` | constructor | Create exact index |
| `HnswIndex(dimensions, metric="l2", ef_construction=200, ef_search=50, m=12)` | constructor | Create ANN index |
| both | `.add(id, vector)` | Insert one vector |
| both | `.add_batch([(id, vector), ...])` | Insert many vectors |
| both | `.search(query, k)` → `list[dict]` | Return k nearest neighbours |
| both | `.delete(id)` → `bool` | Remove a vector |
| both | `.save(path)` | Persist index to JSON |
| both | `cls.load(path)` | Restore index from JSON |
| both | `len(idx)` | Number of stored vectors |
| both | `.dimensions`, `.metric` | Read-only properties |
| `HnswIndex` | `.flush()` | Rebuild HNSW graph immediately |

Metrics: `"l2"`, `"cosine"`, `"dot_product"`.

---

## Development Setup

### 1. Prerequisites

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Python 3.8+ and maturin (only needed for Python bindings)
brew install python          # macOS; use apt/dnf on Linux
pip install maturin
```

### 2. Clone and branch

```bash
git clone https://github.com/vectordb/vectordb.git
cd vectordb
git checkout -b feat/my-change origin/main   # or an existing branch
```

### 3. Python virtual environment

Skip if you only need the Rust server/CLI.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install maturin
```

### 4. Build

```bash
# Rust (server + CLI)
cargo build

# Python bindings (venv must be active)
maturin develop
python3 -c "import vectordb; print('OK')"   # verify
```

> **macOS:** if `cargo build` fails with undefined Python symbols, see
> [Building on macOS](#building-on-macos).

### 5. Test and run

```bash
cargo test                                   # run all tests
RUST_LOG=debug cargo run -p vectordb-server  # start dev server
./target/debug/vdb list                      # use the CLI
```

### Project structure

```
crates/
├── vectordb-core/     # index trait, FlatIndex, HnswIndex
├── vectordb-server/   # Axum HTTP server
├── vectordb-cli/      # vdb CLI binary
└── vectordb-python/   # PyO3 Python bindings
```

---

## REST API

All requests and responses use `application/json`.

### Collections

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/collections` | List all collections |
| `POST` | `/collections/:name` | Create a collection |
| `GET` | `/collections/:name` | Get collection info |
| `DELETE` | `/collections/:name` | Delete a collection |

**Create collection body:**
```json
{
  "dimensions": 1536,
  "metric": "cosine",
  "index_type": "hnsw",
  "hnsw": {
    "ef_construction": 200,
    "ef_search": 50,
    "m": 12
  }
}
```

`metric`: `"l2"` | `"cosine"` | `"dot_product"`
`index_type`: `"flat"` | `"hnsw"` (default: `"hnsw"`)

### Vectors

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/collections/:name/vectors` | Upsert a vector |
| `POST` | `/collections/:name/search` | kNN search |
| `DELETE` | `/collections/:name/vectors/:id` | Delete a vector |

**Upsert body:**
```json
{ "id": 42, "vector": [0.1, 0.2, 0.3, ...] }
```

**Search body:**
```json
{ "vector": [0.1, 0.2, 0.3, ...], "k": 10 }
```

**Search response:**
```json
{
  "results": [
    { "id": 7,  "distance": 0.012 },
    { "id": 99, "distance": 0.034 }
  ]
}
```

---

## CLI

`vdb` is the command-line client for a running `vectordb-server`.

### Build and install

```bash
cargo build --release
# binary lands at target/release/vdb
```

Add `target/release` to your `$PATH`, or run it as `./target/release/vdb`.

### Server URL

By default `vdb` connects to `http://localhost:8080`. Override with `--host` or the `VDB_HOST` environment variable:

```bash
vdb --host http://prod-server:8080 list
VDB_HOST=http://prod-server:8080 vdb list
```

### Commands

#### `vdb list`
List all collections.
```bash
vdb list
```

#### `vdb create <name>`
Create a new collection.

| Flag | Default | Description |
|------|---------|-------------|
| `--dimensions <N>` | *(required)* | Number of dimensions |
| `--metric <m>` | `cosine` | `l2` \| `cosine` \| `dot_product` |
| `--index <type>` | `hnsw` | `flat` \| `hnsw` |

```bash
vdb create my-docs --dimensions 768
vdb create my-docs --dimensions 768 --metric cosine --index hnsw
vdb create exact-idx --dimensions 128 --metric l2 --index flat
```

#### `vdb insert <collection>`
Insert or update a vector (comma-separated floats).

```bash
vdb insert my-docs --id 1 --vector "0.1,0.2,0.3"
vdb insert my-docs --id 42 --vector "0.9,0.1,0.5,0.3"
```

#### `vdb search <collection>`
Search for nearest neighbours. Prints JSON results.

| Flag | Default | Description |
|------|---------|-------------|
| `--vector <floats>` | *(required)* | Comma-separated query vector |
| `--k <N>` | `5` | Number of neighbours to return |

```bash
vdb search my-docs --vector "0.1,0.2,0.3"
vdb search my-docs --vector "0.1,0.2,0.3" --k 10
```

Output:
```json
{
  "results": [
    { "id": 1,  "distance": 0.0 },
    { "id": 99, "distance": 0.042 }
  ]
}
```

#### `vdb delete <collection>`
Delete a single vector by ID.
```bash
vdb delete my-docs --id 42
```

#### `vdb drop <collection>`
Delete an entire collection.
```bash
vdb drop my-docs
```

### End-to-end example

```bash
# Start the server
./target/release/vectordb-server &

# Create a 3-dimensional collection
vdb create colours --dimensions 3 --metric l2 --index flat

# Insert some vectors
vdb insert colours --id 1 --vector "1.0,0.0,0.0"   # red
vdb insert colours --id 2 --vector "0.0,1.0,0.0"   # green
vdb insert colours --id 3 --vector "0.0,0.0,1.0"   # blue

# Search for the 2 closest to "mostly red"
vdb search colours --vector "0.9,0.1,0.0" --k 2

# Remove a vector and the collection
vdb delete colours --id 3
vdb drop colours
```

---

## Index Types

| Index | Recall | Query complexity | Build complexity | Use when |
|-------|--------|------------------|------------------|----------|
| `flat` | 100% | O(N · D) | O(N) | < 100 K vectors, ground-truth eval |
| `hnsw` | ~95–99% (tunable) | O(log N · ef) | O(N · M · log N) | > 100 K vectors, latency-sensitive |

Both index types support disk persistence via `save` / `load` (JSON format). For `HnswIndex`, the graph is not stored — it is rebuilt automatically on load, so the loaded index is immediately ready for ANN search.

### HNSW tuning

| Parameter | Default | Effect |
|-----------|---------|--------|
| `ef_construction` | 200 | Graph quality during build. Higher → better recall, slower build. |
| `ef_search` | 50 | Beam width at query time. Higher → better recall, slower query. |
| `m` | 12 | Edges per node. Higher → better recall, more memory. |

---

## Distance Metrics

| Metric | Formula | Best for |
|--------|---------|----------|
| `l2` | `‖a − b‖₂` | Absolute position matters (coordinates, pixel embeddings) |
| `cosine` | `1 − (a·b) / (‖a‖ ‖b‖)` | Direction matters, magnitude doesn't (NLP embeddings) |
| `dot_product` | `−(a · b)` | Pre-normalised vectors, max-inner-product search |

---

## Building on macOS

This section covers macOS-specific setup and known issues developers encounter
when building the project on Apple hardware (both Intel and Apple Silicon).

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install Python 3.8+ (Homebrew recommended)
brew install python

# Install maturin (required for the Python bindings crate)
pip install maturin
```

### Building the Rust workspace

```bash
cargo build --release
```

### Building the Python bindings

The `vectordb-python` crate is a PyO3 extension module and **must be built
with `maturin`**, not plain `cargo build`:

```bash
# Dev install into your active virtual environment (fast iteration)
maturin develop

# Build a release wheel
maturin build --release
pip install target/wheels/vectordb-*.whl
```

### Apple Silicon (M1/M2/M3) — universal wheel

```bash
rustup target add x86_64-apple-darwin aarch64-apple-darwin
maturin build --release --target universal2-apple-darwin
```

---

### Known issues

#### 1. `cargo build` fails with "Undefined symbols for architecture arm64"

**Symptom:**

```
Undefined symbols for architecture arm64:
  "_PyBaseObject_Type", referenced from: ...
  "_PyBytes_AsString", referenced from: ...
  ...
ld: symbol(s) not found for architecture arm64
```

**Cause:** PyO3 extension modules intentionally leave Python symbols
(`_Py*`) unresolved at link time. They are resolved at runtime when
Python loads the `.so`. Plain `cargo build` does not pass the required
`-undefined dynamic_lookup` linker flag, so the link step fails.

**Fix:** The repository ships a `.cargo/config.toml` that adds this
flag automatically for both `aarch64-apple-darwin` and
`x86_64-apple-darwin` targets. If you still see the error, ensure the
file is present:

```toml
# .cargo/config.toml
[target.aarch64-apple-darwin]
rustflags = ["-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"]

[target.x86_64-apple-darwin]
rustflags = ["-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"]
```

> For building the Python extension the correct tool is always
> `maturin develop` — it sets this flag automatically regardless of
> `.cargo/config.toml`.

---

#### 2. Python not found / wrong Python picked up

**Symptom:** PyO3's build script prints `error: Python version X.Y is
not supported` or links against the wrong interpreter.

**Fix:** Point PyO3 at your chosen Python explicitly:

```bash
export PYO3_PYTHON=$(which python3)
maturin develop
```

---

#### 3. pyenv Python missing shared library

**Symptom:** Linker error referencing `libpythonX.Y.dylib` not found,
or PyO3 build script warning `could not find Python shared library`.

**Cause:** pyenv builds Python without a shared library by default.

**Fix:** Reinstall Python with shared-library support enabled:

```bash
PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.11.9
pyenv global 3.11.9
```

---

#### 4. `zsh: unknown sort specifier` when running Python commands

**Symptom:** zsh prints `zsh: unknown sort specifier` after certain
`python3 -c` invocations.

**Cause:** This is a zsh glob-expansion side effect when the Python
output contains characters zsh tries to interpret. It does not affect
the build — wrap the command in quotes or run it inside a subshell to
suppress it.

---

#### 5. Build fails after Xcode / Command Line Tools update

**Symptom:** After updating macOS or Xcode, `cargo build` or `maturin`
fails with linker errors unrelated to Python.

**Fix:** Reinstall the Command Line Tools:

```bash
sudo xcode-select --reset
xcode-select --install
```

Then re-run `rustup show` to confirm the active toolchain is still
targeting `aarch64-apple-darwin` or `x86_64-apple-darwin` as expected.

---

## License

Apache 2.0
