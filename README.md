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

```bash
# List collections
vdb list

# Create a 768-dimensional cosine collection backed by HNSW
vdb create my-docs --dimensions 768 --metric cosine --index hnsw

# Insert a vector
vdb insert my-docs --id 1 --vector "0.1,0.2,0.3,..."

# Search for top-5 neighbours
vdb search my-docs --vector "0.1,0.2,0.3,..." --k 5

# Delete a vector
vdb delete my-docs --id 1

# Delete a collection
vdb drop my-docs
```

Override the server URL with `--host` or the `VDB_HOST` environment variable:

```bash
VDB_HOST=http://my-server:8080 vdb list
```

---

## Index Types

| Index | Recall | Query complexity | Build complexity | Use when |
|-------|--------|------------------|------------------|----------|
| `flat` | 100% | O(N · D) | O(N) | < 100 K vectors, ground-truth eval |
| `hnsw` | ~95–99% (tunable) | O(log N · ef) | O(N · M · log N) | > 100 K vectors, latency-sensitive |

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
