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

## License

MIT
