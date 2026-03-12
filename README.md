# Quiver

An embedded vector database written in Rust with a Python SDK.

Runs **fully in-process** — no server, no network, no extra processes. Open a path and go.

## Features

- Multiple index types: Flat (exact), HNSW (approximate), QuantizedFlat (int8), IVF, MmapFlat, FAISS (optional)
- Distance metrics: Cosine, L2 (Euclidean), Dot Product
- Payload metadata with rich filter operators (`$eq`, `$ne`, `$in`, `$gt`, `$gte`, `$lt`, `$lte`, `$and`, `$or`)
- Persistent storage via write-ahead log (WAL) or pure in-memory usage
- Python SDK via PyO3/maturin

## Quick start (Python)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release -m crates/quiver-python/Cargo.toml
```

```python
import quiver

db  = quiver.Client(path="./my_data")          # persists to disk
col = db.create_collection("docs", dimensions=384, metric="cosine")

col.upsert(id=1, vector=[...], payload={"text": "Hello world"})
hits = col.search(query=[...], k=5)
```

See [embedded_quiver_usage.md](embedded_quiver_usage.md) for full documentation.

## Build

```bash
# Rust tests
./dev_build.sh

# Python wheel
./dev_build.sh --python

# With FAISS support
./dev_build.sh --faiss --python
```

## Crates

| Crate | Purpose |
|-------|---------|
| `quiver-core` | Embedded vector database engine (indexes, WAL, filtering) |
| `quiver-python` | Python bindings (PyO3/maturin) |

## License

MIT
