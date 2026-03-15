# Quiver — Design & Performance Internals

This document explains how Quiver achieves high performance across all index types. Every optimization is described with the reasoning behind it, the before/after impact, and simplified diagrams.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [SIMD Distance Kernels](#simd-distance-kernels)
3. [Flat Index: Contiguous Memory Layout](#flat-index-contiguous-memory-layout)
4. [HNSW: Graph Storage & Traversal](#hnsw-graph-storage--traversal)
5. [IVF: GEMM-Based Centroid Assignment](#ivf-gemm-based-centroid-assignment)
6. [Quantization: Int8, FP16, Binary](#quantization-int8-fp16-binary)
7. [Product Quantization (IVF-PQ)](#product-quantization-ivf-pq)
8. [Memory-Mapped Index](#memory-mapped-index)
9. [Python ↔ Rust Data Transfer](#python--rust-data-transfer)
10. [Summary of All Optimizations](#summary-of-all-optimizations)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Python Layer                           │
│   quiver_vector_db (PyO3 bindings, type stubs, BM25, etc.)  │
├─────────────────────────────────────────────────────────────┤
│                      Rust Core                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │Collection│ │    DB    │ │ Manager  │ │   WAL    │       │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │
│       │             │            │             │             │
│  ┌────▼─────────────▼────────────▼─────────────▼──────────┐ │
│  │                  Index Layer                            │ │
│  │  Flat │ HNSW │ Int8 │ FP16 │ IVF │ IVF-PQ │ Mmap │ Bin│ │
│  └───────────────────┬────────────────────────────────────┘ │
│                      │                                      │
│  ┌───────────────────▼────────────────────────────────────┐ │
│  │              Distance Kernels (SIMD)                   │ │
│  │  AVX2+FMA (x86-64) │ NEON (AArch64) │ Scalar fallback │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

All eight index types share the same SIMD-accelerated distance kernels. The `VectorIndex` trait defines a common interface, so the collection layer can swap index types without changing any other code.

---

## SIMD Distance Kernels

### The Problem

Distance calculation (L2, cosine, dot product) is the innermost loop of every vector search. For a 128-dimensional vector, each distance call does 128 multiplications and 128 additions. Brute-force search over 10K vectors = 10K × 256 FLOPs = 2.5 million operations per query.

### The Solution: Runtime CPU Dispatch

Quiver detects CPU features at runtime and picks the fastest kernel:

```
                    ┌──────────────────┐
                    │  distance(a, b)  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  CPU detection   │
                    └────────┬─────────┘
                 ┌───────────┼───────────┐
                 ▼           ▼           ▼
          ┌──────────┐ ┌──────────┐ ┌──────────┐
          │ AVX2+FMA │ │   NEON   │ │  Scalar  │
          │ (x86-64) │ │ (AArch64)│ │ fallback │
          │ 8 floats │ │ 4 floats │ │ 1 float  │
          │ per cycle│ │ per cycle│ │ per cycle│
          └──────────┘ └──────────┘ └──────────┘
```

No compile flags needed. The same binary runs optimally on any CPU.

### Fused Multiply-Add (FMA)

The key instruction is FMA: `a * b + c` in a single cycle instead of two.

For L2 distance:
```
Standard:   diff = a - b       (1 cycle)
            sq   = diff * diff (1 cycle)
            acc  = acc + sq    (1 cycle)  ← 3 cycles total

FMA:        diff = a - b                  (1 cycle)
            acc  = fma(diff, diff, acc)   (1 cycle)  ← 2 cycles total
```

With AVX2, each instruction processes **8 floats** simultaneously. So one FMA instruction does `8 × (diff² + acc)` in a single cycle.

### Batch-4: Instruction-Level Parallelism

Modern CPUs have multiple FMA execution units (typically 2). If we only compute one distance at a time, the second unit sits idle. The fix: compute **4 distances in parallel** using independent accumulators.

```
Standard sequential:          Batch-4 ILP:

┌────────────┐                ┌────────────┬────────────┬────────────┬────────────┐
│ distance 1 │ FMA ──→ acc0   │ FMA→acc0   │ FMA→acc1   │ FMA→acc2   │ FMA→acc3   │
│ distance 2 │ FMA ──→ acc0   │ FMA→acc0   │ FMA→acc1   │ FMA→acc2   │ FMA→acc3   │
│ distance 3 │ FMA ──→ acc0   │ FMA→acc0   │ FMA→acc1   │ FMA→acc2   │ FMA→acc3   │
│ distance 4 │ FMA ──→ acc0   │            │            │            │            │
│ ...32 iters│                │ ...8 iters │            │            │            │
└────────────┘                └────────────┴────────────┴────────────┴────────────┘
 Total: 32 cycles              Total: 8 cycles (4× faster)
```

Because `acc0`, `acc1`, `acc2`, `acc3` are independent, the CPU can execute all 4 FMA instructions in a single cycle using both execution units and pipelining. This eliminates the data-dependency stall where each iteration waits for the previous result.

**Where it's used:** HNSW neighbor scanning, IVF centroid search, flat brute-force search.

---

## Flat Index: Contiguous Memory Layout

### The Problem (Before)

The original FlatIndex stored vectors in a `HashMap<u64, Vec<f32>>`:

```
HashMap (hash table)
  ┌────────────────────────────────────────┐
  │ bucket 0 → id=42 → Vec<f32> at 0xA000 │ ← heap allocation
  │ bucket 1 → (empty)                     │
  │ bucket 2 → id=7  → Vec<f32> at 0xF100 │ ← different heap page
  │ bucket 3 → id=99 → Vec<f32> at 0x3200 │ ← another page
  │ ...                                    │
  └────────────────────────────────────────┘
```

Problems:
- Each vector is a separate heap allocation → memory fragmentation
- Sequential scan jumps between random memory addresses → **cache misses**
- Hash lookup overhead (~8-16 cycles) per access

### The Solution (After)

All vectors live in a single contiguous `Vec<f32>`:

```
data: Vec<f32>   (one allocation, cache-friendly)
┌───────────────┬───────────────┬───────────────┬───────────────┐
│ vec0 (128 f32)│ vec1 (128 f32)│ vec2 (128 f32)│ vec3 (128 f32)│
└───────────────┴───────────────┴───────────────┴───────────────┘
  slot 0          slot 1          slot 2          slot 3

ids:   [ 42,  7, 99, 15 ]     ← ID per slot
alive: [  T,  T,  T,  T ]     ← soft-delete marker
id_to_slot: { 42→0, 7→1, 99→2, 15→3 }   ← for delete/update
```

Benefits:
- **Sequential scan** reads memory in order → CPU prefetcher kicks in
- **Single allocation** for all vectors → no fragmentation
- **Batch insert** = `extend_from_slice()` = single `memcpy`

### Soft-Delete

Deleting a vector doesn't move data. It just marks `alive[slot] = false`:

```
Before delete(id=7):    alive = [ T, T, T, T ]
After  delete(id=7):    alive = [ T, F, T, T ]
                                     ↑ skipped during search
```

The data compacts only on explicit `flush()`. This avoids expensive reallocation during mixed insert/delete workloads.

### Result

**Insert throughput: 1.4M → 28.3M vec/s (20× speedup)**

---

## HNSW: Graph Storage & Traversal

### How HNSW Works

HNSW (Hierarchical Navigable Small World) builds a multi-layer graph where higher layers have fewer nodes and longer-range connections. Search starts at the top layer and "descends" to the bottom.

```
Layer 2:    [3] ─────────────────────── [47]         (few nodes, long edges)
             │                            │
Layer 1:    [3] ──── [12] ──── [28] ── [47]          (more nodes)
             │        │         │        │
Layer 0:    [3]─[5]─[12]─[15]─[28]─[33]─[47]─[51]   (all nodes, short edges)
```

### Optimization 1: Flat Vector Storage

All vectors are stored in a single contiguous buffer, indexed by node ID:

```
vectors: Vec<f32>
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│ node 0  │ node 1  │ node 2  │ node 3  │ node 4  │
│ 128 f32 │ 128 f32 │ 128 f32 │ 128 f32 │ 128 f32 │
└─────────┴─────────┴─────────┴─────────┴─────────┘

Access node i: vectors[i * dim .. (i+1) * dim]   ← O(1), zero indirection
```

During graph traversal, the CPU accesses neighbor vectors in quick succession. Contiguous storage means these accesses often hit the L1/L2 cache.

### Optimization 2: Layer-0 Flat Neighbor Storage

Layer 0 contains all nodes and is accessed on every search. Instead of `Vec<Vec<u32>>` (two levels of indirection), neighbors are stored in a flat buffer:

```
Before (Vec<Vec<u32>>):
  layer0[0] → Vec at 0xA000 → [3, 5, 12]
  layer0[1] → Vec at 0xB200 → [0, 2, 7]     ← 2 pointer dereferences
  layer0[2] → Vec at 0xC800 → [1, 5, 8, 12]

After (flat buffer, stride = m_max0):
  layer0: [3,5,12,0 | 0,2,7,0 | 1,5,8,12 | ...]
           ↑ node 0   ↑ node 1   ↑ node 2
  counts: [3, 3, 4, ...]

  Access node i's neighbors: layer0[i * m_max0 .. i * m_max0 + counts[i]]
```

One pointer dereference instead of two. The entire neighbor array fits in a few cache lines.

### Optimization 3: Generation-Counter Visited Set

During search, we need to track which nodes we've already visited to avoid revisiting them. The naive approach uses a `HashSet`:

```
Naive (HashSet):
  search 1: HashSet::new(), insert visited nodes, drop    ← allocate + free
  search 2: HashSet::new(), insert visited nodes, drop    ← allocate + free
  ...                                                      (thousands of times)

Generation counter:
  marks: [0, 0, 0, 0, 0, ...]   (one u32 per node, allocated once)
  gen: 0

  search 1: gen = 1.  Visit node 3 → marks[3] = 1.  Check node 5 → marks[5] != 1 → unvisited
  search 2: gen = 2.  All marks are now "stale" — no reset needed!
```

**O(1) reset** instead of allocating and freeing a HashSet for every search query.

### Optimization 4: Parallel Batch Insert

HNSW insertion requires both reading (searching for neighbors) and writing (linking edges). We use **phase separation** to avoid locks:

```
Phase 1: Sequential mini-batches (build initial graph structure)
  ┌────────────────────────────────────────┐
  │  Insert first N/batch nodes one-by-one │
  │  (ensures graph is well-connected)     │
  └────────────────────────────────────────┘

Phase 2: Parallel neighbor search (read-only graph access)
  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │ Thread 0│  │ Thread 1│  │ Thread 2│  │ Thread 3│
  │ search  │  │ search  │  │ search  │  │ search  │
  │ for node│  │ for node│  │ for node│  │ for node│
  │ A's nbrs│  │ B's nbrs│  │ C's nbrs│  │ D's nbrs│
  └─────────┘  └─────────┘  └─────────┘  └─────────┘
  (no writes to graph → no locks needed)

Phase 3: Sequential linking (write phase)
  ┌────────────────────────────────────────┐
  │  Link all found neighbors into graph   │
  │  (single-threaded, safe mutation)      │
  └────────────────────────────────────────┘
```

Each thread gets its own `VisitedTracker` — no shared state, no contention.

---

## IVF: GEMM-Based Centroid Assignment

### How IVF Works

IVF (Inverted File Index) partitions vectors into clusters using k-means. At search time, only the nearest clusters are scanned.

```
Training:
  vectors ──k-means──→  k centroids (cluster centers)

Insert:
  find nearest centroid → append to that cluster's posting list

Search:
  find nprobe nearest centroids → scan only those posting lists
```

### The Bottleneck: Centroid Assignment

During k-means training and batch insert, every vector must be compared against every centroid. With N=2000 vectors and k=32 centroids, that's 64,000 distance calculations per iteration.

### Optimization 1: GEMM Decomposition

The key insight: L2 distance can be decomposed into a matrix multiply.

```
||x_i - c_j||² = ||x_i||² + ||c_j||² - 2·dot(x_i, c_j)
                  ↑            ↑           ↑
              precompute   precompute   GEMM (single matmul)
              (N values)   (K values)   (N × K matrix)
```

Instead of computing N×K individual dot products, we compute **one matrix multiplication**:

```
               Centroids^T (D × K)
                ┌─────────────┐
                │             │
 Data (N × D)   │             │    =    IP (N × K)
┌──────────┐    │             │    ┌─────────────┐
│ x0  ...  │  × │ c0 c1 .. ck│    │ dot(x0,c0) .│
│ x1  ...  │    │             │    │ dot(x1,c0) .│
│ ...      │    │             │    │ ...         │
│ xn  ...  │    │             │    │ dot(xn,ck) .│
└──────────┘    └─────────────┘    └─────────────┘
```

### Why GEMM is Fast

The `matrixmultiply` crate uses **BLIS-style cache-blocked micro-kernels**:

```
┌───────────────────────────────────────────┐
│              Full Matrix (N × K)           │
│  ┌──────────┐                             │
│  │ L2 block │ ← fits in L2 cache (256KB) │
│  │ ┌──────┐ │                             │
│  │ │L1 blk│ │ ← fits in L1 cache (32KB)  │
│  │ │┌────┐│ │                             │
│  │ ││µker││ │ ← 8×4 register tile (AVX2) │
│  │ │└────┘│ │                             │
│  │ └──────┘ │                             │
│  └──────────┘                             │
└───────────────────────────────────────────┘
```

The matrix is split into blocks that fit in each cache level. The innermost "micro-kernel" keeps data in CPU registers for maximum throughput. On x86-64, this uses AVX2/FMA instructions. On AArch64, it uses NEON.

### Batch Insert Pipeline

After training, batch insert uses a 3-phase pipeline:

```
Phase 1: GEMM assignment (find nearest centroid for all vectors)
  ┌────────────────────────────────────────────┐
  │ matrixmultiply::sgemm(data, centroids^T)   │
  │ → N×K inner products in one call           │
  │ → assemble distances: ||x||² + ||c||² - 2·ip│
  │ → argmin per row → assignments[N]          │
  └────────────────────────────────────────────┘

Phase 2: Count & pre-allocate
  ┌────────────────────────────────────────────┐
  │ counts[c] = # vectors assigned to cluster c │
  │ posting_list[c].reserve(counts[c])          │
  └────────────────────────────────────────────┘

Phase 3: Scatter-append
  ┌────────────────────────────────────────────┐
  │ for each vector:                            │
  │   posting_list[assignment[i]].push(vec_i)   │
  │   (no reallocation — capacity pre-reserved) │
  └────────────────────────────────────────────┘
```

### Contiguous Posting Lists

Each posting list stores vectors contiguously (not as `Vec<Vec<f32>>`):

```
PostingList for cluster 0:
  ids:  [42, 7, 99, 15]
  data: [v42_f32s | v7_f32s | v99_f32s | v15_f32s]   ← contiguous
         ↑ one allocation, sequential scan
```

### Result

**IVF-Flat insert: 198K → 616K vec/s (3.1× speedup from GEMM)**

The remaining gap to faiss (~1.3M on Mac) is because faiss links to Apple's Accelerate framework, which uses the dedicated AMX matrix coprocessor (30 GFLOPS vs NEON's 3-5 GFLOPS). On Linux (where both use software GEMM), the gap is much smaller: Quiver 216K vs faiss 392K (1.8×).

---

## Quantization: Int8, FP16, Binary

### Int8 Scalar Quantization

Each float32 is mapped to an int8 value, reducing memory by 4×:

```
Original (f32):   [ 0.34,  -0.87,   0.12,  0.95, ... ]   ← 4 bytes each

Quantize:
  scale = 127 / max(|values|) = 127 / 0.95 = 133.7

Quantized (i8):   [  45,   -116,     16,   127, ... ]     ← 1 byte each
                    ↑ round(0.34 × 133.7) = 45

Dequantize for search:
  45 / 133.7 = 0.3366 ≈ 0.34  (error < 0.4%)
```

**Impact**: 4× memory reduction, ~99.4% recall preserved.

### FP16 Half-Precision

Each float32 is converted to float16 (IEEE 754 half-precision):

```
f32: [sign:1][exponent:8][mantissa:23]   = 4 bytes
f16: [sign:1][exponent:5][mantissa:10]   = 2 bytes
```

**Impact**: 2× memory reduction, >99.5% recall (better precision than Int8 for extreme values).

### Binary 1-Bit Quantization

Each dimension is reduced to a single bit (positive → 1, negative → 0):

```
Original:  [ 0.34,  -0.87,   0.12,  -0.05, 0.95, -0.23, 0.41, -0.68 ]
Binary:    [   1        0       1       0      1      0      1      0  ]
Packed:    [ 0b10101010 ]   ← 8 dimensions in 1 byte (32× compression)

Distance = Hamming(a XOR b).popcount()
  XOR:  1 cycle per 64 bits
  POPCNT: 1 cycle per 64 bits  (hardware instruction)
```

**Impact**: 32× compression, hardware-accelerated distance via `popcount`. Best used as a first-pass filter before re-ranking with full-precision vectors.

### Comparison

```
                Memory per vector (128-dim)
  f32:    ████████████████████████████████  512 bytes
  f16:    ████████████████                  256 bytes  (2×)
  int8:   ████████                          128 bytes  (4×)
  binary: █                                  16 bytes  (32×)
```

---

## Product Quantization (IVF-PQ)

### The Idea

Split each vector into `m` sub-vectors and independently quantize each to one of 256 centroids. Store only the centroid index (1 byte per sub-vector).

```
Original vector (128-dim):
┌──────────────────────────────────────────────────────┐
│ d0..d15 │ d16..d31 │ d32..d47 │ ... │ d112..d127    │
└──────────────────────────────────────────────────────┘
    ↓           ↓           ↓               ↓
 quantize    quantize    quantize        quantize
 to 256      to 256      to 256          to 256
 centroids   centroids   centroids       centroids
    ↓           ↓           ↓               ↓
┌──────────────────────────────────────────────────────┐
│  code=42  │  code=7   │  code=128 │ ... │  code=99  │
└──────────────────────────────────────────────────────┘
  m=8 sub-vectors × 1 byte each = 8 bytes total

Compression: 512 bytes → 8 bytes = 64×
```

### Asymmetric Distance Computation (ADC)

At search time, we don't decode the vectors. Instead, we precompute a lookup table:

```
Step 1: Precompute distance table (once per query)
  table[sub][code] = distance(query_sub_vector, codebook[sub][code])
  Size: m × 256 entries

Step 2: Scan encoded vectors (O(m) per vector instead of O(D))
  for each encoded vector [c0, c1, c2, ..., c7]:
    dist = table[0][c0] + table[1][c1] + ... + table[7][c7]
```

This turns an O(D) distance calculation into O(m) table lookups — typically a 16-64× reduction.

---

## Memory-Mapped Index

### The Problem

With 100M vectors × 128 dims × 4 bytes = 51 GB, the dataset doesn't fit in RAM.

### The Solution

Store vectors on disk and let the OS virtual memory system page them in on demand:

```
┌──────────────────────────────────────┐
│              Disk File               │
│  Header (32 bytes)                   │
│  Record 0: [id: u64][vector: f32×D]  │
│  Record 1: [id: u64][vector: f32×D]  │
│  ...                                 │
│  Record N: [id: u64][vector: f32×D]  │
└──────────────────────────────────────┘
        ↕ mmap (OS manages paging)
┌──────────────────────────────────────┐
│           Virtual Memory             │
│  Only accessed pages are in RAM      │
│  OS evicts cold pages automatically  │
└──────────────────────────────────────┘
```

### Mutation Model

Writes go to an in-memory staging buffer. Deletes go to a tombstone set. Only on `flush()` does the data get written to the mmap file:

```
Insert → staging_buffer (in-memory Vec)
Delete → tombstones (HashSet)
Search → scan mmap file + staging_buffer, skip tombstones
Flush  → compact (remove tombstones) + append staging → remap file
```

**Impact**: Near-zero RSS for datasets larger than RAM. Sequential scan throughput limited by disk bandwidth (~1-3 GB/s SSD).

---

## Python ↔ Rust Data Transfer

### The Problem

Passing vectors from Python (numpy) to Rust one-by-one is slow:

```python
# Slow: 10,000 Python → Rust calls
for i in range(10000):
    index.add(i, vectors[i].tolist())  # tolist() copies to Python list
                                        # PyO3 extracts each float individually
```

### The Solution: Buffer Protocol (PEP 3118)

Numpy arrays expose their raw memory via the buffer protocol. Quiver reads the underlying C array directly:

```
Python numpy array:
  array.data → pointer to contiguous f32 buffer
  array.shape → (10000, 128)

Rust side:
  PyBuffer::get_bound(obj)    ← get pointer to numpy's memory
  buf.copy_to_slice(&mut data) ← single memcpy of 10000 × 128 × 4 = 5.1 MB
```

No per-element Python overhead. One `memcpy` for the entire batch.

```python
# Fast: 1 Python → Rust call, buffer protocol transfer
vectors = np.random.randn(10000, 128).astype('float32')
index.add_batch_np(vectors, start_id=0)
```

### Performance

```
Per-vector Python list:   ~5 µs × 10,000 = 50 ms
Buffer protocol batch:    ~0.1 ms (single memcpy)
Speedup: ~500×
```

---

## Summary of All Optimizations

| Component | Optimization | Technique | Impact |
|-----------|-------------|-----------|--------|
| **Distance** | AVX2+FMA SIMD | 8 floats per cycle, fused multiply-add | ~8× vs scalar |
| **Distance** | NEON SIMD (ARM) | 4 floats per cycle, ARM vectorization | ~4× vs scalar |
| **Distance** | Batch-4 ILP | 4 independent accumulators, exploit multiple FMA units | ~2-3× vs sequential |
| **Flat** | Contiguous buffer | Single `Vec<f32>` replaces `HashMap<u64, Vec<f32>>` | **20× insert** (1.4M → 28.3M) |
| **Flat** | Soft-delete | Boolean marker instead of data movement | O(1) delete |
| **Flat** | Batch insert | `extend_from_slice` = single memcpy | Near memory-bandwidth insert |
| **HNSW** | Flat vector storage | Contiguous `Vec<f32>` indexed by node ID | O(1) access, cache-hot |
| **HNSW** | Flat layer-0 neighbors | Single buffer with stride, no `Vec<Vec<>>` | 1 vs 2 pointer dereferences |
| **HNSW** | Generation-counter visited | `gen += 1` instead of `HashSet::new()` | O(1) reset per search |
| **HNSW** | Batch-4 neighbor scan | 4 distances in parallel during graph traversal | ~2× neighbor evaluation |
| **HNSW** | Phase-separated parallel insert | Read-only search phase + sequential link phase | ~2-4× insert (multi-thread) |
| **IVF** | Contiguous centroids | `Vec<f32>` instead of `Vec<Vec<f32>>` | 1.3-1.5× |
| **IVF** | Contiguous posting lists | Per-cluster `Vec<f32>` instead of per-vector allocs | 1.5-2× |
| **IVF** | Batch-4 centroid search | ILP on centroid comparisons | 2-3× |
| **IVF** | GEMM matrix multiply | `matrixmultiply::sgemm` for N×K assignment | **3-5×** (cache-blocked micro-kernels) |
| **IVF** | 3-phase batch insert | GEMM assign → count/reserve → scatter-append | No per-vector allocation |
| **Quantization** | Int8 scalar | 4× compression, per-vector scale factor | 9.5× insert speedup |
| **Quantization** | FP16 half-precision | 2× compression, IEEE 754 half | >99.5% recall |
| **Quantization** | Binary 1-bit | 32× compression, hardware `popcount` | 5.6× search speedup |
| **IVF-PQ** | Product quantization | m sub-vectors × 256 centroids each | 64-85× compression |
| **IVF-PQ** | Asymmetric distance (ADC) | Precompute lookup table, O(m) vs O(D) | 16-64× fewer FLOPs |
| **Mmap** | Memory-mapped file | OS-managed virtual memory paging | 100M+ vectors on 8GB RAM |
| **Mmap** | Staging buffer | In-memory inserts, disk flush on demand | Fast writes, lazy persistence |
| **Python** | Buffer protocol | Direct numpy memory access (PEP 3118) | ~500× vs per-element extract |
| **Python** | Batch APIs | `add_batch_np()` — single call for N vectors | Eliminates Python loop overhead |
| **K-Means** | GEMM assignment | Same matmul decomposition as IVF insert | 3-5× per training iteration |
| **K-Means** | Early-stop convergence | Track `changed` flag, stop when stable | Skip unnecessary iterations |

---

## Design Principles

1. **Contiguous memory wins.** Replace `HashMap` and `Vec<Vec<>>` with flat buffers. The CPU cache is the most important hardware feature for performance.

2. **Expose instruction-level parallelism.** Use independent accumulators (batch-4) so the CPU can use all its execution units simultaneously.

3. **Delegate hot loops to optimized libraries.** GEMM via `matrixmultiply` uses cache-blocked micro-kernels that are hard to beat by hand.

4. **Separate read and write phases.** In HNSW parallel insert, the search phase is read-only (safe to parallelize) and the linking phase is sequential (safe to mutate).

5. **Defer expensive operations.** Soft-delete instead of compaction. Staging buffers instead of immediate disk writes. Generation counters instead of hash table allocation.

6. **Runtime dispatch, not compile-time.** Detect AVX2/NEON at runtime so one binary works everywhere, always using the best available instructions.

7. **Zero-copy across the FFI boundary.** Use Python's buffer protocol to share numpy memory directly with Rust, avoiding per-element marshalling.
