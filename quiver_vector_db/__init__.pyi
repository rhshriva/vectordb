"""Type stubs for quiver_vector_db — embedded vector database with Rust core."""

from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence, Tuple, Union

# ── Standalone Index Classes ───────────────────────────────────────────────

class FlatIndex:
    """Exact brute-force vector index. 100% recall, O(N*D) per query.

    Best for small datasets (<100K vectors) or when exact results are required.
    """

    dimensions: int
    """Number of dimensions per vector."""
    metric: str
    """Distance metric ("l2", "cosine", or "dot_product")."""

    def __init__(
        self,
        dimensions: int,
        metric: Literal["l2", "cosine", "dot_product"] = "l2",
    ) -> None: ...

    def add(self, id: int, vector: List[float]) -> None:
        """Add a single vector with the given ID.

        Args:
            id: Unique integer ID for the vector.
            vector: List of floats with length equal to ``dimensions``.

        Raises:
            ValueError: If vector length does not match ``dimensions``.
            KeyError: If ``id`` already exists.
        """
        ...

    def add_batch(self, entries: List[Tuple[int, List[float]]]) -> None:
        """Add multiple vectors at once.

        Args:
            entries: List of ``(id, vector)`` tuples.
        """
        ...

    def add_batch_np(self, vectors: Any, start_id: int = 0) -> None:
        """Add vectors from a 2D numpy array (N x dim) with sequential IDs.

        This is the fastest single-threaded insert method — passes contiguous
        memory directly to Rust with zero per-element Python overhead.

        Args:
            vectors: 2D numpy array of shape (N, dim), dtype float32.
            start_id: First ID to assign (default 0). IDs are start_id..start_id+N-1.
        """
        ...

    def search(self, query: List[float], k: int) -> List[Dict[str, Any]]:
        """Search for the k nearest vectors.

        Args:
            query: Query vector with length equal to ``dimensions``.
            k: Number of nearest neighbors to return.

        Returns:
            List of dicts with keys ``"id"`` (int) and ``"distance"`` (float),
            sorted by distance ascending.
        """
        ...

    def delete(self, id: int) -> bool:
        """Remove a vector by ID.

        Returns:
            True if found and removed, False if not found.
        """
        ...

    def save(self, path: str) -> None:
        """Save index to a binary file."""
        ...

    @staticmethod
    def load(path: str) -> "FlatIndex":
        """Load a previously saved index from a binary file."""
        ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...


class HnswIndex:
    """Graph-based approximate nearest-neighbor index (HNSW). 95-99% recall.

    Best general-purpose index for datasets of any size. Uses Hierarchical
    Navigable Small World graphs with SIMD-accelerated distance computation.
    """

    dimensions: int
    """Number of dimensions per vector."""
    metric: str
    """Distance metric ("l2", "cosine", or "dot_product")."""

    def __init__(
        self,
        dimensions: int,
        metric: Literal["l2", "cosine", "dot_product"] = "l2",
        ef_construction: int = 200,
        ef_search: int = 50,
        m: int = 12,
    ) -> None:
        """
        Args:
            dimensions: Number of dimensions per vector.
            metric: Distance metric.
            ef_construction: Build beam width. Higher = better recall, slower build.
            ef_search: Query beam width. Tunable at runtime via ``flush()``.
            m: Graph edges per node. Higher = better recall, more RAM.
        """
        ...

    def add(self, id: int, vector: List[float]) -> None:
        """Add a single vector with the given ID."""
        ...

    def add_batch(self, entries: List[Tuple[int, List[float]]]) -> None:
        """Add multiple vectors at once."""
        ...

    def add_batch_np(self, vectors: Any, start_id: int = 0) -> None:
        """Add vectors from a 2D numpy array (N x dim) with sequential IDs.

        This is the fastest single-threaded insert method — passes contiguous
        memory directly to Rust with zero per-element Python overhead.

        Args:
            vectors: 2D numpy array of shape (N, dim), dtype float32.
            start_id: First ID to assign (default 0). IDs are start_id..start_id+N-1.
        """
        ...

    def add_batch_parallel(
        self,
        vectors: Any,
        start_id: int = 0,
        num_threads: int = 0,
        micro_batch_size: int = 256,
    ) -> None:
        """Parallel batch insert from a 2D numpy array using micro-batching.

        Uses rayon thread pool for multi-threaded neighbor search during insert.
        The first micro-batch is inserted sequentially to bootstrap the graph,
        then remaining batches use parallel search + sequential linking.

        Args:
            vectors: 2D numpy array of shape (N, dim), dtype float32.
            start_id: First ID to assign (default 0).
            num_threads: Number of threads (default 0 = all available cores).
            micro_batch_size: Vectors per micro-batch (default 256).
        """
        ...

    def search(self, query: List[float], k: int) -> List[Dict[str, Any]]:
        """Search for the k nearest vectors.

        Returns:
            List of dicts with ``"id"`` and ``"distance"`` keys.
        """
        ...

    def delete(self, id: int) -> bool:
        """Remove a vector by ID. Returns True if found."""
        ...

    def flush(self) -> None:
        """Build/rebuild the HNSW graph. Call after bulk inserts."""
        ...

    def save(self, path: str) -> None:
        """Save index to a binary file."""
        ...

    @staticmethod
    def load(path: str) -> "HnswIndex":
        """Load a previously saved index from a binary file."""
        ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...


class QuantizedFlatIndex:
    """Int8 quantized brute-force index. ~4x less RAM with ~99% recall.

    Each f32 component is quantized to int8 using per-vector min/max scaling.
    """

    dimensions: int
    metric: str

    def __init__(
        self,
        dimensions: int,
        metric: Literal["l2", "cosine", "dot_product"] = "l2",
    ) -> None: ...

    def add(self, id: int, vector: List[float]) -> None: ...
    def add_batch(self, entries: List[Tuple[int, List[float]]]) -> None: ...

    def add_batch_np(self, vectors: Any, start_id: int = 0) -> None:
        """Add vectors from a 2D numpy array (N x dim) with sequential IDs.

        This is the fastest single-threaded insert method — passes contiguous
        memory directly to Rust with zero per-element Python overhead.

        Args:
            vectors: 2D numpy array of shape (N, dim), dtype float32.
            start_id: First ID to assign (default 0). IDs are start_id..start_id+N-1.
        """
        ...

    def search(self, query: List[float], k: int) -> List[Dict[str, Any]]: ...
    def delete(self, id: int) -> bool: ...
    def save(self, path: str) -> None: ...

    @staticmethod
    def load(path: str) -> "QuantizedFlatIndex": ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...


class Fp16FlatIndex:
    """Float16 quantized brute-force index. 2x less RAM with >99.5% recall.

    Vectors are stored as IEEE 754 half-precision floats.
    """

    dimensions: int
    metric: str

    def __init__(
        self,
        dimensions: int,
        metric: Literal["l2", "cosine", "dot_product"] = "l2",
    ) -> None: ...

    def add(self, id: int, vector: List[float]) -> None: ...
    def add_batch(self, entries: List[Tuple[int, List[float]]]) -> None: ...

    def add_batch_np(self, vectors: Any, start_id: int = 0) -> None:
        """Add vectors from a 2D numpy array (N x dim) with sequential IDs.

        This is the fastest single-threaded insert method — passes contiguous
        memory directly to Rust with zero per-element Python overhead.

        Args:
            vectors: 2D numpy array of shape (N, dim), dtype float32.
            start_id: First ID to assign (default 0). IDs are start_id..start_id+N-1.
        """
        ...

    def search(self, query: List[float], k: int) -> List[Dict[str, Any]]: ...
    def delete(self, id: int) -> bool: ...
    def save(self, path: str) -> None: ...

    @staticmethod
    def load(path: str) -> "Fp16FlatIndex": ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...


class IvfIndex:
    """Inverted file index. Cluster-based ANN using k-means.

    Auto-trains after ``train_size`` inserts. Good for large datasets (100K-100M).
    """

    dimensions: int
    metric: str

    def __init__(
        self,
        dimensions: int,
        metric: Literal["l2", "cosine", "dot_product"] = "l2",
        n_lists: int = 256,
        nprobe: int = 16,
        train_size: int = 4096,
    ) -> None:
        """
        Args:
            dimensions: Number of dimensions per vector.
            metric: Distance metric.
            n_lists: Number of clusters. Rule of thumb: ``sqrt(N)``.
            nprobe: Clusters scanned per query. Higher = better recall, slower.
            train_size: Vectors buffered before auto-training.
        """
        ...

    def add(self, id: int, vector: List[float]) -> None: ...
    def add_batch(self, entries: List[Tuple[int, List[float]]]) -> None: ...

    def add_batch_np(self, vectors: Any, start_id: int = 0) -> None:
        """Add vectors from a 2D numpy array (N x dim) with sequential IDs.

        This is the fastest single-threaded insert method — passes contiguous
        memory directly to Rust with zero per-element Python overhead.

        Args:
            vectors: 2D numpy array of shape (N, dim), dtype float32.
            start_id: First ID to assign (default 0). IDs are start_id..start_id+N-1.
        """
        ...

    def search(self, query: List[float], k: int) -> List[Dict[str, Any]]: ...
    def delete(self, id: int) -> bool: ...
    def flush(self) -> None: ...
    def save(self, path: str) -> None: ...

    @staticmethod
    def load(path: str) -> "IvfIndex": ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...


class IvfPqIndex:
    """IVF with product quantization. ~96x memory reduction for 1536-dim vectors.

    Combines IVF coarse quantizer with PQ compression. Ideal for million-scale.
    Memory per vector = ``pq_m`` bytes.
    """

    dimensions: int
    metric: str

    def __init__(
        self,
        dimensions: int,
        metric: Literal["l2", "cosine", "dot_product"] = "l2",
        n_lists: int = 256,
        nprobe: int = 16,
        train_size: int = 4096,
        pq_m: int = 8,
        pq_k_sub: int = 256,
    ) -> None:
        """
        Args:
            dimensions: Number of dimensions per vector.
            metric: Distance metric.
            n_lists: Number of IVF clusters.
            nprobe: Clusters scanned per query.
            train_size: Vectors buffered before auto-training.
            pq_m: Number of PQ sub-quantizers (must divide ``dimensions``).
            pq_k_sub: Centroids per sub-quantizer.
        """
        ...

    def add(self, id: int, vector: List[float]) -> None: ...
    def add_batch(self, entries: List[Tuple[int, List[float]]]) -> None: ...

    def add_batch_np(self, vectors: Any, start_id: int = 0) -> None:
        """Add vectors from a 2D numpy array (N x dim) with sequential IDs.

        This is the fastest single-threaded insert method — passes contiguous
        memory directly to Rust with zero per-element Python overhead.

        Args:
            vectors: 2D numpy array of shape (N, dim), dtype float32.
            start_id: First ID to assign (default 0). IDs are start_id..start_id+N-1.
        """
        ...

    def search(self, query: List[float], k: int) -> List[Dict[str, Any]]: ...
    def delete(self, id: int) -> bool: ...
    def flush(self) -> None: ...
    def save(self, path: str) -> None: ...

    @staticmethod
    def load(path: str) -> "IvfPqIndex": ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...


class MmapFlatIndex:
    """Memory-mapped brute-force index. Near-zero RAM usage.

    Vectors are stored in a flat binary file and memory-mapped by the OS.
    Best when dataset is larger than available RAM.
    """

    dimensions: int
    metric: str

    def __init__(
        self,
        dimensions: int,
        metric: Literal["l2", "cosine", "dot_product"] = "l2",
        path: str = "./mmap_index.qvec",
    ) -> None: ...

    def add(self, id: int, vector: List[float]) -> None: ...
    def add_batch(self, entries: List[Tuple[int, List[float]]]) -> None: ...

    def add_batch_np(self, vectors: Any, start_id: int = 0) -> None:
        """Add vectors from a 2D numpy array (N x dim) with sequential IDs.

        This is the fastest single-threaded insert method — passes contiguous
        memory directly to Rust with zero per-element Python overhead.

        Args:
            vectors: 2D numpy array of shape (N, dim), dtype float32.
            start_id: First ID to assign (default 0). IDs are start_id..start_id+N-1.
        """
        ...

    def search(self, query: List[float], k: int) -> List[Dict[str, Any]]: ...
    def delete(self, id: int) -> bool: ...
    def flush(self) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...


class BinaryFlatIndex:
    """Binary (1-bit) quantized brute-force index. 32x less RAM than FlatIndex.

    Each f32 component is reduced to 1 bit (positive=1, negative=0).
    Distance is computed via Hamming distance (popcount).
    """

    dimensions: int
    metric: str

    def __init__(
        self,
        dimensions: int,
        metric: Literal["l2", "cosine", "dot_product"] = "l2",
    ) -> None: ...

    def add(self, id: int, vector: List[float]) -> None: ...
    def add_batch(self, entries: List[Tuple[int, List[float]]]) -> None: ...

    def add_batch_np(self, vectors: Any, start_id: int = 0) -> None:
        """Add vectors from a 2D numpy array (N x dim) with sequential IDs.

        This is the fastest single-threaded insert method — passes contiguous
        memory directly to Rust with zero per-element Python overhead.

        Args:
            vectors: 2D numpy array of shape (N, dim), dtype float32.
            start_id: First ID to assign (default 0). IDs are start_id..start_id+N-1.
        """
        ...

    def search(self, query: List[float], k: int) -> List[Dict[str, Any]]: ...
    def delete(self, id: int) -> bool: ...
    def save(self, path: str) -> None: ...

    @staticmethod
    def load(path: str) -> "BinaryFlatIndex": ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...


# ── Collection & Client ───────────────────────────────────────────────────

class Collection:
    """A named collection of vectors with WAL-backed persistence.

    Obtained via ``Client.create_collection()`` or ``Client.get_collection()``.
    Supports upsert, search, delete, hybrid search, and payload filtering.
    """

    count: int
    """Number of dense vectors in the collection."""
    sparse_count: int
    """Number of sparse vectors in the collection."""
    name: str
    """Collection name."""

    def upsert(
        self,
        id: int,
        vector: List[float],
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update a vector by ID with optional metadata payload.

        Args:
            id: Unique integer ID.
            vector: Dense vector.
            payload: Optional JSON-serializable metadata dict.
        """
        ...

    def upsert_batch(
        self,
        entries: List[Union[
            Tuple[int, List[float]],
            Tuple[int, List[float], Optional[Dict[str, Any]]],
        ]],
    ) -> None:
        """Batch insert or update multiple vectors at once.

        More efficient than calling ``upsert`` in a loop.

        Args:
            entries: List of ``(id, vector)`` or ``(id, vector, payload)`` tuples.
        """
        ...

    def search(
        self,
        query: List[float],
        k: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for the k nearest vectors with optional filtering.

        Filter operators: ``$eq``, ``$ne``, ``$in``, ``$gt``, ``$gte``,
        ``$lt``, ``$lte``, ``$and``, ``$or``.

        Args:
            query: Query vector.
            k: Number of results.
            filter: Optional filter dict, e.g.
                ``{"category": {"$eq": "tech"}}``.

        Returns:
            List of dicts with keys ``"id"``, ``"distance"``, ``"payload"``.
        """
        ...

    def upsert_hybrid(
        self,
        id: int,
        vector: List[float],
        sparse_vector: Optional[Dict[int, float]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Upsert with an optional sparse vector for hybrid search.

        Args:
            id: Vector ID.
            vector: Dense vector.
            sparse_vector: Optional sparse vector as ``{dimension_index: weight}``.
            payload: Optional metadata dict.
        """
        ...

    def search_hybrid(
        self,
        dense_query: List[float],
        sparse_query: Dict[int, float],
        k: int,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid dense+sparse search with weighted score fusion.

        Args:
            dense_query: Dense query vector.
            sparse_query: Sparse query as ``{dimension_index: weight}``.
            k: Number of results.
            dense_weight: Weight for dense similarity (default 0.7).
            sparse_weight: Weight for sparse similarity (default 0.3).
            filter: Optional payload filter.

        Returns:
            List of dicts with keys ``"id"``, ``"score"``,
            ``"dense_distance"``, ``"sparse_score"``, ``"payload"``.
        """
        ...

    def delete(self, id: int) -> bool:
        """Delete a vector by ID. Returns True if found and removed."""
        ...

    def create_snapshot(self, name: str) -> Dict[str, Any]:
        """Create a named snapshot of this collection's current state.

        Returns:
            Dict with keys ``name``, ``created_at``, ``vector_count``, ``sparse_count``.

        Raises:
            KeyError: If a snapshot with this name already exists.
        """
        ...

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all snapshots, sorted by creation time.

        Returns:
            List of dicts with keys ``name``, ``created_at``, ``vector_count``, ``sparse_count``.
        """
        ...

    def restore_snapshot(self, name: str) -> None:
        """Restore this collection to the state captured by snapshot ``name``.

        Raises:
            KeyError: If snapshot does not exist.
        """
        ...

    def delete_snapshot(self, name: str) -> None:
        """Delete a snapshot by name.

        Raises:
            KeyError: If snapshot does not exist.
        """
        ...

    def __repr__(self) -> str: ...


class Client:
    """Persistent vector database client.

    Opens a directory on disk. All writes are WAL-backed and survive restarts.

    Example::

        db = Client(path="./my_data")
        col = db.create_collection("docs", dimensions=384, metric="cosine")
        col.upsert(id=1, vector=[...], payload={"title": "hello"})
        hits = col.search(query=[...], k=5)
    """

    def __init__(self, path: str = "./data") -> None:
        """
        Args:
            path: Directory path for data storage.
        """
        ...

    def create_collection(
        self,
        name: str,
        dimensions: int,
        metric: Literal["cosine", "l2", "dot_product"] = "cosine",
        index_type: Literal[
            "hnsw", "flat", "quantized_flat", "fp16_flat",
            "ivf", "ivf_pq", "mmap_flat", "binary_flat",
        ] = "hnsw",
    ) -> Collection:
        """Create a new collection.

        Args:
            name: Collection name.
            dimensions: Vector dimensions.
            metric: Distance metric.
            index_type: Index algorithm to use.

        Returns:
            The created ``Collection``.

        Raises:
            KeyError: If a collection with this name already exists.
        """
        ...

    def get_collection(self, name: str) -> Collection:
        """Get an existing collection by name.

        Raises:
            KeyError: If collection does not exist.
        """
        ...

    def get_or_create_collection(
        self,
        name: str,
        dimensions: int,
        metric: Literal["cosine", "l2", "dot_product"] = "cosine",
    ) -> Collection:
        """Get a collection by name, or create it if it doesn't exist.

        Uses HNSW index type for newly created collections.
        """
        ...

    def delete_collection(self, name: str) -> bool:
        """Delete a collection and all its data from disk."""
        ...

    def list_collections(self) -> List[str]:
        """List all collection names."""
        ...

    def __repr__(self) -> str: ...


# ── Embedding Function Protocol ───────────────────────────────────────

class EmbeddingFunction(Protocol):
    """Protocol that any embedding provider must satisfy.

    Implement ``__call__`` to embed a batch of texts, and optionally
    ``dimensions`` to report the output dimensionality.
    """

    def __call__(self, texts: List[str]) -> List[List[float]]: ...

    @property
    def dimensions(self) -> Optional[int]: ...


class SentenceTransformerEmbedding:
    """Local embedding via sentence-transformers.

    Requires: ``pip install quiver-vector-db[sentence-transformers]``
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ) -> None: ...

    def __call__(self, texts: List[str]) -> List[List[float]]: ...

    @property
    def dimensions(self) -> int: ...


class OpenAIEmbedding:
    """OpenAI embedding API wrapper.

    Requires: ``pip install quiver-vector-db[openai]``
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> None: ...

    def __call__(self, texts: List[str]) -> List[List[float]]: ...

    @property
    def dimensions(self) -> Optional[int]: ...


# ── BM25 ──────────────────────────────────────────────────────────────

class BM25:
    """Okapi BM25 tokenizer and sparse vector generator.

    Produces sparse vectors compatible with ``Collection.upsert_hybrid()``.
    """

    doc_count: int
    """Total number of indexed documents."""
    vocab_size: int
    """Number of unique terms in the vocabulary."""
    avg_dl: float
    """Average document length (in tokens)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None: ...

    def index_document(self, doc_id: int, text: str) -> Dict[int, float]:
        """Tokenize and index a document. Returns BM25 sparse vector."""
        ...

    def encode_query(self, text: str) -> Dict[int, float]:
        """Encode a query into an IDF-weighted sparse vector."""
        ...

    def remove_document(self, doc_id: int) -> None:
        """Remove a document's contribution to global stats."""
        ...

    def save(self, path: str) -> None:
        """Serialize BM25 state to a JSON file."""
        ...

    @classmethod
    def load(cls, path: str) -> "BM25":
        """Load BM25 state from a JSON file."""
        ...


# ── TextCollection ────────────────────────────────────────────────────

class TextCollection:
    """Document-oriented collection with automatic embedding + BM25.

    Wraps a Rust ``Collection`` and provides text-in / text-out methods.

    Example::

        text_col = TextCollection(col, SentenceTransformerEmbedding())
        text_col.add(ids=[1, 2], documents=["Hello", "World"])
        hits = text_col.query("greeting", k=5)
    """

    count: int
    """Number of documents in the collection."""
    name: str
    """Collection name."""

    def __init__(
        self,
        collection: Collection,
        embedding_function: EmbeddingFunction,
        enable_bm25: bool = True,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        """
        Args:
            collection: A Rust ``Collection`` from ``Client.create_collection()``.
            embedding_function: Any callable satisfying ``EmbeddingFunction``.
            enable_bm25: Enable BM25 full-text indexing (default True).
            bm25_k1: BM25 k1 parameter (term frequency saturation).
            bm25_b: BM25 b parameter (document length normalization).
        """
        ...

    def add(
        self,
        ids: Sequence[int],
        documents: Sequence[str],
        payloads: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> None:
        """Add documents by text. Handles embedding and BM25 indexing.

        Args:
            ids: Unique integer IDs.
            documents: Text strings to embed and index.
            payloads: Optional metadata dicts (one per document).
        """
        ...

    def query(
        self,
        query_text: str,
        k: int = 10,
        mode: Literal["hybrid", "semantic", "keyword"] = "hybrid",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search by natural language query.

        Args:
            query_text: Natural language query string.
            k: Number of results to return.
            mode: ``"hybrid"`` (default), ``"semantic"``, or ``"keyword"``.
            dense_weight: Weight for dense similarity (hybrid mode).
            sparse_weight: Weight for keyword similarity (hybrid mode).
            filter: Optional payload filter dict.

        Returns:
            List of result dicts with ``id``, ``document``, ``payload``,
            and either ``distance`` (semantic) or ``score`` (hybrid/keyword).
        """
        ...

    def delete(self, ids: Sequence[int]) -> None:
        """Delete documents by ID."""
        ...


# ── Multi-Vector / Multi-Modal Collection ─────────────────────────────

class MultiVectorCollection:
    """A collection supporting multiple named vector spaces per document.

    Stores image + text (or any combination of) embeddings for the same
    document IDs, with cross-space fusion search.

    Example::

        multi = MultiVectorCollection(
            client=db,
            name="products",
            vector_spaces={
                "text":  {"dimensions": 384, "metric": "cosine"},
                "image": {"dimensions": 512, "metric": "cosine"},
            },
        )
        multi.upsert(id=1, vectors={"text": [...], "image": [...]}, payload={"title": "Shirt"})
        hits = multi.search_multi(
            queries={"text": [...], "image": [...]},
            k=5,
            weights={"text": 0.6, "image": 0.4},
        )
    """

    name: str
    """Base name of the multi-vector collection."""
    vector_spaces: List[str]
    """Sorted list of vector space names."""
    count: int
    """Number of documents (based on the primary space)."""

    def __init__(
        self,
        client: Client,
        name: str,
        vector_spaces: Dict[str, Dict[str, Any]],
        index_type: str = "hnsw",
    ) -> None:
        """
        Args:
            client: A ``Client`` instance.
            name: Base name for the collection group.
            vector_spaces: Mapping of space_name to config dict with
                ``dimensions`` (int) and optionally ``metric`` (str).
            index_type: Index algorithm for all sub-collections.
        """
        ...

    def upsert(
        self,
        id: int,
        vectors: Dict[str, List[float]],
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update a document with vectors in one or more spaces.

        Args:
            id: Unique integer ID.
            vectors: Mapping of space_name to vector.
            payload: Optional metadata dict.
        """
        ...

    def upsert_batch(
        self,
        entries: Sequence[
            Union[
                Tuple[int, Dict[str, List[float]]],
                Tuple[int, Dict[str, List[float]], Optional[Dict[str, Any]]],
            ]
        ],
    ) -> None:
        """Batch upsert multiple documents at once.

        Args:
            entries: List of ``(id, vectors)`` or ``(id, vectors, payload)`` tuples.
        """
        ...

    def search(
        self,
        vector_space: str,
        query: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search within a single vector space.

        Args:
            vector_space: Which vector space to search.
            query: Query vector.
            k: Number of results.
            filter: Optional payload filter.
        """
        ...

    def search_multi(
        self,
        queries: Dict[str, List[float]],
        k: int = 10,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Search across multiple spaces with weighted fusion.

        Args:
            queries: Mapping of space_name to query vector.
            k: Number of results.
            weights: Optional mapping of space_name to weight.

        Returns:
            List of dicts with ``id``, ``score``, and ``distances``.
        """
        ...

    def delete(self, id: int) -> None:
        """Delete a document across all vector spaces."""
        ...

    def delete_batch(self, ids: Sequence[int]) -> None:
        """Delete multiple documents across all vector spaces."""
        ...


# ── REST API Server ───────────────────────────────────────────────────

def create_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    data_path: str = "./data",
) -> Any:
    """Create a Quiver REST API server.

    Usage::

        server = create_server(port=9090, data_path="./my_data")
        server.serve_forever()

    Or from the command line::

        python -m quiver_vector_db.server --port 9090 --data ./my_data

    Args:
        host: Bind address (default ``"0.0.0.0"``).
        port: Port number (default ``8080``).
        data_path: Path to data directory (default ``"./data"``).
    """
    ...
