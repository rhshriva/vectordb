"""Type stubs for quiver_vector_db — embedded vector database with Rust core."""

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

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
    def search(self, query: List[float], k: int) -> List[Dict[str, Any]]: ...
    def delete(self, id: int) -> bool: ...
    def flush(self) -> None: ...
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
            "ivf", "ivf_pq", "mmap_flat",
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
