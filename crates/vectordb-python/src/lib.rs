use pyo3::exceptions::{PyIOError, PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use vectordb_core::{
    distance::Metric,
    error::VectorDbError,
    index::{
        flat::FlatIndex,
        hnsw::{HnswConfig, HnswIndex},
        VectorIndex,
    },
};

// ---------------------------------------------------------------------------
// Error conversion
// ---------------------------------------------------------------------------

fn vec_err_to_py(e: VectorDbError) -> PyErr {
    match e {
        VectorDbError::DimensionMismatch { expected, got } => {
            PyValueError::new_err(format!("dimension mismatch: expected {expected}, got {got}"))
        }
        VectorDbError::DuplicateId(id) => {
            PyKeyError::new_err(format!("duplicate vector ID: {id}"))
        }
        VectorDbError::NotFound(id) => {
            PyKeyError::new_err(format!("vector ID not found: {id}"))
        }
        VectorDbError::EmptyIndex => PyValueError::new_err("index is empty"),
        VectorDbError::InvalidConfig(msg) => {
            PyValueError::new_err(format!("invalid configuration: {msg}"))
        }
        VectorDbError::Serialization(e) => {
            PyRuntimeError::new_err(format!("serialization error: {e}"))
        }
        VectorDbError::Io(e) => PyIOError::new_err(format!("I/O error: {e}")),
    }
}

// ---------------------------------------------------------------------------
// Metric string helpers
// ---------------------------------------------------------------------------

fn parse_metric(s: &str) -> PyResult<Metric> {
    match s {
        "l2" => Ok(Metric::L2),
        "cosine" => Ok(Metric::Cosine),
        "dot_product" => Ok(Metric::DotProduct),
        other => Err(PyValueError::new_err(format!(
            "unknown metric {other:?}; expected \"l2\", \"cosine\", or \"dot_product\""
        ))),
    }
}

fn metric_to_str(m: Metric) -> &'static str {
    match m {
        Metric::L2 => "l2",
        Metric::Cosine => "cosine",
        Metric::DotProduct => "dot_product",
    }
}

// ---------------------------------------------------------------------------
// FlatIndex Python wrapper
// ---------------------------------------------------------------------------

/// Exact nearest-neighbour index using a flat (brute-force) scan.
///
/// All searches are exact; every query scans all stored vectors.
/// Suitable for up to ~100k vectors where recall must be 100%.
///
/// Args:
///     dimensions (int): Number of dimensions every vector must have.
///     metric (str): Distance metric — ``"l2"``, ``"cosine"``, or ``"dot_product"``.
///                   Defaults to ``"l2"``.
#[pyclass(name = "FlatIndex")]
struct PyFlatIndex {
    inner: FlatIndex,
}

#[pymethods]
impl PyFlatIndex {
    #[new]
    #[pyo3(signature = (dimensions, metric = "l2"))]
    fn new(dimensions: usize, metric: &str) -> PyResult<Self> {
        let m = parse_metric(metric)?;
        Ok(Self {
            inner: FlatIndex::new(dimensions, m),
        })
    }

    /// Add a single vector.
    ///
    /// Args:
    ///     id (int): Caller-supplied identifier (u64).
    ///     vector (list[float]): The vector to insert.
    ///
    /// Raises:
    ///     ValueError: Dimension mismatch.
    ///     KeyError: Duplicate ID.
    #[pyo3(signature = (id, vector))]
    fn add(&mut self, id: u64, vector: Vec<f32>) -> PyResult<()> {
        self.inner.add(id, &vector).map_err(vec_err_to_py)
    }

    /// Add multiple vectors in one call.
    ///
    /// Args:
    ///     entries (list[tuple[int, list[float]]]): ``[(id, vector), ...]``.
    fn add_batch(&mut self, entries: Vec<(u64, Vec<f32>)>) -> PyResult<()> {
        self.inner.add_batch(&entries).map_err(vec_err_to_py)
    }

    /// Search for the *k* nearest neighbours of *query*.
    ///
    /// The GIL is released during the scan so other threads can run.
    ///
    /// Returns:
    ///     list[dict]: Each dict has ``"id"`` (int) and ``"distance"`` (float),
    ///     sorted by ascending distance.
    fn search(&self, py: Python<'_>, query: Vec<f32>, k: usize) -> PyResult<Vec<PyObject>> {
        let results = py
            .allow_threads(|| self.inner.search(&query, k))
            .map_err(vec_err_to_py)?;

        results
            .into_iter()
            .map(|r| {
                let dict = PyDict::new_bound(py);
                dict.set_item("id", r.id)?;
                dict.set_item("distance", r.distance)?;
                Ok(dict.into())
            })
            .collect()
    }

    /// Remove a vector by ID.
    ///
    /// Returns:
    ///     bool: ``True`` if found and removed, ``False`` if not found.
    fn delete(&mut self, id: u64) -> bool {
        self.inner.delete(id)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Number of dimensions each stored vector must have.
    #[getter]
    fn dimensions(&self) -> usize {
        self.inner.config().dimensions
    }

    /// Distance metric used by this index (``"l2"``, ``"cosine"``, or ``"dot_product"``).
    #[getter]
    fn metric(&self) -> &'static str {
        metric_to_str(self.inner.config().metric)
    }

    /// Save the index to a file.
    ///
    /// Args:
    ///     path (str): Destination file path (e.g. ``"index.json"``).
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(vec_err_to_py)
    }

    /// Load an index from a file previously written by :meth:`save`.
    ///
    /// Args:
    ///     path (str): Path to the saved index file.
    ///
    /// Returns:
    ///     FlatIndex: The restored index.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = FlatIndex::load(path).map_err(vec_err_to_py)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "FlatIndex(dimensions={}, metric=\"{}\", len={})",
            self.inner.config().dimensions,
            metric_to_str(self.inner.config().metric),
            self.inner.len()
        )
    }
}

// ---------------------------------------------------------------------------
// HnswIndex Python wrapper
// ---------------------------------------------------------------------------

/// Approximate nearest-neighbour index using HNSW (Hierarchical Navigable Small World).
///
/// Trades a small amount of recall for sub-linear query time.
/// Suitable for millions of vectors where exact search would be too slow.
///
/// **Staging buffer**: vectors are buffered until *flush_threshold* (default 1000)
/// is reached, at which point the HNSW graph is rebuilt automatically.  For bulk
/// loads, call :meth:`add_batch` followed by an explicit :meth:`flush` to avoid
/// surprise latency spikes during insertion.
///
/// **After delete**: the HNSW graph is dropped and searches fall back to
/// brute-force until the next :meth:`flush`.
///
/// Args:
///     dimensions (int): Number of dimensions every vector must have.
///     metric (str): Distance metric — ``"l2"``, ``"cosine"``, or ``"dot_product"``.
///                   Defaults to ``"l2"``.
///     ef_construction (int): HNSW build-time beam width. Defaults to ``200``.
///     ef_search (int): HNSW search-time beam width. Defaults to ``50``.
///     m (int): Max neighbours per HNSW node. Defaults to ``12``.
#[pyclass(name = "HnswIndex")]
struct PyHnswIndex {
    inner: HnswIndex,
}

#[pymethods]
impl PyHnswIndex {
    #[new]
    #[pyo3(signature = (dimensions, metric = "l2", ef_construction = 200, ef_search = 50, m = 12))]
    fn new(
        dimensions: usize,
        metric: &str,
        ef_construction: usize,
        ef_search: usize,
        m: usize,
    ) -> PyResult<Self> {
        let met = parse_metric(metric)?;
        let cfg = HnswConfig {
            ef_construction,
            ef_search,
            m,
        };
        Ok(Self {
            inner: HnswIndex::new(dimensions, met, cfg),
        })
    }

    /// Add a single vector.
    ///
    /// Auto-flushes (rebuilds the HNSW graph) every 1000 inserts.
    /// For bulk loads prefer :meth:`add_batch` + :meth:`flush`.
    #[pyo3(signature = (id, vector))]
    fn add(&mut self, id: u64, vector: Vec<f32>) -> PyResult<()> {
        self.inner.add(id, &vector).map_err(vec_err_to_py)
    }

    /// Add multiple vectors in one call.
    fn add_batch(&mut self, entries: Vec<(u64, Vec<f32>)>) -> PyResult<()> {
        self.inner.add_batch(&entries).map_err(vec_err_to_py)
    }

    /// Search for the *k* nearest neighbours of *query*.
    ///
    /// The GIL is released during graph traversal so other threads can run.
    ///
    /// Returns:
    ///     list[dict]: Each dict has ``"id"`` (int) and ``"distance"`` (float).
    fn search(&self, py: Python<'_>, query: Vec<f32>, k: usize) -> PyResult<Vec<PyObject>> {
        let results = py
            .allow_threads(|| self.inner.search(&query, k))
            .map_err(vec_err_to_py)?;

        results
            .into_iter()
            .map(|r| {
                let dict = PyDict::new_bound(py);
                dict.set_item("id", r.id)?;
                dict.set_item("distance", r.distance)?;
                Ok(dict.into())
            })
            .collect()
    }

    /// Remove a vector by ID.
    ///
    /// After deletion the HNSW graph is invalidated; call :meth:`flush` to
    /// rebuild it if search performance matters.
    ///
    /// Returns:
    ///     bool: ``True`` if found and removed, ``False`` if not found.
    fn delete(&mut self, id: u64) -> bool {
        self.inner.delete(id)
    }

    /// Force an immediate rebuild of the HNSW graph.
    ///
    /// This is an O(N · M · log N) operation; the GIL is released so other
    /// Python threads can continue running while the graph is being built.
    fn flush(&mut self, py: Python<'_>) {
        py.allow_threads(|| self.inner.flush());
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Number of dimensions each stored vector must have.
    #[getter]
    fn dimensions(&self) -> usize {
        self.inner.config().dimensions
    }

    /// Distance metric used by this index (``"l2"``, ``"cosine"``, or ``"dot_product"``).
    #[getter]
    fn metric(&self) -> &'static str {
        metric_to_str(self.inner.config().metric)
    }

    /// Save the index to a file.
    ///
    /// Vectors are persisted; the HNSW graph is rebuilt automatically on load.
    ///
    /// Args:
    ///     path (str): Destination file path (e.g. ``"index.json"``).
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(vec_err_to_py)
    }

    /// Load an index from a file previously written by :meth:`save`.
    ///
    /// The HNSW graph is rebuilt immediately, so the returned index is ready
    /// for ANN search with no extra :meth:`flush` call needed.
    ///
    /// Args:
    ///     path (str): Path to the saved index file.
    ///
    /// Returns:
    ///     HnswIndex: The restored index.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = HnswIndex::load(path).map_err(vec_err_to_py)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "HnswIndex(dimensions={}, metric=\"{}\", len={})",
            self.inner.config().dimensions,
            metric_to_str(self.inner.config().metric),
            self.inner.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn vectordb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFlatIndex>()?;
    m.add_class::<PyHnswIndex>()?;
    Ok(())
}
