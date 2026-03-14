use pyo3::exceptions::{PyIOError, PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::collections::HashMap;

/// Extract a batch of (id, vector) entries from a Python list of tuples.
/// Uses buffer protocol for each vector when possible.
#[inline]
fn extract_batch_entries(entries: &Bound<'_, PyList>) -> PyResult<Vec<(u64, Vec<f32>)>> {
    let mut batch = Vec::with_capacity(entries.len());
    for item in entries.iter() {
        let tuple = item.downcast::<PyTuple>()
            .map_err(|_| PyValueError::new_err("each entry must be a tuple of (id, vector)"))?;
        if tuple.len() < 2 {
            return Err(PyValueError::new_err("each entry must be a tuple of (id, vector)"));
        }
        let id = tuple.get_item(0)?.extract::<u64>()?;
        let vector = extract_f32_vec(&tuple.get_item(1)?)?;
        batch.push((id, vector));
    }
    Ok(batch)
}

/// Extract a contiguous f32 buffer from a 2D numpy array / buffer.
/// Returns (data, n_rows, n_cols) where data is a flat Vec<f32> in row-major order.
#[cfg(any(not(Py_LIMITED_API), Py_3_11))]
#[inline]
fn extract_2d_f32_buffer(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f32>, usize, usize)> {
    let buf = pyo3::buffer::PyBuffer::<f32>::get_bound(obj)
        .map_err(|_| PyValueError::new_err("vectors must support the buffer protocol (e.g. numpy array)"))?;
    let ndim = buf.dimensions();
    if ndim != 2 {
        return Err(PyValueError::new_err(format!(
            "vectors must be 2-dimensional, got {ndim}-dimensional"
        )));
    }
    let shape = buf.shape();
    let n_rows = shape[0];
    let n_cols = shape[1];
    let total = n_rows * n_cols;
    let mut data = vec![0.0f32; total];
    buf.copy_to_slice(obj.py(), &mut data)
        .map_err(|e| PyValueError::new_err(format!("buffer copy failed: {e}")))?;
    Ok((data, n_rows, n_cols))
}

/// Extract a Vec<f32> from a Python object.
///
/// When the buffer protocol is available (PyO3 with non-limited API or Python ≥ 3.11),
/// this uses PEP 3118 for numpy arrays / memoryview / ctypes arrays (fast memcpy).
/// Otherwise falls back to PyO3's element-by-element extraction.
///
/// For a 128-dim vector:
///   - numpy array via buffer protocol: ~100ns (single memcpy of 512 bytes)
///   - Python list via PyO3 extract:    ~5µs  (128 × PyFloat_AsDouble calls)
#[inline]
fn extract_f32_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    // Try buffer protocol when available (non-limited-API or Python ≥ 3.11)
    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
    {
        if let Ok(buf) = pyo3::buffer::PyBuffer::<f32>::get_bound(obj) {
            if buf.dimensions() != 1 {
                return Err(PyValueError::new_err("vector must be 1-dimensional"));
            }
            let len = buf.item_count();
            let mut vec = vec![0.0f32; len];
            buf.copy_to_slice(obj.py(), &mut vec)
                .map_err(|e| PyValueError::new_err(format!("buffer copy failed: {e}")))?;
            return Ok(vec);
        }
    }

    // Fallback: extract from Python list/sequence
    obj.extract::<Vec<f32>>()
}
use quiver_core::{
    collection::{CollectionMeta, IndexType},
    distance::Metric,
    error::VectorDbError,
    index::{
        binary_flat::BinaryFlatIndex,
        flat::FlatIndex,
        hnsw::{HnswConfig, HnswIndex},
        ivf::{IvfConfig, IvfIndex},
        ivf_pq::{IvfPqConfig, IvfPqIndex},
        mmap_flat::MmapFlatIndex,
        quantized_flat::QuantizedFlatIndex,
        quantized_fp16::Fp16FlatIndex,
        sparse::SparseVector,
        VectorIndex,
    },
    manager::CollectionManager,
    payload::FilterCondition,
};

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
        VectorDbError::CollectionAlreadyExists(name) => {
            PyKeyError::new_err(format!("collection already exists: {name}"))
        }
        VectorDbError::CollectionNotFound(name) => {
            PyKeyError::new_err(format!("collection not found: {name}"))
        }
        VectorDbError::WalCorruption { entry, reason } => {
            PyRuntimeError::new_err(format!("WAL corruption at entry {entry}: {reason}"))
        }
        VectorDbError::EmbeddingError(msg) => {
            PyRuntimeError::new_err(format!("embedding error: {msg}"))
        }
        VectorDbError::NoEmbedder(name) => {
            PyRuntimeError::new_err(format!("no embedder configured for collection '{name}'"))
        }
        VectorDbError::SnapshotAlreadyExists(name) => {
            PyKeyError::new_err(format!("snapshot already exists: {name}"))
        }
        VectorDbError::SnapshotNotFound(name) => {
            PyKeyError::new_err(format!("snapshot not found: {name}"))
        }
    }
}

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

fn py_to_json(val: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if val.is_none() {
        Ok(serde_json::Value::Null)
    } else if let Ok(b) = val.extract::<bool>() {
        Ok(serde_json::Value::Bool(b))
    } else if let Ok(i) = val.extract::<i64>() {
        Ok(serde_json::Value::Number(i.into()))
    } else if let Ok(f) = val.extract::<f64>() {
        Ok(serde_json::Number::from_f64(f).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null))
    } else if let Ok(s) = val.extract::<String>() {
        Ok(serde_json::Value::String(s))
    } else if let Ok(dict) = val.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key = k.extract::<String>()?;
            map.insert(key, py_to_json(&v)?);
        }
        Ok(serde_json::Value::Object(map))
    } else if let Ok(list) = val.downcast::<PyList>() {
        let arr: PyResult<Vec<_>> = list.iter().map(|item| py_to_json(&item)).collect();
        Ok(serde_json::Value::Array(arr?))
    } else {
        Ok(serde_json::Value::String(val.str()?.to_string()))
    }
}

fn json_to_py(py: Python<'_>, val: &serde_json::Value) -> PyResult<PyObject> {
    use pyo3::IntoPy;
    match val {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.into_py(py)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_py(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_py(py))
            } else {
                Ok(n.to_string().into_py(py))
            }
        }
        serde_json::Value::String(s) => Ok(s.into_py(py)),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty_bound(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.into())
        }
    }
}

fn py_to_filter(val: &Bound<'_, PyAny>) -> PyResult<FilterCondition> {
    let json_val = py_to_json(val)?;
    serde_json::from_value(json_val)
        .map_err(|e| PyValueError::new_err(format!("invalid filter: {e}")))
}

/// Convert a Python dict `{int: float, ...}` to a `SparseVector`.
fn py_dict_to_sparse(dict: &Bound<'_, PyDict>) -> PyResult<SparseVector> {
    let mut map: HashMap<u32, f32> = HashMap::new();
    for (k, v) in dict.iter() {
        let idx = k.extract::<u32>()?;
        let val = v.extract::<f32>()?;
        map.insert(idx, val);
    }
    Ok(SparseVector::from_map(&map))
}

/// Exact brute-force vector index. 100% recall, O(N*D) per query.
///
/// Best for small datasets (<100K vectors) or when exact results are required.
///
/// Args:
///     dimensions: Number of dimensions per vector.
///     metric: Distance metric - "l2", "cosine", or "dot_product".
///
/// Example:
///     >>> idx = FlatIndex(dimensions=384, metric="cosine")
///     >>> idx.add(id=1, vector=[0.1, 0.2, ...])
///     >>> results = idx.search(query=[0.1, 0.2, ...], k=10)
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
        Ok(Self { inner: FlatIndex::new(dimensions, m) })
    }

    /// Add a single vector with the given ID.
    #[pyo3(signature = (id, vector))]
    fn add(&mut self, id: u64, vector: &Bound<'_, PyAny>) -> PyResult<()> {
        let v = extract_f32_vec(vector)?;
        self.inner.add(id, &v).map_err(vec_err_to_py)
    }

    /// Add multiple vectors at once. Takes a list of (id, vector) tuples.
    fn add_batch(&mut self, entries: &Bound<'_, PyList>) -> PyResult<()> {
        let batch = extract_batch_entries(entries)?;
        self.inner.add_batch(&batch).map_err(vec_err_to_py)
    }

    /// Add vectors from a 2D numpy array (N × dim) with sequential IDs starting from `start_id`.
    /// This is the fastest way to insert vectors — no per-element Python extraction.
    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
    #[pyo3(signature = (vectors, start_id = 0))]
    fn add_batch_np(&mut self, py: Python<'_>, vectors: &Bound<'_, PyAny>, start_id: u64) -> PyResult<()> {
        let (data, _n_rows, n_cols) = extract_2d_f32_buffer(vectors)?;
        py.allow_threads(|| {
            self.inner.add_batch_raw(&data, n_cols, start_id).map_err(vec_err_to_py)
        })
    }

    /// Search for the k nearest vectors. Returns list of {"id": int, "distance": float}.
    fn search(&self, py: Python<'_>, query: &Bound<'_, PyAny>, k: usize) -> PyResult<Vec<PyObject>> {
        let q = extract_f32_vec(query)?;
        let results = py.allow_threads(|| self.inner.search(&q, k)).map_err(vec_err_to_py)?;
        results.into_iter().map(|r| {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", r.id)?;
            dict.set_item("distance", r.distance)?;
            Ok(dict.into())
        }).collect()
    }

    /// Remove a vector by ID. Returns True if found and removed.
    fn delete(&mut self, id: u64) -> bool { self.inner.delete(id) }
    fn __len__(&self) -> usize { self.inner.len() }

    /// Number of dimensions per vector.
    #[getter]
    fn dimensions(&self) -> usize { self.inner.config().dimensions }

    /// Distance metric ("l2", "cosine", or "dot_product").
    #[getter]
    fn metric(&self) -> &'static str { metric_to_str(self.inner.config().metric) }

    /// Save index to a binary file.
    fn save(&self, path: &str) -> PyResult<()> { self.inner.save(path).map_err(vec_err_to_py) }

    /// Load a previously saved index from a binary file.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = FlatIndex::load(path).map_err(vec_err_to_py)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("FlatIndex(dimensions={}, metric=\"{}\", len={})",
            self.inner.config().dimensions, metric_to_str(self.inner.config().metric), self.inner.len())
    }
}

/// Graph-based approximate nearest-neighbour index (HNSW). 95-99% recall.
///
/// Best general-purpose index for datasets of any size.
///
/// Args:
///     dimensions: Number of dimensions per vector.
///     metric: Distance metric - "l2", "cosine", or "dot_product".
///     ef_construction: Build beam width (default 200). Higher = better recall, slower build.
///     ef_search: Query beam width (default 50). Tunable at runtime.
///     m: Graph edges per node (default 12). Higher = better recall, more RAM.
///
/// Example:
///     >>> hnsw = HnswIndex(dimensions=384, metric="cosine")
///     >>> hnsw.add(id=1, vector=[...])
///     >>> hnsw.flush()  # build graph after bulk inserts
///     >>> results = hnsw.search(query=[...], k=10)
#[pyclass(name = "HnswIndex")]
struct PyHnswIndex {
    inner: HnswIndex,
}

#[pymethods]
impl PyHnswIndex {
    #[new]
    #[pyo3(signature = (dimensions, metric = "l2", ef_construction = 200, ef_search = 50, m = 12))]
    fn new(dimensions: usize, metric: &str, ef_construction: usize, ef_search: usize, m: usize) -> PyResult<Self> {
        let met = parse_metric(metric)?;
        let cfg = HnswConfig { ef_construction, ef_search, m };
        Ok(Self { inner: HnswIndex::new(dimensions, met, cfg) })
    }

    #[pyo3(signature = (id, vector))]
    fn add(&mut self, id: u64, vector: &Bound<'_, PyAny>) -> PyResult<()> {
        let v = extract_f32_vec(vector)?;
        self.inner.add(id, &v).map_err(vec_err_to_py)
    }

    fn add_batch(&mut self, entries: &Bound<'_, PyList>) -> PyResult<()> {
        let batch = extract_batch_entries(entries)?;
        self.inner.add_batch(&batch).map_err(vec_err_to_py)
    }

    /// Add vectors from a 2D numpy array (N × dim) with sequential IDs starting from `start_id`.
    /// This is the fastest way to insert vectors — no per-element Python extraction.
    ///
    /// Usage:
    ///     vectors = np.random.randn(10000, 128).astype(np.float32)
    ///     idx.add_batch_np(vectors)            # IDs: 0..9999
    ///     idx.add_batch_np(vectors, start_id=5000)  # IDs: 5000..14999
    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
    #[pyo3(signature = (vectors, start_id = 0))]
    fn add_batch_np(&mut self, py: Python<'_>, vectors: &Bound<'_, PyAny>, start_id: u64) -> PyResult<()> {
        let (data, _n_rows, n_cols) = extract_2d_f32_buffer(vectors)?;
        py.allow_threads(|| {
            self.inner.add_batch_raw(&data, n_cols, start_id).map_err(vec_err_to_py)
        })
    }

    /// Parallel batch insert from a 2D numpy array using micro-batching.
    ///
    /// Splits the batch into micro-batches, inserts the first batch sequentially
    /// (bootstrapping the graph), then processes remaining batches with parallel
    /// neighbor search using `num_threads` threads.
    ///
    /// Args:
    ///     vectors: 2D numpy array of shape (N, dim), dtype float32
    ///     start_id: First vector ID (default 0)
    ///     num_threads: Number of threads (default 0 = all cores)
    ///     micro_batch_size: Vectors per micro-batch (default 256)
    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
    #[pyo3(signature = (vectors, start_id = 0, num_threads = 0, micro_batch_size = 256))]
    fn add_batch_parallel(
        &mut self,
        py: Python<'_>,
        vectors: &Bound<'_, PyAny>,
        start_id: u64,
        num_threads: usize,
        micro_batch_size: usize,
    ) -> PyResult<()> {
        let (data, _n_rows, n_cols) = extract_2d_f32_buffer(vectors)?;
        py.allow_threads(|| {
            self.inner.add_batch_parallel(&data, n_cols, start_id, num_threads, micro_batch_size)
                .map_err(vec_err_to_py)
        })
    }

    fn search(&self, py: Python<'_>, query: &Bound<'_, PyAny>, k: usize) -> PyResult<Vec<PyObject>> {
        let q = extract_f32_vec(query)?;
        let results = py.allow_threads(|| self.inner.search(&q, k)).map_err(vec_err_to_py)?;
        results.into_iter().map(|r| {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", r.id)?;
            dict.set_item("distance", r.distance)?;
            Ok(dict.into())
        }).collect()
    }

    fn delete(&mut self, id: u64) -> bool { self.inner.delete(id) }

    /// Build the HNSW graph. Call after bulk inserts for best performance.
    fn flush(&mut self, py: Python<'_>) { py.allow_threads(|| self.inner.flush()); }

    fn __len__(&self) -> usize { self.inner.len() }

    #[getter]
    fn dimensions(&self) -> usize { self.inner.config().dimensions }

    #[getter]
    fn metric(&self) -> &'static str { metric_to_str(self.inner.config().metric) }

    fn save(&self, path: &str) -> PyResult<()> { self.inner.save(path).map_err(vec_err_to_py) }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = HnswIndex::load(path).map_err(vec_err_to_py)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("HnswIndex(dimensions={}, metric=\"{}\", len={})",
            self.inner.config().dimensions, metric_to_str(self.inner.config().metric), self.inner.len())
    }
}

// ── QuantizedFlatIndex ────────────────────────────────────────────────────────

/// Int8 quantized brute-force index. ~4x less RAM than FlatIndex with ~99% recall.
///
/// Each f32 component is quantized to int8 using min/max scaling.
/// Same API as FlatIndex.
#[pyclass(name = "QuantizedFlatIndex")]
struct PyQuantizedFlatIndex {
    inner: QuantizedFlatIndex,
}

#[pymethods]
impl PyQuantizedFlatIndex {
    #[new]
    #[pyo3(signature = (dimensions, metric = "l2"))]
    fn new(dimensions: usize, metric: &str) -> PyResult<Self> {
        let m = parse_metric(metric)?;
        Ok(Self { inner: QuantizedFlatIndex::new(dimensions, m) })
    }

    #[pyo3(signature = (id, vector))]
    fn add(&mut self, id: u64, vector: &Bound<'_, PyAny>) -> PyResult<()> {
        let v = extract_f32_vec(vector)?;
        self.inner.add(id, &v).map_err(vec_err_to_py)
    }

    fn add_batch(&mut self, entries: &Bound<'_, PyList>) -> PyResult<()> {
        let batch = extract_batch_entries(entries)?;
        self.inner.add_batch(&batch).map_err(vec_err_to_py)
    }

    /// Add vectors from a 2D numpy array (N × dim) with sequential IDs starting from `start_id`.
    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
    #[pyo3(signature = (vectors, start_id = 0))]
    fn add_batch_np(&mut self, py: Python<'_>, vectors: &Bound<'_, PyAny>, start_id: u64) -> PyResult<()> {
        let (data, _n_rows, n_cols) = extract_2d_f32_buffer(vectors)?;
        py.allow_threads(|| {
            self.inner.add_batch_raw(&data, n_cols, start_id).map_err(vec_err_to_py)
        })
    }

    fn search(&self, py: Python<'_>, query: &Bound<'_, PyAny>, k: usize) -> PyResult<Vec<PyObject>> {
        let q = extract_f32_vec(query)?;
        let results = py.allow_threads(|| self.inner.search(&q, k)).map_err(vec_err_to_py)?;
        results.into_iter().map(|r| {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", r.id)?;
            dict.set_item("distance", r.distance)?;
            Ok(dict.into())
        }).collect()
    }

    fn delete(&mut self, id: u64) -> bool { self.inner.delete(id) }
    fn __len__(&self) -> usize { self.inner.len() }

    #[getter]
    fn dimensions(&self) -> usize { self.inner.config().dimensions }

    #[getter]
    fn metric(&self) -> &'static str { metric_to_str(self.inner.config().metric) }

    fn save(&self, path: &str) -> PyResult<()> { self.inner.save(path).map_err(vec_err_to_py) }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = QuantizedFlatIndex::load(path).map_err(vec_err_to_py)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("QuantizedFlatIndex(dimensions={}, metric=\"{}\", len={})",
            self.inner.config().dimensions, metric_to_str(self.inner.config().metric), self.inner.len())
    }
}

// ── Fp16FlatIndex ────────────────────────────────────────────────────────────

/// Float16 quantized brute-force index. 2x less RAM than FlatIndex with >99.5% recall.
///
/// Vectors are stored as half-precision floats and decoded on-the-fly for distance computation.
/// Same API as FlatIndex.
#[pyclass(name = "Fp16FlatIndex")]
struct PyFp16FlatIndex {
    inner: Fp16FlatIndex,
}

#[pymethods]
impl PyFp16FlatIndex {
    #[new]
    #[pyo3(signature = (dimensions, metric = "l2"))]
    fn new(dimensions: usize, metric: &str) -> PyResult<Self> {
        let m = parse_metric(metric)?;
        Ok(Self { inner: Fp16FlatIndex::new(dimensions, m) })
    }

    #[pyo3(signature = (id, vector))]
    fn add(&mut self, id: u64, vector: &Bound<'_, PyAny>) -> PyResult<()> {
        let v = extract_f32_vec(vector)?;
        self.inner.add(id, &v).map_err(vec_err_to_py)
    }

    fn add_batch(&mut self, entries: &Bound<'_, PyList>) -> PyResult<()> {
        let batch = extract_batch_entries(entries)?;
        self.inner.add_batch(&batch).map_err(vec_err_to_py)
    }

    /// Add vectors from a 2D numpy array (N × dim) with sequential IDs starting from `start_id`.
    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
    #[pyo3(signature = (vectors, start_id = 0))]
    fn add_batch_np(&mut self, py: Python<'_>, vectors: &Bound<'_, PyAny>, start_id: u64) -> PyResult<()> {
        let (data, _n_rows, n_cols) = extract_2d_f32_buffer(vectors)?;
        py.allow_threads(|| {
            self.inner.add_batch_raw(&data, n_cols, start_id).map_err(vec_err_to_py)
        })
    }

    fn search(&self, py: Python<'_>, query: &Bound<'_, PyAny>, k: usize) -> PyResult<Vec<PyObject>> {
        let q = extract_f32_vec(query)?;
        let results = py.allow_threads(|| self.inner.search(&q, k)).map_err(vec_err_to_py)?;
        results.into_iter().map(|r| {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", r.id)?;
            dict.set_item("distance", r.distance)?;
            Ok(dict.into())
        }).collect()
    }

    fn delete(&mut self, id: u64) -> bool { self.inner.delete(id) }
    fn __len__(&self) -> usize { self.inner.len() }

    #[getter]
    fn dimensions(&self) -> usize { self.inner.config().dimensions }

    #[getter]
    fn metric(&self) -> &'static str { metric_to_str(self.inner.config().metric) }

    fn save(&self, path: &str) -> PyResult<()> { self.inner.save(path).map_err(vec_err_to_py) }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = Fp16FlatIndex::load(path).map_err(vec_err_to_py)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("Fp16FlatIndex(dimensions={}, metric=\"{}\", len={})",
            self.inner.config().dimensions, metric_to_str(self.inner.config().metric), self.inner.len())
    }
}

// ── IvfIndex ─────────────────────────────────────────────────────────────────

/// Inverted file index. Cluster-based ANN using k-means.
///
/// Auto-trains after train_size inserts. Rule of thumb: n_lists = sqrt(N).
///
/// Args:
///     dimensions: Number of dimensions per vector.
///     metric: Distance metric - "l2", "cosine", or "dot_product".
///     n_lists: Number of clusters (default 256).
///     nprobe: Clusters scanned per query (default 16).
///     train_size: Vectors buffered before auto-training (default 4096).
#[pyclass(name = "IvfIndex")]
struct PyIvfIndex {
    inner: IvfIndex,
}

#[pymethods]
impl PyIvfIndex {
    #[new]
    #[pyo3(signature = (dimensions, metric = "l2", n_lists = 256, nprobe = 16, train_size = 4096))]
    fn new(dimensions: usize, metric: &str, n_lists: usize, nprobe: usize, train_size: usize) -> PyResult<Self> {
        let m = parse_metric(metric)?;
        let cfg = IvfConfig { n_lists, nprobe, train_size, max_iter: 25 };
        Ok(Self { inner: IvfIndex::new(dimensions, m, cfg) })
    }

    #[pyo3(signature = (id, vector))]
    fn add(&mut self, id: u64, vector: &Bound<'_, PyAny>) -> PyResult<()> {
        let v = extract_f32_vec(vector)?;
        self.inner.add(id, &v).map_err(vec_err_to_py)
    }

    fn add_batch(&mut self, entries: &Bound<'_, PyList>) -> PyResult<()> {
        let batch = extract_batch_entries(entries)?;
        self.inner.add_batch(&batch).map_err(vec_err_to_py)
    }

    /// Add vectors from a 2D numpy array (N × dim) with sequential IDs starting from `start_id`.
    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
    #[pyo3(signature = (vectors, start_id = 0))]
    fn add_batch_np(&mut self, py: Python<'_>, vectors: &Bound<'_, PyAny>, start_id: u64) -> PyResult<()> {
        let (data, _n_rows, n_cols) = extract_2d_f32_buffer(vectors)?;
        py.allow_threads(|| {
            self.inner.add_batch_raw(&data, n_cols, start_id).map_err(vec_err_to_py)
        })
    }

    fn search(&self, py: Python<'_>, query: &Bound<'_, PyAny>, k: usize) -> PyResult<Vec<PyObject>> {
        let q = extract_f32_vec(query)?;
        let results = py.allow_threads(|| self.inner.search(&q, k)).map_err(vec_err_to_py)?;
        results.into_iter().map(|r| {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", r.id)?;
            dict.set_item("distance", r.distance)?;
            Ok(dict.into())
        }).collect()
    }

    fn delete(&mut self, id: u64) -> bool { self.inner.delete(id) }

    fn flush(&mut self, py: Python<'_>) { py.allow_threads(|| self.inner.flush()); }

    fn __len__(&self) -> usize { self.inner.len() }

    #[getter]
    fn dimensions(&self) -> usize { self.inner.config().dimensions }

    #[getter]
    fn metric(&self) -> &'static str { metric_to_str(self.inner.config().metric) }

    fn save(&self, path: &str) -> PyResult<()> { self.inner.save(path).map_err(vec_err_to_py) }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = IvfIndex::load(path).map_err(vec_err_to_py)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("IvfIndex(dimensions={}, metric=\"{}\", len={})",
            self.inner.config().dimensions, metric_to_str(self.inner.config().metric), self.inner.len())
    }
}

// ── IvfPqIndex ───────────────────────────────────────────────────────────────

/// IVF with product quantization. Extreme memory reduction (~96x for 1536-dim).
///
/// Combines IVF coarse quantizer with PQ compression for residual vectors.
/// Ideal for million-scale datasets.
///
/// Args:
///     dimensions: Number of dimensions per vector.
///     metric: Distance metric - "l2", "cosine", or "dot_product".
///     n_lists: Number of IVF clusters (default 256).
///     nprobe: Clusters scanned per query (default 16).
///     train_size: Vectors buffered before auto-training (default 4096).
///     pq_m: Number of PQ sub-quantizers (default 8, must divide dimensions).
///     pq_k_sub: Centroids per sub-quantizer (default 256).
#[pyclass(name = "IvfPqIndex")]
struct PyIvfPqIndex {
    inner: IvfPqIndex,
}

#[pymethods]
impl PyIvfPqIndex {
    #[new]
    #[pyo3(signature = (dimensions, metric = "l2", n_lists = 256, nprobe = 16, train_size = 4096, pq_m = 8, pq_k_sub = 256))]
    fn new(
        dimensions: usize, metric: &str,
        n_lists: usize, nprobe: usize, train_size: usize,
        pq_m: usize, pq_k_sub: usize,
    ) -> PyResult<Self> {
        let m = parse_metric(metric)?;
        use quiver_core::index::pq::PqConfig;
        let cfg = IvfPqConfig {
            n_lists, nprobe, train_size, max_iter: 25,
            pq: PqConfig { m: pq_m, k_sub: pq_k_sub, max_iter: 25 },
        };
        Ok(Self { inner: IvfPqIndex::new(dimensions, m, cfg) })
    }

    #[pyo3(signature = (id, vector))]
    fn add(&mut self, id: u64, vector: &Bound<'_, PyAny>) -> PyResult<()> {
        let v = extract_f32_vec(vector)?;
        self.inner.add(id, &v).map_err(vec_err_to_py)
    }

    fn add_batch(&mut self, entries: &Bound<'_, PyList>) -> PyResult<()> {
        let batch = extract_batch_entries(entries)?;
        self.inner.add_batch(&batch).map_err(vec_err_to_py)
    }

    /// Add vectors from a 2D numpy array (N × dim) with sequential IDs starting from `start_id`.
    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
    #[pyo3(signature = (vectors, start_id = 0))]
    fn add_batch_np(&mut self, py: Python<'_>, vectors: &Bound<'_, PyAny>, start_id: u64) -> PyResult<()> {
        let (data, _n_rows, n_cols) = extract_2d_f32_buffer(vectors)?;
        py.allow_threads(|| {
            self.inner.add_batch_raw(&data, n_cols, start_id).map_err(vec_err_to_py)
        })
    }

    fn search(&self, py: Python<'_>, query: &Bound<'_, PyAny>, k: usize) -> PyResult<Vec<PyObject>> {
        let q = extract_f32_vec(query)?;
        let results = py.allow_threads(|| self.inner.search(&q, k)).map_err(vec_err_to_py)?;
        results.into_iter().map(|r| {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", r.id)?;
            dict.set_item("distance", r.distance)?;
            Ok(dict.into())
        }).collect()
    }

    fn delete(&mut self, id: u64) -> bool { self.inner.delete(id) }

    fn flush(&mut self, py: Python<'_>) { py.allow_threads(|| self.inner.flush()); }

    fn __len__(&self) -> usize { self.inner.len() }

    #[getter]
    fn dimensions(&self) -> usize { self.inner.config().dimensions }

    #[getter]
    fn metric(&self) -> &'static str { metric_to_str(self.inner.config().metric) }

    fn save(&self, path: &str) -> PyResult<()> { self.inner.save(path).map_err(vec_err_to_py) }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = IvfPqIndex::load(path).map_err(vec_err_to_py)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("IvfPqIndex(dimensions={}, metric=\"{}\", len={})",
            self.inner.config().dimensions, metric_to_str(self.inner.config().metric), self.inner.len())
    }
}

// ── MmapFlatIndex ────────────────────────────────────────────────────────────

/// Memory-mapped brute-force index. Near-zero RAM usage.
///
/// Vectors are stored in a flat binary file and memory-mapped by the OS.
/// Pages are loaded on demand. Best when dataset is larger than available RAM.
///
/// Args:
///     dimensions: Number of dimensions per vector.
///     metric: Distance metric - "l2", "cosine", or "dot_product".
///     path: Path to the memory-mapped vector file.
#[pyclass(name = "MmapFlatIndex")]
struct PyMmapFlatIndex {
    inner: MmapFlatIndex,
}

#[pymethods]
impl PyMmapFlatIndex {
    #[new]
    #[pyo3(signature = (dimensions, metric = "l2", path = "./mmap_index.qvec"))]
    fn new(dimensions: usize, metric: &str, path: &str) -> PyResult<Self> {
        let m = parse_metric(metric)?;
        let inner = MmapFlatIndex::new(dimensions, m, path).map_err(vec_err_to_py)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (id, vector))]
    fn add(&mut self, id: u64, vector: &Bound<'_, PyAny>) -> PyResult<()> {
        let v = extract_f32_vec(vector)?;
        self.inner.add(id, &v).map_err(vec_err_to_py)
    }

    fn add_batch(&mut self, entries: &Bound<'_, PyList>) -> PyResult<()> {
        let batch = extract_batch_entries(entries)?;
        self.inner.add_batch(&batch).map_err(vec_err_to_py)
    }

    /// Add vectors from a 2D numpy array (N × dim) with sequential IDs starting from `start_id`.
    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
    #[pyo3(signature = (vectors, start_id = 0))]
    fn add_batch_np(&mut self, py: Python<'_>, vectors: &Bound<'_, PyAny>, start_id: u64) -> PyResult<()> {
        let (data, _n_rows, n_cols) = extract_2d_f32_buffer(vectors)?;
        py.allow_threads(|| {
            self.inner.add_batch_raw(&data, n_cols, start_id).map_err(vec_err_to_py)
        })
    }

    fn search(&self, py: Python<'_>, query: &Bound<'_, PyAny>, k: usize) -> PyResult<Vec<PyObject>> {
        let q = extract_f32_vec(query)?;
        let results = py.allow_threads(|| self.inner.search(&q, k)).map_err(vec_err_to_py)?;
        results.into_iter().map(|r| {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", r.id)?;
            dict.set_item("distance", r.distance)?;
            Ok(dict.into())
        }).collect()
    }

    fn delete(&mut self, id: u64) -> bool { self.inner.delete(id) }

    fn flush(&mut self, py: Python<'_>) { py.allow_threads(|| self.inner.flush()); }

    fn __len__(&self) -> usize { self.inner.len() }

    #[getter]
    fn dimensions(&self) -> usize { self.inner.config().dimensions }

    #[getter]
    fn metric(&self) -> &'static str { metric_to_str(self.inner.config().metric) }

    fn __repr__(&self) -> String {
        format!("MmapFlatIndex(dimensions={}, metric=\"{}\", len={})",
            self.inner.config().dimensions, metric_to_str(self.inner.config().metric), self.inner.len())
    }
}

// ── BinaryFlatIndex ──────────────────────────────────────────────────────────

/// Binary (1-bit) quantized brute-force index. 32x less RAM than FlatIndex.
///
/// Each f32 component is reduced to 1 bit (positive=1, negative=0).
/// Distance is computed via Hamming distance (popcount).
/// Same API as FlatIndex.
#[pyclass(name = "BinaryFlatIndex")]
struct PyBinaryFlatIndex {
    inner: BinaryFlatIndex,
}

#[pymethods]
impl PyBinaryFlatIndex {
    #[new]
    #[pyo3(signature = (dimensions, metric = "l2"))]
    fn new(dimensions: usize, metric: &str) -> PyResult<Self> {
        let m = parse_metric(metric)?;
        Ok(Self { inner: BinaryFlatIndex::new(dimensions, m) })
    }

    #[pyo3(signature = (id, vector))]
    fn add(&mut self, id: u64, vector: &Bound<'_, PyAny>) -> PyResult<()> {
        let v = extract_f32_vec(vector)?;
        self.inner.add(id, &v).map_err(vec_err_to_py)
    }

    fn add_batch(&mut self, entries: &Bound<'_, PyList>) -> PyResult<()> {
        let batch = extract_batch_entries(entries)?;
        self.inner.add_batch(&batch).map_err(vec_err_to_py)
    }

    /// Add vectors from a 2D numpy array (N × dim) with sequential IDs starting from `start_id`.
    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
    #[pyo3(signature = (vectors, start_id = 0))]
    fn add_batch_np(&mut self, py: Python<'_>, vectors: &Bound<'_, PyAny>, start_id: u64) -> PyResult<()> {
        let (data, _n_rows, n_cols) = extract_2d_f32_buffer(vectors)?;
        py.allow_threads(|| {
            self.inner.add_batch_raw(&data, n_cols, start_id).map_err(vec_err_to_py)
        })
    }

    fn search(&self, py: Python<'_>, query: &Bound<'_, PyAny>, k: usize) -> PyResult<Vec<PyObject>> {
        let q = extract_f32_vec(query)?;
        let results = py.allow_threads(|| self.inner.search(&q, k)).map_err(vec_err_to_py)?;
        results.into_iter().map(|r| {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", r.id)?;
            dict.set_item("distance", r.distance)?;
            Ok(dict.into())
        }).collect()
    }

    fn delete(&mut self, id: u64) -> bool { self.inner.delete(id) }
    fn __len__(&self) -> usize { self.inner.len() }

    #[getter]
    fn dimensions(&self) -> usize { self.inner.config().dimensions }

    #[getter]
    fn metric(&self) -> &'static str { metric_to_str(self.inner.config().metric) }

    fn save(&self, path: &str) -> PyResult<()> { self.inner.save(path).map_err(vec_err_to_py) }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = BinaryFlatIndex::load(path).map_err(vec_err_to_py)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("BinaryFlatIndex(dimensions={}, metric=\"{}\", len={})",
            self.inner.config().dimensions, metric_to_str(self.inner.config().metric), self.inner.len())
    }
}

// ── Collection ───────────────────────────────────────────────────────────────

/// A named collection of vectors with WAL-backed persistence.
///
/// Obtained via Client.create_collection() or Client.get_collection().
/// Supports upsert, search, delete, hybrid search, and payload filtering.
#[pyclass(name = "Collection")]
struct PyCollection {
    manager: Arc<RwLock<CollectionManager>>,
    name: String,
}

#[pymethods]
impl PyCollection {
    /// Insert or update a vector by ID with optional metadata payload.
    #[pyo3(signature = (id, vector, payload=None))]
    fn upsert(&mut self, py: Python<'_>, id: u64, vector: &Bound<'_, PyAny>, payload: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let v = extract_f32_vec(vector)?;
        let p = match payload {
            Some(v) if !v.is_none() => Some(py_to_json(v)?),
            _ => None,
        };
        let name = self.name.clone();
        let mut mgr = self.manager.write().unwrap();
        let col = mgr.get_collection_mut(&name)
            .ok_or_else(|| PyKeyError::new_err(format!("collection '{name}' not found")))?;
        py.allow_threads(|| col.upsert(id, v, p)).map_err(vec_err_to_py)
    }

    /// Batch insert or update multiple vectors at once. More efficient than
    /// calling upsert in a loop.
    ///
    /// Args:
    ///     entries: List of (id, vector) tuples or (id, vector, payload) tuples.
    #[pyo3(signature = (entries))]
    fn upsert_batch(&mut self, py: Python<'_>, entries: &Bound<'_, PyList>) -> PyResult<()> {
        let mut batch: Vec<(u64, Vec<f32>, Option<serde_json::Value>)> = Vec::with_capacity(entries.len());
        for item in entries.iter() {
            let tuple = item.downcast::<PyTuple>()
                .map_err(|_| PyValueError::new_err("each entry must be a tuple of (id, vector) or (id, vector, payload)"))?;
            let len = tuple.len();
            if len < 2 || len > 3 {
                return Err(PyValueError::new_err(
                    "each entry must be a tuple of (id, vector) or (id, vector, payload)"
                ));
            }
            let id = tuple.get_item(0)?.extract::<u64>()?;
            let vector = extract_f32_vec(&tuple.get_item(1)?)?;
            let payload = if len == 3 {
                let p = tuple.get_item(2)?;
                if p.is_none() { None } else { Some(py_to_json(&p)?) }
            } else {
                None
            };
            batch.push((id, vector, payload));
        }
        let name = self.name.clone();
        let mut mgr = self.manager.write().unwrap();
        let col = mgr.get_collection_mut(&name)
            .ok_or_else(|| PyKeyError::new_err(format!("collection '{name}' not found")))?;
        py.allow_threads(|| col.upsert_batch(batch)).map_err(vec_err_to_py)
    }

    /// Search for the k nearest vectors. Returns list of {"id", "distance", "payload"}.
    ///
    /// Supports optional payload filtering with operators: $eq, $ne, $in, $gt, $gte, $lt, $lte, $and, $or.
    #[pyo3(signature = (query, k, filter=None))]
    fn search(&self, py: Python<'_>, query: &Bound<'_, PyAny>, k: usize, filter: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<PyObject>> {
        let q = extract_f32_vec(query)?;
        let f = match filter {
            Some(v) if !v.is_none() => Some(py_to_filter(v)?),
            _ => None,
        };
        let name = self.name.clone();
        let mgr = self.manager.read().unwrap();
        let col = mgr.get_collection(&name)
            .ok_or_else(|| PyKeyError::new_err(format!("collection '{name}' not found")))?;
        let results = py.allow_threads(|| col.search(&q, k, f.as_ref())).map_err(vec_err_to_py)?;
        results.into_iter().map(|r| {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", r.id)?;
            dict.set_item("distance", r.distance)?;
            if let Some(p) = &r.payload {
                dict.set_item("payload", json_to_py(py, p)?)?;
            }
            Ok(dict.into())
        }).collect()
    }

    /// Upsert with an optional sparse vector for hybrid search.
    ///
    /// Args:
    ///     id: Vector ID.
    ///     vector: Dense vector.
    ///     sparse_vector: Optional dict `{dimension_index: weight}` for sparse search.
    ///     payload: Optional metadata dict.
    #[pyo3(signature = (id, vector, sparse_vector=None, payload=None))]
    fn upsert_hybrid(
        &mut self, py: Python<'_>,
        id: u64, vector: &Bound<'_, PyAny>,
        sparse_vector: Option<&Bound<'_, PyDict>>,
        payload: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let v = extract_f32_vec(vector)?;
        let sv = match sparse_vector {
            Some(d) => Some(py_dict_to_sparse(d)?),
            None => None,
        };
        let p = match payload {
            Some(v) if !v.is_none() => Some(py_to_json(v)?),
            _ => None,
        };
        let name = self.name.clone();
        let mut mgr = self.manager.write().unwrap();
        let col = mgr.get_collection_mut(&name)
            .ok_or_else(|| PyKeyError::new_err(format!("collection '{name}' not found")))?;
        py.allow_threads(|| col.upsert_hybrid(id, v, sv, p)).map_err(vec_err_to_py)
    }

    /// Hybrid dense+sparse search with weighted score fusion.
    ///
    /// Args:
    ///     dense_query: Dense query vector.
    ///     sparse_query: Sparse query dict `{dimension_index: weight}`.
    ///     k: Number of results.
    ///     dense_weight: Weight for dense similarity (default 0.7).
    ///     sparse_weight: Weight for sparse similarity (default 0.3).
    ///     filter: Optional payload filter dict.
    ///
    /// Returns:
    ///     List of dicts with keys: id, score, dense_distance, sparse_score, payload.
    #[pyo3(signature = (dense_query, sparse_query, k, dense_weight=0.7, sparse_weight=0.3, filter=None))]
    fn search_hybrid(
        &self, py: Python<'_>,
        dense_query: &Bound<'_, PyAny>,
        sparse_query: &Bound<'_, PyDict>,
        k: usize,
        dense_weight: f32,
        sparse_weight: f32,
        filter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<PyObject>> {
        let dq = extract_f32_vec(dense_query)?;
        let sq = py_dict_to_sparse(sparse_query)?;
        let f = match filter {
            Some(v) if !v.is_none() => Some(py_to_filter(v)?),
            _ => None,
        };
        let name = self.name.clone();
        let mgr = self.manager.read().unwrap();
        let col = mgr.get_collection(&name)
            .ok_or_else(|| PyKeyError::new_err(format!("collection '{name}' not found")))?;
        let results = py.allow_threads(|| {
            col.search_hybrid(&dq, &sq, k, dense_weight, sparse_weight, f.as_ref())
        }).map_err(vec_err_to_py)?;
        results.into_iter().map(|r| {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", r.id)?;
            dict.set_item("score", r.score)?;
            dict.set_item("dense_distance", r.dense_distance)?;
            dict.set_item("sparse_score", r.sparse_score)?;
            if let Some(p) = &r.payload {
                dict.set_item("payload", json_to_py(py, p)?)?;
            }
            Ok(dict.into())
        }).collect()
    }

    /// Delete a vector by ID. Returns True if found and removed.
    fn delete(&mut self, py: Python<'_>, id: u64) -> PyResult<bool> {
        let name = self.name.clone();
        let mut mgr = self.manager.write().unwrap();
        let col = mgr.get_collection_mut(&name)
            .ok_or_else(|| PyKeyError::new_err(format!("collection '{name}' not found")))?;
        py.allow_threads(|| col.delete(id)).map_err(vec_err_to_py)
    }

    #[getter]
    fn count(&self) -> usize {
        let mgr = self.manager.read().unwrap();
        mgr.get_collection(&self.name).map(|c| c.count()).unwrap_or(0)
    }

    #[getter]
    fn sparse_count(&self) -> usize {
        let mgr = self.manager.read().unwrap();
        mgr.get_collection(&self.name).map(|c| c.sparse_count()).unwrap_or(0)
    }

    #[getter]
    fn name(&self) -> &str { &self.name }

    /// Create a named snapshot of this collection's current state.
    fn create_snapshot(&mut self, name: &str) -> PyResult<PyObject> {
        let col_name = self.name.clone();
        let mut mgr = self.manager.write().unwrap();
        let col = mgr.get_collection_mut(&col_name)
            .ok_or_else(|| PyKeyError::new_err(format!("collection '{col_name}' not found")))?;
        let meta = col.create_snapshot(name).map_err(vec_err_to_py)?;
        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);
            dict.set_item("name", &meta.name)?;
            dict.set_item("created_at", meta.created_at)?;
            dict.set_item("vector_count", meta.vector_count)?;
            dict.set_item("sparse_count", meta.sparse_count)?;
            Ok(dict.into())
        })
    }

    /// List all snapshots for this collection, sorted by creation time.
    fn list_snapshots(&self) -> PyResult<Vec<PyObject>> {
        let mgr = self.manager.read().unwrap();
        let col = mgr.get_collection(&self.name)
            .ok_or_else(|| PyKeyError::new_err(format!("collection '{}' not found", self.name)))?;
        let snaps = col.list_snapshots().map_err(vec_err_to_py)?;
        Python::with_gil(|py| {
            snaps.into_iter().map(|meta| {
                let dict = PyDict::new_bound(py);
                dict.set_item("name", &meta.name)?;
                dict.set_item("created_at", meta.created_at)?;
                dict.set_item("vector_count", meta.vector_count)?;
                dict.set_item("sparse_count", meta.sparse_count)?;
                Ok(dict.into())
            }).collect()
        })
    }

    /// Restore this collection to the state captured by snapshot `name`.
    fn restore_snapshot(&mut self, name: &str) -> PyResult<()> {
        let col_name = self.name.clone();
        let mut mgr = self.manager.write().unwrap();
        let col = mgr.get_collection_mut(&col_name)
            .ok_or_else(|| PyKeyError::new_err(format!("collection '{col_name}' not found")))?;
        col.restore_snapshot(name).map_err(vec_err_to_py)
    }

    /// Delete a snapshot by name.
    fn delete_snapshot(&mut self, name: &str) -> PyResult<()> {
        let col_name = self.name.clone();
        let mut mgr = self.manager.write().unwrap();
        let col = mgr.get_collection_mut(&col_name)
            .ok_or_else(|| PyKeyError::new_err(format!("collection '{col_name}' not found")))?;
        col.delete_snapshot(name).map_err(vec_err_to_py)
    }

    fn __repr__(&self) -> String {
        let count = self.count();
        format!("Collection(name=\"{}\", count={count})", self.name)
    }
}

/// Persistent vector database client.
///
/// Opens a directory on disk. All writes are WAL-backed and survive restarts.
///
/// Args:
///     path: Directory path for data storage (default "./data").
///
/// Example:
///     >>> db = Client(path="./my_data")
///     >>> col = db.create_collection("docs", dimensions=384, metric="cosine")
///     >>> col.upsert(id=1, vector=[...], payload={"title": "hello"})
///     >>> hits = col.search(query=[...], k=5)
#[pyclass(name = "Client")]
struct PyClient {
    manager: Arc<RwLock<CollectionManager>>,
}

#[pymethods]
impl PyClient {
    #[new]
    #[pyo3(signature = (path = "./data"))]
    fn new(path: &str) -> PyResult<Self> {
        let mgr = CollectionManager::open(Path::new(path)).map_err(vec_err_to_py)?;
        Ok(Self { manager: Arc::new(RwLock::new(mgr)) })
    }

    /// Create a new collection.
    ///
    /// Args:
    ///     name: Collection name.
    ///     dimensions: Vector dimensions.
    ///     metric: "cosine", "l2", or "dot_product".
    ///     index_type: "hnsw", "flat", "quantized_flat", "fp16_flat", "ivf", "ivf_pq", or "mmap_flat".
    #[pyo3(signature = (name, dimensions, metric = "cosine", index_type = "hnsw"))]
    fn create_collection(&mut self, name: String, dimensions: usize, metric: &str, index_type: &str) -> PyResult<PyCollection> {
        let m = parse_metric(metric)?;
        let (it, hnsw_config, ivf_config) = match index_type {
            "flat" => (IndexType::Flat, None, None),
            "hnsw" => (IndexType::Hnsw, Some(HnswConfig::default()), None),
            "quantized_flat" => (IndexType::QuantizedFlat, None, None),
            "fp16_flat" => (IndexType::Fp16Flat, None, None),
            "ivf" => (IndexType::Ivf, None, Some(IvfConfig::default())),
            "ivf_pq" => (IndexType::IvfPq, None, None),
            "mmap_flat" => (IndexType::MmapFlat, None, None),
            "binary_flat" => (IndexType::BinaryFlat, None, None),
            other => return Err(PyValueError::new_err(format!("unknown index_type {other:?}"))),
        };
        let ivf_pq_config = if it == IndexType::IvfPq { Some(IvfPqConfig::default()) } else { None };
        let meta = CollectionMeta { name: name.clone(), dimensions, metric: m, index_type: it, hnsw_config, wal_compact_threshold: 50_000, auto_promote_threshold: None, promotion_hnsw_config: None, embedding_model: None, ivf_config, ivf_pq_config, faiss_factory: None };
        let mut mgr = self.manager.write().unwrap();
        mgr.create_collection(meta).map_err(vec_err_to_py)?;
        Ok(PyCollection { manager: Arc::clone(&self.manager), name })
    }

    /// Get an existing collection by name. Raises KeyError if not found.
    fn get_collection(&self, name: &str) -> PyResult<PyCollection> {
        let mgr = self.manager.read().unwrap();
        if mgr.get_collection(name).is_none() {
            return Err(PyKeyError::new_err(format!("collection '{name}' not found")));
        }
        Ok(PyCollection { manager: Arc::clone(&self.manager), name: name.to_string() })
    }

    /// Get a collection by name, or create it if it doesn't exist (uses HNSW).
    #[pyo3(signature = (name, dimensions, metric = "cosine"))]
    fn get_or_create_collection(&mut self, name: &str, dimensions: usize, metric: &str) -> PyResult<PyCollection> {
        {
            let mgr = self.manager.read().unwrap();
            if mgr.get_collection(name).is_some() {
                return Ok(PyCollection { manager: Arc::clone(&self.manager), name: name.to_string() });
            }
        }
        self.create_collection(name.to_string(), dimensions, metric, "hnsw")
    }

    /// Delete a collection and all its data from disk.
    fn delete_collection(&mut self, name: &str) -> PyResult<bool> {
        let mut mgr = self.manager.write().unwrap();
        mgr.delete_collection(name).map_err(vec_err_to_py)
    }

    /// List all collection names.
    fn list_collections(&self) -> Vec<String> {
        let mgr = self.manager.read().unwrap();
        mgr.list_collections().iter().map(|m| m.name.clone()).collect()
    }

    fn __repr__(&self) -> String {
        let count = self.manager.read().unwrap().list_collections().len();
        format!("Client(collections={count})")
    }
}

#[pymodule]
fn quiver_vector_db(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFlatIndex>()?;
    m.add_class::<PyHnswIndex>()?;
    m.add_class::<PyQuantizedFlatIndex>()?;
    m.add_class::<PyFp16FlatIndex>()?;
    m.add_class::<PyIvfIndex>()?;
    m.add_class::<PyIvfPqIndex>()?;
    m.add_class::<PyMmapFlatIndex>()?;
    m.add_class::<PyBinaryFlatIndex>()?;
    m.add_class::<PyClient>()?;
    m.add_class::<PyCollection>()?;
    Ok(())
}
