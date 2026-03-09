use pyo3::exceptions::{PyIOError, PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::Path;
use std::sync::{Arc, Mutex};
use vectordb_core::{
    collection::{CollectionMeta, IndexType},
    distance::Metric,
    error::VectorDbError,
    index::{
        flat::FlatIndex,
        hnsw::{HnswConfig, HnswIndex},
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

    #[pyo3(signature = (id, vector))]
    fn add(&mut self, id: u64, vector: Vec<f32>) -> PyResult<()> {
        self.inner.add(id, &vector).map_err(vec_err_to_py)
    }

    fn add_batch(&mut self, entries: Vec<(u64, Vec<f32>)>) -> PyResult<()> {
        self.inner.add_batch(&entries).map_err(vec_err_to_py)
    }

    fn search(&self, py: Python<'_>, query: Vec<f32>, k: usize) -> PyResult<Vec<PyObject>> {
        let results = py.allow_threads(|| self.inner.search(&query, k)).map_err(vec_err_to_py)?;
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
        let inner = FlatIndex::load(path).map_err(vec_err_to_py)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("FlatIndex(dimensions={}, metric=\"{}\", len={})",
            self.inner.config().dimensions, metric_to_str(self.inner.config().metric), self.inner.len())
    }
}

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
    fn add(&mut self, id: u64, vector: Vec<f32>) -> PyResult<()> {
        self.inner.add(id, &vector).map_err(vec_err_to_py)
    }

    fn add_batch(&mut self, entries: Vec<(u64, Vec<f32>)>) -> PyResult<()> {
        self.inner.add_batch(&entries).map_err(vec_err_to_py)
    }

    fn search(&self, py: Python<'_>, query: Vec<f32>, k: usize) -> PyResult<Vec<PyObject>> {
        let results = py.allow_threads(|| self.inner.search(&query, k)).map_err(vec_err_to_py)?;
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
        let inner = HnswIndex::load(path).map_err(vec_err_to_py)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("HnswIndex(dimensions={}, metric=\"{}\", len={})",
            self.inner.config().dimensions, metric_to_str(self.inner.config().metric), self.inner.len())
    }
}

#[pyclass(name = "Collection")]
struct PyCollection {
    manager: Arc<Mutex<CollectionManager>>,
    name: String,
}

#[pymethods]
impl PyCollection {
    #[pyo3(signature = (id, vector, payload=None))]
    fn upsert(&mut self, py: Python<'_>, id: u64, vector: Vec<f32>, payload: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let p = match payload {
            Some(v) if !v.is_none() => Some(py_to_json(v)?),
            _ => None,
        };
        let name = self.name.clone();
        let mut mgr = self.manager.lock().unwrap();
        let col = mgr.get_collection_mut(&name)
            .ok_or_else(|| PyKeyError::new_err(format!("collection '{name}' not found")))?;
        py.allow_threads(|| col.upsert(id, vector, p)).map_err(vec_err_to_py)
    }

    #[pyo3(signature = (query, k, filter=None))]
    fn search(&self, py: Python<'_>, query: Vec<f32>, k: usize, filter: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<PyObject>> {
        let f = match filter {
            Some(v) if !v.is_none() => Some(py_to_filter(v)?),
            _ => None,
        };
        let name = self.name.clone();
        let mgr = self.manager.lock().unwrap();
        let col = mgr.get_collection(&name)
            .ok_or_else(|| PyKeyError::new_err(format!("collection '{name}' not found")))?;
        let results = py.allow_threads(|| col.search(&query, k, f.as_ref())).map_err(vec_err_to_py)?;
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

    fn delete(&mut self, py: Python<'_>, id: u64) -> PyResult<bool> {
        let name = self.name.clone();
        let mut mgr = self.manager.lock().unwrap();
        let col = mgr.get_collection_mut(&name)
            .ok_or_else(|| PyKeyError::new_err(format!("collection '{name}' not found")))?;
        py.allow_threads(|| col.delete(id)).map_err(vec_err_to_py)
    }

    #[getter]
    fn count(&self) -> usize {
        let mgr = self.manager.lock().unwrap();
        mgr.get_collection(&self.name).map(|c| c.count()).unwrap_or(0)
    }

    #[getter]
    fn name(&self) -> &str { &self.name }

    fn __repr__(&self) -> String {
        let count = self.count();
        format!("Collection(name=\"{}\", count={count})", self.name)
    }
}

#[pyclass(name = "Client")]
struct PyClient {
    manager: Arc<Mutex<CollectionManager>>,
}

#[pymethods]
impl PyClient {
    #[new]
    #[pyo3(signature = (path = "./data"))]
    fn new(path: &str) -> PyResult<Self> {
        let mgr = CollectionManager::open(Path::new(path)).map_err(vec_err_to_py)?;
        Ok(Self { manager: Arc::new(Mutex::new(mgr)) })
    }

    #[pyo3(signature = (name, dimensions, metric = "cosine", index_type = "hnsw"))]
    fn create_collection(&mut self, name: String, dimensions: usize, metric: &str, index_type: &str) -> PyResult<PyCollection> {
        let m = parse_metric(metric)?;
        let (it, hnsw_config) = match index_type {
            "flat" => (IndexType::Flat, None),
            "hnsw" => (IndexType::Hnsw, Some(HnswConfig::default())),
            other => return Err(PyValueError::new_err(format!("unknown index_type {other:?}"))),
        };
        let meta = CollectionMeta { name: name.clone(), dimensions, metric: m, index_type: it, hnsw_config, wal_compact_threshold: 50_000, auto_promote_threshold: None, promotion_hnsw_config: None, embedding_model: None, faiss_factory: None };
        let mut mgr = self.manager.lock().unwrap();
        mgr.create_collection(meta).map_err(vec_err_to_py)?;
        Ok(PyCollection { manager: Arc::clone(&self.manager), name })
    }

    fn get_collection(&self, name: &str) -> PyResult<PyCollection> {
        let mgr = self.manager.lock().unwrap();
        if mgr.get_collection(name).is_none() {
            return Err(PyKeyError::new_err(format!("collection '{name}' not found")));
        }
        Ok(PyCollection { manager: Arc::clone(&self.manager), name: name.to_string() })
    }

    #[pyo3(signature = (name, dimensions, metric = "cosine"))]
    fn get_or_create_collection(&mut self, name: &str, dimensions: usize, metric: &str) -> PyResult<PyCollection> {
        {
            let mgr = self.manager.lock().unwrap();
            if mgr.get_collection(name).is_some() {
                return Ok(PyCollection { manager: Arc::clone(&self.manager), name: name.to_string() });
            }
        }
        self.create_collection(name.to_string(), dimensions, metric, "hnsw")
    }

    fn delete_collection(&mut self, name: &str) -> PyResult<bool> {
        let mut mgr = self.manager.lock().unwrap();
        mgr.delete_collection(name).map_err(vec_err_to_py)
    }

    fn list_collections(&self) -> Vec<String> {
        let mgr = self.manager.lock().unwrap();
        mgr.list_collections().iter().map(|m| m.name.clone()).collect()
    }

    fn __repr__(&self) -> String {
        let count = self.manager.lock().unwrap().list_collections().len();
        format!("Client(collections={count})")
    }
}

#[pymodule]
fn vectordb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFlatIndex>()?;
    m.add_class::<PyHnswIndex>()?;
    m.add_class::<PyClient>()?;
    m.add_class::<PyCollection>()?;
    Ok(())
}
