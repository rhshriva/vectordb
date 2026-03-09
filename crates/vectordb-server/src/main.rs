use std::{
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, RwLock},
};

use axum::{
    extract::{Path, Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Json, Response},
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tracing::info;
use vectordb_core::{
    collection::{CollectionMeta, IndexType},
    distance::Metric,
    index::hnsw::HnswConfig,
    manager::CollectionManager,
    payload::FilterCondition,
    VectorDbError,
};
use vectordb_embeddings::{EmbeddingProvider, OllamaProvider, OpenAiProvider};

// ── Shared state ─────────────────────────────────────────────────────────────

struct AppState {
    manager: RwLock<CollectionManager>,
}

type SharedState = Arc<AppState>;

// ── Request / response types ──────────────────────────────────────────────────

#[derive(Deserialize)]
struct CreateCollectionRequest {
    dimensions: usize,
    metric: Option<Metric>,
    /// "flat", "hnsw", or "faiss" (default: "hnsw")
    index_type: Option<String>,
    hnsw: Option<HnswConfig>,
    /// Automatically promote from Flat to HNSW when vector count reaches this value.
    #[serde(default)]
    auto_promote_threshold: Option<usize>,
    /// HNSW config to use for auto-promotion (optional; defaults to HnswConfig::default()).
    #[serde(default)]
    promotion_hnsw_config: Option<HnswConfig>,
    /// Embedding model identifier, e.g. `"openai/text-embedding-3-small"` or
    /// `"ollama/nomic-embed-text"`. Required for embed-upsert / embed-search endpoints.
    #[serde(default)]
    embedding_model: Option<String>,
    /// FAISS factory string (only used when index_type = "faiss").
    /// Defaults to "Flat". Examples: "IVF1024,Flat", "HNSW32", "IVF256,PQ64".
    #[serde(default)]
    faiss_factory: Option<String>,
}

#[derive(Deserialize)]
struct EmbedUpsertRequest {
    id: u64,
    text: String,
    #[serde(default)]
    payload: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct EmbedSearchRequest {
    text: String,
    k: usize,
    #[serde(default)]
    filter: Option<FilterCondition>,
}

#[derive(Deserialize)]
struct UpsertRequest {
    id: u64,
    vector: Vec<f32>,
    #[serde(default)]
    payload: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct SearchRequest {
    vector: Vec<f32>,
    k: usize,
    #[serde(default)]
    filter: Option<FilterCondition>,
}

#[derive(Serialize)]
struct SearchResultItem {
    id: u64,
    distance: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    payload: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<SearchResultItem>,
}

#[derive(Serialize)]
struct CollectionInfo {
    name: String,
    count: usize,
    dimensions: usize,
    metric: Metric,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

fn err_response(status: StatusCode, msg: impl ToString) -> impl IntoResponse {
    (status, Json(ErrorResponse { error: msg.to_string() }))
}

// ── Auth middleware ───────────────────────────────────────────────────────────

async fn auth_middleware(
    State(api_key): State<Arc<String>>,
    req: Request,
    next: Next,
) -> Response {
    let provided = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "));

    match provided {
        Some(k) if k == api_key.as_str() => next.run(req).await,
        _ => (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "invalid or missing API key".to_string(),
            }),
        )
            .into_response(),
    }
}

// ── Route handlers ────────────────────────────────────────────────────────────

async fn list_collections(State(state): State<SharedState>) -> impl IntoResponse {
    let mgr = state.manager.read().unwrap();
    let names: Vec<String> = mgr.list_collections().iter().map(|m| m.name.clone()).collect();
    Json(names)
}

async fn create_collection(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    Json(req): Json<CreateCollectionRequest>,
) -> impl IntoResponse {
    let metric = req.metric.unwrap_or(Metric::Cosine);
    let index_type_str = req.index_type.as_deref().unwrap_or("hnsw");
    let (index_type, hnsw_config, faiss_factory) = match index_type_str {
        // Pass req.faiss_factory through in all arms so the field is always consumed
        // (prevents dead_code warnings when the faiss feature is disabled).
        "flat" => (IndexType::Flat, None, req.faiss_factory),
        "hnsw" => (IndexType::Hnsw, Some(req.hnsw.unwrap_or_default()), req.faiss_factory),
        "faiss" => {
            #[cfg(not(feature = "faiss"))]
            return err_response(
                StatusCode::BAD_REQUEST,
                "index type 'faiss' requires the server to be compiled with the 'faiss' feature",
            )
            .into_response();
            #[cfg(feature = "faiss")]
            (IndexType::Faiss, None, Some(req.faiss_factory.unwrap_or_else(|| "Flat".to_string())))
        }
        other => {
            return err_response(
                StatusCode::BAD_REQUEST,
                format!("unknown index type '{other}', use 'flat', 'hnsw', or 'faiss'"),
            )
            .into_response()
        }
    };

    let meta = CollectionMeta {
        name: name.clone(),
        dimensions: req.dimensions,
        metric,
        index_type,
        hnsw_config,
        wal_compact_threshold: 50_000,
        auto_promote_threshold: req.auto_promote_threshold,
        promotion_hnsw_config: req.promotion_hnsw_config.clone(),
        embedding_model: req.embedding_model.clone(),
        faiss_factory,
    };

    let mut mgr = state.manager.write().unwrap();
    match mgr.create_collection(meta) {
        Ok(()) => {
            info!("created collection '{name}' ({index_type_str}, {metric:?}, dim={})", req.dimensions);
            StatusCode::CREATED.into_response()
        }
        Err(VectorDbError::CollectionAlreadyExists(_)) => {
            err_response(StatusCode::CONFLICT, format!("collection '{name}' already exists"))
                .into_response()
        }
        Err(e) => err_response(StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    }
}

async fn get_collection(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let mgr = state.manager.read().unwrap();
    match mgr.get_collection(&name) {
        None => err_response(StatusCode::NOT_FOUND, format!("collection '{name}' not found"))
            .into_response(),
        Some(col) => {
            let meta = col.meta();
            Json(CollectionInfo {
                name: name.clone(),
                count: col.count(),
                dimensions: meta.dimensions,
                metric: meta.metric,
            })
            .into_response()
        }
    }
}

async fn delete_collection(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let mut mgr = state.manager.write().unwrap();
    match mgr.delete_collection(&name) {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => {
            err_response(StatusCode::NOT_FOUND, format!("collection '{name}' not found"))
                .into_response()
        }
        Err(e) => err_response(StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    }
}

async fn upsert_vector(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    Json(req): Json<UpsertRequest>,
) -> impl IntoResponse {
    let mut mgr = state.manager.write().unwrap();
    match mgr.get_collection_mut(&name) {
        None => {
            err_response(StatusCode::NOT_FOUND, format!("collection '{name}' not found"))
                .into_response()
        }
        Some(col) => match col.upsert(req.id, req.vector, req.payload) {
            Ok(()) => StatusCode::CREATED.into_response(),
            Err(e) => err_response(StatusCode::BAD_REQUEST, e).into_response(),
        },
    }
}

async fn search_vectors(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    Json(req): Json<SearchRequest>,
) -> impl IntoResponse {
    let mgr = state.manager.read().unwrap();
    match mgr.get_collection(&name) {
        None => {
            err_response(StatusCode::NOT_FOUND, format!("collection '{name}' not found"))
                .into_response()
        }
        Some(col) => match col.search(&req.vector, req.k, req.filter.as_ref()) {
            Ok(results) => Json(SearchResponse {
                results: results
                    .into_iter()
                    .map(|r| SearchResultItem {
                        id: r.id,
                        distance: r.distance,
                        payload: r.payload,
                    })
                    .collect(),
            })
            .into_response(),
            Err(e) => err_response(StatusCode::BAD_REQUEST, e).into_response(),
        },
    }
}

async fn delete_vector(
    State(state): State<SharedState>,
    Path((name, id)): Path<(String, u64)>,
) -> impl IntoResponse {
    let mut mgr = state.manager.write().unwrap();
    match mgr.get_collection_mut(&name) {
        None => {
            err_response(StatusCode::NOT_FOUND, format!("collection '{name}' not found"))
                .into_response()
        }
        Some(col) => match col.delete(id) {
            Ok(true) => StatusCode::NO_CONTENT.into_response(),
            Ok(false) => {
                err_response(StatusCode::NOT_FOUND, format!("vector {id} not found"))
                    .into_response()
            }
            Err(e) => err_response(StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
        },
    }
}

// ── Embedding helpers ─────────────────────────────────────────────────────────

/// Resolve a model identifier string to a live embedding provider.
///
/// Supported prefixes:
/// - `"openai/<model>"` — uses `OPENAI_API_KEY` env var
/// - `"ollama/<model>"` — uses `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
fn make_provider(model_id: &str) -> Result<Box<dyn EmbeddingProvider>, String> {
    if let Some(model) = model_id.strip_prefix("openai/") {
        let provider = OpenAiProvider::from_env(model)
            .map_err(|e| e.to_string())?;
        return Ok(Box::new(provider));
    }
    if let Some(model) = model_id.strip_prefix("ollama/") {
        let base_url = std::env::var("OLLAMA_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        return Ok(Box::new(OllamaProvider::new(base_url, model)));
    }
    Err(format!(
        "unknown model prefix in '{model_id}'; use 'openai/<model>' or 'ollama/<model>'"
    ))
}

async fn embed_upsert(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    Json(req): Json<EmbedUpsertRequest>,
) -> impl IntoResponse {
    // Get the model id without holding the lock
    let model_id = {
        let mgr = state.manager.read().unwrap();
        match mgr.get_collection(&name) {
            None => {
                return err_response(
                    StatusCode::NOT_FOUND,
                    format!("collection '{name}' not found"),
                )
                .into_response()
            }
            Some(col) => match col.meta().embedding_model.clone() {
                Some(m) => m,
                None => {
                    return err_response(
                        StatusCode::BAD_REQUEST,
                        format!("collection '{name}' has no embedding_model configured"),
                    )
                    .into_response()
                }
            },
        }
    };

    let provider = match make_provider(&model_id) {
        Ok(p) => p,
        Err(e) => return err_response(StatusCode::BAD_REQUEST, e).into_response(),
    };

    let vector = match provider.embed(req.text).await {
        Ok(v) => v,
        Err(e) => return err_response(StatusCode::BAD_GATEWAY, e.to_string()).into_response(),
    };

    let mut mgr = state.manager.write().unwrap();
    match mgr.get_collection_mut(&name) {
        None => err_response(StatusCode::NOT_FOUND, format!("collection '{name}' not found"))
            .into_response(),
        Some(col) => match col.upsert(req.id, vector, req.payload) {
            Ok(()) => StatusCode::CREATED.into_response(),
            Err(e) => err_response(StatusCode::BAD_REQUEST, e).into_response(),
        },
    }
}

async fn embed_search(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    Json(req): Json<EmbedSearchRequest>,
) -> impl IntoResponse {
    // Get the model id without holding the lock
    let model_id = {
        let mgr = state.manager.read().unwrap();
        match mgr.get_collection(&name) {
            None => {
                return err_response(
                    StatusCode::NOT_FOUND,
                    format!("collection '{name}' not found"),
                )
                .into_response()
            }
            Some(col) => match col.meta().embedding_model.clone() {
                Some(m) => m,
                None => {
                    return err_response(
                        StatusCode::BAD_REQUEST,
                        format!("collection '{name}' has no embedding_model configured"),
                    )
                    .into_response()
                }
            },
        }
    };

    let provider = match make_provider(&model_id) {
        Ok(p) => p,
        Err(e) => return err_response(StatusCode::BAD_REQUEST, e).into_response(),
    };

    let vector = match provider.embed(req.text).await {
        Ok(v) => v,
        Err(e) => return err_response(StatusCode::BAD_GATEWAY, e.to_string()).into_response(),
    };

    let mgr = state.manager.read().unwrap();
    match mgr.get_collection(&name) {
        None => err_response(StatusCode::NOT_FOUND, format!("collection '{name}' not found"))
            .into_response(),
        Some(col) => match col.search(&vector, req.k, req.filter.as_ref()) {
            Ok(results) => Json(SearchResponse {
                results: results
                    .into_iter()
                    .map(|r| SearchResultItem {
                        id: r.id,
                        distance: r.distance,
                        payload: r.payload,
                    })
                    .collect(),
            })
            .into_response(),
            Err(e) => err_response(StatusCode::BAD_REQUEST, e).into_response(),
        },
    }
}

// ── App builder (shared by main and tests) ────────────────────────────────────

fn build_app(data_dir: PathBuf, api_key: Option<String>) -> Router {
    let manager = CollectionManager::open(&data_dir).expect("failed to open collection manager");
    let state = Arc::new(AppState {
        manager: RwLock::new(manager),
    });
    let router = Router::new()
        .route("/collections", get(list_collections))
        .route("/collections/:name", post(create_collection))
        .route("/collections/:name", get(get_collection))
        .route("/collections/:name", delete(delete_collection))
        .route("/collections/:name/vectors", post(upsert_vector))
        .route("/collections/:name/search", post(search_vectors))
        .route("/collections/:name/vectors/:id", delete(delete_vector))
        .route("/collections/:name/embed-upsert", post(embed_upsert))
        .route("/collections/:name/embed-search", post(embed_search))
        .with_state(state);

    if let Some(key) = api_key {
        let key = Arc::new(key);
        router.layer(axum::middleware::from_fn_with_state(key, auth_middleware))
    } else {
        router
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "vectordb_server=info,tower_http=debug".into()),
        )
        .init();

    let data_dir = std::env::var("VECTORDB_DATA_DIR").unwrap_or_else(|_| "./data".to_string());
    let api_key = std::env::var("VECTORDB_API_KEY").ok();

    if api_key.is_some() {
        info!("API key authentication enabled");
    } else {
        info!("API key authentication disabled (dev mode)");
    }

    let app = build_app(PathBuf::from(data_dir), api_key);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!("vectordb-server listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// ── Integration tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::{Request, StatusCode}};
    use http_body_util::BodyExt;
    use serde_json::{json, Value};
    use tower::ServiceExt;

    fn build_test_app() -> Router {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();
        // Leak the TempDir so it survives for the duration of the test
        std::mem::forget(dir);
        build_app(path, None)
    }

    fn build_test_app_with_key(key: &str) -> Router {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();
        std::mem::forget(dir);
        build_app(path, Some(key.to_string()))
    }

    async fn body_json(body: Body) -> Value {
        let bytes = body.collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn post_json(uri: &str, body: Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap()
    }

    fn post_json_with_auth(uri: &str, body: Value, key: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .header("Authorization", format!("Bearer {key}"))
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap()
    }

    fn delete_req(uri: &str) -> Request<Body> {
        Request::builder().method("DELETE").uri(uri).body(Body::empty()).unwrap()
    }

    fn get_req(uri: &str) -> Request<Body> {
        Request::builder().uri(uri).body(Body::empty()).unwrap()
    }

    // ── Collections ──

    #[tokio::test]
    async fn list_collections_empty() {
        let app = build_test_app();
        let resp = app.oneshot(get_req("/collections")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_json(resp.into_body()).await, json!([]));
    }

    #[tokio::test]
    async fn create_and_list_collection() {
        let app = build_test_app();
        let resp = app
            .oneshot(post_json("/collections/docs", json!({"dimensions": 3, "index_type": "flat"})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
    }

    #[tokio::test]
    async fn create_duplicate_collection_returns_conflict() {
        let app = build_test_app();
        let body = json!({"dimensions": 3, "index_type": "flat"});
        app.clone().oneshot(post_json("/collections/dup", body.clone())).await.unwrap();
        let resp = app.oneshot(post_json("/collections/dup", body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn get_collection_info() {
        let app = build_test_app();
        app.clone()
            .oneshot(post_json("/collections/info", json!({"dimensions": 4, "metric": "l2", "index_type": "flat"})))
            .await.unwrap();
        let resp = app.oneshot(get_req("/collections/info")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp.into_body()).await;
        assert_eq!(body["dimensions"], 4);
        assert_eq!(body["count"], 0);
    }

    #[tokio::test]
    async fn get_nonexistent_collection_returns_404() {
        let app = build_test_app();
        let resp = app.oneshot(get_req("/collections/nope")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn delete_collection() {
        let app = build_test_app();
        app.clone()
            .oneshot(post_json("/collections/del", json!({"dimensions": 2, "index_type": "flat"})))
            .await.unwrap();
        let resp = app.clone().oneshot(delete_req("/collections/del")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        let resp = app.oneshot(get_req("/collections/del")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn delete_nonexistent_collection_returns_404() {
        let app = build_test_app();
        let resp = app.oneshot(delete_req("/collections/ghost")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn create_flat_index_type() {
        let app = build_test_app();
        let resp = app
            .oneshot(post_json("/collections/flat", json!({"dimensions": 2, "index_type": "flat"})))
            .await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
    }

    #[tokio::test]
    async fn create_hnsw_index_type() {
        let app = build_test_app();
        let resp = app
            .oneshot(post_json("/collections/hnsw", json!({"dimensions": 2, "index_type": "hnsw"})))
            .await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
    }

    #[tokio::test]
    async fn create_unknown_index_type_returns_400() {
        let app = build_test_app();
        let resp = app
            .oneshot(post_json("/collections/bad", json!({"dimensions": 2, "index_type": "tree"})))
            .await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp.into_body()).await;
        assert!(body["error"].as_str().unwrap().contains("flat"));
    }

    // ── Vectors ──

    #[tokio::test]
    async fn upsert_and_search_vector() {
        let app = build_test_app();
        app.clone()
            .oneshot(post_json("/collections/s", json!({"dimensions": 2, "index_type": "flat", "metric": "l2"})))
            .await.unwrap();
        app.clone()
            .oneshot(post_json("/collections/s/vectors", json!({"id": 1, "vector": [1.0, 0.0]})))
            .await.unwrap();
        app.clone()
            .oneshot(post_json("/collections/s/vectors", json!({"id": 2, "vector": [0.0, 1.0]})))
            .await.unwrap();
        let resp = app
            .oneshot(post_json("/collections/s/search", json!({"vector": [1.0, 0.0], "k": 1})))
            .await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp.into_body()).await;
        assert_eq!(body["results"][0]["id"], 1);
    }

    #[tokio::test]
    async fn upsert_duplicate_id_acts_as_update() {
        let app = build_test_app();
        app.clone()
            .oneshot(post_json("/collections/u", json!({"dimensions": 2, "index_type": "flat"})))
            .await.unwrap();
        app.clone()
            .oneshot(post_json("/collections/u/vectors", json!({"id": 1, "vector": [1.0, 0.0]})))
            .await.unwrap();
        let resp = app
            .oneshot(post_json("/collections/u/vectors", json!({"id": 1, "vector": [0.5, 0.5]})))
            .await.unwrap();
        // upsert always returns CREATED (first insert) — subsequent upserts also get CREATED
        assert!(resp.status().is_success());
    }

    #[tokio::test]
    async fn upsert_wrong_dimensions_returns_400() {
        let app = build_test_app();
        app.clone()
            .oneshot(post_json("/collections/d", json!({"dimensions": 2, "index_type": "flat"})))
            .await.unwrap();
        let resp = app
            .oneshot(post_json("/collections/d/vectors", json!({"id": 1, "vector": [1.0, 2.0, 3.0]})))
            .await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn upsert_to_nonexistent_collection_returns_404() {
        let app = build_test_app();
        let resp = app
            .oneshot(post_json("/collections/missing/vectors", json!({"id": 1, "vector": [1.0]})))
            .await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn search_nonexistent_collection_returns_404() {
        let app = build_test_app();
        let resp = app
            .oneshot(post_json("/collections/missing/search", json!({"vector": [1.0], "k": 1})))
            .await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn delete_vector() {
        let app = build_test_app();
        app.clone()
            .oneshot(post_json("/collections/dv", json!({"dimensions": 2, "index_type": "flat"})))
            .await.unwrap();
        app.clone()
            .oneshot(post_json("/collections/dv/vectors", json!({"id": 42, "vector": [1.0, 0.0]})))
            .await.unwrap();
        let resp = app.clone().oneshot(delete_req("/collections/dv/vectors/42")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        let resp = app
            .oneshot(post_json("/collections/dv/search", json!({"vector": [1.0, 0.0], "k": 1})))
            .await.unwrap();
        let body = body_json(resp.into_body()).await;
        assert_eq!(body["results"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn delete_nonexistent_vector_returns_404() {
        let app = build_test_app();
        app.clone()
            .oneshot(post_json("/collections/dv2", json!({"dimensions": 2, "index_type": "flat"})))
            .await.unwrap();
        let resp = app.oneshot(delete_req("/collections/dv2/vectors/99")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn all_three_metrics_work() {
        for metric in ["l2", "cosine", "dot_product"] {
            let app = build_test_app();
            let col = format!("/collections/{metric}col");
            app.clone()
                .oneshot(post_json(&col, json!({"dimensions": 2, "index_type": "flat", "metric": metric})))
                .await.unwrap();
            app.clone()
                .oneshot(post_json(&format!("{col}/vectors"), json!({"id": 1, "vector": [1.0, 0.0]})))
                .await.unwrap();
            let resp = app
                .oneshot(post_json(&format!("{col}/search"), json!({"vector": [1.0, 0.0], "k": 1})))
                .await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK, "metric={metric}");
        }
    }

    // ── Payload ──

    #[tokio::test]
    async fn upsert_with_payload_roundtrips() {
        let app = build_test_app();
        app.clone()
            .oneshot(post_json("/collections/p", json!({"dimensions": 2, "index_type": "flat"})))
            .await.unwrap();
        app.clone()
            .oneshot(post_json("/collections/p/vectors", json!({"id": 1, "vector": [1.0, 0.0], "payload": {"tag": "news"}})))
            .await.unwrap();
        let resp = app
            .oneshot(post_json("/collections/p/search", json!({"vector": [1.0, 0.0], "k": 1})))
            .await.unwrap();
        let body = body_json(resp.into_body()).await;
        assert_eq!(body["results"][0]["id"], 1);
        assert_eq!(body["results"][0]["payload"]["tag"], "news");
    }

    #[tokio::test]
    async fn search_with_filter_eq() {
        let app = build_test_app();
        app.clone()
            .oneshot(post_json("/collections/f", json!({"dimensions": 2, "index_type": "flat"})))
            .await.unwrap();
        app.clone()
            .oneshot(post_json("/collections/f/vectors", json!({"id": 1, "vector": [1.0, 0.0], "payload": {"tag": "a"}})))
            .await.unwrap();
        app.clone()
            .oneshot(post_json("/collections/f/vectors", json!({"id": 2, "vector": [0.9, 0.1], "payload": {"tag": "b"}})))
            .await.unwrap();
        app.clone()
            .oneshot(post_json("/collections/f/vectors", json!({"id": 3, "vector": [0.8, 0.2], "payload": {"tag": "a"}})))
            .await.unwrap();

        let resp = app
            .oneshot(post_json("/collections/f/search", json!({"vector": [1.0, 0.0], "k": 3, "filter": {"tag": {"$eq": "a"}}})))
            .await.unwrap();
        let body = body_json(resp.into_body()).await;
        let results = body["results"].as_array().unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r["payload"]["tag"] == "a"));
    }

    #[tokio::test]
    async fn search_without_filter_returns_all_candidates() {
        let app = build_test_app();
        app.clone()
            .oneshot(post_json("/collections/nf", json!({"dimensions": 2, "index_type": "flat"})))
            .await.unwrap();
        app.clone()
            .oneshot(post_json("/collections/nf/vectors", json!({"id": 1, "vector": [1.0, 0.0]})))
            .await.unwrap();
        app.clone()
            .oneshot(post_json("/collections/nf/vectors", json!({"id": 2, "vector": [0.0, 1.0]})))
            .await.unwrap();
        let resp = app
            .oneshot(post_json("/collections/nf/search", json!({"vector": [1.0, 0.0], "k": 2})))
            .await.unwrap();
        let body = body_json(resp.into_body()).await;
        assert_eq!(body["results"].as_array().unwrap().len(), 2);
    }

    // ── Auth ──

    #[tokio::test]
    async fn auth_disabled_all_pass() {
        let app = build_test_app();
        let resp = app.oneshot(get_req("/collections")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn auth_enabled_no_header_returns_401() {
        let app = build_test_app_with_key("secret");
        let resp = app.oneshot(get_req("/collections")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn auth_enabled_wrong_key_returns_401() {
        let app = build_test_app_with_key("secret");
        let req = Request::builder()
            .uri("/collections")
            .header("Authorization", "Bearer wrongkey")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn auth_enabled_correct_key_passes() {
        let app = build_test_app_with_key("secret");
        let req = Request::builder()
            .uri("/collections")
            .header("Authorization", "Bearer secret")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn auth_enabled_create_collection_with_key() {
        let app = build_test_app_with_key("mykey");
        let resp = app
            .oneshot(post_json_with_auth(
                "/collections/auth_col",
                json!({"dimensions": 2, "index_type": "flat"}),
                "mykey",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
    }
}
