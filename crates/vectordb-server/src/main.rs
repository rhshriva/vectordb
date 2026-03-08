use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{Arc, RwLock},
};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tracing::info;
use vectordb_core::{FlatIndex, HnswConfig, HnswIndex, Metric, SearchResult, VectorDbError, VectorIndex};

// ── Shared state ─────────────────────────────────────────────────────────────

type DynIndex = Box<dyn VectorIndex>;

struct AppState {
    /// Named collections, each backed by an index.
    collections: RwLock<HashMap<String, DynIndex>>,
}

type SharedState = Arc<AppState>;

// ── Request / response types ──────────────────────────────────────────────────

#[derive(Deserialize)]
struct CreateCollectionRequest {
    dimensions: usize,
    metric: Option<Metric>,
    /// "flat" or "hnsw" (default: "hnsw")
    index_type: Option<String>,
    hnsw: Option<HnswConfig>,
}

#[derive(Deserialize)]
struct UpsertRequest {
    id: u64,
    vector: Vec<f32>,
}

#[derive(Deserialize)]
struct SearchRequest {
    vector: Vec<f32>,
    k: usize,
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
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

// ── Route handlers ────────────────────────────────────────────────────────────

async fn list_collections(State(state): State<SharedState>) -> impl IntoResponse {
    let cols = state.collections.read().unwrap();
    let names: Vec<String> = cols.keys().cloned().collect();
    Json(names)
}

async fn create_collection(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    Json(req): Json<CreateCollectionRequest>,
) -> impl IntoResponse {
    let mut cols = state.collections.write().unwrap();
    if cols.contains_key(&name) {
        return err_response(StatusCode::CONFLICT, format!("collection '{name}' already exists"))
            .into_response();
    }
    let metric = req.metric.unwrap_or(Metric::Cosine);
    let index_type = req.index_type.as_deref().unwrap_or("hnsw");
    let index: DynIndex = match index_type {
        "flat" => Box::new(FlatIndex::new(req.dimensions, metric)),
        "hnsw" => {
            let cfg = req.hnsw.unwrap_or_default();
            Box::new(HnswIndex::new(req.dimensions, metric, cfg))
        }
        other => {
            return err_response(
                StatusCode::BAD_REQUEST,
                format!("unknown index type '{other}', use 'flat' or 'hnsw'"),
            )
            .into_response();
        }
    };
    cols.insert(name.clone(), index);
    info!("created collection '{name}' ({index_type}, {metric:?}, dim={})", req.dimensions);
    StatusCode::CREATED.into_response()
}

async fn get_collection(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let cols = state.collections.read().unwrap();
    match cols.get(&name) {
        None => err_response(StatusCode::NOT_FOUND, format!("collection '{name}' not found"))
            .into_response(),
        Some(idx) => {
            let cfg = idx.config();
            Json(CollectionInfo {
                name,
                count: idx.len(),
                dimensions: cfg.dimensions,
                metric: cfg.metric,
            })
            .into_response()
        }
    }
}

async fn delete_collection(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let mut cols = state.collections.write().unwrap();
    if cols.remove(&name).is_some() {
        StatusCode::NO_CONTENT
    } else {
        StatusCode::NOT_FOUND
    }
}

async fn upsert_vector(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    Json(req): Json<UpsertRequest>,
) -> impl IntoResponse {
    let mut cols = state.collections.write().unwrap();
    let idx = match cols.get_mut(&name) {
        Some(i) => i,
        None => {
            return err_response(StatusCode::NOT_FOUND, format!("collection '{name}' not found"))
                .into_response()
        }
    };
    match idx.add(req.id, &req.vector) {
        Ok(()) => StatusCode::CREATED.into_response(),
        Err(VectorDbError::DuplicateId(_)) => {
            // Treat as update: delete then re-add.
            idx.delete(req.id);
            match idx.add(req.id, &req.vector) {
                Ok(()) => StatusCode::OK.into_response(),
                Err(e) => err_response(StatusCode::BAD_REQUEST, e).into_response(),
            }
        }
        Err(e) => err_response(StatusCode::BAD_REQUEST, e).into_response(),
    }
}

async fn search_vectors(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    Json(req): Json<SearchRequest>,
) -> impl IntoResponse {
    let cols = state.collections.read().unwrap();
    let idx = match cols.get(&name) {
        Some(i) => i,
        None => {
            return err_response(StatusCode::NOT_FOUND, format!("collection '{name}' not found"))
                .into_response()
        }
    };
    match idx.search(&req.vector, req.k) {
        Ok(results) => Json(SearchResponse { results }).into_response(),
        Err(e) => err_response(StatusCode::BAD_REQUEST, e).into_response(),
    }
}

async fn delete_vector(
    State(state): State<SharedState>,
    Path((name, id)): Path<(String, u64)>,
) -> impl IntoResponse {
    let mut cols = state.collections.write().unwrap();
    let idx = match cols.get_mut(&name) {
        Some(i) => i,
        None => {
            return err_response(StatusCode::NOT_FOUND, format!("collection '{name}' not found"))
                .into_response()
        }
    };
    if idx.delete(id) {
        StatusCode::NO_CONTENT.into_response()
    } else {
        err_response(StatusCode::NOT_FOUND, format!("vector {id} not found")).into_response()
    }
}

// ── App builder (shared by main and tests) ────────────────────────────────────

fn build_app() -> Router {
    let state = Arc::new(AppState {
        collections: RwLock::new(HashMap::new()),
    });
    Router::new()
        .route("/collections", get(list_collections))
        .route("/collections/:name", post(create_collection))
        .route("/collections/:name", get(get_collection))
        .route("/collections/:name", delete(delete_collection))
        .route("/collections/:name/vectors", post(upsert_vector))
        .route("/collections/:name/search", post(search_vectors))
        .route("/collections/:name/vectors/:id", delete(delete_vector))
        .with_state(state)
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

    let app = build_app();

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

    fn delete_req(uri: &str) -> Request<Body> {
        Request::builder().method("DELETE").uri(uri).body(Body::empty()).unwrap()
    }

    fn get_req(uri: &str) -> Request<Body> {
        Request::builder().uri(uri).body(Body::empty()).unwrap()
    }

    // ── Collections ──

    #[tokio::test]
    async fn list_collections_empty() {
        let app = build_app();
        let resp = app.oneshot(get_req("/collections")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_json(resp.into_body()).await, json!([]));
    }

    #[tokio::test]
    async fn create_and_list_collection() {
        let app = build_app();
        let resp = app
            .oneshot(post_json("/collections/docs", json!({"dimensions": 3, "index_type": "flat"})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
    }

    #[tokio::test]
    async fn create_duplicate_collection_returns_conflict() {
        let app = build_app();
        let body = json!({"dimensions": 3, "index_type": "flat"});
        app.clone().oneshot(post_json("/collections/dup", body.clone())).await.unwrap();
        let resp = app.oneshot(post_json("/collections/dup", body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn get_collection_info() {
        let app = build_app();
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
        let app = build_app();
        let resp = app.oneshot(get_req("/collections/nope")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn delete_collection() {
        let app = build_app();
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
        let app = build_app();
        let resp = app.oneshot(delete_req("/collections/ghost")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn create_flat_index_type() {
        let app = build_app();
        let resp = app
            .oneshot(post_json("/collections/flat", json!({"dimensions": 2, "index_type": "flat"})))
            .await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
    }

    #[tokio::test]
    async fn create_hnsw_index_type() {
        let app = build_app();
        let resp = app
            .oneshot(post_json("/collections/hnsw", json!({"dimensions": 2, "index_type": "hnsw"})))
            .await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
    }

    #[tokio::test]
    async fn create_unknown_index_type_returns_400() {
        let app = build_app();
        let resp = app
            .oneshot(post_json("/collections/bad", json!({"dimensions": 2, "index_type": "tree"})))
            .await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ── Vectors ──

    #[tokio::test]
    async fn upsert_and_search_vector() {
        let app = build_app();
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
        let app = build_app();
        app.clone()
            .oneshot(post_json("/collections/u", json!({"dimensions": 2, "index_type": "flat"})))
            .await.unwrap();
        app.clone()
            .oneshot(post_json("/collections/u/vectors", json!({"id": 1, "vector": [1.0, 0.0]})))
            .await.unwrap();
        // Re-insert same ID with different vector — should succeed (200)
        let resp = app
            .oneshot(post_json("/collections/u/vectors", json!({"id": 1, "vector": [0.5, 0.5]})))
            .await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn upsert_wrong_dimensions_returns_400() {
        let app = build_app();
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
        let app = build_app();
        let resp = app
            .oneshot(post_json("/collections/missing/vectors", json!({"id": 1, "vector": [1.0]})))
            .await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn search_nonexistent_collection_returns_404() {
        let app = build_app();
        let resp = app
            .oneshot(post_json("/collections/missing/search", json!({"vector": [1.0], "k": 1})))
            .await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn delete_vector() {
        let app = build_app();
        app.clone()
            .oneshot(post_json("/collections/dv", json!({"dimensions": 2, "index_type": "flat"})))
            .await.unwrap();
        app.clone()
            .oneshot(post_json("/collections/dv/vectors", json!({"id": 42, "vector": [1.0, 0.0]})))
            .await.unwrap();
        let resp = app.clone().oneshot(delete_req("/collections/dv/vectors/42")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        // Confirm it's gone: search returns empty
        let resp = app
            .oneshot(post_json("/collections/dv/search", json!({"vector": [1.0, 0.0], "k": 1})))
            .await.unwrap();
        let body = body_json(resp.into_body()).await;
        assert_eq!(body["results"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn delete_nonexistent_vector_returns_404() {
        let app = build_app();
        app.clone()
            .oneshot(post_json("/collections/dv2", json!({"dimensions": 2, "index_type": "flat"})))
            .await.unwrap();
        let resp = app.oneshot(delete_req("/collections/dv2/vectors/99")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn all_three_metrics_work() {
        for metric in ["l2", "cosine", "dot_product"] {
            let app = build_app();
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
}
