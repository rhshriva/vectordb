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

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "vectordb_server=info,tower_http=debug".into()),
        )
        .init();

    let state = Arc::new(AppState {
        collections: RwLock::new(HashMap::new()),
    });

    let app = Router::new()
        .route("/collections", get(list_collections))
        .route("/collections/:name", post(create_collection))
        .route("/collections/:name", get(get_collection))
        .route("/collections/:name", delete(delete_collection))
        .route("/collections/:name/vectors", post(upsert_vector))
        .route("/collections/:name/search", post(search_vectors))
        .route("/collections/:name/vectors/:id", delete(delete_vector))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!("vectordb-server listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
