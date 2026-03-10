/**
 * Supported distance metrics.
 */
export type Metric = "l2" | "cosine" | "dot_product";

/**
 * Index type for a collection.
 * - "flat"  — exact brute-force (100 % recall, O(N·D) per query)
 * - "hnsw"  — approximate graph search (~95–99 % recall, sub-linear)
 * - "faiss" — FAISS-backed index; server must be compiled with --features faiss
 */
export type IndexType = "flat" | "hnsw" | "faiss";

/**
 * HNSW graph construction and search parameters.
 */
export interface HnswConfig {
  /** Beam width during graph construction. Default: 200. */
  ef_construction?: number;
  /** Beam width during search. Default: 50. */
  ef_search?: number;
  /** Edges per node per layer. Default: 12. */
  m?: number;
}

/**
 * Options for creating a new collection.
 */
export interface CreateCollectionOptions {
  /** Number of dimensions in each vector. */
  dimensions: number;
  /** Distance metric. Default: "cosine". */
  metric?: Metric;
  /** Index type. Default: "hnsw". */
  index_type?: IndexType;
  /** HNSW-specific tuning (ignored for flat index). */
  hnsw?: HnswConfig;
  /**
   * Automatically upgrade from a flat index to HNSW when the collection
   * reaches this many vectors. Useful when you start small but expect growth.
   */
  auto_promote_threshold?: number;
  /** HNSW config to apply on auto-promotion. */
  promotion_hnsw_config?: HnswConfig;
  /**
   * FAISS factory string (only used when index_type = "faiss").
   * Defaults to "Flat". Examples: "IVF1024,Flat", "HNSW32", "IVF256,PQ64".
   * See https://github.com/facebookresearch/faiss/wiki/The-index-factory
   */
  faiss_factory?: string;
}

/**
 * Payload is an arbitrary JSON-serializable object attached to a vector.
 */
export type Payload = Record<string, unknown>;

/**
 * Comparison operators for payload filters.
 */
export interface FieldOps {
  $eq?: unknown;
  $ne?: unknown;
  $in?: unknown[];
  $gt?: number;
  $gte?: number;
  $lt?: number;
  $lte?: number;
}

/**
 * A filter expression for `search`.
 *
 * @example
 * // Single field equality
 * { category: { $eq: "news" } }
 *
 * @example
 * // AND of multiple conditions
 * { $and: [{ score: { $gte: 0.5 } }, { tag: { $in: ["a", "b"] } }] }
 */
export type FilterCondition =
  | { $and: FilterCondition[] }
  | { $or: FilterCondition[] }
  | { [field: string]: FieldOps };

/**
 * A single search result.
 */
export interface SearchResult {
  id: number;
  distance: number;
  payload?: Payload;
}

/**
 * Response from a search request.
 */
export interface SearchResponse {
  results: SearchResult[];
}

/**
 * Metadata about a collection returned by the server.
 */
export interface CollectionInfo {
  name: string;
  count: number;
  dimensions: number;
  metric: Metric;
}

/**
 * Options for constructing a `QuiverClient`.
 */
export interface ClientOptions {
  /** Base URL of the Quiver server. Default: "http://localhost:7070". */
  baseUrl?: string;
  /** Bearer token for API key authentication. */
  apiKey?: string;
}
