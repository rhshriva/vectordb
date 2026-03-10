/**
 * QuiverClient — zero-dependency TypeScript client for the Quiver HTTP server.
 *
 * Uses the global `fetch` API (available in Node 18+, Deno, and all modern browsers).
 *
 * @example
 * ```ts
 * import { QuiverClient } from "quiver-client";
 *
 * const db = new QuiverClient({ baseUrl: "http://localhost:7070" });
 *
 * await db.createCollection("docs", { dimensions: 768, metric: "cosine" });
 * await db.upsert("docs", 1, [0.1, 0.2, ...], { source: "readme" });
 * const hits = await db.search("docs", [0.1, 0.2, ...], 5);
 * ```
 */

import type {
  ClientOptions,
  CollectionInfo,
  CreateCollectionOptions,
  FilterCondition,
  Payload,
  SearchResult,
} from "./types.js";

export class QuiverError extends Error {
  constructor(
    message: string,
    public readonly statusCode?: number,
  ) {
    super(message);
    this.name = "QuiverError";
  }
}

export class QuiverClient {
  private readonly baseUrl: string;
  private readonly headers: Record<string, string>;

  constructor(options: ClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? "http://localhost:7070").replace(/\/$/, "");
    this.headers = {
      "Content-Type": "application/json",
      ...(options.apiKey ? { Authorization: `Bearer ${options.apiKey}` } : {}),
    };
  }

  // ── Collections ────────────────────────────────────────────────────────────

  /**
   * Create a new collection.
   * @throws {QuiverError} if the collection already exists (409) or the
   *   request is malformed (400).
   */
  async createCollection(
    name: string,
    options: CreateCollectionOptions,
  ): Promise<void> {
    await this.request("POST", `/collections/${encodeURIComponent(name)}`, options);
  }

  /**
   * Delete a collection and all its vectors.
   * @returns `true` if the collection was deleted, `false` if it didn't exist.
   */
  async deleteCollection(name: string): Promise<boolean> {
    const resp = await this.rawRequest(
      "DELETE",
      `/collections/${encodeURIComponent(name)}`,
    );
    if (resp.status === 404) return false;
    if (!resp.ok) await this.throwFromResponse(resp);
    return true;
  }

  /**
   * List all collection names.
   */
  async listCollections(): Promise<string[]> {
    const data = await this.request<{ collections: string[] }>("GET", "/collections");
    return data.collections;
  }

  /**
   * Get metadata about a specific collection.
   */
  async getCollection(name: string): Promise<CollectionInfo> {
    return this.request<CollectionInfo>(
      "GET",
      `/collections/${encodeURIComponent(name)}`,
    );
  }

  // ── Vectors ────────────────────────────────────────────────────────────────

  /**
   * Insert or update a vector (upsert by `id`).
   */
  async upsert(
    collection: string,
    id: number,
    vector: number[],
    payload?: Payload,
  ): Promise<void> {
    await this.request(
      "POST",
      `/collections/${encodeURIComponent(collection)}/upsert`,
      { id, vector, payload },
    );
  }

  /**
   * Delete a vector by id.
   * @returns `true` if the vector was found and deleted.
   */
  async delete(collection: string, id: number): Promise<boolean> {
    const data = await this.request<{ deleted: boolean }>(
      "DELETE",
      `/collections/${encodeURIComponent(collection)}/vectors/${id}`,
    );
    return data.deleted;
  }

  /**
   * Search for the `k` nearest neighbours of `query`.
   *
   * @param filter - Optional payload filter; only matching vectors are returned.
   */
  async search(
    collection: string,
    query: number[],
    k: number,
    filter?: FilterCondition,
  ): Promise<SearchResult[]> {
    const body: Record<string, unknown> = { vector: query, k };
    if (filter !== undefined) body.filter = filter;
    const data = await this.request<{ results: SearchResult[] }>(
      "POST",
      `/collections/${encodeURIComponent(collection)}/search`,
      body,
    );
    return data.results;
  }

  // ── Low-level helpers ──────────────────────────────────────────────────────

  private async request<T = void>(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<T> {
    const resp = await this.rawRequest(method, path, body);
    if (!resp.ok) await this.throwFromResponse(resp);
    // 204 No Content or DELETE 200 with no body
    const text = await resp.text();
    return (text ? JSON.parse(text) : undefined) as T;
  }

  private rawRequest(method: string, path: string, body?: unknown): Promise<Response> {
    return fetch(`${this.baseUrl}${path}`, {
      method,
      headers: this.headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  }

  private async throwFromResponse(resp: Response): Promise<never> {
    let message: string;
    try {
      const data = (await resp.json()) as { error?: string };
      message = data.error ?? resp.statusText;
    } catch {
      message = resp.statusText;
    }
    throw new QuiverError(message, resp.status);
  }
}
