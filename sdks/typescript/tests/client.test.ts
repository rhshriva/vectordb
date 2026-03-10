import { describe, it, expect, beforeAll, afterAll, afterEach } from "vitest";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { QuiverClient, QuiverError } from "../src/client.js";

const BASE_URL = "http://localhost:7070";
const client = new QuiverClient({ baseUrl: BASE_URL });
const authedClient = new QuiverClient({ baseUrl: BASE_URL, apiKey: "secret" });

// ── Mock server setup ─────────────────────────────────────────────────────────

const server = setupServer(
  // Create collection
  http.post(`${BASE_URL}/collections/:name`, () => {
    return new HttpResponse(null, { status: 201 });
  }),

  // List collections
  http.get(`${BASE_URL}/collections`, () => {
    return HttpResponse.json({ collections: ["alpha", "beta"] });
  }),

  // Get collection info
  http.get(`${BASE_URL}/collections/:name`, ({ params }) => {
    return HttpResponse.json({
      name: params.name,
      count: 42,
      dimensions: 3,
      metric: "cosine",
    });
  }),

  // Upsert
  http.post(`${BASE_URL}/collections/:name/upsert`, () => {
    return new HttpResponse(null, { status: 200 });
  }),

  // Search
  http.post(`${BASE_URL}/collections/:name/search`, () => {
    return HttpResponse.json({
      results: [
        { id: 1, distance: 0.05 },
        { id: 2, distance: 0.12, payload: { tag: "news" } },
      ],
    });
  }),

  // Delete vector
  http.delete(`${BASE_URL}/collections/:name/vectors/:id`, () => {
    return HttpResponse.json({ deleted: true });
  }),

  // Delete collection
  http.delete(`${BASE_URL}/collections/:name`, ({ params }) => {
    if (params.name === "missing") {
      return new HttpResponse(null, { status: 404 });
    }
    return new HttpResponse(null, { status: 200 });
  }),
);

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("QuiverClient", () => {
  describe("createCollection", () => {
    it("sends correct request and resolves", async () => {
      await expect(
        client.createCollection("docs", { dimensions: 3, metric: "cosine" }),
      ).resolves.toBeUndefined();
    });

    it("throws QuiverError on conflict", async () => {
      server.use(
        http.post(`${BASE_URL}/collections/:name`, () =>
          HttpResponse.json({ error: "collection already exists: docs" }, { status: 409 }),
        ),
      );
      await expect(
        client.createCollection("docs", { dimensions: 3 }),
      ).rejects.toBeInstanceOf(QuiverError);
    });
  });

  describe("listCollections", () => {
    it("returns array of names", async () => {
      const names = await client.listCollections();
      expect(names).toEqual(["alpha", "beta"]);
    });
  });

  describe("getCollection", () => {
    it("returns collection info", async () => {
      const info = await client.getCollection("docs");
      expect(info.name).toBe("docs");
      expect(info.count).toBe(42);
      expect(info.dimensions).toBe(3);
    });
  });

  describe("upsert", () => {
    it("resolves without payload", async () => {
      await expect(client.upsert("docs", 1, [1, 0, 0])).resolves.toBeUndefined();
    });

    it("resolves with payload", async () => {
      await expect(
        client.upsert("docs", 2, [0, 1, 0], { tag: "news" }),
      ).resolves.toBeUndefined();
    });
  });

  describe("search", () => {
    it("returns results without filter", async () => {
      const results = await client.search("docs", [1, 0, 0], 2);
      expect(results).toHaveLength(2);
      expect(results[0].id).toBe(1);
      expect(results[0].distance).toBe(0.05);
    });

    it("returns payload in results", async () => {
      const results = await client.search("docs", [1, 0, 0], 2);
      expect(results[1].payload).toEqual({ tag: "news" });
    });

    it("sends filter in request body", async () => {
      let capturedBody: unknown;
      server.use(
        http.post(`${BASE_URL}/collections/:name/search`, async ({ request }) => {
          capturedBody = await request.json();
          return HttpResponse.json({ results: [] });
        }),
      );
      await client.search("docs", [1, 0, 0], 5, { tag: { $eq: "a" } });
      expect(capturedBody).toMatchObject({ filter: { tag: { $eq: "a" } } });
    });
  });

  describe("delete", () => {
    it("returns true when vector deleted", async () => {
      const deleted = await client.delete("docs", 1);
      expect(deleted).toBe(true);
    });
  });

  describe("deleteCollection", () => {
    it("returns true when collection existed", async () => {
      const ok = await client.deleteCollection("docs");
      expect(ok).toBe(true);
    });

    it("returns false when collection not found", async () => {
      const ok = await client.deleteCollection("missing");
      expect(ok).toBe(false);
    });
  });

  describe("authentication", () => {
    it("sends Authorization header when apiKey provided", async () => {
      let authHeader: string | null = null;
      server.use(
        http.get(`${BASE_URL}/collections`, ({ request }) => {
          authHeader = request.headers.get("Authorization");
          return HttpResponse.json({ collections: [] });
        }),
      );
      await authedClient.listCollections();
      expect(authHeader).toBe("Bearer secret");
    });

    it("throws QuiverError on 401", async () => {
      server.use(
        http.get(`${BASE_URL}/collections`, () =>
          HttpResponse.json({ error: "unauthorized" }, { status: 401 }),
        ),
      );
      await expect(client.listCollections()).rejects.toBeInstanceOf(QuiverError);
    });
  });

  describe("QuiverError", () => {
    it("includes status code", async () => {
      server.use(
        http.get(`${BASE_URL}/collections`, () =>
          HttpResponse.json({ error: "not allowed" }, { status: 403 }),
        ),
      );
      try {
        await client.listCollections();
        expect.fail("should have thrown");
      } catch (err) {
        expect(err).toBeInstanceOf(QuiverError);
        expect((err as QuiverError).statusCode).toBe(403);
      }
    });
  });
});
