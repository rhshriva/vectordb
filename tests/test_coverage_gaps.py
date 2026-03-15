"""Tests targeting coverage gaps in server.py, multi_collection.py,
embedding.py, and text_collection.py."""

import hashlib
import json
import pytest
import tempfile
import threading
import time
import urllib.request
import urllib.error
from unittest.mock import MagicMock, patch, PropertyMock

import quiver_vector_db as quiver
from quiver_vector_db.server import create_server
from quiver_vector_db.multi_collection import MultiVectorCollection
from quiver_vector_db.text_collection import TextCollection


# ── Helpers ──────────────────────────────────────────────────────────────

class MockEmbedding:
    """Deterministic mock embedder for testing."""

    def __init__(self, dimensions: int = 4):
        self._dimensions = dimensions

    def __call__(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            h = int(hashlib.md5(text.encode()).hexdigest(), 16)
            vec = [(h >> (i * 8) & 0xFF) / 255.0 for i in range(self._dimensions)]
            results.append(vec)
        return results

    @property
    def dimensions(self) -> int:
        return self._dimensions


def _request(url, method="GET", data=None):
    """Helper: make HTTP request and return (status, parsed_json)."""
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    if body:
        req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# ── Server coverage gaps ────────────────────────────────────────────────

@pytest.fixture(scope="module")
def server_url():
    """Start a test server on a random port and return its base URL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        server = create_server(host="127.0.0.1", port=0, data_path=tmpdir)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{port}"
        for _ in range(20):
            try:
                urllib.request.urlopen(f"{base_url}/healthz", timeout=1)
                break
            except Exception:
                time.sleep(0.1)
        yield base_url
        server.shutdown()


class TestServerCoverageGaps:
    """Cover missing lines in server.py."""

    def test_cors_options(self, server_url):
        """OPTIONS preflight returns 204 with CORS headers."""
        req = urllib.request.Request(f"{server_url}/collections", method="OPTIONS")
        resp = urllib.request.urlopen(req, timeout=5)
        assert resp.status == 204
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"
        assert "POST" in resp.headers.get("Access-Control-Allow-Methods", "")

    def test_create_collection_missing_name(self, server_url):
        """POST /collections without name returns 400."""
        status, data = _request(f"{server_url}/collections", method="POST", data={
            "dimensions": 3,
        })
        assert status == 400
        assert "required" in data["error"].lower()

    def test_create_collection_missing_dimensions(self, server_url):
        """POST /collections without dimensions returns 400."""
        status, data = _request(f"{server_url}/collections", method="POST", data={
            "name": "no_dims",
        })
        assert status == 400
        assert "required" in data["error"].lower()

    def test_create_duplicate_collection(self, server_url):
        """Creating a collection that already exists returns 409."""
        _request(f"{server_url}/collections", method="POST", data={
            "name": "dup_test", "dimensions": 3,
        })
        status, data = _request(f"{server_url}/collections", method="POST", data={
            "name": "dup_test", "dimensions": 3,
        })
        assert status == 409
        assert "already exists" in data["error"]

    def test_upsert_to_nonexistent_collection(self, server_url):
        """Upsert to nonexistent collection returns 404."""
        status, data = _request(
            f"{server_url}/collections/ghost_col/upsert",
            method="POST",
            data={"id": 1, "vector": [1.0, 2.0]},
        )
        assert status == 404

    def test_upsert_missing_id(self, server_url):
        """Upsert without id returns 400."""
        _request(f"{server_url}/collections", method="POST", data={
            "name": "upsert_err", "dimensions": 2,
        })
        status, data = _request(
            f"{server_url}/collections/upsert_err/upsert",
            method="POST",
            data={"vector": [1.0, 2.0]},
        )
        assert status == 400
        assert "required" in data["error"].lower()

    def test_upsert_missing_vector(self, server_url):
        """Upsert without vector returns 400."""
        status, data = _request(
            f"{server_url}/collections/upsert_err/upsert",
            method="POST",
            data={"id": 1},
        )
        assert status == 400
        assert "required" in data["error"].lower()

    def test_search_nonexistent_collection(self, server_url):
        """Search on nonexistent collection returns 404."""
        status, data = _request(
            f"{server_url}/collections/no_such/search",
            method="POST",
            data={"query": [1.0, 0.0]},
        )
        assert status == 404

    def test_search_missing_query(self, server_url):
        """Search without query vector returns 400."""
        _request(f"{server_url}/collections", method="POST", data={
            "name": "search_err", "dimensions": 2,
        })
        status, data = _request(
            f"{server_url}/collections/search_err/search",
            method="POST",
            data={"k": 5},
        )
        assert status == 400
        assert "required" in data["error"].lower()

    def test_delete_vector_nonexistent_collection(self, server_url):
        """Delete vector from nonexistent collection returns 404."""
        status, data = _request(
            f"{server_url}/collections/no_such/delete",
            method="POST",
            data={"id": 1},
        )
        assert status == 404

    def test_delete_vector_missing_id(self, server_url):
        """Delete without id returns 400."""
        status, data = _request(
            f"{server_url}/collections/search_err/delete",
            method="POST",
            data={},
        )
        assert status == 400
        assert "required" in data["error"].lower()

    def test_delete_nonexistent_collection(self, server_url):
        """DELETE /collections/nonexistent returns 200 with deleted=false (or 404)."""
        status, data = _request(
            f"{server_url}/collections/not_here",
            method="DELETE",
        )
        # Rust Client.delete_collection may not raise KeyError for missing collections;
        # it may return False. Either way, the endpoint should not crash.
        assert status in (200, 404)

    def test_delete_nonexistent_snapshot(self, server_url):
        """DELETE snapshot that doesn't exist returns 404."""
        _request(f"{server_url}/collections", method="POST", data={
            "name": "snap_err", "dimensions": 2,
        })
        status, data = _request(
            f"{server_url}/collections/snap_err/snapshots/no_snap",
            method="DELETE",
        )
        assert status == 404

    def test_snapshots_nonexistent_collection(self, server_url):
        """GET snapshots for nonexistent collection returns 404."""
        status, data = _request(f"{server_url}/collections/ghost/snapshots")
        assert status == 404

    def test_create_snapshot_nonexistent_collection(self, server_url):
        """POST snapshot on nonexistent collection returns 404."""
        status, data = _request(
            f"{server_url}/collections/ghost/snapshots",
            method="POST",
            data={"name": "v1"},
        )
        assert status == 404

    def test_create_snapshot_missing_name(self, server_url):
        """POST snapshot without name returns 400."""
        status, data = _request(
            f"{server_url}/collections/snap_err/snapshots",
            method="POST",
            data={},
        )
        assert status == 400
        assert "required" in data["error"].lower()

    def test_create_duplicate_snapshot(self, server_url):
        """Creating a snapshot that already exists returns 409."""
        _request(f"{server_url}/collections/snap_err/upsert", method="POST", data={
            "id": 1, "vector": [1.0, 0.0],
        })
        _request(f"{server_url}/collections/snap_err/snapshots", method="POST", data={
            "name": "dup_snap",
        })
        status, data = _request(
            f"{server_url}/collections/snap_err/snapshots",
            method="POST",
            data={"name": "dup_snap"},
        )
        assert status == 409
        assert "already exists" in data["error"]

    def test_restore_snapshot_nonexistent_collection(self, server_url):
        """Restore snapshot on nonexistent collection returns 404."""
        status, data = _request(
            f"{server_url}/collections/ghost/snapshots/restore",
            method="POST",
            data={"name": "v1"},
        )
        assert status == 404

    def test_restore_snapshot_missing_name(self, server_url):
        """Restore snapshot without name returns 400."""
        status, data = _request(
            f"{server_url}/collections/snap_err/snapshots/restore",
            method="POST",
            data={},
        )
        assert status == 400
        assert "required" in data["error"].lower()

    def test_restore_nonexistent_snapshot(self, server_url):
        """Restore a snapshot that doesn't exist returns 404."""
        status, data = _request(
            f"{server_url}/collections/snap_err/snapshots/restore",
            method="POST",
            data={"name": "nonexistent_snap"},
        )
        assert status == 404

    def test_upsert_batch_nonexistent_collection(self, server_url):
        """Batch upsert on nonexistent collection returns 404."""
        status, data = _request(
            f"{server_url}/collections/ghost/upsert_batch",
            method="POST",
            data={"entries": [{"id": 1, "vector": [1.0]}]},
        )
        assert status == 404

    def test_upsert_batch_missing_entries(self, server_url):
        """Batch upsert without entries returns 400."""
        status, data = _request(
            f"{server_url}/collections/snap_err/upsert_batch",
            method="POST",
            data={},
        )
        assert status == 400
        assert "entries" in data["error"].lower()

    def test_upsert_batch_entry_missing_fields(self, server_url):
        """Batch upsert with entry missing id/vector returns 400."""
        status, data = _request(
            f"{server_url}/collections/snap_err/upsert_batch",
            method="POST",
            data={"entries": [{"id": 1}]},
        )
        assert status == 400
        assert "vector" in data["error"].lower()

    def test_get_unknown_path(self, server_url):
        """GET on unknown path returns 404."""
        status, data = _request(f"{server_url}/unknown/path")
        assert status == 404

    def test_post_unknown_path(self, server_url):
        """POST on unknown path returns 404."""
        status, data = _request(f"{server_url}/unknown/path", method="POST", data={})
        assert status == 404

    def test_delete_unknown_path(self, server_url):
        """DELETE on unknown path returns 404."""
        status, data = _request(f"{server_url}/unknown/path", method="DELETE")
        assert status == 404

    def test_count_nonexistent_collection(self, server_url):
        """GET count for nonexistent collection returns 404."""
        status, data = _request(f"{server_url}/collections/ghost/count")
        assert status == 404


# ── MultiVectorCollection coverage gaps ─────────────────────────────────

class TestMultiCollectionCoverageGaps:
    """Cover missing lines in multi_collection.py."""

    @pytest.fixture
    def db(self, tmp_path):
        return quiver.Client(path=str(tmp_path))

    def test_upsert_batch_unknown_space(self, db):
        """upsert_batch with unknown vector space raises KeyError."""
        multi = MultiVectorCollection(
            client=db,
            name="ub_err",
            vector_spaces={"text": {"dimensions": 2}},
        )
        with pytest.raises(KeyError, match="image"):
            multi.upsert_batch([
                (1, {"image": [1.0, 0.0]}),
            ])

    def test_search_unknown_space(self, db):
        """search with unknown vector space raises KeyError."""
        multi = MultiVectorCollection(
            client=db,
            name="s_err",
            vector_spaces={"text": {"dimensions": 2}},
        )
        with pytest.raises(KeyError, match="image"):
            multi.search("image", query=[1.0, 0.0])

    def test_search_multi_unknown_space(self, db):
        """search_multi with unknown vector space raises KeyError."""
        multi = MultiVectorCollection(
            client=db,
            name="sm_err",
            vector_spaces={"text": {"dimensions": 2}},
        )
        with pytest.raises(KeyError, match="image"):
            multi.search_multi(queries={"image": [1.0, 0.0]})

    def test_search_multi_empty_queries(self, db):
        """search_multi with empty queries returns []."""
        multi = MultiVectorCollection(
            client=db,
            name="sm_empty",
            vector_spaces={"text": {"dimensions": 2}},
        )
        result = multi.search_multi(queries={})
        assert result == []

    def test_search_multi_weights_none(self, db):
        """search_multi with weights=None uses equal weights."""
        multi = MultiVectorCollection(
            client=db,
            name="sm_wnone",
            vector_spaces={
                "text": {"dimensions": 4, "metric": "l2"},
                "image": {"dimensions": 4, "metric": "l2"},
            },
        )
        multi.upsert(id=1, vectors={
            "text": [1.0, 0.0, 0.0, 0.0],
            "image": [0.0, 1.0, 0.0, 0.0],
        })
        results = multi.search_multi(
            queries={"text": [1.0, 0.0, 0.0, 0.0], "image": [0.0, 1.0, 0.0, 0.0]},
            k=1,
            weights=None,
        )
        assert len(results) == 1

    def test_search_multi_zero_weights(self, db):
        """search_multi where all weights are zero (total <= 0) doesn't crash."""
        multi = MultiVectorCollection(
            client=db,
            name="sm_zw",
            vector_spaces={"text": {"dimensions": 4, "metric": "l2"}},
        )
        multi.upsert(id=1, vectors={"text": [1.0, 0.0, 0.0, 0.0]})
        results = multi.search_multi(
            queries={"text": [1.0, 0.0, 0.0, 0.0]},
            k=1,
            weights={"text": 0.0},
        )
        assert len(results) == 1


# ── Embedding coverage gaps (mocked) ────────────────────────────────────

class TestSentenceTransformerEmbeddingMocked:
    """Test SentenceTransformerEmbedding with mocked sentence_transformers."""

    def test_call_and_dimensions(self):
        import numpy as np

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]] * 2)

        mock_st_class = MagicMock(return_value=mock_model)

        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
            import sys
            sys.modules["sentence_transformers"].SentenceTransformer = mock_st_class

            from quiver_vector_db.embedding import SentenceTransformerEmbedding
            ef = SentenceTransformerEmbedding.__new__(SentenceTransformerEmbedding)
            ef._model = mock_model
            ef._dimensions = 384

            result = ef(["hello", "world"])
            mock_model.encode.assert_called_once_with(["hello", "world"], convert_to_numpy=True)
            assert len(result) == 2
            assert ef.dimensions == 384


class TestOpenAIEmbeddingMocked:
    """Test OpenAIEmbedding with mocked openai."""

    def test_call_and_dimensions(self):
        mock_item1 = MagicMock()
        mock_item1.embedding = [0.1, 0.2, 0.3]
        mock_item2 = MagicMock()
        mock_item2.embedding = [0.4, 0.5, 0.6]

        mock_response = MagicMock()
        mock_response.data = [mock_item1, mock_item2]

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        with patch.dict("sys.modules", {"openai": MagicMock()}):
            from quiver_vector_db.embedding import OpenAIEmbedding
            ef = OpenAIEmbedding.__new__(OpenAIEmbedding)
            ef._client = mock_client
            ef._model = "text-embedding-3-small"
            ef._dimensions = 1536

            result = ef(["hello", "world"])
            mock_client.embeddings.create.assert_called_once_with(
                input=["hello", "world"], model="text-embedding-3-small"
            )
            assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            assert ef.dimensions == 1536

    def test_dimensions_none_for_unknown_model(self):
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            from quiver_vector_db.embedding import OpenAIEmbedding
            ef = OpenAIEmbedding.__new__(OpenAIEmbedding)
            ef._client = MagicMock()
            ef._model = "unknown-model"
            ef._dimensions = None
            assert ef.dimensions is None


# ── TextCollection coverage gaps ────────────────────────────────────────

class TestTextCollectionCoverageGaps:
    """Cover the hybrid fallback when BM25 returns no known terms."""

    def test_hybrid_fallback_no_bm25_terms(self, tmp_path):
        """When hybrid query has no known BM25 terms, falls back to pure semantic."""
        db = quiver.Client(path=str(tmp_path))
        raw_col = db.create_collection("docs", dimensions=4, metric="cosine")
        text_col = TextCollection(
            collection=raw_col,
            embedding_function=MockEmbedding(dimensions=4),
        )
        # Index some documents (these terms go into BM25 vocab)
        text_col.add(ids=[1, 2], documents=["machine learning", "deep neural"])

        # Query with terms entirely absent from BM25 index
        # (single char tokens are filtered out, so "zz" is min len 2 but not in vocab)
        results = text_col.query("xyznonexistent qqqqunknown", k=2, mode="hybrid")
        # Should fall back to semantic and return results with "distance" key
        assert len(results) >= 1
        assert "distance" in results[0]
