"""
Integration tests for the API layer — /search, /search/suggest,
/index/*, /health, /metrics — plus the auth and rate-limit middleware.

These build the FastAPI app directly (bypassing `create_app`'s lifespan,
which loads real embedder/reranker/crawler models) and populate
`app.state` with lightweight fakes, the same pattern used by
tests/test_hybrid_retriever.py for the retriever itself.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.auth import APIKeyMiddleware
from api.metrics import MetricsMiddleware, MetricsRegistry
from api.rate_limit import RateLimitMiddleware
from api.routes import index as index_router
from api.routes import search as search_router
from indexer.bm25 import BM25Index
from indexer.vector_store import VectorStore
from search.hybrid_retriever import HybridRetriever, SearchResult
from search.query_processor import QueryProcessor
from utils.cache import TTLCache


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeReranker:
    """No-op reranker: keeps retriever order, sets a fake rerank_score."""

    def rerank(self, query, results, *, text_map=None, top_k=None):
        for r in results:
            r.rerank_score = r.rrf_score
        return results[:top_k] if top_k else results


class _FakeNeuralFilter:
    """Pass-through filter — keeps every candidate."""

    def filter(self, query, results, *, text_map=None):
        for r in results:
            r.metadata["neural_sim"] = 0.5
        return results


def _seeded_bm25() -> BM25Index:
    bm25 = BM25Index()
    bm25.add(
        "Flash attention reduces memory usage in transformer training.",
        {"url": "http://a.test", "title": "Flash Attention", "text": "Flash attention reduces memory usage in transformer training."},
    )
    bm25.add(
        "The quick brown fox jumps over the lazy dog.",
        {"url": "http://b.test", "title": "Fox", "text": "The quick brown fox jumps over the lazy dog."},
    )
    return bm25


def _build_app(*, api_keys=None, rate_limit_per_minute=0, cache_ttl=30.0) -> FastAPI:
    app = FastAPI()

    app.add_middleware(APIKeyMiddleware, api_keys=api_keys or [])
    app.add_middleware(RateLimitMiddleware, requests_per_minute=rate_limit_per_minute)
    metrics_registry = MetricsRegistry()
    app.add_middleware(MetricsMiddleware, registry=metrics_registry)

    app.include_router(search_router.router, prefix="/search", tags=["search"])
    app.include_router(index_router.router, prefix="/index", tags=["index"])

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    from fastapi.responses import PlainTextResponse

    @app.get("/metrics", response_class=PlainTextResponse)
    async def get_metrics():
        return metrics_registry.render_prometheus()

    bm25 = _seeded_bm25()
    vector_store = VectorStore(dim=4, index_type="flat")
    vector_store.add(0, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), {"url": "http://a.test"})
    vector_store.add(1, np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), {"url": "http://b.test"})

    bm25_hits = [
        (0, 5.0, {"url": "http://a.test", "title": "Flash Attention"}),
        (1, 3.0, {"url": "http://b.test", "title": "Fox"}),
    ]

    class _FakeBM25Retriever:
        """Wraps the real BM25Index but returns a fixed candidate set,
        so /search exercises real fusion logic without depending on
        BM25's actual scoring for this fixture text."""

        def search(self, query, top_k=100):
            return bm25_hits[:top_k]

    class _FakeVectorRetriever:
        def search(self, query_vec, top_k=100):
            return []

    class _FakeEmbedder:
        def encode_query(self, text):
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        def encode_passages(self, texts):
            return np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1)).astype(np.float32)

    retriever = HybridRetriever(
        bm25=_FakeBM25Retriever(),
        vector_store=_FakeVectorRetriever(),
        embedder=_FakeEmbedder(),
        link_graph=None,
    )

    app.state.settings = None
    app.state.bm25 = bm25
    app.state.vector_store = vector_store
    app.state.link_graph = type("LG", (), {"node_count": lambda self: 2, "edge_count": lambda self: 1})()
    app.state.query_proc = QueryProcessor(expand_synonyms=False)
    app.state.retriever = retriever
    app.state.reranker = _FakeReranker()
    app.state.neural_filter = _FakeNeuralFilter()
    app.state.search_cache = TTLCache(maxsize=64, ttl=cache_ttl)

    return app


@pytest.fixture
def client() -> TestClient:
    return TestClient(_build_app())


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_ok(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /search
# ---------------------------------------------------------------------------

def test_search_returns_results(client: TestClient):
    resp = client.post("/search", json={"query": "flash attention", "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "flash attention"
    assert len(body["results"]) > 0
    assert body["cached"] is False


def test_search_second_identical_request_is_cached(client: TestClient):
    payload = {"query": "flash attention", "top_k": 5}
    first = client.post("/search", json=payload).json()
    second = client.post("/search", json=payload).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["results"] == first["results"]


def test_search_offset_paginates_results(client: TestClient):
    full = client.post("/search", json={"query": "attention fox", "top_k": 2, "rerank": True}).json()
    assert len(full["results"]) == 2

    page2 = client.post(
        "/search", json={"query": "attention fox", "top_k": 1, "offset": 1, "rerank": True}
    ).json()
    assert len(page2["results"]) == 1
    assert page2["results"][0]["url"] == full["results"][1]["url"]


def test_search_empty_query_is_rejected(client: TestClient):
    resp = client.post("/search", json={"query": ""})
    assert resp.status_code == 422


def test_search_offset_out_of_range_is_rejected(client: TestClient):
    resp = client.post("/search", json={"query": "x", "offset": -1})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /search/suggest
# ---------------------------------------------------------------------------

def test_suggest_returns_matching_prefix(client: TestClient):
    resp = client.get("/search/suggest", params={"q": "atten"})
    assert resp.status_code == 200
    body = resp.json()
    assert "attention" in body["suggestions"]


def test_suggest_requires_query_param(client: TestClient):
    resp = client.get("/search/suggest")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /index/url (DELETE) and /index/stats
# ---------------------------------------------------------------------------

def test_delete_indexed_url_succeeds(client: TestClient):
    resp = client.delete("/index/url", params={"url": "http://a.test"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] is True
    assert body["doc_id"] == 0


def test_delete_unknown_url_returns_404(client: TestClient):
    resp = client.delete("/index/url", params={"url": "http://not-indexed.test"})
    assert resp.status_code == 404


def test_delete_clears_search_cache(client: TestClient):
    payload = {"query": "flash attention", "top_k": 5}
    first = client.post("/search", json=payload).json()
    assert first["cached"] is False
    # Cache now warm — repeating would normally hit it...
    client.delete("/index/url", params={"url": "http://a.test"})
    # ...but the delete should have cleared it.
    second = client.post("/search", json=payload).json()
    assert second["cached"] is False


def test_index_stats_reports_deleted_counts(client: TestClient):
    client.delete("/index/url", params={"url": "http://a.test"})
    resp = client.get("/index/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["bm25_deleted"] == 1


# ---------------------------------------------------------------------------
# /index/jobs
# ---------------------------------------------------------------------------

def test_list_jobs_empty_initially(client: TestClient):
    resp = client.get("/index/jobs")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_unknown_job_returns_404(client: TestClient):
    resp = client.get("/index/jobs/does-not-exist")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------

def test_metrics_endpoint_reports_request_counts(client: TestClient):
    client.get("/health")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "axon_http_requests_total" in resp.text
    assert 'path="/health"' in resp.text


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------

def test_auth_disabled_by_default_allows_requests():
    app = _build_app(api_keys=[])
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/index/stats").status_code == 200


def test_auth_enabled_rejects_missing_key():
    app = _build_app(api_keys=["secret123"])
    with TestClient(app) as c:
        resp = c.get("/index/stats")
        assert resp.status_code == 401


def test_auth_enabled_accepts_valid_key():
    app = _build_app(api_keys=["secret123"])
    with TestClient(app) as c:
        resp = c.get("/index/stats", headers={"X-API-Key": "secret123"})
        assert resp.status_code == 200


def test_auth_enabled_still_allows_health_without_key():
    app = _build_app(api_keys=["secret123"])
    with TestClient(app) as c:
        resp = c.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Rate limit middleware
# ---------------------------------------------------------------------------

def test_rate_limit_disabled_by_default_allows_many_requests():
    app = _build_app(rate_limit_per_minute=0)
    with TestClient(app) as c:
        for _ in range(10):
            assert c.get("/index/stats").status_code == 200


def test_rate_limit_enabled_blocks_after_threshold():
    app = _build_app(rate_limit_per_minute=3)
    with TestClient(app) as c:
        statuses = [c.get("/index/stats").status_code for _ in range(5)]
        assert statuses[:3] == [200, 200, 200]
        assert 429 in statuses[3:]


def test_rate_limit_exempts_health():
    app = _build_app(rate_limit_per_minute=1)
    with TestClient(app) as c:
        c.get("/index/stats")  # consume the one allowed slot
        for _ in range(5):
            assert c.get("/health").status_code == 200
