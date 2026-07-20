"""
FastAPI application factory.

All heavy objects (BM25, VectorStore, Embedder, Reranker) are
instantiated once during startup via the lifespan context and stored
in app.state so route handlers can access them without globals.

Endpoints
---------
GET  /health                    — liveness probe
POST /search                    — main search
POST /index/url                 — index a single URL on demand
POST /index/batch                — queue a batch of seeds for crawling
GET  /index/stats               — index statistics
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config import get_settings
from indexer.bm25 import BM25Index
from indexer.embedder import Embedder
from indexer.vector_store import VectorStore
from crawler.link_graph import LinkGraph
from search.query_processor import QueryProcessor
from search.hybrid_retriever import HybridRetriever
from search.reranker import CrossEncoderReranker
from search.neural_filter import NeuralFilter
from utils.dedup import DedupFilter
from utils.quality_scorer import QualityScorer

from .routes import search as search_router
from .routes import index as index_router


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load all indices and models on startup; release on shutdown."""
    settings = get_settings()
    logger.info("Axon Search starting up…")

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    bm25_path   = settings.index_dir / "bm25.pkl"
    vector_path = settings.index_dir / "vectors"

    # --- BM25 ---
    if bm25_path.exists():
        bm25 = BM25Index.load(bm25_path)
    else:
        bm25 = BM25Index()

    # --- Embedder ---
    embedder = Embedder(
        settings.embed_model,
        batch_size=settings.embed_batch_size,
        device=settings.embed_device,
        max_length=settings.embed_max_length,
    )

    # --- VectorStore ---
    if (vector_path / "faiss.index").exists():
        vector_store = VectorStore.load(
            vector_path, dim=settings.embed_dim, index_type=settings.vector_index_type
        )
    else:
        vector_store = VectorStore(dim=settings.embed_dim, index_type=settings.vector_index_type)

    # --- Link graph ---
    link_graph = LinkGraph()

    # --- Search pipeline ---
    query_proc = QueryProcessor()
    retriever  = HybridRetriever(
        bm25=bm25,
        vector_store=vector_store,
        embedder=embedder,
        link_graph=link_graph,
        bm25_weight=settings.bm25_weight,
        dense_weight=settings.dense_weight,
        pagerank_alpha=settings.pagerank_alpha,
    )
    reranker = CrossEncoderReranker(settings.rerank_model, device=settings.rerank_device)
    neural_filter = NeuralFilter(embedder, threshold=settings.neural_filter_threshold)

    # --- Indexing helpers ---
    quality = QualityScorer()
    dedup   = DedupFilter(threshold=settings.dedup_threshold, num_perm=settings.dedup_num_perm)

    # Store in app.state
    app.state.settings      = settings
    app.state.bm25          = bm25
    app.state.embedder      = embedder
    app.state.vector_store  = vector_store
    app.state.link_graph    = link_graph
    app.state.query_proc    = query_proc
    app.state.retriever     = retriever
    app.state.reranker      = reranker
    app.state.neural_filter = neural_filter
    app.state.quality       = quality
    app.state.dedup         = dedup

    logger.info("Axon Search ready")
    yield

    # Shutdown: persist indices
    logger.info("Saving indices…")
    bm25.save(bm25_path)
    vector_store.save(vector_path)
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Axon Search",
        description="Hybrid semantic search engine",
        version="0.1.0",
        lifespan=_lifespan,
    )

    # CORS is locked down by default (no cross-origin access) unless the
    # deployer explicitly sets CORS_ALLOW_ORIGINS. The previous default of
    # "*" for origins/methods/headers is not safe for a service that can
    # trigger outbound crawls and index writes.
    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "Authorization"],
        )

    app.include_router(search_router.router, prefix="/search", tags=["search"])
    app.include_router(index_router.router, prefix="/index",  tags=["index"])

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.server:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
    )
