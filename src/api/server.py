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
POST /index/batch               — queue a batch of seeds for crawling
GET  /index/stats               — index statistics
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

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
# Configuration (override with env vars or a config file)
# ---------------------------------------------------------------------------

INDEX_DIR     = Path("./data/index")
EMBED_MODEL   = "BAAI/bge-large-en-v1.5"
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-12-v2"
EMBED_DIM     = 1024    # bge-large dimension
VECTOR_TYPE   = "hnsw"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load all indices and models on startup; release on shutdown."""
    logger.info("Axon Search starting up…")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    bm25_path   = INDEX_DIR / "bm25.pkl"
    vector_path = INDEX_DIR / "vectors"

    # --- BM25 ---
    if bm25_path.exists():
        bm25 = BM25Index.load(bm25_path)
    else:
        bm25 = BM25Index()

    # --- Embedder ---
    embedder = Embedder(EMBED_MODEL, batch_size=64)

    # --- VectorStore ---
    if (vector_path / "faiss.index").exists():
        vector_store = VectorStore.load(vector_path, dim=EMBED_DIM, index_type=VECTOR_TYPE)
    else:
        vector_store = VectorStore(dim=EMBED_DIM, index_type=VECTOR_TYPE)

    # --- Link graph ---
    link_graph = LinkGraph()

    # --- Search pipeline ---
    query_proc = QueryProcessor()
    retriever  = HybridRetriever(
        bm25=bm25,
        vector_store=vector_store,
        embedder=embedder,
        link_graph=link_graph,
    )
    reranker = CrossEncoderReranker(RERANK_MODEL)
    neural_filter = NeuralFilter(embedder, threshold=0.2)

    # --- Indexing helpers ---
    quality = QualityScorer()
    dedup   = DedupFilter()

    # Store in app.state
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
    app = FastAPI(
        title="Axon Search",
        description="Hybrid semantic search engine",
        version="0.1.0",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
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
    uvicorn.run(
        "api.server:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )
