"""
/index routes — on-demand crawl & index, plus stats.

POST /index/url    — crawl and index one URL (synchronous, await)
POST /index/batch  — launch a background crawl job (async)
GET  /index/stats  — current index statistics
GET  /index/jobs/{job_id} — check background job status
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from loguru import logger

from api.schemas import (
    IndexBatchRequest,
    IndexJobStatus,
    IndexStatsResponse,
    IndexURLRequest,
)
from crawler.async_crawler import AsyncCrawler
from crawler.content_extractor import ContentExtractor
from indexer.pipeline import IndexPipeline

router = APIRouter()

# In-memory job registry (replace with Redis/DB for production)
_JOBS: dict[str, IndexJobStatus] = {}


# ---------------------------------------------------------------------------
# Single URL indexing
# ---------------------------------------------------------------------------

@router.post("/url", response_model=IndexJobStatus)
async def index_url(req: IndexURLRequest, request: Request) -> IndexJobStatus:
    """Crawl and index a single URL synchronously."""
    url   = str(req.url)
    state = request.app.state

    crawler   = AsyncCrawler(max_depth=req.depth, max_pages=500, concurrency=8)
    extractor = ContentExtractor()
    pipeline  = _build_pipeline(state, crawler, extractor)

    t0 = time.monotonic()
    try:
        stats = await pipeline.run([url])
    except Exception as exc:
        logger.exception(f"Index URL failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    job = IndexJobStatus(
        job_id=str(uuid.uuid4()),
        status="done",
        crawled=stats.crawled,
        indexed=stats.indexed,
        elapsed_s=round(time.monotonic() - t0, 2),
    )
    _JOBS[job.job_id] = job
    return job


# ---------------------------------------------------------------------------
# Batch crawl job (background)
# ---------------------------------------------------------------------------

@router.post("/batch", response_model=IndexJobStatus)
async def index_batch(
    req: IndexBatchRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> IndexJobStatus:
    """Queue a batch crawl job; returns immediately with job_id."""
    job_id = str(uuid.uuid4())
    job    = IndexJobStatus(job_id=job_id, status="pending")
    _JOBS[job_id] = job

    state = request.app.state
    seeds = [str(u) for u in req.seeds]

    background_tasks.add_task(
        _run_batch_job,
        job_id=job_id,
        seeds=seeds,
        max_pages=req.max_pages,
        max_depth=req.max_depth,
        concurrency=req.concurrency,
        state=state,
    )
    return job


@router.get("/jobs/{job_id}", response_model=IndexJobStatus)
async def get_job(job_id: str) -> IndexJobStatus:
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _JOBS[job_id]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@router.get("/stats", response_model=IndexStatsResponse)
async def get_stats(request: Request) -> IndexStatsResponse:
    state = request.app.state
    return IndexStatsResponse(
        bm25_docs=len(state.bm25),
        vector_docs=len(state.vector_store),
        graph_nodes=state.link_graph.node_count(),
        graph_edges=state.link_graph.edge_count(),
    )


# ---------------------------------------------------------------------------
# Background job runner
# ---------------------------------------------------------------------------

async def _run_batch_job(
    job_id: str,
    seeds: list[str],
    max_pages: int,
    max_depth: int,
    concurrency: int,
    state: Any,
) -> None:
    job = _JOBS[job_id]
    job.status = "running"
    t0 = time.monotonic()

    try:
        crawler   = AsyncCrawler(
            max_depth=max_depth,
            max_pages=max_pages,
            concurrency=concurrency,
        )
        extractor = ContentExtractor()
        pipeline  = _build_pipeline(state, crawler, extractor)
        stats     = await pipeline.run(seeds)

        job.status    = "done"
        job.crawled   = stats.crawled
        job.indexed   = stats.indexed
        job.elapsed_s = round(time.monotonic() - t0, 2)

    except Exception as exc:
        logger.exception(f"Batch job {job_id} failed: {exc}")
        job.status = "failed"
        job.error  = str(exc)
        job.elapsed_s = round(time.monotonic() - t0, 2)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_pipeline(state: Any, crawler: AsyncCrawler, extractor: ContentExtractor) -> IndexPipeline:
    from indexer.pipeline import IndexPipeline
    return IndexPipeline(
        crawler=crawler,
        extractor=extractor,
        quality=state.quality,
        dedup=state.dedup,
        embedder=state.embedder,
        bm25=state.bm25,
        vector_store=state.vector_store,
        link_graph=state.link_graph,
    )
