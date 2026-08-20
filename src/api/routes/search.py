"""
POST /search          — main search endpoint.
GET  /search/suggest   — lightweight autocomplete over indexed vocabulary.

Pipeline
--------
1. Parse + validate request
2. QueryProcessor.process()
3. NeuralFilter gate (optional)
4. HybridRetriever.retrieve()
5. CrossEncoderReranker.rerank() (optional)
6. Snippet generation
7. Serialize SearchResponse

Caching
-------
Identical requests (same query/top_k/offset/rerank/neural_filter/filters)
within SEARCH_CACHE_TTL_S are served from an in-memory TTL cache instead
of re-running retrieval/reranking. Disabled by setting the TTL to 0.

Pagination
----------
`offset` skips the first N fused/reranked results before applying
`top_k`, so callers can page through results without re-scoring
everything from rank 0 each time.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Query, Request
from loguru import logger

from api.schemas import ResultItem, SearchRequest, SearchResponse, SuggestResponse
from search.reranker import CrossEncoderReranker
from utils.cache import make_key

router = APIRouter()


@router.post("", response_model=SearchResponse)
async def search(req: SearchRequest, request: Request) -> SearchResponse:
    t0 = time.perf_counter()

    state       = request.app.state
    query_proc  = state.query_proc
    retriever   = state.retriever
    reranker    = state.reranker
    n_filter    = state.neural_filter
    bm25        = state.bm25
    cache       = getattr(state, "search_cache", None)

    cache_key = make_key(
        req.query, req.top_k, req.offset, req.rerank, req.neural_filter, req.filters
    )
    if cache is not None:
        cached_response = cache.get(cache_key)
        if cached_response is not None:
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return cached_response.model_copy(update={"cached": True, "elapsed_ms": elapsed})

    window = req.offset + req.top_k

    # 1. Process query
    pq = query_proc.process(req.query)
    logger.info(
        f"Search | query={req.query!r} intent={pq.intent} "
        f"operators={pq.operators} offset={req.offset}"
    )

    # 2. Retrieve (over-fetch enough candidates to cover offset + top_k,
    #    plus reranking headroom)
    candidates = retriever.retrieve(
        pq,
        top_k=window * 5 if req.rerank else window,
        filters=req.filters or None,
    )

    if not candidates:
        response = SearchResponse(
            query=req.query,
            intent=pq.intent,
            expanded_query=pq.expanded,
            results=[],
            total_retrieved=0,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        if cache is not None:
            cache.set(cache_key, response)
        return response

    total_retrieved = len(candidates)

    # Build text_map for reranker / filter
    text_map: dict[int, str] = {}
    for r in candidates:
        meta = bm25.get_metadata(r.doc_id)
        text_map[r.doc_id] = meta.get("text", r.title)

    # 3. Neural filter
    if req.neural_filter:
        candidates = n_filter.filter(req.query, candidates, text_map=text_map)

    # 4. Rerank (score/sort the full window, then apply offset below)
    if req.rerank and candidates:
        candidates = reranker.rerank(
            pq.normalised,
            candidates,
            text_map=text_map,
            top_k=window,
        )
    else:
        candidates = candidates[:window]

    # 4b. Pagination — skip the first `offset` results
    candidates = candidates[req.offset:req.offset + req.top_k]

    # 5. Snippets
    for r in candidates:
        passage = text_map.get(r.doc_id, "")
        if passage:
            r.snippet = CrossEncoderReranker.generate_snippet(
                passage, pq.tokens, window=30, max_length=220
            )

    # 6. Serialise
    items = [
        ResultItem(
            url=r.url,
            title=r.title,
            snippet=r.snippet,
            score=round(r.final_score, 6),
            rrf_score=round(r.rrf_score, 6),
            rerank_score=round(r.rerank_score, 4) if r.rerank_score is not None else None,
            pagerank=round(r.pagerank, 6),
            bm25_rank=r.bm25_rank,
            dense_rank=r.dense_rank,
            neural_sim=round(r.metadata.get("neural_sim", 0.0), 4),
        )
        for r in candidates
    ]

    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(f"Search complete | results={len(items)} elapsed={elapsed}ms")

    response = SearchResponse(
        query=req.query,
        intent=pq.intent,
        expanded_query=pq.expanded,
        results=items,
        total_retrieved=total_retrieved,
        elapsed_ms=elapsed,
    )
    if cache is not None:
        cache.set(cache_key, response)
    return response


@router.get("/suggest", response_model=SuggestResponse)
async def suggest(
    request: Request,
    q: str = Query(..., min_length=1, max_length=100, description="Prefix to autocomplete"),
    limit: int = Query(10, ge=1, le=50),
) -> SuggestResponse:
    """Autocomplete indexed vocabulary by prefix, ranked by document frequency."""
    bm25 = request.app.state.bm25
    return SuggestResponse(query=q, suggestions=bm25.suggest(q, limit=limit))
