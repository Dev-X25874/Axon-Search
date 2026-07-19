"""
POST /search — main search endpoint.

Pipeline
--------
1. Parse + validate request
2. QueryProcessor.process()
3. NeuralFilter gate (optional)
4. HybridRetriever.retrieve()
5. CrossEncoderReranker.rerank() (optional)
6. Snippet generation
7. Serialize SearchResponse
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from api.schemas import SearchRequest, SearchResponse, ResultItem
from search.reranker import CrossEncoderReranker

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

    # 1. Process query
    pq = query_proc.process(req.query)
    logger.info(
        f"Search | query={req.query!r} intent={pq.intent} "
        f"operators={pq.operators}"
    )

    # 2. Retrieve
    candidates = retriever.retrieve(
        pq,
        top_k=req.top_k * 5 if req.rerank else req.top_k,
        filters=req.filters or None,
    )

    if not candidates:
        return SearchResponse(
            query=req.query,
            intent=pq.intent,
            expanded_query=pq.expanded,
            results=[],
            total_retrieved=0,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    total_retrieved = len(candidates)

    # Build text_map for reranker / filter
    text_map: dict[int, str] = {}
    for r in candidates:
        meta = bm25.get_metadata(r.doc_id)
        text_map[r.doc_id] = meta.get("text", r.title)

    # 3. Neural filter
    if req.neural_filter:
        candidates = n_filter.filter(req.query, candidates, text_map=text_map)

    # 4. Rerank
    if req.rerank and candidates:
        candidates = reranker.rerank(
            pq.normalised,
            candidates,
            text_map=text_map,
            top_k=req.top_k,
        )
    else:
        candidates = candidates[:req.top_k]

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

    return SearchResponse(
        query=req.query,
        intent=pq.intent,
        expanded_query=pq.expanded,
        results=items,
        total_retrieved=total_retrieved,
        elapsed_ms=elapsed,
    )
