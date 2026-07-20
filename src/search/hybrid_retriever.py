"""
Hybrid retriever.

Fuses results from:
  1. BM25 sparse retrieval      (keyword precision)
  2. FAISS dense retrieval      (semantic recall)
  3. PageRank authority boost   (optional)

Fusion strategy: Reciprocal Rank Fusion (RRF)
  score(d) = Σ  1 / (k + rank_i(d))   over all rankers i
  k = 60 is the standard constant (Cormack et al. 2009).

The final list is sent to CrossEncoderReranker for re-scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from loguru import logger

from indexer.bm25 import BM25Index
from indexer.embedder import Embedder
from indexer.vector_store import VectorStore
from crawler.link_graph import LinkGraph
from .query_processor import ProcessedQuery


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    doc_id:    int
    url:       str
    title:     str
    score:     float
    bm25_rank: Optional[int]  = None
    dense_rank: Optional[int] = None
    rrf_score:  float          = 0.0
    pagerank:   float          = 0.0
    metadata:   dict           = field(default_factory=dict)
    snippet:    str            = ""
    # Filled in by reranker
    rerank_score: Optional[float] = None

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.rrf_score


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

_RRF_K = 60


class HybridRetriever:
    """
    Runs BM25 + dense search and fuses with RRF.

    Parameters
    ----------
    bm25         : BM25Index
    vector_store : VectorStore
    embedder     : Embedder
    link_graph   : LinkGraph (optional — used for PageRank boosting)
    bm25_weight  : relative weight of BM25 in RRF (default 0.5 / 0.5)
    dense_weight : relative weight of dense retrieval
    pagerank_alpha : how much PageRank authority boosts the final score
    """

    def __init__(
        self,
        *,
        bm25: BM25Index,
        vector_store: VectorStore,
        embedder: Embedder,
        link_graph: LinkGraph | None = None,
        bm25_weight: float    = 1.0,
        dense_weight: float   = 1.0,
        pagerank_alpha: float = 0.1,
        candidate_k: int      = 200,
    ):
        self.bm25           = bm25
        self.vector_store   = vector_store
        self.embedder       = embedder
        self.link_graph     = link_graph
        self.bm25_weight    = bm25_weight
        self.dense_weight   = dense_weight
        self.pagerank_alpha = pagerank_alpha
        self.candidate_k    = candidate_k

    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: ProcessedQuery,
        top_k: int = 10,
        *,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """
        Main retrieval method.

        Parameters
        ----------
        query   : ProcessedQuery from QueryProcessor
        top_k   : number of results to return after fusion
        filters : optional metadata filters applied post-retrieval
                  e.g. {"language": "en", "site": "arxiv.org"}
        """

        # --- BM25 candidates ---
        bm25_raw = self.bm25.search(
            query.expanded,
            top_k=self.candidate_k,
        )
        logger.debug(f"BM25 returned {len(bm25_raw)} candidates")

        # --- Dense candidates ---
        query_vec = self.embedder.encode_query(query.normalised)
        dense_raw = self.vector_store.search(query_vec, top_k=self.candidate_k)
        logger.debug(f"Dense returned {len(dense_raw)} candidates")

        # --- RRF fusion ---
        fused = self._rrf_fuse(bm25_raw, dense_raw)

        # --- Apply must-include / exclude / site filters ---
        fused = self._apply_filters(fused, query, filters)

        # --- PageRank boost ---
        if self.link_graph and self.pagerank_alpha > 0:
            fused = self._pagerank_boost(fused)

        # Sort and return top_k
        fused.sort(key=lambda r: r.final_score, reverse=True)
        return fused[:top_k]

    # ------------------------------------------------------------------
    # RRF Fusion
    # ------------------------------------------------------------------

    def _rrf_fuse(
        self,
        bm25_hits: list[tuple[int, float, dict]],
        dense_hits: list[tuple[int, float, dict]],
    ) -> list[SearchResult]:
        scores: dict[int, float]        = {}
        meta_map: dict[int, dict]       = {}
        bm25_rank_map: dict[int, int]   = {}
        dense_rank_map: dict[int, int]  = {}

        # BM25 contribution
        for rank, (doc_id, _score, meta) in enumerate(bm25_hits):
            rrf = self.bm25_weight / (_RRF_K + rank + 1)
            scores[doc_id]        = scores.get(doc_id, 0.0) + rrf
            meta_map[doc_id]      = meta
            bm25_rank_map[doc_id] = rank + 1

        # Dense contribution
        for rank, (doc_id, _score, meta) in enumerate(dense_hits):
            rrf = self.dense_weight / (_RRF_K + rank + 1)
            scores[doc_id]         = scores.get(doc_id, 0.0) + rrf
            meta_map.setdefault(doc_id, meta)
            dense_rank_map[doc_id] = rank + 1

        results = []
        for doc_id, rrf_score in scores.items():
            meta = meta_map.get(doc_id, {})
            results.append(SearchResult(
                doc_id=doc_id,
                url=meta.get("url", ""),
                title=meta.get("title", ""),
                score=rrf_score,
                rrf_score=rrf_score,
                bm25_rank=bm25_rank_map.get(doc_id),
                dense_rank=dense_rank_map.get(doc_id),
                metadata=meta,
            ))

        return results

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _apply_filters(
        self,
        results: list[SearchResult],
        query: ProcessedQuery,
        extra_filters: dict | None,
    ) -> list[SearchResult]:
        ops = query.operators
        filtered = []

        for r in results:
            url = r.url.lower()

            # site: filter
            if ops.site and ops.site not in url:
                continue

            # exclude terms
            if any(term.lower() in url or term.lower() in r.title.lower()
                   for term in ops.exclude_terms):
                continue

            # must include
            if ops.must_include and not all(
                term.lower() in (r.title.lower() + " " + url)
                for term in ops.must_include
            ):
                continue

            # extra metadata filters
            if extra_filters:
                skip = False
                for key, val in extra_filters.items():
                    if r.metadata.get(key) != val:
                        skip = True
                        break
                if skip:
                    continue

            filtered.append(r)

        return filtered

    # ------------------------------------------------------------------
    # PageRank Boost
    # ------------------------------------------------------------------

    def _pagerank_boost(self, results: list[SearchResult]) -> list[SearchResult]:
        for r in results:
            pr = self.link_graph.log_score(r.url) if self.link_graph else 0.0
            r.pagerank  = pr
            r.rrf_score = r.rrf_score * (1 + self.pagerank_alpha * pr)
            r.score     = r.rrf_score
        return results
