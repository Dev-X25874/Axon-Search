"""
Pydantic v2 request / response schemas for the Axon Search API.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, HttpUrl, field_validator


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query:      str = Field(..., min_length=1, max_length=1024, description="Search query")
    top_k:      int = Field(10, ge=1, le=100, description="Number of results to return")
    rerank:     bool = Field(True, description="Apply cross-encoder reranking")
    neural_filter: bool = Field(True, description="Apply neural relevance filter")
    filters:    dict[str, Any] = Field(default_factory=dict, description="Metadata filters")

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class ResultItem(BaseModel):
    url:            str
    title:          str
    snippet:        str
    score:          float
    rrf_score:      float
    rerank_score:   Optional[float] = None
    pagerank:       float           = 0.0
    bm25_rank:      Optional[int]   = None
    dense_rank:     Optional[int]   = None
    neural_sim:     Optional[float] = None


class SearchResponse(BaseModel):
    query:          str
    intent:         str
    expanded_query: str
    results:        list[ResultItem]
    total_retrieved: int
    elapsed_ms:     float


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

class IndexURLRequest(BaseModel):
    url:  HttpUrl = Field(..., description="URL to crawl and index")
    depth: int   = Field(0, ge=0, le=5, description="Crawl depth from this URL")


class IndexBatchRequest(BaseModel):
    seeds:      list[HttpUrl] = Field(..., min_length=1, max_length=500)
    max_pages:  int           = Field(1000, ge=1, le=100_000)
    max_depth:  int           = Field(3, ge=0, le=10)
    concurrency: int          = Field(16, ge=1, le=64)


class IndexJobStatus(BaseModel):
    job_id:    str
    status:    str   # pending | running | done | failed
    crawled:   int   = 0
    indexed:   int   = 0
    elapsed_s: float = 0.0
    error:     Optional[str] = None


class IndexStatsResponse(BaseModel):
    bm25_docs:    int
    vector_docs:  int
    graph_nodes:  int
    graph_edges:  int
