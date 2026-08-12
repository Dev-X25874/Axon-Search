"""
Central application settings.

Everything here is overridable via environment variables or a .env
file (see .env.example), using pydantic-settings. This replaces the
module-level constants that used to be hardcoded in api/server.py —
those constants silently ignored EMBED_MODEL, RERANK_MODEL, etc. even
though the README documented them as configurable.

Usage
-----
from config import get_settings
settings = get_settings()
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Index storage ────────────────────────────────────────────────
    index_dir: Path = Path("./data/index")

    # ── Embedding model (bi-encoder) ────────────────────────────────
    embed_model: str = "BAAI/bge-large-en-v1.5"
    embed_dim: int = 1024
    embed_batch_size: int = 64
    embed_max_length: int = 512
    embed_device: str = "cpu"

    # ── Reranker model (cross-encoder) ──────────────────────────────
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    rerank_batch_size: int = 32
    rerank_max_length: int = 512
    rerank_device: str = "cpu"

    # ── Vector index ─────────────────────────────────────────────────
    vector_index_type: str = "hnsw"  # flat | hnsw | ivf
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64

    # ── Crawler ──────────────────────────────────────────────────────
    crawl_delay: float = 1.0
    crawl_concurrency: int = 32
    crawl_max_depth: int = 3
    crawl_max_pages: int = 10_000
    crawl_request_timeout: float = 15.0
    crawl_max_response_mb: int = 5

    # ── Quality & dedup gates ───────────────────────────────────────
    quality_threshold: float = 0.35
    min_words: int = 80
    max_link_density: float = 0.4
    dedup_threshold: float = 0.8
    dedup_num_perm: int = 128

    # ── Retrieval ────────────────────────────────────────────────────
    neural_filter_threshold: float = 0.25
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    pagerank_alpha: float = 0.1

    # ── API server ───────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_reload: bool = False
    log_level: str = "INFO"

    # CORS: comma-separated origins, e.g. "https://app.example.com,https://admin.example.com"
    # Defaults to no cross-origin access rather than "*" — tighten/loosen per deployment.
    cors_allow_origins: list[str] = []

    # ── Auth ─────────────────────────────────────────────────────────
    # Comma-separated API keys. Empty (default) = auth disabled, matching
    # the CORS pattern above — opt in per deployment via API_KEYS.
    # Callers send the key in the `X-API-Key` header. /health and /metrics
    # stay open even when auth is enabled, so orchestrators can probe them.
    api_keys: list[str] = []

    # ── Rate limiting ────────────────────────────────────────────────
    # Requests allowed per client (per API key if present, else per IP)
    # in a rolling 60s window. 0 (default) disables rate limiting.
    rate_limit_per_minute: int = 0

    # ── Search result cache ─────────────────────────────────────────
    # Short-TTL in-memory cache for identical /search requests. This is
    # a process-local cache with no invalidation on index writes, so keep
    # the TTL short relative to how often the index changes. 0 disables it.
    search_cache_ttl_s: float = 30.0
    search_cache_size: int = 512

    # ── Autocomplete ─────────────────────────────────────────────────
    suggest_max_results: int = 10


@lru_cache
def get_settings() -> Settings:
    """Settings are read once and cached — override via env vars or .env before first use."""
    return Settings()
