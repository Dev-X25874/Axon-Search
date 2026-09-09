# axon-search

A hybrid semantic search engine.

Combines BM25 sparse retrieval with FAISS dense retrieval, fused via
Reciprocal Rank Fusion, then re-ranked with a cross-encoder.

## What it does

- Parses raw queries — operator extraction (`site:`, `filetype:`,
  `-term`, `+"phrase"`, `after:`/`before:`), intent classification,
  WordNet synonym expansion.
- Runs BM25Okapi (keyword precision) and FAISS (semantic recall) in
  parallel, fuses results with RRF, then boosts by PageRank authority.
- Gates candidates through a bi-encoder semantic filter before handing
  off to a cross-encoder reranker, so the final ranking is both
  fast and precise.
- Crawls asynchronously with per-domain rate limiting, TLS
  verification, and an SSRF guard that rejects loopback, RFC1918, and
  cloud-metadata targets before any request is made.
- Ships soft document deletion (immediate exclusion from results), a
  short-TTL search cache (auto-invalidated on delete/reindex),
  autocomplete over indexed vocabulary, paginated results, optional
  API-key auth, optional per-client rate limiting, and a
  Prometheus-format `/metrics` endpoint.

## Why

Most search demos bolt dense retrieval onto a keyword index after the
fact. This one treats them as peers: BM25 for exact-match precision,
FAISS for semantic recall, RRF to merge without tuning per-dataset
weights, and a cross-encoder reranker to fix the fusion's blind spots.
The crawler feeds the pipeline directly — no separate ingestion step.

## Models

| Component | Model |
|---|---|
| Bi-encoder | `BAAI/bge-large-en-v1.5` |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| BM25 | `rank_bm25` (BM25Okapi) |

Swap either model by setting `EMBED_MODEL` / `RERANK_MODEL` in `.env`
— the code is model-agnostic.

## Install

```console
$ git clone https://github.com/Dev-X25874/axon-search
$ cd axon-search
$ pip install -e ".[dev]"
$ playwright install chromium   # only if JS rendering is enabled
```

## Configure

```console
$ cp .env.example .env
# edit .env — EMBED_MODEL, RERANK_MODEL, INDEX_DIR, etc.
```

All settings are read by `src/config.py` (pydantic-settings) at
startup. Nothing is hardcoded.

## Run

```console
$ uvicorn api.server:create_app --factory --host 0.0.0.0 --port 8000
```

Or with Docker:

```console
$ docker compose up --build
```

The compose file starts the API on port 8000 and mounts `./data` for
index persistence.

## Usage

Index a page:

```console
$ curl -X POST http://localhost:8000/index/url \
    -H "Content-Type: application/json" \
    -d '{"url": "https://arxiv.org/abs/2005.14165", "depth": 1}'
```

Batch crawl (runs as a background job):

```console
$ curl -X POST http://localhost:8000/index/batch \
    -H "Content-Type: application/json" \
    -d '{
      "seeds": ["https://arxiv.org", "https://huggingface.co/papers"],
      "max_pages": 10000,
      "max_depth": 3,
      "concurrency": 32
    }'
```

Search:

```console
$ curl -X POST http://localhost:8000/search \
    -H "Content-Type: application/json" \
    -d '{
      "query": "flash attention memory efficient transformers",
      "top_k": 10,
      "rerank": true,
      "neural_filter": true
    }'
```

Autocomplete:

```console
$ curl "http://localhost:8000/search/suggest?q=atten&limit=5"
```

## What this isn't

It doesn't federate across multiple index shards and doesn't try to
replace a production search platform. The in-memory rate limiter
under-counts across workers — use a Redis-backed limiter if you scale
out. The search cache has no per-caller partitioning; set
`SEARCH_CACHE_TTL_S=0` if that matters for your deployment.

## Tests

```console
$ pytest
```

90 tests across BM25, vector store, query processor, dedup, PageRank,
hybrid retrieval, the async crawler, and the full API layer — all via
fakes, so no real ML models are required for most of the suite. CI
runs `ruff check` and the full suite (with coverage) on push/PR across
Python 3.10–3.12.
