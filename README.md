# Axon Search

A hybrid semantic search engine.
Combines BM25 sparse retrieval with FAISS dense retrieval, fused via Reciprocal Rank Fusion, then re-ranked with a cross-encoder.

```
query
  └─► QueryProcessor (normalise, expand, parse operators)
        └─► HybridRetriever
              ├─► BM25Index (sparse, keyword precision)
              └─► VectorStore/FAISS (dense, semantic recall)
        └─► RRF Fusion + PageRank boost
        └─► NeuralFilter (bi-encoder gate)
        └─► CrossEncoderReranker
              └─► ranked results + snippets
```

Also ships: soft document deletion, `/search/suggest` autocomplete,
result-page pagination, a short-TTL search cache, optional API-key auth,
optional rate limiting, and a Prometheus-format `/metrics` endpoint —
see [Additional features](#additional-features) below.

---

## Project layout

```
axon-search/
├── pyproject.toml
├── .env.example
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── src/
│   ├── config.py                 # pydantic-settings — reads .env, single source of truth
│   ├── crawler/
│   │   ├── async_crawler.py      # async BFS + per-domain rate limiting + SSRF guard
│   │   ├── content_extractor.py  # HTML → clean text (trafilatura cascade)
│   │   ├── link_graph.py         # PageRank authority scoring
│   │   └── robots.py             # robots.txt async cache
│   ├── indexer/
│   │   ├── pipeline.py           # streaming crawl→embed→index pipeline
│   │   ├── embedder.py           # BGE dual-encoder (sentence-transformers)
│   │   ├── bm25.py               # incremental BM25Okapi index
│   │   └── vector_store.py       # FAISS (flat / HNSW / IVF)
│   ├── search/
│   │   ├── query_processor.py    # operator parsing, intent, WordNet expansion
│   │   ├── hybrid_retriever.py   # RRF fusion + PageRank boost
│   │   ├── reranker.py           # ms-marco cross-encoder reranker
│   │   └── neural_filter.py      # bi-encoder semantic gate
│   ├── api/
│   │   ├── server.py             # FastAPI app factory + lifespan DI
│   │   ├── schemas.py            # Pydantic v2 request/response models
│   │   ├── auth.py               # optional X-API-Key middleware
│   │   ├── rate_limit.py         # optional per-client rate limiting
│   │   ├── metrics.py            # /metrics — Prometheus-format counters
│   │   └── routes/
│   │       ├── search.py         # POST /search, GET /search/suggest
│   │       └── index.py          # POST /index/url|batch, DELETE /index/url,
│   │                             # GET /index/stats, GET /index/jobs[/{id}]
│   └── utils/
│       ├── text_cleaner.py       # Unicode normalise, chunk, sentence-split
│       ├── dedup.py              # MinHash LSH near-dedup (datasketch)
│       ├── quality_scorer.py     # heuristic quality gate (TTR, link density…)
│       └── cache.py              # TTL+LRU cache backing the search cache
└── tests/
    ├── test_bm25.py
    ├── test_vector_store.py
    ├── test_query_processor.py
    ├── test_dedup.py
    ├── test_link_graph.py
    ├── test_hybrid_retriever.py
    ├── test_async_crawler.py
    └── test_api.py
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/Dev-X25874/axon-search
cd axon-search
pip install -e ".[dev]"
playwright install chromium        # only needed if you enable JS rendering
```

### 2. Configure

```bash
cp .env.example .env
# edit .env — set EMBED_MODEL, RERANK_MODEL, INDEX_DIR, etc.
```

All values in `.env.example` are read by `src/config.py` (pydantic-settings) at
startup — nothing is hardcoded, so anything you set here actually takes effect.

### 3. Start the server

```bash
python src/api/server.py
# or
uvicorn api.server:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

### 4. Index some pages

```bash
# single URL
curl -X POST http://localhost:8000/index/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://arxiv.org/abs/2005.14165", "depth": 1}'

# batch crawl (background job)
curl -X POST http://localhost:8000/index/batch \
  -H "Content-Type: application/json" \
  -d '{
    "seeds": ["https://arxiv.org", "https://huggingface.co/papers"],
    "max_pages": 10000,
    "max_depth": 3,
    "concurrency": 32
  }'

# check job status
curl http://localhost:8000/index/jobs/<job_id>
```

### 5. Search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "flash attention memory efficient transformers",
    "top_k": 10,
    "rerank": true,
    "neural_filter": true
  }'
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest
```

90 tests covering `BM25Index` (incl. soft-delete and `suggest()`), `VectorStore`
(incl. soft-delete), `QueryProcessor`, `DedupFilter`, `LinkGraph`/PageRank,
`HybridRetriever` (RRF fusion, filters, PageRank boost — via fakes, no real ML
models required), `AsyncCrawler`'s URL allow-listing and SSRF guard, and the
API layer (`test_api.py` — search, suggest, delete, jobs, stats, `/metrics`,
auth, and rate-limiting, all via fakes).

`HybridRetriever`'s test module imports through the real `sentence-transformers` /
`torch` / `faiss` stack (same as the app does), so a full `pip install -e ".[dev]"`
is needed for that file specifically; the rest run with just `pytest` + the
lighter-weight deps. `test_api.py` also avoids the real `sentence_transformers`
import when it isn't installed by falling back to a stub package under
`tests/_stubs/` (see `tests/conftest.py`) — a real install always takes priority.

## CI

`.github/workflows/tests.yml` runs `ruff check` and the full `pytest` suite
(with coverage) on push/PR across Python 3.10–3.12.

---

## Docker

```bash
docker compose up --build
```

The compose file starts the API on port 8000 and mounts `./data` for index persistence.

---

---

## Additional features

Beyond the core hybrid-search pipeline, the API also ships:

- **Document deletion** — `DELETE /index/url` soft-deletes a document; it
  disappears from `/search` immediately.
- **Autocomplete** — `GET /search/suggest?q=` completes a prefix against
  indexed vocabulary.
- **Pagination** — `offset` on `POST /search` pages through results without
  re-scoring from rank 0.
- **Search caching** — short-TTL cache for repeated identical queries
  (`SEARCH_CACHE_TTL_S`), auto-invalidated on delete/reindex.
- **Optional API-key auth** and **optional rate limiting**, both opt-in via
  env vars (see [Security notes](#security-notes)).
- **`GET /metrics`** — Prometheus-format request counters and latency.
- **`GET /index/jobs`** — list/filter background crawl jobs, not just look
  one up by id.

Full parameters for each are in [API reference](#api-reference) below.

---

## API reference

### `POST /search`

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | Raw search query |
| `top_k` | int | 10 | Results to return (max 100) |
| `offset` | int | 0 | Results to skip before `top_k` (pagination, max 10,000) |
| `rerank` | bool | true | Apply cross-encoder reranker |
| `neural_filter` | bool | true | Apply bi-encoder semantic gate |
| `filters` | object | {} | Metadata equality filters |

Response includes a `cached: bool` field — `true` when served from the
short-TTL search cache (`SEARCH_CACHE_TTL_S`) instead of re-running
retrieval/reranking. The cache is cleared automatically whenever
`/index/url` (DELETE) removes a document or a batch job finishes indexing.

**Supported query operators**

| Operator | Example | Effect |
|---|---|---|
| `site:` | `site:arxiv.org transformers` | Restrict to domain |
| `filetype:` | `filetype:pdf` | Filter by file type |
| `-term` | `attention -vision` | Exclude term |
| `+"phrase"` | `+"flash attention"` | Must include phrase |
| `after:` | `after:2024-01-01` | Published after date |
| `before:` | `before:2025-01-01` | Published before date |

### `GET /search/suggest`

Autocomplete over indexed vocabulary, ranked by document frequency.

| Param | Type | Default | Description |
|---|---|---|---|
| `q` | string | required | Prefix to complete |
| `limit` | int | 10 | Max suggestions (max 50) |

```bash
curl "http://localhost:8000/search/suggest?q=atten&limit=5"
```

### `POST /index/url`

| Field | Type | Default | Description |
|---|---|---|---|
| `url` | string | required | URL to crawl and index |
| `depth` | int | 0 | Crawl depth from this URL |

### `DELETE /index/url`

Soft-deletes a previously indexed URL — it's immediately excluded from
`/search` results (and the search cache is cleared), without a full
index rebuild. The row stays on disk as a tombstone until you rebuild
the index from scratch.

```bash
curl -X DELETE "http://localhost:8000/index/url?url=https://arxiv.org/abs/2005.14165"
```

Returns `404` if the URL was never indexed.

### `POST /index/batch`

| Field | Type | Default | Description |
|---|---|---|---|
| `seeds` | list[url] | required | Seed URLs |
| `max_pages` | int | 1000 | Page cap |
| `max_depth` | int | 3 | Crawl depth |
| `concurrency` | int | 16 | Concurrent fetches |

### `GET /index/jobs`

Lists background jobs, most recent first.

| Param | Type | Default | Description |
|---|---|---|---|
| `limit` | int | 50 | Max jobs returned (max 500) |
| `status` | string | *(unset)* | Filter: `pending` / `running` / `done` / `failed` |

### `GET /index/jobs/{job_id}`

Status of a single job (also returned synchronously by `/index/url` and
`/index/batch`).

### `GET /index/stats`

Returns `{ bm25_docs, vector_docs, graph_nodes, graph_edges, bm25_deleted, vector_deleted }`.
`bm25_docs`/`vector_docs` already exclude soft-deleted documents;
`*_deleted` is the tombstone count.

### `GET /metrics`

Prometheus text-format request counters and cumulative latency per
route, plus process uptime. Always reachable, even when
`API_KEYS`/rate limiting are enabled, so scrapers don't need a key.

### `GET /health`

Liveness probe — `{ "status": "ok" }`. Always reachable, same as `/metrics`.

---

## Configuration

All settings live in `src/config.py` and can be overridden via environment
variables or a `.env` file (see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `BAAI/bge-large-en-v1.5` | Bi-encoder model |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-12-v2` | Cross-encoder model |
| `EMBED_DIM` | `1024` | Embedding dimension |
| `VECTOR_INDEX_TYPE` | `hnsw` | `flat` / `hnsw` / `ivf` |
| `INDEX_DIR` | `./data/index` | Where indices are persisted |
| `CRAWL_DELAY` | `1.0` | Default per-domain crawl delay (s) |
| `QUALITY_THRESHOLD` | `0.35` | Minimum quality score to index |
| `DEDUP_THRESHOLD` | `0.8` | Jaccard threshold for near-dedup |
| `NEURAL_FILTER_THRESHOLD` | `0.25` | Minimum bi-encoder similarity |
| `CORS_ALLOW_ORIGINS` | *(unset)* | Comma-separated allowed origins. Unset = CORS disabled entirely (no cross-origin access), not `*`. |
| `API_KEYS` | *(unset)* | Comma-separated API keys. Unset = auth disabled. Clients send `X-API-Key`. |
| `RATE_LIMIT_PER_MINUTE` | `0` | Max requests per client (per key, else per IP) per rolling 60s. `0` = disabled. |
| `SEARCH_CACHE_TTL_S` | `30.0` | TTL for cached `/search` responses. `0` disables caching. |
| `SEARCH_CACHE_SIZE` | `512` | Max cached search responses (LRU eviction). |
| `SUGGEST_MAX_RESULTS` | `10` | Default `limit` for `/search/suggest`. |

---

## Security notes

- **CORS is closed by default.** Cross-origin requests are rejected unless
  `CORS_ALLOW_ORIGINS` is explicitly set. Only `GET`/`POST` and
  `Content-Type`/`Authorization` headers are allowed even when it is set.
- **The crawler verifies TLS certificates** on every request and will not
  silently accept invalid/self-signed certs.
- **The crawler blocks SSRF targets.** Seed URLs and discovered outlinks that
  resolve to loopback, private (RFC1918), link-local (including the
  `169.254.169.254` cloud metadata endpoint), or otherwise reserved addresses
  are rejected before any request is made.
- **Authentication is opt-in via `API_KEYS`.** Unset (the dev default) means
  no auth, same as before — set it before exposing `/index/url` and
  `/index/batch` (which trigger outbound crawls) beyond localhost. Clients
  authenticate with a static `X-API-Key` header; `/health` and `/metrics`
  stay reachable without a key so orchestrators/scrapers keep working.
  This is intentionally simple (static shared keys, no rotation/scoping) —
  put a real auth layer (OAuth2, reverse-proxy SSO, etc.) in front for
  multi-tenant or production use.
- **Rate limiting is opt-in via `RATE_LIMIT_PER_MINUTE`.** The limiter is
  in-memory and per-process — fine for `API_WORKERS=1`, but counts aren't
  shared across workers/replicas, so it under-limits if you scale out.
  Use a Redis-backed limiter at that point.
- **The `/index/jobs` registry is in-memory** and — unless `API_KEYS` is
  set — unauthenticated: anyone who can reach the API can read job status.
  Fine for local/dev use; replace with a real store + access control before
  running multi-tenant.
- **The search cache has no cross-request access control.** If you enable
  `API_KEYS`, results are still cached and served across different callers'
  identical queries (no per-key partitioning). Set `SEARCH_CACHE_TTL_S=0`
  if that's not acceptable for your deployment.

---

## Models used

| Component | Model | Why |
|---|---|---|
| Bi-encoder | `BAAI/bge-large-en-v1.5` | Top MTEB open model, asymmetric retrieval |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-12-v2` | Fast, strong MS MARCO reranker |
| BM25 | `rank_bm25` (BM25Okapi) | Best BM25 variant for web text |

Swap either model by setting the env vars — the code is model-agnostic.

---

## Extending

- **Add a new ranker signal** — subclass `HybridRetriever`, override `_rrf_fuse`
- **Swap the vector index** — set `VECTOR_INDEX_TYPE=ivf` for billion-scale
- **GPU reranking** — set `RERANK_DEVICE=cuda` (see `.env.example`)
- **Distributed crawl** — replace `asyncio.Queue` in `pipeline.py` with a Redis stream
- **Passage-level indexing** — pipe `TextCleaner.chunk()` output into the embedder instead of full-page text
- **CI** — none is configured yet; add a `.github/workflows/tests.yml` running
  `pip install -e ".[dev]" && pytest` if you want checks on push/PR
