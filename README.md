# Axon Search

A production-grade hybrid semantic search engine.  
Combines BM25 sparse retrieval with FAISS dense retrieval, fused via Reciprocal Rank Fusion, then re-ranked with a cross-encoder.

```
query
  └─► QueryProcessor (normalise, expand, parse operators)
        └─► HybridRetriever
              ├─► BM25Index          (sparse, keyword precision)
              └─► VectorStore/FAISS  (dense, semantic recall)
                    └─► RRF Fusion + PageRank boost
                          └─► NeuralFilter  (bi-encoder gate)
                                └─► CrossEncoderReranker
                                      └─► ranked results + snippets
```

---

## Project layout

```
axon-search/
├── pyproject.toml
├── .env.example
├── Makefile
├── Dockerfile
├── docker-compose.yml
└── src/
    ├── crawler/
    │   ├── async_crawler.py       # async BFS + per-domain rate limiting
    │   ├── content_extractor.py   # HTML → clean text (trafilatura cascade)
    │   ├── link_graph.py          # PageRank authority scoring
    │   └── robots.py              # robots.txt async cache
    ├── indexer/
    │   ├── pipeline.py            # streaming crawl→embed→index pipeline
    │   ├── embedder.py            # BGE dual-encoder (sentence-transformers)
    │   ├── bm25.py                # incremental BM25Okapi index
    │   └── vector_store.py        # FAISS (flat / HNSW / IVF)
    ├── search/
    │   ├── query_processor.py     # operator parsing, intent, WordNet expansion
    │   ├── hybrid_retriever.py    # RRF fusion + PageRank boost
    │   ├── reranker.py            # ms-marco cross-encoder reranker
    │   └── neural_filter.py       # bi-encoder semantic gate
    ├── api/
    │   ├── server.py              # FastAPI app factory + lifespan DI
    │   ├── schemas.py             # Pydantic v2 request/response models
    │   └── routes/
    │       ├── search.py          # POST /search
    │       └── index.py           # POST /index/url|batch, GET /index/stats
    └── utils/
        ├── text_cleaner.py        # Unicode normalise, chunk, sentence-split
        ├── dedup.py               # MinHash LSH near-dedup (datasketch)
        └── quality_scorer.py      # heuristic quality gate (TTR, link density…)
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/yourname/axon-search
cd axon-search
pip install -e ".[dev]"
playwright install chromium        # only needed if you enable JS rendering
```

### 2. Configure

```bash
cp .env.example .env
# edit .env — set EMBED_MODEL, RERANK_MODEL, INDEX_DIR, etc.
```

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

## Docker

```bash
docker compose up --build
```

The compose file starts the API on port 8000 and mounts `./data` for index persistence.

---

## API reference

### `POST /search`

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | Raw search query |
| `top_k` | int | 10 | Results to return (max 100) |
| `rerank` | bool | true | Apply cross-encoder reranker |
| `neural_filter` | bool | true | Apply bi-encoder semantic gate |
| `filters` | object | {} | Metadata equality filters |

**Supported query operators**

| Operator | Example | Effect |
|---|---|---|
| `site:` | `site:arxiv.org transformers` | Restrict to domain |
| `filetype:` | `filetype:pdf` | Filter by file type |
| `-term` | `attention -vision` | Exclude term |
| `+"phrase"` | `+"flash attention"` | Must include phrase |
| `after:` | `after:2024-01-01` | Published after date |
| `before:` | `before:2025-01-01` | Published before date |

### `POST /index/url`

| Field | Type | Default | Description |
|---|---|---|---|
| `url` | string | required | URL to crawl and index |
| `depth` | int | 0 | Crawl depth from this URL |

### `POST /index/batch`

| Field | Type | Default | Description |
|---|---|---|---|
| `seeds` | list[url] | required | Seed URLs |
| `max_pages` | int | 1000 | Page cap |
| `max_depth` | int | 3 | Crawl depth |
| `concurrency` | int | 16 | Concurrent fetches |

### `GET /index/stats`

Returns `{ bm25_docs, vector_docs, graph_nodes, graph_edges }`.

---

## Configuration

All settings can be overridden via environment variables (see `.env.example`).

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
- **GPU reranking** — set `device=cuda` in `CrossEncoderReranker.__init__`
- **Distributed crawl** — replace `asyncio.Queue` in `pipeline.py` with a Redis stream
- **Passage-level indexing** — pipe `TextCleaner.chunk()` output into the embedder instead of full-page text
