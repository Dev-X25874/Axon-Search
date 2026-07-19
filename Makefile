.PHONY: install dev server index-url index-batch search lint fmt typecheck clean

# ── Setup ──────────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"
	playwright install chromium

# ── Dev server ─────────────────────────────────────────────────────────────

server:
	uvicorn api.server:create_app \
		--factory \
		--host 0.0.0.0 \
		--port 8000 \
		--reload \
		--log-level info \
		--app-dir src

# ── Quick index / search (requires server running) ─────────────────────────

URL ?= https://huggingface.co/papers
index-url:
	curl -s -X POST http://localhost:8000/index/url \
		-H "Content-Type: application/json" \
		-d '{"url": "$(URL)", "depth": 1}' | python -m json.tool

SEEDS ?= ["https://arxiv.org","https://huggingface.co/papers"]
index-batch:
	curl -s -X POST http://localhost:8000/index/batch \
		-H "Content-Type: application/json" \
		-d '{"seeds": $(SEEDS), "max_pages": 5000, "max_depth": 3}' \
		| python -m json.tool

QUERY ?= attention mechanism transformer
search:
	curl -s -X POST http://localhost:8000/search \
		-H "Content-Type: application/json" \
		-d '{"query": "$(QUERY)", "top_k": 10, "rerank": true}' \
		| python -m json.tool

stats:
	curl -s http://localhost:8000/index/stats | python -m json.tool

health:
	curl -s http://localhost:8000/health | python -m json.tool

# ── Code quality ───────────────────────────────────────────────────────────

lint:
	ruff check src/

fmt:
	ruff format src/

typecheck:
	mypy src/ --ignore-missing-imports --no-strict-optional

# ── Docker ─────────────────────────────────────────────────────────────────

docker-build:
	docker build -t axon-search:latest .

docker-up:
	docker compose up --build

docker-down:
	docker compose down

# ── Cleanup ────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .ruff_cache .mypy_cache .pytest_cache
	rm -rf data/index/*.pkl data/index/vectors/

clean-all: clean
	rm -rf data/
