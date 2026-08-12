"""
Minimal stub of `sentence_transformers` for tests that don't need real
embeddings/reranking (e.g. API route tests using fakes). Avoids pulling
in torch/transformers just to satisfy module-level imports in
indexer/embedder.py and search/reranker.py.

Only used when tests/_stubs is placed on sys.path ahead of site-packages
(see tests/conftest.py) — the real package, if installed, always wins.
"""

from __future__ import annotations


class SentenceTransformer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "sentence_transformers is stubbed out in this test environment; "
            "this fake should never be instantiated. Use fakes in app.state instead."
        )
