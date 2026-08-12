"""
Shared pytest fixtures.

`pythonpath = ["src"]` in pyproject.toml puts src/ on sys.path so
tests can `import indexer.bm25`, `import search.query_processor`, etc.
directly, matching how the app itself imports internally.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# `sentence_transformers` (used by indexer/embedder.py and search/reranker.py)
# pulls in torch and real model downloads — unnecessary for tests that only
# exercise route/index logic against fakes (e.g. tests/test_api.py). If it
# isn't installed, fall back to a lightweight stub package so those modules
# still import cleanly. A real install always takes priority.
if importlib.util.find_spec("sentence_transformers") is None:
    sys.path.insert(0, str(Path(__file__).parent / "_stubs"))


@pytest.fixture
def sample_docs() -> list[str]:
    return [
        "Flash attention reduces memory usage in transformer training.",
        "The quick brown fox jumps over the lazy dog.",
        "Attention mechanisms allow transformers to weigh input tokens.",
        "Cats and dogs are common household pets.",
        "Memory-efficient attention enables longer context windows.",
    ]
