"""
Shared pytest fixtures.

`pythonpath = ["src"]` in pyproject.toml puts src/ on sys.path so
tests can `import indexer.bm25`, `import search.query_processor`, etc.
directly, matching how the app itself imports internally.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_docs() -> list[str]:
    return [
        "Flash attention reduces memory usage in transformer training.",
        "The quick brown fox jumps over the lazy dog.",
        "Attention mechanisms allow transformers to weigh input tokens.",
        "Cats and dogs are common household pets.",
        "Memory-efficient attention enables longer context windows.",
    ]
