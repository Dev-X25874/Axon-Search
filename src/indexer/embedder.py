"""
Dense text embedder.

Wraps sentence-transformers to produce L2-normalised float32 embeddings.

Features
--------
- Configurable model (default: BAAI/bge-large-en-v1.5 — MTEB SOTA open)
- Automatic batching with progress bar
- Optional LRU in-process cache for repeated queries
- Instruction prefixes for asymmetric retrieval (query vs passage)
  following the BGE / E5 convention
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Sequence

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Instruction prefixes (BGE convention)
# ---------------------------------------------------------------------------

QUERY_PREFIX   = "Represent this sentence for searching relevant passages: "
PASSAGE_PREFIX = ""   # BGE passages don't need a prefix


class Embedder:
    """
    Wraps a SentenceTransformer model.

    Parameters
    ----------
    model_name   : HuggingFace model id or local path
    device       : "cpu", "cuda", "mps" — auto-detected if None
    batch_size   : inference batch size
    max_length   : token truncation length
    normalize    : L2-normalize output vectors (required for dot-product ANN)
    cache_size   : number of query embeddings to cache in memory (0 = disabled)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        *,
        device: str | None = None,
        batch_size: int = 64,
        max_length: int = 512,
        normalize: bool = True,
        cache_size: int = 4096,
    ):
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading embedder: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_length

        self.batch_size  = batch_size
        self.normalize   = normalize
        self.dim: int    = self.model.get_sentence_embedding_dimension()
        self._cache_size = cache_size

        logger.info(f"Embedder ready | dim={self.dim} device={device}")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: str | Sequence[str],
        *,
        is_query: bool = False,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode one or more texts into float32 vectors.

        Parameters
        ----------
        texts       : single string or list of strings
        is_query    : if True, prepend QUERY_PREFIX for asymmetric retrieval
        show_progress : tqdm progress bar (useful for large corpora)

        Returns
        -------
        np.ndarray of shape (N, dim) or (dim,) for a single string
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        prefix = QUERY_PREFIX if is_query else PASSAGE_PREFIX
        prefixed = [prefix + t for t in texts]

        vecs: np.ndarray = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return vecs[0] if single else vecs

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single search query (with instruction prefix)."""
        return self._cached_query(query)

    def encode_passages(
        self,
        texts: list[str],
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a batch of document passages."""
        return self.encode(texts, is_query=False, show_progress=show_progress)

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _cached_query(self, query: str) -> np.ndarray:
        if self._cache_size <= 0:
            return self.encode(query, is_query=True)
        key = hashlib.md5(query.encode()).hexdigest()
        if not hasattr(self, "_query_cache"):
            self._query_cache: dict[str, np.ndarray] = {}
        if key not in self._query_cache:
            if len(self._query_cache) >= self._cache_size:
                # Evict oldest key (insertion-order dict)
                self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[key] = self.encode(query, is_query=True)
        return self._query_cache[key]

    # ------------------------------------------------------------------
    # Similarity utilities
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Scalar cosine similarity for two L2-normalised vectors."""
        return float(np.dot(a, b))

    @staticmethod
    def batch_cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """query (dim,) × matrix (N, dim) → scores (N,)"""
        return matrix @ query
