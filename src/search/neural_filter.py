"""
Neural relevance filter.

Used as a post-retrieval gate: given (query, passage) pairs it
applies a lightweight bi-encoder threshold to drop results that
are semantically far from the query — even if they ranked high
by BM25 or RRF (keyword collision artefacts).

Also provides a standalone batch scorer for offline quality audits.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from loguru import logger

from indexer.embedder import Embedder
from .hybrid_retriever import SearchResult


class NeuralFilter:
    """
    Drops SearchResults whose semantic similarity to the query
    is below a configurable threshold.

    This is cheaper than full reranking (no cross-attention) but
    removes obvious semantic mismatches before the reranker is invoked.

    Parameters
    ----------
    embedder        : shared Embedder instance (bi-encoder)
    threshold       : minimum cosine similarity to pass (0 – 1)
    text_map        : doc_id → passage text for similarity computation;
                      if None, falls back to title + snippet
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        threshold: float = 0.25,
    ):
        self.embedder  = embedder
        self.threshold = threshold

    # ------------------------------------------------------------------

    def filter(
        self,
        query: str,
        results: list[SearchResult],
        *,
        text_map: dict[int, str] | None = None,
    ) -> list[SearchResult]:
        """
        Return only results with cosine similarity ≥ threshold.

        Also sets result.metadata["neural_sim"] for downstream use.
        """
        if not results:
            return []

        query_vec = self.embedder.encode_query(query)

        # Collect texts
        passages = []
        for r in results:
            if text_map and r.doc_id in text_map:
                passages.append(text_map[r.doc_id][:512])
            else:
                passages.append(f"{r.title} {r.snippet}"[:512])

        # Batch encode
        passage_vecs = self.embedder.encode_passages(passages)

        # Cosine similarity (vectors are L2-normalised → dot product = cosine)
        sims: np.ndarray = passage_vecs @ query_vec

        passed = []
        for result, sim in zip(results, sims):
            result.metadata["neural_sim"] = float(sim)
            if sim >= self.threshold:
                passed.append(result)
            else:
                logger.debug(
                    f"Neural filter dropped doc_id={result.doc_id} "
                    f"sim={sim:.3f} < {self.threshold}"
                )

        logger.debug(
            f"Neural filter: {len(results)} in → {len(passed)} out "
            f"(dropped {len(results) - len(passed)})"
        )
        return passed

    # ------------------------------------------------------------------
    # Batch audit (offline use)
    # ------------------------------------------------------------------

    def score_pairs(
        self,
        queries: list[str],
        passages: list[str],
    ) -> np.ndarray:
        """
        Score N (query, passage) pairs and return cosine similarities.

        queries and passages must have the same length.
        Returns float32 array of shape (N,).
        """
        assert len(queries) == len(passages)
        q_vecs = np.array([self.embedder.encode_query(q) for q in queries])
        p_vecs = self.embedder.encode_passages(passages)
        return np.einsum("ij,ij->i", q_vecs, p_vecs).astype(np.float32)

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold
        logger.info(f"NeuralFilter threshold set to {threshold}")
