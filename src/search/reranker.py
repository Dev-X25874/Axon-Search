"""
Cross-encoder reranker.

Takes the top-K candidates from HybridRetriever and scores each
(query, passage) pair with a full cross-attention transformer.
This is expensive but extremely accurate — only run on the top-K
(typically 20–50) candidates, not the full index.

Default model: cross-encoder/ms-marco-MiniLM-L-12-v2
  - Fast (12-layer MiniLM)
  - Trained on MS MARCO passage ranking
  - Strong generalisation to web search
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from loguru import logger
from sentence_transformers.cross_encoder import CrossEncoder

from .hybrid_retriever import SearchResult
from indexer.bm25 import BM25Index


class CrossEncoderReranker:
    """
    Reranks a list of SearchResult objects using a cross-encoder.

    The reranker needs the passage text for each candidate.
    This is retrieved from the BM25 index's metadata store
    (which holds the original text or a snippet).

    Parameters
    ----------
    model_name : HuggingFace cross-encoder model id
    device     : "cpu" / "cuda" / "mps"
    batch_size : inference batch size (cross-encoders are memory-heavy)
    max_length : token limit for (query, passage) pairs
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        *,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading cross-encoder: {model_name} on {device}")
        self.model      = CrossEncoder(model_name, device=device, max_length=max_length)
        self.batch_size = batch_size
        logger.info("Cross-encoder ready")

    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        *,
        text_map: dict[int, str] | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Rerank results in-place, setting result.rerank_score.

        Parameters
        ----------
        query    : raw or normalised query string
        results  : candidates from HybridRetriever
        text_map : doc_id → passage text.  If None, falls back to
                   result.title + result.snippet.
        top_k    : if set, return only the top_k after reranking

        Returns
        -------
        Sorted list (descending by rerank_score).
        """
        if not results:
            return []

        pairs = []
        for r in results:
            if text_map and r.doc_id in text_map:
                passage = text_map[r.doc_id]
            else:
                passage = f"{r.title}. {r.snippet}".strip()
            pairs.append((query, passage))

        scores = self._batch_score(pairs)

        for result, score in zip(results, scores):
            result.rerank_score = float(score)

        results.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)
        return results[:top_k] if top_k else results

    # ------------------------------------------------------------------

    def _batch_score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch  = pairs[i : i + self.batch_size]
            scores = self.model.predict(batch, show_progress_bar=False)
            all_scores.extend(scores.tolist() if hasattr(scores, "tolist") else scores)
        return np.array(all_scores, dtype=np.float32)

    # ------------------------------------------------------------------
    # Snippet generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_snippet(
        text: str,
        query_terms: list[str],
        *,
        window: int = 30,
        max_length: int = 200,
    ) -> str:
        """
        Extract a passage snippet centred around the best query-term match.

        Returns a plain-text snippet ≤ max_length characters.
        """
        tokens = text.split()
        query_set = {t.lower() for t in query_terms}

        best_start = 0
        best_hits  = 0

        for i in range(len(tokens)):
            chunk  = tokens[i : i + window]
            hits   = sum(1 for t in chunk if t.lower() in query_set)
            if hits > best_hits:
                best_hits  = hits
                best_start = i

        snippet_tokens = tokens[best_start : best_start + window]
        snippet = " ".join(snippet_tokens)

        if best_start > 0:
            snippet = "… " + snippet
        if best_start + window < len(tokens):
            snippet = snippet + " …"

        return snippet[:max_length]
