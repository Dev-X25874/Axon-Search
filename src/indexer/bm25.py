"""
BM25 sparse inverted index.

- Wraps rank_bm25 (BM25Okapi) for scoring
- Maintains a forward store of doc metadata (url, title, etc.)
- Supports incremental adds + full re-build
- Pickle-serialisable for persistence
"""

from __future__ import annotations

import pickle
import re
import string
from pathlib import Path
from typing import Any

import nltk
from loguru import logger
from rank_bm25 import BM25Okapi

# Ensure NLTK data is present
for resource in ("punkt", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

_STOP_WORDS = set(stopwords.words("english"))
_STEMMER    = PorterStemmer()
_PUNCT      = str.maketrans("", "", string.punctuation)
_MULTI_WS   = re.compile(r"\s+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords, stem."""
    text  = text.lower().translate(_PUNCT)
    tokens = _MULTI_WS.sub(" ", text).split()
    return [
        _STEMMER.stem(t)
        for t in tokens
        if t not in _STOP_WORDS and len(t) > 1
    ]


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class BM25Index:
    """
    Incremental BM25 index.

    Documents are accumulated in a staging buffer; the BM25Okapi
    object is rebuilt in bulk when _rebuild() is called.
    Documents can also be added one-at-a-time and the index rebuilt
    lazily on next search.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b

        self._tokenized_corpus: list[list[str]] = []
        self._metadata: list[dict[str, Any]]    = []
        self._bm25: BM25Okapi | None            = None
        self._dirty = False

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> int:
        """
        Add a document; returns its integer doc_id.
        The BM25 object will be rebuilt lazily on next search.
        """
        doc_id = len(self._tokenized_corpus)
        self._tokenized_corpus.append(_tokenize(text))
        self._metadata.append(metadata or {})
        self._dirty = True
        return doc_id

    def add_batch(self, texts: list[str], metadata: list[dict] | None = None) -> list[int]:
        ids = []
        meta = metadata or [{}] * len(texts)
        for text, m in zip(texts, meta):
            ids.append(self.add(text, m))
        return ids

    def _rebuild(self) -> None:
        if not self._tokenized_corpus:
            return
        logger.debug(f"BM25 rebuild: {len(self._tokenized_corpus)} docs")
        self._bm25  = BM25Okapi(
            self._tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )
        self._dirty = False

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 100,
    ) -> list[tuple[int, float, dict]]:
        """
        Return top-k results as (doc_id, bm25_score, metadata).

        Scores are raw BM25 values (not normalised).
        """
        if self._dirty or self._bm25 is None:
            self._rebuild()
        if self._bm25 is None:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Partial sort: get top_k indices without full sort
        import numpy as np
        idx = np.argpartition(scores, -min(top_k, len(scores)))[-top_k:]
        idx = idx[np.argsort(scores[idx])[::-1]]

        return [
            (int(i), float(scores[i]), self._metadata[i])
            for i in idx
            if scores[i] > 0
        ]

    def get_metadata(self, doc_id: int) -> dict:
        if 0 <= doc_id < len(self._metadata):
            return self._metadata[doc_id]
        return {}

    def __len__(self) -> int:
        return len(self._tokenized_corpus)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        if self._dirty:
            self._rebuild()
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "corpus":   self._tokenized_corpus,
                    "metadata": self._metadata,
                    "k1": self.k1,
                    "b":  self.b,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info(f"BM25 index saved to {path} ({len(self)} docs)")

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls(k1=data["k1"], b=data["b"])
        idx._tokenized_corpus = data["corpus"]
        idx._metadata         = data["metadata"]
        idx._dirty = True   # will rebuild on next search
        logger.info(f"BM25 index loaded from {path} ({len(idx)} docs)")
        return idx
