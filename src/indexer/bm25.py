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

        # Soft-delete tombstones: BM25Okapi has no native removal, and
        # rebuilding FAISS/BM25 on every delete is expensive, so deleted
        # doc_ids are just filtered out of search() results.
        self._deleted: set[int] = set()

        # url -> doc_id, populated from add()'s metadata when present.
        # Lets callers delete/look up a document by URL instead of doc_id.
        self._url_to_id: dict[str, int] = {}

        # Raw (unstemmed) word frequencies, for autocomplete. Kept
        # separate from the stemmed tokens used for BM25 scoring since
        # stemmed prefixes ("attent") make poor suggestions.
        self._vocab_freq: dict[str, int] = {}

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

        meta = metadata or {}
        url = meta.get("url")
        if url:
            self._url_to_id[url] = doc_id

        self._update_vocab(text)
        return doc_id

    def add_batch(self, texts: list[str], metadata: list[dict] | None = None) -> list[int]:
        ids = []
        meta = metadata or [{}] * len(texts)
        for text, m in zip(texts, meta):
            ids.append(self.add(text, m))
        return ids

    def _update_vocab(self, text: str) -> None:
        """Track raw lowercase word frequencies for suggest()."""
        words = _MULTI_WS.sub(" ", text.lower().translate(_PUNCT)).split()
        for w in words:
            if w not in _STOP_WORDS and len(w) > 1:
                self._vocab_freq[w] = self._vocab_freq.get(w, 0) + 1

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

        # Partial sort: get top_k indices without full sort. Over-fetch by
        # the number of tombstoned docs so deleted hits don't shrink the
        # result count below top_k when possible.
        import numpy as np
        fetch_k = min(top_k + len(self._deleted), len(scores))
        idx = np.argpartition(scores, -fetch_k)[-fetch_k:]
        idx = idx[np.argsort(scores[idx])[::-1]]

        results = []
        for i in idx:
            doc_id = int(i)
            if doc_id in self._deleted or scores[i] <= 0:
                continue
            results.append((doc_id, float(scores[i]), self._metadata[doc_id]))
            if len(results) >= top_k:
                break
        return results

    def get_metadata(self, doc_id: int) -> dict:
        if 0 <= doc_id < len(self._metadata):
            return self._metadata[doc_id]
        return {}

    def __len__(self) -> int:
        return len(self._tokenized_corpus) - len(self._deleted)

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete(self, doc_id: int) -> bool:
        """Soft-delete a document by doc_id. Returns False if unknown/already deleted."""
        if not (0 <= doc_id < len(self._tokenized_corpus)) or doc_id in self._deleted:
            return False
        self._deleted.add(doc_id)
        return True

    def delete_by_url(self, url: str) -> bool:
        """Soft-delete the document indexed under this URL, if any."""
        doc_id = self._url_to_id.get(url)
        if doc_id is None:
            return False
        return self.delete(doc_id)

    def url_to_doc_id(self, url: str) -> int | None:
        return self._url_to_id.get(url)

    def is_deleted(self, doc_id: int) -> bool:
        return doc_id in self._deleted

    def deleted_count(self) -> int:
        return len(self._deleted)

    # ------------------------------------------------------------------
    # Autocomplete
    # ------------------------------------------------------------------

    def suggest(self, prefix: str, limit: int = 10) -> list[str]:
        """
        Return up to `limit` indexed words starting with `prefix`,
        ranked by frequency across the corpus.

        Uses raw (unstemmed) words, so results read naturally as
        autocomplete suggestions rather than BM25's internal stems.
        """
        prefix = prefix.strip().lower()
        if not prefix:
            return []
        matches = [
            (word, freq) for word, freq in self._vocab_freq.items()
            if word.startswith(prefix)
        ]
        matches.sort(key=lambda pair: (-pair[1], pair[0]))
        return [word for word, _freq in matches[:limit]]

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
                    "deleted":    self._deleted,
                    "url_to_id":  self._url_to_id,
                    "vocab_freq": self._vocab_freq,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info(f"BM25 index saved to {path} ({len(self)} docs)")

    @classmethod
    def load(cls, path: str | Path) -> BM25Index:
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls(k1=data["k1"], b=data["b"])
        idx._tokenized_corpus = data["corpus"]
        idx._metadata         = data["metadata"]
        # New fields are absent in indices pickled before this feature —
        # default to empty rather than failing to load old data.
        idx._deleted     = data.get("deleted", set())
        idx._url_to_id   = data.get("url_to_id", {})
        idx._vocab_freq  = data.get("vocab_freq", {})
        idx._dirty = True   # will rebuild on next search
        logger.info(f"BM25 index loaded from {path} ({len(idx)} docs)")
        return idx
