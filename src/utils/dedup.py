"""
Near-duplicate detection using MinHash + LSH.

Uses the datasketch library's MinHashLSH for sub-linear lookup.

Jaccard threshold 0.8 means documents with ≥80% shingle overlap
are considered duplicates. This catches:
  - Mirror sites
  - Slightly modified reposts
  - Syndicated articles with minor edits

Shingles are character-level 5-grams (robust to tokenisation differences).
"""

from __future__ import annotations

import re
from typing import Optional

from datasketch import MinHash, MinHashLSH
from loguru import logger

_WHITESPACE = re.compile(r"\s+")


def _shingle(text: str, k: int = 5) -> set[str]:
    """Character k-gram shingles."""
    text = _WHITESPACE.sub(" ", text.lower()).strip()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


class DedupFilter:
    """
    Streaming near-duplicate filter.

    Call is_duplicate(text) for each document as it arrives.
    Returns True if the document is a near-duplicate of one already seen.
    Thread-safety: LSH index is NOT thread-safe — use within a single
    asyncio task or protect with a lock.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.8,
        num_perm: int = 128,
        shingle_k: int = 5,
    ):
        self.threshold = threshold
        self.num_perm  = num_perm
        self.shingle_k = shingle_k
        self._lsh   = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._count = 0

    def is_duplicate(self, text: str) -> bool:
        """Return True if text is a near-duplicate of an already-indexed doc."""
        mh = self._build_minhash(text)
        result = self._lsh.query(mh)
        if result:
            logger.debug(f"Dedup hit: similar to {result[0]}")
            return True
        # Not a duplicate — add to index
        key = f"doc_{self._count}"
        self._lsh.insert(key, mh)
        self._count += 1
        return False

    def similarity(self, text_a: str, text_b: str) -> float:
        """Estimate Jaccard similarity between two texts."""
        mh_a = self._build_minhash(text_a)
        mh_b = self._build_minhash(text_b)
        return mh_a.jaccard(mh_b)

    def reset(self) -> None:
        """Clear all stored fingerprints."""
        self._lsh   = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self._count = 0

    def __len__(self) -> int:
        return self._count

    # ------------------------------------------------------------------

    def _build_minhash(self, text: str) -> MinHash:
        mh = MinHash(num_perm=self.num_perm)
        for shingle in _shingle(text, k=self.shingle_k):
            mh.update(shingle.encode("utf-8"))
        return mh
