"""
FAISS-backed dense vector store.

Index types
-----------
flat    : exact L2/IP search — correct, slow for >1M vectors
ivf     : IVF + flat quantizer — fast approximate, good recall
hnsw    : HNSW graph — best recall/speed trade-off for <10M vectors (default)

All vectors are expected to be L2-normalised float32 (use inner-product
== cosine similarity).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Literal

import faiss
import numpy as np
from loguru import logger


IndexType = Literal["flat", "ivf", "hnsw"]


class VectorStore:
    """
    Stores (doc_id → vector) mappings in a FAISS index,
    with a parallel Python list holding per-doc metadata.

    doc_id is an integer that aligns with BM25Index's doc_ids
    so the two indices share the same document space.
    """

    def __init__(
        self,
        dim: int,
        *,
        index_type: IndexType = "hnsw",
        # HNSW params
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
        # IVF params
        ivf_nlist: int = 256,
        ivf_nprobe: int = 32,
    ):
        self.dim        = dim
        self.index_type = index_type
        self._meta: list[dict[str, Any]] = []
        self._doc_ids: list[int]         = []

        # Build the FAISS index
        if index_type == "flat":
            self._index = faiss.IndexFlatIP(dim)

        elif index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self._index.hnsw.efConstruction = hnsw_ef_construction
            self._index.hnsw.efSearch       = hnsw_ef_search

        elif index_type == "ivf":
            quantizer    = faiss.IndexFlatIP(dim)
            self._index  = faiss.IndexIVFFlat(
                quantizer, dim, ivf_nlist, faiss.METRIC_INNER_PRODUCT
            )
            self._index.nprobe = ivf_nprobe
            self._trained = False

        else:
            raise ValueError(f"Unknown index_type: {index_type}")

        logger.info(f"VectorStore | dim={dim} type={index_type}")

    # ------------------------------------------------------------------
    # Adding vectors
    # ------------------------------------------------------------------

    def add(self, doc_id: int, vector: np.ndarray, metadata: dict | None = None) -> None:
        """Add a single vector."""
        vec = self._ensure_float32(vector).reshape(1, -1)
        self._maybe_train(vec)
        self._index.add(vec)
        self._doc_ids.append(doc_id)
        self._meta.append(metadata or {})

    def add_batch(
        self,
        doc_ids: list[int],
        vectors: np.ndarray,
        metadata: list[dict] | None = None,
    ) -> None:
        """Add N vectors at once (faster than N single adds)."""
        vecs = self._ensure_float32(vectors)
        self._maybe_train(vecs)
        self._index.add(vecs)
        self._doc_ids.extend(doc_ids)
        meta = metadata or [{}] * len(doc_ids)
        self._meta.extend(meta)

    def __len__(self) -> int:
        return self._index.ntotal

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 100,
    ) -> list[tuple[int, float, dict]]:
        """
        ANN search.

        Returns
        -------
        list of (doc_id, score, metadata), sorted descending by score.
        """
        if len(self) == 0:
            return []

        q = self._ensure_float32(query_vec).reshape(1, -1)
        k = min(top_k, len(self))

        scores, indices = self._index.search(q, k)
        scores  = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:   # FAISS returns -1 for empty slots
                continue
            doc_id = self._doc_ids[idx]
            results.append((doc_id, float(score), self._meta[idx]))

        return results

    # ------------------------------------------------------------------
    # IVF training
    # ------------------------------------------------------------------

    def _maybe_train(self, vecs: np.ndarray) -> None:
        if self.index_type != "ivf":
            return
        if not self._index.is_trained:
            if len(self) + len(vecs) < self._index.nlist * 10:
                return   # not enough data yet to train
            logger.info("Training IVF index...")
            all_vecs = self._collect_vectors()
            train    = np.vstack([all_vecs, vecs]) if all_vecs is not None else vecs
            self._index.train(train)

    def _collect_vectors(self) -> np.ndarray | None:
        """Reconstruct stored vectors from a flat index (only works for Flat/IVFFlat)."""
        n = len(self)
        if n == 0:
            return None
        vecs = np.zeros((n, self.dim), dtype=np.float32)
        self._index.reconstruct_n(0, n, vecs)
        return vecs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(directory / "faiss.index"))
        with open(directory / "meta.pkl", "wb") as f:
            pickle.dump({"doc_ids": self._doc_ids, "meta": self._meta}, f)
        logger.info(f"VectorStore saved to {directory} ({len(self)} vectors)")

    @classmethod
    def load(cls, directory: str | Path, dim: int, **kwargs) -> "VectorStore":
        directory = Path(directory)
        store = cls(dim, **kwargs)
        store._index = faiss.read_index(str(directory / "faiss.index"))
        with open(directory / "meta.pkl", "rb") as f:
            data = pickle.load(f)
        store._doc_ids = data["doc_ids"]
        store._meta    = data["meta"]
        logger.info(f"VectorStore loaded from {directory} ({len(store)} vectors)")
        return store

    # ------------------------------------------------------------------
    # GPU offload (optional)
    # ------------------------------------------------------------------

    def to_gpu(self, gpu_id: int = 0) -> None:
        """Move index to GPU for faster search (requires faiss-gpu)."""
        res = faiss.StandardGpuResources()
        self._index = faiss.index_cpu_to_gpu(res, gpu_id, self._index)
        logger.info(f"VectorStore moved to GPU {gpu_id}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_float32(arr: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(arr, dtype=np.float32)
