"""
In-memory link graph with iterative PageRank.

- Nodes  : URL strings (stored as integer IDs internally)
- Edges  : outlinks discovered during crawl
- PageRank is recomputed on demand; scores are used as authority
  signals in hybrid retrieval ranking.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional


class LinkGraph:
    """
    Directed link graph.

    IDs are assigned incrementally as new URLs are seen.
    The graph is stored as a CSR-style adjacency list for
    memory efficiency.
    """

    def __init__(self, damping: float = 0.85, iterations: int = 50, tol: float = 1e-6):
        self.damping    = damping
        self.iterations = iterations
        self.tol        = tol

        self._url_to_id: Dict[str, int]   = {}
        self._id_to_url: Dict[int, str]   = {}
        self._out_edges: Dict[int, List[int]] = defaultdict(list)
        self._in_edges:  Dict[int, List[int]] = defaultdict(list)
        self._scores:    Dict[int, float]  = {}
        self._dirty = True

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_link(self, src: str, dst: str) -> None:
        """Record a directed hyperlink src → dst."""
        src_id = self._get_or_create(src)
        dst_id = self._get_or_create(dst)
        if dst_id not in self._out_edges[src_id]:
            self._out_edges[src_id].append(dst_id)
            self._in_edges[dst_id].append(src_id)
            self._dirty = True

    def add_page(self, url: str, outlinks: list[str]) -> None:
        """Add all outlinks for a crawled page at once."""
        for dst in outlinks:
            self.add_link(url, dst)

    def node_count(self) -> int:
        return len(self._url_to_id)

    def edge_count(self) -> int:
        return sum(len(v) for v in self._out_edges.values())

    # ------------------------------------------------------------------
    # PageRank
    # ------------------------------------------------------------------

    def compute_pagerank(self) -> None:
        """Run iterative PageRank until convergence or max iterations."""
        n = len(self._url_to_id)
        if n == 0:
            return

        ids = list(self._url_to_id.values())
        rank: Dict[int, float] = {i: 1.0 / n for i in ids}
        d = self.damping

        for _ in range(self.iterations):
            new_rank: Dict[int, float] = {}
            delta = 0.0

            # dangling_sum is the same for every node — compute once per iteration
            dangling_sum = sum(
                rank[i] / n
                for i in ids
                if not self._out_edges[i]
            )

            for node_id in ids:
                in_sum = 0.0
                for src_id in self._in_edges[node_id]:
                    out_count = len(self._out_edges[src_id])
                    if out_count:
                        in_sum += rank[src_id] / out_count

                new_rank[node_id] = (1 - d) / n + d * (in_sum + dangling_sum)
                delta += abs(new_rank[node_id] - rank[node_id])

            rank = new_rank
            if delta < self.tol:
                break
        self._scores = rank
        self._dirty  = False

    def get_score(self, url: str) -> float:
        """Return the PageRank score for a URL (0.0 if unknown)."""
        if self._dirty:
            self.compute_pagerank()
        node_id = self._url_to_id.get(url)
        if node_id is None:
            return 0.0
        return self._scores.get(node_id, 0.0)

    def top_k(self, k: int = 100) -> list[tuple[str, float]]:
        """Return top-k URLs by PageRank score."""
        if self._dirty:
            self.compute_pagerank()
        ranked = sorted(
            ((self._id_to_url[i], s) for i, s in self._scores.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:k]

    def log_score(self, url: str) -> float:
        """Log-normalised PageRank — useful as a ranking feature."""
        raw = self.get_score(url)
        return math.log1p(raw * 1e6)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        if self._dirty:
            self.compute_pagerank()
        return {
            "nodes": self._url_to_id,
            "edges": {str(k): v for k, v in self._out_edges.items()},
            "scores": {str(k): v for k, v in self._scores.items()},
        }

    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> "LinkGraph":
        g = cls(**kwargs)
        g._url_to_id = data["nodes"]
        g._id_to_url = {v: k for k, v in data["nodes"].items()}
        g._out_edges = defaultdict(list, {int(k): v for k, v in data["edges"].items()})
        g._scores    = {int(k): v for k, v in data["scores"].items()}
        # Rebuild in-edges
        for src_id, dsts in g._out_edges.items():
            for dst_id in dsts:
                g._in_edges[dst_id].append(src_id)
        g._dirty = False
        return g

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, url: str) -> int:
        if url not in self._url_to_id:
            new_id = len(self._url_to_id)
            self._url_to_id[url] = new_id
            self._id_to_url[new_id] = url
        return self._url_to_id[url]
