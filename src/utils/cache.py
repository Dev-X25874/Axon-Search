"""
Small thread-safe TTL + LRU cache.

Used to short-circuit identical /search requests. Deliberately
process-local and dependency-free — swap for Redis if you run
multiple API workers/replicas and need a shared cache.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class TTLCache(Generic[K, V]):
    """Fixed-size cache with per-entry expiry.

    Eviction order is LRU among *unexpired* entries; expired entries
    are dropped lazily on access rather than via a background sweep.
    """

    def __init__(self, maxsize: int = 512, ttl: float = 30.0):
        self.maxsize = maxsize
        self.ttl = ttl
        self._data: OrderedDict[K, tuple[V, float]] = OrderedDict()
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: K) -> V | None:
        if self.ttl <= 0:
            return None
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None

            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._data[key]
                self.misses += 1
                return None

            self._data.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: K, value: V) -> None:
        if self.ttl <= 0 or self.maxsize <= 0:
            return
        with self._lock:
            self._data[key] = (value, time.monotonic() + self.ttl)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def stats(self) -> dict[str, Any]:
        return {"size": len(self), "hits": self.hits, "misses": self.misses}


def make_key(*parts: Any) -> str:
    """Build a stable cache key from arbitrary (hashable-after-repr) parts.

    Dict parts are sorted by key first so equal filters/params always
    produce the same string regardless of insertion order.
    """
    normalised = []
    for part in parts:
        if isinstance(part, dict):
            normalised.append(tuple(sorted(part.items())))
        else:
            normalised.append(part)
    return repr(tuple(normalised))
