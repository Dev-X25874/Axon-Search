"""
Simple in-memory rate limiter.

Sliding-window counter per client, keyed by API key when present
(so authenticated callers get their own budget) and by remote IP
otherwise. Disabled by default via RATE_LIMIT_PER_MINUTE=0.

This is process-local — fine for a single `api_workers=1` deployment.
Replace with a Redis-backed limiter (e.g. `slowapi` + Redis) before
running multiple workers/replicas, since counts won't be shared across
processes.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from .auth import API_KEY_HEADER, EXEMPT_PATHS

_WINDOW_S = 60.0


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, requests_per_minute: int):
        super().__init__(app)
        self._limit = requests_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def _client_key(self, request: Request) -> str:
        api_key = request.headers.get(API_KEY_HEADER)
        if api_key:
            return f"key:{api_key}"
        client = request.client
        return f"ip:{client.host if client else 'unknown'}"

    async def dispatch(self, request: Request, call_next):
        if self._limit <= 0 or request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        now = time.monotonic()
        key = self._client_key(request)

        with self._lock:
            window = self._hits[key]
            while window and now - window[0] > _WINDOW_S:
                window.popleft()

            if len(window) >= self._limit:
                retry_after = max(0.0, _WINDOW_S - (now - window[0]))
                return JSONResponse(
                    {"detail": "Rate limit exceeded. Slow down and retry shortly."},
                    status_code=429,
                    headers={"Retry-After": str(int(retry_after) + 1)},
                )

            window.append(now)

        return await call_next(request)
