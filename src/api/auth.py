"""
Optional API key authentication.

Disabled by default (matches the CORS pattern in config.py) — set
API_KEYS in the environment to require an `X-API-Key` header on every
request except the always-open health/metrics/docs paths.

This is deliberately simple (static shared keys, no scoping/rotation)
to address the README's "no authentication is implemented" gap without
pulling in a full auth stack. Swap for OAuth2/JWT if you need per-user
identity.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

# Paths that stay reachable without a key, even when auth is enabled —
# orchestrators / uptime checks / API explorers need these open.
EXEMPT_PATHS = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}

API_KEY_HEADER = "x-api-key"


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Rejects requests missing a valid `X-API-Key` header.

    No-op when `api_keys` is empty, so existing local/dev deployments
    are unaffected unless they opt in via the API_KEYS env var.
    """

    def __init__(self, app: ASGIApp, api_keys: list[str]):
        super().__init__(app)
        self._api_keys = set(api_keys)

    async def dispatch(self, request: Request, call_next):
        if not self._api_keys or request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        supplied = request.headers.get(API_KEY_HEADER)
        if supplied is None or supplied not in self._api_keys:
            return JSONResponse(
                {"detail": "Missing or invalid API key. Send it via the X-API-Key header."},
                status_code=401,
            )
        return await call_next(request)
