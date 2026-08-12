"""
Minimal Prometheus-text-format metrics — no `prometheus_client`
dependency, just counters guarded by a lock, rendered on GET /metrics.

Tracks per-route request counts (by status code) and cumulative
latency, plus process uptime. Good enough to wire into a scrape
config; swap for `prometheus_client` if you need histograms/summaries.
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._request_counts: dict[tuple[str, str, int], int] = defaultdict(int)
        self._latency_sum: dict[tuple[str, str], float] = defaultdict(float)
        self._latency_count: dict[tuple[str, str], int] = defaultdict(int)
        self.started_at = time.time()

    def record(self, method: str, path: str, status_code: int, duration_s: float) -> None:
        with self._lock:
            self._request_counts[(method, path, status_code)] += 1
            self._latency_sum[(method, path)] += duration_s
            self._latency_count[(method, path)] += 1

    def render_prometheus(self) -> str:
        lines = [
            "# HELP axon_http_requests_total Total HTTP requests handled",
            "# TYPE axon_http_requests_total counter",
        ]
        with self._lock:
            for (method, path, status), count in sorted(self._request_counts.items()):
                lines.append(
                    f'axon_http_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}'
                )

            lines += [
                "# HELP axon_http_request_duration_seconds_sum Cumulative request duration",
                "# TYPE axon_http_request_duration_seconds_sum counter",
            ]
            for (method, path), total in sorted(self._latency_sum.items()):
                lines.append(
                    f'axon_http_request_duration_seconds_sum{{method="{method}",path="{path}"}} {total:.6f}'
                )

            lines += [
                "# HELP axon_http_request_duration_seconds_count Requests timed",
                "# TYPE axon_http_request_duration_seconds_count counter",
            ]
            for (method, path), count in sorted(self._latency_count.items()):
                lines.append(
                    f'axon_http_request_duration_seconds_count{{method="{method}",path="{path}"}} {count}'
                )

        lines += [
            "# HELP axon_uptime_seconds Seconds since process start",
            "# TYPE axon_uptime_seconds gauge",
            f"axon_uptime_seconds {time.time() - self.started_at:.2f}",
        ]
        return "\n".join(lines) + "\n"


# Process-wide singleton — simplest option for a single-worker deployment.
metrics = MetricsRegistry()


class MetricsMiddleware(BaseHTTPMiddleware):
    """Times every request and records it in the shared `metrics` registry."""

    def __init__(self, app: ASGIApp, registry: MetricsRegistry | None = None):
        super().__init__(app)
        self._registry = registry or metrics

    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - t0
        # route.path (e.g. "/search") rather than the raw URL avoids an
        # unbounded label cardinality blow-up from path params / query strings.
        route = request.scope.get("route")
        path = route.path if route is not None else request.url.path
        self._registry.record(request.method, path, response.status_code, duration)
        return response
