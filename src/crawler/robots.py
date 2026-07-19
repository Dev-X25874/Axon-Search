"""
Async robots.txt cache.

- Fetches and parses robots.txt per domain
- Caches results in memory with a 24-hour TTL
- Falls back to allow-all on fetch failure (conservative crawl policy)
- Extracts Crawl-Delay directive to set per-domain rate limits
"""

from __future__ import annotations

import asyncio
import time
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import aiohttp
from loguru import logger

_TTL = 86_400          # 24 hours
_BOT_NAME = "AxonSearchBot"
_FETCH_TIMEOUT = 10.0


class _CachedRobots:
    __slots__ = ("parser", "crawl_delay", "fetched_at")

    def __init__(self, parser: RobotFileParser, crawl_delay: float):
        self.parser       = parser
        self.crawl_delay  = crawl_delay
        self.fetched_at   = time.monotonic()

    def is_expired(self) -> bool:
        return time.monotonic() - self.fetched_at > _TTL


class RobotsCache:
    """
    Thread-safe (asyncio-safe) robots.txt cache.

    Concurrent requests for the same domain are serialized via a
    per-domain lock so we never fetch the same robots.txt twice.
    """

    def __init__(self):
        self._cache:  dict[str, _CachedRobots]    = {}
        self._locks:  dict[str, asyncio.Lock]      = {}
        self._global  = asyncio.Lock()

    async def is_allowed(
        self,
        url: str,
        session: aiohttp.ClientSession | None = None,
    ) -> bool:
        """Return True if the bot is allowed to fetch this URL."""
        domain = self._domain(url)
        cached = await self._get_or_fetch(domain, session)
        return cached.parser.can_fetch(_BOT_NAME, url)

    async def crawl_delay(
        self,
        url: str,
        session: aiohttp.ClientSession | None = None,
    ) -> float:
        """Return the Crawl-Delay for this URL's domain (default 1.0 s)."""
        domain = self._domain(url)
        cached = await self._get_or_fetch(domain, session)
        return cached.crawl_delay

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _get_or_fetch(
        self,
        domain: str,
        session: aiohttp.ClientSession | None,
    ) -> _CachedRobots:
        # Fast path (no lock contention on cache hit)
        entry = self._cache.get(domain)
        if entry and not entry.is_expired():
            return entry

        # Acquire per-domain lock to avoid duplicate fetches
        async with self._global:
            if domain not in self._locks:
                self._locks[domain] = asyncio.Lock()
        lock = self._locks[domain]

        async with lock:
            # Double-check after acquiring lock
            entry = self._cache.get(domain)
            if entry and not entry.is_expired():
                return entry

            entry = await self._fetch(domain, session)
            self._cache[domain] = entry
            return entry

    async def _fetch(
        self,
        domain: str,
        session: aiohttp.ClientSession | None,
    ) -> _CachedRobots:
        robots_url = f"https://{domain}/robots.txt"
        parser = RobotFileParser(robots_url)
        crawl_delay = 1.0

        try:
            headers = {"User-Agent": _BOT_NAME}
            timeout = aiohttp.ClientTimeout(total=_FETCH_TIMEOUT)

            if session:
                async with session.get(robots_url, headers=headers, timeout=timeout) as resp:
                    if resp.status == 200:
                        text = await resp.text(errors="replace")
                        parser.parse(text.splitlines())
                        delay = parser.crawl_delay(_BOT_NAME)
                        if delay:
                            crawl_delay = float(delay)
            else:
                # Fallback: run synchronous fetch in executor
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, parser.read)

        except Exception as exc:
            logger.debug(f"robots.txt fetch failed for {domain}: {exc} — assuming allow-all")
            parser.allow_all = True

        return _CachedRobots(parser, crawl_delay)

    @staticmethod
    def _domain(url: str) -> str:
        return urlparse(url).netloc
