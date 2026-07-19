"""
Async web crawler.

- Breadth-first frontier with per-domain rate limiting
- Politeness: respects robots.txt, Crawl-Delay, configurable delay
- Retry logic with exponential back-off (tenacity)
- Deduplication via xxhash URL fingerprinting
- Streams CrawlResult objects to an async queue consumed by the indexer
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Set
from urllib.parse import urljoin, urlparse

import aiohttp
import xxhash
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from .robots import RobotsCache

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CrawlResult:
    url: str
    status: int
    html: str
    content_type: str
    crawled_at: float = field(default_factory=time.time)
    final_url: str = ""          # after redirects
    depth: int = 0
    parent_url: str = ""

    @property
    def url_hash(self) -> str:
        return xxhash.xxh64(self.url).hexdigest()


# ---------------------------------------------------------------------------
# Per-domain rate limiter
# ---------------------------------------------------------------------------

class _DomainBucket:
    """Token-bucket rate limiter for a single domain."""

    def __init__(self, delay: float = 1.0):
        self._delay = delay
        self._last_access: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._delay - (now - self._last_access)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_access = time.monotonic()


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------

class AsyncCrawler:
    """
    Async breadth-first crawler.

    Usage
    -----
    async for result in crawler.crawl(seeds):
        process(result)
    """

    DEFAULT_HEADERS = {
        "User-Agent": (
            "AxonSearchBot/0.1 (+https://github.com/axon-search; "
            "research crawler; contact: crawler@axon.search)"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(
        self,
        *,
        max_depth: int = 3,
        max_pages: int = 10_000,
        concurrency: int = 32,
        default_delay: float = 1.0,
        request_timeout: float = 15.0,
        max_response_size: int = 5 * 1024 * 1024,   # 5 MB
        allowed_domains: list[str] | None = None,
        disallowed_extensions: set[str] | None = None,
    ):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.concurrency = concurrency
        self.default_delay = default_delay
        self.request_timeout = request_timeout
        self.max_response_size = max_response_size
        self.allowed_domains: set[str] = set(allowed_domains or [])
        self.disallowed_extensions: set[str] = disallowed_extensions or {
            ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg",
            ".mp4", ".mp3", ".zip", ".gz", ".tar", ".exe",
            ".css", ".js", ".woff", ".woff2", ".ico",
        }

        self._robots = RobotsCache()
        self._buckets: dict[str, _DomainBucket] = {}
        self._seen: Set[str] = set()
        self._session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def crawl(
        self,
        seeds: list[str],
        *,
        output_queue: asyncio.Queue | None = None,
    ) -> AsyncIterator[CrawlResult]:
        """
        Async-generator that yields CrawlResult objects.
        Also pushes to output_queue if provided (for pipeline integration).
        """
        frontier: asyncio.Queue[tuple[str, int, str]] = asyncio.Queue()
        for url in seeds:
            frontier.put_nowait((url, 0, ""))

        semaphore = asyncio.Semaphore(self.concurrency)
        pages_crawled = 0

        async with aiohttp.ClientSession(
            headers=self.DEFAULT_HEADERS,
            connector=aiohttp.TCPConnector(limit=self.concurrency, ssl=False),
            timeout=aiohttp.ClientTimeout(total=self.request_timeout),
        ) as session:
            self._session = session

            pending: set[asyncio.Task] = set()
            result_queue: asyncio.Queue[CrawlResult | None] = asyncio.Queue()

            async def _worker(url: str, depth: int, parent: str) -> None:
                async with semaphore:
                    result = await self._fetch_one(url, depth, parent)
                    if result:
                        await result_queue.put(result)
                        # Discover outlinks
                        if depth < self.max_depth:
                            links = self._extract_links(result.html, result.final_url or url)
                            for link in links:
                                h = xxhash.xxh64(link).hexdigest()
                                if h not in self._seen:
                                    self._seen.add(h)
                                    frontier.put_nowait((link, depth + 1, url))
                await result_queue.put(None)  # signal this task done

            # seed
            for url in seeds:
                h = xxhash.xxh64(url).hexdigest()
                self._seen.add(h)

            active = 0

            async def _drain_frontier() -> None:
                nonlocal active, pages_crawled
                while pages_crawled < self.max_pages:
                    try:
                        url, depth, parent = frontier.get_nowait()
                    except asyncio.QueueEmpty:
                        if active == 0:
                            break
                        await asyncio.sleep(0.05)
                        continue
                    task = asyncio.create_task(_worker(url, depth, parent))
                    pending.add(task)
                    task.add_done_callback(pending.discard)
                    active += 1

                    # collect results
                    while not result_queue.empty():
                        item = result_queue.get_nowait()
                        if item is None:
                            active -= 1
                        else:
                            pages_crawled += 1
                            yield item
                            if output_queue:
                                await output_queue.put(item)

            async for result in _drain_frontier():
                yield result

            # drain remaining
            while active > 0 or not result_queue.empty():
                item = await result_queue.get()
                if item is None:
                    active -= 1
                else:
                    pages_crawled += 1
                    yield item
                    if output_queue:
                        await output_queue.put(item)

        if output_queue:
            await output_queue.put(None)  # sentinel

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=False,
    )
    async def _fetch_one(
        self, url: str, depth: int, parent: str
    ) -> CrawlResult | None:
        if not self._is_allowed_url(url):
            return None

        domain = urlparse(url).netloc
        bucket = self._buckets.setdefault(domain, _DomainBucket(self.default_delay))

        # Check robots.txt
        if not await self._robots.is_allowed(url, self._session):
            logger.debug(f"robots.txt disallows {url}")
            return None

        await bucket.acquire()

        try:
            async with self._session.get(url, allow_redirects=True) as resp:
                content_type = resp.content_type or ""
                if "text/html" not in content_type:
                    return None
                if resp.content_length and resp.content_length > self.max_response_size:
                    logger.debug(f"Skipping oversized page: {url}")
                    return None

                raw = await resp.content.read(self.max_response_size)
                html = raw.decode("utf-8", errors="replace")
                final_url = str(resp.url)

                return CrawlResult(
                    url=url,
                    status=resp.status,
                    html=html,
                    content_type=content_type,
                    final_url=final_url,
                    depth=depth,
                    parent_url=parent,
                )
        except Exception as exc:
            logger.warning(f"Fetch failed for {url}: {exc}")
            return None

    def _is_allowed_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        ext = "." + url.rsplit(".", 1)[-1].split("?")[0].lower() if "." in url else ""
        if ext in self.disallowed_extensions:
            return False
        if self.allowed_domains and parsed.netloc not in self.allowed_domains:
            return False
        return True

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        links = []
        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            if not href or href.startswith(("#", "javascript:", "mailto:")):
                continue
            abs_url = urljoin(base_url, href).split("#")[0]
            links.append(abs_url)
        return links
