"""
Async web crawler package.

The crawler implementation lives in async_crawler.py. This file only
re-exports the public names so `from crawler import AsyncCrawler` and
`from crawler.async_crawler import AsyncCrawler` both work.

(Previously this file contained a full byte-for-byte copy of the
AsyncCrawler class from async_crawler.py — a copy/paste bug. Having
two independently-maintained definitions of the same class is a
correctness hazard: fixes to one silently don't apply to the other,
and `crawler.AsyncCrawler` vs `crawler.async_crawler.AsyncCrawler`
could drift apart. Import instead of duplicate.)
"""

from __future__ import annotations

from .async_crawler import AsyncCrawler, CrawlResult
from .robots import RobotsCache

__all__ = ["AsyncCrawler", "CrawlResult", "RobotsCache"]
