"""
End-to-end indexing pipeline.

Stage 1 → AsyncCrawler      : fetch pages
Stage 2 → ContentExtractor  : HTML → clean ExtractedPage
Stage 3 → QualityScorer     : drop low-quality pages
Stage 4 → DedupFilter       : MinHash near-dedup
Stage 5 → Embedder          : text → dense float32 vectors
Stage 6 → BM25Index         : add to sparse inverted index
Stage 7 → VectorStore       : add to FAISS dense index
Stage 8 → LinkGraph         : record outlinks, recompute PageRank

The pipeline is streaming / async — stages run concurrently via
asyncio queues so the crawler, extractor, and GPU embedder overlap.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger

from crawler.async_crawler import AsyncCrawler, CrawlResult
from crawler.content_extractor import ContentExtractor, ExtractedPage
from crawler.link_graph import LinkGraph
from utils.dedup import DedupFilter
from utils.quality_scorer import QualityScorer
from .embedder import Embedder
from .bm25 import BM25Index
from .vector_store import VectorStore


@dataclass
class IndexStats:
    crawled:   int = 0
    extracted: int = 0
    deduped:   int = 0
    quality_dropped: int = 0
    indexed:   int = 0
    elapsed:   float = 0.0

    def log(self) -> None:
        logger.info(
            f"Pipeline stats | crawled={self.crawled} extracted={self.extracted} "
            f"deduped={self.deduped} quality_dropped={self.quality_dropped} "
            f"indexed={self.indexed} elapsed={self.elapsed:.1f}s"
        )


class IndexPipeline:
    """
    Orchestrates the full crawl → index pipeline.

    Parameters
    ----------
    crawler      : configured AsyncCrawler
    extractor    : ContentExtractor (stateless)
    quality      : QualityScorer
    dedup        : DedupFilter (MinHash LSH)
    embedder     : Embedder
    bm25         : BM25Index
    vector_store : VectorStore (FAISS)
    link_graph   : LinkGraph
    embed_batch  : batch size fed to the embedder
    """

    def __init__(
        self,
        *,
        crawler: AsyncCrawler,
        extractor: ContentExtractor,
        quality: QualityScorer,
        dedup: DedupFilter,
        embedder: Embedder,
        bm25: BM25Index,
        vector_store: VectorStore,
        link_graph: LinkGraph,
        embed_batch: int = 64,
        extract_workers: int = 8,
    ):
        self.crawler      = crawler
        self.extractor    = extractor
        self.quality      = quality
        self.dedup        = dedup
        self.embedder     = embedder
        self.bm25         = bm25
        self.vector_store = vector_store
        self.link_graph   = link_graph
        self.embed_batch  = embed_batch
        self.extract_workers = extract_workers

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self, seeds: list[str]) -> IndexStats:
        stats = IndexStats()
        t0    = time.monotonic()

        # Inter-stage queues
        crawl_q:   asyncio.Queue[Optional[CrawlResult]]  = asyncio.Queue(maxsize=256)
        extract_q: asyncio.Queue[Optional[ExtractedPage]] = asyncio.Queue(maxsize=256)

        # Run stages concurrently
        await asyncio.gather(
            self._stage_crawl(seeds, crawl_q, stats),
            self._stage_extract(crawl_q, extract_q, stats),
            self._stage_index(extract_q, stats),
        )

        stats.elapsed = time.monotonic() - t0
        stats.log()
        self.link_graph.compute_pagerank()
        return stats

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    async def _stage_crawl(
        self,
        seeds: list[str],
        out_q: asyncio.Queue,
        stats: IndexStats,
    ) -> None:
        async for result in self.crawler.crawl(seeds):
            stats.crawled += 1
            await out_q.put(result)
        await out_q.put(None)   # sentinel

    async def _stage_extract(
        self,
        in_q: asyncio.Queue,
        out_q: asyncio.Queue,
        stats: IndexStats,
    ) -> None:
        loop = asyncio.get_event_loop()
        semaphore = asyncio.Semaphore(self.extract_workers)

        async def _extract_one(result: CrawlResult) -> None:
            async with semaphore:
                page = await loop.run_in_executor(
                    None, self.extractor.extract, result
                )
                if page is None:
                    return

                # Quality gate
                score = self.quality.score(page)
                if score < self.quality.threshold:
                    stats.quality_dropped += 1
                    return

                # Dedup gate
                if self.dedup.is_duplicate(page.text):
                    stats.deduped += 1
                    return

                stats.extracted += 1
                await out_q.put(page)

                # Record links for PageRank
                self.link_graph.add_page(page.url, page.outlinks)

        tasks: set[asyncio.Task] = set()

        while True:
            item = await in_q.get()
            if item is None:
                break
            task = asyncio.create_task(_extract_one(item))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

        await asyncio.gather(*tasks)
        await out_q.put(None)

    async def _stage_index(
        self,
        in_q: asyncio.Queue,
        stats: IndexStats,
    ) -> None:
        batch_pages: list[ExtractedPage] = []
        loop = asyncio.get_event_loop()

        async def _flush(pages: list[ExtractedPage]) -> None:
            if not pages:
                return
            texts = [p.text for p in pages]
            urls  = [p.url  for p in pages]

            # Embed (CPU/GPU blocking call → run in executor)
            vectors: np.ndarray = await loop.run_in_executor(
                None, self.embedder.encode, texts
            )

            # Add to indices (fast, do on event-loop thread)
            for i, page in enumerate(pages):
                doc_id = self.bm25.add(page.text, metadata={
                    "url":   page.url,
                    "title": page.title,
                    "desc":  page.description,
                    "date":  page.publish_date,
                })
                self.vector_store.add(doc_id, vectors[i], {
                    "url":       page.url,
                    "title":     page.title,
                    "word_count": page.word_count,
                })
                stats.indexed += 1

        while True:
            item = await in_q.get()
            if item is None:
                break
            batch_pages.append(item)
            if len(batch_pages) >= self.embed_batch:
                await _flush(batch_pages)
                batch_pages = []

        await _flush(batch_pages)   # flush remainder
