"""
Content extraction pipeline.

Turns raw HTML into structured, clean text using a cascade:
  1. trafilatura  (best at article/blog content)
  2. readability  (fallback for more structured pages)
  3. raw BS4 body text (last resort)

Also extracts metadata: title, description, publish date, language,
canonical URL, author, outlinks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import trafilatura
from bs4 import BeautifulSoup
from loguru import logger
from readability import Document

from .async_crawler import CrawlResult

# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

@dataclass
class ExtractedPage:
    url: str
    canonical_url: str
    title: str
    text: str                       # main body text, clean
    description: str = ""
    author: str = ""
    language: str = "en"
    publish_date: str = ""
    outlinks: list[str] = field(default_factory=list)
    word_count: int = 0
    char_count: int = 0
    # Quality signals
    link_density: float = 0.0      # links / total words
    avg_sentence_len: float = 0.0

    def is_valid(self, min_words: int = 50) -> bool:
        return self.word_count >= min_words and bool(self.text.strip())


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MULTI_SPACE    = re.compile(r"\s{2,}")
_MULTI_NEWLINE  = re.compile(r"\n{3,}")


class ContentExtractor:
    """
    Stateless extractor — call extract(crawl_result) per page.
    Thread-safe (no shared mutable state).
    """

    def extract(self, page: CrawlResult) -> Optional[ExtractedPage]:
        html = page.html
        url  = page.final_url or page.url

        try:
            text = self._extract_text(html, url)
            if not text:
                return None

            meta   = self._extract_meta(html, url)
            stats  = self._compute_stats(html, text)

            return ExtractedPage(
                url=url,
                canonical_url=meta.get("canonical", url),
                title=meta.get("title", ""),
                text=text,
                description=meta.get("description", ""),
                author=meta.get("author", ""),
                language=meta.get("language", "en"),
                publish_date=meta.get("date", ""),
                outlinks=meta.get("outlinks", []),
                word_count=stats["word_count"],
                char_count=stats["char_count"],
                link_density=stats["link_density"],
                avg_sentence_len=stats["avg_sentence_len"],
            )
        except Exception as exc:
            logger.warning(f"Extraction failed for {url}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Text extraction cascade
    # ------------------------------------------------------------------

    def _extract_text(self, html: str, url: str) -> str:
        # --- Attempt 1: trafilatura ---
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_precision=True,
        )
        if text and len(text.split()) >= 50:
            return self._clean(text)

        # --- Attempt 2: readability ---
        try:
            doc  = Document(html)
            body = BeautifulSoup(doc.summary(), "lxml").get_text(separator="\n")
            body = self._clean(body)
            if len(body.split()) >= 50:
                return body
        except Exception:
            pass

        # --- Attempt 3: raw body text ---
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        raw = soup.get_text(separator="\n")
        return self._clean(raw)

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    def _extract_meta(self, html: str, url: str) -> dict:
        soup = BeautifulSoup(html, "lxml")
        meta: dict = {"outlinks": []}

        # Title
        title_tag = soup.find("title")
        og_title  = soup.find("meta", property="og:title")
        meta["title"] = (
            og_title["content"] if og_title and og_title.get("content")
            else (title_tag.get_text(strip=True) if title_tag else "")
        )

        # Description
        og_desc  = soup.find("meta", property="og:description")
        std_desc = soup.find("meta", attrs={"name": "description"})
        meta["description"] = (
            og_desc["content"] if og_desc and og_desc.get("content")
            else (std_desc["content"] if std_desc and std_desc.get("content") else "")
        )

        # Canonical
        canonical = soup.find("link", rel="canonical")
        meta["canonical"] = canonical["href"] if canonical and canonical.get("href") else url

        # Author
        author_meta = soup.find("meta", attrs={"name": "author"})
        meta["author"] = author_meta["content"] if author_meta and author_meta.get("content") else ""

        # Date — try common patterns
        for selector in [
            {"name": "article:published_time"},
            {"property": "article:published_time"},
            {"itemprop": "datePublished"},
            {"name": "pubdate"},
        ]:
            tag = soup.find("meta", attrs=selector)
            if tag and tag.get("content"):
                meta["date"] = tag["content"]
                break
        else:
            meta["date"] = ""

        # Language
        html_tag = soup.find("html")
        meta["language"] = html_tag.get("lang", "en").split("-")[0] if html_tag else "en"

        # Outlinks (absolute)
        base_domain = urlparse(url).netloc
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("http") and urlparse(href).netloc != base_domain:
                meta["outlinks"].append(href)

        return meta

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _compute_stats(self, html: str, text: str) -> dict:
        words = text.split()
        sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        avg_sent = len(words) / max(len(sentences), 1)

        # Link density: count anchor text words in raw HTML
        soup = BeautifulSoup(html, "lxml")
        link_words = sum(len(a.get_text().split()) for a in soup.find_all("a"))
        density = link_words / max(len(words), 1)

        return {
            "word_count": len(words),
            "char_count": len(text),
            "link_density": round(density, 4),
            "avg_sentence_len": round(avg_sent, 2),
        }

    # ------------------------------------------------------------------
    # Cleaner
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(text: str) -> str:
        text = _MULTI_SPACE.sub(" ", text)
        text = _MULTI_NEWLINE.sub("\n\n", text)
        return text.strip()
