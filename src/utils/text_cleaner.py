"""
Text cleaning utilities used across the pipeline.

Normalises Unicode, collapses whitespace, strips boilerplate
patterns, and splits text into sentence-level chunks.
"""

from __future__ import annotations

import re
import unicodedata

_BOILERPLATE = re.compile(
    r"(cookie\s*policy|accept\s*cookies|privacy\s*policy|terms\s*of\s*service"
    r"|subscribe\s*to\s*our\s*newsletter|all\s*rights\s*reserved"
    r"|©\s*\d{4}|copyright\s*\d{4})",
    re.IGNORECASE,
)
_URL_PATTERN    = re.compile(r"https?://\S+")
_EMAIL_PATTERN  = re.compile(r"\b[\w.+-]+@[\w-]+\.\w+\b")
_MULTI_SPACE    = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE  = re.compile(r"\n{3,}")
_SENTENCE_END   = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


class TextCleaner:
    """
    Stateless text cleaning pipeline.

    Usage
    -----
    cleaner = TextCleaner()
    clean   = cleaner.clean(raw_text)
    chunks  = cleaner.chunk(clean, max_tokens=256)
    """

    def __init__(
        self,
        strip_urls: bool = True,
        strip_emails: bool = True,
        strip_boilerplate: bool = True,
        normalise_unicode: bool = True,
    ):
        self.strip_urls        = strip_urls
        self.strip_emails      = strip_emails
        self.strip_boilerplate = strip_boilerplate
        self.normalise_unicode = normalise_unicode

    def clean(self, text: str) -> str:
        if self.normalise_unicode:
            text = unicodedata.normalize("NFKC", text)

        if self.strip_urls:
            text = _URL_PATTERN.sub(" ", text)

        if self.strip_emails:
            text = _EMAIL_PATTERN.sub(" ", text)

        if self.strip_boilerplate:
            text = _BOILERPLATE.sub(" ", text)

        # Collapse whitespace
        text = _MULTI_SPACE.sub(" ", text)
        text = _MULTI_NEWLINE.sub("\n\n", text)
        return text.strip()

    def sentences(self, text: str) -> list[str]:
        """Split into sentences (heuristic, no dependency on NLTK punkt)."""
        parts = _SENTENCE_END.split(text)
        return [s.strip() for s in parts if s.strip()]

    def chunk(
        self,
        text: str,
        *,
        max_tokens: int = 256,
        overlap_tokens: int = 32,
        approx_chars_per_token: float = 4.5,
    ) -> list[str]:
        """
        Split text into overlapping chunks suitable for embedding.

        Uses a character-based approximation for token count.
        """
        max_chars     = int(max_tokens * approx_chars_per_token)
        overlap_chars = int(overlap_tokens * approx_chars_per_token)

        sents   = self.sentences(text)
        chunks:  list[str] = []
        current = ""

        for sent in sents:
            if len(current) + len(sent) + 1 > max_chars:
                if current:
                    chunks.append(current.strip())
                    # Overlap: keep last overlap_chars of current buffer
                    current = current[-overlap_chars:].strip() + " " + sent
                else:
                    # Single sentence exceeds limit — hard-split
                    for i in range(0, len(sent), max_chars - overlap_chars):
                        chunks.append(sent[i : i + max_chars])
                    current = ""
            else:
                current = (current + " " + sent).strip()

        if current:
            chunks.append(current.strip())

        return chunks

    @staticmethod
    def word_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def avg_word_len(text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        return sum(len(w) for w in words) / len(words)
