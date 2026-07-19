"""
Heuristic content quality scorer.

Produces a 0.0–1.0 quality score for a crawled page.
Pages below `threshold` are dropped before indexing.

Signals used
------------
- word_count         : too short → bad
- avg_sentence_len   : very short (listicle) or very long (auto-gen) → bad
- link_density       : high link ratio → navigation page, not content
- char_count         : sanity check
- type_token_ratio   : vocabulary richness (unique / total words)
- boilerplate ratio  : fraction of text that looks like boilerplate
- digit ratio        : pages that are mostly numbers (data dumps)
- uppercase ratio    : screaming or auto-generated content

Weights are hand-tuned heuristics, not learned. For a production
system you'd replace this with a classifier trained on human labels.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class QualityReport:
    score:          float
    word_count:     int
    ttr:            float
    link_density:   float
    avg_sent_len:   float
    digit_ratio:    float
    upper_ratio:    float
    passed:         bool

    def __str__(self) -> str:
        return (
            f"QualityReport(score={self.score:.3f} words={self.word_count} "
            f"ttr={self.ttr:.3f} link_density={self.link_density:.3f} "
            f"passed={self.passed})"
        )


_BOILERPLATE_RE = re.compile(
    r"(cookie|subscribe|newsletter|copyright|all rights reserved|"
    r"terms of service|privacy policy|follow us)",
    re.IGNORECASE,
)


class QualityScorer:
    """
    Scores an ExtractedPage for content quality.

    Parameters
    ----------
    threshold     : minimum score to pass (0.0–1.0)
    min_words     : hard minimum word count
    max_link_density : hard maximum link density
    """

    def __init__(
        self,
        *,
        threshold: float       = 0.35,
        min_words: int         = 80,
        max_link_density: float = 0.4,
    ):
        self.threshold        = threshold
        self.min_words        = min_words
        self.max_link_density = max_link_density

    # ------------------------------------------------------------------

    def score(self, page) -> float:
        """Score a page and return 0.0–1.0. Uses ExtractedPage attributes."""
        return self.score_text(
            text=page.text,
            word_count=page.word_count,
            link_density=page.link_density,
            avg_sentence_len=page.avg_sentence_len,
        )

    def score_text(
        self,
        text: str,
        *,
        word_count: Optional[int]  = None,
        link_density: float        = 0.0,
        avg_sentence_len: float    = 0.0,
    ) -> float:
        words = text.split()
        wc    = word_count if word_count is not None else len(words)

        # --- Hard gates ---
        if wc < self.min_words:
            return 0.0
        if link_density > self.max_link_density:
            return 0.0

        # --- Soft signals (each 0–1, then weighted average) ---
        signals: list[tuple[float, float]] = []   # (value, weight)

        # Word count — log sigmoid centred at 300 words
        wc_score = 1.0 / (1.0 + math.exp(-0.01 * (wc - 300)))
        signals.append((wc_score, 2.0))

        # Type-token ratio (vocabulary richness)
        ttr = len(set(w.lower() for w in words)) / max(wc, 1)
        ttr_score = min(ttr * 2.0, 1.0)   # 0.5 TTR → 1.0
        signals.append((ttr_score, 1.5))

        # Link density (lower is better)
        ld_score = 1.0 - min(link_density / self.max_link_density, 1.0)
        signals.append((ld_score, 1.5))

        # Sentence length — penalise very short (<6) or very long (>50)
        if avg_sentence_len > 0:
            if 6 <= avg_sentence_len <= 50:
                sl_score = 1.0
            elif avg_sentence_len < 6:
                sl_score = avg_sentence_len / 6.0
            else:
                sl_score = max(0.0, 1.0 - (avg_sentence_len - 50) / 50)
            signals.append((sl_score, 1.0))

        # Digit ratio — penalise data-dump pages
        digits     = sum(1 for c in text if c.isdigit())
        dig_ratio  = digits / max(len(text), 1)
        dig_score  = 1.0 - min(dig_ratio * 5, 1.0)
        signals.append((dig_score, 0.5))

        # Uppercase ratio — penalise ALL-CAPS content
        uppers     = sum(1 for c in text if c.isupper())
        up_ratio   = uppers / max(len(text), 1)
        up_score   = 1.0 - min(up_ratio * 3, 1.0)
        signals.append((up_score, 0.5))

        # Boilerplate density
        boiler_matches = len(_BOILERPLATE_RE.findall(text))
        bp_score = max(0.0, 1.0 - boiler_matches / 10)
        signals.append((bp_score, 0.8))

        # Weighted average
        total_weight = sum(w for _, w in signals)
        score = sum(v * w for v, w in signals) / total_weight

        return round(score, 4)

    def report(self, page) -> QualityReport:
        score = self.score(page)
        words = page.text.split()
        wc    = len(words)
        ttr   = len(set(w.lower() for w in words)) / max(wc, 1)
        digits = sum(1 for c in page.text if c.isdigit())
        uppers = sum(1 for c in page.text if c.isupper())
        return QualityReport(
            score=score,
            word_count=wc,
            ttr=round(ttr, 4),
            link_density=page.link_density,
            avg_sent_len=page.avg_sentence_len,
            digit_ratio=round(digits / max(len(page.text), 1), 4),
            upper_ratio=round(uppers / max(len(page.text), 1), 4),
            passed=score >= self.threshold,
        )
