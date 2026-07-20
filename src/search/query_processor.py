"""
Query understanding and preprocessing.

Steps applied to every raw user query:
1. Normalisation     — lowercase, strip, collapse whitespace
2. Intent detection  — navigational / informational / transactional
3. Entity extraction — site:, filetype:, date: operators
4. Spell correction  — lightweight phonetic similarity (optional)
5. Expansion         — synonym lookup via WordNet
6. Embedding         — encode processed query for dense retrieval
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import nltk
from loguru import logger

for _r in ("wordnet", "averaged_perceptron_tagger", "punkt"):
    try:
        nltk.data.find(f"corpora/{_r}" if "tagger" not in _r else f"taggers/{_r}")
    except LookupError:
        nltk.download(_r, quiet=True)

from nltk.corpus import wordnet


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ParsedOperators:
    site:     Optional[str] = None
    filetype: Optional[str] = None
    date_from: Optional[str] = None
    date_to:   Optional[str] = None
    language:  Optional[str] = None
    exclude_terms: list[str] = field(default_factory=list)
    must_include:  list[str] = field(default_factory=list)


@dataclass
class ProcessedQuery:
    raw:            str
    clean:          str              # after operator stripping
    normalised:     str              # lowercase / deduped whitespace
    expanded:       str              # with synonyms appended
    tokens:         list[str]
    operators:      ParsedOperators
    intent:         str              # informational | navigational | transactional
    is_question:    bool
    # Set by HybridRetriever after embedding
    query_vector:   Optional[object] = None


# ---------------------------------------------------------------------------
# Intent signals
# ---------------------------------------------------------------------------

_NAV_SIGNALS  = {"site", "www", ".com", ".org", ".net", "login", "homepage"}
_TRANS_SIGNALS = {
    "buy", "price", "cheap", "discount", "order", "shop", "download",
    "install", "hire", "subscribe", "sign up", "register",
}
_QUESTION_STARTS = {"what", "why", "how", "when", "where", "who", "which", "is", "are", "can"}

# Operator patterns
_SITE_RE     = re.compile(r"\bsite:(\S+)")
_FILETYPE_RE = re.compile(r"\bfiletype:(\S+)")
_LANG_RE     = re.compile(r"\blang:([a-z]{2})")
_DATE_FROM   = re.compile(r"\bafter:(\d{4}-\d{2}-\d{2})")
_DATE_TO     = re.compile(r"\bbefore:(\d{4}-\d{2}-\d{2})")
_EXCLUDE_RE  = re.compile(r"-(\w+)")
_MUST_RE     = re.compile(r'\+"([^"]+)"|(\+\w+)')


class QueryProcessor:
    """
    Stateless query pre-processor.

    Usage
    -----
    proc = QueryProcessor()
    pq   = proc.process("how does attention work in transformers?")
    """

    def __init__(
        self,
        *,
        expand_synonyms: bool = True,
        max_expansion: int = 3,
        min_synonym_score: float = 0.8,
    ):
        self.expand_synonyms    = expand_synonyms
        self.max_expansion      = max_expansion
        self.min_synonym_score  = min_synonym_score

    def process(self, raw_query: str) -> ProcessedQuery:
        # 1. Extract operators
        ops   = self._parse_operators(raw_query)
        clean = self._strip_operators(raw_query)

        # 2. Normalise
        norm  = " ".join(clean.lower().split())

        # 3. Tokenise
        tokens = norm.split()

        # 4. Intent
        intent = self._detect_intent(tokens)

        # 5. Synonym expansion
        expanded = self._expand(norm) if self.expand_synonyms else norm

        return ProcessedQuery(
            raw=raw_query,
            clean=clean,
            normalised=norm,
            expanded=expanded,
            tokens=tokens,
            operators=ops,
            intent=intent,
            is_question=bool(tokens and tokens[0] in _QUESTION_STARTS),
        )

    # ------------------------------------------------------------------

    def _parse_operators(self, query: str) -> ParsedOperators:
        ops = ParsedOperators()

        m = _SITE_RE.search(query)
        if m: ops.site = m.group(1)

        m = _FILETYPE_RE.search(query)
        if m: ops.filetype = m.group(1)

        m = _LANG_RE.search(query)
        if m: ops.language = m.group(1)

        m = _DATE_FROM.search(query)
        if m: ops.date_from = m.group(1)

        m = _DATE_TO.search(query)
        if m: ops.date_to = m.group(1)

        ops.exclude_terms = _EXCLUDE_RE.findall(query)

        for m in _MUST_RE.finditer(query):
            term = m.group(1) or m.group(2).lstrip("+")
            ops.must_include.append(term)

        return ops

    @staticmethod
    def _strip_operators(query: str) -> str:
        patterns = [
            _SITE_RE, _FILETYPE_RE, _LANG_RE,
            _DATE_FROM, _DATE_TO, _MUST_RE,
        ]
        q = query
        for pat in patterns:
            q = pat.sub("", q)
        # Strip exclude operators (start of string or after whitespace)
        q = re.sub(r"(?:^|\s)-\w+", " ", q)
        return " ".join(q.split()).strip()

    def _detect_intent(self, tokens: list[str]) -> str:
        if not tokens:
            return "informational"

        word_set = set(tokens)
        if word_set & _NAV_SIGNALS:
            return "navigational"
        if word_set & _TRANS_SIGNALS:
            return "transactional"
        return "informational"

    def _expand(self, query: str) -> str:
        tokens = query.split()
        extras: list[str] = []

        for token in tokens:
            synsets = wordnet.synsets(token)
            added = 0
            for synset in synsets:
                if added >= self.max_expansion:
                    break
                for lemma in synset.lemmas():
                    name = lemma.name().replace("_", " ").lower()
                    if name != token and name not in tokens and name not in extras:
                        extras.append(name)
                        added += 1
                        if added >= self.max_expansion:
                            break

        return query + (" " + " ".join(extras) if extras else "")
