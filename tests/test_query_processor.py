from __future__ import annotations

from search.query_processor import QueryProcessor


def test_site_operator_parsed_and_stripped():
    proc = QueryProcessor(expand_synonyms=False)
    pq = proc.process("site:arxiv.org transformers")
    assert pq.operators.site == "arxiv.org"
    assert "site:" not in pq.clean
    assert "transformers" in pq.clean


def test_filetype_operator_parsed():
    proc = QueryProcessor(expand_synonyms=False)
    pq = proc.process("filetype:pdf attention paper")
    assert pq.operators.filetype == "pdf"


def test_exclude_terms_parsed():
    proc = QueryProcessor(expand_synonyms=False)
    pq = proc.process("attention -vision")
    assert "vision" in pq.operators.exclude_terms
    assert "vision" not in pq.clean


def test_must_include_quoted_phrase_parsed():
    proc = QueryProcessor(expand_synonyms=False)
    pq = proc.process('+"flash attention" transformer')
    assert "flash attention" in pq.operators.must_include


def test_date_operators_parsed():
    proc = QueryProcessor(expand_synonyms=False)
    pq = proc.process("after:2024-01-01 before:2025-01-01 news")
    assert pq.operators.date_from == "2024-01-01"
    assert pq.operators.date_to == "2025-01-01"


def test_transactional_intent_detected():
    proc = QueryProcessor(expand_synonyms=False)
    pq = proc.process("buy cheap wireless headphones")
    assert pq.intent == "transactional"


def test_informational_intent_is_default():
    proc = QueryProcessor(expand_synonyms=False)
    pq = proc.process("how does gradient descent work")
    assert pq.intent == "informational"
    assert pq.is_question is True


def test_normalisation_lowercases_and_collapses_whitespace():
    proc = QueryProcessor(expand_synonyms=False)
    pq = proc.process("  Flash   ATTENTION   Transformers  ")
    assert pq.normalised == "flash attention transformers"


def test_expansion_adds_synonyms_when_enabled():
    proc = QueryProcessor(expand_synonyms=True, max_expansion=2)
    pq = proc.process("happy")
    assert len(pq.expanded) >= len(pq.normalised)


def test_expansion_disabled_leaves_query_unchanged():
    proc = QueryProcessor(expand_synonyms=False)
    pq = proc.process("happy")
    assert pq.expanded == pq.normalised
