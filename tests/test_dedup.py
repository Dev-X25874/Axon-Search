from __future__ import annotations

from utils.dedup import DedupFilter


def test_first_occurrence_is_not_duplicate():
    dedup = DedupFilter()
    text = "The quick brown fox jumps over the lazy dog. " * 5
    assert dedup.is_duplicate(text) is False
    assert len(dedup) == 1


def test_exact_repeat_is_duplicate():
    dedup = DedupFilter()
    text = "The quick brown fox jumps over the lazy dog. " * 5
    dedup.is_duplicate(text)
    assert dedup.is_duplicate(text) is True
    # A duplicate should not grow the index.
    assert len(dedup) == 1


def test_unrelated_texts_are_not_duplicates():
    dedup = DedupFilter()
    a = "Flash attention reduces memory usage in transformer training. " * 5
    b = "Sourdough bread requires a long fermentation with wild yeast. " * 5
    dedup.is_duplicate(a)
    assert dedup.is_duplicate(b) is False
    assert len(dedup) == 2


def test_near_duplicate_with_minor_edit_is_caught():
    dedup = DedupFilter(threshold=0.7)
    base = "Attention mechanisms let transformers weigh input tokens by relevance. " * 4
    near = base.replace("relevance", "importance")  # small edit, still ~overlapping shingles
    dedup.is_duplicate(base)
    assert dedup.is_duplicate(near) is True


def test_similarity_is_symmetric_and_bounded():
    dedup = DedupFilter()
    a = "hello world this is a test document about search engines"
    b = "hello world this is a test document about search engines"
    sim = dedup.similarity(a, b)
    assert 0.0 <= sim <= 1.0
    assert sim == dedup.similarity(b, a)


def test_reset_clears_index():
    dedup = DedupFilter()
    dedup.is_duplicate("some content to fingerprint here")
    assert len(dedup) == 1
    dedup.reset()
    assert len(dedup) == 0
