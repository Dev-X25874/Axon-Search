from __future__ import annotations

from pathlib import Path

from indexer.bm25 import BM25Index


def test_add_returns_incrementing_doc_ids():
    idx = BM25Index()
    id0 = idx.add("first document text", {"url": "http://a.test"})
    id1 = idx.add("second document text", {"url": "http://b.test"})
    assert id0 == 0
    assert id1 == 1
    assert len(idx) == 2


def test_search_returns_relevant_doc_first(sample_docs):
    idx = BM25Index()
    for doc in sample_docs:
        idx.add(doc, {"url": f"http://{hash(doc) % 1000}.test", "text": doc})

    results = idx.search("flash attention memory transformer", top_k=5)
    assert results, "expected at least one hit"

    top_doc_id, top_score, _meta = results[0]
    # The flash-attention doc should outrank the unrelated cat/dog doc.
    assert "flash attention" in sample_docs[top_doc_id].lower()
    assert top_score > 0


def test_search_empty_index_returns_empty_list():
    idx = BM25Index()
    assert idx.search("anything") == []


def test_search_query_with_only_stopwords_returns_empty():
    idx = BM25Index()
    idx.add("some real content here", {})
    # "the a an" all get stripped by tokenisation -> no tokens -> no results
    assert idx.search("a an the") == []


def test_get_metadata_out_of_range_returns_empty_dict():
    idx = BM25Index()
    idx.add("doc", {"url": "http://x.test"})
    assert idx.get_metadata(999) == {}
    assert idx.get_metadata(0) == {"url": "http://x.test"}


def test_add_batch_matches_individual_adds(sample_docs):
    idx = BM25Index()
    ids = idx.add_batch(sample_docs, [{"i": i} for i in range(len(sample_docs))])
    assert ids == list(range(len(sample_docs)))
    assert len(idx) == len(sample_docs)


def test_save_and_load_roundtrip(tmp_path: Path, sample_docs):
    idx = BM25Index(k1=1.3, b=0.7)
    idx.add_batch(sample_docs, [{"url": f"http://doc{i}.test"} for i in range(len(sample_docs))])

    save_path = tmp_path / "bm25.pkl"
    idx.save(save_path)
    assert save_path.exists()

    loaded = BM25Index.load(save_path)
    assert len(loaded) == len(idx)
    assert loaded.k1 == 1.3
    assert loaded.b == 0.7

    # Same query should return the same top doc after reload.
    before = idx.search("attention transformer", top_k=1)
    after = loaded.search("attention transformer", top_k=1)
    assert before and after
    assert before[0][0] == after[0][0]


# ---------------------------------------------------------------------------
# Soft-delete
# ---------------------------------------------------------------------------

def test_delete_by_doc_id_removes_from_search_results(sample_docs):
    idx = BM25Index()
    for doc in sample_docs:
        idx.add(doc, {"url": f"http://{hash(doc) % 1000}.test", "text": doc})

    before = idx.search("flash attention memory transformer", top_k=5)
    top_doc_id = before[0][0]

    assert idx.delete(top_doc_id) is True
    after = idx.search("flash attention memory transformer", top_k=5)
    assert top_doc_id not in {doc_id for doc_id, _, _ in after}


def test_delete_unknown_doc_id_returns_false():
    idx = BM25Index()
    idx.add("doc", {})
    assert idx.delete(999) is False


def test_delete_twice_returns_false_second_time():
    idx = BM25Index()
    doc_id = idx.add("doc", {"url": "http://x.test"})
    assert idx.delete(doc_id) is True
    assert idx.delete(doc_id) is False


def test_delete_by_url_removes_correct_doc():
    idx = BM25Index()
    idx.add("first document", {"url": "http://a.test"})
    id_b = idx.add("second document", {"url": "http://b.test"})

    assert idx.delete_by_url("http://b.test") is True
    assert idx.is_deleted(id_b) is True
    assert idx.deleted_count() == 1


def test_delete_by_url_unknown_url_returns_false():
    idx = BM25Index()
    idx.add("doc", {"url": "http://a.test"})
    assert idx.delete_by_url("http://not-indexed.test") is False


def test_len_excludes_deleted_docs():
    idx = BM25Index()
    id0 = idx.add("first", {"url": "http://a.test"})
    idx.add("second", {"url": "http://b.test"})
    assert len(idx) == 2
    idx.delete(id0)
    assert len(idx) == 1


def test_save_load_roundtrip_preserves_deletions(tmp_path: Path):
    idx = BM25Index()
    idx.add("first document", {"url": "http://a.test"})
    id_b = idx.add("second document", {"url": "http://b.test"})
    idx.delete(id_b)

    path = tmp_path / "bm25.pkl"
    idx.save(path)
    loaded = BM25Index.load(path)

    assert loaded.is_deleted(id_b) is True
    assert loaded.deleted_count() == 1
    assert loaded.url_to_doc_id("http://a.test") == 0


# ---------------------------------------------------------------------------
# Autocomplete (suggest)
# ---------------------------------------------------------------------------

def test_suggest_returns_prefix_matches(sample_docs):
    idx = BM25Index()
    for doc in sample_docs:
        idx.add(doc, {})
    suggestions = idx.suggest("atten")
    assert "attention" in suggestions


def test_suggest_ranks_by_frequency():
    idx = BM25Index()
    idx.add("cat cat cat", {})
    idx.add("cat dog", {})
    idx.add("car", {})
    suggestions = idx.suggest("ca", limit=3)
    assert suggestions[0] == "cat"


def test_suggest_empty_prefix_returns_empty_list():
    idx = BM25Index()
    idx.add("some content", {})
    assert idx.suggest("") == []


def test_suggest_respects_limit():
    idx = BM25Index()
    idx.add("apple apricot avocado almond anchovy", {})
    assert len(idx.suggest("a", limit=2)) == 2
