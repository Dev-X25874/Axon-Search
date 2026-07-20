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
