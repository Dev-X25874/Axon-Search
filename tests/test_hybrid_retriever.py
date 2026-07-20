from __future__ import annotations

from search.hybrid_retriever import HybridRetriever, SearchResult
from search.query_processor import QueryProcessor


class _FakeBM25:
    """Returns a fixed candidate list regardless of query text."""

    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k=100):
        return self._hits[:top_k]


class _FakeVectorStore:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query_vec, top_k=100):
        return self._hits[:top_k]


class _FakeEmbedder:
    def encode_query(self, text):
        return [0.0]  # value is irrelevant; VectorStore is faked too


class _FakeLinkGraph:
    def __init__(self, scores: dict[str, float]):
        self._scores = scores

    def log_score(self, url: str) -> float:
        return self._scores.get(url, 0.0)


def _pq(query: str = "attention transformer"):
    return QueryProcessor(expand_synonyms=False).process(query)


def _make_retriever(bm25_hits, dense_hits, link_graph=None, **kwargs):
    return HybridRetriever(
        bm25=_FakeBM25(bm25_hits),
        vector_store=_FakeVectorStore(dense_hits),
        embedder=_FakeEmbedder(),
        link_graph=link_graph,
        **kwargs,
    )


def test_doc_in_both_rankers_outranks_doc_in_one():
    bm25_hits = [(1, 5.0, {"url": "http://a.test", "title": "A"}),
                 (2, 4.0, {"url": "http://b.test", "title": "B"})]
    dense_hits = [(1, 0.9, {"url": "http://a.test", "title": "A"})]

    retriever = _make_retriever(bm25_hits, dense_hits)
    results = retriever.retrieve(_pq(), top_k=10)

    assert results[0].doc_id == 1  # appears in both rankers -> higher RRF score
    assert results[0].bm25_rank == 1
    assert results[0].dense_rank == 1
    assert results[1].doc_id == 2
    assert results[1].dense_rank is None


def test_rrf_fuse_combines_disjoint_hit_sets():
    bm25_hits = [(1, 5.0, {"url": "http://a.test", "title": "A"})]
    dense_hits = [(2, 0.8, {"url": "http://b.test", "title": "B"})]

    retriever = _make_retriever(bm25_hits, dense_hits)
    results = retriever.retrieve(_pq(), top_k=10)

    doc_ids = {r.doc_id for r in results}
    assert doc_ids == {1, 2}


def test_site_filter_excludes_non_matching_urls():
    bm25_hits = [
        (1, 5.0, {"url": "http://arxiv.org/paper1", "title": "A"}),
        (2, 4.0, {"url": "http://other.test/paper2", "title": "B"}),
    ]
    retriever = _make_retriever(bm25_hits, [])
    results = retriever.retrieve(_pq("site:arxiv.org attention"), top_k=10)

    assert len(results) == 1
    assert results[0].doc_id == 1


def test_exclude_term_filters_out_matching_title():
    bm25_hits = [
        (1, 5.0, {"url": "http://a.test", "title": "Vision transformers"}),
        (2, 4.0, {"url": "http://b.test", "title": "Attention is all you need"}),
    ]
    retriever = _make_retriever(bm25_hits, [])
    results = retriever.retrieve(_pq("attention -vision"), top_k=10)

    doc_ids = {r.doc_id for r in results}
    assert 1 not in doc_ids
    assert 2 in doc_ids


def test_extra_metadata_filters_applied():
    bm25_hits = [
        (1, 5.0, {"url": "http://a.test", "title": "A", "language": "en"}),
        (2, 4.0, {"url": "http://b.test", "title": "B", "language": "fr"}),
    ]
    retriever = _make_retriever(bm25_hits, [])
    results = retriever.retrieve(_pq(), top_k=10, filters={"language": "en"})

    assert len(results) == 1
    assert results[0].doc_id == 1


def test_pagerank_boost_reorders_results():
    bm25_hits = [
        (1, 5.0, {"url": "http://low-authority.test", "title": "A"}),
        (2, 5.0, {"url": "http://high-authority.test", "title": "B"}),
    ]
    dense_hits = [
        (1, 0.9, {"url": "http://low-authority.test", "title": "A"}),
        (2, 0.9, {"url": "http://high-authority.test", "title": "B"}),
    ]
    link_graph = _FakeLinkGraph({
        "http://low-authority.test": 0.0,
        "http://high-authority.test": 10.0,
    })
    retriever = _make_retriever(bm25_hits, dense_hits, link_graph=link_graph, pagerank_alpha=0.5)
    results = retriever.retrieve(_pq(), top_k=10)

    assert results[0].url == "http://high-authority.test"


def test_final_score_prefers_rerank_score_when_present():
    r = SearchResult(doc_id=1, url="http://a.test", title="A", score=0.1, rrf_score=0.1)
    assert r.final_score == 0.1
    r.rerank_score = 0.9
    assert r.final_score == 0.9


def test_top_k_truncates_results():
    bm25_hits = [(i, float(10 - i), {"url": f"http://{i}.test", "title": str(i)}) for i in range(20)]
    retriever = _make_retriever(bm25_hits, [])
    results = retriever.retrieve(_pq(), top_k=3)
    assert len(results) == 3
