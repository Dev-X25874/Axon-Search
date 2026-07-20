from __future__ import annotations

from crawler.link_graph import LinkGraph


def test_node_and_edge_counts():
    g = LinkGraph()
    g.add_link("http://a.test", "http://b.test")
    g.add_link("http://a.test", "http://c.test")
    assert g.node_count() == 3
    assert g.edge_count() == 2


def test_duplicate_edges_not_double_counted():
    g = LinkGraph()
    g.add_link("http://a.test", "http://b.test")
    g.add_link("http://a.test", "http://b.test")
    assert g.edge_count() == 1


def test_pagerank_sums_to_approximately_one():
    g = LinkGraph(iterations=50)
    g.add_link("http://a.test", "http://b.test")
    g.add_link("http://b.test", "http://c.test")
    g.add_link("http://c.test", "http://a.test")
    g.compute_pagerank()
    total = sum(g._scores.values())
    assert abs(total - 1.0) < 1e-3


def _build_popular_vs_unpopular_graph() -> LinkGraph:
    """
    Three equal-rank sources (x1, x2, x3) link to 'popular'; one
    equal-rank source (y1) links to 'unpopular'. Every node also links
    to a shared 'sink' (which links back to x1) so nothing is a
    dangling node -- isolating the effect of inlink count rather than
    dangling-mass redistribution, which otherwise dominates in very
    small graphs.
    """
    g = LinkGraph()
    for x in ("x1", "x2", "x3"):
        g.add_link(f"http://{x}.test", "http://popular.test")
        g.add_link(f"http://{x}.test", "http://sink.test")
    g.add_link("http://y1.test", "http://unpopular.test")
    g.add_link("http://y1.test", "http://sink.test")
    g.add_link("http://popular.test", "http://sink.test")
    g.add_link("http://unpopular.test", "http://sink.test")
    g.add_link("http://sink.test", "http://x1.test")
    return g


def test_more_inlinks_yields_higher_pagerank():
    g = _build_popular_vs_unpopular_graph()
    g.compute_pagerank()
    assert g.get_score("http://popular.test") > g.get_score("http://unpopular.test")


def test_unknown_url_has_zero_score():
    g = LinkGraph()
    g.add_link("http://a.test", "http://b.test")
    assert g.get_score("http://never-seen.test") == 0.0


def test_log_score_is_monotonic_with_raw_score():
    g = _build_popular_vs_unpopular_graph()
    g.compute_pagerank()
    assert g.log_score("http://popular.test") > g.log_score("http://unpopular.test")


def test_to_dict_from_dict_roundtrip():
    g = LinkGraph()
    g.add_link("http://a.test", "http://b.test")
    g.add_link("http://b.test", "http://a.test")
    g.compute_pagerank()

    data = g.to_dict()
    g2 = LinkGraph.from_dict(data)

    assert g2.node_count() == g.node_count()
    assert g2.edge_count() == g.edge_count()
    assert abs(g2.get_score("http://a.test") - g.get_score("http://a.test")) < 1e-9
