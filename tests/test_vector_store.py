from __future__ import annotations

from pathlib import Path

import numpy as np

from indexer.vector_store import VectorStore


def _unit_vec(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_add_and_search_flat_returns_nearest():
    store = VectorStore(dim=8, index_type="flat")
    v0, v1 = _unit_vec(0), _unit_vec(1)
    store.add(0, v0, {"url": "http://a.test"})
    store.add(1, v1, {"url": "http://b.test"})

    results = store.search(v0, top_k=2)
    assert results
    assert results[0][0] == 0  # querying with v0 should return doc 0 first


def test_add_batch_matches_len():
    store = VectorStore(dim=8, index_type="flat")
    vecs = np.stack([_unit_vec(i) for i in range(5)])
    store.add_batch(list(range(5)), vecs, [{"i": i} for i in range(5)])
    assert len(store) == 5


def test_search_empty_store_returns_empty_list():
    store = VectorStore(dim=8, index_type="flat")
    assert store.search(_unit_vec(0), top_k=5) == []


def test_save_and_load_roundtrip(tmp_path: Path):
    store = VectorStore(dim=8, index_type="flat")
    v0 = _unit_vec(0)
    store.add(0, v0, {"url": "http://a.test"})

    out_dir = tmp_path / "vectors"
    store.save(out_dir)
    loaded = VectorStore.load(out_dir, dim=8, index_type="flat")

    assert len(loaded) == 1
    results = loaded.search(v0, top_k=1)
    assert results[0][0] == 0


# ---------------------------------------------------------------------------
# Soft-delete
# ---------------------------------------------------------------------------

def test_delete_removes_doc_from_search_results():
    store = VectorStore(dim=8, index_type="flat")
    v0, v1 = _unit_vec(0), _unit_vec(1)
    store.add(0, v0, {"url": "http://a.test"})
    store.add(1, v1, {"url": "http://b.test"})

    assert store.delete(0) is True
    results = store.search(v0, top_k=2)
    doc_ids = {doc_id for doc_id, _, _ in results}
    assert 0 not in doc_ids


def test_delete_unknown_doc_id_returns_false():
    store = VectorStore(dim=8, index_type="flat")
    store.add(0, _unit_vec(0), {})
    assert store.delete(999) is False


def test_delete_twice_returns_false_second_time():
    store = VectorStore(dim=8, index_type="flat")
    store.add(0, _unit_vec(0), {})
    assert store.delete(0) is True
    assert store.delete(0) is False


def test_deleted_count_tracks_deletions():
    store = VectorStore(dim=8, index_type="flat")
    store.add(0, _unit_vec(0), {})
    store.add(1, _unit_vec(1), {})
    assert store.deleted_count() == 0
    store.delete(0)
    assert store.deleted_count() == 1


def test_save_load_roundtrip_preserves_deletions(tmp_path: Path):
    store = VectorStore(dim=8, index_type="flat")
    store.add(0, _unit_vec(0), {"url": "http://a.test"})
    store.add(1, _unit_vec(1), {"url": "http://b.test"})
    store.delete(1)

    out_dir = tmp_path / "vectors"
    store.save(out_dir)
    loaded = VectorStore.load(out_dir, dim=8, index_type="flat")

    assert loaded.is_deleted(1) is True
    assert loaded.deleted_count() == 1
