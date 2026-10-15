"""Regression tests for NSMWGraph (review items 1.1 and 1.2).

1.1 -- ``insert`` must not connect the very first vertex to itself.
1.2 -- the ``m``-start ``search`` must not carry state from one start into
       the next, which makes extra entry points *worse* than fewer.

These are expected to fail until the two defects are fixed.
"""

import numpy as np

from approximate_nearest_neighbor.graph import NSMWGraph, Space, Vertex


def make_graph(n=400, seed=11):
    """A deterministic L2 graph built from ``n`` 4-D normal samples."""
    graph = NSMWGraph(w=10, f=3, space=Space.L2, seed=seed)
    vectors = np.random.default_rng(seed).standard_normal((n, 4))
    for i, vector in enumerate(vectors):
        graph.insert(Vertex(i, vector))
    return graph


def test_first_insert_has_no_self_loop():
    graph = NSMWGraph(w=4, f=2, seed=0)
    vertex = Vertex(0, np.zeros(2))

    graph.insert(vertex)

    assert vertex.neighbors == set()


def test_single_vertex_has_zero_degree():
    graph = NSMWGraph(w=4, f=2, seed=0)

    graph.insert(Vertex(0, np.zeros(2)))

    assert graph.degree_statistics()["max"] == 0.0


def test_multi_start_search_does_not_leak_state(monkeypatch):
    graph = make_graph()
    query = np.random.default_rng(2).standard_normal(4)

    def search_from(entry_ids):
        queued = [graph.vertices[i] for i in entry_ids]
        monkeypatch.setattr(graph, "_random_entry", lambda: queued.pop(0))
        return graph.search(query, m=len(entry_ids), k=5)

    alone = search_from([359])
    both = search_from([71, 359])

    best_alone = min(distance for distance, _, _ in alone)
    best_both = min(distance for distance, _, _ in both)

    assert best_both <= best_alone
