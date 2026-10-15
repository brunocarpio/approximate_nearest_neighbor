"""Regression tests for NSMWGraph (review items 1.1, 1.2 and 1.3).

1.1 -- ``insert`` must not connect the very first vertex to itself.
1.2 -- the ``m``-start ``search`` must not carry state from one start into
       the next, which makes extra entry points *worse* than fewer.
1.3 -- ``_random_entry`` must work for any hashable id, not just scalars,
       and must pick an entry without scanning the whole vertex set
       (otherwise insertion/search degrade to O(n^2)).
2.1 -- ``search`` must not advance the graph's construction RNG, so an extra
       query cannot change the graph that later ``insert`` calls build.
4.1 -- cosine distance must be finite for a zero vector, not NaN.

These are expected to fail until the defects are fixed.
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
        monkeypatch.setattr(graph, "_random_entry", lambda *_: queued.pop(0))
        return graph.search(query, m=len(entry_ids), k=5)

    alone = search_from([359])
    both = search_from([71, 359])

    best_alone = min(distance for distance, _, _ in alone)
    best_both = min(distance for distance, _, _ in both)

    assert best_both <= best_alone


def test_insert_and_search_support_non_scalar_ids():
    graph = NSMWGraph(w=4, f=2, seed=0)
    for i, vertex_id in enumerate([(0, 0), (1, 1), (2, 2)]):
        graph.insert(Vertex(vertex_id, np.full(2, i)))

    results = graph.search(np.array([0.5, 0.5]), m=3, k=2)

    assert len(results) == 2


class CountingDict(dict):
    """A dict that records how many keys have been enumerated."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scanned = 0

    def keys(self):
        self.scanned += len(self)
        return super().keys()

    def __iter__(self):
        self.scanned += len(self)
        return super().__iter__()


def test_random_entry_does_not_scan_all_vertices(monkeypatch):
    graph = make_graph(n=100)
    vertices = CountingDict(graph.vertices)
    monkeypatch.setattr(graph, "vertices", vertices)

    graph._random_entry()

    assert vertices.scanned <= 1


def test_search_does_not_advance_construction_rng():
    graph = make_graph(n=100)
    state = graph.rng.bit_generator.state

    graph.search(graph.vertices[0].vector, m=5, k=5)

    assert graph.rng.bit_generator.state == state


def test_search_does_not_change_graph_construction():
    seed, n = 7, 120
    vectors = np.random.default_rng(seed).standard_normal((n, 4))

    def build(with_query):
        graph = NSMWGraph(w=10, f=3, space=Space.L2, seed=seed)
        for i, vector in enumerate(vectors):
            if with_query and i == n // 2:
                graph.search(vectors[0], m=5, k=5)
            graph.insert(Vertex(i, vector))
        return graph

    def edges(graph):
        return {
            (min(v.id, nb.id), max(v.id, nb.id))
            for v in graph.vertices.values()
            for nb in v.neighbors
        }

    assert edges(build(with_query=False)) == edges(build(with_query=True))


def test_cosine_distance_is_finite_for_zero_vectors():
    graph = NSMWGraph(w=4, f=2, space=Space.COSINE, seed=0)
    graph.insert(Vertex(0, np.zeros(2)))
    graph.insert(Vertex(1, np.array([1.0, 0.0])))

    results = graph.search(np.zeros(2), m=2, k=2)

    assert all(np.isfinite(distance) for distance, _, _ in results)
