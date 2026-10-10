import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from approximate_nearest_neighbor.graph import NSMWGraph, Space, Vertex


def plot_search_path(
    graph: NSMWGraph,
    data: np.ndarray,
    q: np.ndarray,
    path: np.ndarray,
    title: str = "Search path",
) -> None:
    _, ax = plt.subplots(figsize=(7, 6))

    segments = []
    seen = set()
    for v in graph.vertices.values():
        v_xy = v.vector
        for nb in v.neighbors:
            edge = tuple(sorted((v.id, nb.id)))
            if edge in seen:
                continue
            seen.add(edge)
            segments.append([v_xy, nb.vector])

    if segments:
        lc = LineCollection(
            segments,
            colors="steelblue",
            linewidths=0.8,
            alpha=0.12,
            zorder=1,
        )
        ax.add_collection(lc)

    ax.scatter(
        data[:, 0], data[:, 1], s=25, alpha=0.7, label="data", c="gray", zorder=2
    )

    ax.scatter(q[0], q[1], c="orange", s=90, marker="x", label="query", zorder=6)

    ax.plot(
        path[:, 0],
        path[:, 1],
        "-o",
        c="blue",
        linewidth=2,
        markersize=5,
        label="hop path",
        zorder=5,
    )

    ax.scatter(path[0, 0], path[0, 1], c="red", s=80, label="v_entry", zorder=6)

    ax.set_title(f"{title} (hops: {len(path) - 1})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()


def main():
    seed = 42
    rng = np.random.default_rng(seed)

    n_samples = 200
    dimension = 2
    data = rng.standard_normal((n_samples, dimension))

    graph = NSMWGraph(w=20, f=8, space=Space.COSINE, seed=seed)
    for i, v in enumerate(data):
        graph.insert(Vertex(i, v))

    q = rng.standard_normal(dimension)

    results = graph.search(q, m=20, k=10)

    _, _, path_ids = results[0]
    path = np.array([graph.vertices[vid].vector for vid in path_ids])

    plot_search_path(graph, data, q, path)


if __name__ == "__main__":
    main()
