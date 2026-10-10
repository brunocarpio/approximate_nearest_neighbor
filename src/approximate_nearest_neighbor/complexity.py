import matplotlib.pyplot as plt
import numpy as np

from approximate_nearest_neighbor.graph import NSMWGraph, Space, Vertex


def avg_hops(g, q):
    return np.mean([len(path) - 1 for _, _, path in g.search(q, m=m, k=k)])


if __name__ == "__main__":
    sizes = [100, 150, 250, 300, 400, 500, 600]
    m, k, queries = 20, 10, 5

    dim_styles = {
        5: {"color": "blue", "marker": "^"},
        10: {"color": "magenta", "marker": "v"},
        20: {"color": "green", "marker": "D"},
        50: {"color": "navy", "marker": "<"},
        100: {"color": "black", "marker": "s"},
    }

    _, ax = plt.subplots(figsize=(7.2, 4.8))

    for dim, style in dim_styles.items():
        ys = []
        for size in sizes:
            rng = np.random.default_rng(42)
            data = rng.standard_normal((size, dim))

            g = NSMWGraph(w=m, f=min(dim * 3, 32), space=Space.L2)
            for i, v in enumerate(data):
                g.insert(Vertex(i, v))

            hops = [avg_hops(g, data[j]) for j in range(min(queries, size))]
            ys.append(np.mean(hops))

        ax.plot(
            sizes,
            ys,
            label=f"d={dim}",
            color=style["color"],
            marker=style["marker"],
            linewidth=1.2,
            markersize=5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Average hop count")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=True, fancybox=False, edgecolor="black")
    plt.tight_layout()
    plt.show()
