import numpy as np

from approximate_nearest_neighbor.approximate_nearest_neighbor import NSMWGraph, Vertex


def main():
    n_samples = 400
    dimension = 100
    data = np.random.randn(n_samples, dimension)

    graph = NSMWGraph(ef_search=50, max_neighbors=16)

    print(f"Inserting {n_samples} vectors...")
    for i in range(n_samples):
        vertex = Vertex(i, data[i])
        graph.insert(vertex)

    print(f"Vertices: {graph.vertex_count()}")
    print(f"Edges: {graph.edge_count()}")

    print("\nAdjency:")
    print(graph.degree_statistics())


def __init__():
    main()
