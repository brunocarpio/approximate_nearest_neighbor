from enum import Enum
from typing import Any

import numpy as np
from numpy import float32
from sortedcontainers import SortedSet


class Space(Enum):
    L2 = "l2"
    IP = "ip"
    COSINE = "cosine"


class Vertex:
    def __init__(self, id: Any, vector: np.ndarray):
        self.id = id
        self.vector = np.asarray(vector, dtype=np.float32)
        self.neighbors: set[Vertex] = set()

    def connect(self, other: "Vertex"):
        self.neighbors.add(other)


class NSMWGraph:
    def __init__(self, w=20, f=10, space=Space.L2, seed=None):
        self.vertices: dict[Any, Vertex] = {}
        self._ids: list[Any] = []
        self.w = w
        self.f = f
        self.space = space
        self.rng = np.random.default_rng(seed)

    def _random_entry(self) -> Vertex:
        return self.vertices[self._ids[self.rng.integers(len(self._ids))]]

    def _distance_function(self, u: np.ndarray, v: np.ndarray):
        match self.space:
            case Space.L2:
                return np.linalg.norm(u - v)
            case Space.IP:
                return 1.0 - np.inner(u, v)
            case Space.COSINE:
                norm_u = np.linalg.norm(u)
                norm_v = np.linalg.norm(v)
                similarity = np.inner(u, v) / (norm_u * norm_v)
                return 1.0 - similarity
            case _:
                raise ValueError("Unknown space")

    def insert(self, v: Vertex) -> None:
        if not self.vertices:
            self.vertices[v.id] = v
            self._ids.append(v.id)
            return

        v_friends = self.search(v.vector, m=self.w, k=self.f)

        for i in range(min(self.f, len(v_friends))):
            _, friend_id, _ = v_friends[i]
            friend = self.vertices[friend_id]
            friend.connect(v)
            v.connect(friend)

        self.vertices[v.id] = v
        self._ids.append(v.id)

    def _reconstruct_path(self, parent: dict, vid: Any) -> list:
        if vid is None:
            return []

        return self._reconstruct_path(parent, parent.get(vid)) + [vid]

    def search(self, q: np.ndarray, m: int, k: int) -> list[tuple[float32, Any, Any]]:
        if not self.vertices:
            raise ValueError("cannot search an empty graph")

        result = SortedSet(key=lambda x: x[0])
        paths = {}

        for _ in range(m):
            temp_result = SortedSet(key=lambda x: x[0])
            candidates = SortedSet(key=lambda x: x[0])
            visited = set()
            parent = {}

            v_entry = self._random_entry()
            v_distance = self._distance_function(v_entry.vector, q)

            candidates.add((v_distance, v_entry.id))
            visited.add(v_entry.id)

            parent[v_entry.id] = None

            while candidates:
                closest_d, closest_id = candidates.pop(0)

                if len(temp_result) >= k and closest_d >= temp_result[-1][0]:
                    break

                temp_result.add((closest_d, closest_id))

                if len(temp_result) > k:
                    temp_result.pop()

                c = self.vertices[closest_id]

                for friend in c.neighbors:
                    if friend.id not in visited:
                        visited.add(friend.id)
                        friend_distance = self._distance_function(friend.vector, q)

                        candidates.add((friend_distance, friend.id))
                        parent[friend.id] = closest_id

            for _, vid in temp_result:
                paths[vid] = self._reconstruct_path(parent, vid)

            result.update(temp_result)

        return [(vd, vid, paths[vid]) for vd, vid in list(result)[:k]]

    def vertex_count(self) -> int:
        return len(self.vertices)

    def edge_count(self) -> int:
        return sum(len(v.neighbors) for v in self.vertices.values()) // 2

    def degree_statistics(self) -> dict[str, Any]:
        degrees = [len(v.neighbors) for v in self.vertices.values()]
        return {
            "avg": float(np.mean(degrees)),
            "min": float(np.min(degrees)),
            "max": float(np.max(degrees)),
            "std": float(np.std(degrees)),
        }


if __name__ == "__main__":
    n_samples = 20
    dimension = 4

    seed = 42
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_samples, dimension))

    graph = NSMWGraph(w=20, f=12, space=Space.L2, seed=seed)

    q = np.random.randn(1, dimension)[0]

    print(f"Inserting {n_samples} vectors...")
    for i in range(n_samples):
        vertex = Vertex(i, data[i])
        graph.insert(vertex)
        print(
            f"Inserted:   ID: {vertex.id}, distance to q: {graph._distance_function(q, vertex.vector)}"
        )

    print(f"Vertices: {graph.vertex_count()}")
    print(f"Edges: {graph.edge_count()}")

    print("\nAdjency:")
    print(graph.degree_statistics())

    print("\nSearch from q:")
    results = graph.search(q, m=20, k=3)
    print(f"Found {len(results)} nearest neighbors")
    for vd, vid, hops in results:
        print(f"    ID: {vid}, distance: {vd}, hops: {hops}")
