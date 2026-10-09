from enum import Enum
from typing import Any

import numpy as np
from numpy import float32
from sortedcontainers import SortedSet


class Space(Enum):
    L2 = "l2"


class Vertex:
    def __init__(self, id: Any, vector: np.ndarray):
        self.id = id
        self.vector = np.asarray(vector, dtype=np.float32)
        self.neighbors: set[Vertex] = set()

    def connect(self, other: "Vertex"):
        self.neighbors.add(other)


class NSMWGraph:
    def __init__(self, w=20, f=10, space=Space.L2):
        self.vertices: dict[Any, Vertex] = {}
        self.w = w
        self.f = f
        self.space = space

    def _random_entry(self) -> Any:
        key = np.random.choice(list(self.vertices.keys()))
        return self.vertices.get(key)

    def _distance_function(self, u: np.ndarray, v: np.ndarray):
        match self.space:
            case Space.L2:
                return np.linalg.norm(u - v)
            case _:
                raise ValueError("Unknown space")

    def insert(self, v: Vertex) -> None:
        if not self.vertices:
            self.vertices[v.id] = v

        v_friends = self.search(v.vector, m=self.w, k=self.f)

        for i in range(min(self.f, len(v_friends))):
            _, friend_id = v_friends[i]
            friend = self.vertices[friend_id]
            friend.connect(v)
            v.connect(friend)

        self.vertices[v.id] = v

    def search(self, q: np.ndarray, m: int, k: int) -> list[tuple[float32, Any]]:
        if not self.vertices:
            raise ValueError("cannot search an empty graph")

        temp_result = SortedSet(key=lambda x: x[0])
        candidates = SortedSet(key=lambda x: x[0])
        visited = set()
        result = SortedSet(key=lambda x: x[0])

        for _ in range(m):
            v_entry = self._random_entry()
            v_distance = self._distance_function(v_entry.vector, q)

            candidates.add((v_distance, v_entry.id))
            visited.add(v_entry.id)

            while candidates:
                closest_d, closest_id = candidates.pop(0)

                if len(temp_result) >= k and closest_d >= temp_result[-1][0]:
                    break

                temp_result.add((closest_d, closest_id))

                c = self.vertices[closest_id]

                for friend in c.neighbors:
                    if friend.id not in visited:
                        visited.add(friend.id)
                        friend_distance = self._distance_function(friend.vector, q)

                        candidates.add((friend_distance, friend.id))
                        temp_result.add((friend_distance, friend.id))

            result.update(temp_result)

        return list(result)[:k]

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
    n_samples = 500
    dimension = 50
    data = np.random.randn(n_samples, dimension)

    graph = NSMWGraph(w=20, f=10, space=Space.L2)

    print(f"Inserting {n_samples} vectors...")
    for i in range(n_samples):
        vertex = Vertex(i, data[i])
        graph.insert(vertex)

    print(f"Vertices: {graph.vertex_count()}")
    print(f"Edges: {graph.edge_count()}")

    print("\nAdjency:")
    print(graph.degree_statistics())
