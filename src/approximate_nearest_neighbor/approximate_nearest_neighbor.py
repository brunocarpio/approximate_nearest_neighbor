from typing import Any

import numpy as np
from numpy import float32
from sortedcontainers import SortedSet


class Vertex:
    def __init__(self, id: Any, vector: np.ndarray):
        self.id = id
        self.vector = np.asarray(vector, dtype=np.float32)
        self.neighbors: set[Vertex] = set()

    def distance_to(self, other: np.ndarray) -> float32:
        return np.linalg.norm(self.vector - other)

    def connect(self, other: "Vertex"):
        self.neighbors.add(other)


class NSMWGraph:
    def __init__(self, ef_search=100, max_neighbors=16):
        self.vertices: dict[Any, Vertex] = {}
        self.ef_search = ef_search
        self.max_neighbors = max_neighbors

    def _random_entry(self) -> Any:
        key = np.random.choice(list(self.vertices.keys()))
        return self.vertices.get(key)

    def insert(self, v: Vertex) -> None:
        if not self.vertices:
            self.vertices[v.id] = v

        m = self.ef_search
        k = self.max_neighbors

        v_friends = self.search(v.vector, m, k)
        for _, friend_id in v_friends:
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
            v_distance = v_entry.distance_to(q)

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
                        friend_distance = friend.distance_to(q)

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
