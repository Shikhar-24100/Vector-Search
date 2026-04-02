# hnsw.py
import numpy as np
import heapq
import math
import random
from typing import Optional
from .distance import cosine_distance
# import .distance from cosine_distance

class HNSW:
    def __init__(self, M: int = 16, ef_construction: int = 200):
        self.M = M
        self.M0 = M * 2
        self.ef_construction = ef_construction
        self.mL = 1.0 / math.log(M)

        self.vectors: list[np.ndarray] = []
        self.graphs: list[dict[int, list[int]]] = []
        self.entry_point: Optional[int] = None
        self.max_layer: int = -1

    def add(self, vector: np.ndarray) -> int:
        vector = np.array(vector, dtype=np.float32)
        node_id = len(self.vectors)
        self.vectors.append(vector)

        node_layer = self._random_layer()

        while len(self.graphs) <= node_layer:
            self.graphs.append({})

        if self.entry_point is None:
            for lc in range(node_layer + 1):
                self.graphs[lc][node_id] = []
            self.entry_point = node_id
            self.max_layer = node_layer
            return node_id

        ep = self.entry_point
        for lc in range(self.max_layer, node_layer, -1):
            ep = self._search_layer_greedy(vector, ep, lc)

        current_ep = ep
        for lc in range(min(node_layer, self.max_layer), -1, -1):
            candidates = self._search_layer(vector, current_ep, self.ef_construction, lc)
            m_at_layer = self.M0 if lc == 0 else self.M
            
            neighbours = self._select_neighbours_heuristic(vector, candidates, m_at_layer)
            self.graphs[lc][node_id] = neighbours

            for nb in neighbours:
                if nb not in self.graphs[lc]:
                    self.graphs[lc][nb] = []
                nb_neighbours = self.graphs[lc][nb]
                nb_neighbours.append(node_id)

                max_nb = self.M0 if lc == 0 else self.M
                if len(nb_neighbours) > max_nb:
                    self.graphs[lc][nb] = self._prune(self.vectors[nb], nb_neighbours, max_nb)

            if candidates:
                current_ep = heapq.nsmallest(1, candidates, key=lambda x: x[0])[0][1]

        if node_layer > self.max_layer:
            self.max_layer = node_layer
            self.entry_point = node_id

        return node_id

    def search(self, query: np.ndarray, k: int = 10, ef: int = 50) -> list[tuple[float, int]]:
        if self.entry_point is None:
            return []

        query = np.array(query, dtype=np.float32)
        ep = self.entry_point

        for lc in range(self.max_layer, 0, -1):
            ep = self._search_layer_greedy(query, ep, lc)

        candidates = self._search_layer(query, ep, ef, 0)
        result = sorted(candidates, key=lambda x: x[0])
        return result[:k]

    def __len__(self) -> int:
        return len(self.vectors)

    def _random_layer(self) -> int:
        return int(-math.log(random.random()) * self.mL)

    def _dist(self, a: np.ndarray, b_id: int) -> float:
        return cosine_distance(a, self.vectors[b_id])

    def _search_layer_greedy(self, query: np.ndarray, ep: int, layer: int) -> int:
        best = ep
        best_dist = self._dist(query, ep)
        changed = True
        while changed:
            changed = False
            for nb in self.graphs[layer].get(best, []):
                d = self._dist(query, nb)
                if d < best_dist:
                    best_dist = d
                    best = nb
                    changed = True
        return best

    def _search_layer(self, query: np.ndarray, ep: int, ef: int, layer: int) -> list[tuple[float, int]]:
        ep_dist = self._dist(query, ep)
        candidates = [(ep_dist, ep)]
        heapq.heapify(candidates)
        found = [(-ep_dist, ep)]
        heapq.heapify(found)
        visited = {ep}

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            worst_found_dist = -found[0][0]
            if c_dist > worst_found_dist:
                break

            for nb in self.graphs[layer].get(c_id, []):
                if nb in visited:
                    continue
                visited.add(nb)
                nb_dist = self._dist(query, nb)
                worst_found_dist = -found[0][0]

                if nb_dist < worst_found_dist or len(found) < ef:
                    heapq.heappush(candidates, (nb_dist, nb))
                    heapq.heappush(found, (-nb_dist, nb))
                    if len(found) > ef:
                        heapq.heappop(found)

        return [(-d, idx) for (d, idx) in found]

    def _select_neighbours_heuristic(self, query: np.ndarray, candidates: list[tuple[float, int]], m: int, keep_pruned: bool = True) -> list[int]:
        sorted_cands = sorted(candidates, key=lambda x: x[0])
        selected = []
        pruned = []

        for (d_q_e, e) in sorted_cands:
            if len(selected) >= m:
                break

            dominated = False
            for (_, r) in selected:
                d_e_r = cosine_distance(self.vectors[e], self.vectors[r])
                if d_e_r < d_q_e:
                    dominated = True
                    break

            if not dominated:
                selected.append((d_q_e, e))
            else:
                pruned.append((d_q_e, e))

        if keep_pruned:
            for p in pruned:
                if len(selected) >= m:
                    break
                selected.append(p)

        return [idx for (_, idx) in selected]

    def _prune(self, node_vec: np.ndarray, neighbours: list[int], max_m: int) -> list[int]:
        candidates = [(cosine_distance(node_vec, self.vectors[nb]), nb) for nb in neighbours]
        return self._select_neighbours_heuristic(node_vec, candidates, max_m)