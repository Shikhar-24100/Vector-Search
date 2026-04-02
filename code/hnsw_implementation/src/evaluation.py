import numpy as np
from .distance import cosine_distance

def compute_recall(index, queries: np.ndarray, ground_truth: list[list[int]], k: int = 10, ef: int = 50) -> float:
    hits = 0
    total = 0
    for i, q in enumerate(queries):
        results = index.search(q, k=k, ef=ef)
        found_ids = {idx for (_, idx) in results}
        true_ids = set(ground_truth[i][:k])
        hits += len(found_ids & true_ids)
        total += k
    return hits / total if total > 0 else 0.0

def brute_force_knn(vectors: np.ndarray, queries: np.ndarray, k: int) -> list[list[int]]:
    results = []
    for q in queries:
        dists = [cosine_distance(q, v) for v in vectors]
        sorted_ids = sorted(range(len(dists)), key=lambda i: dists[i])
        results.append(sorted_ids[:k])
    return results


