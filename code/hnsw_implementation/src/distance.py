import numpy as np



def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity. Ranges [0, 2]."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 1.0
    return 1.0 - dot / norm



#certain other of these distance functions for testing purposes, but they are not used in the HNSW implementation itself. They are here for completeness and to provide a basis for comparison.

def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """The overlap between two vectors."""
    return np.dot(a, b)


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """The straight-line (Euclidean) distance between two points."""
    return np.linalg.norm(a - b)

# Example usage:
# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])

# print(f"Dot Product: {dot_product_np(a, b)}")
# print(f"Cosine Similarity: {cosine_similarity_np(a, b):.4f}")
# print(f"L2 Distance: {l2_distance_np(a, b):.4f}")