import numpy as np
import random
import os
import pickle

from src.index import HNSW
from src.evaluation import compute_recall, brute_force_knn

def run_experiment():
    np.random.seed(42)
    random.seed(42)

    DIM = 128
    N_VECTORS = 10_000 
    N_QUERIES = 100
    K = 10

    print(f"Generating {N_VECTORS} random {DIM}-dim vectors...")
    data = np.random.randn(N_VECTORS, DIM).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)

    queries = np.random.randn(N_QUERIES, DIM).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    gt_filename = os.path.join("data", f"ground_truth_{N_VECTORS}.pkl")
    
    if os.path.exists(gt_filename):
        print(f"\nLoading cached ground truth from {gt_filename}...")
        with open(gt_filename, "rb") as f:
            ground_truth = pickle.load(f)
    else:
        print("\nComputing ground truth (brute force). This might take a minute...")
        ground_truth = brute_force_knn(data, queries, K)
        #ensure data directory exists
        os.makedirs("data", exist_ok=True)
        with open(gt_filename, "wb") as f:
            pickle.dump(ground_truth, f)
        print(f"Saved ground truth to {gt_filename}")

    # Test M Values
    m_values_to_test = [8, 16, 32]
    
    for test_m in m_values_to_test:
        print(f"\n========================================")
        print(f"Testing HNSW with M = {test_m}")
        print(f"========================================")
        
        index = HNSW(M=test_m, ef_construction=200)
        for i, vec in enumerate(data):
            index.add(vec)
            
        print(f"Index built! Max layer reached: {index.max_layer}")
        
        print("\nEvaluating recall@10 at different ef values:")
        print(f"{'ef':>6}  {'recall@10':>10}")
        print("-" * 20)
        for ef in [10, 20, 50, 100, 200]:
            recall = compute_recall(index, queries, ground_truth, k=K, ef=ef)
            print(f"{ef:>6}  {recall:>10.3f}")

if __name__ == "__main__":
    run_experiment()