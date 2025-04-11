import torch

def knn_search(query, data, k):
    distances = torch.cdist(query, data)
    top_k_dists, top_k_indices = torch.topk(distances, k, largest=False)
    return top_k_dists, top_k_indices

# Set up the tensor of 10000 5-dimensional vectors
N = 10000
dim = 5
data = torch.randn(N, dim)  # Create a tensor of random vectors with shape (10000, 5)

# Perform kNN search for all vectors in the dataset
k = 10  # Let's say we want the 5 nearest neighbors for each vector

import time
for i in range(10):
    start = time.perf_counter()
    top_k_dists, top_k_indices = knn_search(data, data, k)
    end = time.perf_counter()
    print("Time taken for kNN search:", end - start)

# top_k_dists and top_k_indices now contain the k-nearest distances and indices for each vector in the dataset
print("Top K Distances:", top_k_dists)
print("Top K Indices:", top_k_indices)