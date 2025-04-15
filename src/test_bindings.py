import sys
import os
import torch

# Adjust this path if your build output is in a different location
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'build')))

# Import the CUDA module that was built
import pprm_cuda

# Test functions and objects defined in the bindings
def test_pprm_cuda():
    # Example of creating a roadmap
    roadmap = pprm_cuda.PyRoadmap()

    # Get the raw data (pointer) from C++ (assuming they are on the device)
    states_ptr = roadmap.d_states()
    edges_ptr = roadmap.d_edges()
    neighbors_ptr = roadmap.d_neighbors()
    valid_nodes_ptr = roadmap.d_valid_nodes()
    valid_edges_ptr = roadmap.d_valid_edges()
    print(f"States Pointer: {states_ptr}")
    print(f"Edges Pointer: {edges_ptr}")
    print(f"Neighbors Pointer: {neighbors_ptr}")
    print(f"Valid Nodes Pointer: {valid_nodes_ptr}")
    print(f"Valid Edges Pointer: {valid_edges_ptr}")
    # Convert raw pointers to torch tensors (assuming data is on GPU)
    states = torch.from_numpy(states_ptr).to(torch.device("cuda"))
    edges = torch.from_numpy(edges_ptr).to(torch.device("cuda"))
    neighbors = torch.from_numpy(neighbors_ptr).to(torch.device("cuda"))
    valid_nodes = torch.from_numpy(valid_nodes_ptr).to(torch.device("cuda"))
    valid_edges = torch.from_numpy(valid_edges_ptr).to(torch.device("cuda"))

    # Print out basic information about the tensors (just to check the bindings)
    print(f"States: {states}")
    print(f"Edges: {edges}")
    print(f"Neighbors: {neighbors}")
    print(f"Valid Nodes: {valid_nodes}")
    print(f"Valid Edges: {valid_edges}")

    # Ensure they are PyTorch tensors
    assert isinstance(states, torch.Tensor), "States should be a torch tensor"
    assert isinstance(edges, torch.Tensor), "Edges should be a torch tensor"
    assert isinstance(neighbors, torch.Tensor), "Neighbors should be a torch tensor"
    assert isinstance(valid_nodes, torch.Tensor), "Valid Nodes should be a torch tensor"
    assert isinstance(valid_edges, torch.Tensor), "Valid Edges should be a torch tensor"

    print("All tests passed successfully!")

# Run the test
if __name__ == "__main__":
    test_pprm_cuda()
