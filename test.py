import torch
import parallelrm_cuda

def test_prm():
    # Create sample environment
    circles = torch.tensor([
        [2.0, 2.0, 0.5],  # x, y, radius
        [8.0, 8.0, 1.0],
    ], dtype=torch.float32, device='cuda')
    
    rectangles = torch.tensor([
        [4.0, 4.0, 1.0, 2.0],  # x, y, width, height
        [6.0, 2.0, 0.5, 1.5],
    ], dtype=torch.float32, device='cuda')
    
    # Define bounds [lower, upper] for [x, y, theta, pan, tilt]
    bounds = torch.tensor([
        [0.0, 0.0, -3.14159, 0.0, 0.0],  # lower bounds
        [10.0, 10.0, 3.14159, 0.0, 0.0]  # upper bounds
    ], dtype=torch.float32, device='cuda')
    
    # Build PRM
    nodes, node_validity, neighbors, edges, edge_validity = parallelrm_cuda.build_prm(
        circles, rectangles, bounds, num_states=1000, k=5, seed=12345
    )
    
    print(f"Nodes shape: {nodes.shape}")
    print(f"Node validity shape: {node_validity.shape}")
    print(f"Neighbors shape: {neighbors.shape}")
    print(f"Edges shape: {edges.shape}")
    print(f"Edge validity shape: {edge_validity.shape}")
    
    print(f"Valid nodes: {node_validity.sum().item()}/{len(node_validity)}")

if __name__ == "__main__":
    test_prm()

# cd /home/lenman/capstone/parallelrm
# python setup.py build_ext --inplace
# python test_bindings.py