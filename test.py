import time
import torch
import parallelrm_cuda

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import torch
import numpy as np

def visualize_prm_from_tensors(nodes, node_validity, neighbors, edges, edge_validity, 
                              circles=None, rectangles=None, bounds=None):
    """
    Visualize PRM roadmap from torch tensors or numpy arrays.
    
    Args:
        nodes: [NUM_STATES, DIM] - node positions
        node_validity: [NUM_STATES] - boolean array of valid nodes
        neighbors: [NUM_STATES, K] - neighbor indices (-1 for no neighbor)
        edges: [NUM_STATES, K, INTERP_STEPS, DIM] - edge interpolation points
        edge_validity: [NUM_STATES, K] - boolean array of valid edges
        circles: [N, 3] - circle obstacles [x, y, r] (optional)
        rectangles: [N, 4] - rectangle obstacles [x, y, h, w] (optional)
        bounds: [4] - [x_min, x_max, y_min, y_max] (optional, auto-computed if None)
    
    Returns:
        matplotlib figure
    """
    # Convert tensors to numpy if needed
    def to_numpy(tensor):
        if torch.is_tensor(tensor):
            return tensor.cpu().numpy()
        return np.array(tensor)
    
    nodes = to_numpy(nodes)
    node_validity = to_numpy(node_validity)
    neighbors = to_numpy(neighbors)
    edge_validity = to_numpy(edge_validity)
    
    if circles is not None:
        circles = to_numpy(circles)
    if rectangles is not None:
        rectangles = to_numpy(rectangles)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set plot bounds
    if bounds is not None:
        bounds = to_numpy(bounds)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
    else:
        # Auto-compute bounds from nodes (using only x,y coordinates)
        valid_nodes_mask = node_validity.astype(bool)
        if np.any(valid_nodes_mask):
            valid_node_positions = nodes[valid_nodes_mask]
            margin = 1.0
            ax.set_xlim(valid_node_positions[:, 0].min() - margin, 
                       valid_node_positions[:, 0].max() + margin)
            ax.set_ylim(valid_node_positions[:, 1].min() - margin, 
                       valid_node_positions[:, 1].max() + margin)
    
    ax.set_aspect('equal')
    
    # Draw obstacles
    if circles is not None and len(circles) > 0:
        for cx, cy, r in circles:
            circle = Circle((cx, cy), r, fill=True, alpha=0.5, color='green')
            ax.add_patch(circle)
    
    if rectangles is not None and len(rectangles) > 0:
        for rx, ry, h, w in rectangles:  # Note: assuming [x, y, h, w] format
            rect = Rectangle((rx - w/2, ry - h/2), w, h, fill=True, alpha=0.5, color='green')
            ax.add_patch(rect)
    
    # Draw edges
    num_states, k = neighbors.shape
    for node_idx in range(num_states):
        if not node_validity[node_idx]:
            continue
            
        # Extract only x, y coordinates from the node (first 2 dimensions)
        x1, y1 = nodes[node_idx, 0], nodes[node_idx, 1]
        
        for k_idx in range(k):
            neighbor_idx = neighbors[node_idx, k_idx]
            
            # Check if neighbor exists (assuming -1 means no neighbor)
            if neighbor_idx < 0 or neighbor_idx >= num_states:
                continue
                
            # Check if neighbor is valid
            if not node_validity[neighbor_idx]:
                continue
            
            # Extract only x, y coordinates from the neighbor
            x2, y2 = nodes[neighbor_idx, 0], nodes[neighbor_idx, 1]
            
            # Get edge validity
            is_valid_edge = edge_validity[node_idx, k_idx]
            edge_color = 'black' if is_valid_edge else 'red'
            alpha = 0.6 if is_valid_edge else 0.3
            
            ax.plot([x1, x2], [y1, y2], '-', color=edge_color, alpha=alpha, linewidth=1)
    
    # Draw nodes
    for node_idx in range(num_states):
        # Extract only x, y coordinates
        x, y = nodes[node_idx, 0], nodes[node_idx, 1]
        is_valid = node_validity[node_idx]
        
        if is_valid:
            color = 'blue'
            alpha = 0.8
            radius = 0.3
        else:
            color = 'red'
            alpha = 0.5
            radius = 0.2
            
        node_circle = Circle((x, y), radius, fill=True, alpha=alpha, color=color)
        ax.add_patch(node_circle)
        
        # Add node index text (only for valid nodes to avoid clutter)
        if is_valid:
            ax.text(x, y, str(node_idx), ha='center', va='center', 
                   color='white', fontsize=6, weight='bold')
    
    ax.set_title('PRM Roadmap Visualization (X-Y Projection)', fontsize=18)
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=8, label='Valid Nodes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=8, label='Invalid Nodes'),
        Line2D([0], [0], color='black', linewidth=2, label='Valid Edges'),
        Line2D([0], [0], color='red', linewidth=2, label='Invalid Edges')
    ]
    if circles is not None and len(circles) > 0 or rectangles is not None and len(rectangles) > 0:
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                   markersize=10, alpha=0.5, label='Obstacles')
        )
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    return fig

def test_prm():
    # Create sample environment
    circles = torch.tensor([
        [2.0, 2.0, 0.5],  # x, y, radius
        [8.0, 8.0, 1.0],
    ], dtype=torch.float32, device='cuda')
    
    rectangles = torch.tensor([
        [4.0, 4.0, 1.0, 2.0],  # x, y, height, width
        [6.0, 2.0, 0.5, 1.5],
    ], dtype=torch.float32, device='cuda')
    
    # Define bounds [lower, upper] for [x, y, theta, pan, tilt]
    bounds = torch.tensor([
        [0.0, 0.0, -3.14159, 0.0, 0.0],  # lower bounds
        [10.0, 10.0, 3.14159, 0.0, 0.0]  # upper bounds
    ], dtype=torch.float32, device='cuda')
    
   
    t1 = time.time()
    # Build PRM
    nodes, node_validity, neighbors, edges, edge_validity = parallelrm_cuda.build_prm(
        circles, rectangles, bounds, num_states=1000, k=5, seed=12345
    )

    t2 = time.time()
    print(f"PRM build time: {t2 - t1:.4f} seconds")

    print(f"Nodes shape: {nodes.device}, {nodes.shape}")
    print(f"Node validity shape: {node_validity.device}, {node_validity.shape}")
    print(f"Neighbors shape: {neighbors.device}, {neighbors.shape}")
    print(f"Edges shape: {edges.device}, {edges.shape}")
    print(f"Edge validity shape: {edge_validity.device}, {edge_validity.shape}")
    print(f"Valid nodes: {node_validity.sum().item()}/{len(node_validity)}")


    fig = visualize_prm_from_tensors(
        nodes=nodes,
        node_validity=node_validity, 
        neighbors=neighbors,
        edges=edges,
        edge_validity=edge_validity,
        circles=circles,  # optional
        rectangles=rectangles,  # optional
        bounds=[0, 10, 0, 10]  # optional
    )
    plt.show()


if __name__ == "__main__":
    test_prm()