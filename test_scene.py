import time
import cuPRM
import torch

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from fk import FK
from matplotlib.lines import Line2D
from nn import inference as trti
from utils.EnvLoader import EnvironmentLoader
from matplotlib.patches import Circle, Rectangle


def visualize_prm_from_networkx(G, circles=None, rectangles=None, bounds=None, path=None):
    """
    Visualize PRM roadmap from NetworkX graph with optional path highlighting.
    
    Args:
        G: NetworkX graph with node attributes 'x', 'y' for positions
        circles: [N, 3] - circle obstacles [x, y, r] (optional)
        rectangles: [N, 4] - rectangle obstacles [x, y, h, w] (optional)
        bounds: [4] - [x_min, x_max, y_min, y_max] (optional, auto-computed if None)
        path: list of node IDs - path to highlight (optional)
    
    Returns:
        matplotlib figure
    """
    # Convert obstacles to numpy if needed
    def to_numpy(tensor):
        if torch.is_tensor(tensor):
            return tensor.cpu().numpy()
        return np.array(tensor)
    
    circles = to_numpy(circles)
    rectangles = to_numpy(rectangles)
    bounds = to_numpy(bounds)
    
    # Create figure with interactive functionality
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Store path for toggling
    path_data = {'path': path, 'visible': False, 'elements': []}
    
    # Get node positions from x, y attributes
    x_coords = nx.get_node_attributes(G, 'x')
    y_coords = nx.get_node_attributes(G, 'y')
    
    if not x_coords or not y_coords:
        raise ValueError("Graph nodes must have 'x' and 'y' attributes")
    
    # Set plot bounds
    if bounds is not None:
        bounds = to_numpy(bounds)
        ax.set_xlim(bounds[0][0], bounds[1][0])
        ax.set_ylim(bounds[0][1], bounds[1][1])
    else:
        # Auto-compute bounds from node positions
        margin = 1.0
        ax.set_xlim(min(x_coords.values()) - margin, max(x_coords.values()) + margin)
        ax.set_ylim(min(y_coords.values()) - margin, max(y_coords.values()) + margin)
    
    ax.set_aspect('equal')
    
    # Draw obstacles
    if circles is not None and len(circles) > 0:
        for cx, cy, r in circles:
            circle = Circle((cx, cy), r, fill=True, alpha=0.5, color='green')
            ax.add_patch(circle)
    
    if rectangles is not None and len(rectangles) > 0:
        for rx, ry, h, w in rectangles:
            rect = Rectangle((rx - w/2, ry - h/2), w, h, fill=True, alpha=0.5, color='green')
            ax.add_patch(rect)
    
    # Draw edges
    for node1, node2 in G.edges():
        if node1 in x_coords and node2 in x_coords and node1 in y_coords and node2 in y_coords:
            x1, y1 = x_coords[node1], y_coords[node1]
            x2, y2 = x_coords[node2], y_coords[node2]
            
            # All edges are considered valid in this graph structure
            ax.plot([x1, x2], [y1, y2], '-', color='black', alpha=0.6, linewidth=1)
    
    # Draw nodes
    for node in G.nodes():
        if node in x_coords and node in y_coords:
            x, y = x_coords[node], y_coords[node]
            
            # All nodes are considered valid in this graph structure
            color = 'blue'
            alpha = 0.8
            radius = 0.3
                
            node_circle = Circle((x, y), radius, fill=True, alpha=alpha, color=color)
            ax.add_patch(node_circle)
            
            ax.text(x, y, str(node), ha='center', va='center', 
                   color='white', fontsize=6, weight='bold')
    
    def toggle_path():
        """Toggle path visibility"""
        if path_data['path'] is None:
            print("No path specified")
            return
            
        if path_data['visible']:
            # Hide path
            for element in path_data['elements']:
                element.remove()
            path_data['elements'] = []
            path_data['visible'] = False
            print("Path hidden")
        else:
            # Show path
            path_nodes = path_data['path']
            
            # Validate path nodes
            valid_path_nodes = []
            for node_id in path_nodes:
                if node_id in G.nodes() and node_id in x_coords and node_id in y_coords:
                    valid_path_nodes.append(node_id)
                else:
                    print(f"Warning: Node {node_id} not found in graph")
            
            if len(valid_path_nodes) < 2:
                print("Path must have at least 2 valid nodes")
                return
            
            # Draw path edges
            for i in range(len(valid_path_nodes) - 1):
                node1_id = valid_path_nodes[i]
                node2_id = valid_path_nodes[i + 1]
                
                x1, y1 = x_coords[node1_id], y_coords[node1_id]
                x2, y2 = x_coords[node2_id], y_coords[node2_id]
                
                # Draw thick path edge
                line, = ax.plot([x1, x2], [y1, y2], '-', color='orange', 
                              linewidth=4, alpha=0.8, zorder=10)
                path_data['elements'].append(line)
            
            # Highlight path nodes
            for i, node_id in enumerate(valid_path_nodes):
                x, y = x_coords[node_id], y_coords[node_id]
                
                # Different colors for start/end vs intermediate nodes
                if i == 0:  # Start node
                    color = 'lime'
                    radius = 0.25
                    label = 'START'
                elif i == len(valid_path_nodes) - 1:  # End node
                    color = 'red'
                    radius = 0.25
                    label = 'END'
                else:  # Intermediate nodes
                    color = 'orange'
                    radius = 0.1
                    label = str(i)
                
                # Draw highlighted node
                path_circle = Circle((x, y), radius, fill=True, alpha=0.9, 
                                   color=color, zorder=15)
                ax.add_patch(path_circle)
                path_data['elements'].append(path_circle)
                
                # Add path sequence number
                text = ax.text(x, y, label, ha='center', va='center', 
                             color='black', fontsize=8, weight='bold', zorder=20)
                path_data['elements'].append(text)
            
            path_data['visible'] = True
            print(f"Path shown: {len(valid_path_nodes)} nodes")
        
        fig.canvas.draw()
    
    # Add key press event handler
    def on_key(event):
        if event.key == 'p':  # Press 'p' to toggle path
            toggle_path()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Set title and labels
    title = 'PRM Roadmap Visualization (NetworkX Graph)'
    if path is not None:
        title += f"\nPress 'p' to toggle path highlighting"
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=8, label='Valid Nodes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=8, label='Invalid Nodes'),
        Line2D([0], [0], color='black', linewidth=2, label='Valid Edges'),
        Line2D([0], [0], color='red', linewidth=2, label='Invalid Edges')
    ]
    
    if path is not None:
        legend_elements.extend([
            Line2D([0], [0], color='orange', linewidth=4, label='Path'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', 
                   markersize=10, label='Start Node'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label='End Node')
        ])
    
    if circles is not None and len(circles) > 0 or rectangles is not None and len(rectangles) > 0:
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                   markersize=10, alpha=0.5, label='Obstacles')
        )
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    return fig

# motion cost is np.linalg.norm between two nodes (just x, y, theta?)


def tensors_to_networkx(nodes_tensor, neighbors_tensor, scores_tensor=None):
    if nodes_tensor.device.type == 'cuda':
        nodes = nodes_tensor.cpu().numpy()
        neighbors = neighbors_tensor.cpu().numpy()
        if scores_tensor is not None:
            scores = scores_tensor.cpu().numpy()
    else:
        nodes = nodes_tensor.detach().numpy()
        neighbors = neighbors_tensor.detach().numpy()
        if scores_tensor is not None:
            scores = scores_tensor.detach().numpy()
    
    N = nodes.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    
    nx.set_node_attributes(G, dict(enumerate(nodes[:, 0])), 'x')
    nx.set_node_attributes(G, dict(enumerate(nodes[:, 1])), 'y')
    nx.set_node_attributes(G, dict(enumerate(nodes[:, 2])), 'theta')
    nx.set_node_attributes(G, dict(enumerate(nodes[:, 3])), 'pan')
    nx.set_node_attributes(G, dict(enumerate(nodes[:, 4])), 'tilt')
    
    if scores_tensor is not None:
        nx.set_node_attributes(G, dict(enumerate(scores)), 'score')
    
    valid_mask = (neighbors >= 0) & (neighbors < N)
    source_indices, neighbor_indices = np.where(valid_mask)
    target_indices = neighbors[source_indices, neighbor_indices]

    # Calculation of edge weights
    src_positions = nodes[source_indices, :2]  # Shape: (num_edges, 2)
    tgt_positions = nodes[target_indices, :2]  # Shape: (num_edges, 2)
    weights = np.linalg.norm(src_positions - tgt_positions, axis=1)  
    
    edge_list_with_weights = list(zip(source_indices.tolist(), 
                                    target_indices.tolist(), 
                                    weights.tolist()))
    
    G.add_weighted_edges_from(edge_list_with_weights)
    
    return G

def astar_search(graph, start, goal, alpha, beta):
    edge_weights = [data['weight'] for u, v, data in graph.edges(data=True)]
    if edge_weights:
        max_weight = max(edge_weights)
        min_weight = min(edge_weights)
        weight_range = max_weight - min_weight if max_weight > min_weight else 1.0
    else:
        weight_range = 1.0

    def heuristic(u, v):
        # Get node positions (x, y coordinates)
        u_pos = np.array([graph.nodes[u]['x'], graph.nodes[u]['y']])
        v_pos = np.array([graph.nodes[v]['x'], graph.nodes[v]['y']])
        
        # Get node scores
        u_score = graph.nodes[u].get('score', 0)
        v_score = graph.nodes[v].get('score', 0)
        
        # Perception cost (using current node's score)
        
        # Motion cost (Euclidean distance between x, y positions)
        motion_cost = np.linalg.norm(u_pos - v_pos) / weight_range
        
        # Combined heuristic: prioritize high scores, minimize motion cost
        return -beta * min(u_score, v_score) + alpha * motion_cost
    
    try:
        path = nx.astar_path(
            graph, 
            start, 
            goal, 
            heuristic=heuristic,
            weight='weight'
        )
        return path
    except nx.NetworkXNoPath:
        print(f"No path found from {start} to {goal}")
        return None


def build_graph(env: dict, model, seed):
    bounds = env['bounds']
    circles = env['circles']
    rectangles = env['rectangles']
    object_pose_world = env['object_pose']
    obj_label = env['object_label'] 
        
    nodes, node_validity, neighbors, edges, edge_validity = cuPRM.build_prm(
        circles, rectangles, bounds, seed
    )

    valid_neighbors = torch.where(edge_validity, neighbors, -1)  # Replace invalid neighbors with -1
    
    projected_nodes, diffs = FK.calculate_pan_tilt_for_nodes(nodes, object_pose_world)
    diffs = torch.cat((diffs, obj_label.unsqueeze(0).repeat(nodes.shape[0], 1)), dim=1)
    scores = model(diffs)

    graph = tensors_to_networkx(projected_nodes, valid_neighbors, scores)

    
    return graph, {'nodes': projected_nodes, 'node_validity': node_validity, 'neighbors': valid_neighbors, 'edge_validity': edge_validity}

env_config_file = "/home/lenman/capstone/parallelrm/resources/scenes/scene_hostpital_plant_0.yaml"  # Update this path
model_path = "/home/lenman/capstone/parallelrm/resources/models/percscore-nov12-50k.pt"
seed = 25487
source_node = 463  # Update with your source node index
target_node = 747  # Update with your target node index

loader = EnvironmentLoader(device='cuda' if torch.cuda.is_available() else 'cpu')
model =  trti.build_trt_from_dict(model_path, batch_size=10000)  


# Load environment from YAML
env = loader.load_world(env_config_file)
env['bounds'] = torch.concat([env['bounds'], torch.tensor([[-3.14159, 0.0, 0.0], [3.14159, 0.0, 0.0]], device=env['bounds'].device)], dim=1)
print(env['bounds'][:, :2])

env['object_pose'] = torch.tensor([.175, .78, .82, 1.0, 0.0, 0.0, 1.0], dtype=torch.float32, device='cuda')
env['object_label'] = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda')

times = []
for i in range (10):
    seed = seed + 4*i
    print("\n\nBuilding graph...")
    t1 = time.time()
    graph, tensors = build_graph(env, model, seed)
    path = astar_search(graph, source_node, target_node, alpha=1, beta=1)
    if i > 0:
        times.append(time.time() - t1)

print(f"Average time to build graph: {np.mean(times):.4f} seconds")
print(f"Standard deviation: {np.std(times):.4f} seconds")


# fig = visualize_prm_from_networkx(
#     graph, 
#     circles=env['circles'], 
#     rectangles=env['rectangles'],
#     bounds=env['bounds'][:, :2], path=path
# )
# plt.show()