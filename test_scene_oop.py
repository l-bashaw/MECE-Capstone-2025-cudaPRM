import time
import torch

import numpy as np

import matplotlib.pyplot as plt

from prm import PSPRM
from matplotlib.lines import Line2D
from nn.inference import ModelLoader
from utils.EnvLoader import EnvironmentLoader
from matplotlib.patches import Circle, Rectangle

import networkx as nx

def main():
    device = 'cuda' 
    dtype = torch.float32

    env_config_file = "/home/lenman/capstone/parallelrm/resources/scenes/scene_hostpital_plant_0.yaml"  
    model_path = "/home/lenman/capstone/parallelrm/resources/models/percscore-nov12-50k.pt"

    seed = 22363387
    source_node = 463  
    target_node = 747

    print("Loading environment and model...")
    env_loader = EnvironmentLoader(device=device)
    model_loader = ModelLoader(label_size=3, max_batch_size=10000, use_trt=True)
    env = env_loader.load_world(env_config_file)
    env['bounds'] = torch.concat([env['bounds'], torch.tensor([[-3.14159, 0.0, 0.0], [3.14159, 0.0, 0.0]], dtype=dtype, device=device)], dim=1)
    env['object_pose'] = torch.tensor([.175, .78, .82, 1.0, 0.0, 0.0, 1.0], dtype=dtype, device=device)
    env['object_label'] = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)

    model = model_loader.load_model(model_path)

    print("Starting testing...\n\n")

    
    
    prm = PSPRM(model, env)
    prm.build_prm(seed)
    path = prm.a_star_search(source_node, target_node, alpha=1, beta=0.1)
  
    fig = visualize_prm_from_networkx(
        prm.graph, 
        circles=env['circles'], 
        rectangles=env['rectangles'],
        bounds=env['bounds'][:, :2], 
        path=path
    )
    plt.show()
    del fig
    path2 = prm.a_star_search(source_node, target_node, alpha=2, beta=0.3)
    fig = visualize_prm_from_networkx(
        prm.graph, 
        circles=env['circles'], 
        rectangles=env['rectangles'],
        bounds=env['bounds'][:, :2], 
        path=path2
    )
    plt.show()



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
            radius = 0.05
                
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

if __name__ == "__main__":
    main()


# fig = visualize_prm_from_networkx(
#     graph, 
#     circles=env['circles'], 
#     rectangles=env['rectangles'],
#     bounds=env['bounds'][:, :2], path=path
# )
# plt.show()


# circles = torch.tensor([
#         [2.0, 2.0, 0.5],  # x, y, radius
#         [8.0, 8.0, 1.0],
#         [5.0, 5.0, 0.3],
#         [3.0, 7.0, 0.4],
#         [7.0, 3.0, 0.6],
#         [1.0, 9.0, 0.2],
#         [9.0, 2.0, 0.5],
#         [6.0, 6.0, 0.4],
#         [4.0, 1.0, 0.3],
#         [2.5, 3.5, 0.2],
#     ], dtype=torch.float32, device='cuda')
    
#     rectangles = torch.tensor([
#         [4.0, 4.0, 1.0, 2.0],  # x, y, height, width
#         [6.0, 2.0, 0.5, 1.5],
#         [2.0, 8.0, 1.5, 1.0],
#         [9.0, 1.0, 2.0, 1.0],
#         [3.0, 6.0, 1.0, 2.0],
#         [8.0, 4.0, 1.0, 1.0],
#         [1.0, 5.0, 1.0, 1.0],
#         [7.0, 7.0, 1.5, 1.5],
#         [5.0, 3.0, 1.0, 2.0],
#         [2.0, 2.0, 1.0, 1.0],
#     ], dtype=torch.float32, device='cuda')
    
#     # Define bounds [lower, upper] for [x, y, theta, pan, tilt]
#     bounds = torch.tensor([
#         [0.0, 0.0, -3.14159, 0.0, 0.0],  # lower bounds
#         [10.0, 10.0, 3.14159, 0.0, 0.0]  # upper bounds
#     ], dtype=torch.float32, device='cuda')