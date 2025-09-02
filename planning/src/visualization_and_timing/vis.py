import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.patches import Circle, Rectangle

def read_data_from_file(filename):
    with open(filename, 'r') as file:
        content = file.read()
    
    # Parse states (x, y coordinates)
    states_pattern = re.compile(r'State (\d+): ([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+)')
    states_matches = states_pattern.findall(content)
    states = {int(idx): (float(x), float(y)) for idx, x, y, _, _, _ in states_matches}
    
    # Parse neighbors
    neighbors_pattern = re.compile(r'Neighbors (\d+): ([\d\s]+)')
    neighbors_matches = neighbors_pattern.findall(content)
    neighbors = {int(idx): [int(n) for n in neighs.strip().split()] for idx, neighs in neighbors_matches}
    
    # Parse validity for nodes and edges
    valid_node_pattern = re.compile(r'Valid Node (\d+): (\d)')
    valid_node_matches = valid_node_pattern.findall(content)
    valid_nodes = {int(idx): int(v) == 1 for idx, v in valid_node_matches}

    valid_edge_pattern = re.compile(r'Valid Edge (\d+): (\d)')
    valid_edge_matches = valid_edge_pattern.findall(content)
    valid_edges = [int(v) == 1 for _, v in valid_edge_matches]

    # Parse environment bounds
    bounds_pattern = re.compile(r'Bounds: \[([-\d.]+), ([-\d.]+)\] x \[([-\d.]+), ([-\d.]+)\]')
    bounds_match = bounds_pattern.search(content)
    bounds = [float(bounds_match.group(1)), float(bounds_match.group(2)), 
              float(bounds_match.group(3)), float(bounds_match.group(4))]
    
    # Parse circles
    circles_pattern = re.compile(r'Circle (\d+): Center \(([-\d.]+), ([-\d.]+)\), Radius ([-\d.]+)')
    circles_matches = circles_pattern.findall(content)
    circles = [(float(x), float(y), float(r)) for _, x, y, r in circles_matches]
    
    # Parse rectangles
    rectangles_pattern = re.compile(r'Rectangle (\d+): Center \(([-\d.]+), ([-\d.]+)\), Width ([-\d.]+), Height ([-\d.]+)')
    rectangles_matches = rectangles_pattern.findall(content)
    rectangles = [(float(x), float(y), float(w), float(h)) for _, x, y, w, h in rectangles_matches]
    
    return states, neighbors, valid_nodes, valid_edges, bounds, circles, rectangles

def visualize_prm(states, neighbors, valid_nodes, valid_edges, bounds, circles, rectangles):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set plot bounds
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect('equal')
    
    # Draw obstacles
    for cx, cy, r in circles:
        circle = Circle((cx, cy), r, fill=True, alpha=0.5, color='green')
        ax.add_patch(circle)
    
    for rx, ry, w, h in rectangles:
        rect = Rectangle((rx - w/2, ry - h/2), w, h, fill=True, alpha=0.5, color='green')
        ax.add_patch(rect)
    
    # Draw edges
    edge_counter = 0
    for node_idx, neighbor_indices in neighbors.items():
        x1, y1 = states[node_idx]
        for neighbor_idx in neighbor_indices:
            if neighbor_idx not in states:
                continue
            x2, y2 = states[neighbor_idx]

            # Get edge validity by order
            is_valid_edge = valid_edges[edge_counter] if edge_counter < len(valid_edges) else True
            edge_color = 'black' if is_valid_edge else 'red'
            ax.plot([x1, x2], [y1, y2], '-', color=edge_color, alpha=0.6)
            edge_counter += 1
    
    # Draw nodes
    for node_idx, (x, y) in states.items():
        is_valid = valid_nodes.get(node_idx, True)
        color = 'blue' if is_valid else 'red'
        node_circle = Circle((x, y), 0.3, fill=True, alpha=0.7, color=color)
        ax.add_patch(node_circle)
        ax.text(x, y, str(node_idx), ha='center', va='center', color='white', fontsize=8)
    
    ax.set_title('Roadmap in an 8 Obstacle Environment', fontsize=18)
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.grid(True)
    
    return fig

def main():
    filename = "./roadmapN400K10.txt"
    try:
        states, neighbors, valid_nodes, valid_edges, bounds, circles, rectangles = read_data_from_file(filename)
    except FileNotFoundError:
        print(f"File {filename} not found. Please create the file with the given data.")
        return
    
    fig = visualize_prm(states, neighbors, valid_nodes, valid_edges, bounds, circles, rectangles)
    #plt.tight_layout()
    plt.savefig("prm_visualization.jpg", dpi=1000)
    plt.show()

if __name__ == "__main__":
    main()
