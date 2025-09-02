import time
import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from prm import PSPRM, Solution
from nn.inference import ModelLoader
from utils.EnvLoader import EnvironmentLoader
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch



def visualize_environment_and_path(env, prm, path, trajectory, source_node, target_node, animate=True, animation_speed=50):
    """
    Visualize the environment, obstacles, path, and trajectory with optional animation
    
    Args:
        env: Environment dictionary
        prm: PRM object
        path: Planned path
        trajectory: Trajectory array with shape (n_points, 3) where columns are [x, y, theta]
        source_node: Start node
        target_node: Goal node
        animate: Whether to create animation (default: True)
        animation_speed: Animation interval in milliseconds (default: 50)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Extract environment bounds for plotting
    bounds = env['bounds'].cpu().numpy()
    x_min, y_min = bounds[0, :2]
    x_max, y_max = bounds[1, :2]
    
    # Plot 1: Environment overview with path
    ax1 = axes[0]
    ax1.set_xlim(x_min - 0.1, x_max + 0.1)
    ax1.set_ylim(y_min - 0.1, y_max + 0.1)
    ax1.set_aspect('equal')
    ax1.set_title('Environment with Path Planning', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Draw obstacles using the correct format
    def draw_obstacles(ax):
        if 'circles' in env and env['circles'] is not None:
            circles = env['circles'].cpu().numpy() if torch.is_tensor(env['circles']) else np.array(env['circles'])
            if len(circles) > 0:
                for cx, cy, r in circles:
                    circle = Circle((cx, cy), r, fill=True, alpha=0.5, color='green')
                    ax.add_patch(circle)
        
        if 'rectangles' in env and env['rectangles'] is not None:
            rectangles = env['rectangles'].cpu().numpy() if torch.is_tensor(env['rectangles']) else np.array(env['rectangles'])
            if len(rectangles) > 0:
                for rx, ry, h, w in rectangles:
                    rect = Rectangle((rx - w/2, ry - h/2), w, h, fill=True, alpha=0.5, color='green')
                    ax.add_patch(rect)
        
        # Draw object if present
        if 'object_pose' in env:
            obj_pose = env['object_pose'].cpu().numpy()
            circle = Circle((obj_pose[0], obj_pose[1]), 0.05, 
                           facecolor='orange', alpha=0.8, edgecolor='darkorange')
            ax.add_patch(circle)
    
    draw_obstacles(ax1)
    
    # Draw all PRM nodes (light gray)
    positions = {}
    if hasattr(prm, 'graph') and prm.graph.number_of_nodes() > 0:
        for node in prm.graph.nodes():
            node_data = prm.graph.nodes[node]
            if 'pos' in node_data:
                pos = node_data['pos']
                if isinstance(pos, torch.Tensor):
                    pos = pos.cpu().numpy()
                positions[node] = pos[:2]  # Use only x, y coordinates
                ax1.plot(pos[0], pos[1], 'o', color='lightgray', markersize=2, alpha=0.5)
        
        # Draw PRM edges (light gray)
        for edge in prm.graph.edges():
            if edge[0] in positions and edge[1] in positions:
                pos1, pos2 = positions[edge[0]], positions[edge[1]]
                ax1.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                        '-', color='lightgray', alpha=0.3, linewidth=0.5)
    
    # Draw the planned path
    if path and len(path) > 1:
        path_positions = []
        for node in path:
            if node in positions:
                path_positions.append(positions[node])
        
        if path_positions:
            path_x = [pos[0] for pos in path_positions]
            path_y = [pos[1] for pos in path_positions]
            ax1.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.8, label='Planned Path')
            
            # Highlight path nodes
            ax1.scatter(path_x, path_y, c='blue', s=30, alpha=0.8, zorder=5)
    
    # Draw start and goal nodes
    if source_node in positions:
        start_pos = positions[source_node]
        ax1.plot(start_pos[0], start_pos[1], 'go', markersize=12, 
                label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
    
    if target_node in positions:
        goal_pos = positions[target_node]
        ax1.plot(goal_pos[0], goal_pos[1], 'rs', markersize=12, 
                label='Goal', markeredgecolor='darkred', markeredgewidth=2)
    
    ax1.legend(loc='upper right')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    
    # Plot 2: Animated trajectory
    ax2 = axes[1]
    ax2.set_title('Animated Robot Trajectory', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(x_min - 0.1, x_max + 0.1)
    ax2.set_ylim(y_min - 0.1, y_max + 0.1)
    ax2.set_aspect('equal')
    
    draw_obstacles(ax2)
    
    # Initialize animation elements
    trajectory_line = None
    robot_arrow = None
    trail_points = None
    
    if trajectory is not None and len(trajectory) > 0:
        if isinstance(trajectory, torch.Tensor):
            traj_np = trajectory.cpu().numpy()
        else:
            traj_np = np.array(trajectory)
        
        if traj_np.shape[1] >= 3:  # Ensure we have x, y, theta
            # Plot full trajectory path (static)
            ax2.plot(traj_np[:, 0], traj_np[:, 1], 'g--', linewidth=1, alpha=0.3, label='Full Path')
            
            # Mark start and end of trajectory
            ax2.plot(traj_np[0, 0], traj_np[0, 1], 'go', markersize=12, 
                    label='Trajectory Start', markeredgecolor='darkgreen', markeredgewidth=2)
            ax2.plot(traj_np[-1, 0], traj_np[-1, 1], 'rs', markersize=12, 
                    label='Trajectory End', markeredgecolor='darkred', markeredgewidth=2)
            
            if animate:
                # Initialize animated elements
                trajectory_line, = ax2.plot([], [], 'g-', linewidth=3, alpha=0.8, label='Traveled Path')
                trail_points, = ax2.plot([], [], 'ro', markersize=3, alpha=0.6)
                
                # Calculate arrow size based on plot dimensions
                plot_range = max(x_max - x_min, y_max - y_min)
                arrow_length = plot_range * 0.05  # 5% of plot range
                
                # Create robot arrow (will be updated in animation)
                robot_arrow = FancyArrowPatch((0, 0), (arrow_length, 0),
                                            arrowstyle='->', mutation_scale=20,
                                            color='red', linewidth=3, alpha=0.9)
                ax2.add_patch(robot_arrow)
                
                def animate_frame(frame):
                    # Update trajectory line (path traveled so far)
                    trajectory_line.set_data(traj_np[:frame+1, 0], traj_np[:frame+1, 1])
                    
                    # Update trail points (recent positions)
                    trail_start = max(0, frame - 10)  # Show last 10 points
                    trail_points.set_data(traj_np[trail_start:frame+1, 0], 
                                         traj_np[trail_start:frame+1, 1])
                    
                    # Update robot arrow position and orientation
                    if frame < len(traj_np):
                        x, y, theta, _, _ = traj_np[frame]
                        
                        # Calculate arrow end point
                        dx = arrow_length * np.cos(theta)
                        dy = arrow_length * np.sin(theta)
                        
                        # Update arrow position and direction
                        robot_arrow.set_positions((x, y), (x + dx, y + dy))
                    
                    return trajectory_line, trail_points, robot_arrow
                
                # Create animation
                n_frames = len(traj_np)
                ani = animation.FuncAnimation(fig, animate_frame, frames=n_frames,
                                            interval=animation_speed, blit=True, repeat=True)
                
                # Store animation reference to prevent garbage collection
                fig.animation = ani
            
            else:
                # Static visualization with arrows at selected points
                ax2.plot(traj_np[:, 0], traj_np[:, 1], 'g-', linewidth=2, alpha=0.8, label='Trajectory')
                
                # Show arrows at every nth point to avoid clutter
                n_arrows = min(20, len(traj_np))  # Maximum 20 arrows
                arrow_indices = np.linspace(0, len(traj_np)-1, n_arrows, dtype=int)
                
                plot_range = max(x_max - x_min, y_max - y_min)
                arrow_length = plot_range * 0.03
                
                for i in arrow_indices:
                    x, y, theta = traj_np[i]
                    dx = arrow_length * np.cos(theta)
                    dy = arrow_length * np.sin(theta)
                    
                    arrow = FancyArrowPatch((x, y), (x + dx, y + dy),
                                          arrowstyle='->', mutation_scale=15,
                                          color='red', linewidth=2, alpha=0.7)
                    ax2.add_patch(arrow)
    
    ax2.legend(loc='upper right')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    
    plt.tight_layout()
    
    return fig


def plot_environment_with_heatmap(graph, bounds, resolution=100):
    """
    Plots the environment and overlays a heatmap of scores across the environment.

    Parameters:
        graph (nx.Graph): The PRM graph with node attributes 'x', 'y', and 'score'.
        bounds (list): The environment bounds as [[x_min, y_min], [x_max, y_max]].
        resolution (int): The resolution of the heatmap grid.
    """
    # Extract bounds
    x_min, y_min, _, _, _ = bounds[0]
    x_max, y_max, _, _, _ = bounds[1]

    # Create a grid for the heatmap
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(x, y)

    # Initialize the heatmap grid
    heatmap = np.zeros_like(xx)

    # Populate the heatmap grid with scores
    for node, data in graph.nodes(data=True):
        if 'x' in data and 'y' in data and 'score' in data:
            node_x, node_y, score = data['x'], data['y'], data['score']
            # Find the closest grid point
            i = np.argmin(np.abs(x - node_x))
            j = np.argmin(np.abs(y - node_y))
            heatmap[j, i] = score

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(
        heatmap,
        extent=(x_min, x_max, y_min, y_max),
        origin='lower',
        cmap='hot',
        alpha=0.8
    )
    plt.colorbar(label='Score')
    plt.title('Environment Heatmap')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Overlay the PRM graph
    # for u, v in graph.edges():
    #     u_data = graph.nodes[u]
    #     v_data = graph.nodes[v]
    #     if 'x' in u_data and 'y' in u_data and 'x' in v_data and 'y' in v_data:
    #         plt.plot(
    #             [u_data['x'], v_data['x']],
    #             [u_data['y'], v_data['y']],
    #             color='blue',
    #             alpha=0.5
    #         )
    # for node, data in graph.nodes(data=True):
    #     if 'x' in data and 'y' in data:
    #         plt.scatter(data['x'], data['y'], c='black', s=10)

    plt.show()

def main():
    device = 'cuda'
    dtype = torch.float32
    env_config_file = "/home/lb73/cudaPRM/planning/resources/scenes/environment/multigoal_demo.yaml"  
    model_path = "/home/lb73/cudaPRM/planning/resources/models/percscore-nov12-50k.pt"
    seed = 2387
    

    start = {'x': 0.5, 'y': -3.0, 'theta': 3.14159/2}
    goal = {'x': 0.5, 'y': 2.5, 'theta': -3.14159}
    call_id = 0

    print("Loading environment and model...")
    env_loader = EnvironmentLoader(device=device)
    model_loader = ModelLoader(label_size=3, max_batch_size=10000, use_trt=False)
    env = env_loader.load_world(env_config_file)
    env['bounds'] = torch.concat([env['bounds'], torch.tensor([[-3.14159, 0.0, 0.0], [3.14159, 0.0, 0.0]], dtype=dtype, device=device)], dim=1)
    print(env['bounds'])
    # Cup
    env['object_pose'] = torch.tensor([-0.5, 2.5, 0.7, 0.0, 0.0, 0.7071068, -0.7071068], dtype=dtype, device=device)
    env['object_label'] = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
    env['bounds'][0, 0] = -1.5
    env['bounds'][1, 0] = 4
    env['bounds'][0, 1] = -4
    env['bounds'][1, 1] = 4
    model = model_loader.load_model(model_path)


    print("Starting testing...\n")
    start_time = time.time()
    prm = PSPRM(model, env)
    prm.build_prm(seed)   # graph attributes 'x', 'y', 'theta', 'pan', 'tilt'

    print(f"\nNUM EDGES: {len(prm.graph.edges(data=False))}\n")
    import networkx as nx
    print(f"\nNUM CONNECTED: {nx.number_connected_components(prm.graph)}\n")
    start_id, goal_id = prm.add_start_and_goal(start, goal, call_id)  # returns start_id, goal_id

    #plot_environment_with_heatmap(prm.graph, env['bounds'].cpu().numpy(), resolution=200)
    #print(prm.graph.nodes(data=True))
    
    path = prm.a_star_search(start_id=start_id, goal_id=goal_id, alpha=0.2, beta=1)
    end_time = time.time()
    print(f"Path planning completed in {end_time - start_time:.3f} seconds")

    t2 = time.time()
    sol = Solution(path)
    trajectory = sol.generate_trajectory(prm.graph)
    t3 = time.time()
    sol.print_path(prm.graph)
    print(f"Trajectory generation completed in {t3 - t2:.3f} seconds\n")

    # save trajectory to csv
    np.savetxt("./test_trajectory.csv", trajectory, delimiter=",")

    #print_path_info(path, trajectory, prm)
    
    fig = visualize_environment_and_path(env, prm, path, trajectory, start_id, goal_id, animation_speed=10)
    plt.show()
    # Save animation as video if present
    # if hasattr(fig, 'animation'):
    #     fig.animation.save('./prm_animation.gif', writer='ffmpeg', fps=30)
    #     print("Animation saved as 'prm_animation.gif'")
    # else:
    #     # If no animation, save static image
    #     fig.savefig('./prm_visualization.png', dpi=300, bbox_inches='tight')
    #     print("Visualization saved as 'prm_visualization.png'")
    
# Current issue:
    # Pan and tilt are calculated for the sampled thetas, but the trajectory generator creates new thetas
    # Therefore, the pan and tilt values in the trajectory are not accurate

# Solution:
    # Build PRM like normal, but when creating the trajectory, recalculate pan and tilt for each waypoint that is returned by the A* search
    # Then, interpolate between those pan and tilt values when generating the trajectory

if __name__ == "__main__":
    main()


# def visualize_prm_from_networkx(G, circles=None, rectangles=None, bounds=None, path=None):
#     """
#     Visualize PRM roadmap from NetworkX graph with optional path highlighting.
    
#     Args:
#         G: NetworkX graph with node attributes 'x', 'y' for positions
#         circles: [N, 3] - circle obstacles [x, y, r] (optional)
#         rectangles: [N, 4] - rectangle obstacles [x, y, h, w] (optional)
#         bounds: [4] - [x_min, x_max, y_min, y_max] (optional, auto-computed if None)
#         path: list of node IDs - path to highlight (optional)
    
#     Returns:
#         matplotlib figure
#     """
#     # Convert obstacles to numpy if needed
#     def to_numpy(tensor):
#         if torch.is_tensor(tensor):
#             return tensor.cpu().numpy()
#         return np.array(tensor)
    
#     circles = to_numpy(circles)
#     rectangles = to_numpy(rectangles)
#     bounds = to_numpy(bounds)
    
#     # Create figure with interactive functionality
#     fig, ax = plt.subplots(figsize=(12, 10))
    
#     # Store path for toggling
#     path_data = {'path': path, 'visible': False, 'elements': []}
    
#     # Get node positions from x, y attributes
#     x_coords = nx.get_node_attributes(G, 'x')
#     y_coords = nx.get_node_attributes(G, 'y')
    
#     if not x_coords or not y_coords:
#         raise ValueError("Graph nodes must have 'x' and 'y' attributes")
    
#     # Set plot bounds
#     if bounds is not None:
#         bounds = to_numpy(bounds)
#         ax.set_xlim(bounds[0][0], bounds[1][0])
#         ax.set_ylim(bounds[0][1], bounds[1][1])
#     else:
#         # Auto-compute bounds from node positions
#         margin = 1.0
#         ax.set_xlim(min(x_coords.values()) - margin, max(x_coords.values()) + margin)
#         ax.set_ylim(min(y_coords.values()) - margin, max(y_coords.values()) + margin)
    
#     ax.set_aspect('equal')
    
#     # Draw obstacles
#     if circles is not None and len(circles) > 0:
#         for cx, cy, r in circles:
#             circle = Circle((cx, cy), r, fill=True, alpha=0.5, color='green')
#             ax.add_patch(circle)
    
#     if rectangles is not None and len(rectangles) > 0:
#         for rx, ry, h, w in rectangles:
#             rect = Rectangle((rx - w/2, ry - h/2), w, h, fill=True, alpha=0.5, color='green')
#             ax.add_patch(rect)
    
#     # Draw edges
#     for node1, node2 in G.edges():
#         if node1 in x_coords and node2 in x_coords and node1 in y_coords and node2 in y_coords:
#             x1, y1 = x_coords[node1], y_coords[node1]
#             x2, y2 = x_coords[node2], y_coords[node2]
            
#             # All edges are considered valid in this graph structure
#             ax.plot([x1, x2], [y1, y2], '-', color='black', alpha=0.6, linewidth=1)
    
#     # Draw nodes
#     for node in G.nodes():
#         if node in x_coords and node in y_coords:
#             x, y = x_coords[node], y_coords[node]
            
#             # All nodes are considered valid in this graph structure
#             color = 'blue'
#             alpha = 0.8
#             radius = 0.05
                
#             node_circle = Circle((x, y), radius, fill=True, alpha=alpha, color=color)
#             ax.add_patch(node_circle)
            
#             ax.text(x, y, str(node), ha='center', va='center', 
#                    color='white', fontsize=6, weight='bold')
    
#     def toggle_path():
#         """Toggle path visibility"""
#         if path_data['path'] is None:
#             print("No path specified")
#             return
            
#         if path_data['visible']:
#             # Hide path
#             for element in path_data['elements']:
#                 element.remove()
#             path_data['elements'] = []
#             path_data['visible'] = False
#             print("Path hidden")
#         else:
#             # Show path
#             path_nodes = path_data['path']
            
#             # Validate path nodes
#             valid_path_nodes = []
#             for node_id in path_nodes:
#                 if node_id in G.nodes() and node_id in x_coords and node_id in y_coords:
#                     valid_path_nodes.append(node_id)
#                 else:
#                     print(f"Warning: Node {node_id} not found in graph")
            
#             if len(valid_path_nodes) < 2:
#                 print("Path must have at least 2 valid nodes")
#                 return
            
#             # Draw path edges
#             for i in range(len(valid_path_nodes) - 1):
#                 node1_id = valid_path_nodes[i]
#                 node2_id = valid_path_nodes[i + 1]
                
#                 x1, y1 = x_coords[node1_id], y_coords[node1_id]
#                 x2, y2 = x_coords[node2_id], y_coords[node2_id]
                
#                 # Draw thick path edge
#                 line, = ax.plot([x1, x2], [y1, y2], '-', color='orange', 
#                               linewidth=4, alpha=0.8, zorder=10)
#                 path_data['elements'].append(line)
            
#             # Highlight path nodes
#             for i, node_id in enumerate(valid_path_nodes):
#                 x, y = x_coords[node_id], y_coords[node_id]
                
#                 # Different colors for start/end vs intermediate nodes
#                 if i == 0:  # Start node
#                     color = 'lime'
#                     radius = 0.25
#                     label = 'START'
#                 elif i == len(valid_path_nodes) - 1:  # End node
#                     color = 'red'
#                     radius = 0.25
#                     label = 'END'
#                 else:  # Intermediate nodes
#                     color = 'orange'
#                     radius = 0.1
#                     label = str(i)
                
#                 # Draw highlighted node
#                 path_circle = Circle((x, y), radius, fill=True, alpha=0.9, 
#                                    color=color, zorder=15)
#                 ax.add_patch(path_circle)
#                 path_data['elements'].append(path_circle)
                
#                 # Add path sequence number
#                 text = ax.text(x, y, label, ha='center', va='center', 
#                              color='black', fontsize=8, weight='bold', zorder=20)
#                 path_data['elements'].append(text)
            
#             path_data['visible'] = True
#             print(f"Path shown: {len(valid_path_nodes)} nodes")
        
#         fig.canvas.draw()
    
#     # Add key press event handler
#     def on_key(event):
#         if event.key == 'p':  # Press 'p' to toggle path
#             toggle_path()
    
#     fig.canvas.mpl_connect('key_press_event', on_key)
    
#     # Set title and labels
#     title = 'PRM Roadmap Visualization (NetworkX Graph)'
#     if path is not None:
#         title += f"\nPress 'p' to toggle path highlighting"
#     ax.set_title(title, fontsize=16)
#     ax.set_xlabel('X', fontsize=14)
#     ax.set_ylabel('Y', fontsize=14)
#     ax.grid(True, alpha=0.3)
    
#     # Add legend
#     legend_elements = [
#         Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
#                markersize=8, label='Valid Nodes'),
#         Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
#                markersize=8, label='Invalid Nodes'),
#         Line2D([0], [0], color='black', linewidth=2, label='Valid Edges'),
#         Line2D([0], [0], color='red', linewidth=2, label='Invalid Edges')
#     ]
    
#     if path is not None:
#         legend_elements.extend([
#             Line2D([0], [0], color='orange', linewidth=4, label='Path'),
#             Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', 
#                    markersize=10, label='Start Node'),
#             Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
#                    markersize=10, label='End Node')
#         ])
    
#     if circles is not None and len(circles) > 0 or rectangles is not None and len(rectangles) > 0:
#         legend_elements.append(
#             Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
#                    markersize=10, alpha=0.5, label='Obstacles')
#         )
    
#     ax.legend(handles=legend_elements, loc='upper right')
    
#     return fig
