# Start region
# x bounds (0.5, 3)
# y bounds (-3.5, -2)


# Goal region 1
# x bounds (0.5, 3)
# y bounds (2, 3.5)
# theta bounds (-3.14159, 3.14159)

# Goal region 2
# x bounds (-0.5, 0.5)
# y bounds (0.5, 1.5)
# theta bounds (-3.14159, 3.14159)

# With numpy
# Randomly sample 5 random start positons in the start region
# Randomly sample 5 random goal positions from the two goal regions
# Sample random theta in (-3.14159, 3.14159)
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.EnvLoader import EnvironmentLoader

from matplotlib.patches import Circle, Rectangle, FancyArrowPatch


starts = np.random.uniform(low=[0.5, -3.5], high=[3, -2], size=(5, 2))
starts_theta = np.random.uniform(low=-3.14159, high=3.14159, size=(5, 1))
starts = np.hstack((starts, starts_theta, np.zeros((5, 2))))  # add zero pan and tilt

goals_region1 = np.random.uniform(low=[0.5, 2], high=[3, 3.5], size=(3, 2))
goals_region2 = np.random.uniform(low=[-0.5, 0.5], high=[0.5, 1.5], size=(2, 2))
goals_theta = np.random.uniform(low=-3.14159, high=3.14159, size=(5, 1))
goals_region1 = np.hstack((goals_region1, goals_theta[:3], np.zeros((3, 2))))  # add zero pan and tilt
goals_region2 = np.hstack((goals_region2, goals_theta[3:], np.zeros((2, 2))))  # add zero pan and tilt
goals = np.vstack((goals_region1, goals_region2))

print(starts)
print(goals)

env_config_file = "/home/lb73/cudaPRM/planning/resources/scenes/environment/multigoal_demo.yaml"  
env_loader = EnvironmentLoader(device='cuda')
env = env_loader.load_world(env_config_file)
env['bounds'] = torch.concat([env['bounds'], torch.tensor([[-3.14159, 0.0, 0.0], [3.14159, 0.0, 0.0]], dtype=torch.float32, device='cuda')], dim=1)
env['bounds'][0, 0] = -1.5
env['bounds'][1, 0] = 4
env['bounds'][0, 1] = -4
env['bounds'][1, 1] = 4

def visualize_environment_and_path(env, starts, goals):
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

    # Add start and goal x/y positions
    ax1.scatter(starts[:, 0], starts[:, 1], c='blue', s=100, label='Start', marker='o', edgecolors='k')
    ax1.scatter(goals[:, 0], goals[:, 1], c='red', s=100, label='Goal', marker='*', edgecolors='k')
    for i, (sx, sy) in enumerate(starts[:, :2]):
        ax1.text(sx, sy, f'S{i+1}', fontsize=9, ha='right', va='bottom', color='blue', fontweight='bold')
    for i, (gx, gy) in enumerate(goals[:, :2]):
        ax1.text(gx, gy, f'G{i+1}', fontsize=9, ha='left', va='bottom', color='red', fontweight='bold')

    draw_obstacles(ax1)
    
   
    
    ax1.legend(loc='upper right')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')

    
    
    plt.tight_layout()
    # Save as png
    fig.savefig('./environment_visualization.png', dpi=300, bbox_inches='tight')
    

visualize_environment_and_path(env, starts, goals)

# Save starts and goals as json
import json
with open('./starts_goals.json', 'w') as f:
    json.dump({
        "starts": starts.tolist(),
        "goals": goals.tolist()
    }, f, indent=4)