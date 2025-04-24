import torch
import pytorch_kinematics as pk
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

B = 100000

####################################################################################
### We want the z-axis of link_head_nav_cam to point at our object in the world. ###
####################################################################################

# This is the pose of the camera in the robot frame when pan and tilt joints are zero.
cam_pos_robot = torch.tensor([0.0450, 0.0523, 1.2607], dtype=dtype, device=device)
cam_quat_robot = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=dtype, device=device)
cam_rot_robot = torch.tensor(
    [[0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]], dtype=dtype, device=device
)

# Specify some object position in the world frame.
object_pos_world_h = torch.tensor([2.5, 3.0, 0.5, 1.0], dtype=dtype, device=device).unsqueeze(0).expand(B, -1)  # x, y, z, w
zero_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=dtype, device=device)
object_pose_world = torch.cat((object_pos_world_h[0][:3], zero_quat)) # x, y, z, qx, qy, qz, qw`

xy = (torch.rand(B, 2, dtype=dtype, device=device) - 0.5) * 10.0
theta = (torch.rand(B, 1, dtype=dtype, device=device) * 2 - 1) * torch.pi
robot_states = torch.cat([xy, theta], dim=1)
   

# --------Start of FK code to time-------------

# Create batched camera base transforms in robot frame (same for all robots)
T_robot_camera_base = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
T_robot_camera_base[:, :3, :3] = cam_rot_robot.unsqueeze(0).repeat(B, 1, 1)
T_robot_camera_base[:, :3, 3] = cam_pos_robot.unsqueeze(0).repeat(B, 1)

# Build the robot base-to-world transforms for each robot.
cos_theta = torch.cos(robot_states[:, 2])
sin_theta = torch.sin(robot_states[:, 2])

T_world_robot = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)

T_world_robot[:, 0, 0] = cos_theta
T_world_robot[:, 0, 1] = -sin_theta
T_world_robot[:, 1, 0] = sin_theta
T_world_robot[:, 1, 1] = cos_theta
T_world_robot[:, 0, 3] = robot_states[:, 0]
T_world_robot[:, 1, 3] = robot_states[:, 1]

# Step 2: Camera base pose in world
T_world_cam_base = torch.bmm(T_world_robot, T_robot_camera_base)

# Invert camera transforms
T_cam_base_world = torch.inverse(T_world_cam_base)

obj_in_cam = torch.bmm(T_cam_base_world, object_pos_world_h.unsqueeze(-1)).squeeze(-1)[:, :3]
dx, dy, dz = obj_in_cam[:, 0], obj_in_cam[:, 1], obj_in_cam[:, 2]
pan = torch.atan2(dx, dz)
tilt = torch.atan2(-dy, torch.sqrt(dx**2 + dz**2))

# Compute camera's final pose in world frame
# With this corrected version (assuming you want to incorporate pan/tilt):
# First create pan/tilt transform
T_cam_base_pantilt = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
# Apply rotation for pan (around y-axis)
T_cam_base_pantilt[:, 0, 0] = torch.cos(pan)
T_cam_base_pantilt[:, 0, 2] = torch.sin(pan)
T_cam_base_pantilt[:, 2, 0] = -torch.sin(pan)
T_cam_base_pantilt[:, 2, 2] = torch.cos(pan)
# Apply rotation for tilt (around x-axis)
T_tilt = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
T_tilt[:, 1, 1] = torch.cos(tilt)
T_tilt[:, 1, 2] = -torch.sin(tilt)
T_tilt[:, 2, 1] = torch.sin(tilt)
T_tilt[:, 2, 2] = torch.cos(tilt)
# Combine pan and tilt transforms
T_cam_base_pantilt = torch.bmm(T_cam_base_pantilt, T_tilt)
# Apply to get final camera transform
T_world_cam = torch.bmm(T_world_cam_base, T_cam_base_pantilt)



# Extract camera position and orientation
cam_pos_world = T_world_cam[:, :3, 3]
cam_quat_world = pk.matrix_to_quaternion(T_world_cam[:, :3, :3])

# Concatenate camera world pose and quaternion into B x 7 tensor
cam_pose_world = torch.cat([cam_pos_world, cam_quat_world], dim=1)
#print("cam_pose_world", cam_pose_world[0])

# Compute differences between camera pose and object pose
diffs = cam_pose_world - object_pose_world

# --------End of FK code to time-------------


import matplotlib.pyplot as plt
import math
# ========== Visualization ==========
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = ['blue', 'orange', 'purple']
# Only visualize 3 examples
B_vis = 3  # We'll only visualize 3 robots

def add_sphere(ax, center, radius=0.05, color='cyan', alpha=0.1, resolution=20):
    """Add a transparent sphere at the given 3D center."""
    u, v = torch.linspace(0, 2 * math.pi, resolution), torch.linspace(0, math.pi, resolution)
    u, v = torch.meshgrid(u, v, indexing='ij')
    x = radius * torch.cos(u) * torch.sin(v) + center[0]
    y = radius * torch.sin(u) * torch.sin(v) + center[1]
    z = radius * torch.cos(v) + center[2]
    ax.plot_surface(x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy(), color=color, alpha=alpha, linewidth=0, antialiased=True)

# Calculate camera direction from the rotation matrix (z-axis is forward for typical cameras)
cam_dir_world = T_world_cam[:B_vis, :3, :3] @ torch.tensor([0, 0, 1], dtype=dtype, device=device)

for i in range(B_vis):
    rx, ry, rt = robot_states[i].cpu()
    cx, cy, cz = cam_pos_world[i].cpu()
    dx, dy, dz = cam_dir_world[i].cpu()
    
    # Plot robot base
    ax.scatter(rx, ry, 0, color=colors[i], s=80, label=f'Robot {i} Base')
    ax.quiver(rx, ry, 0, 0.4 * math.cos(rt), 0.4 * math.sin(rt), 0, color=colors[i])
    
    # Plot camera position
    ax.scatter(cx, cy, cz, color=colors[i], marker='^', s=80, label=f'Camera {i}')
    center = torch.tensor([cx.item(), cy.item(), cz.item()], dtype=dtype)
    add_sphere(ax, center=center, radius=0.07, color=colors[i])
    
    # Plot camera direction - using the properly calculated direction vector
    ax.quiver(cx, cy, cz, dx * 0.4, dy * 0.4, dz * 0.4, color=colors[i])
    
    # Line from camera to object
    ax.plot(
        [cx, object_pos_world_h[0, 0].cpu().item()],
        [cy, object_pos_world_h[0, 1].cpu().item()],
        [cz, object_pos_world_h[0, 2].cpu().item()],
        linestyle='--',
        color=colors[i],
        linewidth=1.5,
        alpha=0.6
    )

import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def add_cylinder(ax, start, end, radius, color, alpha=0.1, resolution=20):
    """Add a vertical cylinder from start to end (3D points)"""
    # Cylinder from trimesh
    start = start.cpu()
    end = end.cpu()
    height = (end - start).norm().item()
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=resolution)
    cylinder.apply_translation([0, 0, height / 2])  # shift so bottom is at origin
   
    # Translate to start point
    cylinder.apply_translation(start.cpu().numpy())
    # Extract faces for plotting
    for face in cylinder.faces:
        tri = cylinder.vertices[face]
        poly = Poly3DCollection([tri], alpha=alpha, facecolor=color, edgecolor='k', linewidths=0.2)
        ax.add_collection3d(poly)

# Add vertical cylinder per robot
for i in range(B_vis):
    x, y = robot_states[i, 0].item(), robot_states[i, 1].item()
    z_cam = cam_pos_world[i, 2].item()
    base_pos = torch.tensor([x, y, 0.0], dtype=dtype)
    top_pos = torch.tensor([x, y, z_cam], dtype=dtype)
    add_cylinder(ax, base_pos, top_pos, radius=0.1, color=colors[i])

# Object and table visualization
table_center = object_pos_world_h[0, :3].cpu().numpy()
table_width = 0.6   # x dimension
table_depth = 0.8   # y dimension
table_height = 0.4  # z dimension (top surface at object height)

# Bottom corner (bar3d needs lower corner and dimensions)
table_x = table_center[0] - table_width / 2
table_y = table_center[1] - table_depth / 2
table_z = 0.0  # table sits on ground
ax.bar3d(
    table_x, table_y, table_z,
    table_width, table_depth, table_height,
    color='saddlebrown', alpha=0.4, shade=True, edgecolor='black'
)

# Object position
ax.scatter(*table_center, color='red', s=100, label='Target Object')

obstacle_center = table_center
obstacle_width = 0.15
obstacle_depth = 0.15
obstacle_height = 0.15  # height above the table

# Base of the obstacle is on top of the table
obstacle_x = obstacle_center[0] - obstacle_width / 2
obstacle_y = obstacle_center[1] - obstacle_depth / 2
obstacle_z = 0.4  # top of table
ax.bar3d(
    obstacle_x, obstacle_y, obstacle_z,
    obstacle_width, obstacle_depth, obstacle_height,
    color='red', alpha=0.15, shade=True, edgecolor='black'
)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('State Projection Visualization')

# Only use the first few labels in the legend (to avoid duplicates)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

plt.tight_layout()
plt.show()
