import torch
import math
import kornia.geometry.conversions as kornia_conv
import pytorch_kinematics as pk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
urdf = "/home/lb73/cudaPRM/resources/robot/stretch/stretch_spherized.urdf"

# Load robot kinematics chain
chain = pk.build_serial_chain_from_urdf(open(urdf, mode='rb').read(), "link_head_nav_cam", "base_link")
chain = chain.to(device=device, dtype=dtype)

# Get camera mount transform in robot frame
th = [0.0, 0.0]  # pan, tilt at 0 for base pose
ret = chain.forward_kinematics(th, end_only=False)
tg = ret['link_head_nav_cam']
m = tg.get_matrix()
pos = m[:, :3, 3]
rot = pk.matrix_to_quaternion(m[:, :3, :3])
cam_base_rot = kornia_conv.quaternion_to_rotation_matrix(rot)

# Define a batch of robot base states
robot_states = torch.tensor([
    [1.0, 2.0, math.radians(60)],
    [0.0, 0.0, math.radians(0)],
    [2.0, -1.0, math.radians(90)]
], dtype=dtype, device=device)  # shape: (B, 3)

B = robot_states.shape[0]

# Object in world frame (same for all robots)
object_pos_world = torch.tensor([2.5, 3.0, 0.5], dtype=dtype, device=device)

# Static camera mount transform in robot frame
T_robot_camera_base = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
T_robot_camera_base[:, :3, :3] = cam_base_rot
T_robot_camera_base[:, :3, 3] = pos

# Step 1: Build robot base-to-world transforms for each robot
cos_theta = torch.cos(robot_states[:, 2])
sin_theta = torch.sin(robot_states[:, 2])
T_world_robot = torch.eye(4, dtype=dtype, device=device).repeat(B, 1, 1)
T_world_robot[:, 0, 0] = cos_theta
T_world_robot[:, 0, 1] = -sin_theta
T_world_robot[:, 1, 0] = sin_theta
T_world_robot[:, 1, 1] = cos_theta
T_world_robot[:, :2, 3] = robot_states[:, :2]

# Step 2: Camera base pose in world
T_world_cam_base = torch.bmm(T_world_robot, T_robot_camera_base)

# Step 3: Transform object into each camera base frame
obj_world_h = torch.cat([object_pos_world, torch.tensor([1.0], dtype=dtype, device=device)])
obj_world_h = obj_world_h.unsqueeze(0).repeat(B, 1)  # shape: (B, 4)
T_cam_base_world = torch.linalg.inv(T_world_cam_base)
obj_in_cam = torch.bmm(T_cam_base_world, obj_world_h.unsqueeze(-1)).squeeze(-1)[:, :3]

dx, dy, dz = obj_in_cam[:, 0], obj_in_cam[:, 1], obj_in_cam[:, 2]
pan = torch.atan2(dy, dx)
tilt = torch.atan2(-dz, torch.sqrt(dx**2 + dy**2))

# Step 4: Compute camera direction in world frame
def make_rot_matrix(pan, tilt):
    B = pan.shape[0]
    Ry = torch.eye(3, device=device, dtype=dtype).repeat(B, 1, 1)
    Rz = torch.eye(3, device=device, dtype=dtype).repeat(B, 1, 1)
    
    cos_tilt = torch.cos(tilt)
    sin_tilt = torch.sin(tilt)
    cos_pan = torch.cos(pan)
    sin_pan = torch.sin(pan)
    
    Ry[:, 0, 0] = cos_tilt
    Ry[:, 0, 2] = sin_tilt
    Ry[:, 2, 0] = -sin_tilt
    Ry[:, 2, 2] = cos_tilt

    Rz[:, 0, 0] = cos_pan
    Rz[:, 0, 1] = -sin_pan
    Rz[:, 1, 0] = sin_pan
    Rz[:, 1, 1] = cos_pan

    return torch.bmm(Rz, Ry)

R_pan_tilt = make_rot_matrix(pan, tilt)
T_world_camera = T_world_cam_base.clone()
T_world_camera[:, :3, :3] = torch.bmm(T_world_cam_base[:, :3, :3], R_pan_tilt)
cam_pos_world = T_world_camera[:, :3, 3]
cam_dir_world = T_world_camera[:, :3, 0]  # x-axis direction



# ========== Visualization ==========
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['blue', 'orange', 'purple']

def add_sphere(ax, center, radius=0.05, color='cyan', alpha=0.1, resolution=20):
    """Add a transparent sphere at the given 3D center."""
    u, v = torch.linspace(0, 2 * math.pi, resolution), torch.linspace(0, math.pi, resolution)
    u, v = torch.meshgrid(u, v, indexing='ij')

    x = radius * torch.cos(u) * torch.sin(v) + center[0]
    y = radius * torch.sin(u) * torch.sin(v) + center[1]
    z = radius * torch.cos(v) + center[2]

    ax.plot_surface(x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy(), color=color, alpha=alpha, linewidth=0, antialiased=True)


for i in range(B):
    rx, ry, rt = robot_states[i].cpu()
    cx, cy, cz = cam_pos_world[i].cpu()
    dx, dy, dz = cam_dir_world[i].cpu()
    ax.scatter(rx, ry, 0, color=colors[i], s=80, label=f'Robot {i} Base')
    ax.quiver(rx, ry, 0, 0.4 * math.cos(rt), 0.4 * math.sin(rt), 0, color=colors[i])
    ax.scatter(cx, cy, cz, color=colors[i], marker='^', s=80, label=f'Camera {i}')
    add_sphere(ax, center=torch.tensor([cx, cy, cz]), radius=0.07, color=colors[i])
    ax.quiver(cx, cy, cz, dx * 0.4, dy * 0.4, dz * 0.4, color=colors[i])

    ax.plot(
        [cx, object_pos_world[0].cpu().item()],
        [cy, object_pos_world[1].cpu().item()],
        [cz, object_pos_world[2].cpu().item()],
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
for i in range(B):
    x, y = robot_states[i, 0].item(), robot_states[i, 1].item()
    z_cam = cam_pos_world[i, 2].item()

    base_pos = torch.tensor([x, y, 0.0], dtype=dtype)
    top_pos = torch.tensor([x, y, z_cam], dtype=dtype)
    add_cylinder(ax, base_pos, top_pos, radius=0.1, color=colors[i])

table_center = object_pos_world.cpu().numpy()
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
ax.scatter(*object_pos_world.cpu().numpy(), color='red', s=100, label='Target Object')

obstacle_center = object_pos_world.cpu().numpy()

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

# World frame


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('State Projection Visualization')
ax.legend()
plt.tight_layout()
plt.show()
