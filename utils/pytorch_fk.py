import pytorch_kinematics as pk
import torch
from time import time

import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
urdf = "/home/lenman/capstone/parallelrm/resources/robot/stretch/stretch_spherized.urdf"

chain = pk.build_chain_from_urdf(open(urdf, mode='rb').read())
chain.print_tree()

# chain = pk.build_serial_chain_from_urdf(open(urdf, mode='rb').read(), "link_head_nav_cam", "base_link")
# chain = chain.to(device=device, dtype=dtype)

# th = [0.65, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# J = chain.jacobian(th)
# print("Jacobian shape:", J.shape)
# print("Jacobian:", J)


# N = 10000
# th_batch = torch.randn(N, 8, device=device, dtype=dtype)

# for i in range(3):     
#     tg_batch = chain.forward_kinematics(th_batch)


# start = time()
# tg_batch = chain.forward_kinematics(th_batch)
# end = time()
# print("Batch forward kinematics time:", end - start)

print(chain.get_joint_parameter_names())

th = [0.5, 0.4]
ret = chain.forward_kinematics(th, end_only=False)
tg = ret['link_head_nav_cam']
m = tg.get_matrix()
pos = m[:, :3, 3]
rot = pk.matrix_to_quaternion(m[:, :3, :3])

print("Position:", pos)
print("Rotation:", rot)

import numpy as np
from scipy.spatial.transform import Rotation as R

# Example camera pose in the robot frame provided by your FK library.
# Camera position in the robot frame (x, y, z)
cam_pos_robot = pos.cpu().numpy()

# Camera orientation in the robot frame as a quaternion in the [x, y, z, w] order.
cam_quat_robot = rot.cpu().numpy()

# Convert the quaternion to a 3x3 rotation matrix.
rot_cam_robot = R.from_quat(cam_quat_robot).as_matrix()

# Build the homogeneous transformation matrix for the camera in the robot frame.
T_robot_camera = np.eye(4)
T_robot_camera[:3, :3] = rot_cam_robot
T_robot_camera[:3, 3] = cam_pos_robot

# Define the robot's pose in the world frame.
# For example: x=1.0 m, y=2.0 m, theta=30 degrees (converted to radians)
robot_x = 1.0
robot_y = 2.0
robot_theta = np.pi / 4  # 45 degrees in radians

# Create the robot-to-world transformation matrix.
T_world_robot = np.eye(4)
cos_theta = np.cos(robot_theta)
sin_theta = np.sin(robot_theta)
T_world_robot[:3, :3] = np.array([
    [cos_theta, -sin_theta, 0],
    [sin_theta,  cos_theta, 0],
    [0,          0,         1]
])
T_world_robot[:3, 3] = [robot_x, robot_y, 0]

# Chain the transformations: world -> robot -> camera.
T_world_camera = T_world_robot @ T_robot_camera

# Extract the camera's position in the world frame.
cam_pos_world = T_world_camera[:3, 3]

print("Camera position in the world frame:", cam_pos_world)
