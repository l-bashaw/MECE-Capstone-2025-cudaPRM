import torch
import pytorch_kinematics as pk
import sys

# sys.path.append("/home/lb73/cudaPRM/nn")
sys.path.append("/home/lenman/capstone/parallelrm/nn")
import trt_inference as trt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

B = 100000
# model_state_path = "/home/lb73/cudaPRM/resources/models/percscore-nov12-50k.pt"
model_state_path = "/home/lenman/capstone/parallelrm/resources/models/percscore-nov12-50k.pt"
model_trt = trt.build_trt_from_dict(model_state_path, batch_size=B)


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
print("pan", pan[0:10])
print("tilt", tilt[0:10])
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



# --------Start of NN Inference code to time-------------
diffs = torch.cat((diffs, torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).unsqueeze(0).repeat(B, 1)), dim=1)
output = model_trt(diffs)
# --------End of NN Inference code to time-------------


print("output > 0.5", torch.sum(output > 0.5).item())