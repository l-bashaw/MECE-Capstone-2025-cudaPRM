import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
B = 5

cam_pos_robot = torch.tensor([0.0450, 0.0523, 1.2607], dtype=dtype, device=device)
cam_quat_robot = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=dtype, device=device)
cam_rot_robot = torch.tensor(
    [[0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]], dtype=dtype, device=device
)

object_pos_world_h = torch.tensor([2.5, 3.0, 0.5, 1.0], dtype=dtype, device=device).unsqueeze(0).expand(B, -1)  # x, y, z, w

robot_start = torch.tensor([1.0, 1.0 , 0.0           ], dtype=dtype, device=device).unsqueeze(0)
inter1 = torch.tensor(     [1.5, 1.25, torch.pi/4    ], dtype=dtype, device=device).unsqueeze(0)
inter2 = torch.tensor(     [2.0, 1.5 , torch.pi/2    ], dtype=dtype, device=device).unsqueeze(0)
inter3 = torch.tensor(     [2.5, 1.75, (3*torch.pi)/4], dtype=dtype, device=device).unsqueeze(0)
robot_goal = torch.tensor( [3.0, 2.0 , torch.pi      ], dtype=dtype, device=device).unsqueeze(0)

# Join the states into a tensor of dimension (B, 3)
robot_states = torch.cat((robot_start, inter1, inter2, inter3, robot_goal), dim=0)
print("robot_states.shape", robot_states.shape)







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

print(pan)
print()
print(tilt)

pan = pan.cpu().numpy()
tilt = tilt.cpu().numpy()

for i in range(len(pan)):
    if i == 0:
        continue
    diff_p = pan[i] - pan[i-1]
    diff_t = tilt[i] - tilt[i-1]
    print("diff_p", diff_p)
    print("diff_t", diff_t)


# Compare the pan/tilt angles calculated by interpolating states to the pan/tilt angles calculated by doing transforms on the interpolated x, y, theta.    
# # Step 1: Camera base pose in world



# # -----------------------------
# def make_rot_matrix(pan, tilt):
#     # Create batched rotation matrices for each state.
#     B = pan.shape[0]
#     cos_tilt = torch.cos(tilt)
#     sin_tilt = torch.sin(tilt)
#     cos_pan = torch.cos(pan)
#     sin_pan = torch.sin(pan)
    
#     Ry = torch.zeros((B, 3, 3), dtype=dtype, device=device)
#     Rz = torch.zeros((B, 3, 3), dtype=dtype, device=device)
    
#     # Rotation about y-axis (tilt)
#     Ry[:, 0, 0] = cos_tilt
#     Ry[:, 0, 2] = sin_tilt
#     Ry[:, 1, 1] = 1.0
#     Ry[:, 2, 0] = -sin_tilt
#     Ry[:, 2, 2] = cos_tilt
    
#     # Rotation about z-axis (pan)
#     Rz[:, 0, 0] = cos_pan
#     Rz[:, 0, 1] = -sin_pan
#     Rz[:, 1, 0] = sin_pan
#     Rz[:, 1, 1] = cos_pan
#     Rz[:, 2, 2] = 1.0
    
#     return torch.bmm(Rz, Ry)

# R_pan_tilt = make_rot_matrix(pan, tilt)
# # Compose the final camera rotation in world frame by applying pan/tilt to the camera base rotation.
# T_world_camera = T_world_cam_base.clone()
# T_world_camera[:, :3, :3] = torch.bmm(T_world_cam_base[:, :3, :3], R_pan_tilt)
# cam_pos_world = T_world_camera[:, :3, 3]
# cam_dir_world = T_world_camera[:, :3, 0]  # the camera's forward direction (x-axis)
