import torch
import pytorch_kinematics as pk

def calculate_pan_tilt_for_nodes(nodes, object_pose_world):
    """
    Calculate pan and tilt angles for robot states to look at target object,
    then compute camera poses and diffs from object.
    
    Args:
        nodes: tensor of shape [NUM_STATES, 5] with [x, y, theta, pan, tilt]
        object_pose_world: tensor [7] with [x, y, z, qx, qy, qz, qw]
    
    Returns:
        nodes_updated: tensor with updated pan and tilt values
        cam_pose_world: tensor [NUM_STATES, 7] with camera poses in world frame
        diffs: tensor [NUM_STATES, 7] with differences between camera and object poses
    """
    device = nodes.device
    dtype = nodes.dtype
    B = nodes.shape[0]
    
    # Camera configuration (same as original script)
    cam_pos_robot = torch.tensor([0.0450, 0.0523, 1.2607], dtype=dtype, device=device)
    cam_rot_robot = torch.tensor(
        [[0.0, 0.0, 1.0],
         [1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0]], dtype=dtype, device=device
    )
    
    # Extract robot states from nodes
    robot_states = nodes[:, :3]  # [x, y, theta]
    
    # Prepare object position in homogeneous coordinates
    object_pos_world_h = torch.cat([object_pose_world[:3], torch.ones(1, dtype=dtype, device=device)])
    object_pos_world_h = object_pos_world_h.unsqueeze(0).expand(B, -1)
    
    # Create batched camera base transforms in robot frame
    T_robot_camera_base = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
    T_robot_camera_base[:, :3, :3] = cam_rot_robot.unsqueeze(0).repeat(B, 1, 1)
    T_robot_camera_base[:, :3, 3] = cam_pos_robot.unsqueeze(0).repeat(B, 1)
    
    # Build robot base-to-world transforms
    cos_theta = torch.cos(robot_states[:, 2])
    sin_theta = torch.sin(robot_states[:, 2])
    
    T_world_robot = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
    T_world_robot[:, 0, 0] = cos_theta
    T_world_robot[:, 0, 1] = -sin_theta
    T_world_robot[:, 1, 0] = sin_theta
    T_world_robot[:, 1, 1] = cos_theta
    T_world_robot[:, 0, 3] = robot_states[:, 0]
    T_world_robot[:, 1, 3] = robot_states[:, 1]
    
    # Camera base pose in world
    T_world_cam_base = torch.bmm(T_world_robot, T_robot_camera_base)
    
    # Invert camera transforms and transform object to camera frame
    T_cam_base_world = torch.inverse(T_world_cam_base)
    obj_in_cam = torch.bmm(T_cam_base_world, object_pos_world_h.unsqueeze(-1)).squeeze(-1)[:, :3]
    
    # Calculate pan and tilt angles
    dx, dy, dz = obj_in_cam[:, 0], obj_in_cam[:, 1], obj_in_cam[:, 2]
    pan = torch.atan2(dx, dz)
    tilt = torch.atan2(-dy, torch.sqrt(dx**2 + dz**2))
    
    # Create updated nodes tensor with new pan and tilt values
    nodes_updated = nodes.clone()
    nodes_updated[:, 3] = pan   # Update pan
    nodes_updated[:, 4] = tilt  # Update tilt
    
    # Now compute the final camera pose in world frame with pan/tilt applied
    # Create pan/tilt transform
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
    
    # Apply to get final camera transform in world frame
    T_world_cam = torch.bmm(T_world_cam_base, T_cam_base_pantilt)
    
    # Extract camera position and orientation
    cam_pos_world = T_world_cam[:, :3, 3]
    cam_quat_world = pk.matrix_to_quaternion(T_world_cam[:, :3, :3])
    
    # Concatenate camera world pose and quaternion into B x 7 tensor
    cam_pose_world = torch.cat([cam_pos_world, cam_quat_world], dim=1)
    
    # Compute differences between camera pose and object pose
    diffs = cam_pose_world - object_pose_world
    
    return nodes_updated, cam_pose_world, diffs

# # Example usage:
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     dtype = torch.float32
    
#     # Example nodes tensor
#     nodes = torch.tensor([
#         [ 4.0197,  7.4258, -1.4614,  0.0000,  0.0000],
#         [ 9.2034,  3.9005,  2.1030,  0.0000,  0.0000],
#         [ 1.1662,  7.0980, -0.1769,  0.0000,  0.0000],
#         [ 9.7511,  8.9559, -1.7222,  0.0000,  0.0000],
#         [ 0.7213,  0.4114, -1.2706,  0.0000,  0.0000]
#     ], dtype=dtype, device=device)
    
#     # Example object pose [x, y, z, qx, qy, qz, qw]
#     object_pose_world = torch.tensor([2.5, 3.0, 0.5, 0.0, 0.0, 0.0, 1.0], dtype=dtype, device=device)
    
#     # Calculate updated pan and tilt
#     nodes_updated, cam_pose_world, diffs = calculate_pan_tilt_for_nodes(nodes, object_pose_world)
    
#     print("Original nodes:")
#     print(nodes)
#     print("\nUpdated nodes with calculated pan/tilt:")
#     print(nodes_updated)
#     print(f"\nPan angles: {nodes_updated[:, 3]}")
#     print(f"Tilt angles: {nodes_updated[:, 4]}")
#     print(f"\nCamera poses in world frame:")
#     print(cam_pose_world)
#     print(f"\nDifferences between camera and object poses:")
#     print(diffs)