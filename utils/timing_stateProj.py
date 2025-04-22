import torch
import pytorch_kinematics as pk
import sys
import time
import os

sys.path.append("/home/lb73/cudaPRM/nn")
import trt_inference as trt

def run_timing_for_batch_size(batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    B = batch_size
    model_state_path = "/home/lb73/cudaPRM/resources/models/percscore-nov12-50k.pt"
    model_trt = trt.build_trt_from_dict(model_state_path, batch_size=B)
    
    # This is the pose of the camera in the robot frame when pan and tilt joints are zero.
    cam_pos_robot = torch.tensor([0.0450, 0.0523, 1.2607], dtype=dtype, device=device)
    cam_quat_robot = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=dtype, device=device)
    cam_rot_robot = torch.tensor(
        [[0.0, 0.0, 1.0],
         [1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0]], dtype=dtype, device=device
    )
    
    # Specify some object position in the world frame.
    object_pos_world_h = torch.tensor([2.5, 3.0, 0.5, 1.0], dtype=dtype, device=device).unsqueeze(0).expand(B, -1)
    zero_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=dtype, device=device)
    object_pose_world = torch.cat((object_pos_world_h[0][:3], zero_quat))
    
    # Generate random robot states
    xy = (torch.rand(B, 2, dtype=dtype, device=device) - 0.5) * 10.0
    theta = (torch.rand(B, 1, dtype=dtype, device=device) * 2 - 1) * torch.pi
    robot_states = torch.cat([xy, theta], dim=1)
    
    # Prepare for timing
    fk_times = []
    nn_times = []
    
    # Run 100 iterations to get stable timing results
    for i in range(100):
        # -------- FK timing --------
        torch.cuda.synchronize()  # Ensure CUDA operations are synchronized
        fk_start = time.time()
        
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
        T_world_cam = torch.bmm(T_world_cam_base, T_robot_camera_base)
        
        # Extract camera position and orientation
        cam_pos_world = T_world_cam[:, :3, 3]
        cam_quat_world = pk.matrix_to_quaternion(T_world_cam[:, :3, :3])
        
        # Concatenate camera world pose and quaternion into B x 7 tensor
        cam_pose_world = torch.cat([cam_pos_world, cam_quat_world], dim=1)
        
        # Compute differences between camera pose and object pose
        diffs = cam_pose_world - object_pose_world
        
        torch.cuda.synchronize()  # Ensure CUDA operations are completed
        fk_end = time.time()
        fk_times.append(fk_end - fk_start)
        
        # -------- NN Inference timing --------
        # Prepare input for the model
        diffs_input = torch.cat((diffs, torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).unsqueeze(0).repeat(B, 1)), dim=1)
        
        torch.cuda.synchronize()  # Ensure CUDA operations are synchronized
        nn_start = time.time()
        
        output = model_trt(diffs_input)
        
        torch.cuda.synchronize()  # Ensure CUDA operations are completed
        nn_end = time.time()
        nn_times.append(nn_end - nn_start)
    
    # Calculate statistics
    avg_fk_time = sum(fk_times) / len(fk_times)
    avg_nn_time = sum(nn_times) / len(nn_times)
    
    # Create timing directory if it doesn't exist
    os.makedirs("timing_results", exist_ok=True)
    
    # Write results to a file
    with open(f"timing_results/batch_size_{batch_size}.txt", "w") as f:
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Average FK Time: {avg_fk_time:.6f} seconds\n")
        f.write(f"Average NN Inference Time: {avg_nn_time:.6f} seconds\n")
        f.write(f"Total Average Time: {avg_fk_time + avg_nn_time:.6f} seconds\n")
        f.write("\n--- Detailed Timing ---\n")
        f.write("FK Times (seconds):\n")
        for i, t in enumerate(fk_times):
            f.write(f"Run {i+1}: {t:.6f}\n")
        f.write("\nNN Inference Times (seconds):\n")
        for i, t in enumerate(nn_times):
            f.write(f"Run {i+1}: {t:.6f}\n")
    
    print(f"Completed timing for batch size {batch_size}")
    print(f"Average FK Time: {avg_fk_time:.6f} seconds")
    print(f"Average NN Inference Time: {avg_nn_time:.6f} seconds")
    print(f"Results saved to timing_results/batch_size_{batch_size}.txt")
    
    return avg_fk_time, avg_nn_time

def main():
    # Define the batch sizes to test
    batch_sizes = [1000, 10000, 20000, 50000, 100000, 200000, 500000]  # Adjust as needed
    
    # Create a summary file
    with open("timing_results/summary.txt", "w") as f:
        f.write("Batch Size | FK Time (s) | NN Time (s) | Total Time (s)\n")
        f.write("-" * 60 + "\n")
    
    # Run timing for each batch size
    for batch_size in batch_sizes:
        print(f"\nRunning timing for batch size {batch_size}...")
        try:
            avg_fk_time, avg_nn_time = run_timing_for_batch_size(batch_size)
            
            # Append to summary file
            with open("timing_results/summary.txt", "a") as f:
                f.write(f"{batch_size:^10} | {avg_fk_time:.6f} | {avg_nn_time:.6f} | {avg_fk_time + avg_nn_time:.6f}\n")
                
        except Exception as e:
            print(f"Error with batch size {batch_size}: {str(e)}")
            with open("timing_results/summary.txt", "a") as f:
                f.write(f"{batch_size:^10} | Error: {str(e)}\n")

if __name__ == "__main__":
    main()