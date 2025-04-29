import torch
import pytorch_kinematics as pk
import sys
import os
import statistics

sys.path.append("/home/lb73/cudaPRM/nn")
import trt_inference as trt

def run_timing_for_batch_size(batch_size, num_iterations=100, num_warmup=10):
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
    
    # Generate random robot states - create once for consistent testing
    xy = (torch.rand(B, 2, dtype=dtype, device=device) - 0.5) * 10.0
    theta = (torch.rand(B, 1, dtype=dtype, device=device) * 2 - 1) * torch.pi
    robot_states = torch.cat([xy, theta], dim=1)
    
    # Create CUDA events for more accurate GPU timing
    fk_start_event = torch.cuda.Event(enable_timing=True)
    fk_end_event = torch.cuda.Event(enable_timing=True)
    nn_start_event = torch.cuda.Event(enable_timing=True)
    nn_end_event = torch.cuda.Event(enable_timing=True)
    
    # Storage for timing results
    fk_times = []
    nn_times = []
    
    # Create CUDA stream for better performance isolation
    stream = torch.cuda.Stream()
    
    # Warmup runs to ensure GPU is initialized and JIT compilation is done
    print(f"Performing {num_warmup} warmup iterations...")
    with torch.cuda.stream(stream):
        for _ in range(num_warmup):
            # FK warmup
            T_robot_camera_base = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
            T_robot_camera_base[:, :3, :3] = cam_rot_robot.unsqueeze(0).repeat(B, 1, 1)
            T_robot_camera_base[:, :3, 3] = cam_pos_robot.unsqueeze(0).repeat(B, 1)
            
            cos_theta = torch.cos(robot_states[:, 2])
            sin_theta = torch.sin(robot_states[:, 2])
            T_world_robot = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
            T_world_robot[:, 0, 0] = cos_theta
            T_world_robot[:, 0, 1] = -sin_theta
            T_world_robot[:, 1, 0] = sin_theta
            T_world_robot[:, 1, 1] = cos_theta
            T_world_robot[:, 0, 3] = robot_states[:, 0]
            T_world_robot[:, 1, 3] = robot_states[:, 1]
            
            T_world_cam_base = torch.bmm(T_world_robot, T_robot_camera_base)
            T_cam_base_world = torch.inverse(T_world_cam_base)
            obj_in_cam = torch.bmm(T_cam_base_world, object_pos_world_h.unsqueeze(-1)).squeeze(-1)[:, :3]
            dx, dy, dz = obj_in_cam[:, 0], obj_in_cam[:, 1], obj_in_cam[:, 2]
            pan = torch.atan2(dx, dz)
            tilt = torch.atan2(-dy, torch.sqrt(dx**2 + dz**2))
            
            T_world_cam = torch.bmm(T_world_cam_base, T_robot_camera_base)
            cam_pos_world = T_world_cam[:, :3, 3]
            cam_quat_world = pk.matrix_to_quaternion(T_world_cam[:, :3, :3])
            cam_pose_world = torch.cat([cam_pos_world, cam_quat_world], dim=1)
            diffs = cam_pose_world - object_pose_world
            
            # NN warmup
            diffs_input = torch.cat((diffs, torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).unsqueeze(0).repeat(B, 1)), dim=1)
            output = model_trt(diffs_input)
    
    # Wait for warmup to complete
    torch.cuda.synchronize()
    print(f"Warmup complete. Starting {num_iterations} timed iterations...")
    
    # Actual timing runs
    with torch.cuda.stream(stream):
        for i in range(num_iterations):
            # -------- FK timing --------
            torch.cuda.synchronize()  # Ensure previous operations are completed
            fk_start_event.record()
            
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
            
            fk_end_event.record()
            
            # -------- NN Inference timing --------
            # Prepare input for the model
            diffs_input = torch.cat((diffs, torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).unsqueeze(0).repeat(B, 1)), dim=1)
            
            nn_start_event.record()
            
            output = model_trt(diffs_input)
            
            nn_end_event.record()
            
            # Wait for this iteration to complete
            torch.cuda.synchronize()
            
            # Record times in milliseconds, convert to seconds
            fk_times.append(fk_start_event.elapsed_time(fk_end_event) / 1000)
            nn_times.append(nn_start_event.elapsed_time(nn_end_event) / 1000)
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_iterations} iterations")
    
    # Calculate statistics
    avg_fk_time = sum(fk_times) / len(fk_times)
    avg_nn_time = sum(nn_times) / len(nn_times)
    median_fk_time = statistics.median(fk_times)
    median_nn_time = statistics.median(nn_times)
    stddev_fk_time = statistics.stdev(fk_times) if len(fk_times) > 1 else 0
    stddev_nn_time = statistics.stdev(nn_times) if len(nn_times) > 1 else 0
    min_fk_time = min(fk_times)
    min_nn_time = min(nn_times)
    max_fk_time = max(fk_times)
    max_nn_time = max(nn_times)
    
    # Create timing directory if it doesn't exist
    os.makedirs("timing_results2", exist_ok=True)
    
    # Write results to a file
    with open(f"timing_results2/batch_size_{batch_size}.txt", "w") as f:
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Number of iterations: {num_iterations} (after {num_warmup} warmup runs)\n\n")
        
        f.write("=== Forward Kinematics Timing ===\n")
        f.write(f"Average: {avg_fk_time:.6f} seconds\n")
        f.write(f"Median: {median_fk_time:.6f} seconds\n")
        f.write(f"Std Dev: {stddev_fk_time:.6f} seconds\n")
        f.write(f"Min: {min_fk_time:.6f} seconds\n")
        f.write(f"Max: {max_fk_time:.6f} seconds\n\n")
        
        f.write("=== NN Inference Timing ===\n")
        f.write(f"Average: {avg_nn_time:.6f} seconds\n")
        f.write(f"Median: {median_nn_time:.6f} seconds\n")
        f.write(f"Std Dev: {stddev_nn_time:.6f} seconds\n")
        f.write(f"Min: {min_nn_time:.6f} seconds\n")
        f.write(f"Max: {max_nn_time:.6f} seconds\n\n")
        
        f.write(f"Total Average Time: {avg_fk_time + avg_nn_time:.6f} seconds\n\n")
        
        f.write("--- Detailed Timing ---\n")
        f.write("Iteration | FK Time (s) | NN Time (s) | Total (s)\n")
        f.write("-" * 50 + "\n")
        for i in range(num_iterations):
            f.write(f"{i+1:^9} | {fk_times[i]:.6f} | {nn_times[i]:.6f} | {fk_times[i] + nn_times[i]:.6f}\n")
    
    print(f"\nCompleted timing for batch size {batch_size}")
    print(f"Average FK Time: {avg_fk_time:.6f} seconds")
    print(f"Average NN Time: {avg_nn_time:.6f} seconds")
    print(f"Results saved to timing_results2/batch_size_{batch_size}.txt")
    
    return avg_fk_time, avg_nn_time, median_fk_time, median_nn_time, stddev_fk_time, stddev_nn_time

def main():
    # Define the batch sizes to test
    batch_sizes = batch_sizes = [1000, 10000, 20000, 50000, 100000, 200000, 500000]  # Adjust as needed
    
    # Number of iterations for timing
    num_iterations = 100
    num_warmup = 10
    
    # Create a summary file
    with open("timing_results2/summary.txt", "w") as f:
        f.write(f"Performance Timing Summary (After {num_warmup} warmup iterations)\n")
        f.write("=" * 80 + "\n\n")
        f.write("| Batch Size | FK Time (s) | NN Time (s) | Total (s) | FK Median (s) | NN Median (s) | FK StdDev | NN StdDev |\n")
        f.write("|" + "-" * 11 + "|" + "-" * 12 + "|" + "-" * 13 + "|" + "-" * 11 + "|" + "-" * 15 + "|" + "-" * 15 + "|" + "-" * 11 + "|" + "-" * 11 + "|\n")
    
    # Run timing for each batch size
    for batch_size in batch_sizes:
        print(f"\nRunning timing for batch size {batch_size}...")
        try:
            avg_fk, avg_nn, med_fk, med_nn, std_fk, std_nn = run_timing_for_batch_size(
                batch_size, num_iterations=num_iterations, num_warmup=num_warmup
            )
            
            # Append to summary file
            with open("timing_results2/summary.txt", "a") as f:
                f.write(f"| {batch_size:^9} | {avg_fk:.6f} | {avg_nn:.6f} | {avg_fk + avg_nn:.6f} | {med_fk:.6f} | {med_nn:.6f} | {std_fk:.6f} | {std_nn:.6f} |\n")
                
        except Exception as e:
            print(f"Error with batch size {batch_size}: {str(e)}")
            with open("timing_results2/summary.txt", "a") as f:
                f.write(f"| {batch_size:^9} | Error: {str(e)} | | | | | | |\n")
    
    

if __name__ == "__main__":
    main()