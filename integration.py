import time
import torch
import cuPRM

import numpy as np

from fk import fk
from nn import percscorenn as psnn
from nn import trt_inference as trti

def warmup_fk_function(nodes, object_pose_world, warmup_size=1000):
    """Run a small warmup to compile CUDA kernels"""
    device = nodes.device
    dtype = nodes.dtype
    print(f"Warmup FK function on device: {device}, dtype: {dtype}\n")
    # Create small dummy tensors
    dummy_nodes = torch.zeros(warmup_size, 5, dtype=dtype, device=device)
    dummy_object = torch.zeros(7, dtype=dtype, device=device)
    
    # Run your FK function once to trigger compilation
    _, _, _ = fk.calculate_pan_tilt_for_nodes(dummy_nodes, dummy_object)
    
    # Clear cache if needed
    torch.cuda.empty_cache()

def warmup_trt_model(model_trt, warmup_size=1000):
    """Run a small warmup to compile TRT model"""
    device = 'cuda'
    dtype = torch.float32
    print(f"Warmup TRT model on device: {device}, dtype: {dtype}\n")
    # Create dummy input tensor
    dummy_input = torch.zeros((warmup_size, 7 + 3), dtype=dtype, device=device)
    
    # Run the model once to trigger compilation
    _ = model_trt(dummy_input)
    
    # Clear cache if needed
    torch.cuda.empty_cache()


def test_prm():

    # Set device and dtype
    device = 'cuda'
    dtype = torch.float32

    # Create sample environment
    circles = torch.tensor([
        [2.0, 2.0, 0.5],  # x, y, radius
        [8.0, 8.0, 1.0],
        [5.0, 5.0, 0.3],
        [3.0, 7.0, 0.4],
        [7.0, 3.0, 0.6],
        [1.0, 9.0, 0.2],
        [9.0, 2.0, 0.5],
        [6.0, 6.0, 0.4],
        [4.0, 1.0, 0.3],
        [2.5, 3.5, 0.2],
    ], dtype=torch.float32, device='cuda')
    
    rectangles = torch.tensor([
        [4.0, 4.0, 1.0, 2.0],  # x, y, height, width
        [6.0, 2.0, 0.5, 1.5],
        [2.0, 8.0, 1.5, 1.0],
        [9.0, 1.0, 2.0, 1.0],
        [3.0, 6.0, 1.0, 2.0],
        [8.0, 4.0, 1.0, 1.0],
        [1.0, 5.0, 1.0, 1.0],
        [7.0, 7.0, 1.5, 1.5],
        [5.0, 3.0, 1.0, 2.0],
        [2.0, 2.0, 1.0, 1.0],
    ], dtype=torch.float32, device='cuda')
    
    # Define bounds [lower, upper] for [x, y, theta, pan, tilt]
    bounds = torch.tensor([
        [0.0, 0.0, -3.14159, 0.0, 0.0],  # lower bounds
        [10.0, 10.0, 3.14159, 0.0, 0.0]  # upper bounds
    ], dtype=dtype, device=device)

    object_pose_world = torch.tensor([2.5, 3.0, 0.5, 0.0, 0.0, 0.0, 1.0], dtype=dtype, device=device)

    model_state_path = "/home/lenman/capstone/parallelrm/resources/models/percscore-nov12-50k.pt"
    model_trt = trti.build_trt_from_dict(model_state_path, batch_size=100000)

    # Warmup FK function to compile CUDA kernels
    warmup_fk_function(circles, object_pose_world, warmup_size=100)
    warmup_trt_model(model_trt, warmup_size=100)


    t1 = time.time()
    # Build PRM
    nodes, node_validity, neighbors, edges, edge_validity = cuPRM.build_prm(
        circles, rectangles, bounds, seed=12345
    )

    # print(f"PRM build time: {time.time() - t1:.4f} seconds\n")
    # t2 = time.time()

    nodes_updated, cam_pose_world, diffs = fk.calculate_pan_tilt_for_nodes(nodes, object_pose_world)
    diffs = torch.cat((diffs, torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).unsqueeze(0).repeat(nodes.shape[0], 1)), dim=1)
    
    # print(f"FK calculation time: {time.time() - t2:.4f} seconds\n")
    # t3 = time.time()

    output = model_trt(diffs)

    # print("Inference time:", time.time() - t3, "seconds\n")
    print("Total time:", time.time() - t1, "seconds\n")
    print("output > 0.2", torch.sum(output > 0.2).item())

if __name__ == "__main__":
    test_prm()