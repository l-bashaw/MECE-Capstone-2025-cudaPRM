import time
import torch

import numpy as np
import pytorch_kinematics as pk

from prm import PSPRM, Solution
from nn.inference import ModelLoader
from utils.EnvLoader import EnvironmentLoader

def compute_camera_diffs(nodes: np.ndarray, object_pose_world: np.ndarray):
    """
    Given robot states (with pan/tilt already filled) and an object position in world,
    compute diffs = [obj_in_cam (3), cam_quat_in_world (4)] for each node.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
   
    nodes_t = torch.tensor(nodes, dtype=dtype, device=device)
    B = nodes_t.shape[0]

    # Camera mount config (zero pan/tilt reference)
    cam_pos_robot = torch.tensor([0.0450, 0.0523, 1.2607], dtype=dtype, device=device)
    cam_rot_robot = torch.tensor(
        [[0.0, 0.0, 1.0],
         [1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0]], dtype=dtype, device=device
    )

    # Extract states
    robot_xytheta = nodes_t[:, :3]  # (x, y, theta)
    pan = nodes_t[:, 3]
    tilt = nodes_t[:, 4]

    # Object homogeneous
    object_pos_world_h = torch.cat([object_pose_world[:3], torch.ones(1, dtype=dtype, device=device)])
    object_pos_world_h = object_pos_world_h.unsqueeze(0).expand(B, -1)

    # Camera mount in robot frame
    T_robot_camera_base = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
    T_robot_camera_base[:, :3, :3] = cam_rot_robot.unsqueeze(0).repeat(B, 1, 1)
    T_robot_camera_base[:, :3, 3] = cam_pos_robot.unsqueeze(0).repeat(B, 1)

    # Robot -> world
    cos_theta = torch.cos(robot_xytheta[:, 2])
    sin_theta = torch.sin(robot_xytheta[:, 2])
    T_world_robot = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
    T_world_robot[:, 0, 0] = cos_theta
    T_world_robot[:, 0, 1] = -sin_theta
    T_world_robot[:, 1, 0] = sin_theta
    T_world_robot[:, 1, 1] = cos_theta
    T_world_robot[:, 0, 3] = robot_xytheta[:, 0]
    T_world_robot[:, 1, 3] = robot_xytheta[:, 1]

    # Camera base in world
    T_world_cam_base = torch.bmm(T_world_robot, T_robot_camera_base)

    # Pan/tilt transform
    def make_pan_tilt_transform(pan, tilt):
        B = pan.shape[0]
        T = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)

        # Pan: rotation around Y
        cos_p, sin_p = torch.cos(pan), torch.sin(pan)
        R_pan = torch.stack([
            torch.stack([ cos_p, torch.zeros_like(pan), sin_p], dim=1),
            torch.stack([ torch.zeros_like(pan), torch.ones_like(pan), torch.zeros_like(pan)], dim=1),
            torch.stack([-sin_p, torch.zeros_like(pan), cos_p], dim=1)
        ], dim=1)

        # Tilt: rotation around X
        cos_t, sin_t = torch.cos(tilt), torch.sin(tilt)
        R_tilt = torch.stack([
            torch.stack([ torch.ones_like(tilt), torch.zeros_like(tilt), torch.zeros_like(tilt)], dim=1),
            torch.stack([ torch.zeros_like(tilt), cos_t, -sin_t], dim=1),
            torch.stack([ torch.zeros_like(tilt), sin_t,  cos_t], dim=1)
        ], dim=1)

        R = torch.bmm(R_tilt, R_pan)  # tilt ∘ pan
        T[:, :3, :3] = R
        return T

    T_cam_base_cam = make_pan_tilt_transform(pan, tilt)
    T_world_cam = torch.bmm(T_world_cam_base, T_cam_base_cam)
    T_cam_world = torch.inverse(T_world_cam)

    # Object in true camera frame
    obj_in_cam = torch.bmm(T_cam_world, object_pos_world_h.unsqueeze(-1)).squeeze(-1)[:, :3]

    # Camera orientation quaternion
    R_cam_world = T_cam_world[:, :3, :3]
    cam_quat = pk.matrix_to_quaternion(R_cam_world)

    diffs = torch.cat([-obj_in_cam, cam_quat], dim=1)  # [B, 7]

    return diffs

def calculate_trajectory_perc_score(trajectory, object_pose, object_label, model):
    diffs = compute_camera_diffs(trajectory, object_pose_world=object_pose)  # [N, 7]
    diffs = torch.cat((diffs, object_label.unsqueeze(0).repeat(diffs.shape[0], 1)), dim=1)
    p_scores = model(diffs)
    
    # Move to CPU and numpy
    p_scores = p_scores.detach().cpu().numpy()
    # return stats
    return {
        'mean': float(np.mean(p_scores)),
        'std': float(np.std(p_scores)),
        'min': float(np.min(p_scores)),
        'max': float(np.max(p_scores))
    }

def main():
    device = 'cuda'
    dtype = torch.float32
    env_config_file = "/home/lb73/cudaPRM/planning/resources/scenes/environment/multigoal_demo.yaml"  
    model_path = "/home/lb73/cudaPRM/planning/resources/models/percscore-nov12-50k.pt"
    seed = 2387

    env_loader = EnvironmentLoader(device=device)
    env = env_loader.load_world(env_config_file)

    model_loader = ModelLoader(label_size=3, max_batch_size=10000, use_trt=False)
    model = model_loader.load_model(model_path)

    # ----------------------------------- Experiment parameters -----------------------------------
    starts = np.array([
        [0.5, -3, 3.14159/2, 0.0, 0.0],
        [1.0, -3, 3.14159/2, 0.0, 0.0],
        [1.5, -3, 3.14159/2, 0.0, 0.0],
        [2.0, -3, 3.14159/2, 0.0, 0.0],
        [2.5, -3, 3.14159/2, 0.0, 0.0]
    ])
    goals = np.array([
        [0.5, 2.5, -3.14159, 0.0, 0.0],
        [1.0, 2.5, -3.14159, 0.0, 0.0],
        [1.5, -2.5, -3.14159, 0.0, 0.0],
        [2.0, -2.5, -3.14159, 0.0, 0.0],
        [2.5, -3, -3.14159, 0.0, 0.0]
    ])

    MAX_SKIP_DIST = 4

    # label_map = {
    #   "human": [1, 0, 0],
    #   "monitor": [0, 1, 0],
    #   "cup": [0, 0, 1]
    # }

    # ----------------------------------- Object positions -----------------------------------

    human_pose = torch.tensor([0.2, -0.2, 1.65,   0.0, 0.0, 0.0, 1.0], dtype=dtype, device=device)
    human_label = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)

    monitor_pose = torch.tensor([1.25, 0, 0.9,   0.0, 0.0, -0.7071068, -0.7071068], dtype=dtype, device=device)
    monitor_label = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)

    cup_pose = torch.tensor([-0.5, 2.5, 0.7,   0.0, 0.0, 0.7071068, -0.7071068], dtype=dtype, device=device)
    cup_label = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)

    objects = [
        (human_pose, human_label, 'human'),
        (monitor_pose, monitor_label, 'monitor'),
        (cup_pose, cup_label, 'cup')
    ]

    # ----------------------------------- Setup Env -----------------------------------
    
    env['bounds'] = torch.concat([env['bounds'], torch.tensor([[-3.14159, 0.0, 0.0], [3.14159, 0.0, 0.0]], dtype=dtype, device=device)], dim=1)
    env['bounds'][0, 0] = -1.5  # clip them to the walls to avoid weird sampling
    env['bounds'][1, 0] = 4
    env['bounds'][0, 1] = -4
    env['bounds'][1, 1] = 4

    # ----------------------------------- Warmup Planner -----------------------------------

    print("Warming up planner...")
    env['object_pose'] = objects[0][0]
    env['object_label'] = objects[0][1]
    prm = PSPRM(model, env)
    prm.build_prm(seed)
    s_id, g_id = prm.addStartAndGoal(starts[0], goals[0])
    path = prm.a_star_search(start_id=s_id, goal_id=g_id, alpha=1, beta=1)
    sol = Solution(path)
    sol.simplify(prm, env, max_skip_dist=MAX_SKIP_DIST)
    trajectory = sol.generate_trajectory_rsplan(prm, turning_radius=1)      
    trajectory = sol.project_trajectory(env['object_pose'])
    print("Planner warmup complete.")

    # ------------------------------------------------------------------------------------------
    # ----------------------------------- Expierment/Planning ----------------------------------
    # ------------------------------------------------------------------------------------------
    
    trajectories = {}
    
    # Set object in env
    for obj in objects:  # human, monitor, cup
        print(f"Planning for {obj[2]}")
        env['object_pose'] = obj[0]
        env['object_label'] = obj[1]
        obj_name = obj[2]
        trajectories[obj_name] = []
        
        # Plan for each start/goal pair
        for start in starts:
            for goal in goals:
                times = []
                for i in range(100):  # 100 trials per start/goal pair, 
                    t1 = time.time()
                    # Build PRM and add start and goal
                    prm = PSPRM(model, env)
                    prm.build_prm(seed)
                    s_id, g_id = prm.addStartAndGoal(start, goal)

                    # Find the path and simplify it
                    path = prm.a_star_search(start_id=s_id, goal_id=g_id, alpha=1, beta=1)
                    sol = Solution(path)
                    sol.simplify(prm, env, max_skip_dist=MAX_SKIP_DIST)

                    # Generate trajectory and project it to the object
                    trajectory = sol.generate_trajectory_rsplan(prm, turning_radius=1)
                    trajectory = sol.project_trajectory(env['object_pose'])
                    t2 = time.time()
                    times.append(t2 - t1)

                times = np.array(times)
                timing_stats = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times)
                }
                # print(timing_stats)

                trajectory_perc_score = calculate_trajectory_perc_score(trajectory, env['object_pose'], env['object_label'], model=model)
                trajectories[obj_name].append({
                    "start": start.tolist(),
                    "goal": goal.tolist(),
                    # "trajectory": trajectory.tolist(), 
                    "timing": timing_stats, 
                    "perc_score": trajectory_perc_score
                })  

    # Save trajectories to json
    import json
    with open('./comparison_experiment_trajectories.json', 'w') as f:
        json.dump(trajectories, f, indent=4)

    return

    
    

if __name__ == "__main__":
    main()


