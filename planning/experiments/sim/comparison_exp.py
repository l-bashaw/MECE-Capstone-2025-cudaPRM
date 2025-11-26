import os
import time
import torch

import numpy as np
import pytorch_kinematics as pk

from prm import PSPRM, Solution
from nn.inference import ModelLoader
from utils.EnvLoader import EnvironmentLoader


def calculate_trajectory_perc_score(prm: PSPRM, trajectory, object_pose, object_label, model):
    trajectory = torch.tensor(trajectory, dtype=torch.float32, device='cuda')  # (N, 5)
    model_input = prm.create_model_input(trajectory, object_pose, object_label)
    # print(model_input)
    # print any NaN values in model_input
    # if torch.any(torch.isnan(model_input)):
    #     print("Warning: NaN values in model input")
    #     # print the nan values
    #     print("NaN values:", model_input[torch.isnan(model_input)])
    p_scores = model(model_input)

    # print(p_scores)
    # Move to CPU and numpy
    p_scores = p_scores.detach().cpu().numpy()
    # print(p_scores)

    # Catch if any NaN values
    # if np.any(np.isnan(p_scores)):
    #     print("Warning: NaN values in perc scores, setting to 0")
    #     # print number of NaN values
    #     print("Number of NaN values:", np.sum(np.isnan(p_scores)))
    #     # print the nan values
    #     print("NaN values:", p_scores[np.isnan(p_scores)])
        # print(p_scores)   # return stats
    return {
        'mean': float(np.mean(p_scores)),
        'std': float(np.std(p_scores)),
        'min': float(np.min(p_scores)),
        'max': float(np.max(p_scores)),
        'sum': float(np.sum(p_scores))
    }

def main():
    device = 'cuda'
    dtype = torch.float32
    env_config_file = "/home/lb73/cudaPRM/planning/resources/scenes/environment/multigoal_demo.yaml"  
    model_path = "/home/lb73/cudaPRM/planning/resources/models/percscore-nov12-50k.pt"
    SEED = 2387

    env_loader = EnvironmentLoader(device=device)
    env = env_loader.load_world(env_config_file)

    model_loader = ModelLoader(label_size=3, max_batch_size=10000, use_trt=False)
    MODEL = model_loader.load_model(model_path)

    # ----------------------------------- Experiment parameters -----------------------------------
    starts = np.array([
        [2.826, -2.448,  1.021, 0.0, 0.0],
        [0.533, -3.432,  2.610, 0.0, 0.0],
        [2.037, -3.141,  2.309, 0.0, 0.0],
        [0.645, -2.227,  2.928, 0.0, 0.0],
        [2.681, -2.115, -1.097, 0.0, 0.0],
    ])

    goals = np.array([
        [2.757,  2.963, -0.903, 0.0, 0.0],
        [2.215,  2.920, -0.072, 0.0, 0.0],
        [2.861,  3.286, -0.013, 0.0, 0.0],
        [0.378,  1.193,  1.093, 0.0, 0.0],
        [-0.273, 0.668,  2.012, 0.0, 0.0],
    ])

    MAX_SKIP_DIST = 1.5

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
        # (human_pose, human_label, 'human'),
        # (monitor_pose, monitor_label, 'monitor'),
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
    prm = PSPRM(MODEL, env)
    prm.build_prm(SEED)
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
    
        
    def generate_trajectory(env, model, seed, start, goal):
        prm = PSPRM(model, env)
        t1 = time.time()
        prm.build_prm(seed)
        t2 = time.time()

        t1x = time.time()
        s_id, g_id = prm.addStartAndGoal(start, goal)
        path = prm.a_star_search(start_id=s_id, goal_id=g_id, alpha=.25, beta=.5)
        sol = Solution(path)
        sol.simplify(prm, env, max_skip_dist=MAX_SKIP_DIST)
        trajectory = sol.generate_trajectory_rsplan(prm, turning_radius=1)      
        trajectory = sol.project_trajectory(env['object_pose'])
        t2x = time.time()
        return trajectory, (t2 - t1), (t2x - t1x), sol.path

    trajectories = {}
    
    # Set object in env
    for obj in objects:  # human, monitor, cup
        print(f"Planning for {obj[2]}")
        env['object_pose'] = obj[0]
        env['object_label'] = obj[1]
        obj_name = obj[2]
        trajectories[obj_name] = []
        
        # Plan for each start/goal pair
        plan = 0
        for start in starts:
            for goal in goals:
                b_times = []
                p_times = []
                for i in range(10):  # 100 trials per start/goal pair, 
                    trajectory, build_time, plan_time, path = generate_trajectory(env, MODEL, SEED, start, goal)
                    b_times.append(build_time)
                    p_times.append(plan_time)
                    print(len(path))
                    # print(len(trajectory))
                # Delete first time (warmup)

                b_times = b_times[1:]
                p_times = p_times[1:]
                b_times = np.array(b_times)
                p_times = np.array(p_times)
                timing_stats = {
                    'mean_build': np.mean(b_times),
                    'mean_plan': np.std(p_times)
                    # 'min': np.min(times),
                    # 'max': np.max(times)
                }
                # # Timing boxplot stats
                # boxplot_stats = {
                #     'whislo': np.min(times),  # Bottom whisker position
                #     'q1': np.percentile(times, 25),  # First quartile (
                #     'med': np.median(times),  # Median
                #     'q3': np.percentile(times, 75),  # Third quartile
                #     'whishi': np.max(times),  # Top whisker position
                # }
                # print(timing_stats)

                trajectory_perc_score = calculate_trajectory_perc_score(prm, trajectory, env['object_pose'], env['object_label'], model=MODEL)
                # diffs = np.diff(trajectory, axis=0)              # shape (N-1, D)
                # segment_lengths = np.linalg.norm(diffs, axis=1)  # shape (N-1,)
                # path_length = np.sum(segment_lengths)

                # Get x, y, theta, pan, tilt for nodes in path
                netx_graph = prm.graph
                path_ = []
                for node_id in path:
                    node = netx_graph.nodes[node_id]
                    x = node['x']
                    y = node['y']
                    theta = node['theta']
                    pan = node['pan']
                    tilt = node['tilt']
                    path_.append([x, y, theta, pan, tilt])
                path_ = np.array(path_)  # shape (N, 5)
                diffs = np.diff(path_, axis=0)              # shape (N-1, D)
                segment_lengths = np.linalg.norm(diffs, axis=1)  # shape (N-1,)
                path_length = np.sum(segment_lengths)


                trajectories[obj_name].append({
                    # "start": start.tolist(),
                    # "goal": goal.tolist(),
                    "trajectory": trajectory.tolist(), 
                    "timing": timing_stats, 
                    # "boxplot": boxplot_stats,
                    # "perc_score": trajectory_perc_score,
                    "path_length": float(path_length)
                })  


                # if not os.path.exists(f'./trajectories_for_e22/plan_{plan}/'):
                #     os.makedirs(f'./trajectories_for_e22/plan_{plan}/')
                
                # # Save trajectory to csv and reorder to pan tilt x y theta
                # traj_ = trajectory[:, [3, 4, 0, 1, 2]]  # reorder to pan tilt x y theta
                # np.savetxt(f'./trajectories_for_e22/plan_{plan}/path.csv', traj_, delimiter=',')

                # plan += 1

    for obj in objects:
        obj_name = obj[2]
        path_lengths = [traj['path_length'] for traj in trajectories[obj_name]]
        avg_path_length = float(np.mean(path_lengths))
        trajectories['Average Path Length'] = avg_path_length
    
    for obj in objects:
        obj_name = obj[2]
        build_times = [traj['timing']['mean_build'] for traj in trajectories[obj_name]]
        avg_build_time = float(np.mean(build_times))
        trajectories['Average Build Time'] = avg_build_time
    
    for obj in objects:
        obj_name = obj[2]
        plan_times = [traj['timing']['mean_plan'] for traj in trajectories[obj_name]]
        avg_plan_time = float(np.mean(plan_times))
        trajectories['Average Plan Time'] = avg_plan_time

    # Save trajectories to json
    import json
    with open('./trajs_for_e.json', 'w') as f:
        json.dump(trajectories, f, indent=4)

    return

    
    

if __name__ == "__main__":
    main()


