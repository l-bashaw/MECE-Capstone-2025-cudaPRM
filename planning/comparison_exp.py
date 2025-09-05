import time
import torch

import numpy as np

from prm import PSPRM, Solution
from nn.inference import ModelLoader
from utils.EnvLoader import EnvironmentLoader


def calculate_trajectory_perc_score(trajectory, object_pose, object_label):
    return 999.0  # Placeholder implementation

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

                trajectory_perc_score = calculate_trajectory_perc_score(trajectory, env['object_pose'], env['object_label'])
                trajectories[obj_name].append({
                    "start": start.tolist(),
                    "goal": goal.tolist(),
                    "trajectory": trajectory.tolist(), 
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


