import time
import torch
import numpy as np
from planning.prm import PSPRM, Solution
from planning.nn.inference import ModelLoader
from planning.utils.EnvLoader import EnvironmentLoader

def main():
    device = 'cuda'
    dtype = torch.float32
    env_config_file = "/planning/resources/scenes/environment/multigoal_demo.yaml"  
    model_path = "/planning/resources/models/percscore-nov12-50k.pt"
    seed = 2387

    # make start and goal tensors
    start = np.array([1, -3, 3.14159/2, 0.0, 0.0]) 
    goal = np.array([1, 2.5, 3.14159/2, 0.0, 0.0])
    
    # Load environment from yaml
    print("Loading environment and model...")
    env_loader = EnvironmentLoader(device=device)
    env = env_loader.load_world(env_config_file)
    env['bounds'] = torch.concat([env['bounds'], torch.tensor([[-3.14159, 0.0, 0.0], [3.14159, 0.0, 0.0]], dtype=dtype, device=device)], dim=1)
    env['object_pose'] = torch.tensor([0.2, -0.2, 1.65, 0, 0, 0, 1], dtype=dtype, device=device)
    env['object_label'] = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
    env['bounds'][0, 0] = -1.5
    env['bounds'][1, 0] = 4
    env['bounds'][0, 1] = -4
    env['bounds'][1, 1] = 4
    # label_map = {
    #   "human": [1, 0, 0],
    #   "monitor": [0, 1, 0],
    #   "cup": [0, 0, 1]
    # }

    # Load model
    model_loader = ModelLoader(label_size=3, max_batch_size=10000, use_trt=False)
    model = model_loader.load_model(model_path)

    # Create PRM and plan path, multiple times to benchmark
    for i in range(5):
        prm = PSPRM(model, env)
        t1 = time.time()
        
        prm.build_prm(seed)    # call cuda code to construct the PRM
        st, go = prm.addStartAndGoal(start, goal)  # add the start and goal nodes
        path = prm.a_star_search(start_id=st, goal_id=go, alpha=0.5, beta=.2)  # plan path using A*
        sol = Solution(path)  # create solution object and process path into a trajectory
        sol.simplify(prm, env, max_skip_dist=1.5)
        trajectory = sol.generate_trajectory_rsplan(prm, turning_radius=1)
        trajectory = sol.project_trajectory(env['object_pose'])  # project Stretch2's camera for the entire trajectory
        
        t2 = time.time()
        print(f"Time: {t2 - t1:.4f} seconds")

    sol.print_path(prm.graph)  # prints the path (not the trajectory)

if __name__ == "__main__":
    main()