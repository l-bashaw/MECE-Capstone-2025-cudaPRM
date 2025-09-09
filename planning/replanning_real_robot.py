import os
import sys
import time
import websocket
from ros_bridge import pose_receiver as pr
from ros_bridge import trajectory_publisher as tp

import time
import torch
import numpy as np
from prm import PSPRM, Solution
from nn.inference import ModelLoader


TOPIC = "/trajectory"
MSG_TYPE = "std_msgs/Float32MultiArray"
ROSBRIDGE_SERVER = "ws://localhost:9090"


DEVICE = 'cuda'
DTYPE = torch.float32
MODEL_PATH = "resources/models/percscore-nov12-50k.pt"
MODEL_LOADER = ModelLoader(label_size=3, max_batch_size=10000, use_trt=False)
MODEL = MODEL_LOADER.load_model(MODEL_PATH)
SEED = 2387
MAX_SKIP_DIST = 1.2
REPLANNING_THRESHOLD = 1

START = np.array([0.0, 2.0, 0.0, 0.0, 0.0])
GOAL = np.array([0.1, -1.8, -1.57, 0.0, 0.0])

ROBOT_NAME = 'stretch_odom'

MONITORING_OBJ = "chair_black"
MO_LABEL = torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE, device=DEVICE)

BOUNDS = torch.tensor([
    [-0.7, -2.5, -3.14159, 0.0, 0.0],
    [ 1.2,  2.0,  3.14159, 0.0, 0.0]], 
    dtype=DTYPE, 
    device=DEVICE
)

# Print env in matplotlib
def print_env_and_trajectory(env, trajectory):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    fig, ax = plt.subplots()
    ax.set_xlim(BOUNDS[0,0].item()-1, BOUNDS[1,0].item()+1)
    ax.set_ylim(BOUNDS[0,1].item()-1, BOUNDS[1,1].item()+1)

    if env['rectangles'] is not None and env['rectangles'].numel() > 0:
        for rect in env['rectangles'].cpu().numpy():
            x, y, w, h = rect
            ax.add_patch(Rectangle((x - w/2, y - h/2), w, h, color='blue', alpha=0.5))

    if env['circles'] is not None and env['circles'].numel() > 0:
        for circ in env['circles'].cpu().numpy():
            x, y, r = circ
            ax.add_patch(Circle((x, y), r, color='red', alpha=0.5))

    if env['object_pose'] is not None and env['object_pose'].numel() > 0:
        obj = env['object_pose'].cpu().numpy()
        ax.plot(obj[0], obj[1], 'go', markersize=10)  # Object position

    # Print object orientation as a line
        if env['object_pose'].numel() == 7:
            import scipy.spatial.transform
            quat = obj[3:7]
            rot = scipy.spatial.transform.Rotation.from_quat(quat)
            dir_vec = rot.apply(np.array([1, 0, 0]))
            ax.plot([obj[0], obj[0] + dir_vec[0]], [obj[1], obj[1] + dir_vec[1]], 'g-')

    plt.gca().set_aspect('equal', adjustable='box')

    if trajectory is not None and len(trajectory) > 0:
        ax.plot(trajectory[:,2], trajectory[:,3], 'k.-')
    ax.plot(trajectory[0,2], trajectory[0,3], 'go', markersize=10)  # Start
    ax.plot(trajectory[-1,2], trajectory[-1,3], 'ro', markersize=10)  # Goal

    plt.grid(True)
    plt.show()

def get_robot_state(poses, names):
    for pose, name in zip(poses, names):
        if name == ROBOT_NAME:
            position = pose['position']
            orientation = pose['orientation']
            print(orientation)
            # Convert quaternion to theta in xy plane
            theta = np.arctan2(2.0 * (orientation['w'] * orientation['z'] + orientation['x'] * orientation['y']),
                               1.0 - 2.0 * (orientation['y']**2 + orientation['z']**2))
            return np.array([position['x'], position['y'], theta, 0.0, 0.0])
    print("ROBOT NOT FOUND")
    return None

def update_env_from_stream(env, poses, names):
    env['rectangles'] = torch.empty((0, 4), dtype=DTYPE, device=DEVICE)
    env['circles'] = torch.empty((0, 3), dtype=DTYPE, device=DEVICE)
    env['object_pose'] = torch.empty((7,), dtype=DTYPE, device=DEVICE)

    for i, (pose, name) in enumerate(zip(poses, names)):
        # If name contains chair, use the r 0.75
        # Append to env['circles']
        if name.startswith("chair"):
            env['circles'] = torch.cat((
                env['circles'], 
                torch.tensor(
                    [
                        [pose['position']['x'], pose['position']['y'], 0.4]
                    ], 
                dtype=DTYPE, 
                device=DEVICE
            )), dim=0)
            
           
        # elif name == "table":
        #     env['rectangles'] = torch.tensor(
        #         [
        #             [pose['position']['x'], pose['position']['y'], 0.75, 1.5]
        #         ], 
        #     dtype=DTYPE, 
        #     device=DEVICE
        # )
        

        if name == MONITORING_OBJ:
            env['object_pose'] = torch.tensor(
                [
                    pose['position']['x'], 
                    pose['position']['y'], 
                    1.5, 
                    pose['orientation']['x'], 
                    pose['orientation']['y'], 
                    pose['orientation']['z'], 
                    pose['orientation']['w']
                ], 
            dtype=DTYPE, 
            device=DEVICE
        )

def generate_trajectory(env, start, goal):
    prm = PSPRM(MODEL, env)
    prm.build_prm(SEED)
    s_id, g_id = prm.addStartAndGoal(start, goal)
    path = prm.a_star_search(start_id=s_id, goal_id=g_id, alpha=.3, beta=.25)
    sol = Solution(path)
    sol.simplify(prm, env, max_skip_dist=MAX_SKIP_DIST)
    trajectory = sol.generate_trajectory_rsplan(prm, turning_radius=1)      
    trajectory = sol.project_trajectory(env['object_pose'])

    # Traj is x, y, theta, pan, tilt
    # Reorder to pan tilt x y theta
    trajectory = trajectory[:, [3, 4, 0, 1, 2]]
    return trajectory

def need_to_replan(initial_env, current_env):

    # return False

    if initial_env['rectangles'].numel() > 0 and initial_env['circles'].numel() > 0:
        for ri, rc, ci, cc in zip(initial_env['rectangles'].cpu(), current_env['rectangles'].cpu(), initial_env['circles'].cpu(), current_env['circles'].cpu()):
            if (np.linalg.norm(rc - ri) > REPLANNING_THRESHOLD) or (np.linalg.norm(cc - ci) > REPLANNING_THRESHOLD):
                # print("REPLANNING")
                return True
    # If rectangles is empty tensor or none
    elif initial_env['rectangles'] is None or initial_env['rectangles'].numel() == 0:
        for ci, cc in zip(initial_env['circles'].cpu(), current_env['circles'].cpu()):
            # print(np.linalg.norm(cc - ci))
            # Implement the norm from scratch
            # norm = (cc[0] - ci[0])
            if (np.linalg.norm(cc - ci) > REPLANNING_THRESHOLD):
                # print("REPLANNING")
                return True
    elif initial_env['circles'] is None or initial_env['circles'].numel() == 0:
        for ri, rc in zip(initial_env['rectangles'].cpu(), current_env['rectangles'].cpu()):
            if (np.linalg.norm(rc - ri) > REPLANNING_THRESHOLD):
                # print("REPLANNING")
                return True
            
    if initial_env['object_pose'].numel() > 0 and current_env['object_pose'].numel() > 0:
        print(np.linalg.norm(current_env['object_pose'][3:].cpu() - initial_env['object_pose'][3:].cpu()))
        if (np.linalg.norm(current_env['object_pose'][3:].cpu() - initial_env['object_pose'][3:].cpu()) > REPLANNING_THRESHOLD*2):
            # print("REPLANNING")
            return True
    return False

def main():

    base_env = {
        'bounds': BOUNDS,  # FILL IN... x_min, y_min, -PI, 0, 0;   x_max, y_max, theta_max, PI, 0, 0
        'rectangles': None,
        'circles': None,
        'object_pose': None,
        'object_label': MO_LABEL
    }
    ws = websocket.create_connection(ROSBRIDGE_SERVER)
    pr.start_rosbridge_thread()

    # Wait for three seconds
    time.sleep(3)

    poses, names = pr.get_latest_poses_and_names()
    if poses and names:
        print("Initial poses and names received.")
        update_env_from_stream(base_env, poses, names)

    env = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in base_env.items()}
    # print(env)

    start = get_robot_state(poses, names)
    goal = GOAL

    traj = generate_trajectory(env, start, goal)
    # print_env_and_trajectory(env, traj)

    # print(traj[:5])
    # print()
    # print(traj[-5:])
    tp.publish_trajectory(ws, traj)
    
    replan = False
    # Set t_restart to yesterday so that it replans immediately
    t_restart = time.time() - 24 * 60 * 60

    while True:
        poses, names = pr.get_latest_poses_and_names()
        if poses and names:
            update_env_from_stream(env, poses, names)

        replan = need_to_replan(initial_env=base_env, current_env=env)
        # print(f"Current object pose: {env['object_pose']}")
        if replan:
            # if time.time() - t_restart < 2:
            #     replan = False
            #     continue
            print("Replanning...")
            poses, names = pr.get_latest_poses_and_names()
            
            start = get_robot_state(poses, names)
            if start is None:
                print("Robot not found, skipping replanning.")
                continue
            update_env_from_stream(env, poses, names)
            
            traj = generate_trajectory(env, start, goal)
            # print_env_and_trajectory(env, traj)

            tp.publish_trajectory(ws, traj)
            t_restart = time.time()
            for k, v in env.items():
                base_env[k] = v.clone() if isinstance(v, torch.Tensor) else v
            replan = False

if __name__ == "__main__":
    main()