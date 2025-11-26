import os
import sys
import time
import websocket
from ros_bridge import pose_receiver as pr
from ros_bridge import trajectory_publisher as tp
from scipy.spatial.transform import Rotation as R
import time
import torch
import numpy as np
from prm import PSPRM, Solution
from nn.inference import ModelLoader
import copy


TOPIC = "/trajectory"
MSG_TYPE = "std_msgs/Float32MultiArray"
ROSBRIDGE_SERVER = "ws://localhost:9090"


DEVICE = 'cuda'
DTYPE = torch.float32
MODEL_PATH = "resources/models/percscore-nov12-50k.pt"
MODEL_LOADER = ModelLoader(label_size=3, max_batch_size=10000, use_trt=False)
MODEL = MODEL_LOADER.load_model(MODEL_PATH)
SEED = 2387
MAX_SKIP_DIST = 1.5

REPLANNING_THRESHOLD = 0.5
ANGLE_THRESH = 1  # Radians

# START = np.array([0.0, 2.0, 0.0, 0.0, 0.0])
# GOAL = np.array([0.2, -2, -1.57, 0.0, 0.0])

GOAL1 = np.array([1.10, 0, 0.0, 0.0, 0.0])
GOAL2 = np.array([-.65, 0, 0.0, 0.0, 0.0])

ROBOT_NAME = 'stretch_odom'

MONITORING_OBJ = "sword"
MO_LABEL = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE, device=DEVICE)

BOUNDS = torch.tensor([
    [-0.7, -.75, -3.14159, 0.0, 0.0],
    [ 1.2,  1.25,  3.14159, 0.0, 0.0]], 
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

    # Print object orientation as an arrow
    if env['object_pose'].numel() == 7:
        import scipy.spatial.transform
        quat = obj[3:7]
        rot = scipy.spatial.transform.Rotation.from_quat(quat)
        dir_vec = rot.apply(np.array([0.5, 0, 0]))
        ax.arrow(obj[0], obj[1], dir_vec[0], dir_vec[1], head_width=0.1, head_length=0.1, fc='g', ec='g')
        ax.text(obj[0], obj[1]+0.2, f"Quat:[{quat[0]:.2f}, {quat[1]:.2f}, {quat[2]:.2f}, {quat[3]:.2f}]", fontsize=8)


    plt.gca().set_aspect('equal', adjustable='box')

    if trajectory is not None and len(trajectory) > 0:
        ax.plot(trajectory[:,2], trajectory[:,3], 'k.-')
    ax.plot(trajectory[0,2], trajectory[0,3], 'go', markersize=10)  # Start
    ax.plot(trajectory[-1,2], trajectory[-1,3], 'ro', markersize=10)  # Goal

    plt.grid(True)

    # Save to png
    plt.savefig("./env_and_trajectory.png")
    plt.show()

def get_robot_state(poses, names):
    for pose, name in zip(poses, names):
        if name == ROBOT_NAME:
            position = pose['position']
            orientation = pose['orientation']
            # Convert quaternion to theta in xy plane
            theta = np.arctan2(2.0 * (orientation['w'] * orientation['z'] + orientation['x'] * orientation['y']),
                               1.0 - 2.0 * (orientation['y']**2 + orientation['z']**2))
            return np.array([position['x'], position['y'], theta, 0.0, 0.0])
    print("ROBOT NOT FOUND")
    return None

# def update_env_from_streamOLD(env, poses, names):
#     env['rectangles'] = torch.empty((0, 4), dtype=DTYPE, device=DEVICE)
#     env['circles'] = torch.empty((0, 3), dtype=DTYPE, device=DEVICE)
#     env['object_pose'] = torch.empty((7,), dtype=DTYPE, device=DEVICE)

#     for i, (pose, name) in enumerate(zip(poses, names)):
#         # If name contains chair, use the r 0.75
#         # Append to env['circles']
#         if name.startswith("chair"):
#             env['circles'] = torch.cat((
#                 env['circles'], 
#                 torch.tensor(
#                     [
#                         [pose['position']['x'], pose['position']['y'], 0.4]
#                     ], 
#                 dtype=DTYPE, 
#                 device=DEVICE
#             )), dim=0)
            
           
#         # elif name == "table":
#         #     env['rectangles'] = torch.tensor(
#         #         [
#         #             [pose['position']['x'], pose['position']['y'], 0.75, 1.5]
#         #         ], 
#         #     dtype=DTYPE, 
#         #     device=DEVICE
#         # )
        

#         if name == MONITORING_OBJ:
#             env['object_pose'] = torch.tensor(
#                 [
#                     pose['position']['x'], 
#                     pose['position']['y'], 
#                     1.5, 
#                     pose['orientation']['x'], 
#                     pose['orientation']['y'], 
#                     pose['orientation']['z'], 
#                     pose['orientation']['w']
#                 ], 
#             dtype=DTYPE, 
#             device=DEVICE
#         )

def flip_quat_180_z(quat):
    """
    Flip a quaternion 180 degrees about the z axis.
    quat: array-like, [x, y, z, w]
    Returns: flipped quaternion [x, y, z, w]
    """
    from scipy.spatial.transform import Rotation as R
    q = R.from_quat(quat)
    q_flip = R.from_quat([0, 0, 1, 0])  # 180 deg about z
    q_new = q_flip * q
    return q_new.as_quat()

def flip_quat_90_ccw_z(quat):
    """
    Flip a quaternion 90 degrees counter-clockwise about the z axis.
    quat: array-like, [x, y, z, w]
    Returns: flipped quaternion [x, y, z, w]
    """
    from scipy.spatial.transform import Rotation as R
    q = R.from_quat(quat)
    q_flip = R.from_quat([0, 0, 0.7071067811865476, 0.7071067811865476])  # 90 deg CCW about z
    q_new = q_flip * q
    return q_new.as_quat()

def update_env_from_stream(env, poses, names):
    # Initialize or reset rectangles and circles (commented for throwing demo)
    if env['rectangles'] is None or env['rectangles'].numel() == 0:
        env['rectangles'] = torch.empty((0, 4), dtype=DTYPE, device=DEVICE)
    # else:
    #     env['rectangles'].resize_(0, 4)

    # if env['circles'] is None or env['circles'].numel() == 0:
    #     env['circles'] = torch.empty((0, 3), dtype=DTYPE, device=DEVICE)
    # else:
    #     env['circles'].resize_(0, 3)

    # Initialize object_pose if not already
    if env['object_pose'] is None or env['object_pose'].numel() == 0:
        env['object_pose'] = torch.zeros((7,), dtype=DTYPE, device=DEVICE)

    # Hardcode humans for object throwing demo
    env['circles'] = torch.tensor(
        [
            [1.1, -0.4, 0.25],
            [-.65, -0.4, 0.25],
        ],
        dtype=DTYPE,
        device=DEVICE
    )

    for i, (pose, name) in enumerate(zip(poses, names)):
        # Chairs (except the monitored object) go into circles (commented for throwing demo)
        # if name.startswith("chair"):
        #     circle_tensor = torch.tensor(
        #         [[pose['position']['x'], pose['position']['y'], 0.6]],
        #         dtype=DTYPE,
        #         device=DEVICE
        #     )
        #     env['circles'] = torch.cat((env['circles'], circle_tensor), dim=0)

        # Rectangles (if you add tables later)
        # if name == "table":
        #     rect_tensor = torch.tensor(
        #         [[pose['position']['x'], pose['position']['y'], 0.75, 1.5]],
        #         dtype=DTYPE,
        #         device=DEVICE
        #     )
        #     env['rectangles'] = torch.cat((env['rectangles'], rect_tensor), dim=0)


        # Monitoring object gets updated in-place
        flipped_quat = flip_quat_90_ccw_z([
            pose['orientation']['x'],
            pose['orientation']['y'],
            pose['orientation']['z'],
            pose['orientation']['w']   
        ])

        if name == MONITORING_OBJ:
            env['object_pose'][:] = torch.tensor([
                pose['position']['x'],
                pose['position']['y'],
                pose['position']['z'],
                flipped_quat[0],
                flipped_quat[1],
                flipped_quat[2],
                flipped_quat[3]
            ], dtype=DTYPE, device=DEVICE)


def generate_trajectory(env, start, goal):
    prm = PSPRM(MODEL, env)
    prm.build_prm(SEED)
    s_id, g_id = prm.addStartAndGoal(start, goal)
    path = prm.a_star_search(start_id=s_id, goal_id=g_id, alpha=.5, beta=.25)
    sol = Solution(path)
    sol.simplify(prm, env, max_skip_dist=MAX_SKIP_DIST)
    trajectory = sol.generate_trajectory_rsplan(prm, turning_radius=1)      
    trajectory = sol.project_trajectory(env['object_pose'])

    # from debug import plot_environment_with_heatmap
    # plot_environment_with_heatmap(prm.graph, env['bounds'].cpu().numpy(), resolution=200)

    # Traj is x, y, theta, pan, tilt
    # Reorder to pan tilt x y theta
    trajectory = trajectory[:, [3, 4, 0, 1, 2]]
    return trajectory

# def quaternion_angleOLD(q1, q2):
 
#     q1 = q1 / np.linalg.norm(q1)
#     q2 = q2 / np.linalg.norm(q2)
    
#     q_rel = R.from_quat(q1).inv() * R.from_quat(q2)
    
#     return q_rel.magnitude()

def quaternion_angle(q1, q2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    return 2 * np.arccos(dot)

def need_to_replan(initial_env, current_env):

    if initial_env['rectangles'].numel() > 0 and initial_env['circles'].numel() > 0:
        for ri, rc, ci, cc in zip(initial_env['rectangles'].cpu(), current_env['rectangles'].cpu(), initial_env['circles'].cpu(), current_env['circles'].cpu()):
            
            norm_c = np.linalg.norm(cc[:2] - ci[:2])
            norm_r = np.linalg.norm(rc[:2] - ri[:2])
            if (norm_r > REPLANNING_THRESHOLD) or (norm_c > REPLANNING_THRESHOLD):
                print(f"BING: norm_r={norm_r}, norm_c={norm_c}")
                return True
            

    elif initial_env['rectangles'] is None or initial_env['rectangles'].numel() == 0:
        for ci, cc in zip(initial_env['circles'].cpu(), current_env['circles'].cpu()):
            
            # Use x, y only, not radius
            norm = np.linalg.norm(cc[:2] - ci[:2])
            if (norm > REPLANNING_THRESHOLD):
                print(f"BOOM: norm_c={norm}")
                return True
            

    elif initial_env['circles'] is None or initial_env['circles'].numel() == 0:
        for ri, rc in zip(initial_env['rectangles'].cpu(), current_env['rectangles'].cpu()):
            
            norm = np.linalg.norm(rc[:2] - ri[:2])
            if (norm > REPLANNING_THRESHOLD):
                print(f"BANG: norm_r={norm}")
                return True
            

    if initial_env['object_pose'].numel() > 0 and current_env['object_pose'].numel() > 0:
        cur_q = current_env['object_pose'][3:].cpu().numpy()
        init_q = initial_env['object_pose'][3:].cpu().numpy()
        # print("cur_q:", cur_q
        #       , "\ninit_q:", init_q)
        ang = quaternion_angle(cur_q, init_q)
        # print("quat angle diff (rad):", ang)

        norm = np.linalg.norm(current_env['object_pose'][:2].cpu().numpy() - initial_env['object_pose'][:2].cpu().numpy())
        if ang > ANGLE_THRESH or norm > REPLANNING_THRESHOLD:
            print(f"CLANG: norm_obj={norm}, ang={ang}")
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
    print(f"Initial base object_pose: {base_env['object_pose']}")
    env = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in base_env.items()}
    # print(env)

    start = get_robot_state(poses, names)


    goal = GOAL1

    traj = generate_trajectory(copy.deepcopy(env), start, goal)
    
    # print_env_and_trajectory(env, traj)

    tp.publish_trajectory(ws, traj)
    
    replan = False
    # Set t_restart to yesterday so that it replans immediately
    # t_restart = time.time() - 24 * 60 * 60

    while True:
        poses, names = pr.get_latest_poses_and_names()
        # print(f"BASE:    {base_env['object_pose']}")
        # print(f"CURR: {env['object_pose']}")
        if poses and names:
            update_env_from_stream(env, poses, names)
        # print(f"BASE22:    {base_env['object_pose']}")
        # print(f"CURR22: {env['object_pose']}")
        replan = need_to_replan(initial_env=base_env, current_env=env)

        if replan:

            # print(f"Before replanning - base object_pose: {base_env['object_pose']}")
            # print(f"Before replanning - env  object_pose: {env['object_pose']}")
            # print(f"They are equal: {torch.equal(base_env['object_pose'], env['object_pose'])}")
            # print()
            print("Replanning...")

            poses, names = pr.get_latest_poses_and_names()
            start = get_robot_state(poses, names)
            
            if start is None:
                print("Robot not found, skipping replanning.")
                continue

            # Set the goal to whichever the monitored object closest to
            env_obj_pos = env['object_pose'][:2].cpu().numpy()
            dist_to_goal1 = np.linalg.norm(env_obj_pos - GOAL1[:2])
            dist_to_goal2 = np.linalg.norm(env_obj_pos - GOAL2[:2])
            goal = GOAL1 if dist_to_goal1 < dist_to_goal2 else GOAL2

            # Generate trajectory
            traj = generate_trajectory(copy.deepcopy(env), start, goal)

            tp.publish_trajectory(ws, traj)
            base_env = copy.deepcopy(env)
            replan = False
            time.sleep(0.1)
            # print(f"After replanning - base object_pose: {base_env['object_pose']}")
            # print(f"After replanning - env  object_pose: {env['object_pose']}")
            # print(f"They are equal: {torch.equal(base_env['object_pose'], env['object_pose'])}")
            # print('--------------------------------------------------------------------------')
            # time.sleep(2)

if __name__ == "__main__":
    main()