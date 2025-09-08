import time
import torch

import numpy as np
import pytorch_kinematics as pk

from prm import PSPRM, Solution
from nn.inference import ModelLoader
from utils.EnvLoader import EnvironmentLoader

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from isaacsim import SimulationApp
import argparse
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Run the simulation app")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Whether to run the simulation in headless mode",
    )
    parser.add_argument(
        "--world-cam",
        action="store_true",
        help="Whether to render the world camera view",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Path to the model file, if displaying model evaluation",
    )
    parser.add_argument(
        "--plan-csv",
        type=str,
        default="",
        help="Path to the plan data file",
    )
    return parser.parse_args()

args = parse_args()
# Start the simulation app
simulation_app = SimulationApp(launch_config={"headless": False, "sync_loads": True})
from pathlib import Path
import numpy as np
import omni
import yaml
import time
import cv2
import os
from omni.isaac.core import World
from omni.isaac.core.utils.extensions import get_extension_path_from_name
import omni.isaac.core.utils.numpy.rotations as rot_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.importer.urdf import _urdf
import omni.kit.commands
import omni.usd
from omni.isaac.core.robots import Robot
from omni.isaac.core.prims import XFormPrim
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.materials import OmniPBR
from omni.isaac.core.objects.ground_plane import GroundPlane
from omni.isaac.debug_draw import _debug_draw
from utils.SimUtils import IsaacSimCamera

def set_robot_state(robot, state):
    # Set the joint positions
    pan_index = 3
    tilt_index = 4
    pan_val = state[3]
    tilt_val = state[4]
    robot.set_joint_positions(np.array([pan_val, tilt_val]), joint_indices=np.array([pan_index, tilt_index]))

    # Set the world pose
    theta = state[2]
    w = np.cos(theta / 2.0)
    z = np.sin(theta / 2.0)
    quaternion = [w, 0.0, 0.0, z]
    robot.set_world_pose(position=[state[0], state[1], 0.0], orientation=quaternion)


def main():

     # Initialize the world
    world = World()
    world.clear()
    world.scene.add_default_ground_plane()
    GroundPlane(prim_path="/World/groundPlane", size=20, color=np.array([0.85, 0.65, 0.6]), z_position=0.01)
    action_registry = omni.kit.actions.core.get_action_registry()
    action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_off")
    action.execute()

    prim_utils.create_prim(
        "/model_urdf_0/Light_1",
        "DomeLight",
        position=np.array([0,0,-1]),
        attributes={
            "inputs:intensity": 500,
            "inputs:color": (1.0, 1.0, 1.0)
        }
    )
    file_path = "resources/robots/stretch/stretch.urdf"
    yaml_path = "resources/scenes/environment/replanning_test_env.yaml"

    ################## Load the URDF##################
    # Acquire the URDF extension interface
    urdf_interface = _urdf.acquire_urdf_interface()
    # Set the settings in the import config
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = False
    import_config.default_drive_strength = 1047.19751
    import_config.default_position_drive_damping = 52.35988
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0
    extension_path = get_extension_path_from_name("omni.importer.urdf")


    #################### Load the robot ####################
    result, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=file_path,
        import_config=import_config,
    )
    robot = world.scene.add(Robot(prim_path=prim_path, name="stretch_robot"))
    robot_prim = Robot(prim_path=prim_path, name="fetch")
    robot = world.scene.add(robot_prim)
    world.reset()
    world.step(render=True)

    camera = Camera(
        prim_path="/stretch/camera_color_optical_frame/camera",
        resolution=(640, 480),
        # Need to initialize the camera with the correct orientation
        orientation=rot_utils.euler_angles_to_quats(np.array([0, -90, 0]), degrees=True),
    )
    camera.set_clipping_range(0.3, 10.0)
    camera.initialize()
    print(camera.get_focal_length())
    camera.set_focal_length(2.0)

    with open(yaml_path, "r") as stream:
        try:
            yaml_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    prim_dict = {}
    for obj in yaml_data["world"]["collision_objects"]:
        if "meshes" not in obj:
            # TODO: add support for other types of objects e.g. primitives
            continue
        dimension = np.array(obj["meshes"][0]["dimensions"]) 
        # if "wall" in obj["id"]:
            # dimension[2]*=50
        positions = np.array(obj["mesh_poses"][0]["position"])
        orientations = np.array(obj["mesh_poses"][0]["orientation"])
        orientations = orientations / np.linalg.norm(orientations)
        resource = str(Path(obj["meshes"][0]["resource"]).parent / Path("model.usdz"))
        # if has usd, add reference to stage
        usd_path_usdz = str(Path(obj["meshes"][0]["resource"]).parent / Path("model.usdz"))
        usd_path_usd = str(Path(obj["meshes"][0]["resource"]).parent / Path("model.usd"))
        dae_path = str(Path(obj["meshes"][0]["resource"]).parent / Path("model.dae"))	
        if os.path.exists(usd_path_usdz) and os.access(usd_path_usdz, os.R_OK):
            print(f"Loading {obj['id']} from {usd_path_usdz}")
            prim_path="/World/humans/" + obj["id"]
            stage_utils.add_reference_to_stage(usd_path=usd_path_usdz, prim_path=prim_path)
        
        elif os.path.exists(usd_path_usd) and os.access(usd_path_usd, os.R_OK):
            print(f"Loading {obj['id']} from {usd_path_usd}")
            prim_path="/World/humans/" + obj["id"]
            stage_utils.add_reference_to_stage(usd_path=usd_path_usd, prim_path=prim_path)
        elif os.path.exists(dae_path) and os.access(dae_path, os.R_OK):
            print(f"Loading {obj['id']} from {dae_path}")
            prim_path="/World/humans/" + obj["id"]
            stage_utils.add_reference_to_stage(usd_path=dae_path, prim_path=prim_path)
        else:
            continue                 
        prim = XFormPrim(prim_path, name=obj["id"], scale=dimension, position=positions, orientation=np.array([orientations[3], orientations[0], orientations[1], orientations[2]]), visible=True)
        prim_dict[obj["id"]] = prim
    world.reset()

    def active_sleep(duration):
            curr_time = time.time()
            while time.time() - curr_time < duration:
                world.step(render=True)

    def countdown(t):
        while t:
            mins, secs = divmod(t, 60)
            timer = '{:02d}:{:02d}'.format(mins, secs)
            print(timer, end="\r")
            active_sleep(1)
            t -= 1

    # -----------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
    

    device = 'cuda'
    dtype = torch.float32
    model_path = "resources/models/percscore-nov12-50k.pt"
    seed = 2387

    env_loader = EnvironmentLoader(device=device)
    env = env_loader.load_world(yaml_path)

    model_loader = ModelLoader(label_size=3, max_batch_size=10000, use_trt=False)
    model = model_loader.load_model(model_path)



    # ----------------------------------- Experiment parameters -----------------------------------
    start = np.array(
        [1, -2.5, 3.14159/2, 0.0, 0.0]
    )
    goal = np.array(
        [1, 2.5, 3.14159/2, 0.0, 0.0]
    )

    MAX_SKIP_DIST = 3
    # ----------------------------------- Object positions -----------------------------------

    human_start = np.array(
        [1.2, 0.0, 1.65,   0, 0, 0.707107, 0.707107],  # start
    )
    human_goal = np.array(
         [1, 1.50, 1.65,   0.707107, 0, 0, 0.707107]
    )

    chair_goal = np.array(
         [0, 1, 0,   -0.707107, 0, 0, 0.707107]
    )

    human_label = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
    monitoring_object = 'human'
    
    # ----------------------------------- Setup Env -----------------------------------
    
    env['bounds'] = torch.concat([env['bounds'], torch.tensor([[-3.14159, 0.0, 0.0], [3.14159, 0.0, 0.0]], dtype=dtype, device=device)], dim=1)
    env['bounds'][0, 0] = -1.5  # clip them to the walls to avoid weird sampling
    env['bounds'][1, 0] = 4
    env['bounds'][0, 1] = -4
    env['bounds'][1, 1] = 4
    env['object_pose'] = torch.tensor(human_start, dtype=dtype, device=device)
    env['object_label'] = human_label

    # ----------------------------------- Warmup Planner -----------------------------------

    print("Warming up planner...")
    prm = PSPRM(model, env)
    prm.build_prm(seed)
    s_id, g_id = prm.addStartAndGoal(start, goal)
    path = prm.a_star_search(start_id=s_id, goal_id=g_id, alpha=1, beta=1)
    sol = Solution(path)
    sol.simplify(prm, env, max_skip_dist=MAX_SKIP_DIST)
    trajectory = sol.generate_trajectory_rsplan(prm, turning_radius=1)      
    trajectory = sol.project_trajectory(env['object_pose'])
    print("Planner warmup complete.")

    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
    def generate_trajectory(env, model, start, goal, MAX_SKIP_DIST, seed):
        prm = PSPRM(model, env)
        prm.build_prm(seed)
        s_id, g_id = prm.addStartAndGoal(start, goal)
        path = prm.a_star_search(start_id=s_id, goal_id=g_id, alpha=1, beta=1)
        sol = Solution(path)
        sol.simplify(prm, env, max_skip_dist=MAX_SKIP_DIST)
        trajectory = sol.generate_trajectory_rsplan(prm, turning_radius=1)      
        trajectory = sol.project_trajectory(env['object_pose'])
        return trajectory


    def need_to_replan(initial_env, current_env, threshold=0.5):
        for ri, rc in zip(initial_env['rectangles'].cpu(), current_env['rectangles'].cpu()):
            print(np.linalg.norm(rc - ri) > 0)
            if np.linalg.norm(rc - ri) > threshold:
                return True


    def move_obj(new_pose, env, monitoring_obj, obj="human"):
        # New pose is in [x, y, z, qx, qy, qz, qw] format
        # Set in isaac sim
        obj_ = prim_dict[obj]
        obj_.set_world_pose(position=new_pose[:3], orientation=new_pose[3:])

        # Set in env
        id = env_loader.map[obj]
        print(id)
        old_obs = env['rectangles'][id]

        env['rectangles'][id] = torch.tensor(
            [new_pose[0], new_pose[1], old_obs[2], old_obs[3]], 
            dtype=dtype, 
            device=device
        )

        if obj == monitoring_obj:
            env['object_pose'] = torch.tensor(
                [new_pose[0], new_pose[1], new_pose[2], new_pose[4], new_pose[5], new_pose[6], new_pose[3]],
                dtype=dtype,
                device=device
            )        

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

    
    countdown(3)  # Countdown for 5 seconds before starting the video recording

    ################ Write the camera view to a video #################
    
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    recording_dir = f"videos/recording_{timestamp}"
    os.makedirs(recording_dir, exist_ok=True)
    video_filename = os.path.join(recording_dir, "isaacsim_camera.mp4")
    fixed_video_filename = os.path.join(recording_dir, "fixed_isaacsim_camera.mp4")
    frame_width = 640
    frame_height = 480
    fps = 20
    if args.world_cam:
        world_camera = IsaacSimCamera(
        cam_pos=[5.0, 0, 4.5],
        target_pos=[0.0, 0, 0],
        fov=80,
        width=1280,
        height=720,
        near=0.15,
        far=15,
        )
        world_cam_filename = os.path.join(recording_dir, "world_cam_isaacsim.mp4")
        world_cam_video_out = cv2.VideoWriter(world_cam_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))

    time_between_states = 0.001 # time between states
    time_between_paths = 3 # time between paths
    time_between_loops = 10 # time between loops
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 file
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
    i = 0
    traj_n = 0
    print("Starting video recording")
    ################# Set the robot states #################
    # Read in the robot states from the CSV file
    draw = _debug_draw.acquire_debug_draw_interface()
    record_time = 5
    start_record_time = time.time()
    first_image = True
    for k in range(100):
        world.step(render=True)
    start_sim_time = time.time()
    def take_image():
        camera.get_current_frame()
        # Get the RGBA image data
        image_data = camera.get_rgba()[:, :, :3]  # Assuming it returns a NumPy array
        # Convert from RGB to BGR for OpenCV
        image = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
        # Write the frame to the video file
        video_writer.write(image)
        if args.world_cam:
            world_image = world_camera.get_image()
            world_cam_video_out.write(world_image)
    last_cam_time = time.time()
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

    # Need a deep copy
    env_i = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in env.items()}
    states = generate_trajectory(env, model, start, goal, MAX_SKIP_DIST, seed)
    
    move_human = False
    replan = False
    need_to_move = True

    if simulation_app.is_running():
        t1 = time.time()
        while True:
            print(env)
            if replan:
                print("Replanning...")
                states = generate_trajectory(env, model, start, goal, MAX_SKIP_DIST, seed)
                replan = False
                
            for robot_state in states:
                
                set_robot_state(robot, robot_state)
                
                if time.time() - t1 > 2 and need_to_move:
                    move_obj(chair_goal, env, monitoring_obj='human', obj="chair")
                    move_obj(human_goal, env, monitoring_obj='human', obj="human")
                    need_to_move = False
                   
                    if need_to_replan(env_i, env, threshold=0.5):
                        
                        print("Env has changed significantly...")
                        start = robot_state
                        replan = True
                        break

                for _ in range(2):
                    world.step(render=True)
                
                if time.time() - last_cam_time > 1/fps:
                    take_image()
                    last_cam_time = time.time()
                # sleep between each state, if timestamp is not available
                if timestamp == -1:
                    active_sleep(time_between_states)
                # if time.time() - start_record_time > record_time:
                #     break
                # 10 ms delay to allow for rendering
                active_sleep(0.1)

            if not replan:
                break
                
        # Release the video writer
        video_writer.release()
        if args.world_cam:
            world_cam_video_out.release()
        print(f"Finished writing video: {video_filename}")
        
    # Close the simulation app
    simulation_app.close()
    
    

if __name__ == "__main__":
    main()


