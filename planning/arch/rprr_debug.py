"""
Test harness for your replanning script.
Simulates poses for: stretch_odom, chair_black, chair_white.
- chair_white moves in x,y first
- chair_black rotates later
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# --- fake ros_bridge.pose_receiver ---
class FakePoseReceiver:
    def __init__(self):
        self.t0 = time.time()
        self.black_rotated = False
        self.white_moved = False

    def get_latest_poses_and_names(self):
        t = time.time() - self.t0

        # Robot stays fixed
        robot_pose = {
            "position": {"x": 0.0, "y": 2.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
        }

        # Chair_white moves left and right between x=-0.5 and 0.5
        if t > 5.0 and not self.white_moved:
            # Move it once in +x
            print("Moving chair_white in +x")
            time.sleep(0.1)
            x_white, y_white = 1.2, 1.0
            self.white_moved = True
        else:
            # Initial position
            x_white, y_white = 0.0, 0.0

        chair_white_pose = {
            "position": {"x": x_white, "y": y_white},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
        }

        # Chair_black rotates around z after 15 seconds
        if t > 10.0 and not self.black_rotated:
            # Apply a fixed 45° yaw once
            print("Rotating chair_black by 45 degrees")
            time.sleep(0.1)
            r = R.from_euler("z", np.deg2rad(45))
            quat_black = r.as_quat()  # [x,y,z,w]
            self.black_rotated = True
        else:
            # Initial orientation
            quat_black = [0.0, 0.0, 0.0, 1.0]

        chair_black_pose = {
            "position": {"x": 0.5, "y": 1.0},
            "orientation": {
                "x": float(quat_black[0]),
                "y": float(quat_black[1]),
                "z": float(quat_black[2]),
                "w": float(quat_black[3])
            }
        }

        poses = [robot_pose, chair_black_pose, chair_white_pose]
        names = ["stretch_odom", "chair_black", "chair_white"]
        return poses, names


# --- patch your module to use FakePoseReceiver ---
import types
import replanning_real_robot as main_script  # replace with actual filename (without .py)

fake_pr = FakePoseReceiver()
main_script.pr.get_latest_poses_and_names = types.MethodType(
    lambda self=None: fake_pr.get_latest_poses_and_names(),
    fake_pr
)

# --- run main loop (with plotting) ---
if __name__ == "__main__":
    main_script.main()
