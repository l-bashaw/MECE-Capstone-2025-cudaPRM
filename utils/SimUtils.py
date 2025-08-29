
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

"""@dataclass
class OrientedBoundingBox:
    # translation and rotation offset to the object's pose
    translation: NDArray = np.zeros(3)
    # only handle z-axis rotation
    rotation: int = 0
    # [length, width, height] eq. [x, y, z]
    extents: NDArray = np.zeros(3)"""

    
class IsaacSimCamera():
    def __init__(self, cam_pos=[-4,-4,3], target_pos=[0,0,0], fov=60, width=640, height=480, near=0.1, far=100):
        
        import omni.isaac.core.utils.prims as prim_utils
        from omni.isaac.sensor import Camera as IsaacSimCamera
        
        self.pos = cam_pos
        self.target_pos = target_pos
        self.fov = fov
        self.width = width
        self.height = height
        self.near = near
        self.far = far
        
        self.camera_config = {}
        self.camera_config["prim_path"] = "/World/Camera_World"
        prim_utils.create_prim(self.camera_config["prim_path"], "Xform")
        self.camera_config["prim_path"] = self.camera_config["prim_path"] + "/camera"
        self.camera = IsaacSimCamera(
            prim_path=self.camera_config["prim_path"],
            name="Camera",
            resolution=(self.width, self.height),
        )
        
        self.camera.initialize()
        
        camera_pose = self.get_camera_pose(np.array(self.pos), np.array(self.target_pos))
        self.camera.set_world_pose(camera_pose[0], camera_pose[1])
        
        focal_length = 2.0
        
        horizontal_aperture = 2 * focal_length * np.tan(fov * np.pi / 180 / 2)
        
        self.camera.set_focal_length(focal_length)
        self.camera.set_horizontal_aperture(horizontal_aperture)
        self.camera.set_clipping_range(near_distance=self.near, far_distance=self.far)
        
        print("Camera initialized at path: ", self.camera_config["prim_path"])
    
    def get_camera_pose(self, world_pose, target_pose = np.array([0,0,0]), world_up = np.array([0, 0, 1])):
    
        camera_position = np.array(world_pose)
        target_position = np.array(target_pose)

        # vector (x axis front)
        forward = target_position - camera_position
        forward = forward / np.linalg.norm(forward)
        right = np.cross(world_up, forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)

        # R matrix
        rotation_matrix = np.array([
            forward,    # Camera's x-axis aligned with forward direction
            right,      # Camera's y-axis aligned with the 'right' direction
            up       # Camera's z-axis aligned with the 'up' direction
        ]).T
        
        rotation_scipy = R.from_matrix(rotation_matrix)
        quat = rotation_scipy.as_quat()
        quat = [quat[3], quat[0], quat[1], quat[2]]

        return [world_pose, quat]
    
    def get_image(self):
        rgb = self.camera.get_rgba()[:, :, :3]
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    