from typing import Dict
import yaml
import cv2
from pathlib import Path
from dataclasses import dataclass
from numpy.typing import NDArray

import numpy as np

import torch
import pytorch_kinematics as pk
import time
import xmltodict

def add_base_joints(urdf_path, type="ground"):
    with open(urdf_path, 'r') as f:
        xml = xmltodict.parse(f.read())    
    # Add world
    xml['robot']['link'].insert(0, {'@name': "world", 'visual': [], 'collision': []})
    # Add base_link_x
    xml['robot']['link'].append({'@name': "base_link_x", })
    xml['robot']['joint'].append({'@name': "joint_base_link_x", '@type': "prismatic", 'origin': {'@rpy': "0 0 0", '@xyz': "0 0 0"}, 'parent': {'@link': "world"}, 'child': {'@link': "base_link_x"}, 'axis': {'@xyz': "1 0 0"}, 'limit': {'@effort': "100", '@lower': "-10", '@upper': "10", '@velocity': "1.0"}})
    # Add base_link_y
    xml['robot']['link'].append({'@name': "base_link_y", })
    xml['robot']['joint'].append({'@name': "joint_base_link_y", '@type': "prismatic", 'origin': {'@rpy': "0 0 0", '@xyz': "0 0 0"}, 'parent': {'@link': "base_link_x"}, 'child': {'@link': "base_link_y"}, 'axis': {'@xyz': "0 1 0"}, 'limit': {'@effort': "100", '@lower': "-10", '@upper': "10", '@velocity': "1.0"}})
    # if floating, add base_link_z
    if type == "floating":
        xml['robot']['link'].append({'@name': "base_link_z", })
        xml['robot']['joint'].append({'@name': "joint_base_link_z", '@type': "prismatic", 'origin': {'@rpy': "0 0 0", '@xyz': "0 0 0"}, 'parent': {'@link': "base_link_y"}, 'child': {'@link': "base_link_z"}, 'axis': {'@xyz': "0 0 1"}, 'limit': {'@effort': "100", '@lower': "-10", '@upper': "10", '@velocity': "1.0"}})
    
    
    # Add base_link_yaw
    xml['robot']['link'].append({'@name': "base_link_yaw", })
    if type == "floating":
        xml['robot']['joint'].append({'@name': "joint_base_link_yaw", '@type': "revolute", 'origin': {'@rpy': "0 0 0", '@xyz': "0 0 0"}, 'parent': {'@link': "base_link_z"}, 'child': {'@link': "base_link_yaw"}, 'axis': {'@xyz': "0 0 1"}, 'limit': {'@effort': "100", '@lower': "-3.14159", '@upper': "3.14159", '@velocity': "1.0"}})
    else:
        xml['robot']['joint'].append({'@name': "joint_base_link_yaw", '@type': "revolute", 'origin': {'@rpy': "0 0 0", '@xyz': "0 0 0"}, 'parent': {'@link': "base_link_y"}, 'child': {'@link': "base_link_yaw"}, 'axis': {'@xyz': "0 0 1"}, 'limit': {'@effort': "100", '@lower': "-3.14159", '@upper': "3.14159", '@velocity': "1.0"}})
    # Add joint_base_link
    xml['robot']['joint'].append({'@name': "joint_base_link", '@type': "fixed", 'origin': {'@rpy': "0 0 0", '@xyz': "0 0 0"}, 'parent': {'@link': "base_link_yaw"}, 'child': {'@link': "base_link"}})
    # Write to file
    patched_urdf_path = urdf_path.replace(".urdf", "_patched.urdf")
    with open(patched_urdf_path, 'w') as f:
        f.write(xmltodict.unparse(xml, pretty=True))
    return patched_urdf_path

class BatchFk():
    
    def __init__(self, urdf_path, ee_link_name, device, batch_size = 1):
        self.device = device
        # check if tuned urdf_file.pk.urdf file exists e.g. stretch.urdf will have a stretch.pk.urdf file
        
        
        # floating can be moved in xyz and theta, grounded can only be moved in xy and theta. TODO: Have not found a way to make this generic
        urdf_path = add_base_joints(urdf_path, type =  "ground")
        
        chain = pk.build_serial_chain_from_urdf(open(urdf_path, "rb").read(), ee_link_name)
        
        self.dof = 5
       
        self.chain = chain.to(device=device)
        self.batch_size = batch_size
        self.states = torch.zeros((batch_size, self.dof), dtype=torch.float32, device=device)
        self.pk_states = torch.zeros((batch_size, self.dof), dtype=torch.float32, device=device)
       
        self.camera_matrices = torch.zeros((batch_size, 4, 4), dtype=torch.float32, device=device)
      
        self.camera_positions_world = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)
        self.camera_quaternions_world = torch.zeros((batch_size, 4), dtype=torch.float32, device=device)
        

    def batch_fk(self, states : NDArray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Convert states to tensor and move to device
        # assert states.shape[0] == self.batch_size
        if isinstance(states, np.ndarray):
            self.states = torch.tensor(states, dtype=torch.float32, device=self.device)
        else:
            self.states = states
        # Separate base states and joint states if using standard URDF
        # self.base_states = states[:, 2:]  # [x, y, theta]
        # self.joint_states = states[:, :2]  # [pan, tilt]
        # These depend on how the states are ordered in the URDF, how they are interpreted by PytorchKinematics, and how Grapeshot sends the states
        # print("states[0]", states[0])
        self.pk_states = self.states
        
        # Perform FK in batch
        ret = self.chain.forward_kinematics(self.pk_states, end_only=True)
        tg = ret
        self.camera_matrices = tg.get_matrix()  # Shape: (batch_size, 4, 4)
        
        # For URDF with base joints
        self.camera_positions_world = self.camera_matrices[:, :3, 3]  # Shape: (batch_size, 3)
        self.camera_quaternions_world = pk.matrix_to_quaternion(self.camera_matrices[:, :3, :3])  # Shape: (batch_size, 4)
        # wxyz to xyzw
        self.camera_quaternions_world = torch.roll(self.camera_quaternions_world, shifts=-1, dims=1)
        
        return self.camera_positions_world, self.camera_quaternions_world, self.camera_matrices
    
    def transform_poses_batch(self, from_transforms, to_transforms):
        # Convert object pose to tensors
        # if receiving only one to_transform matrix, repeat it to match the batch size
        if len(to_transforms.size()) < 3 and len(from_transforms.size()) == 3:
            # repeat this 4,4 matrix to (batch_size, 4, 4)
            to_transforms = to_transforms.unsqueeze(0).repeat(from_transforms.size(0), 1, 1)
        # opposite case
        elif len(from_transforms.size()) < 3 and len(to_transforms.size()) ==3:
            # repeat this 4,4 matrix to (batch_size, 4, 4)
            from_transforms = from_transforms.unsqueeze(0).repeat(to_transforms.size(0), 1, 1)
        else:
            # repeat both over batch size 1
            from_transforms = from_transforms.unsqueeze(0).repeat(to_transforms.size(0), 1, 1)
            to_transforms = to_transforms.unsqueeze(0).repeat(from_transforms.size(0), 1, 1)
        # Preallocate output tensors
        transformed_matrices_tensor = torch.empty((from_transforms.size(0), 4, 4), dtype=torch.float32, device=self.device)
        transformed_poses_tensor = torch.empty((from_transforms.size(0), 7), dtype=torch.float32, device=self.device)
        # Start timing
        start_time = time.time()
        # Use batch_fk to compute all camera poses in a batch
        from_transforms_inv = self.invert_transformation_batch(from_transforms)
        transformed_matrices_tensor = torch.bmm(from_transforms_inv, to_transforms)
        # Extract positions from world transforms
        transformed_poses_tensor[:, :3] = transformed_matrices_tensor[:, :3, 3]  # Shape: (batch_size, 3)
        # Extract rotation matrices correctly
        transformed_poses_tensor[:, 3:] = pk.matrix_to_quaternion(transformed_matrices_tensor[:, :3, :3])  # Ensure shape: (batch_size, 3, 3)
        transformed_poses_tensor[:, 3:] = torch.roll(transformed_poses_tensor[:, 3:], shifts=-1, dims=1)
        return transformed_poses_tensor
    
    def pose_to_transformation_matrix(self, pose):
        transformation_matrix = torch.zeros((4, 4), dtype=torch.float32, device=self.device)
        # transform to wxyz from xyzw
        pose[3:] = torch.roll(pose[3:], shifts=1, dims=0)
        transformation_matrix[:3, :3] = pk.quaternion_to_matrix(pose[3:])
        transformation_matrix[:3, 3] = pose[:3]
        transformation_matrix[3, 3] = 1
        return transformation_matrix
    
    def invert_transformation_batch(self, T):
        """Invert a batch of 4x4 transformation matrices."""
        R = T[:, :3, :3]  # Rotation matrices
        t = T[:, :3, 3:]  # Translation vectors
        
        R_inv = R.transpose(-1, -2)  # Transpose the rotation matrix
        t_inv = -torch.bmm(R_inv, t)  # Invert the translation

        # Construct the inverted transformation matrix
        T_inv = torch.eye(4, device=T.device).repeat(T.shape[0], 1, 1)  # Batch of identity matrices
        T_inv[:, :3, :3] = R_inv
        T_inv[:, :3, 3:] = t_inv

        return T_inv
    
    def warmup(self):
        # do a dummy forward kinematics to warm up the model
        for i in range(5):
            self.batch_fk(torch.zeros((self.batch_size, self.dof), dtype=torch.float32, device=self.device))
            self.transform_poses_batch(torch.zeros((self.batch_size, 4, 4), dtype=torch.float32, device=self.device), torch.zeros((4, 4), dtype=torch.float32, device=self.device))
    


