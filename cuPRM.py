import torch

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
import numpy as np
# faiss, cuVS 

# need to build a state sampler class
    # This samples states in batch 
    # Uses an instance of ValidityChecker 


# need to build a ValidityChecker class
    # this checks validity in-batch for:
        # collision (world and self)
        # joint limits
        # pointing at the object

# need a Graph class to store the PRM graph
    # Will have methods to add nodes and edges
    # Will have method to find kNNs or aNNs (faiss, cuVS)
    # Will have method to assign scores to edges (perception model)

# Maybe Faiss and cuVS classes to handle faiss and cuVS operations

# need a PerceptionModel class
    # This will be a torch model
    # Will have a method to assign scores to nodes/edges

class PerceptionPRMPlanner():
    def __init__(
            self,
            robot_file,
            world_file,
            world_config,
            collision_activation_distance,
            num_nodes,
            perc_model,
            interpolation_steps=10,
      
                 
                 
                 ):
        
        self.robot_file = robot_file
        self.world_file = world_file
        self.world_config = world_config
        self.collision_activation_distance = collision_activation_distance

        self.config = RobotWorldConfig.load_from_config(
            robot_file, world_file, collision_activation_distance
        )

        self.curobo_fn = RobotWorld(self.config)

        self.num_nodes = num_nodes
        self.interpolation_steps = interpolation_steps

        self.model = perc_model

        self.prm_graph = None
        self.path = None

        self.tensor_args = TensorDeviceType()


        pass

    def sample_collision_free(self, curobo_fn, tensor_args) -> torch.Tensor:
        q_s_valid = curobo_fn.sample(self.num_nodes, mask_valid=True)   # Setting flag to false will return some samples in collision
        return q_s_valid
         
    def project(self, samples : torch.Tensor) -> torch.Tensor:
        pass



    def find_nearest_neighbors(self, samples : torch.Tensor) -> np.ndarray:
        samples = samples.cpu().numpy()
        # pass to faiss
        # return indices of nearest neighbors
        pass

    def create_edges(self) -> torch.Tensor:
        
        pass

    def assign_perc_scores(self) -> torch.Tensor:
        
        
        pass    

    def build_graph(self):
        samples = self.sample_collision_free()
        nearest_neighbors = self.find_nearest_neighbors(samples)
        edges = self.create_edges(nearest_neighbors)
        perc_scores = self.assign_perc_scores(edges)
        return samples, nearest_neighbors, edges, perc_scores
        
        

    def find_path(self):
        pass