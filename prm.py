import cuPRM
import torch

import numpy as np
import networkx as nx
import pytorch_kinematics as pk

from scipy.interpolate import make_interp_spline

# Each PSPRM instance has an associated model that it uses and environment that it represents,
# as well as a graph that it builds based on the environment.
class PSPRM:
    def __init__(self, model, env):
        self.graph = None   # netx graph
        self.model = model
        self.env = env

    def build_prm(self, seed):
        
        bounds = self.env['bounds']
        circles = self.env['circles']
        rectangles = self.env['rectangles']
        object_pose_world = self.env['object_pose']
        obj_label = self.env['object_label'] 
            
        nodes, node_validity, neighbors, edges, edge_validity = cuPRM.build_prm(
            circles, rectangles, bounds, seed
        )

        neighbors = torch.where(edge_validity, neighbors, -1)  # Replace invalid neighbors with -1
        diffs = self.project_nodes(nodes, object_pose_world)    # Project nodes and return diffs (difference between camera pose and object pose at each node)
        diffs = torch.cat((diffs, obj_label.unsqueeze(0).repeat(nodes.shape[0], 1)), dim=1)

        p_scores = self.model(diffs)
        self.graph = self.tensors_to_networkx(nodes, neighbors, p_scores)
        return
    

    def project_nodes(self, nodes, object_pose_world):
        """
        Calculate pan and tilt angles for robot states to look at target object,
        then compute camera poses and diffs from object.
        """
        device = nodes.device
        dtype = nodes.dtype
        B = nodes.shape[0]
        
        # Camera configuration (same as original script)
        cam_pos_robot = torch.tensor([0.0450, 0.0523, 1.2607], dtype=dtype, device=device)
        cam_rot_robot = torch.tensor(
            [[0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]], dtype=dtype, device=device
        )
        
        # Extract robot states from nodes
        robot_states = nodes[:, :3]  # [x, y, theta]

        # Prepare object position in homogeneous coordinates
        object_pos_world_h = torch.cat([object_pose_world[:3], torch.ones(1, dtype=dtype, device=device)])
        object_pos_world_h = object_pos_world_h.unsqueeze(0).expand(B, -1)
        
        # Create batched camera base transforms in robot frame
        T_robot_camera_base = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
        T_robot_camera_base[:, :3, :3] = cam_rot_robot.unsqueeze(0).repeat(B, 1, 1)
        T_robot_camera_base[:, :3, 3] = cam_pos_robot.unsqueeze(0).repeat(B, 1)
        
        # Build robot base-to-world transforms
        cos_theta = torch.cos(robot_states[:, 2])
        sin_theta = torch.sin(robot_states[:, 2])
        
        T_world_robot = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
        T_world_robot[:, 0, 0] = cos_theta
        T_world_robot[:, 0, 1] = -sin_theta
        T_world_robot[:, 1, 0] = sin_theta
        T_world_robot[:, 1, 1] = cos_theta
        T_world_robot[:, 0, 3] = robot_states[:, 0]
        T_world_robot[:, 1, 3] = robot_states[:, 1]
        
        # Camera base pose in world
        T_world_cam_base = torch.bmm(T_world_robot, T_robot_camera_base)
        
        # Invert camera transforms and transform object to camera frame
        T_cam_base_world = torch.inverse(T_world_cam_base)
        obj_in_cam = torch.bmm(T_cam_base_world, object_pos_world_h.unsqueeze(-1)).squeeze(-1)[:, :3]
        
        # Calculate pan and tilt angles
        dx, dy, dz = obj_in_cam[:, 0], obj_in_cam[:, 1], obj_in_cam[:, 2]
        pan = torch.atan2(dx, dz)
        tilt = torch.atan2(-dy, torch.sqrt(dx**2 + dz**2))
       
        # Pan bounds are [-3.9, 1.5]
        # Tilt bounds are [-1.53, 0.79]
        pan = torch.clamp(pan, -3.9, 1.5)
        tilt = torch.clamp(tilt, -1.53, 0.79)
        # pan = torch.clamp(torch.atan2(dx, dz), -3.9, 1.5)
        # tilt = torch.clamp(torch.atan2(-dy, torch.sqrt(dx**2 + dz**2)), -1.53, 0.79)

        # Modify nodes in place with calculated pan and tilt
        nodes[:, 3] = pan  
        nodes[:, 4] = tilt  
        
        # Now compute the final camera pose in world frame with pan/tilt applied
        # Create pan/tilt transform
        T_cam_base_pantilt = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
        
        # Apply rotation for pan (around y-axis)
        T_cam_base_pantilt[:, 0, 0] = torch.cos(pan)
        T_cam_base_pantilt[:, 0, 2] = torch.sin(pan)
        T_cam_base_pantilt[:, 2, 0] = -torch.sin(pan)
        T_cam_base_pantilt[:, 2, 2] = torch.cos(pan)
        
        # Apply rotation for tilt (around x-axis)
        T_tilt = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
        T_tilt[:, 1, 1] = torch.cos(tilt)
        T_tilt[:, 1, 2] = -torch.sin(tilt)
        T_tilt[:, 2, 1] = torch.sin(tilt)
        T_tilt[:, 2, 2] = torch.cos(tilt)
        
        # Combine pan and tilt transforms
        T_cam_base_pantilt = torch.bmm(T_cam_base_pantilt, T_tilt)
        
        # Apply to get final camera transform in world frame
        T_world_cam = torch.bmm(T_world_cam_base, T_cam_base_pantilt)
        
        # Extract camera position and orientation
        cam_pos_world = T_world_cam[:, :3, 3]
        cam_quat_world = pk.matrix_to_quaternion(T_world_cam[:, :3, :3])
        
        # Concatenate camera world pose and quaternion into B x 7 tensor
        cam_pose_world = torch.cat([cam_pos_world, cam_quat_world], dim=1)
        
        # Compute differences between camera pose and object pose
        diffs = cam_pose_world - object_pose_world
        return diffs
    
    
    def tensors_to_networkx(self, nodes_tensor, neighbors_tensor, scores_tensor=None):
        if nodes_tensor.device.type == 'cuda':
            nodes = nodes_tensor.cpu().numpy()
            neighbors = neighbors_tensor.cpu().numpy()
            if scores_tensor is not None:
                scores = scores_tensor.cpu().numpy()
        else:
            nodes = nodes_tensor.detach().numpy()
            neighbors = neighbors_tensor.detach().numpy()
            if scores_tensor is not None:
                scores = scores_tensor.detach().numpy()
        
        N = nodes.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(N))
        
        nx.set_node_attributes(G, dict(enumerate(nodes[:, 0])), 'x')
        nx.set_node_attributes(G, dict(enumerate(nodes[:, 1])), 'y')
        nx.set_node_attributes(G, dict(enumerate(nodes[:, 2])), 'theta')
        nx.set_node_attributes(G, dict(enumerate(nodes[:, 3])), 'pan')
        nx.set_node_attributes(G, dict(enumerate(nodes[:, 4])), 'tilt')
        
        if scores_tensor is not None:
            nx.set_node_attributes(G, dict(enumerate(scores)), 'score')
        
        valid_mask = (neighbors >= 0) & (neighbors < N)
        source_indices, neighbor_indices = np.where(valid_mask)
        target_indices = neighbors[source_indices, neighbor_indices]

        # Calculation of edge weights
        src_positions = nodes[source_indices, :2]  # Shape: (num_edges, 2)
        tgt_positions = nodes[target_indices, :2]  # Shape: (num_edges, 2)
        weights = np.linalg.norm(src_positions - tgt_positions, axis=1)  
        
        edge_list_with_weights = list(zip(source_indices.tolist(), 
                                        target_indices.tolist(), 
                                        weights.tolist()))
        
        G.add_weighted_edges_from(edge_list_with_weights)
        
        return G
    
    def a_star_search(self, start_node, goal_node, alpha = 1.0, beta = 0.5):

        edge_weights = [data['weight'] for u, v, data in self.graph.edges(data=True)]
        if edge_weights:
            max_weight = max(edge_weights)
            min_weight = min(edge_weights)
            weight_range = max_weight - min_weight if max_weight > min_weight else 1.0
        else:
            weight_range = 1.0

        def heuristic(u, v):
            # Get node positions (x, y coordinates)
            u_pos = np.array([self.graph.nodes[u]['x'], self.graph.nodes[u]['y']])
            v_pos = np.array([self.graph.nodes[v]['x'], self.graph.nodes[v]['y']])
            
            # Get node scores
            u_score = self.graph.nodes[u].get('score', 0)
            v_score = self.graph.nodes[v].get('score', 0)
            
            # Perception cost (using current node's score)
            
            # Motion cost (Euclidean distance between x, y positions)
            motion_cost = np.linalg.norm(u_pos - v_pos) / weight_range
            
            # Combined heuristic: prioritize high scores, minimize motion cost
            return -beta * min(u_score, v_score) + alpha * motion_cost
        
        try:
            path = nx.astar_path(
                self.graph, 
                start_node, 
                goal_node, 
                heuristic=heuristic,
                weight='weight'
            )
            return path
        except nx.NetworkXNoPath:
            print(f"No path found from {start_node} to {goal_node}")
            return None




class Solution:
    def __init__(self, path):
        self.path = path
        self.trajectory = None
    
    def set_path(self, path):
        self.path = path
        self.trajectory = None

    def generate_trajectory(self, graph, num_points=1000, degree=3):

        if not self.path or len(self.path) < 2:
            raise ValueError("Path must contain at least 2 nodes")
    
        x_coords = nx.get_node_attributes(graph, 'x')
        y_coords = nx.get_node_attributes(graph, 'y')
        pan = nx.get_node_attributes(graph, 'pan')
        tilt = nx.get_node_attributes(graph, 'tilt')


        if not x_coords or not y_coords or not pan or not tilt:
            raise ValueError("Graph nodes must have all attributes")
        
        # Vectorized coordinate extraction using list comprehension + array conversion
        try:
            waypoints = np.array([[x_coords[node_id], y_coords[node_id], pan[node_id], tilt[node_id]] for node_id in self.path])
        except KeyError as e:
            raise ValueError(f"Node {e} not found in graph")

        # Handle edge case with two waypoints
        if len(waypoints) == 2:
            # Linear interpolation for 2 points
            t = np.linspace(0, 1, num_points)[:, np.newaxis]  
            trajectory = waypoints[0] + t * (waypoints[1] - waypoints[0]) 
            self.trajectory = trajectory
            return trajectory

        t = np.linspace(0, 1, len(waypoints))
    
        spline_x = make_interp_spline(t, waypoints[:, 0], k=degree)
        spline_y = make_interp_spline(t, waypoints[:, 1], k=degree)
        spline_pan = make_interp_spline(t, waypoints[:, 2], k=degree)
        spline_tilt = make_interp_spline(t, waypoints[:, 3], k=degree)

        # Get analytical derivatives directly from the splines
        spline_x_dot = spline_x.derivative()
        spline_y_dot = spline_y.derivative()


        t_new = np.linspace(0, 1, num_points)
        theta = np.arctan2(spline_y_dot(t_new), spline_x_dot(t_new))

        trajectory = np.column_stack([spline_x(t_new), spline_y(t_new), theta, spline_pan(t_new), spline_tilt(t_new)])
        #velocity = np.column_stack([spline_x.derivative()(t_new), spline_y.derivative()(t_new)])
        
        return trajectory #, velocity