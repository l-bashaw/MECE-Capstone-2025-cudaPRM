import torch

import numpy as np
import networkx as nx
import pytorch_kinematics as pk

import cuPRM
from scipy.interpolate import make_interp_spline, CubicHermiteSpline

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
        print(diffs[:5])
        p_scores = self.model(diffs)
        print(f"Max score: {p_scores.max().item():.4f}, Min score: {p_scores.min().item():.4f}, Mean score: {p_scores.mean().item():.4f}")
        self.graph = self.tensors_to_networkx(nodes, neighbors, p_scores)
        return
    
    def add_start_and_goal(self, start, goal, call_id, k_neighbors=10):
        """
        Given start and goal dicts with keys 'x', 'y', 'theta', calculate pan/tilt to look at object,
        add them to the graph, and connect to k nearest neighbors.

        """
        device = torch.device('cuda')
        dtype = torch.float32
        
        start_tensor = torch.tensor([[start['x'], start['y'], start['theta'], 0.0, 0.0]], dtype=dtype, device=device)
        goal_tensor = torch.tensor([[goal['x'], goal['y'], goal['theta'], 0.0, 0.0]], dtype=dtype, device=device)
        
        combined = torch.cat((start_tensor, goal_tensor), dim=0)
        object_pose_world = self.env['object_pose']
        combined_diffs = self.project_nodes(combined, object_pose_world)
        scores = self.model(torch.cat((combined_diffs, self.env['object_label'].unsqueeze(0).repeat(2, 1)), dim=1))
        
        start_attrs = {
            'x': start['x'],
            'y': start['y'],
            'theta': start['theta'],
            'pan': combined[0, 3].cpu().item(),
            'tilt': combined[0, 4].cpu().item(),
            'score': scores[0].cpu().item()
        }
        
        goal_attrs = {
            'x': goal['x'],
            'y': goal['y'],
            'theta': goal['theta'],
            'pan': combined[1, 3].cpu().item(),
            'tilt': combined[1, 4].cpu().item(),
            'score': scores[1].cpu().item()
        }

        print(start_attrs)

        G = self.graph
        # Add start and goal nodes
        offset = 2 * call_id
        start_id = max(G.nodes) + 1 + offset
        goal_id = start_id + 1

        def get_neighbors(node_dict, k):
            nodes = np.array([[G.nodes[n]['x'], G.nodes[n]['y'], G.nodes[n]['theta']] for n in G.nodes])
            node_vec = np.array([node_dict['x'], node_dict['y'], node_dict['theta']])
            dists = np.linalg.norm(nodes - node_vec, axis=1)
            neighbor_ids = np.argsort(dists)[:k]
            return [list(G.nodes)[i] for i in neighbor_ids]
        
        start_neighbors = get_neighbors(start, k_neighbors)
        goal_neighbors = get_neighbors(goal, k_neighbors)

        G.add_node(start_id, **start_attrs)
        G.add_node(goal_id, **goal_attrs)

        for nid in start_neighbors:
            src = np.array([start['x'], start['y']])
            tgt = np.array([G.nodes[nid]['x'], G.nodes[nid]['y']])
            weight = np.linalg.norm(src - tgt)
            G.add_edge(start_id, nid, weight=weight)

        for nid in goal_neighbors:
            src = np.array([goal['x'], goal['y']])
            tgt = np.array([G.nodes[nid]['x'], G.nodes[nid]['y']])
            weight = np.linalg.norm(src - tgt)
            G.add_edge(goal_id, nid, weight=weight)

        
        return start_id, goal_id

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
        
        # Invert camera transforms and transform object to camera frame, calculate pan/tilt
        T_cam_base_world = torch.inverse(T_world_cam_base)
        obj_in_cam = torch.bmm(T_cam_base_world, object_pos_world_h.unsqueeze(-1)).squeeze(-1)[:, :3]
        dx, dy, dz = obj_in_cam[:, 0], obj_in_cam[:, 1], obj_in_cam[:, 2]
        pan = torch.atan2(dx, dz)
        tilt = torch.atan2(dy, torch.sqrt(dx**2 + dz**2))
        print(obj_in_cam.shape)
        # Pan bounds are [-3.9, 1.5]
        # Tilt bounds are [-1.53, 0.79]
        pan = torch.clamp(pan, -3.9, 1.5)
        tilt = torch.clamp(tilt, -1.53, 0.79)
       
        # Modify nodes in place with calculated pan and tilt
        nodes[:, 3] = pan  
        nodes[:, 4] = tilt  
        
        # # Now compute the final camera pose in world frame with pan/tilt applied
        # # Create pan/tilt transform
        # T_cam_base_pantilt = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
        
        # # Apply rotation for pan (around y-axis)
        # T_cam_base_pantilt[:, 0, 0] = torch.cos(pan)
        # T_cam_base_pantilt[:, 0, 2] = torch.sin(pan)
        # T_cam_base_pantilt[:, 2, 0] = -torch.sin(pan)
        # T_cam_base_pantilt[:, 2, 2] = torch.cos(pan)
        
        # # Apply rotation for tilt (around x-axis)
        # T_tilt = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
        # T_tilt[:, 1, 1] = torch.cos(tilt)
        # T_tilt[:, 1, 2] = -torch.sin(tilt)
        # T_tilt[:, 2, 1] = torch.sin(tilt)
        # T_tilt[:, 2, 2] = torch.cos(tilt)
        
        # T_cam_base_pantilt = torch.bmm(T_cam_base_pantilt, T_tilt)
        
        # # Apply to get final camera transform in world frame
        # T_world_cam = torch.bmm(T_world_cam_base, T_cam_base_pantilt)
        
        # # Extract camera position and orientation
        # cam_pos_world = T_world_cam[:, :3, 3]
        # cam_quat_world = pk.matrix_to_quaternion(T_world_cam[:, :3, :3])
        
        # cam_pose_world = torch.cat([cam_pos_world, cam_quat_world], dim=1)
        
        # # Compute differences between camera pose and object pose
        # diffs = cam_pose_world - object_pose_world

        # Compute the relative orientation as a quaternion
        # The camera's orientation in the world frame is T_world_cam_base[:, :3, :3]
        # The relative orientation of the object in the camera frame is derived from T_cam_base_world
        R_cam_base_world = T_cam_base_world[:, :3, :3]
        obj_quat_in_cam = pk.matrix_to_quaternion(R_cam_base_world)

        diffs = torch.cat([obj_in_cam, obj_quat_in_cam], dim=1)  # Shape: [B, 7]
        return diffs
    
    
    def tensors_to_networkx(self, nodes_tensor, neighbors_tensor, scores_tensor):
        if nodes_tensor.device.type == 'cuda':
            nodes = nodes_tensor.cpu().numpy()
            neighbors = neighbors_tensor.cpu().numpy()
            scores = scores_tensor.detach().cpu().numpy()
        else:
            nodes = nodes_tensor.detach().numpy()
            neighbors = neighbors_tensor.detach().numpy()
            scores = scores_tensor.detach().numpy()
        
        N = nodes.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(N))
        
        nx.set_node_attributes(G, dict(enumerate(nodes[:, 0])), 'x')
        nx.set_node_attributes(G, dict(enumerate(nodes[:, 1])), 'y')
        nx.set_node_attributes(G, dict(enumerate(nodes[:, 2])), 'theta')
        nx.set_node_attributes(G, dict(enumerate(nodes[:, 3])), 'pan')
        nx.set_node_attributes(G, dict(enumerate(nodes[:, 4])), 'tilt')
        nx.set_node_attributes(G, dict(enumerate(scores)), 'score')
        
        valid_mask = (neighbors >= 0) & (neighbors < N)
        source_indices, neighbor_indices = np.where(valid_mask)
        target_indices = neighbors[source_indices, neighbor_indices]

        # Calculation of edge weights
        src_positions = nodes[source_indices, :3]  # Shape: (num_edges, 2)  #change to 3 for 3D
        tgt_positions = nodes[target_indices, :3]  # Shape: (num_edges, 2)
        weights = np.linalg.norm(src_positions - tgt_positions, axis=1)  
        
        edge_list_with_weights = list(zip(source_indices.tolist(), 
                                        target_indices.tolist(), 
                                        weights.tolist()))
        
        G.add_weighted_edges_from(edge_list_with_weights)
        
        return G
    

    def a_star_search(self, start_id, goal_id, alpha=1, beta=1):
        """
        Call_id: unique identifier for this search instance, needed to know which start and goal we're talking about.
        Runs A* from start to goal, does not need to take them as params.
        """
        G = self.graph
        
        # Heuristic function
        edge_weights = [data['weight'] for u, v, data in G.edges(data=True)]
        if edge_weights:
            max_weight = max(edge_weights)
            min_weight = min(edge_weights)
            weight_range = max_weight - min_weight if max_weight > min_weight else 1.0
        else:
            weight_range = 1.0

        max_score = max(nx.get_node_attributes(G, 'score').values())
        min_score = min(nx.get_node_attributes(G, 'score').values())
        score_range = max_score - min_score if max_score > min_score else 1.0

        def heuristic(u, v):
            u_score = (G.nodes[u]['score'] - min_score) / score_range
            v_score = (G.nodes[v]['score'] - min_score) / score_range

            # Set motion cost to the edge weight
            if G.has_edge(u, v):
                motion_cost = G[u][v]['weight'] / weight_range  
                
            else:
                motion_cost = float('inf')  # No direct edge, set to infinity

            # Use the average of the scores instead of the minimum
            score_term = (u_score + v_score) / 2
            val = -beta * score_term + alpha * motion_cost

            # Adjust the heuristic to prioritize high scores more strongly
            return val
        
        try:
            path = nx.astar_path(
                G,
                start_id,
                goal_id,
                heuristic=heuristic,
                weight='weight'
            )

            total_cost = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_cost = G[u][v]['weight']
                heuristic_cost = heuristic(u, v)
                total_cost += edge_cost
                print(f"\nEdge ({u} -> {v}): edge_cost={edge_cost:.4f}, heuristic_cost={heuristic_cost:.4f}, total_cost={total_cost:.4f}")
            
            print(f"\nFinal total cost: {total_cost:.4f}")

            return path
        
        except nx.NetworkXNoPath:
            print(f"No path found from {start_id} to {goal_id}")
            return None




class Solution:
    def __init__(self, path):
        self.path = path
        self.trajectory = None
    
    def set_path(self, path):
        self.path = path
        self.trajectory = None

    def generate_trajectory2(self, graph, num_points=1000, degree=3):
        if not self.path or len(self.path) < 2:
            raise ValueError("Path must contain at least 2 nodes")
        
        x_coords = nx.get_node_attributes(graph, 'x')
        y_coords = nx.get_node_attributes(graph, 'y')
        theta_coords = nx.get_node_attributes(graph, 'theta')
        pan_coords = nx.get_node_attributes(graph, 'pan')
        tilt_coords = nx.get_node_attributes(graph, 'tilt')

        if not x_coords or not y_coords or not theta_coords:
            raise ValueError("Graph nodes must have all attributes")
        
        # Vectorized coordinate extraction using list comprehension + array conversion
        try:
            waypoints = np.array([[x_coords[node_id], y_coords[node_id], theta_coords[node_id], pan_coords[node_id],tilt_coords[node_id]] for node_id in self.path])
        except KeyError as e:
            raise ValueError(f"Node {e} not found in graph")
        
        # Extract coordinates
        x_waypoints = waypoints[:, 0]
        y_waypoints = waypoints[:, 1]
        thetas = waypoints[:, 2]
        pans = waypoints[:, 3] 
        tilts = waypoints[:, 4]
        
        # Calculate cumulative distances for parameterization
        # This gives us a more natural parameterization than simple indexing
        distances = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x_waypoints)**2 + np.diff(y_waypoints)**2))])
        
        # Convert theta angles to tangent vectors (derivatives)
        # We need to estimate appropriate magnitudes for the tangent vectors
        # Use the distances between consecutive waypoints as a scaling factor
        tangent_magnitudes = np.ones(len(thetas))
        
        # For interior points, use average of adjacent segment lengths
        if len(thetas) > 2:
            segment_lengths = np.diff(distances)
            tangent_magnitudes[1:-1] = (segment_lengths[:-1] + segment_lengths[1:]) / 2
            tangent_magnitudes[0] = segment_lengths[0]
            tangent_magnitudes[-1] = segment_lengths[-1]
        else:
            # For just 2 points, use the distance between them
            tangent_magnitudes[:] = distances[-1]
        
        # Calculate tangent vectors (vectorized)
        dx_dt = tangent_magnitudes * np.cos(thetas)
        dy_dt = tangent_magnitudes * np.sin(thetas)
        
        # Create cubic Hermite splines for x and y coordinates
        spline_x = CubicHermiteSpline(distances, x_waypoints, dx_dt)
        spline_y = CubicHermiteSpline(distances, y_waypoints, dy_dt)
        spline_pan = CubicHermiteSpline(distances, pans, np.gradient(pans, distances))
        spline_tilt = CubicHermiteSpline(distances, tilts, np.gradient(tilts, distances))
        
        # Generate trajectory points
        t_trajectory = np.linspace(distances[0], distances[-1], num_points)
        x_trajectory = spline_x(t_trajectory)
        y_trajectory = spline_y(t_trajectory)
        pan_trajectory = spline_pan(t_trajectory)
        tilt_trajectory = spline_tilt(t_trajectory)
        
        # Calculate tangent angles along the trajectory (optional, for verification)
        dx_trajectory = spline_x.derivative()(t_trajectory)
        dy_trajectory = spline_y.derivative()(t_trajectory)
        theta_trajectory = np.arctan2(dy_trajectory, dx_trajectory)
        
        # Return trajectory as structured array or dictionary
        trajectory = np.column_stack([
            x_trajectory,
            y_trajectory,
            theta_trajectory,
            pan_trajectory,
            tilt_trajectory
        ])
        self.trajectory = trajectory
        return trajectory
        
    def generate_trajectory(self, graph, num_points=1000, degree=3):
        if not self.path or len(self.path) < 2:
            raise ValueError("Path must contain at least 2 nodes")

        x_coords = nx.get_node_attributes(graph, 'x')
        y_coords = nx.get_node_attributes(graph, 'y')
        theta_coords = nx.get_node_attributes(graph, 'theta')
        pan = nx.get_node_attributes(graph, 'pan')
        tilt = nx.get_node_attributes(graph, 'tilt')

        if not x_coords or not y_coords or not pan or not tilt or not theta_coords:
            raise ValueError("Graph nodes must have all attributes")

        # Vectorized coordinate extraction using list comprehension + array conversion
        try:
            waypoints = np.array([[x_coords[node_id], y_coords[node_id], theta_coords[node_id], pan[node_id], tilt[node_id]] for node_id in self.path])
        except KeyError as e:
            raise ValueError(f"Node {e} not found in graph")

        # Handle edge case with two waypoints
        if len(waypoints) == 2:
            t = np.linspace(0, 1, num_points)[:, np.newaxis]
            trajectory = waypoints[0] + t * (waypoints[1] - waypoints[0])
            self.trajectory = trajectory
            return trajectory

        t = np.linspace(0, 1, len(waypoints))
        spline_x = make_interp_spline(t, waypoints[:, 0], k=degree)
        spline_y = make_interp_spline(t, waypoints[:, 1], k=degree)
        spline_theta = make_interp_spline(t, waypoints[:, 2], k=degree)
        spline_pan = make_interp_spline(t, waypoints[:, 3], k=degree)
        spline_tilt = make_interp_spline(t, waypoints[:, 4], k=degree)

        t_new = np.linspace(0, 1, num_points)
        theta = spline_theta(t_new)

        # Ensure first and last theta match start/end
        theta[0] = waypoints[0, 2]
        theta[-1] = waypoints[-1, 2]

        trajectory = np.column_stack([
            spline_x(t_new),
            spline_y(t_new),
            theta,
            spline_pan(t_new),
            spline_tilt(t_new)
        ])
        self.trajectory = trajectory
        return trajectory


    def print_path(self, graph):
        if not self.path:
            print("No path to print")
            return
        
        for node_id in self.path:
            node_data = graph.nodes[node_id]
            print(f"Node {node_id}: x={node_data['x']:.2f}, y={node_data['y']:.2f}, theta={node_data['theta']:.2f}, pan={node_data['pan']:.2f}, tilt={node_data['tilt']:.2f}, score={node_data.get('score', 0):.4f}")




# # Each PSPRM instance has an associated model that it uses and environment that it represents,
# # as well as a graph that it builds based on the environment.
# class PSPRM:
#     def __init__(self, model, env):
#         self.graph = None   # netx graph
#         self.model = model
#         self.env = env

#     def build_prm(self, seed):
        
#         bounds = self.env['bounds']
#         circles = self.env['circles']
#         rectangles = self.env['rectangles']
#         object_pose_world = self.env['object_pose']
#         obj_label = self.env['object_label'] 
            
#         nodes, node_validity, neighbors, edges, edge_validity = cuPRM.build_prm(
#             circles, rectangles, bounds, seed
#         )

#         neighbors = torch.where(edge_validity, neighbors, -1)  # Replace invalid neighbors with -1
#         diffs = self.project_nodes(nodes, object_pose_world)    # Project nodes and return diffs (difference between camera pose and object pose at each node)
#         diffs = torch.cat((diffs, obj_label.unsqueeze(0).repeat(nodes.shape[0], 1)), dim=1)

#         p_scores = self.model(diffs)
#         print(f"Max score: {p_scores.max().item():.4f}, Min score: {p_scores.min().item():.4f}, Mean score: {p_scores.mean().item():.4f}")
#         self.graph = self.tensors_to_networkx(nodes, neighbors, p_scores)
#         return
    

#     def project_nodes(self, nodes, object_pose_world):
#         """
#         Calculate pan and tilt angles for robot states to look at target object,
#         then compute camera poses and diffs from object.
#         """
#         device = nodes.device
#         dtype = nodes.dtype
#         B = nodes.shape[0]
        
#         # Camera configuration (same as original script)
#         cam_pos_robot = torch.tensor([0.0450, 0.0523, 1.2607], dtype=dtype, device=device)
#         cam_rot_robot = torch.tensor(
#             [[0.0, 0.0, 1.0],
#             [1.0, 0.0, 0.0],
#             [0.0, 1.0, 0.0]], dtype=dtype, device=device
#         )
        
#         # Extract robot states from nodes
#         robot_states = nodes[:, :3]  # [x, y, theta]

#         # Prepare object position in homogeneous coordinates
#         object_pos_world_h = torch.cat([object_pose_world[:3], torch.ones(1, dtype=dtype, device=device)])
#         object_pos_world_h = object_pos_world_h.unsqueeze(0).expand(B, -1)
        
#         # Create batched camera base transforms in robot frame
#         T_robot_camera_base = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
#         T_robot_camera_base[:, :3, :3] = cam_rot_robot.unsqueeze(0).repeat(B, 1, 1)
#         T_robot_camera_base[:, :3, 3] = cam_pos_robot.unsqueeze(0).repeat(B, 1)
        
#         # Build robot base-to-world transforms
#         cos_theta = torch.cos(robot_states[:, 2])
#         sin_theta = torch.sin(robot_states[:, 2])
        
#         T_world_robot = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
#         T_world_robot[:, 0, 0] = cos_theta
#         T_world_robot[:, 0, 1] = -sin_theta
#         T_world_robot[:, 1, 0] = sin_theta
#         T_world_robot[:, 1, 1] = cos_theta
#         T_world_robot[:, 0, 3] = robot_states[:, 0]
#         T_world_robot[:, 1, 3] = robot_states[:, 1]
        
#         # Camera base pose in world
#         T_world_cam_base = torch.bmm(T_world_robot, T_robot_camera_base)
        
#         # Invert camera transforms and transform object to camera frame
#         T_cam_base_world = torch.inverse(T_world_cam_base)
#         obj_in_cam = torch.bmm(T_cam_base_world, object_pos_world_h.unsqueeze(-1)).squeeze(-1)[:, :3]
        
#         # Calculate pan and tilt angles
#         dx, dy, dz = obj_in_cam[:, 0], obj_in_cam[:, 1], obj_in_cam[:, 2]
#         pan = torch.atan2(dx, dz)
#         tilt = torch.atan2(-dy, torch.sqrt(dx**2 + dz**2))
       
#         # Pan bounds are [-3.9, 1.5]
#         # Tilt bounds are [-1.53, 0.79]
#         pan = torch.clamp(pan, -3.9, 1.5)
#         tilt = torch.clamp(tilt, -1.53, 0.79)
#         # pan = torch.clamp(torch.atan2(dx, dz), -3.9, 1.5)
#         # tilt = torch.clamp(torch.atan2(-dy, torch.sqrt(dx**2 + dz**2)), -1.53, 0.79)

#         # Modify nodes in place with calculated pan and tilt
#         nodes[:, 3] = pan  
#         nodes[:, 4] = tilt  
        
#         # Now compute the final camera pose in world frame with pan/tilt applied
#         # Create pan/tilt transform
#         T_cam_base_pantilt = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
        
#         # Apply rotation for pan (around y-axis)
#         T_cam_base_pantilt[:, 0, 0] = torch.cos(pan)
#         T_cam_base_pantilt[:, 0, 2] = torch.sin(pan)
#         T_cam_base_pantilt[:, 2, 0] = -torch.sin(pan)
#         T_cam_base_pantilt[:, 2, 2] = torch.cos(pan)
        
#         # Apply rotation for tilt (around x-axis)
#         T_tilt = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
#         T_tilt[:, 1, 1] = torch.cos(tilt)
#         T_tilt[:, 1, 2] = -torch.sin(tilt)
#         T_tilt[:, 2, 1] = torch.sin(tilt)
#         T_tilt[:, 2, 2] = torch.cos(tilt)
        
#         # Combine pan and tilt transforms
#         T_cam_base_pantilt = torch.bmm(T_cam_base_pantilt, T_tilt)
        
#         # Apply to get final camera transform in world frame
#         T_world_cam = torch.bmm(T_world_cam_base, T_cam_base_pantilt)
        
#         # Extract camera position and orientation
#         cam_pos_world = T_world_cam[:, :3, 3]
#         cam_quat_world = pk.matrix_to_quaternion(T_world_cam[:, :3, :3])
        
#         # Concatenate camera world pose and quaternion into B x 7 tensor
#         cam_pose_world = torch.cat([cam_pos_world, cam_quat_world], dim=1)
        
#         # Compute differences between camera pose and object pose
#         diffs = cam_pose_world - object_pose_world
#         return diffs
    
    
#     def tensors_to_networkx(self, nodes_tensor, neighbors_tensor, scores_tensor):
#         if nodes_tensor.device.type == 'cuda':
#             nodes = nodes_tensor.cpu().numpy()
#             neighbors = neighbors_tensor.cpu().numpy()
#             scores = scores_tensor.detach().cpu().numpy()
#         else:
#             nodes = nodes_tensor.detach().numpy()
#             neighbors = neighbors_tensor.detach().numpy()
#             scores = scores_tensor.detach().numpy()
        
#         N = nodes.shape[0]
#         G = nx.Graph()
#         G.add_nodes_from(range(N))
        
#         nx.set_node_attributes(G, dict(enumerate(nodes[:, 0])), 'x')
#         nx.set_node_attributes(G, dict(enumerate(nodes[:, 1])), 'y')
#         nx.set_node_attributes(G, dict(enumerate(nodes[:, 2])), 'theta')
#         nx.set_node_attributes(G, dict(enumerate(nodes[:, 3])), 'pan')
#         nx.set_node_attributes(G, dict(enumerate(nodes[:, 4])), 'tilt')
#         nx.set_node_attributes(G, dict(enumerate(scores)), 'score')
        
#         valid_mask = (neighbors >= 0) & (neighbors < N)
#         source_indices, neighbor_indices = np.where(valid_mask)
#         target_indices = neighbors[source_indices, neighbor_indices]

#         # Calculation of edge weights
#         src_positions = nodes[source_indices, :2]  # Shape: (num_edges, 2)
#         tgt_positions = nodes[target_indices, :2]  # Shape: (num_edges, 2)
#         weights = np.linalg.norm(src_positions - tgt_positions, axis=1)  
        
#         edge_list_with_weights = list(zip(source_indices.tolist(), 
#                                         target_indices.tolist(), 
#                                         weights.tolist()))
        
#         G.add_weighted_edges_from(edge_list_with_weights)
        
#         return G
    

#     def a_star_search(self, start, goal, alpha=1.0, beta=0.5, k_neighbors=10):
#         """
#         start, goal: dicts with keys 'x', 'y', 'theta'.
#         Adds start/goal to graph, connects to nearest neighbors, and runs A*.
#         """
#         import copy
#         G = self.graph
#         next_node_id = max(G.nodes) + 1
#         start_id = next_node_id
#         goal_id = next_node_id + 1

#         # Helper to copy attributes from nearest neighbor
#         def copy_attrs(neighbor_id, node_dict):
#             attrs = {}
#             for key in ['pan', 'tilt', 'score']:
#                 attrs[key] = G.nodes[neighbor_id].get(key, 0)
#             attrs.update(node_dict)
#             return attrs

#         # Add start/goal nodes
#         # Find nearest neighbors in (x, y, theta) space
#         def get_neighbors(node_dict, k):
#             nodes = np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in G.nodes])
#             node_vec = np.array([node_dict['x'], node_dict['y']])
#             dists = np.linalg.norm(nodes - node_vec, axis=1)
#             neighbor_ids = np.argsort(dists)[:k]
#             return [list(G.nodes)[i] for i in neighbor_ids]

#         start_neighbors = get_neighbors(start, k_neighbors)
#         goal_neighbors = get_neighbors(goal, k_neighbors)

#         # Copy pan/tilt/score from nearest neighbor
#         start_attrs = copy_attrs(start_neighbors[0], start)
#         goal_attrs = copy_attrs(goal_neighbors[0], goal)

#         G.add_node(start_id, **start_attrs)
#         G.add_node(goal_id, **goal_attrs)

#         # Connect start/goal to neighbors
#         for nid in start_neighbors:
#             src = np.array([start['x'], start['y']])
#             tgt = np.array([G.nodes[nid]['x'], G.nodes[nid]['y']])
#             weight = np.linalg.norm(src - tgt)
#             G.add_edge(start_id, nid, weight=weight)

#         for nid in goal_neighbors:
#             src = np.array([goal['x'], goal['y']])
#             tgt = np.array([G.nodes[nid]['x'], G.nodes[nid]['y']])
#             weight = np.linalg.norm(src - tgt)
#             G.add_edge(goal_id, nid, weight=weight)

#         # Heuristic function
#         edge_weights = [data['weight'] for u, v, data in G.edges(data=True)]
#         if edge_weights:
#             max_weight = max(edge_weights)
#             min_weight = min(edge_weights)
#             weight_range = max_weight - min_weight if max_weight > min_weight else 1.0
#         else:
#             weight_range = 1.0

#         def heuristic(u, v):
#             u_pos = np.array([G.nodes[u]['x'], G.nodes[u]['y']])
#             v_pos = np.array([G.nodes[v]['x'], G.nodes[v]['y']])
#             u_score = G.nodes[u].get('score', 0)
#             v_score = G.nodes[v].get('score', 0)
#             motion_cost = np.linalg.norm(u_pos - v_pos) / weight_range
#             return -beta * min(u_score, v_score) + alpha * motion_cost

#         try:
#             path = nx.astar_path(
#                 G,
#                 start_id,
#                 goal_id,
#                 heuristic=heuristic,
#                 weight='weight'
#             )
#             return path
#         except nx.NetworkXNoPath:
#             print(f"No path found from {start_id} to {goal_id}")
#             return None




# class Solution:
#     def __init__(self, path):
#         self.path = path
#         self.trajectory = None
    
#     def set_path(self, path):
#         self.path = path
#         self.trajectory = None

#     def generate_trajectory(self, graph, num_points=1000, degree=3):

#         if not self.path or len(self.path) < 2:
#             raise ValueError("Path must contain at least 2 nodes")
    
#         x_coords = nx.get_node_attributes(graph, 'x')
#         y_coords = nx.get_node_attributes(graph, 'y')
#         pan = nx.get_node_attributes(graph, 'pan')
#         tilt = nx.get_node_attributes(graph, 'tilt')

        

#         if not x_coords or not y_coords or not pan or not tilt:
#             raise ValueError("Graph nodes must have all attributes")
        
#         # Vectorized coordinate extraction using list comprehension + array conversion
#         try:
#             waypoints = np.array([[x_coords[node_id], y_coords[node_id], pan[node_id], tilt[node_id]] for node_id in self.path])
#         except KeyError as e:
#             raise ValueError(f"Node {e} not found in graph")

#         # Handle edge case with two waypoints
#         if len(waypoints) == 2:
#             # Linear interpolation for 2 points
#             t = np.linspace(0, 1, num_points)[:, np.newaxis]  
#             trajectory = waypoints[0] + t * (waypoints[1] - waypoints[0]) 
#             self.trajectory = trajectory
#             return trajectory

#         t = np.linspace(0, 1, len(waypoints))
    
#         spline_x = make_interp_spline(t, waypoints[:, 0], k=degree)
#         spline_y = make_interp_spline(t, waypoints[:, 1], k=degree)
#         spline_pan = make_interp_spline(t, waypoints[:, 2], k=degree)
#         spline_tilt = make_interp_spline(t, waypoints[:, 3], k=degree)

#         # Get analytical derivatives directly from the splines
#         spline_x_dot = spline_x.derivative()
#         spline_y_dot = spline_y.derivative()


#         t_new = np.linspace(0, 1, num_points)
#         theta = np.arctan2(spline_y_dot(t_new), spline_x_dot(t_new))

#         trajectory = np.column_stack([spline_x(t_new), spline_y(t_new), theta, spline_pan(t_new), spline_tilt(t_new)])
#         #velocity = np.column_stack([spline_x.derivative()(t_new), spline_y.derivative()(t_new)])
        
#         return trajectory #, velocity


#     def print_path(self, graph):
#         if not self.path:
#             print("No path to print")
#             return
        
#         for node_id in self.path:
#             node_data = graph.nodes[node_id]
#             print(f"Node {node_id}: x={node_data['x']:.2f}, y={node_data['y']:.2f}, theta={node_data['theta']:.2f}, pan={node_data['pan']:.2f}, tilt={node_data['tilt']:.2f}, score={node_data.get('score', 0):.4f}")