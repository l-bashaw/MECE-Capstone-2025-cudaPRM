import torch
import rsplan
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

        self.tensors = {
            'nodes': None, 
            'node_validity': None, 
            'neighbors': None, 
            'edges': None, 
            'edge_validity': None
        }
        

    def build_prm(self, seed):
        
        bounds = self.env['bounds']
        circles = self.env['circles']
        rectangles = self.env['rectangles']
        object_pose_world = self.env['object_pose']
        obj_label = self.env['object_label'] 
            
        nodes, node_validity, neighbors, edges, edge_validity = cuPRM.build_prm(
            circles, rectangles, bounds, seed
        )  # TODO: add obj pose

        self.tensors['nodes'] = nodes
        self.tensors['node_validity'] = node_validity
        self.tensors['neighbors'] = neighbors
        self.tensors['edges'] = edges
        self.tensors['edge_validity'] = edge_validity

        neighbors = torch.where(edge_validity, neighbors, -1)  # Replace invalid neighbors with -1
        diffs = self.project_nodesFIX(nodes, object_pose_world)    # Project nodes and return diffs (difference between camera pose and object pose at each node)
        diffs = torch.cat((diffs, obj_label.unsqueeze(0).repeat(nodes.shape[0], 1)), dim=1)
        p_scores = self.model(diffs)
        # print(f"Max score: {p_scores.max().item():.4f}, Min score: {p_scores.min().item():.4f}, Mean score: {p_scores.mean().item():.4f}")
        self.graph = self.tensors_to_networkx(nodes, neighbors, p_scores)
        return 

    # Start and goal are numpy arrays of shape (5,)
    # Add them to the networkx graph as new nodes, connect to nearest neighbors, and update the graph
    # Member of PSPRM
    def addStartAndGoal(self, start, goal):  # tensors of shape 5
    
        # nodes = self.tensors['nodes']
        # node_validity = self.tensors['node_validity']
        # neighbors = self.tensors['neighbors']
        # edges = self.tensors['edges']
        # edge_validity = self.tensors['edge_validity']
        # circles = self.env['circles']
        # rectangles = self.env['rectangles']

        # start_state, start_neighbors, start_edges, start_valid_edges, goal_state, goal_neighbors, goal_edges, goal_valid_edges = cuPRM.addStartAndGoal_(
        #     nodes, node_validity, neighbors, edges, edge_validity,
        #     start, goal,
        #     circles, rectangles
        # )
        # print(f"\nActual goal: {goal}\n")
        # print(f"\nReturned goal tensor: {goal_state}\n")
        # print(f"Actual start: {start}\n")
        # print(f"\nReturned start tensor: {start_state}\n")
        # # Add start and goal to the netx graph 

        start_id = max(self.graph.nodes) + 1
        goal_id = start_id + 1

        start_attrs = {
            'x': start[0],
            'y': start[1],
            'theta': start[2],
            'pan': 0,
            'tilt': 0,
            'score': 0.0  # Placeholders, can be updated if needed
        }
        goal_attrs = {
            'x': goal[0],
            'y': goal[1],
            'theta': goal[2],
            'pan': 0,
            'tilt': 0,
            'score': 0.0  # Placeholders, can be updated if needed
        }

        self.graph.add_node(start_id, **start_attrs)
        self.graph.add_node(goal_id, **goal_attrs)
        # print(start_id, goal_id)

        # Find nearest neighbors in the existing graph, manually
        all_nodes = np.array([[data['x'], data['y'], data['theta']] for _, data in self.graph.nodes(data=True)])
        start_dists = np.linalg.norm(all_nodes - start[:3], axis=1)
        goal_dists = np.linalg.norm(all_nodes - goal[:3], axis=1)
        start_neighbor_ids = np.argsort(start_dists)[:10]  # Get indices of 10 nearest neighbors
        goal_neighbor_ids = np.argsort(goal_dists)[:10]
        # print("Start neighbor IDs:", start_neighbor_ids)
        # print("Goal neighbor IDs:", goal_neighbor_ids)
        
        # Ignore collisions for now
        for nid in start_neighbor_ids:
            src = np.array([start[0], start[1], start[2]])
            tgt = np.array([self.graph.nodes[nid]['x'], self.graph.nodes[nid]['y'], self.graph.nodes[nid]['theta']])
            weight = np.linalg.norm(src - tgt)
            self.graph.add_edge(start_id, nid, weight=weight)
        for nid in goal_neighbor_ids:
            src = np.array([goal[0], goal[1], goal[2]])
            tgt = np.array([self.graph.nodes[nid]['x'], self.graph.nodes[nid]['y'], self.graph.nodes[nid]['theta']])
            weight = np.linalg.norm(src - tgt)
            self.graph.add_edge(goal_id, nid, weight=weight)
        return start_id, goal_id

        # # Connect edges for start
        # for i, valid in enumerate(start_valid_edges):
        #     if valid:
        #         nid = start_neighbors[i].item()
        #         src = np.array([start_state[0].item(), start_state[1].item(), start_state[2].item()])
        #         tgt = np.array([self.graph.nodes[nid]['x'], self.graph.nodes[nid]['y'], self.graph.nodes[nid]['theta']])
        #         weight = np.linalg.norm(src - tgt)
        #         self.graph.add_edge(start_id, nid, weight=weight)
            
        # # Connect edges for goal
        # for i, valid in enumerate(goal_valid_edges):
        #     if valid:
        #         nid = goal_neighbors[i].item()
        #         src = np.array([goal_state[0].item(), goal_state[1].item(), start_state[2].item()])
        #         tgt = np.array([self.graph.nodes[nid]['x'], self.graph.nodes[nid]['y'], self.graph.nodes[nid]['theta']])
        #         weight = np.linalg.norm(src - tgt)
        #         self.graph.add_edge(goal_id, nid, weight=weight)

        # return {'start': [start_state, start_neighbors, start_edges, start_valid_edges],
        #         'goal': [goal_state, goal_neighbors, goal_edges, goal_valid_edges]}

        


    def project_nodesFIX(self, nodes, object_pose_world):
        """
        Calculate pan and tilt angles for robot states to look at target object,
        update camera transforms with pan/tilt, and compute diffs.
    
        Args:
            nodes: Tensor [B, 5] = (x, y, theta, pan, tilt)
            object_pose_world: Tensor [3] = object xyz in world
    
        Returns:
            diffs: [B, 7] = (object xyz in cam frame, cam orientation quat)
        """
        device = nodes.device
        dtype = nodes.dtype
        B = nodes.shape[0]
    
        # Camera mount config (robot -> camera base)
        cam_pos_robot = torch.tensor([0.0450, 0.0523, 1.2607], dtype=dtype, device=device)
        cam_rot_robot = torch.tensor(
            [[0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]], dtype=dtype, device=device
        )
    
        # Extract robot states
        robot_states = nodes[:, :3]  # [x, y, theta]
    
        # Object position homogeneous
        object_pos_world_h = torch.cat([object_pose_world[:3], torch.ones(1, dtype=dtype, device=device)])
        object_pos_world_h = object_pos_world_h.unsqueeze(0).expand(B, -1)
    
        # Camera base in robot frame
        T_robot_camera_base = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
        T_robot_camera_base[:, :3, :3] = cam_rot_robot.unsqueeze(0).repeat(B, 1, 1)
        T_robot_camera_base[:, :3, 3] = cam_pos_robot.unsqueeze(0).repeat(B, 1)
    
        # Robot -> world
        cos_theta = torch.cos(robot_states[:, 2])
        sin_theta = torch.sin(robot_states[:, 2])
        T_world_robot = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
        T_world_robot[:, 0, 0] = cos_theta
        T_world_robot[:, 0, 1] = -sin_theta
        T_world_robot[:, 1, 0] = sin_theta
        T_world_robot[:, 1, 1] = cos_theta
        T_world_robot[:, 0, 3] = robot_states[:, 0]
        T_world_robot[:, 1, 3] = robot_states[:, 1]
    
        # Camera base in world
        T_world_cam_base = torch.bmm(T_world_robot, T_robot_camera_base)
    
        # Object in camera base frame (before pan/tilt)
        T_cam_base_world = torch.inverse(T_world_cam_base)
        obj_in_cam_base = torch.bmm(T_cam_base_world, object_pos_world_h.unsqueeze(-1)).squeeze(-1)[:, :3]
        dx, dy, dz = obj_in_cam_base[:, 0], obj_in_cam_base[:, 1], obj_in_cam_base[:, 2]
    
        # Pan/tilt calculation
        pan = torch.atan2(dx, dz)
        tilt = torch.atan2(dy, torch.sqrt(dx**2 + dz**2))
        pan = torch.clamp(pan, -3.9, 1.5)
        tilt = torch.clamp(tilt, -1.53, 0.79)
    
        # Save into nodes
        nodes[:, 3] = pan
        nodes[:, 4] = tilt
    
        # --- Apply pan/tilt to camera transform ---
        def make_pan_tilt_transform(pan, tilt, dtype, device):
            B = pan.shape[0]
            T = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)

            # Pan: rotation around Y
            cos_p, sin_p = torch.cos(pan), torch.sin(pan)
            R_pan = torch.stack([
                torch.stack([ cos_p, torch.zeros_like(pan), sin_p], dim=1),
                torch.stack([ torch.zeros_like(pan), torch.ones_like(pan), torch.zeros_like(pan)], dim=1),
                torch.stack([-sin_p, torch.zeros_like(pan), cos_p], dim=1)
            ], dim=1)

            # Tilt: rotation around X
            cos_t, sin_t = torch.cos(tilt), torch.sin(tilt)
            R_tilt = torch.stack([
                torch.stack([ torch.ones_like(tilt), torch.zeros_like(tilt), torch.zeros_like(tilt)], dim=1),
                torch.stack([ torch.zeros_like(tilt), cos_t, -sin_t], dim=1),
                torch.stack([ torch.zeros_like(tilt), sin_t,  cos_t], dim=1)
            ], dim=1)

            R = torch.bmm(R_tilt, R_pan)  # tilt ∘ pan
            T[:, :3, :3] = R
            return T

        T_cam_base_cam = make_pan_tilt_transform(pan, tilt, dtype, device)
        T_world_cam = torch.bmm(T_world_cam_base, T_cam_base_cam)
        T_cam_world = torch.inverse(T_world_cam)
    
        # Object in true camera frame
        obj_in_cam = torch.bmm(T_cam_world, object_pos_world_h.unsqueeze(-1)).squeeze(-1)[:, :3]
    
        # Camera orientation as quaternion
        R_cam_world = T_cam_world[:, :3, :3]
        obj_quat_in_cam = pk.matrix_to_quaternion(R_cam_world)
    
        # Final diffs
        diffs = torch.cat([-obj_in_cam, obj_quat_in_cam], dim=1)  # [B, 7]
        return diffs

   
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
        # Pan bounds are [-3.9, 1.5]
        # Tilt bounds are [-1.53, 0.79]
        pan = torch.clamp(pan, -3.9, 1.5)
        tilt = torch.clamp(tilt, -1.53, 0.79)
       
        # Modify nodes in place with calculated pan and tilt
        nodes[:, 3] = pan  
        nodes[:, 4] = tilt  
        
        # Compute the relative orientation as a quaternion
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
        # print(f"Source positions shape: {src_positions.shape}, Target positions shape: {tgt_positions.shape}")
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
        # edge_weights = [data['weight'] for u, v, data in G.edges(data=True)]
        # if edge_weights:
        #     max_weight = max(edge_weights)
        #     min_weight = min(edge_weights)
        #     weight_range = max_weight - min_weight if max_weight > min_weight else 1.0
        # else:
        #     weight_range = 1.0

        max_score = max(nx.get_node_attributes(G, 'score').values())
        min_score = min(nx.get_node_attributes(G, 'score').values())
        score_range = max_score - min_score if max_score > min_score else 1.0
        
        goal_x = G.nodes[goal_id]['x']
        goal_y = G.nodes[goal_id]['y']
        goal_theta = G.nodes[goal_id]['theta']

        def heuristic(u, v):
            u_score = (G.nodes[u]['score'] - min_score) / score_range
            v_score = (G.nodes[v]['score'] - min_score) / score_range

            # Set motion cost to the edge weight
            if G.has_edge(u, v):
                # motion_cost = G[u][v]['weight'] / weight_range  
                ux = G.nodes[u]['x']
                uy = G.nodes[u]['y']
                ut = G.nodes[u]['theta']
                motion_cost = ((ux - goal_x) ** 2 + (uy - goal_y) ** 2 ) ** 0.5
            else:
                motion_cost = float('inf')  # No direct edge, set to infinity

            # Use the average of the scores instead of the minimum
            score_term = (u_score + v_score) / 2
            val = -alpha * score_term + beta * motion_cost

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

            # total_cost = 0
            # for i in range(len(path) - 1):
            #     u, v = path[i], path[i + 1]
            #     edge_cost = G[u][v]['weight']
            #     heuristic_cost = heuristic(u, v)
            #     total_cost += edge_cost
                # print(f"\nEdge ({u} -> {v}): edge_cost={edge_cost:.4f}, heuristic_cost={heuristic_cost:.4f}, total_cost={total_cost:.4f}")
            
            # print(f"\nFinal total cost: {total_cost:.4f}")

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

    def simplify(self, prm, env, max_skip_dist):
        if self.path is None or len(self.path) < 2:
            raise ValueError("Path must contain at least 2 nodes to simplify")

        circles = env['circles'].cpu().numpy()
        rectangles = env['rectangles'].cpu().numpy()
        graph = prm.graph

        x_coords = nx.get_node_attributes(graph, 'x')
        y_coords = nx.get_node_attributes(graph, 'y')
        theta_coords = nx.get_node_attributes(graph, 'theta')
        
        def in_collision(p1, p2, num_samples=10):
            t = np.linspace(0, 1, num_samples)[:, np.newaxis]
            samples = p1 + t * (p2 - p1)

            if circles.size > 0:
                centers = circles[:, :2]
                radii = circles[:, 2]
                dists = np.linalg.norm(
                    samples[:, np.newaxis, :2] - centers[np.newaxis, :, :],
                    axis=2
                )
                if np.any(dists <= radii):
                    return True

            if rectangles.size > 0:
                x, y, w, h = rectangles.T
                in_x_bounds = (samples[:, np.newaxis, 0] >= x) & (samples[:, np.newaxis, 0] <= x + w)
                in_y_bounds = (samples[:, np.newaxis, 1] >= y) & (samples[:, np.newaxis, 1] <= y + h)
                if np.any(in_x_bounds & in_y_bounds):
                    return True

            return False

        try:
            waypoints = np.array([
                [x_coords[node_id], y_coords[node_id], theta_coords[node_id]]
                for node_id in self.path
            ])
        except KeyError as e:
            raise ValueError(f"Node {e} not found in graph")

        simplified_path = [self.path[0]]
        last_kept_index = 0
        for i in range(2, len(self.path)):
            p1 = waypoints[last_kept_index]
            p2 = waypoints[i]

            # Euclidean distance in XY
            dist = np.linalg.norm(p2[:3] - p1[:3])
            
            if (in_collision(p1, p2) or dist > max_skip_dist): 
                
                simplified_path.append(self.path[i - 1])
                last_kept_index = i - 1
            else:
                continue

        simplified_path.append(self.path[-1])
        self.path = simplified_path
        
    def project_trajectory(self, object_pose_world: torch.Tensor) -> np.ndarray:
        """
        Update nodes[:,3] and nodes[:,4] with pan and tilt angles needed
        for the robot camera to point at the object.
        
        Args:
            nodes: np.ndarray of shape (N,5) with columns [x, y, theta, pan, tilt].
            object_pose_world: torch.Tensor of shape (4,) or (3,) representing object pose in world.
        
        Returns:
            nodes: same np.ndarray, with pan/tilt columns updated in place.
        """
        nodes = self.trajectory
        # Ensure numpy -> torch conversion
        device = object_pose_world.device
        dtype = object_pose_world.dtype
        N = nodes.shape[0]

        nodes_t = torch.from_numpy(nodes[:, :3]).to(device=device, dtype=dtype)  # [N,3]

        # Camera extrinsics in robot frame
        cam_pos_robot = torch.tensor([0.0450, 0.0523, 1.2607], dtype=dtype, device=device)
        cam_rot_robot = torch.tensor(
            [[0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]], dtype=dtype, device=device
        )

        # Object position homogeneous
        obj_pos_h = torch.cat([object_pose_world[:3], torch.ones(1, device=device, dtype=dtype)])
        obj_pos_h = obj_pos_h.unsqueeze(0).expand(N, -1)

        # Camera base transform
        T_robot_cam = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(N, 1, 1)
        T_robot_cam[:, :3, :3] = cam_rot_robot.unsqueeze(0).repeat(N, 1, 1)
        T_robot_cam[:, :3, 3] = cam_pos_robot.unsqueeze(0).repeat(N, 1)

        # Robot base-to-world transforms
        cos_t = torch.cos(nodes_t[:, 2])
        sin_t = torch.sin(nodes_t[:, 2])
        T_world_robot = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(N, 1, 1)
        T_world_robot[:, 0, 0] = cos_t
        T_world_robot[:, 0, 1] = -sin_t
        T_world_robot[:, 1, 0] = sin_t
        T_world_robot[:, 1, 1] = cos_t
        T_world_robot[:, 0, 3] = nodes_t[:, 0]
        T_world_robot[:, 1, 3] = nodes_t[:, 1]

        # World -> camera transforms
        T_world_cam = torch.bmm(T_world_robot, T_robot_cam)
        T_cam_world = torch.inverse(T_world_cam)

        # Transform object into camera frame
        obj_in_cam = torch.bmm(T_cam_world, obj_pos_h.unsqueeze(-1)).squeeze(-1)[:, :3]
        dx, dy, dz = obj_in_cam[:, 0], obj_in_cam[:, 1], obj_in_cam[:, 2]

        # Pan/tilt computation
        pan = torch.atan2(dx, dz)
        tilt = torch.atan2(dy, torch.sqrt(dx**2 + dz**2))

        # Clamp to limits
        pan = torch.clamp(pan, -3.9, 1.5)
        tilt = torch.clamp(tilt, -1.53, 0.79)

        # Write back to numpy
        nodes[:, 3] = pan.cpu().numpy()
        nodes[:, 4] = tilt.cpu().numpy()
        self.trajectory = nodes
        return nodes
    
    def generate_trajectory_rsplan(self, prm:PSPRM, turning_radius=1.0):
        
        graph = prm.graph
        x_coords = nx.get_node_attributes(graph, 'x')
        y_coords = nx.get_node_attributes(graph, 'y')
        theta_coords = nx.get_node_attributes(graph, 'theta')
        
        waypoints = np.array([[x_coords[node_id], y_coords[node_id], theta_coords[node_id]] for node_id in self.path])
        traj = []
        for i in range(len(waypoints) - 1):
            rspath = rsplan.path((waypoints[i,0], waypoints[i,1], waypoints[i,2]),
                                 (waypoints[i+1,0], waypoints[i+1,1], waypoints[i+1,2]),
                                    turning_radius,
                                0,
                                0.05)
            xs, ys, ts = rspath.coordinates_tuple()
            segment = np.vstack((xs, ys, ts)).T
            traj.append(segment)
        # Concatenate all segments into a single trajectory
        traj = np.vstack(traj)
        pan_tilt = np.zeros((traj.shape[0], 2))
        traj = np.hstack((traj, pan_tilt))
        self.trajectory = traj
        return traj

    def print_path(self, graph):
        if not self.path:
            print("No path to print")
            return
        waypoints = []
        for node_id in self.path:
            node_data = graph.nodes[node_id]
            waypoints.append((node_data['x'], node_data['y'], node_data['theta'], node_data['pan'], node_data['tilt'], node_data['score']))

        # save waypoints to a text file
        with open("waypoints.txt", "w") as f:
            for wp in waypoints:
                f.write(f"{wp[0]} {wp[1]:.4f} {wp[2]:.4f} {wp[3]:.4f} {wp[4]:.4f} {wp[5]:.4f}\n")



# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------

# def generate_trajectory2(self, prm:PSPRM):
#         """
#         Generate a trajectory using the interpolated values of x, y, theta, pan, and tilt
#         stored in prm.tensors.edges for the nodes in the A* path.
#         """
#         if not self.path or len(self.path) < 2:
#             raise ValueError("Path must contain at least 2 nodes")

#         edges = prm.tensors['edges']  # Shape: [NUM_STATES, K, INTERP_STEPS, DIM]
#         neighbors = prm.tensors['neighbors']  # Shape: [NUM_STATES, K]

#         trajectory = []

#         for i in range(len(self.path) - 1):
#             current_node = self.path[i]
#             next_node = self.path[i + 1]

#             # Find the neighbor index of the next node in the current node's neighbors
#             neighbor_indices = neighbors[current_node].cpu().numpy()
#             if next_node not in neighbor_indices:
#                 # get the edge from the other direction, the next node has the neighbor relationship, not the current node
#                 current_node, next_node = next_node, current_node
#                 neighbor_indices = neighbors[current_node].cpu().numpy()
#                 neighbor_idx = list(neighbor_indices).index(next_node)
#                 edge_data = edges[current_node, neighbor_idx].cpu().numpy()
#                 trajectory.extend(edge_data)
#                 continue

#             neighbor_idx = list(neighbor_indices).index(next_node)

#             # Extract the interpolated edge data
#             edge_data = edges[current_node, neighbor_idx].cpu().numpy()  # Shape: [INTERP_STEPS, DIM]

#             # Append the interpolated points to the trajectory
#             trajectory.extend(edge_data)

#         self.trajectory = np.array(trajectory)  # Convert to numpy array for visualization
#         return self.trajectory

#     def generate_trajectory_reeds_shepp(self, prm:PSPRM, turning_radius=1.0):
#         graph = prm.graph  # networkx graph

#         x_coords = nx.get_node_attributes(graph, 'x')
#         y_coords = nx.get_node_attributes(graph, 'y')
#         theta_coords = nx.get_node_attributes(graph, 'theta')
        
#         try:
#             waypoints = np.array([[x_coords[node_id], y_coords[node_id], theta_coords[node_id]] for node_id in self.path])
#         except KeyError as e:
#             raise ValueError(f"Node {e} not found in graph")

#          # Create trajectory generator
#         traj_gen = ReedsSheppTrajectory(turning_radius=turning_radius)
        
#         # Generate trajectory
#         trajectory, lengths = traj_gen.connect_waypoints(waypoints, num_samples=30)
#         # Add dummy pan and tilt values (0.0)
#         pan_tilt = np.zeros((trajectory.shape[0], 2))
#         trajectory = np.hstack((trajectory, pan_tilt))
#         self.trajectory = trajectory
#         return trajectory
# import numpy as np
# import matplotlib.pyplot as plt
# from typing import List, Tuple, Optional, Union
# from enum import Enum

# class SegmentType(Enum):
#     LEFT = 0
#     RIGHT = 1
#     STRAIGHT = 2
#     NOP = 3

# class ReedsSheppPath:
    
#     PATH_TYPES = [
#         [SegmentType.LEFT, SegmentType.RIGHT, SegmentType.LEFT, SegmentType.NOP, SegmentType.NOP],         # 0
#         [SegmentType.RIGHT, SegmentType.LEFT, SegmentType.RIGHT, SegmentType.NOP, SegmentType.NOP],        # 1
#         [SegmentType.LEFT, SegmentType.RIGHT, SegmentType.LEFT, SegmentType.RIGHT, SegmentType.NOP],       # 2
#         [SegmentType.RIGHT, SegmentType.LEFT, SegmentType.RIGHT, SegmentType.LEFT, SegmentType.NOP],       # 3
#         [SegmentType.LEFT, SegmentType.RIGHT, SegmentType.STRAIGHT, SegmentType.LEFT, SegmentType.NOP],    # 4
#         [SegmentType.RIGHT, SegmentType.LEFT, SegmentType.STRAIGHT, SegmentType.RIGHT, SegmentType.NOP],   # 5
#         [SegmentType.LEFT, SegmentType.STRAIGHT, SegmentType.RIGHT, SegmentType.LEFT, SegmentType.NOP],    # 6
#         [SegmentType.RIGHT, SegmentType.STRAIGHT, SegmentType.LEFT, SegmentType.RIGHT, SegmentType.NOP],   # 7
#         [SegmentType.LEFT, SegmentType.RIGHT, SegmentType.STRAIGHT, SegmentType.RIGHT, SegmentType.NOP],   # 8
#         [SegmentType.RIGHT, SegmentType.LEFT, SegmentType.STRAIGHT, SegmentType.LEFT, SegmentType.NOP],    # 9
#         [SegmentType.RIGHT, SegmentType.STRAIGHT, SegmentType.RIGHT, SegmentType.LEFT, SegmentType.NOP],   # 10
#         [SegmentType.LEFT, SegmentType.STRAIGHT, SegmentType.LEFT, SegmentType.RIGHT, SegmentType.NOP],    # 11
#         [SegmentType.LEFT, SegmentType.STRAIGHT, SegmentType.RIGHT, SegmentType.NOP, SegmentType.NOP],     # 12
#         [SegmentType.RIGHT, SegmentType.STRAIGHT, SegmentType.LEFT, SegmentType.NOP, SegmentType.NOP],     # 13
#         [SegmentType.LEFT, SegmentType.STRAIGHT, SegmentType.LEFT, SegmentType.NOP, SegmentType.NOP],      # 14
#         [SegmentType.RIGHT, SegmentType.STRAIGHT, SegmentType.RIGHT, SegmentType.NOP, SegmentType.NOP],    # 15
#         [SegmentType.LEFT, SegmentType.RIGHT, SegmentType.STRAIGHT, SegmentType.LEFT, SegmentType.RIGHT],  # 16
#         [SegmentType.RIGHT, SegmentType.LEFT, SegmentType.STRAIGHT, SegmentType.RIGHT, SegmentType.LEFT]   # 17
#     ]
    
#     def __init__(self, path_type_index=-1, lengths=None, total_length=999.0):
#         self.path_type_index = path_type_index
#         self.lengths = lengths if lengths is not None else [0.0, 0.0, 0.0, 0.0, 0.0]
#         self.total_length = total_length
        
#         if path_type_index >= 0 and lengths is not None:
#             self.total_length = sum(abs(l) for l in lengths)

# def mod2pi(x):
#     """Normalize angle to [-pi, pi]."""
#     return np.arctan2(np.sin(x), np.cos(x))

# def polar(x, y):
#     """Convert Cartesian to polar coordinates."""
#     r = np.sqrt(x*x + y*y)
#     theta = np.arctan2(y, x)
#     return r, theta

# def tau_omega(u, v, xi, eta, phi):
#     """Helper function for Reeds-Shepp path computation."""
#     delta = mod2pi(u - v)
#     A = np.sin(u) - np.sin(delta)
#     B = np.cos(u) - np.cos(delta) - 1.0
#     t1 = np.arctan2(eta * A - xi * B, xi * A + eta * B)
#     t2 = 2.0 * (np.cos(delta) - np.cos(v) - np.cos(u)) + 3.0
#     tau = mod2pi(t1 + np.pi) if t2 < 0 else mod2pi(t1)
#     omega = mod2pi(tau - u + v - phi)
#     return tau, omega

# # Path computation functions
# def LpSpLp(x, y, phi):
#     """Left-Straight-Left path computation."""
#     r, t = polar(x - np.sin(phi), y - 1.0 + np.cos(phi))
#     u = r
#     if t >= -1e-6:
#         v = mod2pi(phi - t)
#         if v >= -1e-6:
#             return True, t, u, v
#     return False, 0, 0, 0

# def LpSpRp(x, y, phi):
#     """Left-Straight-Right path computation."""
#     r, t1 = polar(x + np.sin(phi), y - 1.0 - np.cos(phi))
#     u1_squared = r * r
#     if u1_squared >= 4.0:
#         u = np.sqrt(u1_squared - 4.0)
#         theta = np.arctan2(2.0, u)
#         t = mod2pi(t1 + theta)
#         v = mod2pi(t - phi)
#         return t >= -1e-6 and v >= -1e-6, t, u, v
#     return False, 0, 0, 0

# def LpRmL(x, y, phi):
#     """Left-Right-Left path computation."""
#     xi = x - np.sin(phi)
#     eta = y - 1.0 + np.cos(phi)
#     r, theta = polar(xi, eta)
#     if r <= 4.0:
#         u = -2.0 * np.arcsin(0.25 * r)
#         t = mod2pi(theta + 0.5 * u + np.pi)
#         v = mod2pi(phi - t + u)
#         return t >= -1e-6 and u <= 1e-6, t, u, v
#     return False, 0, 0, 0

# def LpRupLumRm(x, y, phi):
#     """Path computation for CCCC family."""
#     xi = x + np.sin(phi)
#     eta = y - 1.0 - np.cos(phi)
#     rho = 0.25 * (2.0 + np.sqrt(xi*xi + eta*eta))
#     if rho <= 1.0:
#         u = np.arccos(rho)
#         t, v = tau_omega(u, -u, xi, eta, phi)
#         return t >= -1e-6 and v <= 1e-6, t, u, v
#     return False, 0, 0, 0

# def LpRumLumRp(x, y, phi):
#     """Path computation for CCCC family."""
#     xi = x + np.sin(phi)
#     eta = y - 1.0 - np.cos(phi)
#     rho = (20.0 - xi*xi - eta*eta) / 16.0
#     if 0.0 <= rho <= 1.0:
#         u = -np.arccos(rho)
#         if u >= -np.pi/2:
#             t, v = tau_omega(u, u, xi, eta, phi)
#             return t >= -1e-6 and v >= -1e-6, t, u, v
#     return False, 0, 0, 0

# def LpRmSmLm(x, y, phi):
#     """Path computation for CCSC family."""
#     xi = x - np.sin(phi)
#     eta = y - 1.0 + np.cos(phi)
#     rho, theta = polar(xi, eta)
#     if rho >= 2.0:
#         r = np.sqrt(rho*rho - 4.0)
#         u = 2.0 - r
#         t = mod2pi(theta + np.arctan2(r, -2.0))
#         v = mod2pi(phi - np.pi/2 - t)
#         return t >= -1e-6 and u <= 1e-6 and v <= 1e-6, t, u, v
#     return False, 0, 0, 0

# def LpRmSmRm(x, y, phi):
#     """Path computation for CCSC family."""
#     xi = x + np.sin(phi)
#     eta = y - 1.0 - np.cos(phi)
#     rho, theta = polar(-eta, xi)
#     if rho >= 2.0:
#         t = theta
#         u = 2.0 - rho
#         v = mod2pi(t + np.pi/2 - phi)
#         return t >= -1e-6 and u <= 1e-6 and v <= 1e-6, t, u, v
#     return False, 0, 0, 0

# def LpRmSLmRp(x, y, phi):
#     """Path computation for CCSCC family."""
#     xi = x + np.sin(phi)
#     eta = y - 1.0 - np.cos(phi)
#     rho, theta = polar(xi, eta)
#     if rho >= 2.0:
#         u = 4.0 - np.sqrt(rho*rho - 4.0)
#         if u <= 1e-6:
#             t = mod2pi(np.arctan2((4.0 - u)*xi - 2.0*eta, -2.0*xi + (u - 4.0)*eta))
#             v = mod2pi(t - phi)
#             return t >= -1e-6 and v >= -1e-6, t, u, v
#     return False, 0, 0, 0

# def CSC(x, y, phi, best_path):
#     """Curve-Straight-Curve path family."""
#     paths_to_try = [
#         (lambda: LpSpLp(x, y, phi), 14, 1, 1, 1),
#         (lambda: LpSpLp(-x, y, -phi), 14, -1, -1, -1),
#         (lambda: LpSpLp(x, -y, -phi), 15, 1, 1, 1),
#         (lambda: LpSpLp(-x, -y, phi), 15, -1, -1, -1),
#         (lambda: LpSpRp(x, y, phi), 12, 1, 1, 1),
#         (lambda: LpSpRp(-x, y, -phi), 12, -1, -1, -1),
#         (lambda: LpSpRp(x, -y, -phi), 13, 1, 1, 1),
#         (lambda: LpSpRp(-x, -y, phi), 13, -1, -1, -1),
#     ]
    
#     for path_func, path_type, sign_t, sign_u, sign_v in paths_to_try:
#         valid, t, u, v = path_func()
#         if valid:
#             length = abs(t) + abs(u) + abs(v)
#             if length < best_path.total_length:
#                 best_path.path_type_index = path_type
#                 best_path.lengths = [sign_t*t, sign_u*u, sign_v*v, 0, 0]
#                 best_path.total_length = length

# def CCC(x, y, phi, best_path):
#     """Curve-Curve-Curve path family."""
#     # Forward paths
#     paths_to_try = [
#         (lambda: LpRmL(x, y, phi), 0, 1, 1, 1),
#         (lambda: LpRmL(-x, y, -phi), 0, -1, -1, -1),
#         (lambda: LpRmL(x, -y, -phi), 1, 1, 1, 1),
#         (lambda: LpRmL(-x, -y, phi), 1, -1, -1, -1),
#     ]
    
#     for path_func, path_type, sign_t, sign_u, sign_v in paths_to_try:
#         valid, t, u, v = path_func()
#         if valid:
#             length = abs(t) + abs(u) + abs(v)
#             if length < best_path.total_length:
#                 best_path.path_type_index = path_type
#                 best_path.lengths = [sign_t*t, sign_u*u, sign_v*v, 0, 0]
#                 best_path.total_length = length
    
#     # Backward paths
#     xb = x * np.cos(phi) + y * np.sin(phi)
#     yb = x * np.sin(phi) - y * np.cos(phi)
    
#     backward_paths = [
#         (lambda: LpRmL(xb, yb, phi), 0, 1, 1, 1),
#         (lambda: LpRmL(-xb, yb, -phi), 0, -1, -1, -1),
#         (lambda: LpRmL(xb, -yb, -phi), 1, 1, 1, 1),
#         (lambda: LpRmL(-xb, -yb, phi), 1, -1, -1, -1),
#     ]
    
#     for path_func, path_type, sign_t, sign_u, sign_v in backward_paths:
#         valid, t, u, v = path_func()
#         if valid:
#             length = abs(t) + abs(u) + abs(v)
#             if length < best_path.total_length:
#                 best_path.path_type_index = path_type
#                 best_path.lengths = [sign_v*v, sign_u*u, sign_t*t, 0, 0]  # Reversed order
#                 best_path.total_length = length

# def CCCC(x, y, phi, best_path):
#     """Curve-Curve-Curve-Curve path family."""
#     paths_to_try = [
#         (lambda: LpRupLumRm(x, y, phi), 2, 1, 1, 1, 1),
#         (lambda: LpRupLumRm(-x, y, -phi), 2, -1, -1, -1, -1),
#         (lambda: LpRupLumRm(x, -y, -phi), 3, 1, 1, 1, 1),
#         (lambda: LpRupLumRm(-x, -y, phi), 3, -1, -1, -1, -1),
#         (lambda: LpRumLumRp(x, y, phi), 2, 1, 1, 1, 1),
#         (lambda: LpRumLumRp(-x, y, -phi), 2, -1, -1, -1, -1),
#         (lambda: LpRumLumRp(x, -y, -phi), 3, 1, 1, 1, 1),
#         (lambda: LpRumLumRp(-x, -y, phi), 3, -1, -1, -1, -1),
#     ]
    
#     for path_func, path_type, sign_t, sign_u, sign_v, sign_w in paths_to_try:
#         valid, t, u, v = path_func()
#         if valid:
#             length = abs(t) + 2 * abs(u) + abs(v)
#             if length < best_path.total_length:
#                 best_path.path_type_index = path_type
#                 best_path.lengths = [sign_t*t, sign_u*u, -sign_u*u, sign_v*v, 0]
#                 best_path.total_length = length

# def CCSC(x, y, phi, best_path):
#     """Curve-Curve-Straight-Curve path family."""
#     paths_to_try = [
#         (lambda: LpRmSmLm(x, y, phi), 4, 1, 1, 1, 1),
#         (lambda: LpRmSmLm(-x, y, -phi), 4, -1, -1, -1, -1),
#         (lambda: LpRmSmLm(x, -y, -phi), 5, 1, 1, 1, 1),
#         (lambda: LpRmSmLm(-x, -y, phi), 5, -1, -1, -1, -1),
#         (lambda: LpRmSmRm(x, y, phi), 8, 1, 1, 1, 1),
#         (lambda: LpRmSmRm(-x, y, -phi), 8, -1, -1, -1, -1),
#         (lambda: LpRmSmRm(x, -y, -phi), 9, 1, 1, 1, 1),
#         (lambda: LpRmSmRm(-x, -y, phi), 9, -1, -1, -1, -1),
#     ]
    
#     for path_func, path_type, sign_t, sign_u, sign_v, sign_w in paths_to_try:
#         valid, t, u, v = path_func()
#         if valid:
#             length = abs(t) + abs(u) + abs(v) + np.pi/2
#             if length < best_path.total_length:
#                 best_path.path_type_index = path_type
#                 best_path.lengths = [sign_t*t, sign_u*np.pi/2, sign_v*u, sign_w*v, 0]
#                 best_path.total_length = length
    
#     # Try reverse path types (SCCC)
#     reverse_paths = [
#         (lambda: LpRmSmLm(x, y, phi), 6, 1, 1, 1, 1),
#         (lambda: LpRmSmLm(-x, y, -phi), 6, -1, -1, -1, -1),
#         (lambda: LpRmSmLm(x, -y, -phi), 7, 1, 1, 1, 1),
#         (lambda: LpRmSmLm(-x, -y, phi), 7, -1, -1, -1, -1),
#         (lambda: LpRmSmRm(x, y, phi), 10, 1, 1, 1, 1),
#         (lambda: LpRmSmRm(-x, y, -phi), 10, -1, -1, -1, -1),
#         (lambda: LpRmSmRm(x, -y, -phi), 11, 1, 1, 1, 1),
#         (lambda: LpRmSmRm(-x, -y, phi), 11, -1, -1, -1, -1),
#     ]
    
#     for path_func, path_type, sign_t, sign_u, sign_v, sign_w in reverse_paths:
#         valid, t, u, v = path_func()
#         if valid:
#             length = abs(t) + abs(u) + abs(v) + np.pi/2
#             if length < best_path.total_length:
#                 best_path.path_type_index = path_type
#                 best_path.lengths = [sign_t*v, sign_u*u, sign_v*np.pi/2, sign_w*t, 0]
#                 best_path.total_length = length

# def CCSCC(x, y, phi, best_path):
#     """Curve-Curve-Straight-Curve-Curve path family."""
#     paths_to_try = [
#         (lambda: LpRmSLmRp(x, y, phi), 16, 1, 1, 1, 1, 1),
#         (lambda: LpRmSLmRp(-x, y, -phi), 16, -1, -1, -1, -1, -1),
#         (lambda: LpRmSLmRp(x, -y, -phi), 17, 1, 1, 1, 1, 1),
#         (lambda: LpRmSLmRp(-x, -y, phi), 17, -1, -1, -1, -1, -1),
#     ]
    
#     for path_func, path_type, sign_t, sign_u, sign_v, sign_w, sign_x in paths_to_try:
#         valid, t, u, v = path_func()
#         if valid:
#             length = abs(t) + abs(u) + abs(v) + abs(u) + abs(t)  # |seg1| = |seg5|, |seg2| = |seg4|
#             if length < best_path.total_length:
#                 best_path.path_type_index = path_type
#                 best_path.lengths = [sign_t*t, sign_u*u, sign_v*v, sign_w*u, sign_x*t]
#                 best_path.total_length = length

# def compute_reeds_shepp_path(start, end, turning_radius):
#     """
#     Compute optimal Reeds-Shepp path between two poses.
    
#     Args:
#         start: (x, y, theta) starting pose or numpy array
#         end: (x, y, theta) ending pose or numpy array
#         turning_radius: minimum turning radius
        
#     Returns:
#         ReedsSheppPath object
#     """
#     # Convert to numpy arrays if needed
#     start = np.array(start) if not isinstance(start, np.ndarray) else start
#     end = np.array(end) if not isinstance(end, np.ndarray) else end
    
#     # Transform to local coordinates
#     dx = end[0] - start[0]
#     dy = end[1] - start[1]
#     c = np.cos(start[2])
#     s = np.sin(start[2])
#     x = (c * dx + s * dy) / turning_radius
#     y = (-s * dx + c * dy) / turning_radius
#     phi = end[2] - start[2]
    
#     # Find best path
#     best_path = ReedsSheppPath()  # Invalid path initially
    
#     CSC(x, y, phi, best_path)
#     CCC(x, y, phi, best_path)
#     CCCC(x, y, phi, best_path)
#     CCSC(x, y, phi, best_path)
#     CCSCC(x, y, phi, best_path)
    
#     return best_path


# def interpolate_path(start, path, t, turning_radius):
#     """
#     Interpolate along a Reeds-Shepp path.
    
#     Args:
#         start: starting pose (x, y, theta) or numpy array
#         path: ReedsSheppPath object
#         t: interpolation parameter [0, 1]
#         turning_radius: turning radius
        
#     Returns:
#         numpy array [x, y, theta] pose at parameter t
#     """
#     # Convert to numpy array if needed
#     start = np.array(start) if not isinstance(start, np.ndarray) else start
    
#     if path.path_type_index == -1:
#         return np.array([999.0, 999.0, 999.0])  # Invalid path
    
#     seg = t * path.total_length
#     phi = start[2]
#     x, y = 0.0, 0.0
    
#     path_type = ReedsSheppPath.PATH_TYPES[path.path_type_index]
    
#     for i in range(5):
#         if seg <= 0:
#             break
            
#         length = path.lengths[i]
#         if length == 0:
#             continue
            
#         if length < 0:
#             v = max(-seg, length)
#             seg += v
#         else:
#             v = min(seg, length)
#             seg -= v
        
#         current_phi = phi
#         segment_type = path_type[i]
        
#         if segment_type == SegmentType.LEFT:
#             x += np.sin(current_phi + v) - np.sin(current_phi)
#             y += -np.cos(current_phi + v) + np.cos(current_phi)
#             phi = current_phi + v
#         elif segment_type == SegmentType.RIGHT:
#             x += -np.sin(current_phi - v) + np.sin(current_phi)
#             y += np.cos(current_phi - v) - np.cos(current_phi)
#             phi = current_phi - v
#         elif segment_type == SegmentType.STRAIGHT:
#             x += v * np.cos(current_phi)
#             y += v * np.sin(current_phi)
    
#     # Transform back to world coordinates
#     final_x = x * turning_radius + start[0]
#     final_y = y * turning_radius + start[1]
#     final_theta = phi

    
#     return np.array([final_x, final_y, final_theta])

# class ReedsSheppTrajectory:
#     """Generate smooth Reeds-Shepp trajectories connecting A* waypoints."""
    
#     def __init__(self, turning_radius):
#         self.turning_radius = turning_radius
    
#     def connect_waypoints(self, waypoints, num_samples=50):
#         """
#         Connect waypoints with Reeds-Shepp curves.
        
#         Args:
#             waypoints: List of (x, y, theta) tuples or numpy array of shape (n, 3)
#             num_samples: Number of samples per segment
            
#         Returns:
#             tuple: (trajectory_points as numpy array, segment_lengths as list)
#         """
#         # Convert to numpy array if needed
#         if not isinstance(waypoints, np.ndarray):
#             waypoints = np.array(waypoints)
        
#         if waypoints.shape[0] < 2:
#             raise ValueError("Need at least 2 waypoints")
        
#         if waypoints.shape[1] != 3:
#             raise ValueError("Waypoints must have shape (n, 3) for (x, y, theta)")
        
#         trajectory_points = []
#         segment_lengths = []
#         for i in range(len(waypoints) - 1):
#             start = waypoints[i]
#             end = waypoints[i + 1]
#             # Compute Reeds-Shepp path
#             path = compute_reeds_shepp_path(start, end, self.turning_radius)
            
#             # Generate trajectory segment
#             segment = []
#             for j in range(num_samples):
#                 t = j / (num_samples - 1) if num_samples > 1 else 0.0
#                 pose = interpolate_path(start, path, t, self.turning_radius)
#                 segment.append(pose)
            
#             # Add to trajectory (avoid duplicating waypoints)
#             if i == 0:
#                 trajectory_points.extend(segment)
#             else:
#                 trajectory_points.extend(segment[1:])
            
#             segment_lengths.append(path.total_length * self.turning_radius)
#         print(trajectory_points[-1])
#         return np.array(trajectory_points), segment_lengths
    
#     def plot_trajectory(self, waypoints, trajectory, title="Reeds-Shepp Trajectory"):
#         """
#         Plot trajectory with waypoints and orientation arrows.
        
#         Args:
#             waypoints: List of waypoints or numpy array
#             trajectory: numpy array of trajectory points
#             title: Plot title
#         """
#         plt.figure(figsize=(10, 8))
        
#         # Convert waypoints to numpy array if needed
#         if not isinstance(waypoints, np.ndarray):
#             waypoints = np.array(waypoints)
        
#         # Plot trajectory
#         plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Reeds-Shepp Trajectory')
        
#         # Plot waypoints
#         plt.plot(waypoints[:, 0], waypoints[:, 1], 'ro-', markersize=8, linewidth=1, label='A* Waypoints')
        
#         # Plot orientation arrows
#         arrow_step = max(1, len(trajectory) // 20)
#         for i in range(0, len(trajectory), arrow_step):
#             x, y, theta = trajectory[i]
#             if abs(x) < 900:  # Skip invalid points
#                 dx = 0.1 * np.cos(theta)
#                 dy = 0.1 * np.sin(theta)
#                 plt.arrow(x, y, dx, dy, head_width=0.05, head_length=0.03, fc='green', ec='green', alpha=0.7)
        
#         plt.grid(True, alpha=0.3)
#         plt.axis('equal')
#         plt.xlabel('X Position')
#         plt.ylabel('Y Position')
#         plt.title(title)
#         plt.legend()
#         plt.tight_layout()
#         plt.show()







#     # Members of Solution class
#     # def generate_trajectoryOLD(self, prm, num_points=1000, degree=3):
#     #     graph = prm.graph
#     #     if not self.path or len(self.path) < 2:
#     #         raise ValueError("Path must contain at least 2 nodes")

#     #     x_coords = nx.get_node_attributes(graph, 'x')
#     #     y_coords = nx.get_node_attributes(graph, 'y')
#     #     theta_coords = nx.get_node_attributes(graph, 'theta')
#     #     pan = nx.get_node_attributes(graph, 'pan')
#     #     tilt = nx.get_node_attributes(graph, 'tilt')

#     #     if not x_coords or not y_coords or not pan or not tilt or not theta_coords:
#     #         raise ValueError("Graph nodes must have all attributes")

#     #     # Vectorized coordinate extraction using list comprehension + array conversion
#     #     try:
#     #         waypoints = np.array([[x_coords[node_id], y_coords[node_id], theta_coords[node_id], pan[node_id], tilt[node_id]] for node_id in self.path])
#     #     except KeyError as e:
#     #         raise ValueError(f"Node {e} not found in graph")

#     #     # Handle edge case with two waypoints
#     #     if len(waypoints) == 2:
#     #         t = np.linspace(0, 1, num_points)[:, np.newaxis]
#     #         trajectory = waypoints[0] + t * (waypoints[1] - waypoints[0])
#     #         self.trajectory = trajectory
#     #         return trajectory

#     #     t = np.linspace(0, 1, len(waypoints))
#     #     spline_x = make_interp_spline(t, waypoints[:, 0], k=degree)
#     #     spline_y = make_interp_spline(t, waypoints[:, 1], k=degree)
#     #     spline_theta = make_interp_spline(t, waypoints[:, 2], k=degree)
#     #     spline_pan = make_interp_spline(t, waypoints[:, 3], k=degree)
#     #     spline_tilt = make_interp_spline(t, waypoints[:, 4], k=degree)

#     #     t_new = np.linspace(0, 1, num_points)
#     #     theta = spline_theta(t_new)

#     #     # Ensure first and last theta match start/end
#     #     theta[0] = waypoints[0, 2]
#     #     theta[-1] = waypoints[-1, 2]

#     #     trajectory = np.column_stack([
#     #         spline_x(t_new),
#     #         spline_y(t_new),
#     #         theta,
#     #         spline_pan(t_new),
#     #         spline_tilt(t_new)
#     #     ])
#     #     self.trajectory = trajectory
#     #     return trajectory
#     def generate_trajectory2(self, prm:PSPRM):
#         """
#         Generate a trajectory using the interpolated values of x, y, theta, pan, and tilt
#         stored in prm.tensors.edges for the nodes in the A* path.
#         """
#         if not self.path or len(self.path) < 2:
#             raise ValueError("Path must contain at least 2 nodes")

#         edges = prm.tensors['edges']  # Shape: [NUM_STATES, K, INTERP_STEPS, DIM]
#         neighbors = prm.tensors['neighbors']  # Shape: [NUM_STATES, K]

#         trajectory = []

#         for i in range(len(self.path) - 1):
#             current_node = self.path[i]
#             next_node = self.path[i + 1]

#             # Find the neighbor index of the next node in the current node's neighbors
#             neighbor_indices = neighbors[current_node].cpu().numpy()
#             if next_node not in neighbor_indices:
#                 # get the edge from the other direction, the next node has the neighbor relationship, not the current node
#                 current_node, next_node = next_node, current_node
#                 neighbor_indices = neighbors[current_node].cpu().numpy()
#                 neighbor_idx = list(neighbor_indices).index(next_node)
#                 edge_data = edges[current_node, neighbor_idx].cpu().numpy()
#                 trajectory.extend(edge_data)
#                 continue

#             neighbor_idx = list(neighbor_indices).index(next_node)

#             # Extract the interpolated edge data
#             edge_data = edges[current_node, neighbor_idx].cpu().numpy()  # Shape: [INTERP_STEPS, DIM]

#             # Append the interpolated points to the trajectory
#             trajectory.extend(edge_data)

#         self.trajectory = np.array(trajectory)  # Convert to numpy array for visualization
#         return self.trajectory


    # Member of PSPRM
    # def addStartAndGoalBROKEN(self, start, goal):  # tensors of shape 5
    #     nodes = self.tensors['nodes']
    #     node_validity = self.tensors['node_validity']
    #     neighbors = self.tensors['neighbors']
    #     edges = self.tensors['edges']
    #     edge_validity = self.tensors['edge_validity']
    #     circles = self.env['circles']
    #     rectangles = self.env['rectangles']

    #     start_state, start_neighbors, start_edges, start_valid_edges, goal_state, goal_neighbors, goal_edges, goal_valid_edges = cuPRM.addStartAndGoal_(
    #         nodes, node_validity, neighbors, edges, edge_validity,
    #         start, goal,
    #         circles, rectangles
    #     )
    #     print(f"\nActual goal: {goal}\n")
    #     print(f"\nReturned goal tensor: {goal_state}\n")
    #     print(f"Actual start: {start}\n")
    #     print(f"\nReturned start tensor: {start_state}\n")
    #     # Add start and goal to the netx graph 
    #     start_id = max(self.graph.nodes) + 1
    #     goal_id = start_id + 1

    #     start_attrs = {
    #         'x': start_state[0].item(),
    #         'y': start_state[1].item(),
    #         'theta': start_state[2].item(),
    #         'pan': 0,
    #         'tilt': 0,
    #         'score': 0.0  # Placeholders, can be updated if needed
    #     }
    #     goal_attrs = {
    #         'x': goal_state[0].item(),
    #         'y': goal_state[1].item(),
    #         'theta': goal_state[2].item(),
    #         'pan': 0,
    #         'tilt': 0,
    #         'score': 0.0  # Placeholders, can be updated if needed
    #     }

    #     self.graph.add_node(start_id, **start_attrs)
    #     self.graph.add_node(goal_id, **goal_attrs)
    #     print(start_id, goal_id)
    #     # Connect edges for start
    #     for i, valid in enumerate(start_valid_edges):
    #         if valid:
    #             nid = start_neighbors[i].item()
    #             src = np.array([start_state[0].item(), start_state[1].item(), start_state[2].item()])
    #             tgt = np.array([self.graph.nodes[nid]['x'], self.graph.nodes[nid]['y'], self.graph.nodes[nid]['theta']])
    #             weight = np.linalg.norm(src - tgt)
    #             self.graph.add_edge(start_id, nid, weight=weight)
            
    #     # Connect edges for goal
    #     for i, valid in enumerate(goal_valid_edges):
    #         if valid:
    #             nid = goal_neighbors[i].item()
    #             src = np.array([goal_state[0].item(), goal_state[1].item(), start_state[2].item()])
    #             tgt = np.array([self.graph.nodes[nid]['x'], self.graph.nodes[nid]['y'], self.graph.nodes[nid]['theta']])
    #             weight = np.linalg.norm(src - tgt)
    #             self.graph.add_edge(goal_id, nid, weight=weight)

    #     return {'start': [start_state, start_neighbors, start_edges, start_valid_edges],
    #             'goal': [goal_state, goal_neighbors, goal_edges, goal_valid_edges]}
