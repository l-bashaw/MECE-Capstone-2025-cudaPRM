import torch
import rsplan
import numpy as np
from utils.BatchFK import BatchFk
import networkx as nx
import pytorch_kinematics as pk

import cuPRM
# from scipy.interpolate import make_interp_spline, CubicHermiteSpline

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

        self.urdf_path = "resources/robots/stretch/stretch.urdf"
        self.ee_link_name = 'camera_color_optical_frame_rotated'
        self.device = 'cuda'
        self.bfk = BatchFk(self.urdf_path, self.ee_link_name, device=self.device, batch_size = 1560)

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
        _ = self.project_nodes(nodes, object_pose_world)    # Project nodes 
        
        if (self.bfk.batch_size != nodes.shape[0]):
            raise ValueError(f"Batch size {self.bfk.batch_size} does not match number of nodes {nodes.shape[0]}")
        
        model_input = self.create_model_input(nodes, object_pose_world, obj_label)
        p_scores = self.model(model_input)

        self.graph = self.tensors_to_networkx(nodes, neighbors, p_scores)
        return 

    def addStartAndGoal(self, start, goal):  # tensors of shape 5
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

        # Find nearest neighbors in the existing graph, manually
        all_nodes = np.array([[data['x'], data['y'], data['theta']] for _, data in self.graph.nodes(data=True)])
        start_dists = np.linalg.norm(all_nodes - start[:3], axis=1)
        goal_dists = np.linalg.norm(all_nodes - goal[:3], axis=1)
        start_neighbor_ids = np.argsort(start_dists)[:10]  # Get indices of 10 nearest neighbors
        goal_neighbor_ids = np.argsort(goal_dists)[:10]
 
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

       
    def create_model_input(self, nodes, object_pose, label):
       
        camera_positions, camera_quaternions, camera_matrices = self.bfk.batch_fk(nodes)
        model_input_tensor = torch.zeros((self.bfk.batch_size, 10), dtype=torch.float32).to(torch.device("cuda"))
        object_matrix = self.bfk.pose_to_transformation_matrix(object_pose)
        pose_list_tensor = self.bfk.transform_poses_batch(camera_matrices, object_matrix)
        # print any NaN values in pose_list_tensor
        if torch.any(torch.isnan(pose_list_tensor)):
            print("BANG: NaN values in pose_list_tensor")
            print("NaN values:", pose_list_tensor[torch.isnan(pose_list_tensor)])
        # insert the label
        num_poses = pose_list_tensor.shape[0]
        model_input_tensor[:num_poses, :7] = pose_list_tensor[:num_poses]
        model_input_tensor[:, 7:] = label
        # print any NaN values in model_input_tensor
        if torch.any(torch.isnan(model_input_tensor)):
            print("CLANG: NaN values in model_input_tensor")
            print("NaN values:", model_input_tensor[torch.isnan(model_input_tensor)])
        return model_input_tensor
    

    def project_nodes(self, nodes, object_pose_world):
        """
        Calculate pan and tilt angles for robot states to look at target object,
        update camera transforms with pan/tilt, and compute relative pose between camera and object.

        Args:
            nodes: Tensor [B, 5] = (x, y, theta, pan, tilt)
            object_pose_world: Tensor [7] = object pose [x, y, z, qx, qy, qz, qw] in world

        Returns:
            diffs: [B, 7] = (object xyz in cam frame, relative orientation quaternion)
        """
        # ...existing code for pan/tilt calculation (unchanged)...
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

        # Object position homogeneous (only use position for pan/tilt calculation)
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

        # Object in camera base frame (before pan/tilt) - for pan/tilt calculation
        T_cam_base_world = torch.inverse(T_world_cam_base)
        obj_in_cam_base = torch.bmm(T_cam_base_world, object_pos_world_h.unsqueeze(-1)).squeeze(-1)[:, :3]
        dx, dy, dz = obj_in_cam_base[:, 0], obj_in_cam_base[:, 1], obj_in_cam_base[:, 2]

        # Pan/tilt calculation (unchanged)
        pan = torch.atan2(dx, dz)
        tilt = torch.atan2(dy, torch.sqrt(dx**2 + dz**2))
        pan = torch.clamp(pan, -3.9, 1.5)
        tilt = torch.clamp(tilt, -1.53, 0.79)

        # Save into nodes
        nodes[:, 3] = pan
        nodes[:, 4] = tilt
     
        return T_cam_base_world
        
      

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
       
        # get average of src and target scores
        src_scores = scores[source_indices]
        tgt_scores = scores[target_indices]
        avg_scores = (src_scores + tgt_scores) / 2

        weights = np.linalg.norm(src_positions - tgt_positions, axis=1) * ((1-avg_scores)**2)

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
            raise KeyError("No weights in graph")

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
            # if G.has_edge(u, v):
            ux = G.nodes[u]['x']
            uy = G.nodes[u]['y']
            ut = G.nodes[u]['theta']
            motion_cost = ((ux - goal_x) ** 2 + (uy - goal_y) ** 2 ) ** 0.5 / weight_range
            # else:
            #     motion_cost = float('inf')  # No direct edge, set to infinity

            # Use the average of the scores instead of the minimum
            score_term = (u_score + v_score)/2 
            # if motion_cost != float('inf'):
            #     print(f"score term {score_term}")
            #     print(f"motion cost {motion_cost}")
            # print(f"Score term: {score_term:.4f}, Motion cost: {motion_cost:.4f}")
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
            print("No need to simplify. Path: ", self.path)
            return

        pan_bounds = (-3.9, 1.5)
        tilt_bounds = (-1.53, 0.79)

        circles = env['circles'].cpu().numpy()
        rectangles = env['rectangles'].cpu().numpy()
        graph = prm.graph

        x_coords = nx.get_node_attributes(graph, 'x')
        y_coords = nx.get_node_attributes(graph, 'y')
        theta_coords = nx.get_node_attributes(graph, 'theta')
        pans = nx.get_node_attributes(graph, 'pan')

    
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

        # print(f"Original path: {self.path}")
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

        if simplified_path[-1] != self.path[-1]:
            simplified_path.append(self.path[-1])

        # print(f"Simplified path: {simplified_path}") 
        
        # After simplification, try to skip nodes with low scores
        final_path = [simplified_path[0]]
        for i in range(1, len(simplified_path) - 1):
            node_id = simplified_path[i]
            if graph.nodes[node_id]['score'] < 0.3:
                if not in_collision(
                    np.array([x_coords[final_path[-1]], y_coords[final_path[-1]], theta_coords[final_path[-1]]]),
                    np.array([x_coords[simplified_path[i + 1]], y_coords[simplified_path[i + 1]], theta_coords[simplified_path[i + 1]]])
                ):
                    continue
            final_path.append(node_id)
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
