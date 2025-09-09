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
        self.bfk = BatchFk(self.urdf_path, self.ee_link_name, device=self.device, batch_size = 1500)

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
        _ = self.project_nodesFIX3(nodes, object_pose_world)    # Project nodes 

        # cam_mat = self.project_nodesFIX3(nodes, object_pose_world)
        


        model_input = self.create_model_input(nodes, object_pose_world, obj_label)
        p_scores = self.model(model_input)
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

        # Print edges for start and goal
        # print("Start edges:", list(self.graph.edges(start_id, data=True)))
        # print("Goal edges:", list(self.graph.edges(goal_id, data=True)))
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

    def create_model_input(self, nodes, object_pose, label):
       
       
        camera_positions, camera_quaternions, camera_matrices = self.bfk.batch_fk(nodes)
        model_input_tensor = torch.empty((self.bfk.batch_size, 10), dtype=torch.float32).to(torch.device("cuda"))
        object_matrix = self.bfk.pose_to_transformation_matrix(object_pose)
        pose_list_tensor = self.bfk.transform_poses_batch(camera_matrices, object_matrix)
        # insert the label
        num_poses = pose_list_tensor.shape[0]
        model_input_tensor[:num_poses, :7] = pose_list_tensor[:num_poses]
        model_input_tensor[:, 7:] = label
        return model_input_tensor
        
        
        
        
        
        
        
        object_matrix = self.pose_to_transformation_matrix(object_pose)
        pose_list_tensor = self.transform_poses_batch(camera_matrices, object_matrix)
        # camera_positions_cpu = camera_positions.cpu().numpy()
        model_input_tensor = torch.empty((nodes.shape[0], 10), dtype=torch.float32).to(torch.device("cuda"))
     
        object_matrix = self.pose_to_transformation_matrix(object_pose)
        pose_list_tensor = self.transform_poses_batch(camera_matrices, object_matrix)

        # insert the label
        num_poses = min(pose_list_tensor.shape[0], model_input_tensor.shape[0])
        model_input_tensor[:num_poses, :7] = pose_list_tensor[:num_poses]
        model_input_tensor[:, 7:] = label

        return model_input_tensor
    

    def project_nodesFIX3(self, nodes, object_pose_world):
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
        # print(pan[:5])
        # print(tilt[:5])
        return T_cam_base_world
        
        # # --- Apply pan/tilt to camera transform ---
        # def make_pan_tilt_transform(pan, tilt, dtype, device):
        #     B = pan.shape[0]
        #     T = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)

        #     # Pan: rotation around Y
        #     cos_p, sin_p = torch.cos(pan), torch.sin(pan)
        #     R_pan = torch.stack([
        #         torch.stack([ cos_p, torch.zeros_like(pan), sin_p], dim=1),
        #         torch.stack([ torch.zeros_like(pan), torch.ones_like(pan), torch.zeros_like(pan)], dim=1),
        #         torch.stack([-sin_p, torch.zeros_like(pan), cos_p], dim=1)
        #     ], dim=1)

        #     # Tilt: rotation around X
        #     cos_t, sin_t = torch.cos(tilt), torch.sin(tilt)
        #     R_tilt = torch.stack([
        #         torch.stack([ torch.ones_like(tilt), torch.zeros_like(tilt), torch.zeros_like(tilt)], dim=1),
        #         torch.stack([ torch.zeros_like(tilt), cos_t, -sin_t], dim=1),
        #         torch.stack([ torch.zeros_like(tilt), sin_t,  cos_t], dim=1)
        #     ], dim=1)

        #     R = torch.bmm(R_tilt, R_pan)  # tilt ∘ pan
        #     T[:, :3, :3] = R
        #     return T

        # T_cam_base_cam = make_pan_tilt_transform(pan, tilt, dtype, device)
        # T_world_cam = torch.bmm(T_world_cam_base, T_cam_base_cam)
        # T_cam_world = torch.inverse(T_world_cam)

        # # --- Now compute full relative pose including object orientation ---
        
        # # Create object transform in world frame
        # object_pos = object_pose_world[:3]
        # object_quat = object_pose_world[3:7]  # [qx, qy, qz, qw]
        # object_rot_matrix = pk.quaternion_to_matrix(object_quat.unsqueeze(0)).squeeze(0)  # [3, 3]
        
        # # Object transform in world frame
        # T_world_object = torch.eye(4, dtype=dtype, device=device)
        # T_world_object[:3, :3] = object_rot_matrix
        # T_world_object[:3, 3] = object_pos
        # T_world_object = T_world_object.unsqueeze(0).expand(B, -1, -1)  # [B, 4, 4]

        # # Object pose in camera frame
        # T_cam_object = torch.bmm(T_cam_world, T_world_object)
        
        # # Extract relative position (object position in camera frame)
        # obj_pos_in_cam = T_cam_object[:, :3, 3]  # [B, 3]
        
        # # Extract relative orientation (object orientation in camera frame)
        # obj_rot_in_cam = T_cam_object[:, :3, :3]  # [B, 3, 3]
        # obj_quat_in_cam = pk.matrix_to_quaternion(obj_rot_in_cam)  # [B, 4]

        # # Final diffs: [relative_position, relative_orientation]
        # diffs = torch.cat([obj_pos_in_cam, obj_quat_in_cam], dim=1)  # [B, 7]
        # return diffs

    def project_nodesFIX2(self, nodes, object_pose_world):
        """
        Calculate pan and tilt angles for robot states to look at target object,
        update camera transforms with pan/tilt, and compute relative pose between camera and object.

        Args:
            nodes: Tensor [B, 5] = (x, y, theta, pan, tilt)
            object_pose_world: Tensor [3] = object xyz in world

        Returns:
            diffs: [B, 7] = (object xyz in cam frame, relative orientation quaternion)
        """
        # ...existing code...
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

        # Compute relative orientation quaternion
        # Since pan/tilt already points camera Z-axis at object, the relative orientation
        # represents the rotation needed to align camera with "canonical" view of object
        
        # Method 1: Use the camera's current orientation relative to world
        # This gives you how the camera is oriented when looking at the object
        R_cam_world = T_cam_world[:, :3, :3]
        relative_quat = pk.matrix_to_quaternion(R_cam_world)
        
        # Method 2: Compute orientation relative to "looking directly at object"
        # If you want the quaternion that represents deviation from perfect alignment:
        # 
        # # Camera Z-axis after pan/tilt should point at object
        # obj_direction = obj_in_cam / torch.norm(obj_in_cam, dim=1, keepdim=True)
        # camera_z = torch.tensor([0., 0., 1.], device=device, dtype=dtype).expand_as(obj_direction)
        # 
        # # The relative quaternion represents rotation from perfect alignment
        # # For perfect alignment, obj_direction should equal -camera_z (camera looks down -Z)
        # # Compute quaternion that rotates camera_z to align with -obj_direction
        # target_direction = -obj_direction
        # cross_prod = torch.cross(camera_z, target_direction, dim=1)
        # dot_prod = torch.sum(camera_z * target_direction, dim=1, keepdim=True)
        # 
        # # Handle the case where vectors are aligned or opposite
        # cross_norm = torch.norm(cross_prod, dim=1, keepdim=True)
        # 
        # relative_quat = torch.zeros(B, 4, device=device, dtype=dtype)
        # relative_quat[:, 3] = (1 + dot_prod.squeeze()) / 2  # w component
        # 
        # # For non-parallel vectors
        # mask = cross_norm.squeeze() > 1e-6
        # if mask.any():
        #     relative_quat[mask, :3] = cross_prod[mask] / (2 * relative_quat[mask, 3:4])

        # Final diffs: [object_position_in_camera, camera_orientation_quaternion]
        diffs = torch.cat([obj_in_cam, relative_quat], dim=1)  # [B, 7]
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
        
        # # After simplicification, attempt to skip nodes where pan is at the bounds 
        # # Note that collision checking must be performed
        # final_path = [simplified_path[0]]
        # for i in range(1, len(simplified_path) - 1):
        #     node_id = simplified_path[i]
        #     pan = pans[node_id]
        #     if pan <= pan_bounds[0] + 0.1 or pan >= pan_bounds[1] - 0.1:
        #         if not in_collision(
        #             np.array([x_coords[final_path[-1]], y_coords[final_path[-1]], theta_coords[final_path[-1]]]),
        #             np.array([x_coords[simplified_path[i + 1]], y_coords[simplified_path[i + 1]], theta_coords[simplified_path[i + 1]]])
        #         ):
        #             continue
        #     else:
        #         final_path.append(node_id)
        # final_path.append(simplified_path[-1])

        # print(f"Final path after pan/tilt simplification: {final_path}")
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
# import torch

# import math

# import kornia.geometry.conversions as kornia_conv

# import pytorch_kinematics as pk

# import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D





# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dtype = torch.float64


# urdf = "/home/lb73/cudaPRM/planning/resources/robots/stretch/stretch.urdf"



# # Load robot kinematics chain

# chain = pk.build_serial_chain_from_urdf(open(urdf, mode='rb').read(), "link_head", "base_link")
# chain = chain.to(device=device, dtype=dtype)



# # Get camera mount transform in robot frame

# th = [0.0, 0.0]  # pan, tilt at 0 for base pose

# ret = chain.forward_kinematics(th, end_only=False)

# tg = ret['link_head_nav_cam']

# m = tg.get_matrix()

# pos = m[:, :3, 3]

# rot = pk.matrix_to_quaternion(m[:, :3, :3])

# cam_base_rot = kornia_conv.quaternion_to_rotation_matrix(rot)

# print(rot)
# print(pos)
# print(cam_base_rot)