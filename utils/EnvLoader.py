import yaml
import numpy as np
import torch
import trimesh
from typing import Dict, List, Tuple, Optional
from scipy.spatial.transform import Rotation

class EnvironmentLoader:
    def __init__(self, device='cuda'):
        """
        Initialize the environment loader.
        
        Args:
            device (str): PyTorch device ('cuda' or 'cpu')
        """
        self.device = device
        self.cc_objects = {}
        self.rectangles = []
        self.circles = []
        self.bounds = None
        
    def load_world(self, config_file: str, bounds: Optional[List[List[float]]] = None) -> Dict:
        """
        Load world from YAML configuration and convert to PyTorch format.
        
        Args:
            config_file (str): Path to YAML configuration file
            bounds (Optional[List[List[float]]]): Custom bounds [[x_min, y_min], [x_max, y_max]]
                                                 If None, will compute from obstacles
        
        Returns:
            Dict: Environment dictionary with PyTorch tensors
        """
        # Load YAML configuration
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        cc_objects_list = config["world"]["collision_objects"]
        self.cc_objects = {obj["id"]: obj for obj in cc_objects_list}
        
        # Reset containers
        self.rectangles = []
        self.circles = []
        
        # Process each collision object
        for cc_object_name, cc_object in self.cc_objects.items():
            print(f"Processing object: {cc_object_name}")
            
            # Get pose information
            pose = cc_object["mesh_poses"][0]
            rotation = pose["orientation"]  # [x, y, z, w]
            translation = pose["position"]
            
            # Get mesh information
            mesh_info = cc_object["meshes"][0]
            dimensions = mesh_info["dimensions"]
            resource_path = mesh_info["resource"]
            
            print(f"  Resource: {resource_path}")
           
            try:
                # Load and process mesh using trimesh
                mesh = trimesh.load(resource_path, force='mesh')
                
                # Handle case where trimesh returns a Scene object
                if isinstance(mesh, trimesh.Scene):
                    # Get the combined mesh from all geometries in the scene
                    mesh = mesh.dump(concatenate=True)
                
                if len(mesh.vertices) == 0:
                    print(f"  Warning: Could not load mesh for {cc_object_name}, skipping...")
                    continue
                
                # Scale mesh vertices
                mesh.vertices = mesh.vertices * np.array(dimensions)
                
                # Apply transformations
                mesh.apply_translation(translation)
                
                # Convert quaternion [x, y, z, w] to rotation matrix
                quat_xyzw = rotation
                rot = Rotation.from_quat(quat_xyzw)
                rotation_matrix = rot.as_matrix()
                
                # Create 4x4 transformation matrix
                transform = np.eye(4)
                transform[:3, :3] = rotation_matrix
                mesh.apply_transform(transform)
                
                # Get 2D bounding box
                bbox_3d = mesh.bounds
                
                # Convert to 2D representation
                self._process_object_to_2d(cc_object_name, bbox_3d, cc_object)
                
            except Exception as e:
                print(f"  Error processing {cc_object_name}: {e}")
                print(f"  Creating bounding box from dimensions and position...")
                self._create_bbox_from_dimensions(cc_object_name, translation, dimensions)
        
        # Set bounds
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = self._compute_bounds_from_obstacles()
        
        # Convert to PyTorch format
        return self._create_pytorch_environment()
    
    def _process_object_to_2d(self, object_name: str, bbox_3d, cc_object: Dict):
        """Process 3D bounding box to 2D representation."""
        # bbox_3d is a (2, 3) array with min and max bounds
        min_bound = bbox_3d[0]
        max_bound = bbox_3d[1]
        
        # Calculate 2D center and extents
        center_x = (max_bound[0] + min_bound[0]) / 2
        center_y = (max_bound[1] + min_bound[1]) / 2
        width = max_bound[0] - min_bound[0]
        height = max_bound[1] - min_bound[1]
        
        # Decide whether to represent as circle or rectangle
        if self._should_be_circle(width, height, object_name):
            # Use circle representation
            radius = max(width, height) / 2
            self.circles.append([center_x, center_y, radius])
        else:
            # Use rectangle representation
            self.rectangles.append([center_x, center_y, height, width])  # x, y, height, width
        
        # Store in cc_objects for reference
        cc_object["bbox_2d"] = [center_x, center_y, width/2, height/2]
        cc_object["representation"] = "circle" if self._should_be_circle(width, height, object_name) else "rectangle"
    
    def _create_bbox_from_dimensions(self, object_name: str, translation: List[float], dimensions: List[float]):
        """Create bounding box directly from dimensions when mesh loading fails."""
        center_x, center_y = translation[0], translation[1]
        width, height = dimensions[0], dimensions[1]
        
        if self._should_be_circle(width, height, object_name):
            radius = max(width, height) / 2
            self.circles.append([center_x, center_y, radius])
            print(f"  Added as circle (from dimensions): center=({center_x:.2f}, {center_y:.2f}), radius={radius:.2f}")
        else:
            self.rectangles.append([center_x, center_y, height, width])
            print(f"  Added as rectangle (from dimensions): center=({center_x:.2f}, {center_y:.2f}), size=({height:.2f}x{width:.2f})")
    
    def _should_be_circle(self, width: float, height: float, object_name: str) -> bool:
        """
        Determine if an object should be represented as a circle or rectangle.
        
        Args:
            width (float): Object width
            height (float): Object height
            object_name (str): Name of the object
        
        Returns:
            bool: True if should be circle, False if rectangle
        """
        # Heuristics for circle vs rectangle decision
        aspect_ratio = max(width, height) / min(width, height)
        
        # If nearly square and certain object types, use circle
        circular_objects = ['table', 'chair', 'plant', 'lamp', 'person', 'human']
        is_circular_type = any(obj_type.lower() in object_name.lower() for obj_type in circular_objects)
        
        # Use circle if aspect ratio is close to 1 and it's a typically circular object
        if aspect_ratio < 1.3 and is_circular_type:
            return True
        
        # For very small objects, use circles for simplicity
        if max(width, height) < 0.5:
            return True
            
        return False
    
    def _compute_bounds_from_obstacles(self) -> List[List[float]]:
        """Compute environment bounds from obstacle positions."""
        if not self.rectangles and not self.circles:
            return [[0.0, 0.0], [10.0, 10.0]]  # Default bounds
        
        all_coords = []
        
        # Add rectangle bounds
        for rect in self.rectangles:
            x, y, height, width = rect
            all_coords.extend([
                [x - width/2, y - height/2],
                [x + width/2, y + height/2]
            ])
        
        # Add circle bounds
        for circle in self.circles:
            x, y, radius = circle
            all_coords.extend([
                [x - radius, y - radius],
                [x + radius, y + radius]
            ])
        
        if all_coords:
            coords_array = np.array(all_coords)
            min_bounds = coords_array.min(axis=0)
            max_bounds = coords_array.max(axis=0)
            
            # Add some padding
            padding = 1.0
            min_bounds -= padding
            max_bounds += padding
            
            return [min_bounds.tolist(), max_bounds.tolist()]
        else:
            return [[0.0, 0.0], [10.0, 10.0]]
    
    def _create_pytorch_environment(self) -> Dict:
        """Convert loaded environment to PyTorch format."""
        env = {
            'bounds': torch.tensor(self.bounds, dtype=torch.float32, device=self.device)
        }
        
        # Convert rectangles
        if self.rectangles:
            env['rectangles'] = torch.tensor(
                self.rectangles, 
                dtype=torch.float32, 
                device=self.device
            )
        else:
            env['rectangles'] = torch.empty((0, 4), dtype=torch.float32, device=self.device)
        
        # Convert circles
        if self.circles:
            env['circles'] = torch.tensor(
                self.circles, 
                dtype=torch.float32, 
                device=self.device
            )
        else:
            env['circles'] = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        
        return env
    
    def visualize_environment(self, env: Dict, figsize: Tuple[int, int] = (10, 8), 
                            show_grid: bool = True, save_path: Optional[str] = None):
        """
        Visualize the 2D environment with rectangles and circles.
        
        Args:
            env (Dict): Environment dictionary with PyTorch tensors
            figsize (Tuple[int, int]): Figure size (width, height)
            show_grid (bool): Whether to show grid lines
            save_path (Optional[str]): Path to save the figure, if None just display
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get bounds
        bounds = env['bounds'].cpu().numpy()
        x_min, y_min = bounds[0]
        x_max, y_max = bounds[1]
        
        # Draw rectangles
        rectangles = env['rectangles'].cpu().numpy()
        for i, rect in enumerate(rectangles):
            x, y, height, width = rect
            # Create rectangle patch (bottom-left corner coordinates)
            rect_patch = patches.Rectangle(
                (x - width/2, y - height/2), width, height,
                linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.7
            )
            ax.add_patch(rect_patch)
            # Add label
            ax.text(x, y, f'R{i}', ha='center', va='center', fontsize=8, weight='bold')
        
        # Draw circles
        circles = env['circles'].cpu().numpy()
        for i, circle in enumerate(circles):
            x, y, radius = circle
            circle_patch = patches.Circle(
                (x, y), radius,
                linewidth=2, edgecolor='red', facecolor='lightcoral', alpha=0.7
            )
            ax.add_patch(circle_patch)
            # Add label
            ax.text(x, y, f'C{i}', ha='center', va='center', fontsize=8, weight='bold')
        
        # Set axis properties
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('2D Environment: Rectangles and Circles')
        
        # Add grid if requested
        if show_grid:
            ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = []
        if len(rectangles) > 0:
            legend_elements.append(patches.Patch(color='lightblue', label=f'Rectangles ({len(rectangles)})'))
        if len(circles) > 0:
            legend_elements.append(patches.Patch(color='lightcoral', label=f'Circles ({len(circles)})'))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')
        
        # Add bounds info as text
        bounds_text = f'Bounds: [{x_min:.1f}, {y_min:.1f}] to [{x_max:.1f}, {y_max:.1f}]'
        ax.text(0.02, 0.98, bounds_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Environment visualization saved to: {save_path}")
        
        plt.show()
    
    def print_environment_summary(self, env: Dict):
        """Print a summary of the loaded environment."""
        print("\n" + "="*50)
        print("ENVIRONMENT SUMMARY")
        print("="*50)
        print(f"Device: {env['bounds'].device}")
        print(f"Bounds: {env['bounds'].cpu().numpy().tolist()}")
        print(f"Rectangles: {env['rectangles'].shape[0]} objects")
        if env['rectangles'].shape[0] > 0:
            print("  Rectangle details (x, y, height, width):")
            for i, rect in enumerate(env['rectangles'].cpu().numpy()):
                print(f"    {i}: [{rect[0]:.2f}, {rect[1]:.2f}, {rect[2]:.2f}, {rect[3]:.2f}]")
        
        print(f"Circles: {env['circles'].shape[0]} objects")
        if env['circles'].shape[0] > 0:
            print("  Circle details (x, y, radius):")
            for i, circle in enumerate(env['circles'].cpu().numpy()):
                print(f"    {i}: [{circle[0]:.2f}, {circle[1]:.2f}, {circle[2]:.2f}]")
        print("="*50)

# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = EnvironmentLoader(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load environment from YAML
    config_file = "/home/lenman/capstone/parallelrm/resources/scenes/scene_hostpital_plant_0.yaml"  # Update this path
    
    # Option 1: Auto-compute bounds
    env = loader.load_world(config_file)
    
    # Option 2: Custom bounds
    # custom_bounds = [[-5.0, -5.0], [5.0, 5.0]]
    # env = loader.load_world(config_file, bounds=custom_bounds)
    
    # Print summary
    loader.print_environment_summary(env)
    loader.visualize_environment(env, show_grid=True)
    # Your environment is now ready for use with PRM planning
    print(f"\nEnvironment ready! Bounds shape: {env['bounds'].shape}")
    print(f"Rectangles shape: {env['rectangles'].shape}")
    print(f"Circles shape: {env['circles'].shape}")