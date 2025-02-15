import numpy as np

def generate_random_poses(n):
    """
    Generate a list of n random poses.
    Each pose is a 7-element vector: [x, y, z, qx, qy, qz, qw]
    """
    poses = np.random.uniform(-1, 1, size=(n, 7))  # Generate random values in range [-1, 1]
    
    # Normalize quaternion part (last 4 values per pose)
    for i in range(n):
        quaternion = poses[i, 3:]
        quaternion /= np.linalg.norm(quaternion)
    
    return poses

# Example usage
n = 10000  # Number of poses
import time
start = time.time()
generated_poses = generate_random_poses(n)
print("Time taken: ", time.time() - start)