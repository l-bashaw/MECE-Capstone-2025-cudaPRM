import xml.etree.ElementTree as ET
import numpy as np

def parse_urdf_spheres(urdf_file_path):
    """
    Parse a URDF file and extract links and their associated spheres.
    
    Args:
        urdf_file_path (str): Path to the URDF file
        
    Returns:
        dict: Dictionary with link names as keys and lists of sphere information as values.
              Each sphere is represented as a dictionary with 'radius' and 'origin' keys.
    """
    # Check if the file exists
    import os
    if not os.path.isfile(urdf_file_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_file_path}")
    # Parse the XML
    tree = ET.parse(urdf_file_path)
    root = tree.getroot()
    
    # Dictionary to store link name and associated spheres
    link_spheres = {}
    
    # Find all link elements
    for link in root.findall('.//link'):
        link_name = link.get('name')
        spheres = []
        
        # Find all collision elements in this link
        for collision in link.findall('.//collision'):
            # Check if the geometry contains a sphere
            sphere_elem = collision.find('.//geometry/sphere')
            if sphere_elem is not None:
                # Get sphere radius
                radius = float(sphere_elem.get('radius'))
                
                # Get origin information
                origin_elem = collision.find('origin')
                if origin_elem is not None:
                    # Extract xyz coordinates
                    xyz = origin_elem.get('xyz')
                    if xyz:
                        x, y, z = map(float, xyz.split())
                    else:
                        x, y, z = 0.0, 0.0, 0.0
                    
                    # Extract rpy (roll, pitch, yaw) if available
                    rpy = origin_elem.get('rpy')
                    if rpy:
                        roll, pitch, yaw = map(float, rpy.split())
                    else:
                        roll, pitch, yaw = 0.0, 0.0, 0.0
                else:
                    # Default values if origin is not specified
                    x, y, z = 0.0, 0.0, 0.0
                    roll, pitch, yaw = 0.0, 0.0, 0.0
                
                # Add sphere information to the list
                sphere_info = {
                    'radius': radius,
                    'origin': {
                        'xyz': [x, y, z],
                        'rpy': [roll, pitch, yaw]
                    }
                }
                spheres.append(sphere_info)
        
        # Add link and its spheres to the dictionary
        if spheres:  # Only add links that have spheres
            link_spheres[link_name] = spheres
    
    return link_spheres

def save_sphere_dict(sphere_dict, output_file):
    """
    Save the sphere dictionary to a file.
    
    Args:
        sphere_dict (dict): Dictionary with link names and sphere information
        output_file (str): Path to save the dictionary
    """
    import json
    with open(output_file, 'w') as f:
        json.dump(sphere_dict, f, indent=4)

def load_sphere_dict(input_file):
    """
    Load the sphere dictionary from a file.
    
    Args:
        input_file (str): Path to the saved dictionary
        
    Returns:
        dict: Dictionary with link names and sphere information
    """
    import json
    with open(input_file, 'r') as f:
        return json.load(f)

# Example usage
if __name__ == "__main__":
    import sys
    
    
    
    urdf_file = "/home/lenman/capstone/parallelrm/resources/robot/stretch/stretch_spherized.urdf"
    
    # Default output file or user-specified
    output_file = "./link_spheres.json"
    
    # Parse the URDF and extract sphere information
    link_spheres = parse_urdf_spheres(urdf_file)
    
    # Print the result
    print(f"Found {len(link_spheres)} links with spheres:")
    for link_name, spheres in link_spheres.items():
        print(f"  {link_name}: {len(spheres)} sphere(s)")
    
    # Save to file
    save_sphere_dict(link_spheres, output_file)
    print(f"Saved sphere information to {output_file}")