import json
import sys

def convert_json_to_text_format(json_file_path, output_file_path):
    """
    Convert the JSON file containing link and sphere information to the specified text format.
    
    Args:
        json_file_path (str): Path to the input JSON file
        output_file_path (str): Path to the output text file
    """
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        link_spheres = json.load(f)
    
    # Open the output file for writing
    with open(output_file_path, 'w') as f:
        # Write the header
        f.write("collision_spheres:\n")
        
        # Process each link
        for link_name, spheres in link_spheres.items():
            # Write the link name with proper indentation
            f.write(f"            {link_name}:\n")
            
            # Process spheres for this link
            for sphere in spheres:
                # Extract sphere information
                radius = sphere['radius']
                x, y, z = sphere['origin']['xyz']
                
                # Write sphere center with proper indentation
                f.write(f"                - \"center\": [{x}, {y}, {z}]\n")
                
                # Write sphere radius with proper indentation
                f.write(f"                \"radius\": {radius}\n")
    
    print(f"Successfully converted {json_file_path} to {output_file_path}")

if __name__ == "__main__":
    # Check command-line arguments
    
    
    # Get input JSON file path
    json_file = '/home/lenman/link_spheres.json'
    
    # Set output file path (default or user-specified)
    output_file = './link_spheres.txt'
    
    # Convert JSON to text format
    convert_json_to_text_format(json_file, output_file)