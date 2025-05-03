from PIL import Image
import numpy as np
from typing import Union, Tuple
import os

def image_to_rgb_array(image_path: str) -> Union[np.ndarray, None]:
    """
    Reads an image file and converts it to a 3D numpy array containing sRGB values.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: 3D array of shape (height, width, 3) containing RGB values
                      Each value is in range 0-255
        None: If there's an error reading or processing the image
    """
    try:
        # Verify file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Open and convert image to RGB mode
        with Image.open(image_path) as img:
            # Convert to RGB mode if image is in a different mode (e.g., RGBA, L)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array
            rgb_array = np.array(img)
            
            # Verify array shape and type
            if len(rgb_array.shape) != 3 or rgb_array.shape[2] != 3:
                raise ValueError("Invalid image format: Expected RGB image")
                
            return rgb_array
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def save_rgb_array(rgb_array: np.ndarray, output_path: str, include_metadata: bool = True) -> bool:
    """
    Save RGB array to a text file with optional metadata.
    
    Args:
        rgb_array (numpy.ndarray): 3D array containing RGB values
        output_path (str): Path where to save the text file
        include_metadata (bool): Whether to include array metadata in the file
    """
    try:
        with open(output_path, 'w') as f:
            # Write metadata if requested
            if include_metadata:
                f.write(f"Image Array Metadata:\n")
                f.write(f"Dimensions: {rgb_array.shape}\n")
                f.write(f"Data Type: {rgb_array.dtype}\n")
                f.write(f"Value Range: [{np.min(rgb_array)}, {np.max(rgb_array)}]\n")
                f.write("-" * 50 + "\n\n")
            
            # Write array data
            height, width, _ = rgb_array.shape
            f.write(f"RGB Values ({height}x{width} pixels):\n")
            
            for y in range(height):
                for x in range(width):
                    r, g, b = rgb_array[y, x]
                    f.write(f"({r},{g},{b}) ")
                f.write("\n")
                
        print(f"Array saved successfully to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error saving array: {str(e)}")
        return False

def create_threshold_image(rgb_array: np.ndarray, output_path: str, threshold: Tuple[int, int, int] = (100, 100, 100)) -> bool:
    """
    Create and save a binary image where pixels are black if they exceed the threshold and white if under.
    
    Args:
        rgb_array (numpy.ndarray): 3D array containing RGB values
        output_path (str): Path where to save the thresholded image
        threshold (tuple): RGB threshold values
    """
    try:
        height, width, _ = rgb_array.shape
        # Create new array for binary image
        binary_array = np.zeros((height, width), dtype=np.uint8)
        
        # Apply threshold
        for y in range(height):
            for x in range(width):
                r, g, b = rgb_array[y, x]
                # If all values exceed threshold, make pixel black (0)
                # Otherwise make it white (255)
                if r > threshold[0] and g > threshold[1] and b > threshold[2]:
                    binary_array[y, x] = 0
                else:
                    binary_array[y, x] = 255
        
        # Convert to PIL Image and save
        binary_image = Image.fromarray(binary_array, mode='L')
        binary_image.save(output_path)
        print(f"Threshold image saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating threshold image: {str(e)}")
        return False

def print_rgb_array(rgb_array: np.ndarray, max_rows: int = 10, max_cols: int = 10):
    """
    Print RGB array in a formatted way, with optional size limits.
    """
    if rgb_array is None:
        print("No array to print")
        return

    height, width, _ = rgb_array.shape
    
    # Determine how many rows and columns to print
    rows_to_print = min(height, max_rows)
    cols_to_print = min(width, max_cols)
    
    print(f"\nRGB Array Preview ({height}x{width} pixels):")
    print("-" * 50)
    
    for y in range(rows_to_print):
        row_str = ""
        for x in range(cols_to_print):
            r, g, b = rgb_array[y, x]
            row_str += f"({r:3d},{g:3d},{b:3d}) "
        
        # Add ellipsis if we're not showing all columns
        if width > max_cols:
            row_str += "..."
            
        print(row_str)
    
    # Add ellipsis if we're not showing all rows
    if height > max_rows:
        print("...")
    
    print("-" * 50)
    print(f"Array shape: {rgb_array.shape}")
    print(f"Data type: {rgb_array.dtype}")
    print(f"Value range: [{np.min(rgb_array)}, {np.max(rgb_array)}]")

# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual paths
    image_path = "/home/aashu/temp/unnamed.jpg"  # Your input image path
    rgb_txt_path = "/home/aashu/temp/rgb_values.txt"  # Where to save the RGB values text file
    threshold_image_path = "/home/aashu/temp/contour_image.png"  # Where to save the threshold image
    
    # Convert image to RGB array
    rgb_array = image_to_rgb_array(image_path)
    
    if rgb_array is not None:
        # Print preview of the array
        print_rgb_array(rgb_array)
        
        # Save RGB values to text file
        save_rgb_array(rgb_array, rgb_txt_path)
        
        # Create and save threshold image
        create_threshold_image(rgb_array, threshold_image_path)
