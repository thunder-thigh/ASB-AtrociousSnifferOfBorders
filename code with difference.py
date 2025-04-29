from PIL import Image
import numpy as np
from typing import Union, Tuple
import os

def image_to_rgb_array(image_path: str) -> Union[np.ndarray, None]:
    """
    Reads an image file and converts it to a 3D numpy array containing sRGB values.
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            rgb_array = np.array(img)
            
            if len(rgb_array.shape) != 3 or rgb_array.shape[2] != 3:
                raise ValueError("Invalid image format: Expected RGB image")
                
            return rgb_array
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def calculate_adjacent_differences(rgb_array: np.ndarray) -> np.ndarray:
    """
    Calculate the sum of differences between each pixel and its adjacent pixels.
    
    Args:
        rgb_array (numpy.ndarray): 3D array containing RGB values
        
    Returns:
        numpy.ndarray: 2D array containing difference values
    """
    height, width, _ = rgb_array.shape
    difference_array = np.zeros((height, width), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            total_diff = 0
            center = rgb_array[y, x]
            
            # Check adjacent pixels (up, down, left, right)
            adjacent_positions = [
                (y-1, x),  # up
                (y+1, x),  # down
                (y, x-1),  # left
                (y, x+1)   # right
            ]
            
            for adj_y, adj_x in adjacent_positions:
                # Check if adjacent position is within bounds
                if 0 <= adj_y < height and 0 <= adj_x < width:
                    adjacent = rgb_array[adj_y, adj_x]
                    # Calculate RGB difference and add to total
                    diff = np.sum(np.abs(center - adjacent))
                    total_diff += diff
            
            difference_array[y, x] = total_diff
            
    return difference_array

def create_difference_image(rgb_array: np.ndarray, output_path: str, threshold: float = 125) -> bool:
    """
    Create and save an image based on adjacent pixel differences.
    
    Args:
        rgb_array (numpy.ndarray): 3D array containing RGB values
        output_path (str): Path where to save the processed image
        threshold (float): Threshold for binary conversion
    """
    try:
        # Calculate differences
        difference_array = calculate_adjacent_differences(rgb_array)
        
        # Normalize the differences to 0-255 range
        min_diff = np.min(difference_array)
        max_diff = np.max(difference_array)
        if max_diff > min_diff:  # Avoid division by zero
            normalized = ((difference_array - min_diff) * 655 / (max_diff - min_diff))
        else:
            normalized = np.zeros_like(difference_array)
        
        # Convert to binary based on threshold
        binary_array = np.where(normalized > threshold, 0, 255).astype(np.uint8)
        
        # Create and save image
        binary_image = Image.fromarray(binary_array, mode='L')
        binary_image.save(output_path)
        print(f"Difference-based image saved to {output_path}")
        
        # Print some statistics
        print(f"\nDifference Statistics:")
        print(f"Min difference: {min_diff:.2f}")
        print(f"Max difference: {max_diff:.2f}")
        print(f"Mean difference: {np.mean(difference_array):.2f}")
        print(f"Median difference: {np.median(difference_array):.2f}")
        
        return True
        
    except Exception as e:
        print(f"Error creating difference image: {str(e)}")
        return False

def save_rgb_array(rgb_array: np.ndarray, output_path: str, include_metadata: bool = True) -> bool:
    """
    Save RGB array to a text file with optional metadata.
    """
    try:
        with open(output_path, 'w') as f:
            if include_metadata:
                f.write(f"Image Array Metadata:\n")
                f.write(f"Dimensions: {rgb_array.shape}\n")
                f.write(f"Data Type: {rgb_array.dtype}\n")
                f.write(f"Value Range: [{np.min(rgb_array)}, {np.max(rgb_array)}]\n")
                f.write("-" * 50 + "\n\n")
            
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

# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual paths
    image_path = "/home/aashu/temp/unnamed.jpg"  # Your input image path
    rgb_txt_path = "/home/aashu/temp/rgb_values.txt"  # Where to save the RGB values text file
    difference_image_path = "/home/aashu/temp/difference_image.png"  # Where to save the processed image
    
    # Convert image to RGB array
    rgb_array = image_to_rgb_array(image_path)
    
    if rgb_array is not None:
        # Save RGB values to text file
        save_rgb_array(rgb_array, rgb_txt_path)
        
        # Create and save difference-based image
        create_difference_image(rgb_array, difference_image_path)
