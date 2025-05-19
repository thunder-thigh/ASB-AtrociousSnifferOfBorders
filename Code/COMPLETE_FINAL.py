# Check and install required packages if needed
try:
    import numpy as np
except ImportError:
    print("numpy not found. Installing numpy...")
    import subprocess
    subprocess.check_call(["pip", "install", "numpy"])
    import numpy as np

try:
    import cv2
except ImportError:
    print("OpenCV not found. Installing opencv-python...")
    import subprocess
    subprocess.check_call(["pip", "install", "opencv-python"])
    import cv2

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not found. Installing matplotlib...")
    import subprocess
    subprocess.check_call(["pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError:
    print("Pillow not found. Installing Pillow...")
    import subprocess
    subprocess.check_call(["pip", "install", "Pillow"])
    from PIL import Image

def asb_edge_detection(image_path, use_gaussian_blur=False, blur_kernel_size=5, show_steps=False, 
                 thinning_method=None, threshold_factor=1.0):
    """
    ASB: A Fast, Robust Edge Detection Algorithm implementation.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image file.
    use_gaussian_blur : bool, optional
        Whether to apply Gaussian blur pre-processing. Default is False.
    blur_kernel_size : int, optional
        Size of Gaussian blur kernel if used. Default is 5.
    show_steps : bool, optional
        Whether to display intermediate results. Default is False.
    thinning_method : str, optional
        Edge thinning method to use. Options: None, 'threshold', 'morph', 'skeleton', 'nms'
    threshold_factor : float, optional
        Factor to multiply the threshold by. Values > 1 produce thinner edges.
    
    Returns:
    --------
    tuple
        (original_image, variance_map, edge_image)
    """
    # Read the image
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")
    
    # Convert BGR to RGB format
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Pre-processing: Apply Gaussian blur if requested
    if use_gaussian_blur:
        img = cv2.GaussianBlur(original_img, (blur_kernel_size, blur_kernel_size), 0)
    else:
        img = original_img.copy()
    
    # Get image dimensions
    L, M, _ = img.shape  # L=height, M=width
    
    # Create a matrix to store variance values (V)
    variance_matrix = np.zeros((L, M), dtype=np.float64)
    
    # Calculate variance for each pixel
    for i in range(L):
        for j in range(M):
            # Define the neighborhood (handle boundary conditions)
            i_start = max(0, i-1)
            i_end = min(L, i+2)
            j_start = max(0, j-1)
            j_end = min(M, j+2)
            
            # Calculate how many neighbors are missing (for boundary correction)
            # A regular interior pixel should have 8 neighbors
            max_neighbors = 8
            actual_neighbors = (i_end - i_start) * (j_end - j_start) - 1  # -1 to exclude the pixel itself
            
            # Calculate variance
            for c in range(3):  # RGB channels
                for l in range(i_start, i_end):
                    for m in range(j_start, j_end):
                        if l == i and m == j:
                            continue  # Skip the pixel itself
                        
                        # Calculate absolute difference and add to variance
                        variance_matrix[i, j] += abs(int(img[i, j, c]) - int(img[l, m, c]))
    
    # Boundary correction factor β
    # For simplicity, we'll use a fixed multiplier for boundary pixels
    # based on how many neighbors they have
    for i in range(L):
        for j in range(M):
            # Count missing neighbors
            missing = 0
            if i == 0:  # Top edge
                missing += 3
            elif i == L-1:  # Bottom edge
                missing += 3
            
            if j == 0:  # Left edge
                missing += 3
                if i == 0 or i == L-1:
                    missing -= 1  # Avoid double counting corners
            elif j == M-1:  # Right edge
                missing += 3
                if i == 0 or i == L-1:
                    missing -= 1  # Avoid double counting corners
            
            # If this is a boundary pixel, adjust the variance
            if missing > 0:
                beta = 8 / (8 - missing)  # Correction factor
                variance_matrix[i, j] *= beta
    
    # Normalize variance matrix for visualization (0-255)
    variance_normalized = variance_matrix.copy()
    if variance_normalized.max() > 0:  # Avoid division by zero
        variance_normalized = (variance_normalized / variance_normalized.max()) * 255
    
    # Calculate threshold α (RMS of variance values)
    alpha = np.sqrt(np.mean(np.square(variance_matrix))) * threshold_factor
    
    # Apply threshold to create edge image
    edge_image = np.zeros((L, M), dtype=np.uint8)
    edge_image[variance_matrix > alpha] = 255
    
    # Apply edge thinning if requested
    if thinning_method == 'morph':
        # Morphological thinning
        kernel = np.ones((2, 2), np.uint8)
        edge_image = cv2.erode(edge_image, kernel, iterations=1)
    
    elif thinning_method == 'skeleton':
        # Skeletonization (requires scikit-image)
        try:
            from skimage.morphology import skeletonize
            edge_image_binary = edge_image > 0
            edge_image_thinned = skeletonize(edge_image_binary)
            edge_image = np.uint8(edge_image_thinned) * 255
        except ImportError:
            print("Warning: skimage not installed. Installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "scikit-image"])
            from skimage.morphology import skeletonize
            edge_image_binary = edge_image > 0
            edge_image_thinned = skeletonize(edge_image_binary)
            edge_image = np.uint8(edge_image_thinned) * 255
    
    elif thinning_method == 'nms':
        # Non-maximum suppression (simple implementation)
        temp_edges = np.zeros((L, M), dtype=np.uint8)
        for i in range(1, L-1):
            for j in range(1, M-1):
                if edge_image[i, j] > 0:  # Only process edge pixels
                    # Check if it's a local maximum in 3x3 neighborhood
                    patch = variance_matrix[i-1:i+2, j-1:j+2]
                    if variance_matrix[i, j] >= np.max(patch):
                        temp_edges[i, j] = 255
        edge_image = temp_edges
    
    # Prepare result images
    variance_img = np.uint8(variance_normalized)
    
    if show_steps:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.title("Original Image")
        plt.imshow(original_img)
        plt.axis('off')
        
        plt.subplot(132)
        plt.title("Variance Map")
        plt.imshow(variance_img, cmap='gray')
        plt.axis('off')
        
        plt.subplot(133)
        plt.title("Edge Detection Result")
        plt.imshow(edge_image, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return original_img, variance_img, edge_image

def thin_edges_demo(image_path):
    """
    Demonstrate different edge thinning methods with the ASB algorithm.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image file.
    """
    # Run ASB with different thinning methods
    original, variance, edges_normal = asb_edge_detection(
        image_path, threshold_factor=1.0, thinning_method=None)
    
    _, _, edges_threshold = asb_edge_detection(
        image_path, threshold_factor=1.5, thinning_method=None)
    
    _, _, edges_morph = asb_edge_detection(
        image_path, threshold_factor=1.0, thinning_method='morph')
    
    _, _, edges_skeleton = asb_edge_detection(
        image_path, threshold_factor=1.0, thinning_method='skeleton')
    
    _, _, edges_nms = asb_edge_detection(
        image_path, threshold_factor=1.0, thinning_method='nms')
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(231)
    plt.title("Original Image")
    plt.imshow(original)
    plt.axis('off')
    
    plt.subplot(232)
    plt.title("Standard ASB Edges")
    plt.imshow(edges_normal, cmap='gray')
    plt.axis('off')
    
    plt.subplot(233)
    plt.title("Higher Threshold (1.5x)")
    plt.imshow(edges_threshold, cmap='gray')
    plt.axis('off')
    
    plt.subplot(234)
    plt.title("Morphological Thinning")
    plt.imshow(edges_morph, cmap='gray')
    plt.axis('off')
    
    plt.subplot(235)
    plt.title("Skeletonization")
    plt.imshow(edges_skeleton, cmap='gray')
    plt.axis('off')
    
    plt.subplot(236)
    plt.title("Non-Maximum Suppression")
    plt.imshow(edges_nms, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    # Run ASB with different thinning methods
    original, variance, edges_normal = asb_edge_detection(
        image_path, threshold_factor=1.0, thinning_method=None)
    
    _, _, edges_threshold = asb_edge_detection(
        image_path, threshold_factor=1.5, thinning_method=None)
    
    _, _, edges_morph = asb_edge_detection(
        image_path, threshold_factor=1.0, thinning_method='morph')
    
    _, _, edges_skeleton = asb_edge_detection(
        image_path, threshold_factor=1.0, thinning_method='skeleton')
    
    _, _, edges_nms = asb_edge_detection(
        image_path, threshold_factor=1.0, thinning_method='nms')
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(231)
    plt.title("Original Image")
    plt.imshow(original)
    plt.axis('off')
    
    plt.subplot(232)
    plt.title("Standard ASB Edges")
    plt.imshow(edges_normal, cmap='gray')
    plt.axis('off')
    
    plt.subplot(233)
    plt.title("Higher Threshold (1.5x)")
    plt.imshow(edges_threshold, cmap='gray')
    plt.axis('off')
    
    plt.subplot(234)
    plt.title("Morphological Thinning")
    plt.imshow(edges_morph, cmap='gray')
    plt.axis('off')
    
    plt.subplot(235)
    plt.title("Skeletonization")
    plt.imshow(edges_skeleton, cmap='gray')
    plt.axis('off')
    
    plt.subplot(236)
    plt.title("Non-Maximum Suppression")
    plt.imshow(edges_nms, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    # Get ASB results
    _, _, asb_edges = asb_edge_detection(image_path, use_gaussian_blur=use_gaussian_blur)
    
    # Read image for traditional methods
    img = cv2.imread(image_path, 0)  # Read as grayscale
    
    # Apply Gaussian blur if requested
    if use_gaussian_blur:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Apply Sobel
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel = np.uint8(255 * sobel / sobel.max())
    
    # Apply Canny
    canny = cv2.Canny(img, 100, 200)
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(221)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(222)
    plt.title("ASB Edge Detection")
    plt.imshow(asb_edges, cmap='gray')
    plt.axis('off')
    
    plt.subplot(223)
    plt.title("Sobel Edge Detection")
    plt.imshow(sobel, cmap='gray')
    plt.axis('off')
    
    plt.subplot(224)
    plt.title("Canny Edge Detection")
    plt.imshow(canny, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "unnamed.png"
    
    # Run ASB edge detection with visualization
    original, variance, edges = asb_edge_detection(
        image_path, 
        show_steps=False,
        # Uncomment and modify these to try different edge thinning approaches:
         thinning_method='morph',  # Options: 'morph', 'skeleton', 'nms'
         threshold_factor=1,     # Values > 1 give thinner edges
    )
    
    # Optionally, show thinning method comparison
    # thin_edges_demo(image_path)
    
    # Optionally, compare with traditional methods
    # compare_with_traditional(image_path)
    
    # Save results
    Image.fromarray(edges).save("asb_edges.png")
    Image.fromarray(variance).save("variance_map.png")
    print("Edge detection completed. Results saved as asb_edges.png and variance_map.png")
