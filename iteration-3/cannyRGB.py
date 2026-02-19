import cv2
import sys
import numpy as np

# Check arguments
if len(sys.argv) < 3:
    print("Usage: python canny_color_channels.py <input_image> <output_image>")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]
single_output_path = sys.argv[3]

# Read image
img = cv2.imread(input_path)

if img is None:
    print("Error: Could not read image.")
    sys.exit(1)

cv2.imwrite(single_output_path, cv2.Canny(img, 100, 200))

# Split into B, G, R channels
b, g, r = cv2.split(img)

# Apply Canny on each channel
edges_b = cv2.Canny(b, 100, 200)
edges_g = cv2.Canny(g, 100, 200)
edges_r = cv2.Canny(r, 100, 200)

# Combine edges using OR operation
combined_edges = cv2.bitwise_or(edges_b, edges_g)
combined_edges = cv2.bitwise_or(combined_edges, edges_r)

# Save result
cv2.imwrite(output_path, combined_edges)

print(f"Output saved to {output_path}")
