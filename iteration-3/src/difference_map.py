import cv2
import numpy as np

def compute_neighbourhood_difference(image):
    h, w = image.shape
    image = image.astype(np.int32)
    diff_image = np.zeros((h,w), dtype=np.int32)
    neighbours = [
            (-1,-1), (-1,0), (-1,1),
            (0,-1), (0,1),
            (1,-1),(1,0),(1,1)
            ]
    for y in range(h):
        for x in range(w):
            centre=image[y,x]
            total_diff=0
            for dy,dx in neighbours:
                ny,nx = y+dy, x+dx
                if 0<= ny < h and 0 <= nx <w:
                    total_diff +=abs(centre - image[ny, nx])
            diff_image[y,x]=total_diff
        return diff_image
