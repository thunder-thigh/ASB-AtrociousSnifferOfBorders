import numpy as np
import cv2
import display_functions as DF
import compute_functions as CF
import sys




import cv2
import numpy as np

def canny_on_difference_channels(image):
    # Your custom difference kernel outputs (int32)
    BKI = CF.apply_difference_kernel_fast(image, 0)
    GKI = CF.apply_difference_kernel_fast(image, 1)
    RKI = CF.apply_difference_kernel_fast(image, 2)
    # Normalize to 0–255 and convert to uint8 (required for Canny)
    BKI_8 = cv2.normalize(BKI, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    GKI_8 = cv2.normalize(GKI, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    RKI_8 = cv2.normalize(RKI, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Apply Canny
    B_edges = cv2.Canny(BKI_8, 100, 100)
    G_edges = cv2.Canny(GKI_8, 100, 100)
    R_edges = cv2.Canny(RKI_8, 100, 100)
    # Combine edges (OR is better than + for binary images)
    final_edges = cv2.bitwise_or(B_edges, G_edges)
    final_edges = cv2.bitwise_or(final_edges, R_edges)
    # Display
    cv2.imshow("Final Combined Canny", final_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return final_edges

def display(image):
    height, width = image.shape[:2]
    dsize = (width//4, height//4)
    print(dsize)
    print(image.dtype)
    #image = cv2.resize(image, dsize, interpolation=cv2.INTER_AREA)
    #DF.display_image(image, 0)
    #DF.display_image(image, 1)
    #DF.display_image(image, 2)
    DF.display_image(image, 3)
    #CF.apply_difference_kernel(image, 0)
    #CF.apply_difference_kernel(image, 1)
    #CF.apply_difference_kernel(image, 2)
    #DF.display_image(CF.apply_difference_kernel(image, 0), 3)
    #DF.display_image(CF.apply_difference_kernel(image, 1), 3)
    #DF.display_image(CF.apply_difference_kernel(image, 2), 3)
    cv2.waitKey(0)

def asb3(image):
#   XKI = X kernelled image
#   MEAN before RMS since MEAN<=RMS is always true
    image=CF.resize_image(image, 1)
    #image=cv2.GaussianBlur(image, (5, 5), 0)
    canny_on_difference_channels(image)
    DF.display_image(image, 3)
    BKI=CF.apply_difference_kernel_fast(image, 0)
    GKI=CF.apply_difference_kernel_fast(image, 1)
    RKI=CF.apply_difference_kernel_fast(image, 2)
    DF.display_npint32_image("BKI", BKI)
    DF.display_npint32_image("GKI", GKI)
    DF.display_npint32_image("RKI", RKI)
    all_channel_kernelled_image=RKI+GKI+BKI
    DF.display_npint32_image('All channel kernelling', all_channel_kernelled_image)
    (MEAN, RMS)=CF.calculate_MEAN_RMS(all_channel_kernelled_image)
    #final_image=CF.apply_thresholding(all_channel_kernelled_image, (MEAN, RMS))
    final_image=CF.apply_mean_std_threshold(all_channel_kernelled_image, k=2)
    final_image_cleaned=all_channel_kernelled_image*final_image
    #CF.apply_thin_dilate_erode(image)
    DF.display_npint32_image('final ASB image', final_image)
    DF.display_npint32_image('final cleaned image', final_image_cleaned-all_channel_kernelled_image)
    #thinned ~vomit~
    #DF.display_npint32_image('final ASB image', CF.thin_image(final_image))
    cv2.waitKey(0)
    
image = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
#image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
if image is None:
    print("Error image not loaded")
else:
    #display(image)
    asb3(image)
    cv2.imwrite("canny.jpg", cv2.Canny(image, 100, 200))
    cv2.waitKey(0)
