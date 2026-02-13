import cv2
import numpy as np

def calculate_MEAN_RMS(image):
    dimensions=image.shape
    RMS=0
    MEAN=0
    for h in range(1, dimensions[0]-1):
        for w in range(1, dimensions[1]-1):
            RMS=RMS+(int(image[h,w])*int(image[h,w])/(dimensions[0]*dimensions[1]))
            MEAN=MEAN+(image[h,w]//(dimensions[0]*dimensions[1]))
    #RMS=int((RMS**(0.5))//(dimensions[0]*dimensions[1]))
    #MEAN=(MEAN//(h*w))
    return MEAN, RMS

def apply_thresholding(image, thresholds):
    thresholds=(thresholds[0], thresholds[1])
    dimensions=image.shape
    thresholded_image=np.zeros(dimensions, dtype=np.uint8)
    print(image.dtype)
    for h in range(1, dimensions[0]-1):
        for w in range(1, dimensions[1]-1):
            if (thresholds[0]<=image[h,w]<=thresholds[1]):
                thresholded_image[h,w]=255
            else:
                thresholded_image[h,w]=0
    return thresholded_image

def apply_mean_std_threshold(image, k=2.0):
    img = image.astype(np.float64)
    mean = np.mean(img)
    std  = np.std(img)
    T = mean + k * std
    print(f"Mean: {mean:.2f}, Std: {std:.2f}, Threshold: {T:.2f}")
    binary = np.zeros_like(img, dtype=np.uint8)
    binary[img >= T] = 255
    return binary

def thin_image(image):
    img = image
    return (cv2.ximgproc.thinning(img))

def apply_thin_dilate_erode(image):
    '''
    #Order: opening, closing, erode
    kernel = np.ones((3,3),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.erode(image,kernel,iterations = 1)
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


def apply_difference_kernel(image_input, channel):
#    image=cv2.GaussianBlur(image_input, (3,3), 0)
    if channel not in [0,1,2]:
        print("Error specify correct channel for kernelling")
        return image_input
    image=image_input
    dimensions = image.shape
    print(dimensions, channel)
    #print(image.dtype)
    #Output is single channel
    kernelled_image = np.zeros((dimensions[0], dimensions[1]), dtype=np.int32)
    for h in range(1, dimensions[0] - 1):
        for w in range(1, dimensions[1] - 1):
            c = int(image[h, w, channel])
            kernelled_image[h, w] = (
                abs(c - int(image[h+1, w+1, channel])) +
                abs(c - int(image[h-1, w+1, channel])) +
                abs(c - int(image[h+1, w-1, channel])) +
                abs(c - int(image[h-1, w-1, channel])))
    # Normalize for display (since int32 won't show properly)
    #display_img = cv2.normalize(kernelled_image, None, 0, 255, cv2.NORM_MINMAX)
    #display_img = display_img.astype(np.uint8)
    #cv2.imshow("Kernelled Image", display_img)
    #cv2.waitKey(0)
    return kernelled_image

def apply_difference_kernel_fast(image_input, channel):
    #Slop from GPT, I am ashamed to say this smaller func just werks, and faster than mine :-(
    img = image_input[:, :, channel].astype(np.int32)
    center = img[1:-1, 1:-1]
    br = img[2:, 2:]
    tr = img[:-2, 2:]
    bl = img[2:, :-2]
    tl = img[:-2, :-2]
    result = (
        np.abs(center - br) +
        np.abs(center - tr) +
        np.abs(center - bl) +
        np.abs(center - tl)
    )
    return result

def resize_image(image, ratio):
    height, width = image.shape[:2]
    dsize = (width//ratio, height//ratio)
    resized_image = cv2.resize(image, dsize, interpolation=cv2.INTER_AREA)
    return resized_image
