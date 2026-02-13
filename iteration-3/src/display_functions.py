import cv2
import numpy as np


def display_image(image, color):
	if color==0:
		cv2.imshow("Blue Channel", image[:,:,0])
	elif color==1:
		cv2.imshow("Green Channel", image[:,:,1])
	elif color==2:
		cv2.imshow("Red Channel", image[:,:,2])
	elif color==3:
		cv2.imshow("All Channel", image)
	else:
		print("Error specify correct channel type")
	cv2.waitKey(0)

def display_npint32_image(name_of_window, image):
	display_image=cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
	display_image=display_image.astype(np.uint8)
	cv2.imshow(name_of_window, display_image)
	cv2.waitKey(0)
