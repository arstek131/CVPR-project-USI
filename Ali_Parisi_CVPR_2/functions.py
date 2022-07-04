import cv2
import numpy as np

# Creating a mask thresholding the image
def get_image_mask(input_img, hsv_lower, hsv_upper):
	hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
	img_mask = cv2.inRange(hsv_img, hsv_lower, hsv_upper)
	return img_mask

def is_good(frame):
    ref = cv2.imread("WSC_sample.png")
    N, M = ref.shape[:2]

    lower_green = np.array([41,100,100])
    upper_green = np.array([61,255,255])
    t_frame = get_image_mask(frame, lower_green, upper_green)
    t_ref = get_image_mask(ref, lower_green, upper_green)

    mean_diff = np.sum(abs(t_frame-t_ref))/(N*M)

    MIN_VAL, MAX_VAL = 6, 11

    return MIN_VAL < mean_diff < MAX_VAL