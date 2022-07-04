import numpy as np
import cv2

def preprocessing_corner(img):
    M, N = img.shape[0], img.shape[1]
    DIM_KERNEL = 31

    ## Convert to HSV and isolate H layer
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Threshold on the green
    lower_green = np.array([53, 0, 0])
    upper_green = np.array([65, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    #res = cv2.bitwise_and(img, img, mask=mask)

    # Erosion and dilation (in order to smooth the contours)
    kernel = np.ones((DIM_KERNEL, DIM_KERNEL), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)


    canny = cv2.Canny(dilation, 100, 200)
    kernel = np.ones((7, 7), np.uint8)
    canny = cv2.dilate(canny, kernel, iterations=1)

    return canny

def find_corners(img):

    G = preprocessing_corner(img)

    contours, _ = cv2.findContours(G, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    # Approximate the founded contour to a quadrilateral
    poly = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    inner_corner = [[x,y,1] for [[x,y]] in poly]
    return inner_corner