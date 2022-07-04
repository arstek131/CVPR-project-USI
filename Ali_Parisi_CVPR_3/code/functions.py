import cv2
import numpy as np

# Given a frame, draw the (connected) border of the table
def get_pockets(frame):
    G = preprocessing_corner(frame)

    # Find the contours, take the one with minimum area and find the rectangle that best approximate that contour
    contours, _ = cv2.findContours(G, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    rbox = cv2.minAreaRect(cnt)
    pts = cv2.boxPoints(rbox).astype(np.int32)

    # Compute the position of the two side pockets
    p1, p2, p3, p4 = pts
    if distance(p1,p2) > distance(p2,p3):
        spt1, spt2 = (p1+p2)/2, (p3+p4)/2
    else:
        spt1, spt2 = (p2+p3)/2, (p4+p1)/2
    spt1, spt2 = np.array(spt1.astype(np.int)), np.array(spt2.astype(np.int))
    pts = np.append(pts, [spt1, spt2],axis=0)

    return (pts, rbox[1])


def preprocessing_corner(img):
    ## Convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Blur the image
    blur = cv2.blur(hsv, (10,10))

    # Threshold on the green
    lower_green = np.array([53, 0, 0])
    upper_green = np.array([65, 255, 255])
    mask = cv2.inRange(blur, lower_green, upper_green)

    #Erosion and dilation (in order to smooth the contours)
    kernel = np.ones((31, 31), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=2)
    kernel = np.ones((17, 17), np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    # Canny edge detection
    canny = cv2.Canny(dilation, 100, 200)

    # Dilation (in order to thicken the contours)
    kernel = np.ones((7, 7), np.uint8)
    canny = cv2.dilate(canny, kernel, iterations=1)

    return canny

def find_balls(img):
    G = preprocessing_balls(img)
    # Find circle using hough transform
    circles = cv2.HoughCircles( image=G,
                                method=cv2.HOUGH_GRADIENT,
                                dp=1,
                                minDist=20,
                                param1=10,
                                param2=10,
                                minRadius=5,
                                maxRadius=10)

    if circles is None:
        return []

    circles = np.uint16(np.around(circles))[0]
    return [[x,y] for [x,y,_] in circles]


def preprocessing_balls(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(hsv)
    s = s**3 /(v**2+1)
    s = np.clip(s,0,255)
    hsv = cv2.merge([h,s,v])
    
    # Threshold
    lower_red = np.array([0,255,93])
    upper_red = np.array([12,255,255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Erosion and dilation (in order to clean up the mask)
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=1)

    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations=2)

    # Canny edge detection
    canny = cv2.Canny(dilation, 50, 200)

    # Dilation (in order to thicken the contours)
    kernel = np.ones((3, 3), np.uint8)
    canny = cv2.dilate(canny, kernel, iterations=1)

    return canny



def check_score(pockets, balls):
    ERR = 20
    for ball in balls:
        for pocket in pockets:
            if distance(ball,pocket) < ERR:
                return (pocket, ball)
    return False

def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
