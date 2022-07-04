import cv2
import numpy as np


# Given a frame, draw the (connected) border of the table
def draw_contour(frame, color=(255,0,0), thickness=5):
    bord = detect_border(frame)
    # Find the corner using the lines
    edges = []
    for i in range(4):
        edges.append(find_intersection(bord[i-1], bord[i]))
    for i in range(4):
        cv2.line(frame, edges[i-1], edges[i], color, thickness)
    return edges



# Given a frame, return the four edges of the table
def detect_border(image):
    blurred = cv2.blur(image,(20,20))

    lower_green = np.array([41,100,100])
    upper_green = np.array([61,255,255])
    threshed = get_image_mask(blurred, lower_green, upper_green)
    canny = cv2.Canny(threshed, 100, 200)
    kernel = np.ones((7,7), np.uint8)
    canny = cv2.dilate(canny, kernel, 1)

    linesP = cv2.HoughLinesP(   image=canny, 
                                rho=1, 
                                theta=np.pi/360, 
                                threshold=400, 
                                maxLineGap=200, 
                                minLineLength=100)
    candidate = [None for _ in range(4)]                    # [LEFT, UP, RIGHT, DOWN]
    err = 20
    m = 500
    for line in linesP:                                    # line: [x1, y1, x2, y2]
        # if we have all the desired candidate, we don't need to search for others
        if all([x is not None for x in candidate]):
            break
        line = line[0]
        x1,y1,x2,y2 = line
        # if x1=x2 and x1 < M, we found the left side (same reasoning for the other cases)
        if abs(x1-x2) < err:
            if (candidate[0] is None) and x1 <= m:                  # left
                candidate[0] = line
            elif (candidate[2] is None) and x1 > m:                 # right
                candidate[2] = line
        elif abs(y1-y2) < err:
            if (candidate[1] is None) and y1 <= m:                  # down
                candidate[1] = line
            elif (candidate[3] is None) and y1 > m:                 # up
                candidate[3] = line
    return candidate

# Given two segment (definded by a couple of points), find the intersection of the corresponding line 
def find_intersection(l1, l2):
    m1,b1 = find_eq(l1)
    m2,b2 = find_eq(l2)
    x = (b2-b1)/(m1-m2)
    y = m1*x+b1
    return (int(x),int(y))

# Given two points, find the equation (y=mx+b)
def find_eq(l):
    x1,y1,x2,y2 = l
    m = (y2-y1)/(x2-x1)
    b = y1 - m*x1
    return (m,b)


# Creating a mask thresholding the image
def get_image_mask(input_img, hsv_lower, hsv_upper):
	hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
	img_mask = cv2.inRange(hsv_img, hsv_lower, hsv_upper)
	return img_mask

# Given a frame, detect and draw red balls
def draw_balls(frame, original, edge, color=(255,0,0), thickness=7):

    def on_table(p):
        # Given a point p, check if it's on the table
        x,y = p
        _,upleft,_,downright = edge
        return (x > upleft[0] and x < downright[0] and y > upleft[1] and y < downright[1])

    # Convert the frame from BGR to HSV
    imghsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(imghsv)
    s = s**3 /(v**2+1)
    s = np.clip(s,0,255)
    imghsv = cv2.merge([h,s,v])
    frame = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    # define range of red ball color in LAB
    lower_red = np.array([0,100,100])
    upper_red = np.array([10,255,255])
    # Threshold
    threshed = get_image_mask(frame, lower_red, upper_red)
    # Apply Canny edge detector
    canny = cv2.Canny(threshed, 100, 200)
    # Find circle using hough transform
    circles = cv2.HoughCircles( image=canny,
                                method=cv2.HOUGH_GRADIENT,
                                dp=1,
                                minDist=20,
                                param1=10,
                                param2=10,
                                minRadius=6,
                                maxRadius=10)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # if the circle is not on the table, skip it
        if on_table((i[0],i[1])):
            # draw the outer circle
            #cv2.circle(original,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(original,(i[0],i[1]),2,color,thickness)