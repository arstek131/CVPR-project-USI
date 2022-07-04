import cv2
import numpy as np
from functions import *

def main():

    cap = cv2.VideoCapture('sample.mp4')
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(length): 
    # Skip odd frames
        if(i%2 != 0):
            continue
        # Take each frame
        _, frame = cap.read()

        tmp = frame.copy()
        # Draw the contour of the table
        boundaries = draw_contour(frame, color=(0,0,255))
        # Draw circle and center of red balls
        draw_balls(tmp, frame, boundaries)
        # Resize the frame
        frame=cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        # Show the frame
        cv2.imshow('frame',frame)
        # Stop video conditions
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()