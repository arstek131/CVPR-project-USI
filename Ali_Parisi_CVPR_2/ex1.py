import cv2
import numpy as np
from functions import *


def main():

    cap = cv2.VideoCapture('WSC.mp4')                   # framerate = 30 
    width  = cap.get(3)  # float `width`
    height = cap.get(4)  # float `height`
    size = (int(width/2), int(height/2))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(f'filtered.mp4', fourcc, 30, size)

    black = np.zeros((int(size[1]), int(size[0]), 1), dtype = "uint8")
    start_frame = 30 * 60 * 40
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    count = 0
    for i in range(30*60): 
        _, frame = cap.read()

        # if the frame contains the snooker table, show it
        if is_good(frame):
            # Show the frames
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            cv2.imshow('frame',frame)
            out.write(frame)
            count += 1
        else:
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            cv2.imshow('frame', black)

        cv2.imshow('original', frame)
        # Stop video conditions
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    print(f"Number of good frames: {count}")


    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()