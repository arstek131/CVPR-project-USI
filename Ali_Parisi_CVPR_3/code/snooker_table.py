import cv2
from functions import *
import os

def main():

    videos = ['normal','rotated-010','rotated-025','rotated-045']
    GAP = 30
    C = 16

    for video in videos:
        cap = cv2.VideoCapture(f'{video}.mp4')
        os.chdir(f'./results/{video}')
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        last_score = GAP
        num_score = 0

        for i in range(length): 

            last_score += 1
            # Take each frame
            _, frame = cap.read()
            # In the first frame, compute the contour of the table and pocket positions
            if i == 0:
                pockets, dim = get_pockets(frame)
                RECT_AREA = dim[0]*dim[1]

            overlay = frame.copy()

            # Draw outline of the table
            cv2.drawContours(overlay, [pockets[:4]], -1, (255, 0, 0), 5, cv2.LINE_AA)
            # Mark pokets
            for pocket in pockets:
                cv2.circle(overlay, pocket, 30, (0,255,255), -1)
            
            # Take all the ball detected that are in the contour of the table
            balls = [b for b in find_balls(frame)]
            for ball in balls:
                x,y = ball
                cv2.rectangle(overlay, (x-C,y-C),(x+C,y+C), (192,15,252),3)

            if last_score > GAP:
                score = check_score(pockets, balls)
                if score:
                    last_score = 0
                    num_score += 1
                    pocket, _ = score
                    tmp = frame.copy()
                    cv2.circle(tmp,(pocket[0],pocket[1]), 40,(0,0,255),4)
                    cv2.imwrite(f'frame_{i}.jpg', tmp)

            final = cv2.addWeighted(frame,0.3, overlay, 0.7, gamma=0)
            cv2.imshow('final',final)

            # Stop video conditions
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

        print(f"video: {video}, total score: {num_score}")
        # Release everything if job is finished
        cap.release()
        cv2.destroyAllWindows()
        os.chdir('../..')


if __name__ == "__main__":
    main()