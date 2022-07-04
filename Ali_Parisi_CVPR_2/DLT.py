import numpy as np
import cv2
from find_corners import *



def calculate(x_i, X_i):
    x = np.zeros((2, 12))
    x[0, 4:8] = X_i * -1
    x[0, 8:12] = x_i[1] * X_i
    x[1, 0:4] = X_i
    x[1, 8:12] = -x_i[0] * X_i
    return x

def svd_decomposition(A):
    _, _, vh = np.linalg.svd(A, full_matrices=True)
    p = vh.T[:,-1] 
    p = p.reshape((3, 4))
    print(f"p matrix:\n{p}\n")
    return p

def main():
    img = cv2.imread('WSC_sample.png', cv2.IMREAD_COLOR)
    M, N, _ = img.shape
    # Automatically found the inner corners of the snooker table
    inner_corners = find_corners(img)

    # Manually found the outer corners of the snooker table and the non-red balls
    outer_corners = [[903, 55, 1],
                    [1026, 610, 1],
                    [255, 610, 1],
                    [378, 55, 1]]
    balls = [   [548, 143, 1],
                [726, 143, 1],
                [638, 143, 1],
                [640, 288, 1],
                [638, 433, 1],
                [638, 544, 1]]

    # 3d points homogeneous coordinates of the corners and the non-red balls
    X = np.array([
        [0.889, 1.7845, 0, 1],
        [0.889, -1.7845, 0, 1],
        [-0.889, -1.7845, 0, 1],
        [-0.889, 1.7845, 0, 1],

        [0.894, 1.8345, 0.04, 1],
        [0.894, -1.8345, 0.04, 1],
        [-0.894, -1.8345, 0.04, 1],
        [-0.894, 1.8345, 0.04, 1], 
        
        [-0.292, 1.0475, 0, 1],
        [0.292, 1.0475, 0, 1],
        [0, 1.0475, 0, 1],
        [0, 0, 0, 1],
        [0, -0.89225, 0, 1],
        [0, -1.4605, 0, 1],
    ])

    # corresponding 2d points homogeneous coordinates of the corners and the non-red balls
    x = np.array([*inner_corners, *outer_corners, *balls])

    A = None
    for x_i, X_i in zip(x, X):
        if A is None:
            A = calculate(x_i, X_i)
        else:
            A = np.concatenate((A, calculate(x_i, X_i)))

    p = svd_decomposition(A)


    K, R, _, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    print(f"Calibration matrix K = \n{K}\n")
    print(f"Rotation matrix R = \n{R}\n")

    C = np.matmul(-np.linalg.inv(p[:,:-1]), p[:,3])
    C = np.append(C,1)
    print(f"Camera matrix C = \n{C}\n")

    # We can check if the computation was right, checking if p*C = 0
    print(f"Check: {np.matmul(p,C)}")

    for point in x:
        x0,y0,_ = point
        cv2.circle(img, (x0,y0), 4, (0, 0, 255), 10)
    cv2.imshow('result', img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()