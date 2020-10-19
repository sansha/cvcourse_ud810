import numpy as np
import os
from least_square_matrix import least_square_matrix
if __name__ == "__main__":
    file2d = open("input/pts2d-norm-pic_a.txt")
    points2d = np.loadtxt("input/pts2d-norm-pic_a.txt")
    points3d = np.loadtxt("input/pts3d-norm.txt")
    M = least_square_matrix(points2d, points3d)
    print(M)

