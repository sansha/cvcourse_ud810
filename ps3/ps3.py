import numpy as np
import os
from least_square_matrix import least_square_matrix
from project import project
if __name__ == "__main__":
    file2d = open("input/pts2d-norm-pic_a.txt")
    points2d = np.loadtxt("input/pts2d-norm-pic_a.txt")
    points3d = np.loadtxt("input/pts3d-norm.txt")
    M = least_square_matrix(points2d, points3d)
    print(M)
    example3d = [ 1.2323, 1.4421, 0.4506]
    example2d = project(M, example3d)
    print(example2d)

