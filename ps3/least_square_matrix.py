import numpy as np

def least_square_matrix(points2d, points3d):
    #print("heavy algorithm")

    A = np.zeros((points2d.shape[0] * 2, 12))
    for i in range(points2d.shape[0]):
        X, Y, Z = points3d[i][0], points3d[i][1], points3d[i][2]
        u, v = points2d[i][0], points2d[i][1]
        A[2 * i] =   [X, Y, Z, 1, 0, 0, 0, 0, - u * X, - u * Y, - u * Z, -u]
        A[2 * i+1] = [0, 0, 0, 0, X, Y, Z, 1, - v * X, - v * Y, - v * Z, -v]

    u, s, vh = np.linalg.svd(A)
    m = np.zeros((12, 1))
    m[11][0] = 1
    M = vh[-1, :].reshape((3,4))
    M = M * -1
    return M