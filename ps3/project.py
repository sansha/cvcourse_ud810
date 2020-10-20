import numpy as np

def project(M, point3d):
    assert M.shape == (3,4)
    homogeneous3d = np.append(point3d, [1])
    u, v, w = np.matmul(M, homogeneous3d)
    point2d = np.array([u / w, v / w ])
    return point2d

def project_many(M, points3d):
    assert M.shape == (3,4)
    points2d = np.zeros((points3d.shape[0], 2))

    for i in range(points3d.shape[0]):
        x, y, z = points3d[i]
        u, v, w = np.matmul(M, [x, y, z, 1])
        points2d[i] = [u / w, v / w]
    return points2d

