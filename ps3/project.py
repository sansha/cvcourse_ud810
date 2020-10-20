import numpy as np

def project(M, point3d):
    assert M.shape == (3,4)
    homogeneous3d = np.append(point3d, [1])
    u, v, w = np.matmul(M, homogeneous3d)
    point2d = np.array([u / w, v / w ])
    return point2d


