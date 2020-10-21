import numpy as np

def fundamental_matrix(pts1, pts2):
    assert pts1.shape == pts2.shape
    A = np.zeros((pts1.shape[0], 9))

    for i in range(pts1.shape[0]):
        u, v = pts1[i]
        x, y = pts2[i]
        A[i] = [x * u, x * v, x, y * u, y * v, y, u, v, 1]

    u, s, vh = np.linalg.svd(A)
    M = vh[-1, :].reshape(3,3)
    return M

def reduce_rank(M):
    u, s, vh = np.linalg.svd(M)
    s[2] = 0.0 # the smallest singular value
    F = (u * s) @ vh
    return F