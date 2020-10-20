import numpy as np

def compute_center(M):
    Q = M[:, :-1]
    m4 = M[:, -1]
    C = - np.matmul(np.linalg.inv(Q), m4)
    return C