import numpy as np
import cv2

def calc_angle_map(grad_x, grad_y):
    assert grad_x.shape == grad_y.shape
    angels = np.zeros_like(grad_x)
    for i in grad_x.shape[0]:
        for j in grad_x.shape[1]:
            angels[i, j] = np.arctan2(grad_x[i, j], grad_y[i, j])
    return angels

def get_keypoints_from_feature_list(features, angels, size=3):
    keypoints = []
    for point in features:
        x, y = point
        keypoint = cv2.KeyPoint(x, y, size, angels[x, y], _octave=0)
        keypoints.append(keypoint)
    return keypoints