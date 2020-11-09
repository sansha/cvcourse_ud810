import numpy as np
import cv2
import random


## COMMON RANSAC FUNCTIONS

def calc_N(confidence, outlier_percentage, num_model_params):
    denominator = np.log(1 - ((1 - outlier_percentage) ** num_model_params))
    if denominator == 0:
        return np.inf
    return np.log(1 - confidence) / denominator


def ransac_get_inliers(matched_keypoints,
                       pointsA,
                       pointsB,
                       get_transform,
                       num_samples_per_draw,
                       desired_confidence=0.99,
                       tolerance=2):
    """

    :param matched_keypoints: the matched keypoints from sift descriptors
    :param get_transform: a function that returns a transformation given the samples
    :param num_samples_per_draw: 2 for a transform, 4 for similarity
    :param desired_confidence: the desired p, defines number of samples drawn
    :param tolerance: defines when a point is considered inlier
    :return: best match, inliers, outliers
    """
    desired_confidence = 0.99
    outlier_percentage = 1.0  # init with worst case
    num_samples = 0
    best_match = None
    best_inliers = {}
    best_outliers = {}
    best_transform = None
    inliers, outliers = {}, {}
    N = np.inf
    while N > num_samples:
        # get samples
        matches = np.random.choice(matched_keypoints, num_samples_per_draw, replace=False)
        # calc transform
        transform = get_transform(matches, pointsA, pointsB)
        # calc inliers / outliers
        inliers, outliers = calc_inliers_outliers(transform, matched_keypoints, pointsA, pointsB, tolerance)

        num_samples += 1
        # if we have a new best, update best values
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_outliers = outliers
            best_match = matches
            best_transform = transform
            # update N
            outlier_percentage = len(outliers) / (len(inliers) + len(outliers))
            N = calc_N(desired_confidence, outlier_percentage, 2)

    return best_match, best_transform, best_inliers, best_outliers


def calc_inliers_outliers(func, matches, pointsA, pointsB, tolerance):
    inliers = set()
    outliers = set()
    for match in matches:
        expected = np.array(pointsB[match.trainIdx].pt)
        actual = np.array(func(pointsA[match.queryIdx].pt))
        distance = np.sum((expected - actual) ** 2)
        print(distance)
        if distance < tolerance:
            inliers.add(match)
        else:
            outliers.add(match)

    return inliers, outliers


def get_transform_transformation(matched_keypoint, pointsA, pointsB):
    assert len(matched_keypoint) == 1
    p1 = pointsA[matched_keypoint[0].queryIdx]
    p2 = pointsB[matched_keypoint[0].trainIdx]
    offset_x = p2.pt[0] - p1.pt[0]
    offset_y = p2.pt[1] - p1.pt[1]

    def transform(point):
        return point[0] - offset_x, point[1] - offset_y

    return transform


def ransac_transform(matched_keypoints,
                     pointsA, pointsB,
                     desired_confidence=0.99,
                     tolerance=2000):
    best_match, best_transform, inliers, outliers = ransac_get_inliers(matched_keypoints,
                                                                       pointsA,
                                                                       pointsB,
                                                                       get_transform_transformation,
                                                                       1,
                                                                       desired_confidence=desired_confidence,
                                                                       tolerance=tolerance)
    ## TODO calculate overall best match
    return best_transform, inliers, outliers
