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
    desired_confidence = 0.9999
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
        matches = np.random.choice(matched_keypoints, num_samples_per_draw, replace=True)
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
            print("new outlier percentage:", outlier_percentage)
            N = calc_N(desired_confidence, outlier_percentage, num_samples_per_draw * 2)
    print("found result after", num_samples, "iterations")
    print("num inliers:", len(best_inliers), "of", len(matched_keypoints))
    return best_match, best_transform, best_inliers, best_outliers


def calc_inliers_outliers(func, matches, pointsA, pointsB, tolerance):
    inliers = set()
    outliers = set()
    for match in matches:
        expected = np.array(pointsB[match.trainIdx].pt)
        actual = np.array(func(pointsA[match.queryIdx].pt))
        distance = np.sum((expected - actual) ** 2)
        #print(distance)
        if distance < tolerance:
            inliers.add(match)
        else:
            outliers.add(match)

    return inliers, outliers


## TRANSFORM SPECIFIC FUNCTIONS

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


## SIMILARITY SPECIFIC FUNCTIONS

def get_transform_similarity(matched_keypoint, pointsA, pointsB):
    assert len(matched_keypoint) == 2
    ## 1st step: solve linear equation for 4 variables
    A = np.zeros((4, 4))
    b = np.zeros((4,1))
    points = [
        (pointsA[matched_keypoint[0].queryIdx],
         pointsB[matched_keypoint[0].trainIdx]),
        (pointsA[matched_keypoint[1].queryIdx],
         pointsB[matched_keypoint[1].trainIdx])
    ]
    ## populate matrix & vector
    for i in range(2):
        u, v = points[i][0].pt # input
        A[i * 2, :] =  [u, -v, 1, 0]
        A[i * 2 + 1, :] = [v, u, 0, 1]
        u_dash, v_dash = points[i][1].pt # expected output
        b[i * 2] = u_dash
        b[i * 2 + 1] = v_dash
    # solve
    x, residuals, rank, singular_values = np.linalg.lstsq(A, b)
    assert len(x) == 4
    ## create actual transformation matrix
    a, b, c, d = x[0][0], x[1][0], x[2][0], x[3][0]
    M = np.array([
        [a, -b, c],
        [b, a, d]
    ])
    def transform(point):
        homo_point = np.append(point, 1)
        row_vector = homo_point[np.newaxis].T
        result = M @ row_vector
        return result[0][0], result[1][0]
    #print("M in this iteration is ", M)
    return transform

def similarity_from_inliers(inliers, pointsA, pointsB):
    ## 1st step: solve linear equation for 4 variables
    A = np.zeros((len(inliers) * 2, 4))
    b = np.zeros((len(inliers) * 2, 1))
    points = []
    for inlier in inliers:
        points.append(
            (pointsA[inlier.queryIdx].pt,
             pointsB[inlier.trainIdx].pt)
        )
    ## populate matrix & vector
    for i in range(len(points)):
        u, v = points[i][0]
        A[i * 2, :] = [u, -v, 1, 0]
        A[i * 2 + 1, :] = [v, u, 0, 1]
        u_dash, v_dash = points[i][1]
        b[i * 2] = u_dash
        b[i * 2 + 1] = v_dash
    # solve
    x, residuals, rank, singular_values = np.linalg.lstsq(A, b)
    assert len(x) == 4
    ## create actual transformation matrix
    a, b, c, d = x[0][0], x[1][0], x[2][0], x[3][0]
    M = np.array([
        [a, -b, c],
        [b, a, d]
    ])

    def transform(point):
        homo_point = np.append(point, 1)
        row_vector = homo_point[np.newaxis].T
        result = M @ row_vector
        return result[0][0], result[1][0]

    return transform, M

def ransac_similarity(matched_keypoints,
                     pointsA, pointsB,
                     desired_confidence=0.99,
                     tolerance=2000):
    best_match, best_transform, inliers, outliers = ransac_get_inliers(matched_keypoints,
                                                                       pointsA,
                                                                       pointsB,
                                                                       get_transform_similarity,
                                                                       2,
                                                                       desired_confidence=desired_confidence,
                                                                       tolerance=tolerance)
    best_transform, M = similarity_from_inliers(inliers, pointsA, pointsB)
    print(M)
    inliers, outliers = calc_inliers_outliers(best_transform, matched_keypoints, pointsA, pointsB, tolerance)
    print("with the new ssd transform we get: (inliers / outliers", len(inliers), len(outliers))
    return best_transform, inliers, outliers, M