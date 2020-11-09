import numpy as np
import cv2
import harris as h

def angle_conversion(angle_rad):
    """
    converts from numpy angle to opencv keypoint angle
    :param angle_rad: angle in radiant [-pi, pi]
    :return: angle in degree [0, 360], 0 is going right
    """
    degrees = np.degrees(angle_rad)
    return degrees if degrees >= 0 else 360 + degrees

def calc_angle_map(grad_x, grad_y):
    assert grad_x.shape == grad_y.shape
    angels = np.zeros_like(grad_x)
    for i in range(grad_x.shape[0]):
        for j in range(grad_x.shape[1]):
            angle_rad = np.arctan2(grad_y[i, j], grad_x[i, j]) # [-pi, pi]
            angels[i, j] = angle_conversion(angle_rad)
    return angels

def get_keypoints_from_feature_list(features, angels, size=3):
    keypoints = []
    for point in features:
        i, j = point
        keypoint = cv2.KeyPoint(j, i, size, angels[i, j], _octave=0)
        keypoints.append(keypoint)
    return keypoints

def do_the_keypoint_thing(fname, img):
    basename = "output/harris_imgs/" + fname + "_harris"
    harris = np.loadtxt(basename + ".txt")
    gradx = np.loadtxt(basename + "_gradx.txt")
    grady = np.loadtxt(basename + "_grady.txt")
    corner_matrix = h.find_corner_in_harris(harris, threshold=0.3)

    corner_list = h.get_list_from_corner_img(corner_matrix, thresh=0.1)
    angles = calc_angle_map(gradx, grady)
    keypoints = get_keypoints_from_feature_list(corner_list, angles, size=10)
    keypoint_img = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    outfilename = "output/keypoints/" + fname
    cv2.imwrite(outfilename + ".png", keypoint_img)

def get_sift_descriptors(fname, img):
    basename = "output/harris_imgs/" + fname + "_harris"
    harris = np.loadtxt(basename + ".txt")
    gradx = np.loadtxt(basename + "_gradx.txt")
    grady = np.loadtxt(basename + "_grady.txt")
    corner_matrix = h.find_corner_in_harris(harris, threshold=0.3)

    corner_list = h.get_list_from_corner_img(corner_matrix, thresh=0.1)
    angles = calc_angle_map(gradx, grady)
    keypoints = get_keypoints_from_feature_list(corner_list, angles, size=10)
    sift = cv2.xfeatures2d.SIFT_create()
    points, descriptors = sift.compute(img, keypoints)
    return points, descriptors

def drawMatches(img1, points1, img2, points2, matches):
    img = np.hstack((img1, img2))
    x_offset = img1.shape[1]
    for match in matches:
        p1 = tuple(map(int, points1[match.queryIdx].pt))
        p2 = tuple(map(int, points2[match.trainIdx].pt))
        p2 = p2[0] + x_offset, p2[1]
        color = tuple(np.random.choice(255, 3).tolist())
        cv2.line(img, p1, p2, color, 1)
    return img

def calc_matching_pairs(fname_pair, img_pair):
    A_points, A_descriptors = get_sift_descriptors(fname_pair[0], img_pair[0])
    B_points, B_descriptors = get_sift_descriptors(fname_pair[1], img_pair[1])

    bfm = cv2.BFMatcher()
    matches = bfm.match(A_descriptors, B_descriptors)
    ## this is cheating but check if it works
    #match_img = cv2.drawMatches(img_pair[0], A_points, img_pair[1], B_points, matches, None)
    match_img = drawMatches(img_pair[0], A_points, img_pair[1], B_points, matches)
    cv2.imwrite(fname_pair[0] + "_matches_custom.png", match_img)

