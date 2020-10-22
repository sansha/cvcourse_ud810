import numpy as np
import os
from least_square_matrix import least_square_matrix
from project import project, project_many
from compute_center import compute_center


def compute_residuals(pts1, pts2):
    diff = pts1 - pts2
    sq_diff = np.square(diff)
    sum = np.sum(sq_diff, axis=1)
    # print(sum.shape)
    sqrt = np.sqrt(sum)
    # print(sqrt)
    return sqrt


def task1():
    points2d = np.loadtxt("input/pts2d-norm-pic_a.txt")
    points3d = np.loadtxt("input/pts3d-norm.txt")
    M = least_square_matrix(points2d, points3d)
    print(M)
    C = compute_center(M)
    print(C)
    example3d = [1.2323, 1.4421, 0.4506]
    example2d = project(M, example3d)
    # print(example2d)

    projected2d = project_many(M, points3d)
    sqrt = compute_residuals(projected2d, points2d)
    np.savetxt("output/M_norm_pic_a.txt", M)
    np.savetxt("output/projected_norm_pic_a.txt", projected2d)
    np.savetxt("output/residuals_norm_pic_a.txt", sqrt)


## second task
def task2(num_points):
    points2d_all = np.loadtxt("input/pts2d-pic_b.txt")
    points3d_all = np.loadtxt("input/pts3d.txt")
    curr_avg = np.inf
    bestM = []
    all_residuals = np.zeros(10)
    for i in range(10):
        sampled_idx = np.random.choice(points2d_all.shape[0], num_points, replace=False)
        points2d = points2d_all[sampled_idx, :]
        points3d = points3d_all[sampled_idx, :]
        M = least_square_matrix(points2d, points3d)
        unused_idx = np.delete(np.arange(0, points3d_all.shape[0]), sampled_idx)
        test_idx = np.random.choice(unused_idx, 4, replace=False)
        # print(sampled_idx)
        # print(test_idx)
        testpts2d = points2d_all[test_idx, :]
        testpts3d = points3d_all[test_idx, :]
        project2d = project_many(M, testpts3d)
        residuals = compute_residuals(testpts2d, project2d)
        avg_res = np.average(residuals)
        all_residuals[i] = avg_res
        # print(avg_res)
        if (avg_res < curr_avg):
            # print("new best!")
            bestM = M
            curr_avg = avg_res
    return bestM, all_residuals


# task1()

# Ms = []
# for num_points in [8, 12, 16]:
#     bestM, residuals = task2(num_points=8)
#     C = compute_center(bestM)
#     print(C)
#  #   Ms.append(bestM)
#     #np.savetxt("output/M_pic_best_" + str(num_points) + ".txt", bestM)
#     #np.savetxt("output/residuals_pic_best_" + str(num_points) + ".txt", residuals)
#
# points2d_all = np.loadtxt("input/pts2d-pic_b.txt")
# points3d_all = np.loadtxt("input/pts3d.txt")
# M = least_square_matrix(points2d_all, points3d_all)
# C = compute_center(M)
# print(C)
from fundamental_matrix import fundamental_matrix, reduce_rank
import cv2


def to_normal(epipoint):
    assert epipoint.shape == (3,)
    return (epipoint[0] / epipoint[2]), (epipoint[1] / epipoint[2])

def to_int(normalpoint):
    rounded = np.around(normalpoint)
    return int(rounded[0]), int(rounded[1])

def to_drawable(epipoint):
    return to_int(to_normal(epipoint))

def task3():
    ptsA = np.loadtxt("input/pts2d-pic_a.txt")
    ptsB = np.loadtxt("input/pts2d-pic_b.txt")

    num_points = 20  # ptsA.shape[0]

    M = fundamental_matrix(ptsA, ptsB)
    print(M)
    print(np.linalg.matrix_rank(M))

    F = reduce_rank(M)
    print(F)
    print(np.linalg.matrix_rank(F))
    epipolar_lines_b = np.zeros((num_points, 3))  # store epipolar lines
    epipolar_lines_a = np.zeros((num_points, 3))  # store epipolar lines

    for i in range(num_points):
        epipolar_lines_b[i] = F @ np.append(ptsA[i], 1)  # from a to b
        epipolar_lines_a[i] = np.transpose(F) @ np.append(ptsB[i], 1)  # from b to a
    img_a = cv2.imread("input/pic_a.jpg")
    img_b = cv2.imread("input/pic_b.jpg")

    print(img_a.shape)
    wrong_axis = True
    if not wrong_axis:
        P_UL = [0, 0, 1]
        P_BL = [img_a.shape[0] - 1, 0, 1]
        P_UR = [1, img_a.shape[1] - 1, 1]
        P_BR = [img_a.shape[0] - 1, img_a.shape[1] - 1, 1]
    else:
        P_UL = [0, 0, 1]
        P_BL = [0, img_a.shape[0] - 1, 1]
        P_UR = [img_a.shape[1] - 1, 0, 1]
        P_BR = [img_a.shape[1] - 1, img_a.shape[0] - 1, 1]

    l_L = np.cross(P_UL, P_BL)
    l_R = np.cross(P_UR, P_BR)
    window_name = 'Image'
    green = (0, 255, 0)
    red = (0, 0, 255)
    for i in range(num_points):
        pt_left = np.cross(epipolar_lines_a[i], l_L)
        pt_right = np.cross(epipolar_lines_a[i], l_R)
        offset = 40
        ptl = to_drawable(pt_left)
        ptr = to_drawable(pt_right)
        adjusted_pt_l = (ptl[0], ptl[1] - offset)
        adjusted_pt_r = (ptr[0], ptr[1] - offset)
        cv2.line(img_a, ptr, ptl, green, thickness=1)

        point = (int(ptsA[i][0]), int(ptsA[i][1]),)
        cv2.circle(img_a, point, 5, red, thickness=4)

    for i in range(num_points):
        pt_left = np.cross(epipolar_lines_b[i], l_L)
        pt_right = np.cross(epipolar_lines_b[i], l_R)
        ptl = to_drawable(pt_left)
        ptr = to_drawable(pt_right)
        cv2.line(img_b, ptl, ptr, green, thickness=1)

        point = (int(ptsB[i][0]), int(ptsB[i][1]))
        cv2.circle(img_b, point, 5, red, thickness=4)

    cv2.imshow(window_name, img_a)
    cv2.imwrite("output/pic_a_lines.png", img_a)
    cv2.waitKey(0)
    cv2.imshow(window_name, img_b)
    cv2.imwrite("output/pic_b_lines.png", img_b)
    cv2.waitKey(0)


task3()

# reference M
# points2d = np.loadtxt("input/pts2d-pic_b.txt")
# points3d = np.loadtxt("input/pts3d.txt")
# M = least_square_matrix(points2d, points3d)
# for testM in Ms:
#     sqdiff = np.square(testM) - np.square(M)
#     print(sqdiff)
#     sqrtsum = np.sum(sqdiff)
#     print(sqrtsum)

# 8 max / min: 0.023870663523427538, 0.0024005544074070387
# 12 max min: 0.009588077965169034, 0.0020345500539973482
# 16 max, min: 0.01775494910102892, 0.0023477473260722316
