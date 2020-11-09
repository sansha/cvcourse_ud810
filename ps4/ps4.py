import harris as h
import cv2
import numpy as np
import sift
import ransac


def save_normalized(filename, img, float_normalize=True):
    if float_normalize:
        norm = h.normalize(img)
    else:
        norm = img
    cv2.imwrite(filename, (norm * 255.0).astype('uint8'))


def task1_a():
    transA = cv2.imread("input/transA.jpg")
    simA = cv2.imread("input/simA.jpg")
    check = cv2.imread("input/check.bmp")
    ## CREATE GRADIENT PAIRS ##################################################
    transA_grad_x, transA_grad_y = h.compute_gradients(transA, filter_size=5)
    simA_grads = h.compute_gradients(simA, filter_size=5)
    # grad_x, grad_y = h.normalize(grad_x), h.normalize(grad_y)
    transA_gradient_pair = np.hstack((transA_grad_x, transA_grad_y))
    simA_gradient_pair = np.hstack(simA_grads)
    simA_grad_scaled = h.normalize(simA_gradient_pair)
    transA_grad_scaled = h.normalize(transA_gradient_pair)
    save_normalized("output/transA-gradient_pair.png", transA_grad_scaled)
    save_normalized("output/simA-gradient_pair.png", simA_grad_scaled)


## after some empirical tests, I found the best result is generated for
# gauss window size 5
# harris moment window size 7
# todo: find corner in harris threshold & radios tests
gauss_window_size = 5
h_moment_window_size = 7


def task1_b():
    recompute = False
    check = cv2.imread("input/check.bmp")
    check_rot = cv2.imread("input/check_rot.bmp")
    transA = cv2.imread("input/transA.jpg")
    transB = cv2.imread("input/transB.jpg")
    simA = cv2.imread("input/simA.jpg")
    simB = cv2.imread("input/simB.jpg")

    check = cv2.imread("input/check.bmp")
    ## CREATE GRADIENT PAIRS ##################################################
    check = cv2.cvtColor(check, cv2.COLOR_BGR2GRAY)
    check_rot = cv2.cvtColor(check_rot, cv2.COLOR_BGR2GRAY)
    transA = cv2.cvtColor(transA, cv2.COLOR_BGR2GRAY)
    transB = cv2.cvtColor(transB, cv2.COLOR_BGR2GRAY)
    simA = cv2.cvtColor(simA, cv2.COLOR_BGR2GRAY)
    simB = cv2.cvtColor(simB, cv2.COLOR_BGR2GRAY)

    fnames = ["check", "check_rot", "transA", "transB", "simA", "simB"]
    input = [check, check_rot, transA, transB, simA, simB]
    if recompute:

        for i in range(4, len(fnames)):
            filename = "output/" + fnames[i] + "_harris"
            print("calculate ", filename)
            harris, gradx, grady = h.compute_harris(input[i], gauss_window=gauss_window_size,
                                                    moment_window=h_moment_window_size)
            np.savetxt(filename + ".txt", harris)
            np.savetxt(filename + "_gradx.txt", gradx)
            np.savetxt(filename + "_grady.txt", grady)
            h.save_normalized(filename + ".png", harris)
        ##
        # transB_h = h.compute_harris(transB)
        # np.savetxt("output/transB_harris.txt", transB_h)
        # h.save_normalized("output/transB_harris.png", transB_h)
        # ##
        # simA_h = h.compute_harris(simA)
        # np.savetxt("output/simA_harris.txt", simA_h)
        # h.save_normalized("output/simA_harris.png", simA_h)
        # ##
        # simB_h = h.compute_harris(simB)
        # np.savetxt("output/simB_harris.txt", simB_h)
        # h.save_normalized("output/simB_harris.png", simB_h)
    else:
        transA_harris = np.loadtxt("output/transA_harris_final.txt")

        # transB_h = np.loadtxt("output/transB_harris.txt")
        # simA_h = np.loadtxt("output/simA_harris.txt")
        # simB_h = np.loadtxt("output/simB_harris.txt")

    # for thresh in [0.3, 0.5, 0.7]:
    #     transA_corner_matrix = h.find_corner_in_harris(transA_harris, threshold=thresh, radius=3)
    #     composite_transA = np.zeros(transA.shape + (3,)).astype('float')
    #     composite_transA[:,:,0] = cv2.normalize(transA_harris, None, 0, 255, cv2.NORM_MINMAX2)
    #     _, composite_transA[:,:, 1] = cv2.threshold((transA_corner_matrix * 255.0).astype('uint8'), 1, 255, cv2.THRESH_BINARY)
    #     fname = "output/tansA_composite_thresh_nonmax" + str(thresh)
    #     h.save_normalized(fname + ".png", composite_transA)
    #     print("bla")

    # transB_harris = h.compute_harris(transB)


# def mark_harris_corners(fname, threshold, radius):
#     basename = "output/harris_imgs/" + fname + "_harris"
#     harris = np.loadtxt(basename + ".txt")
#     gradx = np.loadtxt(basename + "_gradx.txt")
#     grady = np.loadtxt(basename + "_grady.txt")
#
#     corner_matrix = h.find_corner_in_harris(harris, threshold=threshold, radius=radius)
#     raw_img = cv2.imread("input/" + fname + ".jpg")
#     _, corners = cv2.threshold((corner_matrix * 255.0).astype('uint8'), 1, 255, cv2.THRESH_BINARY)
#     raw_img //= 2
#     raw_img[:,:, 1] += corners
#     cv2.imwrite("output/" + fname + "_harris_marked.png", corners)


def task2_a():
    check = cv2.imread("input/check.bmp")
    check_rot = cv2.imread("input/check_rot.bmp")
    transA = cv2.imread("input/transA.jpg")
    transB = cv2.imread("input/transB.jpg")
    simA = cv2.imread("input/simA.jpg")
    simB = cv2.imread("input/simB.jpg")
    fnames = ["check", "check_rot", "transA", "transB", "simA", "simB"]
    input = [check, check_rot, transA, transB, simA, simB]
    for i in range(len(fnames)):
        sift.do_the_keypoint_thing(fnames[i], input[i])


def task2_b():
    check = cv2.imread("input/check.bmp")
    check_rot = cv2.imread("input/check_rot.bmp")
    transA = cv2.imread("input/transA.jpg")
    transB = cv2.imread("input/transB.jpg")
    simA = cv2.imread("input/simA.jpg")
    simB = cv2.imread("input/simB.jpg")
    fname_pairs = [("check", "check_rot"), ("transA", "transB"), ("simA", "simB")]
    input_pairs = [(check, check_rot), (transA, transB), (simA, simB)]
    for i in range(len(fname_pairs)):
        sift.calc_matching_pairs(fname_pairs[i], input_pairs[i], draw_pairs=True)


def task3_a():
    transA = cv2.imread("input/transA.jpg")
    transB = cv2.imread("input/transB.jpg")
    matches, A_points, B_points = sift.calc_matching_pairs(("transA", "transB"), (transA, transB))
    np.save("output/matches/trans.txt", matches)
    np.save("output/matches/transA_points", A_points)
    np.save("output/matches/transB_points", B_points)
    best_transform, inliers, outliers = ransac.ransac_transform(matches, pointsA=A_points, pointsB=B_points)
    match_img = cv2.drawMatches(transA, A_points, transB, B_points, list(inliers), None)
    cv2.imwrite("output/ransac/transA_inliers.png", match_img)
    match_img = cv2.drawMatches(transA, A_points, transB, B_points, list(outliers), None)
    cv2.imwrite("output/ransac/transA_outliers.png", match_img)


def task3_a2():  # TODO
    transA = cv2.imread("input/check.bmp")
    transB = cv2.imread("input/check_rot.bmp")
    matches, A_points, B_points = sift.calc_matching_pairs(("check", "check_rot"), (transA, transB))
    np.save("output/matches/check.txt", matches)
    np.save("output/matches/check_points", A_points)
    np.save("output/matches/check_rot_points", B_points)
    best_transform, inliers, outliers = ransac.ransac_transform(matches, pointsA=A_points, pointsB=B_points,
                                                                tolerance=5000)
    match_img = cv2.drawMatches(transA, A_points, transB, B_points, list(inliers), None)
    cv2.imwrite("output/ransac/check_inliers.png", match_img)
    match_img = cv2.drawMatches(transA, A_points, transB, B_points, list(outliers), None)
    cv2.imwrite("output/ransac/check_outliers.png", match_img)


def task3_b():
    recalc = True
    transA = cv2.imread("input/simA.jpg")
    transB = cv2.imread("input/simB.jpg")
    if recalc:
        matches, A_points, B_points = sift.calc_matching_pairs(("simA", "simB"), (transA, transB))
        np.save("output/matches/sim", matches)
        np.save("output/matches/simA_points", A_points)
        np.save("output/matches/simB_points", B_points)
    else:
        # m#atches  = np.load("output/matches/sim.npy")
        A_points = np.load("output/matches/simA_points")
        B_points = np.load("output/matches/simB_points")

    for tolerance in [10000, 5000, 4000, 3000, 2000, 1000, 500, 200]:
        best_transform, inliers, outliers = ransac.ransac_similarity(matches, pointsA=A_points, pointsB=B_points,
                                                                    tolerance=tolerance)
        match_img = cv2.drawMatches(transA, A_points, transB, B_points, list(inliers), None)
        cv2.imwrite("output/ransac/simA_inliers" + str(tolerance) + ".png", match_img)
        match_img = cv2.drawMatches(transA, A_points, transB, B_points, list(outliers), None)
        cv2.imwrite("output/ransac/simB_outliers" + str(tolerance) + ".png", match_img)


# task1_a()
# task2_b()
task3_b()
