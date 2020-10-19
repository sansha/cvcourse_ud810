import math
import numpy as np
import cv2


# cv2.matchTemplate(R_strip, tpl, method=cv2.TM_SQDIFF_NORMED)

def disparity_ncorr(L, R):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))

    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """
    visualization = False
    windowSize = 7
    disparity_window = 150
    offset = math.floor(windowSize / 2)

    R_mod = cv2.copyMakeBorder(R, offset, offset, offset, offset, cv2.BORDER_CONSTANT, 0).astype('float32')
    L_mod = cv2.copyMakeBorder(L, offset, offset, offset, offset, cv2.BORDER_CONSTANT, 0).astype('float32')
    # find patch of left image in right slice
    D = np.zeros(L.shape)
    best_matches = np.zeros(L.shape)
    cv2.namedWindow('patches', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('patches', 512, 512)

    for lineIdx in range(L.shape[0]):
        print("calculate line", lineIdx, "out of", L.shape[0])
        # get slice
        lineIdx += offset  # image padding
        lineMin = lineIdx - offset
        lineMax = lineIdx + offset + 1
        for colIdx in range(L.shape[1]):
            colIdx += offset  # padding
            disp_step = math.floor(disparity_window / 2)
            colLeftMin = colIdx - offset
            colLeftMax = colIdx + offset + 1
            colRightMin = max(0, colLeftMin - disp_step)
            colRightMax = min(R_mod.shape[1], colLeftMax + disp_step)
            right_slice = R_mod[lineMin:lineMax, colRightMin:colRightMax]

            patch_left = L_mod[lineMin:lineMax, colLeftMin:colLeftMax]

            # sq_diff = ssd_template(offset, patch_left, right_slice) # THIS IS SLOW
            sq_diff = cv2.matchTemplate(right_slice, patch_left, method=cv2.TM_SQDIFF)

            c_tf = max(colIdx - colRightMin - offset, 0)
            dist = np.arange(sq_diff.shape[1]) - c_tf
            # cost = sq_diff + np.abs(dist * lambda_factor)
            _, _, _, max_loc = cv2.minMaxLoc(sq_diff)
            if visualization:
                disp = np.zeros((right_slice.shape[0], R_mod.shape[1]))
                disp[:, colRightMin:colRightMax] = right_slice
                disp_left = np.zeros((patch_left.shape[0], R_mod.shape[1]))
                disp_left[:, colLeftMin:colLeftMax] = patch_left
                disp_selected_patch = np.zeros((patch_left.shape[0], R_mod.shape[1]))
                x_val = dist[max_loc[0]] + colLeftMin
                disp_selected_patch[:, x_val:x_val + offset * 2 + 1] = right_slice[:,
                                                                       min_loc[0]:min_loc[0] + patch_left.shape[1]]

                normalizedD = cv2.normalize(D[0:lineMin + 1, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                            dtype=cv2.CV_8U)
                cv2.imshow("patches", np.vstack((normalizedD,
                                                 R_mod[lineMin:lineMax, offset:-offset],
                                                 L_mod[lineMin:lineMax, offset:-offset],
                                                 disp_left[:, offset: -offset],
                                                 disp[:, offset:-offset],
                                                 disp_selected_patch[:, offset:-offset])))
                cv2.waitKey(1)
            D[lineIdx - offset][colIdx - offset] = dist[max_loc[0]]

            # best_match_idx = np.argmin(sq_diff) + colRightMin # offset due to right slice size
            # disparity = (colIdx - offset) - best_match_idx
            # best_matches[lineIdx - offset][colIdx - offset] = best_match_idx
            # D[lineIdx - offset][colIdx - offset] = disparity
    return D

    # TODO: Your code here


def ssd_template(offset, patch_left, right_slice):
    sq_diff = []
    for matchingIdx in range(right_slice.shape[1] - offset * 2):
        matchingIdx += offset  # padding
        patch_right = right_slice[:, matchingIdx - offset:matchingIdx + offset + 1]
        ssd = np.sum((patch_left - patch_right) ** 2)
        # if colIdx == 20 and lineIdx == 20:
        #   cv2.imshow("patch left", patch_left)
        # cv2.waitKey(0)
        #  cv2.imshow("patch right", patch_right)
        # cv2.waitKey(0)
        sq_diff.append(ssd)
    return sq_diff


