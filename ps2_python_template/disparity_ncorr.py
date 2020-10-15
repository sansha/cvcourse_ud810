import cv2
import numpy as np
import math

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

def disparity_ncorr(L, R):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """
    windowSize = 7
    offset = math.floor(windowSize / 2)

    R_mod = cv2.copyMakeBorder(R, offset, offset, offset, offset, cv2.BORDER_CONSTANT, 0).astype('float32')
    L_mod = cv2.copyMakeBorder(L, offset, offset, offset, offset, cv2.BORDER_CONSTANT, 0).astype('float32')
    # find patch of left image in right slice
    D = np.zeros(L.shape)
    best_matches = np.zeros(L.shape)

    for lineIdx in range(L.shape[0]):
        print("calculate line", lineIdx, "out of", L.shape[0])
        # get slice
        lineIdx += offset  # image padding
        right_slice = R_mod[lineIdx - offset:lineIdx + offset + 1, :]
        left_slice = L_mod[lineIdx - offset:lineIdx + offset + 1, :]
        #cv2.imshow("twoslices", np.vstack((left_slice, right_slice)))
        #cv2.waitKey(1)
        for colIdx in range(L.shape[1]):
            colIdx += offset  # padding
            patch_left = left_slice[:, colIdx - offset:colIdx + offset + 1]
            disp = np.zeros(right_slice.shape)
            disp[0:patch_left.shape[0], colIdx - offset:colIdx + offset + 1] = patch_left
            #cv2.imshow("patch & slice", np.vstack((right_slice, disp)))
            #cv2.waitKey(1)
            #sq_diff = ssd_template(offset, patch_left, right_slice) # THIS IS SLOW
            sq_diff = cv2.matchTemplate(right_slice, patch_left, method=cv2.TM_SQDIFF_NORMED)
            best_match_idx = np.argmin(sq_diff)  # not a padded number
            disparity = (colIdx - offset) - (best_match_idx - offset)
            best_matches[lineIdx - offset][colIdx - offset] = best_match_idx
            disp_r = np.zeros(right_slice.shape)
            disp_r[:, best_match_idx:best_match_idx + patch_left.shape[1]] = right_slice[:, best_match_idx:best_match_idx + patch_left.shape[1]]
            #cv2.imshow(("patch & match"), np.vstack((right_slice, disp_r, left_slice, disp)))
            #cv2.waitKey(10)
            D[lineIdx - offset][colIdx - offset] = disparity

    return D
    # TODO: Your code here
