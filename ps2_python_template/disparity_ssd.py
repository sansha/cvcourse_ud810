import math
import numpy as np
import cv2



#cv2.matchTemplate(R_strip, tpl, method=cv2.TM_SQDIFF_NORMED)

def disparity_ssd(L, R):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """
    windowSize = 5
    disparity_window = 30
    offset = math.floor(windowSize / 2)

    R_mod = cv2.copyMakeBorder(R, offset, offset, offset, offset, cv2.BORDER_REFLECT_101).astype('float32')
    L_mod = cv2.copyMakeBorder(L, offset, offset, offset, offset, cv2.BORDER_REFLECT_101).astype('float32')
    # find patch of left image in right slice
    D = np.zeros(L.shape)
    best_matches = np.zeros(L.shape)

    for lineIdx in range(L.shape[0]):
        print("calculate line", lineIdx, "out of", L.shape[0])
        # get slice
        lineIdx += offset # image padding
        lineMin = lineIdx - offset
        lineMax = lineIdx + offset + 1
        for colIdx in range(L.shape[1]):
            colIdx += offset # padding
            disp_step = math.floor(disparity_window / 2)
            colLeftMin = colIdx - offset
            colLeftMax = colIdx + offset + 1
            colRightMin = max(0, colLeftMin - disp_step)
            colRightMax = min(R_mod.shape[1] - 1, colLeftMax + disp_step)
            right_slice = R_mod[lineMin:lineMax, colRightMin:colRightMax]

            patch_left = L_mod[lineMin:lineMax, colLeftMin:colLeftMax]
            #sq_diff = ssd_template(offset, patch_left, right_slice) # THIS IS SLOW
            sq_diff = cv2.matchTemplate(right_slice, patch_left, method=cv2.TM_SQDIFF_NORMED)

            best_match_idx = np.argmin(sq_diff) + colRightMin # offset due to right slice size
            disparity = (colIdx - offset) - best_match_idx
            best_matches[lineIdx - offset][colIdx - offset] = best_match_idx
            D[lineIdx - offset][colIdx - offset] = disparity
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


