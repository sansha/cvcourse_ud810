import math
import numpy as np
import cv2
def disparity_ssd(L, R):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """
    windowSize = 5
    offset = math.floor(windowSize / 2)

    R_mod = cv2.copyMakeBorder(R, offset, offset, offset, offset, cv2.BORDER_REFLECT_101)
    L_mod = cv2.copyMakeBorder(L, offset, offset, offset, offset, cv2.BORDER_REFLECT_101)
    # find patch of left image in right slice
    D = np.zeros(L.shape)
    best_matches = np.zeros(L.shape)

    for lineIdx in range(L.shape[0]):
        print("calculate line", lineIdx, "out of", L.shape[0])
        # get slice
        lineIdx += offset # image padding
        right_slice = R_mod[lineIdx - offset:lineIdx + offset + 1, :]
        left_slice = L_mod[lineIdx - offset:lineIdx + offset + 1, :]
        for colIdx in range(L.shape[1]):
            colIdx += offset # padding
            patch_left = left_slice[:, colIdx - offset:colIdx + offset + 1]
            sq_diff = []
            for matchingIdx in range(R.shape[1]):
                matchingIdx += offset # padding
                patch_right = right_slice[:, matchingIdx - offset:matchingIdx + offset + 1]
                ssd = np.sum((patch_left - patch_right) ** 2)
                #if colIdx == 20 and lineIdx == 20:
                 #   cv2.imshow("patch left", patch_left)
                   # cv2.waitKey(0)
                  #  cv2.imshow("patch right", patch_right)
                   # cv2.waitKey(0)
                sq_diff.append(ssd)
            best_match_idx = np.argmin(sq_diff) # not a padded number
            disparity = (colIdx - offset) - best_match_idx
            best_matches[lineIdx - offset][colIdx - offset] = best_match_idx
            D[lineIdx - offset][colIdx - offset] = disparity
    return D




    # TODO: Your code here


