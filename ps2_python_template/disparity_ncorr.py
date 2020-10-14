import cv2
import numpy as np
import math

def disparity_ncorr(L, R):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """
    windowSize = 10
    offset = math.floor(windowSize / 2)

    R_mod = cv2.copyMakeBorder(R, offset, offset, offset, offset, cv2.BORDER_REFLECT_101).astype('float32')
    L_mod = cv2.copyMakeBorder(L, offset, offset, offset, offset, cv2.BORDER_REFLECT_101).astype('float32')
    # find patch of left image in right slice
    D = np.zeros(L.shape)
    best_matches = np.zeros(L.shape)

    for lineIdx in range(L.shape[0]):
        print("calculate line", lineIdx, "out of", L.shape[0])
        # get slice
        lineIdx += offset  # image padding
        right_slice = R_mod[lineIdx - offset:lineIdx + offset + 1, :]
        left_slice = L_mod[lineIdx - offset:lineIdx + offset + 1, :]
        for colIdx in range(L.shape[1]):
            colIdx += offset  # padding
            patch_left = left_slice[:, colIdx - offset:colIdx + offset + 1]
            # sq_diff = ssd_template(offset, patch_left, right_slice) # THIS IS SLOW
            sq_diff = cv2.matchTemplate(right_slice, patch_left, method=cv2.TM_CCORR_NORMED)
            best_match_idx = np.argmax(sq_diff)  # not a padded number
            disparity = (colIdx - offset) - best_match_idx
            best_matches[lineIdx - offset][colIdx - offset] = best_match_idx
            D[lineIdx - offset][colIdx - offset] = disparity
    return D
    # TODO: Your code here
