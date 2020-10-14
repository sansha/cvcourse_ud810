# ps2
import os
import numpy as np
import cv2

## 1-a
# Read images
L = cv2.imread(os.path.join('input', 'pair0-L.png'), 0) * (1.0 / 255.0)  # grayscale, [0, 1]
R = cv2.imread(os.path.join('input', 'pair0-R.png'), 0) * (1.0 / 255.0)

# Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
from disparity_ssd import disparity_ssd

first_exercise = False
second_exercise = True
if first_exercise:
    L = cv2.imread(os.path.join('input', 'pair0-L.png'), 0) * (1.0 / 255.0)  # grayscale, [0, 1]
    R = cv2.imread(os.path.join('input', 'pair0-R.png'), 0) * (1.0 / 255.0)
    D_L  = disparity_ssd(L, R)
    D_R = disparity_ssd(R, L)


    # TODO: Save output images (D_L as output/ps2-1-a-1.png and D_R as output/ps2-1-a-2.png)
    # Note: They may need to be scaled/shifted before saving to show results properly
    D_L_scaled = D_L * (255.0 / (np.max(D_L) - np.min(D_L)) + np.min(D_L)).astype('uint8')
    D_R_scaled = D_R * (255.0 / (np.max(D_R) - np.min(D_R)) + np.min(D_R)).astype('uint8')

    D_L_scaled = (D_L * (255.0 / 2.0)).astype('uint8')
    D_R_scaled = (D_R * (255.0 / -2.0)).astype('uint8')


    cv2.imshow("", D_L_scaled)
    cv2.waitKey(0)
    cv2.imshow("", D_R_scaled)
    cv2.waitKey(0)

    cv2.imwrite("output/ps2-1-a-3.png", D_L_scaled)
    cv2.imwrite("output/ps2-1-a-4.png", D_R_scaled)
# TODO: Rest of your code here
if second_exercise:
    L_rgb = cv2.imread(os.path.join('input', 'pair1-L.png'))   # grayscale, [0, 1]
    R_rgb = cv2.imread(os.path.join('input', 'pair1-R.png'))
    L = cv2.cvtColor(L_rgb, cv2.COLOR_BGR2GRAY) * (1.0 / 255.0)
    R = cv2.cvtColor(R_rgb, cv2.COLOR_BGR2GRAY) * (1.0 / 255.0)
    D_L = disparity_ssd(L, R)
    cv2.imshow( "actual image",D_L)
    cv2.imwrite("output/ps2-3-a-1 real picture save.png", D_L)


