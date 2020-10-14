import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

img = cv2.imread("input/surfing.jpg", cv2.IMREAD_REDUCED_COLOR_2)
img2 = cv2.imread("input/4.2.06.tiff")
#cv2.imshow("image", img)
cv2.imwrite("smaller.jpg", img)
print(img.shape)
swapped = np.copy(img)
swapped[:,:,0] = img[:,:,2]
swapped[:,:,2] = img[:,:,0]
cv2.imshow("colors changed", swapped)

mono_g = img[:,:,1]
cv2.imshow("mono g", mono_g)

mono1 = img[:,:,2]
mono2 = img2[:,:,2]

img1_center_x = int(img.shape[1] / 2)
img1_center_y = int(img.shape[0] / 2)

center100 = img[img1_center_y - 50:img1_center_y + 50, img1_center_x - 50:img1_center_x + 50]
img2_c_x = int(img2.shape[1] / 2)
img2_c_y = int(img2.shape[0] / 2)
mixed = np.copy(img2)
mixed[img2_c_y - 50:img2_c_y + 50, img2_c_x - 50: img2_c_x + 50] = center100
cv2.imwrite("output/mixed.jpg", mixed)

### exercise 4
min, max, mean = np.min(mono1), np.max(mono1), np.mean(mono1),
std = np.std(mono1)
print(min, max, mean)
changed = np.copy(mono1).astype('float')
changed = (changed - mean) / std * 10 + mean
min2, max2 = changed.min(), changed.max()
print(min2, max2)
cv2.imshow("changed", changed)

shifted = np.zeros(mono1.shape).astype('uint8')
shift_count = 1
shifted[:,:-shift_count] = np.copy(mono1)[:,shift_count:]
cv2.imshow("shifted", shifted)
cv2.imwrite("output/shifted.jpg", shifted)
#cv2.imshow("mono r", mono2)
subtracted = mono1 - shifted
cv2.imshow("subtracted", subtracted)
cv2.imwrite("output/subtracted_shift.jpg", shifted)

#### exercise 5 NOISE
sigma = 10
mean = 0
noise = np.random.normal(mean, sigma, (img.shape[0], img.shape[1]))
print(noise.shape)
noisy = img.astype('float')
cv2.imshow("green noise", noisy)
print(noise.min(), noise.max(), noise.mean())
noisy[:,:,1] +=  noise
noisy = cv2.normalize(noisy, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imshow("green noise", noisy)
cv2.imwrite("output/noise_on_green.jpg", noisy)
noisy_b = img.astype('float')
noisy_b[:,:,0] += noise
noisy_b = cv2.normalize(noisy_b, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imshow("blue noise", noisy_b)
cv2.imwrite("output/noise_on_blue.jpg", noisy_b)



cv2.waitKey(0)
#plt.imshow(img)
#plt.show()