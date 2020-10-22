import numpy as np
import cv2

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def compute_gradients(img, sigma=0, filter_size=9):
    img = img.astype('float')
    blur_filter = cv2.getGaussianKernel(filter_size, sigma)
    blurred = cv2.sepFilter2D(img, -1, blur_filter, blur_filter, borderType=cv2.BORDER_REFLECT_101)
    gradient_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])
    gradient_y = np.transpose(gradient_x)
    x_img = cv2.filter2D(blurred, -1, gradient_x, borderType=cv2.BORDER_REFLECT_101)
    y_img = cv2.filter2D(blurred, -1, gradient_y, borderType=cv2.BORDER_REFLECT_101)
    return x_img, y_img
    #blurred = cv2.GaussianBlur(img, (filter_shape), sigma)

def compute_second_moment_matrix(pos, grad_x, grad_y, window_size=9):
    window = np.ones((window_size, window_size))
    offset = int(np.floor(window_size / 2))
    M = np.zeros((2,2))
    ## TODO check for corner cases / at img boundaries
    ## think about normalizing the matrix in corner cases
    ##
    for i in np.arange(pos[0] - offset, pos[1] + offset + 1):
        for j in np.arange(-offset, offset + 1):
            pass

    assert window_size % 2 == 1



def compute_harris_value(M, alpha=0.04):
    pass


def compute_harris(img):
    grad_x, grad_y = compute_gradients(img)
    harris_img = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            M = compute_second_moment_matrix((i,j), grad_x, grad_y)
            harris_img[i, j] = compute_harris_value(M)

    return harris_img