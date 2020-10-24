import numpy as np
import cv2

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def save_normalized(filename, img, float_normalize=True):
    if float_normalize:
        norm = normalize(img)
    else:
        norm = img
    cv2.imwrite(filename, (norm * 255.0).astype('uint8'))


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
    assert window_size % 2 == 1

    #window = np.ones((window_size, window_size))
    gaussian = cv2.getGaussianKernel(window_size, 0)
    window = np.transpose(gaussian) * gaussian
    offset = int(np.floor(window_size / 2))
    M = np.zeros((2,2))
    grad_x_pad = cv2.copyMakeBorder(grad_x, offset, offset, offset, offset, cv2.BORDER_CONSTANT, value=0)
    grad_y_pad = cv2.copyMakeBorder(grad_y, offset, offset, offset, offset, cv2.BORDER_CONSTANT, value=0)

    for i in np.arange(0, window_size):
        for j in np.arange(0, window_size):
            I_x = grad_x_pad[i + pos[0], i + pos[1]]
            I_y = grad_y_pad[i + pos[0], i + pos[1]]
            M_temp = np.array([
                [I_x ** 2, I_x * I_y],
                [I_x * I_y, I_y ** 2]
            ])
            M = M + (window[i, j] * M_temp)

    return M




def compute_harris_value(M, alpha=0.04):
    det = np.linalg.det(M)
    trace = np.trace(M)
    return det - alpha * (trace ** 2)


def compute_harris(img, sigma=0, gauss_window=9, moment_window=9):
    grad_x, grad_y = compute_gradients(img, sigma, gauss_window)
    harris_img = np.zeros_like(img).astype('float')
    for i in range(img.shape[0]):
        print("computing line ", i + 1, " of ", img.shape[0])
        for j in range(img.shape[1]):
            M = compute_second_moment_matrix((i,j), grad_x, grad_y, moment_window)
            harris_img[i, j] = compute_harris_value(M)

    return harris_img

def non_max_suppression(img, radius=9):
    assert radius % 2 == 1
    offset = int(np.floor(radius / 2))
    padded = cv2.copyMakeBorder(img, offset, offset, offset, offset, cv2.BORDER_CONSTANT, value=0)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i:i+radius, j:j+radius]
            if window.max() > img[i, j]:
                img[i, j] = 0
    return img

def find_corner_in_harris(img, threshold=0.1, radius=9):
    img = normalize(img)
    _, thresh = cv2.threshold(img.astype('float32'), threshold, 1.0, cv2.THRESH_TOZERO)
    #save_normalized("thresh.png", thresh)
    max = non_max_suppression(thresh, radius=radius)
    #save_normalized("max.png", max)
    return max

