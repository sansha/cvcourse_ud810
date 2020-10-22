import harris as h
import cv2
import numpy as np

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
    #grad_x, grad_y = h.normalize(grad_x), h.normalize(grad_y)
    transA_gradient_pair = np.hstack((transA_grad_x, transA_grad_y))
    simA_gradient_pair = np.hstack(simA_grads)
    simA_grad_scaled = h.normalize(simA_gradient_pair)
    transA_grad_scaled = h.normalize(transA_gradient_pair)
    save_normalized("output/transA-gradient_pair.png", transA_grad_scaled)
    save_normalized("output/simA-gradient_pair.png", simA_grad_scaled)

def task1_b():
    transA = cv2.imread("input/transA.jpg")
    simA = cv2.imread("input/simA.jpg")
    check = cv2.imread("input/check.bmp")
    ## CREATE GRADIENT PAIRS ##################################################

    transA_harris = h.compute_harris(transA)
    save_normalized("transA_harris.png", transA_harris)

task1_a()
task1_b()
