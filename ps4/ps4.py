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
    recompute = False

    transA = cv2.imread("input/transA.jpg")
    transB = cv2.imread("input/transB.jpg")
    simA = cv2.imread("input/simA.jpg")
    simB = cv2.imread("input/simB.jpg")

    check = cv2.imread("input/check.bmp")
    ## CREATE GRADIENT PAIRS ##################################################
    transA = cv2.cvtColor(transA, cv2.COLOR_BGR2GRAY)
    transB = cv2.cvtColor(transB, cv2.COLOR_BGR2GRAY)
    simA = cv2.cvtColor(simA, cv2.COLOR_BGR2GRAY)
    simB = cv2.cvtColor(simB, cv2.COLOR_BGR2GRAY)
    if recompute:
        transA_harris = h.compute_harris(transA)
        np.savetxt("output/transA_harris.txt", transA_harris)
        h.save_normalized("output/transA_harris.png", transA_harris)
        ##
        transB_h = h.compute_harris(transB)
        np.savetxt("output/transB_harris.txt", transB_h)
        h.save_normalized("output/transB_harris.png", transB_h)
        ##
        simA_h = h.compute_harris(simA)
        np.savetxt("output/simA_harris.txt", simA_h)
        h.save_normalized("output/simA_harris.png", simA_h)
        ##
        simB_h = h.compute_harris(simB)
        np.savetxt("output/simB_harris.txt", simB_h)
        h.save_normalized("output/simB_harris.png", simB_h)
    else:
        transA_harris = np.loadtxt("output/transA_harris.txt")
        transB_h = np.loadtxt("output/transB_harris.txt")
        simA_h = np.loadtxt("output/simA_harris.txt")
        simB_h = np.loadtxt("output/simB_harris.txt")

    transA_corner_matrix = h.find_corner_in_harris(transA_harris)
    composite_transA = np.zeros(transA.shape + (3,)).astype('float')
    composite_transA[:,:,0] = transA
    composite_transA[:,:, 1] = transA_corner_matrix
    h.save_normalized("output/tansA_composite.png",composite_transA)
    print("bla")

    #transB_harris = h.compute_harris(transB)

#task1_a()
task1_b()
