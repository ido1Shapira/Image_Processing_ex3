import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def gaussian_Kernel(kernel_size: int):
    sigma = int(round(0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8))
    g_kernel = cv2.getGaussianKernel(kernel_size, sigma)
    g_kernel = g_kernel * g_kernel.transpose()
    return g_kernel


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[y,x]...], [[dU,dV]...] for each points
    """
    assert (win_size % 2 == 1)
    assert (im1.shape == im2.shape)

    Gx = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    Gy = Gx.transpose()
    w = win_size // 2
    Ix = cv2.filter2D(im2, -1, Gx, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(im2, -1, Gy, borderType=cv2.BORDER_REPLICATE)
    It = im2 - im1

    u_v = []
    j_i = []
    k = 0
    for i in range(step_size, im1.shape[0], step_size):
        for j in range(step_size, im1.shape[1], step_size):

            Nx = Ix[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Ny = Iy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Nt = It[i - w:i + w + 1, j - w:j + w + 1].flatten()

            A = np.array([[sum(Nx[k] ** 2 for k in range(len(Nx))), sum(Nx[k] * Ny[k] for k in range(len(Nx)))],
                          [sum(Nx[k] * Ny[k] for k in range(len(Nx))), sum(Ny[k] ** 2 for k in range(len(Ny)))]])

            b = np.array([[-1 * sum(Nx[k] * Nt[k] for k in range(len(Nx))),
                           -1 * sum(Ny[k] * Nt[k] for k in range(len(Ny)))]]).reshape(2, 1)

            ev1, ev2 = np.linalg.eigvals(A)
            if ev2 < ev1:  # sort them
                temp = ev1
                ev1 = ev2
                ev2 = temp
            if ev2 >= ev1 > 1 and ev2 / ev1 < 100:  # check the conditions
                velo = np.dot(np.linalg.pinv(A), b)
                u = velo[0][0]
                v = velo[1][0]
                u_v.append(np.array([u, v]))
            else:
                k += 1
                # print('ev1: {0} ev2: {1}', ev1, ev2, k)
                u_v.append(np.array([0.0, 0.0]))

            j_i.append(np.array([j, i]))
    return np.array(j_i), np.array(u_v)

def blurImage2(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param kernel_size: Kernel size
    :return: The Blurred image
    """
    assert (kernel_size % 2 == 1)
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    in_image = cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    in_image = cv2.filter2D(in_image, -1, np.transpose(kernel), borderType=cv2.BORDER_REPLICATE)
    return in_image

def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    img = img[0: np.power(2, levels) * int(img.shape[0] / np.power(2, levels)),
            0: np.power(2, levels) * int(img.shape[1] / np.power(2, levels))]

    temp_img = img.copy()
    pyr = [temp_img]
    for i in range(levels - 1):
        temp_img = blurImage2(temp_img, 5)
        temp_img = temp_img[::2, ::2]
        pyr.append(temp_img)
    return pyr


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    expand = np.zeros((img.shape[0] * 2, img.shape[1] * 2))
    expand[::2, ::2] = img
    expand = cv2.filter2D(expand, -1, gs_k, borderType=cv2.BORDER_REPLICATE)
    return expand

def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    pyr = []
    g_ker = gaussian_Kernel(5)
    g_ker *= 4
    gaussian_pyr = gaussianPyr(img, levels)
    for i in range(levels - 1):
        extend_level = gaussExpand(gaussian_pyr[i + 1], g_ker)
        lap_level = gaussian_pyr[i] - extend_level
        pyr.append(lap_level.copy())
    pyr.append(gaussian_pyr[-1])
    return pyr


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    pyr_updated = lap_pyr.copy()
    guss_k = gaussian_Kernel(5) * 4
    cur_layer = lap_pyr[-1]
    for i in range(len(pyr_updated) - 2, -1, -1):
        cur_layer = gaussExpand(cur_layer, guss_k) + pyr_updated[i]
    return cur_layer


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return:  (Naive blend, Blended Image)
    """
    assert(img_1.shape == img_2.shape)

    img_1 = img_1[0: np.power(2, levels) * int(img_1.shape[0] / np.power(2, levels)),
            0: np.power(2, levels) * int(img_1.shape[1] / np.power(2, levels))]
    img_2 = img_2[0: np.power(2, levels) * int(img_2.shape[0] / np.power(2, levels)),
            0: np.power(2, levels) * int(img_2.shape[1] / np.power(2, levels))]
    mask = mask[0: np.power(2, levels) * int(mask.shape[0] / np.power(2, levels)),
            0: np.power(2, levels) * int(mask.shape[1] / np.power(2, levels))]

    im_blend = np.zeros(img_1.shape)
    if len(img_1.shape) == 3 or len(img_2.shape) == 3:  # the image is RGB
        for color in range(3):
            part_im1 = img_1[:, :, color]
            part_im2 = img_2[:, :, color]
            part_mask = mask[:, :, color]
            im_blend[:, :, color] = pyrBlend_helper(part_im1, part_im2, part_mask, levels)

    else:  # the image is grayscale
        im_blend = pyrBlend_helper(img_1, img_2, mask, levels)

    # Naive blend
    naive_blend = mask * img_1 + (1 - mask) * img_2

    return naive_blend, im_blend

def pyrBlend_helper(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> np.ndarray:
    """
        Blends two images using PyramidBlend method
        :param img_1: Image 1
        :param img_2: Image 2
        :param mask: Blend mask
        :param levels: Pyramid depth
        :return:  Blended Image
        """
    L1 = laplaceianReduce(img_1, levels)
    L2 = laplaceianReduce(img_2, levels)
    Gm = gaussianPyr(mask, levels)
    Lout = []
    for k in range(levels):
        curr_lup = Gm[k] * L1[k] + (1 - Gm[k]) * L2[k]
        Lout.append(curr_lup)
    im_blend = laplaceianExpand(Lout)
    im_blend = np.clip(im_blend, 0, 1)  # check if need this

    return im_blend
