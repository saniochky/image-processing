import numpy as np
import cv2
from PIL import Image


def blur(img_path, regular=True, output_f="output.png"):
    """
    Function for blurring images using convolutions
    :param img_path: str
    :param regular: bool, if false then gaussian blur is applied
    :param output_f: str
    :return: None
    """
    regular_kernel = np.ones((11, 11), dtype="float") * (1.0 / (11 * 11))
    image = np.array(Image.open(img_path).convert('RGB'))
    if regular:
        convolved = cv2.filter2D(image, -1, regular_kernel)
    else:
        convolved = cv2.GaussianBlur(image, (11, 11), 2)

    convolved = Image.fromarray(convolved, mode="RGB")
    convolved.save(output_f)


def sharpen(img_path, output_f="output.png"):
    """
    Function for sharpening images using convolutions
    :param img_path: str
    :param output_f: str
    :return: None
    """
    image = np.array(Image.open(img_path).convert('RGB'))
    sharpen_kernel = np.array([[-0.5, -1.0, -0.5],
                               [-1.0, 7.0, -1.0],
                               [-0.5, -1.0, -0.5]])
    convolved = cv2.filter2D(image, -1, sharpen_kernel)

    convolved = Image.fromarray(convolved, mode="RGB")
    convolved.save(output_f)


def edge_detection(img_path, vertical=True, output_f="output.png"):
    """
    Function for detecting edges in images using convolutions
    :param img_path: str
    :param vertical: bool, if false then horizontal edges are detected
    :param output_f: str
    :return: None
    """
    sobel_x = np.array([[-0.125, -0.25, -0.125],
                        [0.0, 0.0, 0.0],
                        [0.125, 0.25, 0.125]])
    sobel_y = np.array([[-0.125, 0, 0.125],
                        [-0.25, 0.0, 0.25],
                        [-0.125, 0, 0.125]])
    image = np.array(Image.open(img_path).convert('RGB'))
    if vertical:
        convolved = cv2.filter2D(image, -1, sobel_y) * 4
    else:
        convolved = cv2.filter2D(image, -1, sobel_x) * 4

    convolved = Image.fromarray(convolved, mode="RGB")
    convolved.save(output_f)
