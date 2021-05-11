import numpy
import cv2
from PIL import Image


def image2matrix(path: str, rgb: bool = False):
    if rgb:
        image = numpy.array(Image.open(path).convert('RGB'))
        return (image,
                numpy.matrix(image[:, :, :1].reshape(image.shape[0], -1)),
                numpy.matrix(image[:, :, 1:2].reshape(image.shape[0], -1)),
                numpy.matrix(image[:, :, 2:].reshape(image.shape[0], -1)))
    else:
        return numpy.matrix(Image.open(path).convert('L'))


def rank_r_approx(image, r, rgb: bool = False):
    if not rgb:
        matrix = image2matrix(image)

        # performing SVD factorization on a matrix
        u, sigma, v_trans = numpy.linalg.svd(matrix, full_matrices=True)

        # making an approximation by dropping the corresponding terms
        # in the singular value decomposition of matrix
        sigma_r = numpy.diag(sigma[:r])
        u_r = u[0:len(u), 0:r]
        v_trans_r = v_trans[0:r, 0:len(v_trans)]

        # performing dot product on three matrices U, Sigma and V transposed
        sigma_v_trans_dot = numpy.dot(sigma_r, v_trans_r)
        result = numpy.dot(u_r, sigma_v_trans_dot).astype(int).clip(min=0, max=255)
        matrix[:] = result[:]
        return matrix
    else:
        matrix, red, green, blue = image2matrix(image, rgb=True)

        # performing SVD factorization on a matrix
        u_blue, sigma_blue, v_trans_blue = numpy.linalg.svd(blue, full_matrices=True)
        u_green, sigma_green, v_trans_green = numpy.linalg.svd(green, full_matrices=True)
        u_red, sigma_red, v_trans_red = numpy.linalg.svd(red, full_matrices=True)

        # making an approximation by dropping the corresponding terms
        # in the singular value decomposition of matrix
        sigma_r_blue, sigma_r_green, sigma_r_red = numpy.diag(sigma_blue[:r]), numpy.diag(sigma_green[:r]), \
                                                   numpy.diag(sigma_red[:r])
        u_r_blue, u_r_green, u_r_red = u_blue[0:len(u_blue), 0:r], u_green[0:len(u_green), 0:r], u_red[0:len(u_red),
                                                                                                 0:r]
        v_trans_r_blue, v_trans_r_green, v_trans_r_red = v_trans_blue[0:r, 0:len(v_trans_blue)], \
                                                         v_trans_green[0:r, 0:len(v_trans_green)], \
                                                         v_trans_red[0:r, 0:len(v_trans_red)]

        # performing dot product on three matrices U, Sigma and V transposed
        sigma_v_trans_dot_blue, sigma_v_trans_dot_green, sigma_v_trans_dot_red = numpy.dot(sigma_r_blue,
                                                                                           v_trans_r_blue), \
                                                                                 numpy.dot(sigma_r_green,
                                                                                           v_trans_r_green), \
                                                                                 numpy.dot(sigma_r_red, v_trans_r_red)

        result_blue, result_green, result_red = numpy.dot(u_r_blue, sigma_v_trans_dot_blue), \
                                                numpy.dot(u_r_green, sigma_v_trans_dot_green), \
                                                numpy.dot(u_r_red, sigma_v_trans_dot_red)

        result = cv2.merge((result_red, result_green, result_blue)).astype(int).clip(min=0, max=255)
        matrix[:] = result[:]

        return matrix


def matrix2image(mat, filename: str = 'output.png', rgb: bool = False):
    mode = ('L', 'RGB')[rgb]
    image = Image.fromarray(mat, mode=mode)
    image.save(filename)


def approx(input_f: str, output_f: str, rank, rgb: bool = False):
    matrix2image(rank_r_approx(input_f, rank, rgb=rgb), output_f, rgb=rgb)
