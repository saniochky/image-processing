import pandas as pd
from math import pow, log10
import numpy
from rank_approx import rank_r_approx, image2matrix


def frobenius_norm(matrix, approx_matrix):
    return numpy.linalg.norm(matrix - approx_matrix, ord='fro', keepdims=False)


def mean_squared_error(matrix, approx_matrix):
    m, n = matrix.shape[0], matrix.shape[1]
    result = 0
    for row_ind in range(m):
        for col_ind in range(n):
            result += pow((matrix[row_ind, col_ind] - approx_matrix[row_ind, col_ind]), 2)
    return result / (m * n)


def peak_signal_to_noise_ratio(matrix, approx_matrix):
    num_of_max_possible_int_levels = 255
    mse = mean_squared_error(matrix, approx_matrix)
    return 10 * log10(pow(num_of_max_possible_int_levels, 2) / mse)


def analysis_table_rank_r_approx(image):
    matrix = image2matrix(image).astype(int).clip(min=0, max=255)
    ranks = list(range(10, 100, 10))
    mse_values = list()
    psnr_values = list()
    frob_norms = list()
    for r in ranks:
        approx_matrix = rank_r_approx(image, r)
        frob_norms.append(frobenius_norm(matrix, approx_matrix))
        mse_values.append(mean_squared_error(matrix, approx_matrix))
        psnr_values.append(peak_signal_to_noise_ratio(matrix, approx_matrix))
    d = {'ranks': ranks, 'Frobenius norm': frob_norms, 'MSE': mse_values, 'PSNR': psnr_values}
    df = pd.DataFrame(data=d)
    return df
