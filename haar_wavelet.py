from math import sqrt
from numpy import abs, array, linalg, matrix, matmul, ndarray, newaxis, where
from PIL import Image


SQRT_8 = 1 / sqrt(8)
SQRT_2 = 1 / sqrt(2)
HW_TRANSFORM_MATRIX = matrix([[0.125, 0.125, 0.25, 0., 0.5, 0., 0., 0.],
                              [0.125, 0.125, 0.25, 0., -0.5, 0., 0., 0.],
                              [0.125, 0.125, -0.25, 0., 0., 0.5, 0., 0.],
                              [0.125, 0.125, -0.25, 0., 0., -0.5, 0., 0.],
                              [0.125, -0.125, 0., 0.25, 0., 0., 0.5, 0.],
                              [0.125, -0.125, 0., 0.25, 0., 0., -0.5, 0.],
                              [0.125, -0.125, 0., -0.25, 0., 0., 0., 0.5],
                              [0.125, -0.125, 0., -0.25, 0., 0., 0., -0.5]])
HW_TRANSFORM_MATRIX_INVERSE = linalg.inv(HW_TRANSFORM_MATRIX)
HW_TRANSFORM_MATRIX_TRANSPOSED = HW_TRANSFORM_MATRIX.transpose()
HW_TRANSFORM_MATRIX_TRANSPOSED_INVERSE = linalg.inv(HW_TRANSFORM_MATRIX_TRANSPOSED)
HW_TRANSFORM_MATRIX_ORTHOGONAL = matrix([[SQRT_8, SQRT_8, 0.5, 0., SQRT_2, 0., 0., 0.],
                                         [SQRT_8, SQRT_8, 0.5, 0., -SQRT_2, 0., 0., 0.],
                                         [SQRT_8, SQRT_8, -0.5, 0., 0., SQRT_2, 0., 0.],
                                         [SQRT_8, SQRT_8, -0.5, 0., 0., -SQRT_2, 0., 0.],
                                         [SQRT_8, -SQRT_8, 0., 0.5, 0., 0., SQRT_2, 0.],
                                         [SQRT_8, -SQRT_8, 0., 0.5, 0., 0., -SQRT_2, 0.],
                                         [SQRT_8, -SQRT_8, 0., -0.5, 0., 0., 0., SQRT_2],
                                         [SQRT_8, -SQRT_8, 0., -0.5, 0., 0., 0., -SQRT_2]])
HW_TRANSFORM_MATRIX_ORTHOGONAL_TRANSPOSED = HW_TRANSFORM_MATRIX_ORTHOGONAL.transpose()


def image2matrix(path: str, rgb: bool = False):
    if rgb:
        image = array(Image.open(path).convert('RGB'))
        return (image,
                matrix(image[:, :, :1].reshape(image.shape[0], -1)),
                matrix(image[:, :, 1:2].reshape(image.shape[0], -1)),
                matrix(image[:, :, 2:].reshape(image.shape[0], -1)))
    else:
        return matrix(Image.open(path).convert('L'))


def matrix2image(mat, filename: str = 'output.png', rgb: bool = False):
    mode = ('L', 'RGB')[rgb]
    image = Image.fromarray(mat, mode=mode)
    image.save(filename)


def divide_matrix(mat: matrix, n: int = 8) -> list:
    if (n & (n - 1) != 0) or n == 0:
        raise ValueError('N must be positive integer and power of 2')
    if (shape := mat.shape)[0] % n != 0 or shape[1] % n != 0:
        raise ValueError('Image length and width must be equally divided by N')

    height, length, subs = shape[0], shape[1], list()

    for i in range(height // n):
        for j in range(length // n):
            subs.append(mat[i * n:(i + 1) * n, j * n:(j + 1) * n])

    return subs


def union_matrices(mat: matrix, subs: list, n: int = 8) -> None:
    height, length = mat.shape[0], mat.shape[1]
    cur_h, cur_l = n, n

    for sub in subs:
        mat[cur_h - n:cur_h, cur_l - n:cur_l] = sub
        cur_l += n

        if cur_l > length:
            cur_h += n
            cur_l = n


def compress_matrix(mat: matrix) -> ndarray:
    return matmul(HW_TRANSFORM_MATRIX_TRANSPOSED, matmul(mat, HW_TRANSFORM_MATRIX))


def compress_matrix_n(mat: matrix) -> ndarray:
    return matmul(HW_TRANSFORM_MATRIX_ORTHOGONAL_TRANSPOSED, matmul(mat, HW_TRANSFORM_MATRIX_ORTHOGONAL))


def decompress_matrix(mat: matrix) -> ndarray:
    return matmul(HW_TRANSFORM_MATRIX_TRANSPOSED_INVERSE, matmul(mat, HW_TRANSFORM_MATRIX_INVERSE))


def decompress_matrix_n(mat: matrix) -> ndarray:
    return matmul(HW_TRANSFORM_MATRIX_ORTHOGONAL, matmul(mat, HW_TRANSFORM_MATRIX_ORTHOGONAL_TRANSPOSED))


def compress_lossy(mat: matrix, ratio: float = 1.) -> None:
    height, length = mat.shape[0], mat.shape[1]
    zeros = where(mat == 0.)[:2]
    num_zeros = len(zeros[0])
    num_non_zeros = height * length - num_zeros
    num_replacements = int(num_non_zeros - num_non_zeros / ratio)

    for i in range(num_zeros):
        mat[zeros[0][i], zeros[1][i]] = 256.

    abs_matrix = abs(mat)

    while True:
        nearest = abs_matrix.argmin()
        smallest = abs_matrix[nearest // length, nearest % length]
        coordinates = where(abs_matrix == smallest)[:2]

        for i in range(len(coordinates[0])):
            if num_replacements < 1:
                break
            abs_matrix[coordinates[0][i], coordinates[1][i]] = 256.
            mat[coordinates[0][i], coordinates[1][i]] = 256.
            num_replacements -= 1
        else:
            continue
        break

    zeros = where(mat == 256.)[:2]
    num_zeros = len(zeros[0])

    for i in range(num_zeros):
        mat[zeros[0][i], zeros[1][i]] = 0.


def compress_image(mat: matrix, normalization: bool = False) -> matrix:
    comp_fn = (compress_matrix, compress_matrix_n)[normalization]
    subs = divide_matrix(mat)

    for i in range(len(subs)):
        subs[i] = comp_fn(subs[i])

    fl_matrix = matrix([[0. for _ in range(mat.shape[1])] for i in range(mat.shape[0])])
    union_matrices(fl_matrix, subs)
    return fl_matrix


def decompress_image(mat: matrix, org_matrix: matrix, normalization: bool = False) -> None:
    decomp_fn = (decompress_matrix, decompress_matrix_n)[normalization]
    subs = divide_matrix(mat)

    for i in range(len(subs)):
        subs[i] = decomp_fn(subs[i]).clip(min=0., max=255.)

    union_matrices(org_matrix, subs)


def compress(input_f: str, output_f: str = "output.png", ratio: float = 1., rgb: bool = False, normalization: bool = False) -> None:
    if rgb:
        rgb_array, org_matrix_r, org_matrix_g, org_matrix_b = image2matrix(input_f, rgb=True)
        compressed_matrix_r = compress_image(org_matrix_r, normalization=normalization)
        compressed_matrix_g = compress_image(org_matrix_g, normalization=normalization)
        compressed_matrix_b = compress_image(org_matrix_b, normalization=normalization)

        for mat in (compressed_matrix_r, compressed_matrix_g, compressed_matrix_b):
            compress_lossy(mat, ratio)

        decompress_image(compressed_matrix_r, org_matrix_r, normalization=normalization)
        decompress_image(compressed_matrix_g, org_matrix_g, normalization=normalization)
        decompress_image(compressed_matrix_b, org_matrix_b, normalization=normalization)

        rgb_array[:, :, :1] = array(org_matrix_r)[:, :, newaxis]
        rgb_array[:, :, 1:2] = array(org_matrix_g)[:, :, newaxis]
        rgb_array[:, :, 2:] = array(org_matrix_b)[:, :, newaxis]

        matrix2image(rgb_array, output_f, rgb=True)
    else:
        org_matrix = image2matrix(input_f)
        compressed_matrix = compress_image(org_matrix, normalization=normalization)
        compress_lossy(compressed_matrix, ratio)
        decompress_image(compressed_matrix, org_matrix, normalization=normalization)
        matrix2image(org_matrix, output_f)
