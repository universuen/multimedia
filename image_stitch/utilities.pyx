import numpy as np
from libc.math cimport tan, atan, cos, pi
cimport cython

# 对的传入图片作柱面投影
@cython.boundscheck(False)
@cython.wraparound(False)
def cylindrical_project(unsigned char[:, :, :]image):
    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]
    cdef unsigned char[:, :, :] result = np.zeros_like(image)
    cdef Py_ssize_t center_x = int(cols / 2)
    cdef Py_ssize_t center_y = int(rows / 2)
    cdef double alpha = pi / 4
    cdef double f = cols / (2 * tan(alpha / 2))
    cdef double theta
    cdef Py_ssize_t point_x, point_y
    cdef Py_ssize_t x, y

    for y in range(rows):
        for x in range(cols):
            theta = atan((x - center_x) / f)
            point_x = <int>(f * tan((x - center_x) / f)) + center_x
            point_y = <int>((y - center_y) / cos(theta)) + center_y

            if point_x >= cols or point_x < 0 or point_y >= rows or point_y < 0:
                pass
            else:
                result[y, x, :] = image[point_y, point_x, :]
    return np.array(np.resize(result, (rows, cols, 3)))

# 根据偏移量拼接两幅图片
@cython.boundscheck(False)
@cython.wraparound(False)
def blend(unsigned char[:, :, :]img_1, unsigned char[:, :, :]img_2, int offset_x, int offset_y):
    cdef int result_shape_0 = img_1.shape[0]
    cdef int result_shape_1 = img_1.shape[1]
    cdef int shape_0 = img_2.shape[0]
    cdef int shape_1 = img_2.shape[1]
    cdef int i, j
    cdef float alpha, src_len, test_len

    for i in range(offset_y, offset_y + shape_0):
        for j in range(offset_x, offset_x + shape_1):
            if img_2[i-offset_y][j-offset_x][0] + img_2[i-offset_y][j-offset_x][1] + img_2[i-offset_y][j-offset_x][2] != 0:
                img_1[i][j][0] = img_2[i-offset_y][j-offset_x][0]
                img_1[i][j][1] = img_2[i-offset_y][j-offset_x][1]
                img_1[i][j][2] = img_2[i-offset_y][j-offset_x][2]

    return np.array(np.resize(img_1, (result_shape_0, result_shape_1, 3)))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def diffusion(unsigned char[:, :, :]result, unsigned char[:, :, :]img_1, unsigned char[:, :, :]img_2, int offset_x, int offset_y):
    cdef int result_shape_0 = result.shape[0]
    cdef int result_shape_1 = result.shape[1]
    cdef int left = 0
    cdef int right = result.shape[0]
    cdef int i, j
    cdef float src_len, test_len, alpha

    for i in range(offset_y, img_2.shape[0]):
        for j in range(offset_x, img_2.shape[1]):
            if img_2[i-offset_y][j-offset_x][0] + img_2[i-offset_y][j-offset_x][1] + img_2[i-offset_y][j-offset_x][2] != 0:
                src_len = <float>abs(j - left)
                test_len = <float>abs(j - right)
                alpha = src_len/(src_len + test_len)
                # print(i, j)
                for k in range(3):
                    result[i][j][k] = <int>(<float>img_2[i-offset_y][j-offset_x][k]*(1-alpha) + <float>img_1[i][j][k] * alpha)
                    if result[i][j][k] > 255:
                        result[i][j][k] = 255

    return np.array(np.resize(result, (result_shape_0, result_shape_1, 3)))
