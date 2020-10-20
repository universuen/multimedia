import cv2
import numpy as np
import os
import random
from utilities import cylindrical_project, blend
import matplotlib.pyplot as plt


# # 获得图片的高斯金字塔
# def _gaussian_pyramid(image):
#     _G = image
#     gp = [_G]
#     for _ in range(6):
#         _G = cv2.pyrDown(_G)
#         gp.append(_G)
#     return gp
#
#
# # 调整两幅图像至相同大小
# def _same_size(img_a, img_b):
#     shape = (min(img_a.shape[1], img_b.shape[1]), min(img_a.shape[0], img_b.shape[0]))
#     img_a = cv2.resize(img_a, shape)
#     img_b = cv2.resize(img_b, shape)
#     return img_a, img_b
#
#
# # 根据高斯金字塔获得图片的拉普拉斯金字塔
# def _laplacian_pyramid(gp):
#     lp = [gp[5]]
#     for i in range(5, 0, -1):
#         _GE = cv2.pyrUp(gp[i])
#         img_a, img_b = _same_size(gp[i - 1], _GE)
#         _L = cv2.subtract(img_a, img_b)
#         lp.append(_L)
#     return lp
#
#
# def _diffusion(image, offset_x, offset_y):
#     # 根据offset_x和offset_y对image进行切割
#     img_a = image[:offset_x]
#     img_b = image[offset_x:]
#     # 生成img_a和img_b的的高斯金字塔
#     gp_a = _gaussian_pyramid(img_a)
#     gp_b = _gaussian_pyramid(img_b)
#     # 根据高斯金字塔生成拉普拉斯金字塔
#     lp_a = _laplacian_pyramid(gp_a)
#     lp_b = _laplacian_pyramid(gp_b)
#     # 融合两个拉普拉斯金字塔
#     image_lp = list()
#     for i, j in zip(lp_a, lp_b):
#         layer = np.zeros((i.shape[0]+j.shape[0], i.shape[1], 3), dtype="uint8")
#         layer[:i.shape[0]] = i
#         layer[i.shape[0]:i.shape[0]+j.shape[0]] = j
#         image_lp.append(layer)
#     # 根据拉普拉斯金字塔重建图片
#     image = image_lp[0]
#     for i in range(1, 6):
#         image = cv2.pyrUp(image)
#         img_a, img_b = _same_size(image, image_lp[i])
#         image = cv2.add(img_a, img_b)
#
#     # 根据offset_y对image进行纵向切割
#     img_a = image[:, :offset_y]
#     img_b = image[:, offset_y:]
#     # 生成img_a和img_b的的高斯金字塔
#     gp_a = _gaussian_pyramid(img_a)
#     gp_b = _gaussian_pyramid(img_b)
#     # 根据高斯金字塔生成拉普拉斯金字塔
#     lp_a = _laplacian_pyramid(gp_a)
#     lp_b = _laplacian_pyramid(gp_b)
#     # 融合两个拉普拉斯金字塔
#     image_lp = list()
#     for i, j in zip(lp_a, lp_b):
#         layer = np.zeros((i.shape[0], i.shape[1] + j.shape[1], 3), dtype="uint8")
#         layer[:, :i.shape[1]] = i
#         layer[:, i.shape[1]:i.shape[1] + j.shape[1]] = j
#         image_lp.append(layer)
#     # 根据拉普拉斯金字塔重建图片
#     image = image_lp[0]
#     for i in range(1, 6):
#         image = cv2.pyrUp(image)
#         img_a, img_b = _same_size(image, image_lp[i])
#         image = cv2.add(img_a, img_b)
#     return image


# 曝光均衡
def _adjust_color(img_a, img_b):
    hsv_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)
    rate = np.mean(hsv_a[:, :, 2]) / np.mean(hsv_b[:, :, 2])
    hsv_b = hsv_b.astype("float64")
    print(rate)
    if rate > 1:
        rate *= 0.92
    else:
        rate *= 1.12
    hsv_b[:, :, 2] *= rate
    hsv_b[:, :, 2][hsv_b[:, :, 2] > 255] = 255
    hsv_b = hsv_b.astype("uint8")
    return cv2.cvtColor(hsv_a, cv2.COLOR_HSV2BGR), cv2.cvtColor(hsv_b, cv2.COLOR_HSV2BGR)


# 拼接
def _blend(img_1, img_2, offset_x, offset_y):
    return blend(img_1, img_2, offset_x, offset_y)


# 获取图片的柱面投影
def _cylindrical_project(image):
    return cylindrical_project(image)


# 获取随机颜色
def _random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    return b, g, r


# 显示关键点连线
def _show_matches(image_a, image_b, keypoints_a, keypoints_b, matches):
    height_a, width_a = image_a.shape[:2]
    height_b, width_b = image_b.shape[:2]
    result = np.zeros([max(height_a, height_b), width_a + width_b, 3], dtype="uint8")
    result[:height_a, :width_a] = image_a
    result[:height_b, width_a:width_a + width_b] = image_b
    for (i, j) in matches:
        point_a = (int(keypoints_a[i].pt[0]), int(keypoints_a[i].pt[1]))
        point_b = (int(keypoints_b[j].pt[0]) + width_a, int(keypoints_b[j].pt[1]))
        color = _random_color()
        cv2.circle(result, point_a, 10, color, 10)
        cv2.circle(result, point_b, 10, color, 10)
        cv2.line(result, point_a, point_b, color, 10)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    plt.imshow(result)
    plt.show()


# 将图片调整到合适的位置
def _move(image):
    offset = []
    for i in range(image.shape[0]):
        if not np.all(image[i] == 0):
            offset.append(i)
            break
    for i in range(image.shape[0] - 1, -1, -1):
        if not np.all(image[i] == 0):
            offset.append(i)
            break
    for i in range(image.shape[1]):
        if not np.all(image[i] == 0):
            offset.append(i)
            break
    for i in range(image.shape[1] - 1, -1, -1):
        if not np.all(image[:, i] == 0):
            offset.append(i)
            break
    image = image[offset[0]:offset[1], offset[2]:offset[3]]
    return image


# 分析图片，返回其关键点和描述符
def _analyze(image):  # -> keypoints, descriptors
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    result = sift.detectAndCompute(gray_img, None)
    del sift
    return result


# 根据输入的关键点和描述符计算单应性矩阵
def _match(keypoints_b, descriptors_b, keypoints_a, descriptors_a):
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=100)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_a, descriptors_b, k=2)
    good_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            good_matches.append((m.trainIdx, m.queryIdx))
    if len(good_matches) > 4:
        good_keypoints_a = np.float32([keypoints_a[i].pt for (_, i) in good_matches])
        good_keypoints_b = np.float32([keypoints_b[i].pt for (i, _) in good_matches])
        homography, _ = cv2.findHomography(good_keypoints_a, good_keypoints_b, cv2.RANSAC)
        return homography, good_matches
    else:
        return None


class Stitcher:
    def __init__(self, path, ratio=0.5, debug=False):
        self.ratio = ratio
        self.debug = debug
        self.images = list()
        filenames = os.listdir(path)
        self.num = 0
        # 读取所有图片
        for i in filenames:
            file_path = path + "\\" + i
            image = cv2.imread(file_path)
            image = cv2.resize(image, (1024, 768))
            print("Cylindrical projecting {}...".format(i))
            self.images.append(image)
            self.num += 1
        # 曝光补偿
        for i in range(1, len(self.images)):
            self.images[i - 1], self.images[i] = _adjust_color(self.images[i - 1], self.images[i])
        # 柱面投影
        for i in range(len(self.images)):
            self.images[i] = _cylindrical_project(self.images[i])

    # 逐个缝合所有图片
    def stitch(self):
        # 缝合所有图片
        result = self.images[0]
        cnt = 0
        for image in self.images:
            cnt += 1
            print("Stitching {} of {}:".format(cnt, self.num))
            result = self._stitch_two(result, image)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result

    # 缝合两张相邻图片
    def _stitch_two(self, img_a, img_b):
        print("\tAnalyzing image A...")
        keypoints_a, descriptors_a = _analyze(img_a)

        print("\tAnalyzing image B...")
        keypoints_b, descriptors_b = _analyze(img_b)

        print("\tCalculating homography...")
        homography, matches = _match(keypoints_a, descriptors_a, keypoints_b, descriptors_b)
        if self.debug:
            _show_matches(img_a, img_b, keypoints_a, keypoints_b, matches)

        print("\tCalculating offset...")
        inverse_homography = np.linalg.inv(homography)
        start = np.dot(inverse_homography, np.array([0, 0, 1]))
        start /= start[-1]
        inverse_homography[0][-1] += abs(start[0])
        inverse_homography[1][-1] += abs(start[1])
        end = np.dot(inverse_homography, np.array([img_a.shape[1], img_a.shape[0], 1]))
        offset_x = abs(int(start[0]))
        offset_y = abs(int(start[1]))
        dimension_size = (int(end[0]) + offset_x + img_b.shape[1], int(end[1]) + offset_y + img_b.shape[0])
        print("\tStitching...")
        wrapped_a = cv2.warpPerspective(img_a, inverse_homography, dimension_size)
        # result[offset_y:offset_y + img_b.shape[0], offset_x:offset_x + img_b.shape[1]] = img_b
        result = _blend(wrapped_a, img_b, offset_x, offset_y)
        # result = self._diffusion(result, img_b, offset_x, offset_y)
        result = _move(result)
        # result = _diffusion(result, offset_x, offset_y)
        print("\tDone!")
        return result


if __name__ == '__main__':
    stitcher = Stitcher("./images")
    result = stitcher.stitch()
    plt.imshow(result)
    plt.show()
    cv2.create
    # cv2.imshow("Result", result)
    # cv2.waitKey()
