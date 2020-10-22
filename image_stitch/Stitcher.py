import cv2
import numpy as np
import os
import random
from utilities import cylindrical_project, blend
import matplotlib.pyplot as plt


# 亮度均衡
def _adjust_brightness(img_a, img_b):
    hsv_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)
    rate = np.mean(hsv_a[:, :, 2]) / np.mean(hsv_b[:, :, 2])
    hsv_b = hsv_b.astype("float64")
    print(rate)
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


# 展示关键点匹配结果
def _show_matches(image_a, image_b, keypoints_a, keypoints_b, matches):
    result_a, result_b = None, None
    # 画出两张图的关键点
    result_a = cv2.drawKeypoints(image_a, keypoints_a, result_a, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    result_b = cv2.drawKeypoints(image_b, keypoints_b, result_b, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    result_a = cv2.cvtColor(result_a, cv2.COLOR_BGR2RGB)
    result_b = cv2.cvtColor(result_b, cv2.COLOR_BGR2RGB)
    plt.imshow(result_a)
    plt.show()
    plt.imshow(result_b)
    plt.show()
    # 画出关键点连线
    height_a, width_a = image_a.shape[:2]
    height_b, width_b = image_b.shape[:2]
    result = np.zeros([max(height_a, height_b), width_a + width_b, 3], dtype="uint8")
    result[:height_a, :width_a] = image_a
    result[:height_b, width_a:width_a + width_b] = image_b
    for (i, j) in matches:
        point_a = (int(keypoints_a[i].pt[0]), int(keypoints_a[i].pt[1]))
        point_b = (int(keypoints_b[j].pt[0]) + width_a, int(keypoints_b[j].pt[1]))
        color = _random_color()
        cv2.circle(result, point_a, 2, color, 2)
        cv2.circle(result, point_b, 2, color, 2)
        cv2.line(result, point_a, point_b, color, 2)
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
        if not np.all(image[:, i] == 0):
            offset.append(i)
            break
    for i in range(image.shape[1] - 1, -1, -1):
        if not np.all(image[:, i] == 0):
            offset.append(i)
            break
    image = image[offset[0]:offset[1], offset[2]:offset[3]]
    return image


# 分析图片，返回其关键点和描述符
def _analyze(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_img, None)
    return keypoints, descriptors


# 根据输入的关键点和描述符计算单应性矩阵和优质匹配组合
def _match(keypoints_b, descriptors_b, keypoints_a, descriptors_a):
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=100)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_a, descriptors_b, k=2)
    good_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 1 * n.distance:
            good_matches.append((m.trainIdx, m.queryIdx))
    if len(good_matches) > 4:
        good_keypoints_a = np.float32([keypoints_a[i].pt for (_, i) in good_matches])
        good_keypoints_b = np.float32([keypoints_b[i].pt for (i, _) in good_matches])
        homography, _ = cv2.findHomography(good_keypoints_a, good_keypoints_b, cv2.RANSAC)
        return homography, good_matches
    else:
        return None


class Stitcher:
    def __init__(self, path, debug=False):
        self.debug = debug
        self.images = list()
        # 从传入的path读取目录下所有图片
        filenames = os.listdir(path)
        self.num = 0
        for i in filenames:
            file_path = path + "\\" + i
            image = cv2.imread(file_path)
            image = cv2.resize(image, (1024, 768))
            print("Cylindrical projecting {}...".format(i))
            self.images.append(image)
            self.num += 1
        # 亮度均衡
        for i in range(1, len(self.images)):
            self.images[i - 1], self.images[i] = _adjust_brightness(self.images[i - 1], self.images[i])
        # 柱面投影
        for i in range(len(self.images)):
            self.images[i] = _cylindrical_project(self.images[i])

    # 两两缝合所有图片
    def stitch(self):
        result = self.images[0]
        cnt = 1
        for image in self.images[1:]:
            cnt += 1
            print("Stitching {} of {}:".format(cnt, self.num))
            result = self._stitch_two(result, image)
        # 将结果由BGR转换为RGB
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result

    # 缝合两张相邻图片
    def _stitch_two(self, img_a, img_b):
        # 获取img_a的关键点和描述符
        print("\tAnalyzing image A...")
        keypoints_a, descriptors_a = _analyze(img_a)
        # 获取img_b的关键点和描述符
        print("\tAnalyzing image B...")
        keypoints_b, descriptors_b = _analyze(img_b)
        # 根据关键点和描述符计算单应性矩阵
        print("\tCalculating homography...")
        homography, matches = _match(keypoints_a, descriptors_a, keypoints_b, descriptors_b)
        if self.debug:
            _show_matches(img_a, img_b, keypoints_a, keypoints_b, matches)
        # 计算单应性矩阵的逆矩阵，使参考视角从img_a转为img_b
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
        # 对img_a作透视变换
        wrapped_a = cv2.warpPerspective(img_a, inverse_homography, dimension_size)
        # 拼接img_b和透视变换后的img_a
        result = _blend(wrapped_a, img_b, offset_x, offset_y)
        # 调整拼接后图像的位置，删去空白部分
        result = _move(result)
        print("\tDone!")
        return result


if __name__ == '__main__':
    stitcher = Stitcher("./images")
    result = stitcher.stitch()
    plt.imshow(result)
    plt.show()
