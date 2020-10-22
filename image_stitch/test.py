import numpy as np
import cv2
import sys

import time
import copy
import matplotlib.pyplot as plt


# 图像匹配
# 提取特征、特征匹配、构造单应性矩阵
class matchers:
    def __init__(self):
        self.surf = cv2.SIFT_create()  # 特征获取
        # FLANN是快速最近邻搜索包的简称。它是一个对大数据集和高维特征进行最近邻搜索的算法的集合
        # 使用FLANN匹配，需要传入两个字典作为参数
        index_params = dict(algorithm=0, trees=5)  # 第一个是indexParams，配置我们要使用的算法
        search_params = dict(checks=100)  # 第二个是SearchParams。它用来指定递归遍历的次数
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 提取图像特征
    def get_SURF_features(self, image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 使用Hessian算法检测关键点，并且对每个关键点周围的区域计算特征向量，返回关键点的信息和描述符
        keypoints, descriptor = self.surf.detectAndCompute(image_gray, None)
        return {'keypoints': keypoints, 'descriptor': descriptor}

    # 进行图像匹配
    def match(self, image1, image2):
        # 获取两幅图片的特征
        fea_image1 = self.get_SURF_features(image1)
        fea_image2 = self.get_SURF_features(image2)
        # 对两幅图片的特征进行匹配
        # knnMatch：给定查询集合中的每个特征描述子，寻找k个最佳匹配
        # matches得到许多组两两匹配的关键点
        matches = self.flann.knnMatch(fea_image2['descriptor'], fea_image1['descriptor'], k=2)

        # img2 = cv2.drawKeypoints(image1, fea_image1['keypoints'], None)
        # cv2.imwrite('draw_ky.jpg', img2)
        # img3 = cv2.drawMatchesKnn(image1, fea_image1['keypoints'], image2, fea_image2['keypoints'], matches,None,flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        # cv2.imwrite('drawmatch.jpg', img3)
        good_matches = []
        # good = []
        for i, (m, n) in enumerate(matches):
            # 检测出的匹配点可能有一些是错误正例。抛弃距离大于0.7的值，则可以避免几乎90%的错误匹配
            if m.distance < 0.7 * n.distance:
                # trainIdx：测试图像的特征点下标
                # queryIdx：样本图像的特征点下标
                # good.append((m, n))
                good_matches.append((m.trainIdx, m.queryIdx))
        # img3 = cv2.drawMatchesKnn(image1, fea_image1['keypoints'], image2, fea_image2['keypoints'], good,None,flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        # cv2.imwrite('drawmatch.jpg', img3)
        if len(good_matches) > 4:
            points_1 = fea_image1['keypoints']
            points_2 = fea_image2['keypoints']
            matchedPointsPrev = np.float32([points_1[i].pt for (i, __) in good_matches])
            matchedPointsCurrent = np.float32([points_2[i].pt for (__, i) in good_matches])
            # matchedPointsCurrent：源平面中点的坐标矩阵
            # matchedPointsPrev：目标平面中点的坐标矩阵
            # RANSAC:计算单应矩阵所使用的方法-基于RANSAC的鲁棒算法
            # 第四个参数是误差阈值
            H, status = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
            return H
        else:
            return None


class Stitch:
    def __init__(self, args):
        self.path = args
        fp = open(self.path, 'r')
        filenames = [each.rstrip('\r\n') for each in fp.readlines()]
        print(filenames)
        self.images = [cv2.resize(cv2.imread(each), (480, 640)) for each in filenames]
        self.count = len(self.images)
        self.matcher_obj = matchers()

    def find_the_top(self, H, shape):
        # left top
        [w, h, tem] = shape
        v2 = [0, 0, 1]
        v1 = np.dot(H, v2)
        top = []
        top.append([v1[0] / v2[2], v1[1] / v2[2]])

        # left bottom
        v2[0] = 0
        v2[1] = w
        v2[2] = 1
        v1 = np.dot(H, v2)
        top.append([v1[0] / v2[2], v1[1] / v2[2]])

        # right top
        v2[0] = h
        v2[1] = 0
        v2[2] = 1
        v1 = np.dot(H, v2)
        top.append([v1[0] / v2[2], v1[1] / v2[2]])
        return top

    def two_in_one(self, imageA, imageB, begin_w, last_w):
        (hA, wA, tem) = imageA.shape
        (hB, wB, tem) = imageB.shape
        h = min(hA, hB)
        over = int(wA - begin_w - last_w)
        begin_w = int(begin_w)
        imageA[0:h, 0:begin_w] = imageB[0:h, 0:begin_w]
        for now_w in range(begin_w, wB):
            for now_h in range(0, h):
                alpha = (now_w * 1.0 - begin_w) / over * 1.0
                if (imageA[now_h][now_w][0] == 0) & (imageA[now_h][now_w][1] == 0) & (imageA[now_h][now_w][2] == 0):
                    alpha = 0
                imageA[now_h][now_w][0] = imageA[now_h][now_w][0] * alpha + imageB[now_h][now_w][0] * (1 - alpha)
                imageA[now_h][now_w][1] = imageA[now_h][now_w][1] * alpha + imageB[now_h][now_w][1] * (1 - alpha)
                imageA[now_h][now_w][2] = imageA[now_h][now_w][2] * alpha + imageB[now_h][now_w][2] * (1 - alpha)
        return imageA

    def shift(self):
        a = self.images[0]
        for b in self.images[1:]:
            H = self.matcher_obj.match(a, b)  # 特征点匹配
            top = self.find_the_top(H, b.shape)
            last_w = int(min(top[0][0], top[1][0]))
            tmp = cv2.warpPerspective(b, H, (b.shape[1] + last_w, max(a.shape[0], b.shape[0])))
            cv2.imwrite('tmp_a.jpg', tmp)
            result = self.two_in_one(tmp, a, max(top[0][0], top[1][0]), last_w)
            cv2.imwrite('result.jpg', result)
            a = tmp  # 为循环做准备
        return a


if __name__ == '__main__':
    try:
        args = sys.argv[1]
    except:
        args = "./images.txt"
    finally:
        print("Parameters : ", args)
    s = Stitch(args)
    leftimage = s.shift()
    print("done")
    cv2.imwrite("test_yanque.jpg", leftimage)
    print("image written~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
