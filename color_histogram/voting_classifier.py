import cv2
import os
from tqdm import tqdm
import numpy as np


# 计算颜色分布并绘制颜色直方图（可选）
def calc_hist(path, is_RGB=False):
    # 读取图片
    image = cv2.imread(path)
    if is_RGB:
        hist = []
        # 计算BGR三种颜色分布
        for i in range(3):
            hist.append(cv2.calcHist([image], [i], None, [256], [0.0, 255.0]))
        return np.array(hist)
    else:
        # 转换颜色模式
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # 计算颜色分布
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        return hist


# 投票分类器
class VotingClassifier:
    def __init__(self, src_path, is_RGB=False):
        self.mode = is_RGB
        # 以所有类别名为键构建字典
        self.hist = {
            "buildings": [],
            "forest": [],
            "glacier": [],
            "mountain": [],
            "sea": [],
            "street": [],
        }
        # 分别读取每一个类别下的所有图片，计算颜色分布并存储到字典中
        for i in tqdm(self.hist.keys()):
            dir_path = src_path + '\\' + i
            filenames = os.listdir(dir_path)
            for j in filenames:
                file_path = dir_path + "\\" + j
                hist = calc_hist(file_path, self.mode)
                self.hist[i].append([hist, file_path])

    def predict(self, file_path, limit=5, show=False):
        # 计算输入图片的颜色分布
        hist = calc_hist(file_path, self.mode)
        results = []  # 用于存储相似度计算结果
        for i in self.hist.keys():
            for j in self.hist[i]:
                # 计算相似度
                correlation = abs(cv2.compareHist(hist, j[0], cv2.HISTCMP_CORREL))
                # 将类别和相似度存储至results
                results.append([i, correlation, j[1]])
        # 将列表元素按相似度排序，取相似度最高的前limit个元素进行投票
        results.sort(key=lambda x: x[1], reverse=True)
        # 展示图片
        if show:
            import matplotlib.pyplot as plt
            plt.figure()
            # 画出原图
            plt.subplot(2, 3, 1)
            plt.title("original")
            img = plt.imread(file_path)
            plt.imshow(img)
            plt.axis('off')
            # 画出相似图
            for i in range(5):
                plt.subplot(2, 3, i + 2)
                plt.title("similar_{}".format(i + 1))
                img = plt.imread(results[i][2])
                plt.imshow(img)
                plt.axis('off')
            plt.show()
        # 投票计算
        vote = {
            "buildings": 0.0,
            "forest": 0.0,
            "glacier": 0.0,
            "mountain": 0.0,
            "sea": 0.0,
            "street": 0.0,
        }
        for i in results[:limit]:
            vote[i[0]] += i[1]
        # 返回相似度之和最大的类别
        return sorted(vote.items(), key=lambda x: x[1], reverse=True)[0][0]
