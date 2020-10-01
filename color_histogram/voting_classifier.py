import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np


# 计算颜色分布并绘制颜色直方图（可选）
def calc_hist(path, hist_size=256, show=False):
    image = cv2.imread(path)
    color = ["blue", "green", "red"]
    result = list()
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [hist_size], [0, 256])
        result.append(hist)
        plt.plot(hist, color[i])
    if show:
        plt.title("Histogram of image")
        plt.show()
    return np.array(result)


def hello():
    print(1)


# 投票分类器
class VotingClassifier:
    def __init__(self, src_path):
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
                hist = calc_hist(file_path)
                self.hist[i].append(hist)

    def predict(self, file_path):
        # 计算输入图片的颜色分布
        hist = calc_hist(file_path)
        results = []  # 用于存储相似度计算结果
        for i in self.hist.keys():
            for j in self.hist[i]:
                # 计算相似度
                correlation = abs(cv2.compareHist(hist, j, cv2.HISTCMP_CORREL))
                # 将类别和相似度存储至results
                results.append([i, correlation])
        # 将列表元素按相似度排序，取相似度最高的前5个元素进行投票
        results.sort(key=lambda x: x[1], reverse=True)
        vote = {
            "buildings": 0.0,
            "forest": 0.0,
            "glacier": 0.0,
            "mountain": 0.0,
            "sea": 0.0,
            "street": 0.0,
        }
        for i in results[:5]:
            vote[i[0]] += i[1]
        # 返回相似度之和最大的类别
        return sorted(vote.items(), key=lambda x: x[1], reverse=True)[0][0]
