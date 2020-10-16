from color_histogram.voting_classifier import VotingClassifier
import os
from tqdm import tqdm
import random

vc = VotingClassifier("seg_train", is_RGB=True)

# 结果计数器 [TP, FN, FP]
counter = {
    "buildings": [0, 0, 0],
    "forest": [0, 0, 0],
    "glacier": [0, 0, 0],
    "mountain": [0, 0, 0],
    "sea": [0, 0, 0],
    "street": [0, 0, 0],
}

# 对测试集中的每一张图片的类别进行预测并统计结果
path = "seg_test"
all_file_paths = list()
for i in tqdm(counter.keys()):
    dir_path = path + '\\' + i
    filenames = os.listdir(dir_path)
    for j in filenames:
        file_path = dir_path + "\\" + j
        all_file_paths.append(file_path)
        pred = vc.predict(file_path, 50)
        if pred == i:
            counter[i][0] += 1
        else:
            counter[i][1] += 1
            counter[pred][2] += 1

# # 展示来自不同类别的10幅图片并选出与图片最接近的5幅图片
# chosen_files = random.choices(all_file_paths, k=10)
# for i in chosen_files:
#     vc.predict(i, show=True)

# 计算每一类的查全率和查准率
for i in counter.keys():
    tp, fn, fp = counter[i]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("针对{}的查准率为{}, 查全率为{}".format(i, precision, recall))

# # 计算不同颜色模式下的查准率和查全率
# results = {
#     "RGB": list(),
#     "HSV": list()
# }
# for mode in ["RGB", "HSV"]:
#     # 设置颜色模式
#     vc = VotingClassifier("seg_train", mode is "RGB")
#     # 重置结果计数器 [TP, FN, FP]
#     counter = {
#         "buildings": [0, 0, 0],
#         "forest": [0, 0, 0],
#         "glacier": [0, 0, 0],
#         "mountain": [0, 0, 0],
#         "sea": [0, 0, 0],
#         "street": [0, 0, 0],
#     }
#     # 对测试集中的每一张图片的类别进行预测并统计结果
#     path = "seg_test"
#     for i in tqdm(counter.keys()):
#         dir_path = path + '\\' + i
#         filenames = os.listdir(dir_path)
#         for j in filenames:
#             file_path = dir_path + "\\" + j
#             pred = vc.predict(file_path)
#             if pred == i:
#                 counter[i][0] += 1
#             else:
#                 counter[i][1] += 1
#                 counter[pred][2] += 1
#     # 记录结果
#     precision_list = []
#     recall_list = []
#     for i in counter.keys():
#         tp, fn, fp = counter[i]
#         precision = tp / (tp + fp)
#         recall = tp / (tp + fn)
#         precision_list.append(precision)
#         recall_list.append(recall)
#     results[mode].append(precision_list)
#     results[mode].append(recall_list)
# # 展示结果
# import matplotlib.pyplot as plt
#
# classes = [
#     "buildings",
#     "forest",
#     "glacier",
#     "mountain",
#     "sea",
#     "street",
# ]
# plt.plot(classes, results["RGB"][0], color='r', label="RGB precision")
# plt.plot(classes, results["HSV"][0], color='b', label="HSV precision")
# plt.plot(classes, results["RGB"][1], color='r', linestyle="--", label="RGB recall")
# plt.plot(classes, results["HSV"][1], color='b', linestyle="--", label="HSV recall")
# plt.legend()
# plt.show()
# plt.cla()
#
#
# # 计算不同参考图片数下的查准率和查全率
# vc = VotingClassifier("seg_train", "RGB")
# x = [i for i in range(5, 101, 5)]
# results = {
#     "buildings": [list(), list()],
#     "forest": [list(), list()],
#     "glacier": [list(), list()],
#     "mountain": [list(), list()],
#     "sea": [list(), list()],
#     "street": [list(), list()]
# }
# for vote_count in tqdm(x):
#     # 重置结果计数器 [TP, FN, FP]
#     counter = {
#         "buildings": [0, 0, 0],
#         "forest": [0, 0, 0],
#         "glacier": [0, 0, 0],
#         "mountain": [0, 0, 0],
#         "sea": [0, 0, 0],
#         "street": [0, 0, 0],
#     }
#     # 对测试集中的每一张图片的类别进行预测并统计结果
#     path = "seg_test"
#     for i in counter.keys():
#         dir_path = path + '\\' + i
#         filenames = os.listdir(dir_path)
#         for j in filenames:
#             file_path = dir_path + "\\" + j
#             pred = vc.predict(file_path, vote_count)
#             if pred == i:
#                 counter[i][0] += 1
#             else:
#                 counter[i][1] += 1
#                 counter[pred][2] += 1
#     # 计算每一类的查全率和查准率
#     for i in counter.keys():
#         tp, fn, fp = counter[i]
#         precision = tp / (tp + fp)
#         recall = tp / (tp + fn)
#         results[i][0].append(precision)
#         results[i][1].append(recall)
# # 展示结果
# import matplotlib.pyplot as plt
# for i in results.keys():
#     plt.plot(x, results[i][0], label=i)
# plt.title("precision")
# plt.legend()
# plt.show()
# plt.cla()
# for i in results.keys():
#     plt.plot(x, results[i][1], label=i)
# plt.title("recall")
# plt.legend()
# plt.show()