from color_histogram.voting_classifier import VotingClassifier
import os
from tqdm import tqdm

vc = VotingClassifier("seg_train")

# 结果计数器 [TP, FN, FP]
counter = {
    "buildings": [0, 0, 0],
    "forest": [0, 0, 0],
    "glacier": [0, 0, 0],
    "mountain": [0, 0, 0],
    "sea": [0, 0, 0],
    "street": [0, 0, 0],
}

# 对测试集中的每一张图片的类别进行预测
path = "seg_test"
for i in tqdm(counter.keys()):
    dir_path = path + '\\' + i
    filenames = os.listdir(dir_path)
    for j in filenames:
        file_path = dir_path + "\\" + j
        pred = vc.predict(file_path)
        if pred == i:
            counter[i][0] += 1
        else:
            counter[i][1] += 1
            counter[pred][2] += 1

# 计算每一类的查全率和查准率
for i in counter.keys():
    tp = counter[i][0]
    fn = counter[i][1]
    fp = counter[i][2]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("针对{}的查准率为{}, 查全率为{}".format(i, precision, recall))
