import time
import cv2
import matplotlib.pyplot as plt
import os

def calculate_duration(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # 函数开始时间
        func(*args, **kwargs)
        end_time = time.perf_counter()  # 函数结束时间
        duration = end_time - start_time
        return duration
    return wrapper

@calculate_duration
def compress(img_path, rate):
    """
    :param img_path: 图片存放路径
    :param rate: 压缩率
    :return: 压缩用时
    """

    quality = int(100*(1-rate)) # 设置图片质量
    src_img = cv2.imread(img_path) # 读取图片
    cv2.imwrite("{}_{}".format(rate, img_path), src_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality]) # 保存压缩后的图片

if __name__ == '__main__':
    # 设置x轴和y轴
    x = []
    y = []
    plt.xlabel("rate")
    plt.ylabel("cost time")

    # 尝试不同压缩率并计算压缩时间
    for i in range(100):
        rate = i/100
        x.append(rate)
        y.append(compress("image.jpg", rate))

    # 画出压缩时间与压缩率之间的关系
    plt.plot(x, y)
    plt.show()

    # 删除运算过程中产生的垃圾文件
    for i in range(100):
        file_name = "{}_image.jpg".format(i/100)
        os.remove(file_name)
