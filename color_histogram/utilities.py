import cv2
import matplotlib.pyplot as plt


# 计算颜色分布并绘制颜色直方图（可选）
def calc_histogram(img, hist_size:int = 256, show:bool = False):
    color = ["blue", "green", "red"]
    result = list()
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [hist_size], [0, 256])
        result.append(hist)
        plt.plot(hist, color[i])
    if show:
        plt.title("Histogram of image")
        plt.show()
    return result

if __name__ == '__main__':
    img = cv2.imread("image.jpg")
    # cv2.imshow('img', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    var = calc_histogram(img, show=True)
    pass