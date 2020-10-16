import matplotlib.pyplot as plt
from math import pi
import numpy as np
import cv2


class Filter:
    def __init__(self, img):
        self.img = img
        self.shape = self.img.shape

    def apply(self, mask):
        assert mask.shape == self.shape
        # 生成频谱
        f_img = np.fft.fft2(self.img)
        fs_img = np.fft.fftshift(f_img)

        # 滤波
        fs_img *= mask
        fs_img[fs_img == 0] = 1
        # 将滤波后的频谱加入画板
        plt.subplot(121)
        plt.imshow(np.log(abs(fs_img)), cmap="gray")
        plt.axis("off")
        # 重构图片
        f_img = np.fft.ifftshift(fs_img)
        img = np.real(np.fft.ifft2(f_img))
        img[img < 0] = 0
        img[img > 255] = 255
        # 将滤波后重建的图片加入画板
        plt.subplot(122)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        # 显示画板
        plt.show()

# 以灰度形式读入图片
img = cv2.imread("image_2.png")
# 实例化Filter
f = Filter(img)
# 记录图片规格
M, N = f.shape

# 定义滤波器0
mask_0 = np.ones([M, N])
# 激活滤波器0
f.apply(mask_0)

# 定义滤波器1
mask_1 = np.ones([M, N])
M0 = M / 2
N0 = N / 2
for i in range(M):
    for j in range(N):
        if j != N0:
            k = (i - M0) / (j - N0)
            if np.tan(40 * pi / 180) <= k <= np.tan(pi / 2):
                mask_1[i][j] = 4
# 激活滤波器1
f.apply(mask_1)

# 定义滤波器2
mask_2 = np.ones([M, N])
M0 = M / 2
N0 = N / 2
for i in range(M):
    for j in range(N):
        if j != N0:
            k = (i - M0) / (j - N0)
            if np.tan(-15 * pi / 180) <= k <= np.tan(15 * pi / 180):
                mask_2[i][j] = 0
# 激活滤波器2
f.apply(mask_2)

# 定义滤波器3
mask_3 = np.ones([M, N])
M0 = M / 2
N0 = N / 2
for i in range(M):
    for j in range(N):
        if j != N0:
            k = (i - M0) / (j - N0)
            if np.tan(50 * pi / 180) <= k <= np.tan(88 * pi / 180):
                mask_3[i][j] = 0
# 激活滤波器3
f.apply(mask_3)

# 定义滤波器4
mask_4 = np.ones([M, N])
M0 = M / 2
N0 = N / 2
for i in range(M):
    for j in range(N):
        if j != N0:
            k = (i - M0) / (j - N0)
            if np.tan(60 * pi / 180) <= k <= np.tan(70 * pi / 180):
                mask_4[i][j] = 0
# 激活滤波器4
f.apply(mask_4)

# 定义滤波器5
mask_5 = np.zeros([M, N])
M0 = M / 2
N0 = N / 2
for i in range(M):
    for j in range(N):
        if np.sqrt(((i - M0) ** 2 + (j - N0) ** 2)) <= 50:
            mask_5[i][j] = 1
# 激活滤波器5
f.apply(mask_5)

# 定义滤波器6
mask_6 = np.ones([M, N])
M0 = M / 2
N0 = N / 2
for i in range(M):
    for j in range(N):
        if np.sqrt(((i - M0) ** 2 + (j - N0) ** 2)) <= 40:
            mask_6[i][j] = 0
# 激活滤波器6
f.apply(mask_6)
