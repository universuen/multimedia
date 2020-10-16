import matplotlib.pyplot as plt
import numpy as np

# 构建图片
M = 256
N = 256
image = np.zeros([M, N])
box = np.ones([64, 64])
image[97:161, 97:161] = box

# 显示图片
plt.imshow(image, cmap="gray")
plt.axis("off")
plt.show()

# 对图像进行傅里叶变换并将将零频分量移到频谱中心，展示原图像频谱
F = np.fft.fft2(image)
plt.imshow(abs(np.fft.fftshift(F) / (M * N)), cmap="jet")
plt.axis("off")
plt.show()

from filter.ppt import Filter
f = Filter(image)
# 定义理想低通滤波器
mask_lp = np.zeros([M, N])
M0 = M / 2
N0 = N / 2
for i in range(M):
    for j in range(N):
        if np.sqrt(((i - M0) ** 2 + (j - N0) ** 2)) <= 20:
            mask_lp[i][j] = 1

# 显示滤波器图像
plt.imshow(mask_lp, cmap="gray")
plt.axis("off")
plt.show()
# 激活理想低通滤波器
f.apply(mask_lp)
