import matplotlib.pyplot as plt
import numpy as np
import cv2

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

# 对图像进行傅里叶变换并将零频分量移至频谱中心，展示原图像频谱
F = np.fft.fft2(image)
plt.imshow(abs(np.fft.fftshift(F)/(M*N)), cmap="jet")
plt.axis("off")
plt.show()

# 设置Butterworth滤波器
u0 = 20  # 设置截止频率
u = np.array([i for i in range(M)])
v = np.array([i for i in range(N)])
idx = np.argwhere(u>M/2)
u[idx] -= M
idy = np.argwhere(v>N/2)
v[idy] -= N
[V, U] = np.meshgrid(v, u)
H = np.zeros([M, N])
for i in range(M):
    for j in range(N):
        UVw = (U[i][j]*U[i][j] + V[i][j]*V[i][j])/(u0*u0)
        H[i][j] = 1/(1 + UVw*UVw)
H = np.fft.fftshift(H)

# 显示滤波器频谱
plt.imshow(H, cmap="gray")
plt.axis("off")
plt.show()

from filter.ppt import Filter
f = Filter(image)
f.apply(H)
