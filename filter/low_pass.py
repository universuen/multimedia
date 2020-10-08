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

# 对图像进行傅里叶变换并将零频分量移至频谱中心，展示原图像频谱
F = np.fft.fft2(image)
plt.imshow(abs(np.fft.fftshift(F)/(M*N)), cmap="jet")
plt.axis("off")
plt.show()

# 设置低通滤波器
u0 = 20  # 设置截止频率
u = np.array([i for i in range(M)])
v = np.array([i for i in range(N)])
idx = np.argwhere(u>M/2)
u[idx] -= M
idy = np.argwhere(v>N/2)
v[idy] -= N
[V, U] = np.meshgrid(v, u)
D = np.sqrt(U**2 + V**2)
H = (D<=u0).astype('double')

# 显示将零频分量移至频谱中心后的滤波器频谱
plt.imshow(np.fft.fftshift(H), cmap="gray")
plt.axis("off")
plt.show()

# 对原图像进行滤波并做傅里叶逆变换
G = H*F
g = np.real(np.fft.ifft2(G))

# 显示滤波后的图片
plt.imshow(g, cmap="gray")
plt.axis("off")
plt.show()
