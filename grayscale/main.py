from PIL import Image
import numpy as np

# 读入图像
img = Image.open('image.jpg')
# 将图像矩阵化
img_array = np.array(img)
# 输出图像大小
print(img_array.shape)
# img.show()

# 将图像转化为灰度图
L = img.convert('L')
# 将灰度图矩阵化
L_array = np.array(L)
# 输出灰度图大小
print(L_array.shape)
# L.show()
