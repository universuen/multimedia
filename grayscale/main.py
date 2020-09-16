from PIL import Image
import numpy as np

img = Image.open('image.jpg')
img_array = np.array(img)
print(img_array.shape)
img.show()

L = img.convert('L')
L_array = np.array(L)
print(L_array.shape)
L.show()