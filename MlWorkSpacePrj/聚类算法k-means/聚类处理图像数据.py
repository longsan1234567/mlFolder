# -*- coding: utf-8 -*-
# @Time    : 2019/1/14 11:16 AM
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : 聚类处理图像数据.py
# @Software: PyCharm
import numpy as np
from PIL import Image

image = Image.open('./xiaoren.png')
print(image)
image = np.array(image)
print(image)
print(image.shape)


image = np.reshape(image,(600 * 510,3))
print(image.shape)
print(image)