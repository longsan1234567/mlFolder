# -*- coding: utf-8 -*-
# @Time    : 2019/1/11 10:39 AM
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : 高斯分布数据.py
# @Software: PyCharm

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

'''
make_blobs: 产生一个服从给定均值和标准差的高斯分布
最终返回数据样本组成的x特征矩阵 以及样本对应的类别(当前数据属于哪个均值哪个标准差的数据分布)
'''
x,y = make_blobs(n_samples=50,n_features=2,centers=3)
# 50个样本 2个特征 3个簇中心


print(x)
print(y)
print(x.shape)
print(y.shape)
# (50, 2)
# (50,)

# s表示点的半径大小
plt.scatter(x[:, 0], x[:, 1], c=y, s=3)
plt.show()


