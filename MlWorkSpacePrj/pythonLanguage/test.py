# -*- coding: utf-8 -*-
# @Time    : 2018/11/1 下午10:25
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : test.py
# @Software: PyCharm

import sklearn
import numpy as np


print(sklearn.__version__)


x1 = np.linspace(1,10) # 等差数列 默认返回50条
x2 = np.linspace(1,10,10) # 指定返回数据条数
# print(x1)
# print(x2)

'''
    1 np.linspace() 等差数列
'''
X = np.linspace(0,1,3)
Y = np.linspace(0,1,5)
xv,yv = np.meshgrid(X,Y)

'''
   2 meshgrid(*xi, **kwargs)
    功能：从一个坐标向量中返回一个坐标矩阵
    新产生的矩阵维度  是数据的最大个数 sparse:True x的维度就是X的个数 Y就是y的个数
'''
# xv,yv = np.meshgrid(X,Y,sparse=True)
print(xv)
print('-------------------')
print(yv)

# [[0.  0.5 1. ]
#  [0.  0.5 1. ]
#  [0.  0.5 1. ]
#  [0.  0.5 1. ]
#  [0.  0.5 1. ]]
# -------------------
# [[0.   0.   0.  ]
#  [0.25 0.25 0.25]
#  [0.5  0.5  0.5 ]
#  [0.75 0.75 0.75]
#  [1.   1.   1.  ]]

'''
   3 numpy.ndarray.flatten()函数
   功能: 返回一个折叠成一维的数组。但是该函数只能适用于numpy对象，即array或者mat，普通的list列表是不行的。
'''
a = np.array([[1,2],[3,4],[5,6]])  ###此时a是一个array对象

print(a.flatten())
# out: [1 2 3 4 5 6]


'''
    zip() 函数
    功能: 将可迭代的对象 打包成元组列表返回
'''

c = [1,2,3]
d = [4,5,6]
print(list(zip(c,d)))
# out [(1, 4), (2, 5), (3, 6)]








