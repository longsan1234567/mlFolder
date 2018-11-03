# -*- coding: utf-8 -*-
# @Time    : 2018/11/1 下午11:18
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : 一维二维图形.py
# @Software: PyCharm
'''
 梯度下降法
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# 一维函数
def func1(x):
    return 0.5 * ( x - 0.25) **2

X = np.arange(-4,4.5,0.01)
Y = np.array(list(map(lambda t:func1(t),X))) ## ? lambda 表达式


# # 画图
# plt.figure(facecolor='w')
# plt.plot(X, Y, 'r-', linewidth=2)
# plt.title(u'函数$y=0.5 * (θ - 0.25)^2$')
# plt.show()
# 效果对应 一维梯度图片

# 二维图形
def func2(x1,x2):
    return 0.6 * (x1+x2)**2 - x1*x2


X1 = np.arange(-4,4.5,0.2)
X2 = np.arange(-4,4.5,0.2)

X1,X2 = np.meshgrid(X1,X2)

# print(X1,X2)

Y2 = np.array(list(map(lambda t: func2(t[0],t[1]), zip(X1.flatten(),X2.flatten()))))

Y2.shape = X1.shape;


# 画图
fig = plt.figure(facecolor='w')
ax = Axes3D(fig)
ax.plot_surface(X1, X2, Y2, rstride = 1, cstride=1, cmap = plt.cm.jet)
ax.set_title(u'函数$y=0.6 * (θ1 + θ2)^2 - θ1 * θ2$')
plt.show()
# 效果:二维梯度图片








