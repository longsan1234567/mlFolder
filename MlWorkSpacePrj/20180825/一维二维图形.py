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


# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# 一维函数
def func1(x):
    return 0.5 * ( x - 0.25) **2

X = np.arange(-4,4.5,0.01)
Y = np.array(list(map(lambda t:func1(t),X))) ## ? lambda 表达式

# print(X)
# print(Y)

# 画图
# plt.figure(facecolor='w')
# plt.plot(X, Y, 'r-', linewidth=2)
# plt.title(u'函数$y=0.5 * (θ - 0.25)^2$')
# plt.show()

li = [11,22,33]
new_list = list(map(lambda x : x+1,li))
print(new_list)





