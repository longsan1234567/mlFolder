# -*- coding: utf-8 -*-
# @Time    : 2018/11/5 下午1:58
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : 最小二乘.py
# @Software: PyCharm


'''
 最小二乘算法
 最小化误差的平方寻找数据的最佳函数
 θ = (XTX)−1XTY
'''

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
# path = '../datas/household_power_consumption_200.txt'
path = './datas/household_power_consumption_1000.txt'
df = pd.read_csv(path, sep=';')
# print(df.head(2))
# df.info()

# 2. 获取功率值作为作为特征属性X，电流作为目标属性Y
X = df.iloc[:,2:4]
# print(X.head(2))
Y = df.iloc[:,5]

# 3. 获取训练数据和测试数据
train_x,test_x,train_y,test_y = train_test_split(X, Y, train_size=0.8,test_size=0.2, random_state=28)
print("总样本数目:{}, 训练数据样本数目:{}, 测试数据样本数目:{}"
      .format(X.shape, train_x.shape, test_x.shape))

# 4. 训练模型
# a. 训练数据转换为矩阵形式
x = np.mat(train_x)
y = np.mat(train_y).reshape(-1, 1)
# b. 训练模型参数θ值
theta = (x.T * x).I * x.T * y
print(theta.shape)
print("求解出来的theta值:{}".format(theta))

# 5. 模型效果评估
y_hat = np.mat(test_x) * theta


# 画图看一下效果
t = np.arange(len(test_x))
plt.figure(facecolor='w')
plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测值')
plt.plot(t, test_y, 'r-', linewidth=2, label=u'真实值')
plt.legend(loc='lower right')
plt.title('最小二乘线性回归')
plt.show()


theta1 = theta[0]
theta2 = theta[1]
Global_active_power  = 4.216
Global_reactive_power  = 0.418
print('预测值{}'.format(theta1 * Global_active_power + theta2 * Global_reactive_power))














