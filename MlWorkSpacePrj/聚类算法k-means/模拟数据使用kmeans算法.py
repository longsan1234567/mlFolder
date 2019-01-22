# -*- coding: utf-8 -*-
# @Time    : 2019/1/14 9:50 AM
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : 模拟数据使用kmeans算法.py
# @Software: PyCharm

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

N = 1000
n_centers = 4
X,Y = make_blobs(n_samples=N,n_features=2,centers=n_centers,random_state=12)


# 1# 模型构建
# algo = KMeans(n_clusters=n_centers)
# algo.fit(X)


# 2 模型构建使用网格交叉验证
'''
定义: 将原始数据集划分为两部分 一部分分为训练集 训练模型 另一部分分为测试集合验证模型效果
交叉验证的目的是为了验证训练模型的拟合程度

网格交叉验证(网格搜索) GridSearchCV 对估计器的指定参数值穷举搜索  通过给定不同参数值的组合 验证选取一组最优的参数parameters
'''
parameters = {
    'n_clusters':[2,3,4,5,6],
    'random_state':[0,14,28]
}
model = KMeans(n_clusters=n_centers)
algo = GridSearchCV(estimator=model,param_grid=parameters,cv=5)
algo.fit(X)


# 数据预测
x_test =[
    [-4, 8],
    [-3, 7],
    [0, 5],
    [0, -5],
    [8, -8],
    [7, -9]
]
# 预测值:[2 2 3 3 1 1]
print('预测值:{}'.format(algo.predict(x_test)))

# print("中心点坐标{}".format(algo.cluster_centers_))
# print("目标函数的损失函数:(所有样本到簇中心点的距离平方和)")
#
# print(algo.inertia_)
# print(algo.inertia_/N)

## 交叉验证
print("最优模型参数{}".format(algo.best_params_))
print('中心点坐标{}'.format(algo.best_estimator_.cluster_centers_))
print("目标函数损失值{}".format(algo.best_estimator_.inertia_))
print(algo.best_estimator_.inertia_/N)


plt.scatter(X[:,0],X[:,1],c = Y,s=30)
plt.show()