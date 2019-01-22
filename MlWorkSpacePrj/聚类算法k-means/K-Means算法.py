# -*- coding: utf-8 -*-
# @Time    : 2019/1/14 11:23 AM
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : K-Means算法.py
# @Software: PyCharm
import matplotlib
matplotlib.use("TkAgg")
import numpy as  np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import KMeans #引入kmeans



## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# 1 生成模拟数据
N = 1500
centers = 4

# 产生等方差的数据集(中心点随机)
data1,y1 = ds.make_blobs(N,n_features=2,centers=centers,random_state=12)
# 产生指定中心点和方差的数据集
data2,y2 = ds.make_blobs(N,n_features=2,centers= [(-10,-8), (-5,8), (5,2), (8,-7)],cluster_std=[1.5, 2.5, 1.9, 1],random_state=12)
# 产生方差相同 样本数量不同的数据集
data3 = np.vstack((data1[y1 == 0][:200],
                   data1[y1 == 1][:100],
                   data1[y1 == 2][:10],
                   data1[y1 == 3][:50]))

y3 = np.array([0] * 200 + [1] * 100 + [2] * 10 + [3] * 50)

# 2 模型构建
km = KMeans(n_clusters=centers,init='random',random_state=12)
km.fit(data1)


# 模型预测
y_hat = km.predict(data1)
print('所有样本距离簇中心点的总距离和:',km.inertia_)
print('距离聚簇中的平均距离',(km.inertia_/N))
cluster_centers = km.cluster_centers_
print('聚簇中心点\n',cluster_centers)

y_hat2 = km.fit_predict(data2)
y_hat3 = km.fit_predict(data3)

def expandBorder(a,b):
    d = (b - a) * 0.1
    return  a-d,b+d

# 绘图
cm = mpl.colors.ListedColormap(list('rgbmyc'))
plt.figure(figsize=(15, 9), facecolor='w')

plt.subplot(241)
plt.scatter(data1[:, 0], data1[:, 1], c=y1, s=30, cmap=cm, edgecolors='none')
x1_min, x2_min = np.min(data1, axis=0)
x1_max, x2_max = np.max(data1, axis=0)
x1_min, x1_max = expandBorder(x1_min, x1_max)
x2_min, x2_max = expandBorder(x2_min, x2_max)
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.title(u'原始数据')
plt.grid(True)



plt.subplot(242)
plt.scatter(data1[:, 0], data1[:, 1], c = y_hat, s=30, cmap=cm, edgecolors='none')
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.title(u'K-Means算法聚类结果')
plt.grid(True)


# 对数据做一个旋转
m = np.array(((1, -5), (0.5, 5)))
data_r = data1.dot(m)
y_r_hat = km.fit_predict(data_r)


plt.subplot(243)
plt.scatter(data_r[:, 0], data_r[:, 1], c=y1, s=30, cmap=cm, edgecolors='none')
x1_min, x2_min = np.min(data_r, axis=0)
x1_max, x2_max = np.max(data_r, axis=0)
x1_min, x1_max = expandBorder(x1_min, x1_max)
x2_min, x2_max = expandBorder(x2_min, x2_max)
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.title(u'数据旋转后原始数据图')
plt.grid(True)


plt.subplot(244)
plt.scatter(data_r[:, 0], data_r[:, 1], c=y_r_hat, s=30, cmap=cm, edgecolors='none')
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.title(u'数据旋转后预测图')
plt.grid(True)

plt.subplot(245)
plt.scatter(data2[:, 0], data2[:, 1], c=y2, s=30, cmap=cm, edgecolors='none')
x1_min, x2_min = np.min(data2, axis=0)
x1_max, x2_max = np.max(data2, axis=0)
x1_min, x1_max = expandBorder(x1_min, x1_max)
x2_min, x2_max = expandBorder(x2_min, x2_max)
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.title(u'不同方差的原始数据')
plt.grid(True)

plt.subplot(246)
plt.scatter(data2[:, 0], data2[:, 1], c=y_hat2, s=30, cmap=cm, edgecolors='none')
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.title(u'不同方差簇数据的K-Means算法聚类结果')
plt.grid(True)

plt.subplot(247)
plt.scatter(data3[:, 0], data3[:, 1], c=y3, s=30, cmap=cm, edgecolors='none')
x1_min, x2_min = np.min(data3, axis=0)
x1_max, x2_max = np.max(data3, axis=0)
x1_min, x1_max = expandBorder(x1_min, x1_max)
x2_min, x2_max = expandBorder(x2_min, x2_max)
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.title(u'不同簇样本数量原始数据图')
plt.grid(True)

plt.subplot(248)
plt.scatter(data3[:, 0], data3[:, 1], c=y_hat3, s=30, cmap=cm, edgecolors='none')
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.title(u'不同簇样本数量的K-Means算法聚类结果')
plt.grid(True)

plt.tight_layout(2, rect=(0, 0, 1, 0.97))
plt.suptitle(u'数据分布对KMeans聚类的影响', fontsize=18)

plt.show()

# 使用轮廓系数作为衡量指标
from sklearn import metrics
km22 = KMeans(n_clusters=4, init='random',random_state=28)
km22.fit(data1, y1)
y_hat22 = km22.predict(data1)
km_score2 = metrics.silhouette_score(data1, y_hat22)
print("KMeans算法的轮廓系数指标（簇中心:%d）:%.3f" % (len(km22.cluster_centers_), km_score2))


























