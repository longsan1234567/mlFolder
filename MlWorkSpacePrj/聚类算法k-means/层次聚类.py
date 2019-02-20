# -*- coding: utf-8 -*-
# @Time    : 2019/1/26 3:37 PM
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : 层次聚类.py
# @Software: PyCharm

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph ## KNN的K近邻计算
import sklearn.datasets as ds
import warnings

## 设置属性防止中文乱码及拦截异常信息
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings(action='ignore', category=UserWarning)


np.random.seed(0)
n_clusters = 4
centers = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
N = 3000
X1,Y1 = ds.make_blobs(n_samples=N,centers=centers,cluster_std=0.5,random_state=12)

n_noise = int(0.1*N)
print(n_noise)
r = np.random.rand(n_noise,2)
min1,min2 = np.min(X1,axis=0)
max1,max2 = np.max(X1,axis=0)
r[:,0] = r[:,0] * (max1-min1)+ min1
r[:,1] = r[:,1] * (max2-min2) + min2


data1_noise = np.concatenate((X1,r),axis=0)
y1_noise = np.concatenate((Y1,[4]*n_noise))


# 模拟月牙形数据
X2,Y2 = ds.make_moons(n_samples=N,noise=0.05)
X2 = np.array(X2)

n_noise = int(0.1*N)
r = np.random.rand(n_noise,2)
min1,min2 = np.min(X2,axis=0)
max1,max2 = np.max(X2,axis=0)
r[:,0] = r[:,0] * (max1-min1)+ min1
r[:,1] = r[:,1] * (max2-min2) + min2

data2_noise = np.concatenate((X2,r),axis=0)
y2_noise = np.concatenate((Y2,[3] * n_noise))

def expandBorder(a,b):
    d = (b - a) * 0.1
    return a-d,b+d;

# 绘图
cm = mpl.colors.ListedColormap(['#FF0000', '#00FF00',  '#0000FF', '#ffd966', '#5c5a5a'])
plt.figure(figsize=(14,12),facecolor='w')
linkages = ("ward", "complete", "average")#把几种距离方法，放到list里，后面直接循环取值 建议使用word策略对样本进行分类。

# (4: 表示 4个簇中心 2: 表示 2个簇中心)
for index,(n_clusters,data,y) in enumerate(((4, X1, Y1), (4, data1_noise, y1_noise),
                                           (2, X2, Y2), (2, data2_noise, y2_noise))):


    plt.subplot(4,4,4*index+1) #前面的两个4表示几行几列 第三个参数表示第几个图(从1开始数 从左到右)
    plt.scatter(data[:,0],data[:,1],c = y,cmap=cm)
    plt.title(u'原始数据',fontsize = 11)
    plt.grid(b=True,ls=':')
    min1,min2 = np.min(data,axis=0)
    max1,max2 = np.max(data,axis=0)
    plt.xlim(expandBorder(min1,max1))
    plt.ylim(expandBorder(min2,max2))

    # 计算类别与类别的距离(只计算最接近的七个样本的距离) -- 希望在agens算法中，在计算过程中不需要重复性的计算点与点之间的距离
    connectivity = kneighbors_graph(data, n_neighbors=7, mode='distance', metric='minkowski', p=2, include_self=False)
    connectivity = (connectivity + connectivity.T)
    for i, linkage in enumerate(linkages):
        ##进行建模，并传值
        print(n_clusters)
        ac = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',
                                     connectivity=connectivity, linkage=linkage)
        ac.fit(data)
        # 获取得到训练数据data所对应的标签值
        y = ac.labels_

        plt.subplot(4, 4, i+2+4*index)
        plt.scatter(data[:, 0], data[:, 1], c=y, cmap=cm)
        plt.title(linkage, fontsize=11)
        plt.grid(b=True, ls=':')
        plt.xlim(expandBorder(min1, max1))
        plt.ylim(expandBorder(min2, max2))


plt.suptitle(u'AGNES层次聚类的不同合并策略', fontsize=15)
plt.tight_layout(0.5, rect=(0, 0, 1, 0.95))
plt.show()















