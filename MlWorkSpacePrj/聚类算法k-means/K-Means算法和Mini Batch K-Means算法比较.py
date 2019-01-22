# -*- coding: utf-8 -*-
# @Time    : 2019/1/22 2:18 PM
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : K-Means算法和Mini Batch K-Means算法比较.py
# @Software: PyCharm

import matplotlib
matplotlib.use("TkAgg")
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 初始化三个中心
centers = [[1,1],[-1,-1],[1,-1]]
clusters = len(centers)

# 产生3000组2维数据 中心三个中心  标准差0.5
X,Y = make_blobs(30000,centers = centers,cluster_std = 0.5,random_state = 12)


# 构建kmeans算法
k_means = KMeans(init = 'k-means++',n_clusters = clusters,random_state = 12)
tk_time = time.time()
k_means.fit(X)
tk_second = time.time()-tk_time
print("kmeans算法模型训练耗时%.4fs"%(tk_second))

# 构建MiniBatchKMeans算法
batch_size = 100
mbk = MiniBatchKMeans(init='k-means++',n_clusters = clusters,batch_size = batch_size,random_state = 12)
mb_time = time.time()
mbk.fit(X)
mb_second =  time.time()-mb_time
print("MiniBatchKMeans算法模型训练耗时%.4fs"%(mb_second))

# 预测结果
km_prd = k_means.predict(X)
mb_prd = mbk.predict(X)
print("K-Means算法预测值%s"%km_prd[:5])
print("MiniBatchK-Means算法预测值%s"%mb_prd[:5])

# 获取簇的中心点 并按簇类中心点排序
k_means_cluster_center = k_means.cluster_centers_
mbk_means_cluster_centers = mbk.cluster_centers_

'''
pairwise_distances_argmin默认情况下 该API的功能是将X,Y的元素做一个按大到小的排序
然后将排序后的X,Y的值两两组合
API 实际返回是针对x中的每个元素中的对应y中的每个值得下标索引
'''
order = pairwise_distances_argmin(
    X=k_means_cluster_center,Y=mbk_means_cluster_centers)
print(order)


## 画图
plt.figure(figsize=(12,6),facecolor='w')
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.9)
cm = mpl.colors.ListedColormap(['#FFC2CC', '#C2FFCC', '#CCC2FF'])
cm2 = mpl.colors.ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


# 子图1:原始数据
plt.subplot(221)
plt.scatter(X[:,0],X[:,1],c=Y,s=6,cmap=cm,edgecolors='none')
plt.title(u'原始数据分布图')
plt.xticks(())
plt.yticks(())
plt.grid(True)


# 子图2 K-Means算法聚类结果图
plt.subplot(222)
plt.scatter(X[:,0],X[:,1],c=km_prd,s=6,cmap=cm,edgecolors='none')
plt.scatter(k_means_cluster_center[:,0],k_means_cluster_center[:,1],
            c=range(clusters),s=60,cmap=cm2,edgecolors='none')
plt.title(u'k-means算法聚类结果图')
plt.xticks(())
plt.yticks(())
plt.text(-2.8,3,'train time:%.2fms'%(tk_second*1000))
plt.grid(True)


# 子图3 mini batch k-means算法聚类算法
plt.subplot(223)
plt.scatter(X[:,0],X[:,1],c=mb_prd,s=6,cmap=cm,edgecolors='none')
plt.scatter(mbk_means_cluster_centers[:,0],mbk_means_cluster_centers[:,1],
            c= range(clusters),s=60,cmap=cm2,edgecolors='none')
plt.title(u'mini batch k-means算法聚类结果图')
plt.xticks(())
plt.yticks(())
plt.text(-2.8,3,'train time:%.2fms'%(mb_second*1000))
plt.grid(True)


# 子图4 找到两种算法中预测的不同的样本数目

different = list(map(lambda x: (x!=0) & (x!=1) & (x!=2), mb_prd))
for k in range(clusters):
    different += ((km_prd == k) != (mb_prd == order[k]))
identic = np.logical_not(different)
different_nodes = len(list(filter(lambda x:x, different)))


plt.subplot(224)
# 两者预测相同的
plt.plot(X[identic, 0], X[identic, 1], 'w', markerfacecolor='#bbbbbb', marker='.')
# 两者预测不相同的
plt.plot(X[different, 0], X[different, 1], 'w', markerfacecolor='r', marker='.')
plt.title(u'Mini Batch K-Means和K-Means算法预测结果不同的点')
plt.xticks(())
plt.yticks(())
plt.text(-2.8, 2,  'different nodes: %d' % (different_nodes))

plt.show()














