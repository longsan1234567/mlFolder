# -*- coding: utf-8 -*-
# @Time    : 2019/1/26 10:50 AM
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : K-Means算法和Mini Batch K-Means算法效果评估.py 了解即可
# @Software: PyCharm

import matplotlib
matplotlib.use("TkAgg")
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


centers = [[1,1],[-1,-1],[1,-1]]
clusters = len(centers)

X,Y = make_blobs(n_samples=3000,centers=centers,cluster_std=0.5,random_state=12)


'''
 目的: 比较k-means算法和mini-batch k-means算法耗时

 结果: k-means算法消耗时间:0.0700s
      Mini Batch K-Means算法消耗时间:0.0160s
'''
k_means = KMeans(init='k-means++',n_clusters=clusters,random_state=12)
tkbt = time.time()
k_means.fit(X)
kmt = time.time()-tkbt
print('k-means算法消耗时间:%.4fs'%kmt)

batch_size = 1000
mbk_means = MiniBatchKMeans(init='k-means++',n_clusters=clusters,
                            batch_size=batch_size,random_state=12)
mbbt = time.time()
mbk_means.fit(X)
mbt = time.time()-mbbt
print('Mini Batch K-Means算法消耗时间:%.4fs'%mbt)


'''
  目的:输出样本所属的类别

  [0 2 2 ... 2 1 0]
  [1 2 2 ... 2 0 1]

'''
km_y_hat = k_means.labels_
mbkm_y_hat = mbk_means.labels_
print(km_y_hat)
print(mbkm_y_hat)

'''
 目的: 聚类算法聚类中心
'''
k_means_cluster_centers = k_means.cluster_centers_
mb_means_cluster_centers = mbk_means.cluster_centers_
print('k-means算法聚类中心点%s'%k_means_cluster_centers)
print('Mini Batch K-Means算法聚类中心点%s'%mb_means_cluster_centers)
order = pairwise_distances_argmin(k_means_cluster_centers,mb_means_cluster_centers)
# 计算X中每个样本与k_means_cluster_centers中的哪个样本最近。也就是获取所有对象的所属的类标签
print(order)



'''
 目的: 模型效果评估

 结果: k-means算法:adjusted_rand_score评估函数
      计算的评估结果值:0.9181
    耗时时间:0.001
    Mini Batch K-Means算法:adjusted_rand_score评估函数
    计算的评估结果值:0.9173
    耗时时间:0.001


    k-means算法:v_measure_score评估函数
    计算的评估结果值:0.8754
    耗时时间:0.001
    Mini Batch K-Means算法:v_measure_score评估函数
    计算的评估结果值:0.8745
    耗时时间:0.001


    k-means算法:adjusted_mutual_info_score评估函数
    计算的评估结果值:0.8753
     耗时时间:0.003
    Mini Batch K-Means算法:adjusted_mutual_info_score评估函数
    计算的评估结果值:0.8744
    耗时时间:0.003


    k-means算法:mutual_info_score评估函数
    计算的评估结果值:0.9617
     耗时时间:0.001
    Mini Batch K-Means算法:mutual_info_score评估函数
    计算的评估结果值:0.9608
    耗时时间:0.001

'''
score_funcs = [
    metrics.adjusted_rand_score,#ARI
    metrics.v_measure_score,#均一性和完整性的加权平均
    metrics.adjusted_mutual_info_score,#AMI
    metrics.mutual_info_score,#互信息
]

# 迭代对每个评估函数进行评估操作
for score_func in score_funcs:
    t0 = time.time()
    km_scores = score_func(Y,km_y_hat)
    print('k-means算法:%s评估函数\n计算的评估结果值:%.4f\n 耗时时间:%.3f'%
          (score_func.__name__,km_scores,time.time()-t0))



    t0 = time.time()
    mbkm_scores = score_func(Y,mbkm_y_hat)
    print('Mini Batch K-Means算法:%s评估函数\n 计算的评估结果值:%.4f\n 耗时时间:%.3f'
          %(score_func.__name__,mbkm_scores,time.time()-t0))

    print('\n')


'''
  目的 使用轮廓系数评估模型

  轮廓系数:越接近1表示样本i聚类越合理 越接近-1 样本i越属于另外的簇  近似等于0 表示样本i在簇的边界上

  结果: K-Means算法的轮廓系数指标 (簇中心:3个):0.9617
       Mini Batch K-means轮廓系数指标:0.9608
'''
km_score = metrics.silhouette_score(X,km_y_hat)
mbkm_score = metrics.silhouette_score(X,mbkm_y_hat)

print('K-Means算法的轮廓系数指标 (簇中心:%d个):%.4f'%(clusters,km_scores))
print('Mini Batch K-means轮廓系数指标:%.4f'%(mbkm_scores))




























