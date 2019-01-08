# -*- coding: utf-8 -*-
# @Time    : 2018/12/14 上午11:22
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : KNN.py
# @Software: PyCharm

'''
莺尾花分类问题 使用KNN算法构建模型进行预测
'''


import numpy as np
import pandas as  pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier ##KNN分类
from sklearn.preprocessing import StandardScaler # 数据标准化

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False


## 1 数据加载
path = "datas/iris.data"
names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'cla']
df = pd.read_csv(path, header=None, names=names)
# print(df.T.head())


# 2 数据处理
def parseRecord(record):
    result = []
    r = zip(names,record)
    for name,v in r:
        if name == 'cla':
            if v ==  'Iris-setosa':
                result.append(1)
            elif v == 'Iris-versicolor':
                result.append(2)
            elif v == 'Iris-virginica':
                result.append(3)
            else:
                result.append(np.nan)
        else:
            result.append(float(v))
    return result

# 数据转换 axis = 1 表示行读取 pd.Series() 行数据
datas = df.apply(lambda r: pd.Series(parseRecord(r),index=names),axis=1)
# print(datas.head())

# 异常数据删除
datas = datas.dropna(how='any')
# print(datas.head())

# 3 数据分割处理
X = datas[names[0:-1]]
Y = datas[names[-1]]
# print(X.shape)
# print(Y.shape)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8,test_size=0.2,random_state=1)

print('原始数据个数%d 训练数据个数%d 训练数据数据特征个数%d 测试数据条数%d'%(len(X),len(X_train),
                                                 X_train.shape[1],X_test.shape[0]))

# 数据标准化
# ss = StandardScaler()
# X_train = ss.fit_transform(X_train)
# X_test = ss.fit_transform(X_test)
#

# 5 加载模型  n_neighbors 临近数目
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,Y_train)



# 6 预测与 模型效果
knn_predict_y = knn.predict(X_test)
print('KNN算法准确率:',knn.score(X_train,Y_train))

## 画图2：预测结果画图
x_test_len = range(len(X_test))
plt.figure(figsize=(12, 9), facecolor='w')
plt.ylim(0.5,3.5)
plt.plot(x_test_len, Y_test, 'ro',markersize = 6, zorder=3, label=u'真实值')
plt.plot(x_test_len, knn_predict_y, 'yo', markersize = 16, zorder=1,
         label=u'KNN算法预测值,准确率=%.3f' % knn.score(X_test, Y_test))
plt.legend(loc = 'upper right')
plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'种类', fontsize=18)
plt.title(u'鸢尾花数据分类', fontsize=20)
plt.show()






