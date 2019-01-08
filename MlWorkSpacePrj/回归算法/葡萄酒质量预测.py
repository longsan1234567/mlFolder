# -*- coding: utf-8 -*-
# @Time    : 2018/12/12 上午10:31
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : 葡萄酒质量预测.py
# @Software: PyCharm

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import label_binarize
from sklearn import metrics


## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
## 拦截异常
warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)


# 1 加载数据

names =  ["fixed acidity","volatile acidity","citric acid","residual sugar",
          "chlorides","free sulfur dioxide","total sulfur dioxide","density",
          "pH","sulphates","alcohol","quality"]

path_red = 'datas/winequality-red.csv'
df_red = pd.read_csv(path_red,sep=';',header = None,names = names)
df_red['type'] = 1  # 1设置为红色葡萄酒
# print(df_red)

path_white = 'datas/winequality-white.csv'
df_white = pd.read_csv(path_white,sep=';',header = None,names = names)
df_white['type'] = 0 # 0 设置为白色葡萄酒

df = pd.concat([df_red,df_white],axis=1)


#  2 过滤异常数据
datas = df.replace("?", np.nan)
datas = df.dropna(axis=0, how='any')

# print(datas)


## 数据提取
X = datas[names[0:-1]]
Y = datas[names[-1]]


X = X.astype(np.float)
Y = Y.astype(np.float)

# 数据分割
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.1,train_size = 0.9,
                                                 random_state = 1)

print ("训练数据条数:%d；数据特征个数:%d；测试数据条数:%d"
       % (X_train.shape[0], X_train.shape[1], X_test.shape[0]))

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)

# 数据标准化
ss = MinMaxScaler() ## 归一化 将数据缩放在某一范围 (0,1)
X_train = ss.fit_transform(X_train)

lr = LogisticRegressionCV(Cs = np.logspace(-4,1,50),fit_intercept = True,
                          penalty = 'l2', solver = 'lbfgs',tol=0.01, multi_class='ovr')
lr.fit(X_train, Y_train)




