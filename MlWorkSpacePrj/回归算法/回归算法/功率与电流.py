# -*- coding: utf-8 -*-
# @Time    : 2018/11/28 上午10:42
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : 功率与电流.py
# @Software: PyCharm

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import time


from sklearn.model_selection import train_test_split # 数据划分的类
from sklearn.preprocessing import StandardScaler # 数据标准化
from sklearn.linear_model import LinearRegression # 线性回归的类
from sklearn.externals import joblib


## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus']= False


# 1 加载数据
dataPath = './datas/household_power_consumption_1000.txt'
df = pd.read_csv(dataPath,sep=';',low_memory=False)

# 日期、时间、有功功率、无功功率、电压、电流、厨房用电功率、洗衣机用电功率、热水器用电功率
# print(df.head())


# 2 数据清洗
new_df = df.replace('?',np.nan) #替换非法字符为np.nan

datas = new_df.dropna(axis=0, how='any') # 只要有一个数据为空，就进行行删除操作


# print(datas.describe()) #统计数据的指标 count mean std min 25% 50% 75% max
# print(datas.info()) #数据的格式信息

# 3 提取特征值和标签
X = datas.iloc[:,2:4]
Y = datas.iloc[:,5]
print(X.head(2))
print(Y.head(2))


# 4 数据分割
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8,test_size=0.2,random_state=0)


# 5数据标准化
ss = StandardScaler();
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)


# 6 模型训练
lr = LinearRegression(fit_intercept=True)
lr.fit(X_train,Y_train)

y_predict = lr.predict(X_test)

# 7 模型效果
print('训练集R2:',lr.score(X_train,Y_train))
print('测试集R2:',lr.score(X_test,Y_test))

## 预测值和实际值画图比较
t = np.arange(len(X_test))
plt.figure(facecolor='w')#建一个画布，facecolor是背景色
plt.plot(t, Y_test, 'r-', linewidth=2, label='真实值')
plt.plot(t, y_predict, 'g-', linewidth=2, label='预测值')
plt.legend(loc = 'upper left')#显示图例，设置图例的位置
plt.title("线性回归预测功率与电流之间的关系", fontsize=20)
plt.grid(b=True)#加网格
plt.show()




