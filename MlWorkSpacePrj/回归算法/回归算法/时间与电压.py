# -*- coding: utf-8 -*-
# @Time    : 2018/11/28 下午3:08
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : 时间与电压.py
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus']= False


# # 1 加载数据
# dataPath = './datas/household_power_consumption_1000.txt'
# df = pd.read_csv(dataPath,sep=';',low_memory=False)
#
#
# ## 创建一个时间字符串格式化字符串
# def date_format(dt):
#     import time
#     t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
#     return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)
#
#
#
# # 2 数据清洗
# new_df = df.replace('?',np.nan) #替换非法字符为np.nan
#
# datas = new_df.dropna(axis=0, how='any') # 只要有一个数据为空，就进行行删除操作
#
#
#
# #3 提取特征值
# X = datas.iloc[:,0:2]; # 取时间 date time
# X = X.apply(lambda x:pd.Series(date_format(x)),axis=1)
# Y = datas['Voltage']
#
# X = X.astype(np.float)
# Y = Y.astype(np.float)
# print(X)
# print(Y)
#
#
# #4 划分数据集
# X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,train_size=0.8,random_state=0)
#
# print(X_train.shape)
# #5 数据标准化
# ss = StandardScaler();
# X_train = ss.fit_transform(X_train)
# X_test = ss.fit_transform(X_test)
#
#
# #6 训练模型
# lr = LinearRegression(fit_intercept=True)
# lr.fit(X_train,Y_train)
#
# #7 预测结果
# y_predict = lr.predict(X_test)
#
# # 检查模型效果
# print('训练集的R2:',lr.score(X_train,Y_train))
# print('测试集的R2:',lr.score(X_test,Y_test))
# print('mse:',np.average((y_predict-Y_test)**2))
#
# t = np.arange(len(X_test))
# plt.figure(facecolor='w')#建一个画布，facecolor是背景色
# plt.plot(t, Y_test, 'r-', linewidth=2, label='真实值')
# plt.plot(t, y_predict, 'g-', linewidth=2, label='预测值')
# plt.legend(loc = 'upper left')#显示图例，设置图例的位置
# plt.title("线性回归预测时间和电压之间的关系", fontsize=20)
# plt.grid(b=True)#加网格
# plt.show()
#


'''
 多项式拓展
'''
# 1 加载数据
dataPath = './datas/household_power_consumption_1000.txt'
df = pd.read_csv(dataPath,sep=';',low_memory=False)


## 创建一个时间字符串格式化字符串
def date_format(dt):
    import time
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)



# 2 数据清洗
new_df = df.replace('?',np.nan) #替换非法字符为np.nan

datas = new_df.dropna(axis=0, how='any') # 只要有一个数据为空，就进行行删除操作



#3 提取特征值
X = datas.iloc[:,0:2]; # 取时间 date time
X = X.apply(lambda x:pd.Series(date_format(x)),axis=1)
Y = datas['Voltage']

X = X.astype(np.float)
Y = Y.astype(np.float)
print(X)
print(Y)


#4 划分数据集
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,train_size=0.8,random_state=0)

print(X_train.shape)


# 5 模型训练
## 管道流操作
model = Pipeline([
    ('Poly',PolynomialFeatures()),
    ('Linear',LinearRegression(fit_intercept=True))
])

t = np.arange(len(X_test))
print(t)    # [1,2....199]

N = 5
d_pool = np.arange(1,N,1)

print(d_pool)   #[1,2,3,4]
m = d_pool.size;

# 生成颜色集合
clrs = []
for c in  np.linspace(16711680,255,m):
    clrs.append('#%06x'%int(c))

linewidth = 1;
print(enumerate(d_pool))
for i,d in enumerate(d_pool):
    plt.subplot(N-1,1,i+1)
    plt.plot(t,Y_test,'g-',linewidth = linewidth, label=u'真实值',ms = 10,zorder = N)

    model.set_params(Poly__degree = d)  ## 设置多项式的阶乘
    model.fit(X_train,Y_train) #模型训练


    y_predict = model.predict(X_test)   # 模型预测
    label = u'预测值%d阶,R2 = %.3f'%(d,model.score(X_test,Y_test))
    plt.plot(t, y_predict, color = clrs[i], linewidth=linewidth, label=label)
    plt.legend(loc = 'upper left')#显示图例，设置图例的位置
    plt.grid(True)
    plt.ylabel(u'%d阶结果'%d,fontsize=10)

plt.suptitle("时间与电压多项式拓展关系", fontsize=15)
plt.grid(b=True)#加网格
plt.show()


























