# -*- coding: utf-8 -*-
# @Time    : 2018/11/5 下午2:31
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : 加载模型训练测试.py
# @Software: PyCharm

'''
    加载磁盘中的训练模型对数据进行预测
'''
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.externals import joblib

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1 加载模型
ss = joblib.load('./model/ss.m')
algo = joblib.load('./model/lr.m')

# 2 加载数据
path = './datas/household_power_consumption_201.txt'
df = pd.read_csv(path,sep=';')

# 3 清洗数据
df.replace('?', np.nan, inplace=True)
df.dropna(axis=0, how='any', inplace=True)

# 4 需要预测的数据
X = df[['Global_active_power', 'Global_reactive_power']]
Y = df['Global_intensity']

X.type = np.float64
Y.type = np.float64

# 5 查看预测结果
y_predict = algo.predict(ss.transform(X))
print('预测结果{}'.format(y_predict))
print('预测效果得分{}'.format(algo.score(ss.transform(X),Y)))



# 画图直观展示
t = np.arange(len(X))
plt.figure(facecolor='w')
plt.plot(t, y_predict, 'g-', linewidth=2, label=u'预测值')
plt.plot(t, Y, 'r-', linewidth=2, label=u'真实值')
plt.legend(loc='lower right')
plt.title('加载模型-->线性回归')
plt.show()



