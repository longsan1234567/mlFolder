# -*- coding: utf-8 -*-
# @Time    : 2018/12/26 下午3:56
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : 决策树模型可视化.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import pydotplus
from sklearn import tree

from sklearn.model_selection  import train_test_split #测试集和训练集
from sklearn.preprocessing import MinMaxScaler #归一化处理库

from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import SelectKBest #特征选择
from sklearn.feature_selection import chi2 #卡方统计量
from sklearn.decomposition import PCA #主成分分析

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=FutureWarning)


path = './datas/iris.data'
data = pd.read_csv(path, header = None)
x = data[list(range(4))]
# print(np.array(data[4]))
y = pd.Categorical(data[4]).codes  #这个分类的分类代码
# print(y)
print("总样本数目:%d 特征属性数目%d"%(x.shape))

x_train, x_test, y_train, y_test = train_test_split(x, y,train_size = 0.8,
                                                    test_size=0.2,random_state=14)
## 因为需要体现以下是分类模型，因为DecisionTreeClassifier是分类算法，要求y必须是int类型
y_train = y_train.astype(np.int)
y_test = y_test.astype(np.int)

# 数据标准化
ss = MinMaxScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)

# 特征选择
ch2 = SelectKBest(chi2,k=3)
x_train = ch2.fit_transform(x_train,y_train)
x_test = ch2.transform(x_test)
select_name_index = ch2.get_support(indices=True)
print ("对类别判断影响最大的三个特征属性分布是:",ch2.get_support(indices=False))
print(select_name_index)

# 降维处理
pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train,y_train)
x_test = pca.transform(x_test)




# 模型加载
model = DecisionTreeClassifier(criterion='entropy',max_depth=20,random_state=0)
model.fit(x_train,y_train)

predict_hat = model.predict(x_test)

print("预测值%s"%(predict_hat))
print("训练集上的准确率%s"%(model.score(x_train,y_train)))
print("测试集准确率%s"%(model.score(x_test,y_test)))
print("特征的权重%s"%(model.feature_importances_)) #重要性权重，值越大表示该特征对于目标属性y的影响越大


dot_data = tree.export_graphviz(model,out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_png("0.png")
graph.write_pdf("iris2.pdf")

