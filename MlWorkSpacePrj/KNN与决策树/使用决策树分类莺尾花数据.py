# -*- coding: utf-8 -*-
# @Time    : 2018/12/20 下午4:55
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : 使用决策树分类莺尾花数据.py
# @Software: PyCharm

'''
 DecisionTreeClassifier 决策树
'''

import pandas as pd
import numpy as np
import matplotlib as mpl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1、加载数据
names = ['A', 'B', 'C', 'D', 'label']
path = './datas/iris.data'
df = pd.read_csv(path, sep=',', header=None, names=names)


# 2 数据清洗
# df = df.replace('?', np.nan)
df.replace('?', np.nan, inplace=True)
# 删除所有具有非数字的样本
# how: 指定如何删除数据，如果设置为any，默认值, 那么只要行或者列中存在nan值，那么就进行删除操作；如果设置为all，那边要求行或者列中的所有值均为nan的时候，才可以进行删除操作
df.dropna(axis=0, how='any', inplace=True)
print(df.head(5))


# 3 提取数据集的特征和标签
X = df[names[:-1]]
# X.info()
Y = df[names[-1]]
Y_label_values = np.unique(Y)
Y_label_values.sort()
print("Y的取值可能:{}".format(Y_label_values))
random_index = np.random.permutation(len(Y))[:5]
print("随机的索引:{}".format(random_index))
print("原始随机的5个Y值:{}".format(np.array(Y[random_index])))
for i in range(len(Y_label_values)):
    Y[Y == Y_label_values[i]] = i
Y = Y.astype(np.float)
print("做完标签值转换后的随机的5个Y值:{}".format(np.array(Y[random_index])))



# 4 数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8,test_size=0.2, random_state=28)


# 5 数据标准化
ss = StandardScaler()
x_train = ss.fit_transform(x_train, y_train)
x_test = ss.transform(x_test)

# 6 模型选择
algo = DecisionTreeClassifier(criterion="entropy",max_depth = 20,random_state=0)
algo.fit(x_train,y_train)

# 7 模型效果评估
y_hat = algo.predict(x_test)
print("在训练集上的模型效果(分类算法中为准确率):{}".format(algo.score(x_train, y_train)))
print("在测试集上的模型效果(分类算法中为准确率):{}".format(algo.score(x_test, y_test)))
print("在测试集上的召回率的值:{}".format(recall_score(y_true=y_test, y_pred=y_hat, average='micro')))


# 8 输出分类的一些API
print('预测值%s'%(y_hat))

print("预测值真实值%s"%([np.array(Y_label_values)[int(value)] for value in y_hat]))

print("预测值属于各个类别的概率值%s"%(algo.predict_proba(x_test)))



# 9输出决策树各个特征属性的权重系数
print(list(zip(names[:-1],algo.feature_importances_)))

'''
/anaconda3/envs/mlenvment/bin/python3.7 /Users/long/Desktop/ml_worksapce/MlGitHubCode/MlWorkSpacePrj/回归算法/KNN与决策树/使用决策树分类莺尾花数据.py
/anaconda3/envs/mlenvment/lib/python3.7/site-packages/sklearn/externals/joblib/externals/cloudpickle/cloudpickle.py:47: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
  import imp
     A    B    C    D        label
0  5.1  3.5  1.4  0.2  Iris-setosa
1  4.9  3.0  1.4  0.2  Iris-setosa
2  4.7  3.2  1.3  0.2  Iris-setosa
3  4.6  3.1  1.5  0.2  Iris-setosa
4  5.0  3.6  1.4  0.2  Iris-setosa
Y的取值可能:['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
随机的索引:[ 82 146  86  28  25]
原始随机的5个Y值:['Iris-versicolor' 'Iris-virginica' 'Iris-versicolor' 'Iris-setosa'
 'Iris-setosa']
/Users/long/Desktop/ml_worksapce/MlGitHubCode/MlWorkSpacePrj/回归算法/KNN与决策树/使用决策树分类莺尾花数据.py:52: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  Y[Y == Y_label_values[i]] = i
做完标签值转换后的随机的5个Y值:[1. 2. 1. 0. 0.]
在训练集上的模型效果(分类算法中为准确率):1.0
在测试集上的模型效果(分类算法中为准确率):0.9666666666666667
在测试集上的召回率的值:0.9666666666666667
预测值[0. 2. 1. 0. 2. 1. 2. 1. 1. 0. 2. 0. 1. 1. 2. 0. 2. 2. 2. 1. 0. 0. 1. 2.
 1. 0. 2. 2. 0. 1.]
预测值真实值['Iris-setosa', 'Iris-virginica', 'Iris-versicolor', 'Iris-setosa', 'Iris-virginica', 'Iris-versicolor', 'Iris-virginica', 'Iris-versicolor', 'Iris-versicolor', 'Iris-setosa', 'Iris-virginica', 'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor', 'Iris-virginica', 'Iris-setosa', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-versicolor', 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica', 'Iris-versicolor', 'Iris-setosa', 'Iris-virginica', 'Iris-virginica', 'Iris-setosa', 'Iris-versicolor']
预测值属于各个类别的概率值[[1. 0. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]]
[('A', 0.0), ('B', 0.0), ('C', 0.054834837911029874), ('D', 0.9451651620889702)]

Process finished with exit code 0
'''

