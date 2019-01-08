# -*- coding: utf-8 -*-
# @Time    : 2019/1/4 4:57 PM
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : 随机森林宫颈癌预测.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import label_binarize
from sklearn import metrics

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False



names = [u'Age', u'Number of sexual partners', u'First sexual intercourse',
       u'Num of pregnancies', u'Smokes', u'Smokes (years)',
       u'Smokes (packs/year)', u'Hormonal Contraceptives',
       u'Hormonal Contraceptives (years)', u'IUD', u'IUD (years)', u'STDs',
       u'STDs (number)', u'STDs:condylomatosis',
       u'STDs:cervical condylomatosis', u'STDs:vaginal condylomatosis',
       u'STDs:vulvo-perineal condylomatosis', u'STDs:syphilis',
       u'STDs:pelvic inflammatory disease', u'STDs:genital herpes',
       u'STDs:molluscum contagiosum', u'STDs:AIDS', u'STDs:HIV',
       u'STDs:Hepatitis B', u'STDs:HPV', u'STDs: Number of diagnosis',
       u'STDs: Time since first diagnosis', u'STDs: Time since last diagnosis',
       u'Dx:Cancer', u'Dx:CIN', u'Dx:HPV', u'Dx', u'Hinselmann', u'Schiller',
       u'Citology', u'Biopsy'] #df.columns

path = "datas/risk_factors_cervical_cancer.csv"

data = pd.read_csv(path)


## 模型存在多个需要预测的y值，如果是这种情况下，简单来讲可以直接模型构建，在模型内部会单独的处理每个需要预测的y值，相当于对每个y创建一个模型
X = data[names[0:-4]]
Y = data[names[-4:]]
X.head(1) #随机森林可以处理多个目标变量的情况

print(X.shape)
print(Y.shape)
print(X.head(2))
print(Y.head(2))

#空值的处理
X = X.replace("?", np.NAN)
# 使用Imputer给定缺省值，默认的是以mean
# 对于缺省值，进行数据填充；默认是以列/特征的均值填充
imputer = Imputer(missing_values="NaN")
X = imputer.fit_transform(X, Y)
print(X.shape)
print(Y.shape)


#数据分割
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print ("训练样本数量:%d,特征属性数目:%d,目标属性数目:%d" % (x_train.shape[0],x_train.shape[1],y_train.shape[1]))
print ("测试样本数量:%d" % x_test.shape[0])



# 标准化
ss = MinMaxScaler()#分类模型，经常使用的是minmaxscaler归一化，回归模型经常用standardscaler
x_train = ss.fit_transform(x_train, y_train)
x_test = ss.transform(x_test)
x_train.shape

# 降维
pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
print(x_train.shape)
print(pca.explained_variance_ratio_)


# 随机森林模型
forest = RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=2,random_state=0)
forest.fit(x_train,y_train)
print('训练数据准确率%.2f%%'%(forest.score(x_train,y_train)*100))
print('测试数据准确率%.2f%%'%(forest.score(x_test,y_test)*100))

'''
predict_proba 返回n行 k列的数组 第i行第j列上的数值是模型预测第i个预测样本为某个标签的概率  每行的概率和为1

'''
forest_y_score = forest.predict_proba(x_test)

print('预测概率的结构:{}'.format(np.array(forest_y_score).shape))

















