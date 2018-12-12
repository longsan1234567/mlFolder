# -*- coding: utf-8 -*-
# @Time    : 2018/12/6 下午5:44
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : 鸢尾花数据分类(分类问题).py
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
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.preprocessing import label_binarize
from sklearn import metrics

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
## 拦截异常
warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)


path = "datas/iris.data"
names = ['sepal length', 'sepal width', 'petal length','petal width', 'cla']
df = pd.read_csv(path, header=None, names=names)
df['cla'].value_counts()
df.head()

# 数据哑编码处理
def parseRecord(record):
    result=[]
    r = zip(names,record)
    for name,v in r:
        if name == 'cla':
            if v == 'Iris-setosa':
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


### 数据转换
datas = df.apply(lambda r: pd.Series(parseRecord(r),index=names), axis=1)

### 异常数据删除
datas = datas.dropna(how='any')

### 数据分割
X = datas[names[0:-1]]
Y = datas[names[-1]]

### 数据抽样(训练数据和测试数据分割)
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

print ("原始数据条数:%d；训练数据条数:%d；特征个数:%d；测试样本条数:%d"
  % (len(X), len(X_train), X_train.shape[1], X_test.shape[0]))


# 数据标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)



lr = LogisticRegressionCV(Cs=np.logspace(-4,1,50),
  cv=3,fit_intercept=True, penalty='l2', solver='lbfgs',
    tol=0.01, multi_class='multinomial')
#solver：‘newton-cg’,'lbfgs','liblinear','sag'  default:liblinear
#'sag'=mini-batch
#'multi_clss':
lr.fit(X_train, Y_train)


## 将正确的数据转换为矩阵形式
y_test_hot = label_binarize(Y_test,classes=(1,2,3))
print(y_test_hot)
## 得到预测的损失值
lr_y_score = lr.decision_function(X_test)
## 计算roc的值
lr_fpr, lr_tpr, lr_threasholds = metrics.roc_curve(y_test_hot.ravel(),lr_y_score.ravel())
#threasholds阈值
## 计算auc的值
lr_auc = metrics.auc(lr_fpr, lr_tpr)
print ("Logistic算法R值：", lr.score(X_train, Y_train))
print ("Logistic算法AUC值：", lr_auc)
### 7. 模型预测
print(lr_y_score)
lr_y_predict = lr.predict(X_test)
print(lr.predict_proba(X_test))




x_test_len = range(len(X_test))
## 画图1：ROC曲线画图
plt.figure(figsize=(8, 6), facecolor='w')
plt.plot(lr_fpr,lr_tpr,c='r',lw=2,label=u'Logistic算法,AUC=%.3f' % lr_auc)

plt.plot((0,1),(0,1),c='#a0a0a0',lw=2,ls='--')
plt.xlim(-0.01, 1.02) #设置X轴的最大和最小值
plt.ylim(-0.01, 1.02) #设置y轴的最大和最小值
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))


plt.xlabel('False Positive Rate(FPR)', fontsize=16)
plt.ylabel('True Positive Rate(TPR)', fontsize=16)
plt.grid(b=True, ls=':')
plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
plt.title(u'鸢尾花数据LogisticROC/AUC', fontsize=18)
plt.show()

# 绘制预测
# plt.figure(figsize=(12, 9), facecolor='w')
# plt.ylim(0.5,3.5)
# plt.plot(x_test_len, Y_test, 'ro',markersize = 6,
#   zorder=3, label=u'真实值')
# plt.plot(x_test_len, lr_y_predict, 'go', markersize = 10, zorder=2,
#   label=u'Logis算法预测值,$R^2$=%.3f' % lr.score(X_test, Y_test))
# plt.legend(loc = 'lower right')
# plt.xlabel(u'数据编号', fontsize=18)
# plt.ylabel(u'种类', fontsize=18)
# plt.title(u'鸢尾花数据分类', fontsize=20)
# plt.show()



