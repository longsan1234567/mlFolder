# -*- coding: utf-8 -*-
# @Time    : 2018/12/20 下午7:11
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : 决策树莺尾花(全).py
# @Software: PyCharm

'''
 决策树案例 莺尾花数据分类
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

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


# 画图 横竖轴
# 横轴采样范围 x1_min x1_max
x1_min = np.min((x_train.T[0].min(),x_test.T[0].min()))
x1_max = np.max((x_test.T[0].max(),x_train.T[0].max()))

x2_min = np.min((x_train.T[1].min(),x_test.T[1].min()))
x2_max = np.max((x_train.T[1].max(),x_test.T[1].max()))

print("横轴采样范围 [%s %s]"%(x1_min,x1_max))
print("纵轴采样范围 [%s %s]"%(x2_min,x2_max))

N = 100 #采样点
t1 = np.linspace(x1_min,x1_max,N)
t2 = np.linspace(x2_min,x2_max,N)
x1, x2 = np.meshgrid(t1,t2) # 生成网格采样点
x_show = np.dstack((x1.flat,x2.flat))[0] #测试点

y_show_hat = model.predict(x_show) #预测值

y_show_hat = y_show_hat.reshape(x1.shape)  #使之与输入的形状相同

print(y_show_hat.shape)
print(y_show_hat[0])
print(y_show_hat[1])


# 画图 画样本点
plt_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
plt_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

plt.figure(facecolor='w')

## 画一个区域
plt.pcolormesh(x1,x2,y_show_hat,cmap = plt_light)

# 画测试数据的点信息
plt.scatter(x_test.T[0], x_test.T[1], c=y_test.ravel(), edgecolors='k', s=150, zorder=10, cmap=plt_dark, marker='*')  # 测试数据
# 画训练数据的点信息
plt.scatter(x_train.T[0], x_train.T[1], c=y_train.ravel(), edgecolors='k', s=40, cmap=plt_dark)  # 全部数据
plt.xlabel(u'特征属性x1', fontsize=15)
plt.ylabel(u'特征属性x2', fontsize=15)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid(True)
plt.title(u'鸢尾花数据的决策树分类', fontsize=18)
plt.show()

#基于原始数据前3列比较一下决策树在不同深度的情况下错误率
x_train4, x_test4, y_train4, y_test4 = train_test_split(x.iloc[:,:2],y,train_size=0.7,test_size=0.3,random_state=14)

depths = np.arange(1,15)
err_list = []
for d in depths:
    clf = DecisionTreeClassifier(criterion='entropy',max_depth=d,min_samples_split=10)
    clf.fit(x_train4,y_train4)

    score = clf.score(x_test4,y_test4)
    err = 1- score
    err_list.append(err)
    print('树的深度%d 训练集上的正确率%.5f'%(d,clf.score(x_train4,y_train4)))
    print("树的深度%d 测试集合上正确率%.5f"%(d,score))


plt.figure(facecolor='w')
plt.plot(depths, err_list,'ro-',lw=3)
plt.xlabel(u'决策树的深度',fontsize=16)
plt.ylabel(u'错误率',fontsize=16)
plt.grid(True)
plt.title(u'树的深度导致过拟合(欠拟合)问题')
plt.show()























