# -*- coding: utf-8 -*-
# @Time    : 2018/11/29 下午5:02
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : 过拟合.py
# @Software: PyCharm

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

from sklearn.linear_model import LinearRegression,LassoCV,RidgeCV,ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import  Pipeline
from sklearn.linear_model.coordinate_descent import ConvergenceWarning


## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
## 拦截异常
warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)


# 1 创建数据
np.random.seed(100)
np.set_printoptions(linewidth=1000, suppress=True)
N = 10
x = np.linspace(0,6,N) + np.random.randn(N)
y = 1.8 *x ** 3 + x**2 - 14*x - 7 + np.random.randn(N)


## 将其设置为矩阵
x.shape = -1, 1
y.shape = -1, 1

print(x)
print(y)

models = [
    Pipeline([('Poly',PolynomialFeatures(include_bias=True)),
              ('Linear',LinearRegression(fit_intercept=False))
              ]),

    Pipeline([('Poly',PolynomialFeatures(include_bias=True)),
              ('Linear',RidgeCV(alphas=np.logspace(-3,2,50),fit_intercept=False))
              ]),

    Pipeline([('Poly',PolynomialFeatures(include_bias=True)),
              ('Linear',LassoCV(alphas=np.logspace(0,1,10),fit_intercept=False))
              ]),

    Pipeline([
            ('Poly', PolynomialFeatures(include_bias=True)),
            # l1_ratio：给定EN算法中L1正则项在整个惩罚项中的比例，这里给定的是一个列表；
            # 表示的是在CV交叉验证的过程中，EN算法L1正则项的权重比例的可选值的范围
            ('Linear', ElasticNetCV(alphas=np.logspace(0,1,10), l1_ratio=[.1, .5, .7, .9, .95, 1], fit_intercept=False))
        ])
]


## 线性模型过拟合图形识别
plt.figure(facecolor='w')
degree = np.arange(1,N,2) # 阶 1 3 5 7 9
print(degree)
dm = degree.size

colors = [] # 颜色
for c in np.linspace(16711680, 255, dm):
    colors.append('#%06x' % int(c))


titles = [u'线性回归', u'Ridge回归', u'Lasso回归', u'ElasticNet']
for t in range(4):
    model = models[t]#选择了模型--具体的pipeline(线性、Lasso、Ridge、EN)
    plt.subplot(2,2,t+1) # 选择具体的子图
    plt.plot(x, y, 'ro', ms = 10, zorder = N - 1) # 在子图中画原始数据点； zorder：图像显示在第几层

    # 遍历不同的多项式的阶，看不同阶的情况下，模型的效果
    for i,d in enumerate(degree):
        # 设置阶数(多项式)
        model.set_params(Poly__degree=d)
        # 模型训练
        model.fit(x, y.ravel())

        # 获取得到具体的算法模型
        # model.get_params()方法返回的其实是一个dict对象，后面的Linear其实是dict对应的key
        # 也是我们在定义Pipeline的时候给定的一个名称值
        lin = model.get_params()['Linear']

        # 打印数据
        output = u'%s:%d阶，系数为：' % (titles[t],d)
        # 判断lin对象中是否有对应的属性
        if hasattr(lin, 'alpha_'): # 判断lin这个模型中是否有alpha_这个属性
            idx = output.find(u'系数')
            output = output[:idx] + (u'alpha=%.6f, ' % lin.alpha_) + output[idx:]
        if hasattr(lin, 'l1_ratio_'): # 判断lin这个模型中是否有l1_ratio_这个属性
            idx = output.find(u'系数')
            output = output[:idx] + (u'l1_ratio=%.6f, ' % lin.l1_ratio_) + output[idx:]
        # line.coef_：获取线性模型的参数列表，也就是我们ppt中的theta值，ravel()将结果转换为1维数据
        print (output, lin.coef_.ravel())

        # 产生模拟数据
        x_hat = np.linspace(x.min(), x.max(), num = 100) ## 产生模拟数据
        x_hat.shape = -1,1
        # 数据预测
        y_hat = model.predict(x_hat)
        # 计算准确率
        s = model.score(x, y)

        # 当d等于5的时候，设置为N-1层，其它设置0层；将d=5的这条线凸显出来
        z = N + 1 if (d == 5) else 0
        label = u'%d阶, 正确率=%.3f' % (d,s)
        plt.plot(x_hat, y_hat, color=colors[i], lw=2, alpha=0.75, label=label, zorder=z)

    plt.legend(loc = 'upper left')
    plt.grid(True)
    plt.title(titles[t])
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)

plt.tight_layout(1, rect=(0,0,1,0.95))
plt.suptitle(u'各种不同线性回归过拟合显示', fontsize=22)
plt.show()









