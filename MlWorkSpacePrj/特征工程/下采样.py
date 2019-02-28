# -*- coding: utf-8 -*-
# @Time    : 2019/2/28 4:52 PM
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : 下采样.py
# @Software: PyCharm

'''
 下采样  减少多数样本的数量,从而使数据平衡
'''

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 随机数种子
np.random.seed(12)

'''
下采样操作
从给定的DataFrame对象df中抽取出sample number条样本数据，并将数据返回
:param df:
:param sample_number:
:return:
'''
def lower_sample_data(df,sample_number):
    #1 获取总的样本数目
    rows = len(df)

    #2 进行样本数目判断 如果样本数目小于抽取的sample number的数目 那么直接返回
    if rows <= sample_number:
        return None, df

    #3 随机生成需要抽取的数据对应的下标
    row_index = set()
    while len(row_index) != sample_number:
        index = np.random.randint(0,rows,1)[0]
        print(index)
        row_index.add(index)

    #4 进行数据的抽取操作
    sample_df = df.iloc[list(row_index)]
    other_row_index = [i for i in  range(rows) if i not in row_index]
    other_df = df.iloc[list(other_row_index)].reset_index(drop=True)


    #5 任何结果
    return other_df,sample_df



if __name__ == '__main__':

    # 1 产生模拟数据 [0~10] 10000*5的随机数
    category1 = np.random.randint(0,10,[10000,5]).astype(np.float)
    label1 = np.array([1]*10000).reshape(-1,1)
    data1 = np.concatenate((category1,label1),axis=1)
    # 产生模拟数据 [8~18] 10*5的随机数
    category2 = np.random.randint(8,18,[10,5]).astype(np.float)
    label2 = np.array([0]*10).reshape(-1,1)
    data2 = np.concatenate((category2,label2),axis=1)

    # print(data1)
    # print(data2)

    # 2 构建DataFrame
    name = ['A','B','C','D','E','Label']
    data = np.concatenate((data1,data2),axis=0)
    df = pd.DataFrame(data,columns=name)
    print(df)


    # 3
    small_df = df[df.Label == 0.0] #获取小众类数据
    big_df = df[df.Label == 1.0] #获取大众类数据

    # 4 进行下采样操作
    sample_number = 100
    big_df, sample_big_category_df = lower_sample_data(big_df, sample_number)

    # 5 合并数据
    train_df = pd.concat([small_df,sample_big_category_df],ignore_index=True)


     # 数据划分操作
    df = train_df
    x = df.drop('Label', axis = 1)
    y = df['Label']

    print('--- 训练数据 --- x')
    print(x)
    print('--- 训练数据 --- y')
    print(y)
    # 模型训练
    algo = LogisticRegression(fit_intercept=True)
    algo.fit(x,y)

    # # 原始数据
    plt.plot(category1[:, 0], category1[:, 1], 'ro', markersize=3)
    plt.plot(category2[:, 0], category2[:, 1], 'bo', markersize=3)
    plt.show()


# a = [{'Time': '2017-09-19', 'News': '楚了何人之手。今日，楚天都市报'}]
# for i in a:
#     for k,v in i.items():
#         print('%s %s'%(k,v))
