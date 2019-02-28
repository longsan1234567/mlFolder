# -*- coding: utf-8 -*-
# @Time    : 2019/2/28 10:50 AM
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : 上采样.py
# @Software: PyCharm

'''
上采样操作 随机有放回抽取少数类目的数据 增加少数类目数据量
'''
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as  pd
from sklearn.linear_model import LogisticRegression
import warnings


# 拦截异常信息
warnings.filterwarnings(action='ignore', category=UserWarning)

np.random.seed(12)
'''
 进行上采样
 df: DataFrame对象，进行上采样过程中的原始数据集
 sample_number： 需要采样的数量
 label: label名称，必须是字符串
'''

def upper_sample_data(df,sample_number,label):

    #1. 获取Dateframe的列数和行数
    df_column_size = df.columns.size

    df_row_size = len(df)

    #2. 做抽样操作
    sample_df = pd.DataFrame(columns=df.columns)

    for i  in range(sample_number):
        #1 随机一个样本下标值
        idx = np.random.randint(0,df_row_size,1)[0]

        #2 获取下标对应的标签值
        item = df.iloc[idx]

        #3 获取原始数据的标准值
        label_value = item[label]

        #4 删除标签值
        del item[label]

        #5 对剩下的特征属性做一个偏移操作
        item = item + [np.random.random()-0.5 for j in range(df_column_size-1)]

        #6 将标签还原
        item[label] = label_value

        #7 将数据添加到DataFrame
        sample_df.loc[i] = item

    return sample_df


if __name__ == '__main__':

    # 1产生模拟数据
    category1 = np.random.rand(10000,2)* 10  #随机[1,10]的数据
    # print(category1)
    label1 = np.array([1] * 10000).reshape(-1,1)
    # print(label1)
    data1 = np.concatenate((category1,label1),axis = 1) # 合并数据


    category2 = np.random.rand(100,2)/(1-0) * 3 -2 #[-2,1]
    label2 = np.array([0]*100).reshape(-1,1)
    # print(category2)
    # print(label2)
    data2 = np.concatenate((category2,label2),axis = 1) # 合并数据



    name = ['A','B','label']
    data = np.concatenate((data1,data2),axis = 0)

    # print(data)
    df = pd.DataFrame(data, columns = name)

    # 获取小众样本数据
    small_category = df[df.label == 0.0]

    # 上采样产生新的数据
    sample_num_new = 2000
    small_category_data_new = upper_sample_data(small_category,sample_num_new,label = 'label')

    print(small_category_data_new.head())


    # 合并新的小众数据和大众数据
    final_df = pd.concat([df,small_category_data_new],ignore_index = True)


    # 数据划分操作
    df = final_df
    x = df.drop('label', axis = 1)
    y = df['label']

    # 模型训练
    algo = LogisticRegression(fit_intercept=True)
    algo.fit(x,y)

    w1, w2 = algo.coef_[0]
    c = algo.intercept_

    print('模型参数:{}-{}-{}'.format(w1,w2,c))

    # 原始数据
    plt.plot(category1[:, 0], category1[:, 1], 'ro', markersize=3)
    plt.plot(category2[:, 0], category2[:, 1], 'bo', markersize=3)
    plt.plot([-10, 10.0 * w1 / w2 - c / w2], [10, -10.0 * w1 / w2 - c / w2], 'g-')
    plt.show()



















