# -*- coding: utf-8 -*-
# @Time    : 2018/12/18 上午10:11
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : 使用交叉验证模型预测.py
# @Software: PyCharm


'''
 使用本地保存的模型 预测数值
'''

import numpy as np
import pickle
from sklearn.externals import joblib


# 1. 加载模型[5.0, 3.4, 1.5, 0.2], [6.6, 3.0, 4.4, 1.4]
ss = joblib.load('./model/ss.pkl')
gcv = joblib.load('./model/gcv.pkl')
knn = joblib.load('./model/knn.pkl')
y_label_index_2_label_dict = pickle.load(open('./model/y_label_value.pkl', 'rb'))


print(y_label_index_2_label_dict)
def predict1(x):
    return gcv.predict(ss.transform(x))

def predict2(x):
    return knn.predict(ss.transform(x))

if __name__ == '__main__':
    # 输入值 进行预测
    x = [[5.0, 3.4, 1.5, 0.2], [6.6, 3.0, 4.4, 1.4]]
    y_predict1 = predict1(x)
    print(y_predict1)
    y_predict2 = predict2(x)
    print(y_predict2)

    print(predict2)
    result_label = []
    for index in y_predict2:
        label = y_label_index_2_label_dict[int(index)]
        result_label.append(label)
    print("预测结果%s"%(result_label))
    result = list(zip(y_predict2,result_label))
    print(result)











