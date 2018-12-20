# -*- coding: utf-8 -*-
# @Time    : 2018/12/20 下午7:24
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : testpd.py
# @Software: PyCharm

'''
 pandas库中API测试
'''
import pandas as pd


print(pd.Categorical([1,2,3,1,2,3]))
# Categories (3, int64): [1, 2, 3]

print(pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c']))

print(pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c']).categories) #这个类别的类别
print(pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c']).codes) #这个分类的分类代码
print(pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c']).ordered) #类别是否具有有序关系
print(pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c']).dtype) #在CategoricalDtype此实例


