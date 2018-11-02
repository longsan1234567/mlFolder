# -*- coding: utf-8 -*-
# @Time    : 2018/11/2 上午11:32
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : testlambda.py
# @Software: PyCharm

from functools import reduce
add = lambda x,y: x + y
print(add(1,2))
# 3

## 内置函数

## 1 map函数 遍历序列，对序列中每个元素进行操作，最终获取新的序列。
X = [11, 22, 33]
print(list(map(lambda x:x+100,X)))
# [111, 122, 133]


## 2 filter函数   对于序列中的元素进行筛选，最终获取符合条件的序列
X2 = [11,22,33]
print(list(filter(lambda x:x>20,X)))


## 3 reduce函数 对于序列内所有元素进行累计操作
# 在Python 3里,reduce()函数已经被从全局名字空间里移除了,它现在被放置在fucntools模块里 用的话要 先引
# 入：
X3 = [1,2,3,4]
print(reduce(lambda x1,x2:x1+x2,X3))

















