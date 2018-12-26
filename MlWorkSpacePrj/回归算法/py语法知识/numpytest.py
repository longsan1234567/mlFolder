# -*- coding: utf-8 -*-
# @Time    : 2018/12/17 下午4:53
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : numpytest.py
# @Software: PyCharm

import numpy as np


'''
unique 函数
功能 去除重复数据 并从大到小返回新数据
return_index:True 返回新列表 数据在旧列表中的位置
'''
A = [1, 2, 2, 5,3, 4, 3]
a = np.unique(A)
print(a)

B = (1, 2, 2,5, 3, 4, 3)
b = np.unique(B)
print(b)


C = ['fgfh','asd','fgfh','asdfds','wrh']
c = np.unique(C) #按字母大小排列
print(c)


d,s = np.unique(A,return_index=True)
print(d,s)

'''
输出:
[1 2 3 4 5]
[1 2 3 4 5]
['asd' 'asdfds' 'fgfh' 'wrh']
[1 2 3 4 5] [0 1 4 5 3]
'''


'''
numpy.dstack(）函数
等价于：np.concatenate(tup, axis=2)
'''
x1 = np.array((1,2,3))
x2 = np.array((2,3,4))
print(np.dstack((x1,x2)))

'''
输出
[[[1 2]
  [2 3]
  [3 4]]]
'''



'''
 numpy.ravel()函数
 返回扁平连续的数组
'''
print("----------------")
x = np.array([[1,2,3],[4,5,6]])

print(np.ravel(x))
print(x.reshape(-1))

print(np.ravel(x,order='F'))
print(np.ravel(x.T))

print(np.ravel(x.T,order='A'))




