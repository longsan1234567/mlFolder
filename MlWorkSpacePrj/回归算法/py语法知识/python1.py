# -*- coding: utf-8 -*-
# @Time    : 2018/11/29 上午11:43
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : python1.py
# @Software: PyCharm

import numpy as np

a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
'''
 np.ceil(x, *args, **kwargs)
 向正无穷取整
'''
print(np.ceil(a)) #[-1. -1. -0.  1.  2.  2.  2.]


'''
 logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)
    等比数列
    10的幂
'''
print(np.logspace(0,0,10)) #[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] 为什么是1 因为10的0次幂是1
print(np.logspace(0,9,10))

# 改变基数 2的0 2的1 ..... 2的9
print(np.logspace(0,9,10,base=2)) # [  1.   2.   4.   8.  16.  32.  64. 128. 256. 512.]


'''
    降维 flatten() ravel()
    将多维数组降为一维
    #  flattenh函数# 传入'F'参数表示列序优先和ravel函数在降维时默认是行序优先
'''

x = np.array([[1,2],[3,4]])
print(x)
print(x.flatten())
print(x.ravel())

# 传入'F'参数表示列序优先
print(x.ravel('F'))
print(x.flatten('F'))


















