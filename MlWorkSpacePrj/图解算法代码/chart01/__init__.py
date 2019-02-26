# -*- coding: utf-8 -*-
# @Time    : 2019/2/25 10:07 AM
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : __init__.py.py
# @Software: PyCharm

num1 = int(input('请输入一个整数'))
num2 = int(input('请输入另一个整数'))

if num1 < num2:
    tmp = num1
    num1 = num2
    num2 = num1

while num2 != 0:
    tmp = num1%num2
    num1 = num2
    num2 = tmp

print('最大公约数为{}'.format(num1))