# -*- coding: utf-8 -*-
# @Time    : 2018/11/4 下午11:08
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : 清洗数据.py
# @Software: PyCharm

file_paths = ['../datas/000','../datas/001']
# file_path = '../datas/000'
for file_path in  file_paths:
    with open(file_path,encoding='gb2312',errors='ignore') as file:
     flag = False
     content_dict = {}
     for line in file:
         line = line.strip()
         if line.startswith('From:'):
             content_dict['from'] = line[5:]
         elif line.startswith('To:'):
             content_dict['to'] = line[3:]
         elif line.startswith('Date:'):
             content_dict['date'] = line[5:]
         elif not line:
             flag = True

         if flag:
            if 'content' not in  content_dict:
                content_dict['content'] = line
            else:
                content_dict['content'] += line


     print(content_dict)




