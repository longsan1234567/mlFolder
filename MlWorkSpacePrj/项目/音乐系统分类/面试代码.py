# -*- coding: utf-8 -*-
# @Time    : 2019/2/20 3:35 PM
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : 面试代码.py
# @Software: PyCharm

'''
代码实现:
input 目录下有若干个.wav格式的音频文件 编写一段代码将input下的每个音频文件等分成
长度为2s的音频文件 并存放在output目录下

'''
from pydub import AudioSegment
import os

# glob 是文件操作相关模块
import glob
file_paths = glob.glob('./input/*.wav')

out_file_path = './output'

if not os.path.exists(out_file_path):
    os.makedirs(out_file_path)

index = 1
for file_path in file_paths:
    song = AudioSegment.from_file(file_path)
    song_length = len(song)
    start = 0
    count = 1;
    while start < song_length:
        end = int(min(start+2000,song_length))
        song[start:end].export(out_file_path + '/{}_{}.wav'.format(index,count),format='wav')
        start = end
        count += 1
    index += 1

print('音频文件切割完成')
