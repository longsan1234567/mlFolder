# -*- coding: utf-8 -*-
# @Time    : 2019/2/18 3:39 PM
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : pydub_demo.py
# @Software: PyCharm

'''
声音频率在20HZ~20kHz之间的声波，称为音频
音频的采样频率
22050 常用的的采样频率
44100 cd音质
超过48000/960000的采样对人耳没有多大意义
'''

from  pydub import AudioSegment
import numpy as np
import array

# 1 读取数据
path = './data/20Hz-stero.wav'

song = AudioSegment.from_file(file=path)

print(song)

# 2 音频文件相关属性
size = len(song)
print("音频文件的长度信息(ms):{}".format(size))

channel = song.channels
print('音频通道数目:{}'.format(channel))

frame_rate = song.frame_rate
print('音频的抽样频率:{}'.format(frame_rate))

sample_width = song.sample_width
print('音频的样本宽度:{}'.format(sample_width))

# <pydub.audio_segment.AudioSegment object at 0x108d0f198>
# 音频文件的长度信息(ms):300000
# 音频通道数目:2
# 音频的抽样频率:44100
# 音频的样本宽度:2 采样位数 常见1(8bit) 2(16bit cd标准) 4(32bit)

# 3 设置相关特征属性
# 如果降低音乐通道会带来音频的损失 但是增加不会带来损失(频率和样本宽度一样)
song = song.set_channels(channels=1)
song = song.set_frame_rate(frame_rate=22050)
song = song.set_sample_width(sample_width=1)

print('---- 设置后 -----')
print('音频的长度信息.{}'.format(len(song)))
print('音频的通道数目.{}'.format(song.channels))
print('音频的抽样频率.{}'.format(song.frame_rate))
print('音频的样本宽度.{}'.format(song.sample_width))


# 4 音频文件的保存
song.export('./data/out/01.wav',format='wav')


# 5 获取部分数据并保存
# 前10s的数据
song[:10000].export('./data/out/02.wav',format='wav')
# 后10s的数据
song[-10000:].export('./data/out/03.wav',format='wav')

# 中间10s的数据
mid = len(song)//2
song[mid - 5000 : mid + 5000].export('./data/out/04.wav',format='wav')


# 6 填充保存
# a. 将song对象转换成numpy的array对象
samples = np.array(song.get_array_of_samples()).reshape(-1)
print(samples.shape)
print(samples[10:])
print(samples[:10])

# b. 填充操作
append_size = 60 * song.channels * song.frame_rate

'''
pad_width: 给定在什么位置填充 以及填充多少个值  在samples数组前面填充append_size个值 后面添加0个值
mode: 填充方式 constant表示常量填充 常量为constant_values
constant_values: 填充常量
'''
samples = np.pad(samples,pad_width=(append_size,0),mode='constant',constant_values=(0, 0))
print(samples.shape)
print(samples[:10])

# c 将截取的数组转为segment对象
song = song._spawn(array.array(song.array_type, samples))
song.export('./data/out/05.wav',format='wav')


# 对音乐文件进行处理
# 音调 音频增大和减小
(song + 100).export('./data/out/07.wav',format='wav')
(song - 100).export('./data/out/08.wav',format='wav')

samples = np.array(song.get_array_of_samples()).reshape(-1)
samples = (samples * 2).astype(np.int)
song = song._spawn(array.array(song.array_type,samples))
song.export('./data/out/09.wav',format='wav')


# 8 音乐循环 2倍
(song * 2).export('./data/out/10.wav',format = 'wav')

























