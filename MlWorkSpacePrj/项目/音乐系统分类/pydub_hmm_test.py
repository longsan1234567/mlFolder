# -*- coding: utf-8 -*-
# @Time    : 2019/2/19 9:35 AM
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : pydub_hmm_test.py
# @Software: PyCharm

import warnings
from pydub import AudioSegment
import numpy as  np
from hmmlearn import hmm

warnings.filterwarnings('ignore')

paths = ['./data/1KHz-stero.wav',
         './data/10KHz-stero.wav',
         './data/20Hz-stero.wav']

x = None
for path in paths:
    song = AudioSegment.from_file(file=path)
    song = song[:10000]
    samples = np.array(song.get_array_of_samples()).reshape((1,-1))
    if x is None:
        x = samples
    else:
        x = np.append(x,samples,axis=0)

print(x.shape)
x = x.reshape(3,-1,2 * 1000)
print(x.shape)


# 使用HMM提取特征
sample = x.reshape(-1,2 * 1000).astype(np.int32)
print('样本大小:{}'.format(sample.shape))

lengths = []
for i in  range(len(paths)):
    lengths.append(x.shape[1])
print('序列:{}'.format(lengths))

n = 100
model = hmm.GaussianHMM(n_components=n,random_state=12)
model.fit(sample, lengths = lengths)


# 预测获取结果
print("模型训练得到的参数π:")
print(model.startprob_)
print("模型训练得到的参数A:")
print(model.transmat_)


# 预测
y_hat = model.predict(sample,lengths)

new_x = y_hat.reshape(3,-1)
print('新的转换(得到新的特征)')
print(new_x)
print(new_x.shape)



















