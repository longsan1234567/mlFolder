# -*- coding: utf-8 -*-
# @Time    : 2019/2/25 4:13 PM
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : tianchipy.py
# @Software: PyCharm

import pandas as  pd
import time

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
'''
比赛链接：https://tianchi.aliyun.com/competition/information.htm?spm=5176.11165320.5678.2.45056f7cm4yHGQ&raceId=231631

'''

df = pd.read_csv('./datas/train.csv')
print(df.head())
df.info()

# 提取x y
Y = df['Label']
X = df.drop(['ID','V_Time','Label'],1,inplace=False)
print(X.head())

# 分割数据
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=12)
print('训练样本数据数量:{}'.format(x_train.shape[0]))
print('测试样本数据数量:{}'.format(x_test.shape[1]))

print(y_train.value_counts())

t1 = time.time()

# 模型训练
lr = DecisionTreeClassifier(class_weight={0:1,1:5},random_state=12,max_depth=3)
lr.fit(x_train,y_train)
print('训练耗时%.4f{}s'.format(time.time()-t1))

# 训练数据集上的指标
print('----- 训练数据集上的指标 ----- ')
train_predict = lr.predict(x_train)
print('得分%.4f'%(f1_score(y_train,train_predict)))
print('召回率%.4f'%(recall_score(y_train,train_predict)))
print('精准率%.4f'%(precision_score(y_train,train_predict)))
print('准确率%.4f'%(accuracy_score(y_train,train_predict)))

print('---- 测试数据集上的指标 ------ ')
test_predict = lr.predict(x_test)
print('得分%.4f'%(f1_score(y_test,test_predict)))
print('召回率%.4f'%(recall_score(y_test,test_predict)))
print('精准率%.4f'%(precision_score(y_test,test_predict)))
print('准确率%.4f'%(accuracy_score(y_test,test_predict)))


# 加载预测数据集
prdf = pd.read_csv('./datas/pred.csv')
prX = prdf.drop(['ID','V_Time'],1, inplace=False)

print(prX.head())


## 输出预测结果 写入文件
result = pd.DataFrame()
result['ID'] = prdf['ID']
result['Label'] = lr.predict(prX)

result.to_csv('./datas/out/result.csv',index=False)


print('Done!!!')


'''
log 输出结果
/anaconda3/envs/mlenvment/bin/python3.7 /Users/long/Desktop/ml_worksapce/MlGitHubCode/MlWorkSpacePrj/项目/天池风险识别项目/tianchipy.py
       ID  V_Time        V1  ...         V29       V30  Label
0  254359  156699 -0.935008  ...    0.042881  0.771583      0
1  244959  152554  2.039188  ...    0.002934 -1.225918      0
2   79483   58048 -0.377984  ...    0.013761 -3.124638      0
3  164477  116748  1.985660  ...    0.006732 -2.550306      0
4  184542  126292  1.930330  ...    0.000068  0.139250      0

[5 rows x 33 columns]
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100000 entries, 0 to 99999
Data columns (total 33 columns):
ID        100000 non-null int64
V_Time    100000 non-null int64
V1        100000 non-null float64
V2        100000 non-null float64
V3        100000 non-null float64
V4        100000 non-null float64
V5        100000 non-null float64
V6        100000 non-null float64
V7        100000 non-null float64
V8        100000 non-null float64
V9        100000 non-null float64
V10       100000 non-null float64
V11       100000 non-null float64
V12       100000 non-null float64
V13       100000 non-null float64
V14       100000 non-null float64
V15       100000 non-null float64
V16       100000 non-null float64
V17       100000 non-null float64
V18       100000 non-null float64
V19       100000 non-null float64
V20       100000 non-null float64
V21       100000 non-null float64
V22       100000 non-null float64
V23       100000 non-null float64
V24       100000 non-null float64
V25       100000 non-null float64
V26       100000 non-null float64
V27       100000 non-null float64
V28       100000 non-null float64
V29       100000 non-null float64
V30       100000 non-null float64
Label     100000 non-null int64
dtypes: float64(30), int64(3)
memory usage: 25.2 MB
         V1        V2        V3    ...          V28       V29       V30
0 -0.935008  0.820946  1.067777    ...     0.051355  0.042881  0.771583
1  2.039188 -0.264982 -1.235053    ...    -0.055856  0.002934 -1.225918
2 -0.377984  0.917614  1.714673    ...    -0.407508  0.013761 -3.124638
3  1.985660 -0.752667 -1.669258    ...    -0.049041  0.006732 -2.550306
4  1.930330  2.563490 -4.759537    ...     0.109086  0.000068  0.139250

[5 rows x 30 columns]
训练样本数据数量:70000
测试样本数据数量:30
0    69789
1      211
Name: Label, dtype: int64
训练耗时%.4f1.0837130546569824s
----- 训练数据集上的指标 -----
得分0.8723
召回率0.8578
精准率0.8873
准确率0.9992
---- 测试数据集上的指标 ------
得分0.8000
召回率0.7865
精准率0.8140
准确率0.9988
         V1        V2        V3    ...          V28       V29       V30
0 -2.389003  0.508246  0.955227    ...    -0.221525  0.035281 -2.741180
1  1.324636  0.095398 -0.105591    ...    -0.015829  0.002406 -7.049453
2 -0.083895  0.543350 -0.244593    ...     0.086205  0.112603 -1.469655
3  2.027800 -0.080261 -1.167567    ...    -0.068097  0.011066 -0.322347
4 -0.621263 -0.400795  1.599899    ...    -0.167883  0.049452 -2.212872

[5 rows x 30 columns]
Done!!!
'''


