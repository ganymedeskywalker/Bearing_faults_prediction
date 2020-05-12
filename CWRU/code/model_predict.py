# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:55:12 2020
代码功能：模型测试
部署运行请注意：完成step1后直接运行即可
代码输入：142个测试集，路径来源同目录下的新生成文件夹fea_test2
代码输出：1个csv文件，new_result.csv，同目录下
@author: Ganymede 陈
"""

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error,median_absolute_error,r2_score
import pandas as pd
import numpy as np
import json
import joblib
import os

#常数定义
params = {}
#模型路径
params['model'] = '../cwru.model'
#特征提取后测试集路径
params['test'] = './fea_test2'
#答案输出路径
params['opath']='./new_result.csv'

#加载模型
model = joblib.load(params['model'])

files_path = params['test']
filelists = os.listdir(files_path) #读取文件，得到文件夹中所有csv文件名
filelists.sort(key=lambda x:int(x[4:-4])) #按照文件名数字由小到大排序

result = pd.DataFrame()

for info in filelists:
    file_path = os.path.join(files_path,info)
    test_csv = pd.read_csv(file_path)
    y_pred = model.predict(test_csv)
    counts = np.bincount(y_pred)
    res = int(np.argmax(counts))
    predict_df = pd.DataFrame()
    predict_df['label'] = [res]
    predict_df['filename'] = info[:-4]
    result = pd.concat([result,predict_df])
   
result.to_csv(params['opath'],index=False)

