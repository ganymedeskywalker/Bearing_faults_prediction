# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 19:36:28 2020
代码功能：模型训练与保存
说明：模型训练的参数已经调整到最优
@author: Ganymede 陈
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import os
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

#这里路径设置为训练集的路径
train_data = pd.read_csv('./feature_train/feature_train.csv')

# 模型初始化，设置random_state保证可复现性，便于观察优化

train_data_y = train_data['label']
# 除去标签的所有列就是特征
train_data_x = train_data.drop(['label'], axis=1)

model_lgb_default = lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.07, max_depth=7, num_leaves=240, 
               min_child_samples=21, min_child_weight=0.001, min_split_gain=0.0,
               n_estimators=100, n_jobs=-1,objective='multiclass',num_class=4,
               random_state=2019, reg_alpha=0.0, reg_lambda=0.0, silent=True,
               subsample=1.0, subsample_for_bin=200000,
               feature_fraction=0.8,bagging_fraction=0.3,bagging_freq=5)   # subsample_freq=0,
# 模型训练
model_lgb_default.fit(train_data_x, train_data_y)
joblib.dump(model_lgb_default, './model/lightgbm2_best.model') 
