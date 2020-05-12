# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:17:50 2020
代码功能：特征提取
部署运行请注意：需要修改params['path']为测试集文件夹路径
环境信息：python3.6
代码输入：142个测试集，路径来源自定义
代码输出：142个测试集，路径在同目录下的新生成文件夹fea_test2
@author: Ganymede 陈
"""

#需要导入的包
import pandas as pd
import numpy as np
from scipy import stats,fftpack,signal
from pywt import wavedec
import os

#常数定义：
params = {}
#142个原始数据存放的文件夹路径，默认放置在与我提交的项目的同一目录下
params['path'] = '../../cwru'
#在本目录下创建fea_test2文件夹，用于存放特征提取后的csv文件
if not os.path.exists('./fea_test2'):
    os.makedirs('./fea_test2')
params['opath'] = './fea_test2'

#特征列表
df_out_columns = ['time_mean','time_std','time_max','time_min','time_rms','time_ptp','time_median','time_iqr','time_pr','time_skew','time_kurtosis','time_var','time_amp',
                    'time_smr','time_wavefactor','time_peakfactor','time_pulse','time_margin','freq_mean','freq_std','freq_max','freq_min','freq_rms','freq_median',
                    'freq_iqr','freq_pr','freq_f2','freq_f3','freq_f4','freq_f5','freq_f6','freq_f7','freq_f8']
DE_columns = ['DE_' + i for i in df_out_columns]
label_columns = ['label']
full_columns = DE_columns +  label_columns  

#采样率
sampleRate = 12000   

#特征提取函数
def featureget(df_line):
    feature_list = []
    #----------  time-domain feature,18
    #依次为均值，标准差，最大值，最小值，均方根，峰峰值，中位数，四分位差，百分位差，偏度，
    #峰度，方差，整流平均值，方根幅值，波形因子，峰值因子，脉冲值，裕度
    time_mean = df_line.mean()
    time_std = df_line.std()
    time_max = df_line.max()
    time_min = df_line.min()
    time_rms = np.sqrt(np.square(df_line).mean())
    time_ptp = df_line.ptp()
    time_median = np.median(df_line)
    time_iqr = np.percentile(df_line,75)-np.percentile(df_line,25)
    time_pr = np.percentile(df_line,90)-np.percentile(df_line,10)
    time_skew = stats.skew(df_line)
    time_kurtosis = stats.kurtosis(df_line)
    time_var = np.var(df_line)
    time_amp = np.abs(df_line).mean()
    time_smr = np.square(np.sqrt(np.abs(df_line)).mean())
    #下面四个特征需要注意分母为0或接近0问题，可能会发生报错
    time_wavefactor = time_rms/time_amp
    time_peakfactor = time_max/time_rms
    time_pulse = time_max/time_amp
    time_margin = time_max/time_smr
    #----------  freq-domain feature,15
    #采样频率25600Hz
    df_fftline = fftpack.fft(df_line)
    freq_fftline = fftpack.fftfreq(len(df_line),1/2000)
    df_fftline = abs(df_fftline[freq_fftline>=0])
    freq_fftline = freq_fftline[freq_fftline>=0]
    #基本特征,依次为均值，标准差，最大值，最小值，均方根，中位数，四分位差，百分位差
    freq_mean = df_fftline.mean()
    freq_std = df_fftline.std()
    freq_max = df_fftline.max()
    freq_min = df_fftline.min()
    freq_rms = np.sqrt(np.square(df_fftline).mean())
    freq_median = np.median(df_fftline)
    freq_iqr = np.percentile(df_fftline,75)-np.percentile(df_fftline,25)
    freq_pr = np.percentile(df_fftline,90)-np.percentile(df_fftline,10)
    #f2 f3 f4反映频谱集中程度
    freq_f2 = np.square((df_fftline-freq_mean)).sum()/(len(df_fftline)-1)
    freq_f3 = pow((df_fftline-freq_mean),3).sum()/(len(df_fftline)*pow(freq_f2,1.5))
    freq_f4 = pow((df_fftline-freq_mean),4).sum()/(len(df_fftline)*pow(freq_f2,2))
    #f5 f6 f7 f8反映主频带位置
    freq_f5 = np.multiply(freq_fftline,df_fftline).sum()/df_fftline.sum()
    freq_f6 = np.sqrt(np.multiply(np.square(freq_fftline),df_fftline).sum())/df_fftline.sum()
    freq_f7 = np.sqrt(np.multiply(pow(freq_fftline,4),df_fftline).sum())/np.multiply(np.square(freq_fftline),df_fftline).sum()
    freq_f8 = np.multiply(np.square(freq_fftline),df_fftline).sum()/np.sqrt(np.multiply(pow(freq_fftline,4),df_fftline).sum()*df_fftline.sum())
    feature_list.extend([time_mean,time_std,time_max,time_min,time_rms,time_ptp,time_median,time_iqr,time_pr,time_skew,time_kurtosis,time_var,time_amp,
                         time_smr,time_wavefactor,time_peakfactor,time_pulse,time_margin,freq_mean,freq_std,freq_max,freq_min,freq_rms,freq_median,
                         freq_iqr,freq_pr,freq_f2,freq_f3,freq_f4,freq_f5,freq_f6,freq_f7,freq_f8]) 
    return feature_list



files_path = params['path']
ofiles_path = params['opath']
filelists = os.listdir(files_path) #读取文件，得到文件夹中所有csv文件名
filelists.sort(key=lambda x:int(x[4:-4])) #按照文件名数字由小到大排序

fea_selected = pd.DataFrame()

for info in filelists:
    file_path = os.path.join(files_path,info)
    ofile_path = os.path.join(ofiles_path,info)
    test = pd.read_csv(file_path)
    feature_test2 = []
    windowSize = 60*sampleRate/test['RPM'][0]
    for n in range(int(len(test)/windowSize)):  
        DE = featureget(test.loc[n*windowSize+1:(n+1)*windowSize,'DE_time'])
        feature_test2.append(DE)
    #换成数据帧格式
    feature_test2 = pd.DataFrame(feature_test2,columns=DE_columns)
    #fea_selected = pd.concat([fea_selected,feature_test2])

    feature_test2.to_csv(ofile_path,index=False)
