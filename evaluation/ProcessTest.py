from config import *
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from config import *
from CalculateAcc import calculate_acc
from models.Losses import new_mape_numpy, new_weekly_mape_numpy, new_monthly_mape_numpy
from sklearn.metrics import r2_score
from PlotFunction import *


def mse(x, y):
    return np.mean(np.square(x - y))


def mae(x, y):
    return np.mean(np.abs(x - y))


path = '../NEW_4G_ONLY'
files = os.listdir(path)

mape_list = []
increase = pd.DataFrame()
fiveG_result = []
for file_name in files:
    split_name = file_name.split('_')
    data = pd.read_csv(path + '/' + file_name)
    # data2 = pd.read_csv(path[:-3] +'_60' + '/' + file_name)
    true = data['true'].values
    try:
        predict = data['pred'].values
    except:
        predict = data['predict'].values
    result = [split_name[0], '_'.join(split_name[1:3]),
              new_mape_numpy(true, predict),
              new_monthly_mape_numpy(true, predict),
              new_weekly_mape_numpy(true, predict),
              (np.sum(predict) - np.sum(true)) / np.sum(true),
              mse(true, predict),
              mae(true, predict)]

    fiveG_result.append(result)
fiveG_result = pd.DataFrame(fiveG_result, columns=['mode', '扇区编号', '整体平均值偏差', '月粒度平均值偏差',
                                                   '周粒度平均值偏差', '增幅', 'mse', 'mae'])
fiveG_result.to_csv('../result/single4g.csv')

# sum_result_5g = calculate_acc(fiveG_result, 'scene')
fiveG_result.to_csv('../result/statistic_result_' + mode + '.csv', index=False)

plt.figure(figsize=(10, 10))
binsize = np.arange(0, 60, 5)

sns.distplot(a=fiveG_result['整体平均值偏差'], label="整体平均值偏差", kde=False, bins=binsize)
sns.distplot(a=fiveG_result['月粒度平均值偏差'], label="月粒度平均值偏差", kde=False, bins=binsize)
sns.distplot(a=fiveG_result['周粒度平均值偏差'], label="周粒度平均值偏差", kde=False, bins=binsize)
plt.title('4，5g同覆盖扇区 ' + mode + '流量预测结果')
plt.xlabel('偏差率')
plt.ylabel('基站计数')
plt.legend()
plt.savefig('../figures/' + mode + '_result.png')
plt.show()
