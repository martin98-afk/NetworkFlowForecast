from config import *
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from config import *
from CalculateAcc import calculate_acc
from models.Losses import new_weekly_mape_numpy, new_monthly_mape_numpy
from sklearn.metrics import r2_score
from PlotFunction import *


# 平均值偏差numpy版
def new_mape_numpy(x, y):
    mean_x = np.mean(np.abs(x))
    mean_y = np.mean(np.abs(y))
    return np.abs((mean_x - mean_y) / ((mean_x + mean_y) / 2)) * 100


def msle(x, y):
    return np.mean(np.square(np.log(1 + x) - np.log(1 + y)))


def mse(x, y):
    return np.mean(np.square(x - y))


def mae(x, y):
    return np.mean(np.abs(x - y))


path = "../han's_result/20-30扬州"
files = os.listdir(path)

mape_list = []
increase = pd.DataFrame()
fiveG_result = []
for file_name in files:
    print(os.path.join(path, file_name))
    data = pd.read_csv(os.path.join(path, file_name))
    # data2 = pd.read_csv(path[:-3] +'_60' + '/' + file_name)
    true = data['true'].values
    predict = data['pred'].values
    file_name.replace('1-10', '')
    result = ['na_test_4g_01_10', file_name,
              new_mape_numpy(true, predict),
              new_monthly_mape_numpy(true, predict),
              new_weekly_mape_numpy(true, predict),
              (np.sum(predict) - np.sum(true)) / np.sum(true),
              mse(true, predict),
              mae(true, predict)]

    fiveG_result.append(result)

fiveG_result = pd.DataFrame(fiveG_result, columns=['mode', '扇区编号', '整体平均值偏差', '月粒度平均值偏差',
                                                   '周粒度平均值偏差', '增幅', 'mse', 'mae'])
# sum_result_5g = calculate_acc(fiveG_result, 'scene')
fiveG_result.to_csv('../result/statistic_result_han_' + path.split('/')[2] + '.csv',
                    index=False)

plt.figure(figsize=(10, 10))
binsize = np.arange(0, 60, 5)

sns.distplot(a=fiveG_result['整体平均值偏差'], label="整体平均值偏差", kde=False, bins=binsize)
sns.distplot(a=fiveG_result['月粒度平均值偏差'], label="月粒度平均值偏差", kde=False, bins=binsize)
sns.distplot(a=fiveG_result['周粒度平均值偏差'], label="周粒度平均值偏差", kde=False, bins=binsize)
plt.title('4，5g同覆盖扇区 ' + path.split('/')[2][-2:] + '_' + mode + '流量预测结果')
plt.xlabel('偏差率')
plt.ylabel('基站计数')
plt.legend()
plt.savefig('../figures/' + path.split('/')[2][-2:] + '_' + mode + '_result.png')
plt.show()
