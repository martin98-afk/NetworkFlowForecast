import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from config import *
from CalculateAcc import calculate_acc

from sklearn.metrics import r2_score


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


path = '.' + result_path
files = os.listdir(path)

mape_list = []
increase = pd.DataFrame()
fiveG_result = []
for file_name in files:
    split_name = file_name.split('_')
    if len(split_name) == 4 and mode == '4g':
        data = pd.read_csv(path + '/' + file_name)
        data = data.iloc[-180:]
        true = data['true'].values
        predict = data['predict'].values
        result_increase = pd.DataFrame(np.array([split_name[3][:-4], split_name[2],
                                                 np.sum(true), np.sum(predict),
                                                 (np.sum(predict) - np.sum(true)) / np.sum(
                                                     true)]).reshape(1, 5),
                                       columns=['cgi', '场景名', '6月真实总量', '6月预测总量', '180天预测总量增幅'])
        increase = pd.concat([increase, result_increase])

        mape_list.append([split_name[0], split_name[1], split_name[2], split_name[3][:-4],
                          new_mape_numpy(true, predict),
                          np.abs((np.sum(predict) - np.sum(true)) / np.sum(true)),
                          mse(true, predict),
                          mae(true, predict)
                          ])
    elif len(split_name) == 3:
        data = pd.read_csv(path + '/' + file_name)
        # data2 = pd.read_csv(path[:-3] +'_60' + '/' + file_name)
        true = data['true'].values
        predict = data['predict'].values
        result = [split_name[0], split_name[1], split_name[2][:-4],
                  new_mape_numpy(true, predict),
                  (np.sum(predict) - np.sum(true)) / np.sum(true),
                  mse(true, predict),
                  mae(true, predict)]

        fiveG_result.append(result)

if mode == '5g':
    fiveG_result = pd.DataFrame(fiveG_result, columns=['city', 'scene', 'cgi', 'mape', 'increase',
                                                       'mse', 'mae'])
    sum_result_5g = calculate_acc(fiveG_result, 'scene')

    sum_result_5g.to_csv('../result/statistic_result5g.csv', index=False)

elif mode == '4g':
    #
    increase.to_csv('../result/180天预测增幅结果.csv', index=False)
    mape_df = pd.DataFrame(mape_list, columns=['city', 'region', 'scene', 'cgi', 'mape', 'increase',
                                               'mse', 'mae'])

    sum_result_4g = calculate_acc(mape_df, 'scene')
    sum_result_4g.to_csv('../result/statistic_result4g.csv', index=False)
    plt.figure(figsize=(50, 20))
    sns.violinplot("scene", "mape", data=mape_df)
    plt.xlabel('覆盖场景', font1)
    plt.ylabel('平均值偏差', font1)
    plt.xticks(rotation=90)
    plt.ylim([0, 100])
    plt.title('秦淮区各覆盖场景小区预测准确率分布', font1)
    plt.savefig('../figures/scene_result.png')
    plt.show()
    # # 剔除有问题的覆盖场景计算总的准确率
    delete = ['集贸市场', '会展中心', '城中村', '别墅群', '高铁', '高速公路', '风景区', '地铁', '其他', '高校']

    index = 'scene'
    id = ['< 10', '< 20', '< 30', '< 40', '> 40']
    count = [0, 0, 0, 0, 0]
    result = []
    scene_list = mape_df[index].unique()
    for scene_name in scene_list:
        if scene_name not in delete:

            filter_data = mape_df[mape_df[index] == scene_name]
            for mape in filter_data['mape']:
                if mape <= 10:
                    count[0] += 1
                elif mape <= 20:
                    count[1] += 1
                elif mape <= 30:
                    count[2] += 1
                elif mape <= 40:
                    count[3] += 1
                else:
                    count[4] += 1
    count = np.array(count)
    print('剔除有问题场景后计算的总小区准确率:')
    count = [
        str(np.round(count[0] / np.sum(count) * 100)) + '%',
        str(np.round(np.sum(count[:2]) / np.sum(count) * 100)) + '%',
        str(np.round(np.sum(count[:3]) / np.sum(count) * 100)) + '%',
        str(np.round(np.sum(count[:4]) / np.sum(count) * 100)) + '%',
        str(np.round(count[-1] / np.sum(count) * 100)) + '%']
    print(count)

    # # 画预测结果准确率分布图
    base_station_info = pd.read_csv('../data/剔除缺失值全量小区工参.csv')
    mape_df = mape_df.merge(base_station_info, left_on='cgi', right_on='CGI')
    print(mape_df.head())

    changjia_mean = pd.DataFrame(mape_df.groupby(['厂家名称'])['mape'].median())
    plt.bar(changjia_mean.index, changjia_mean['mape'].values)
    plt.show()

    plt.figure(figsize=(20, 20))
    changjia_mean = pd.DataFrame(mape_df.groupby(['region'])['mape'].median()).sort_values(
        by='mape')
    plt.bar(changjia_mean.index, changjia_mean['mape'].values)
    plt.savefig('../figures/region_result.png')
    pl.xticks(rotation=90)
    plt.show()

    plt.figure(figsize=(20, 20))
    changjia_mean = pd.DataFrame(mape_df.groupby(['覆盖场景'])['mape'].median()).sort_values(
        by='mape', ascending=False)
    plt.barh(changjia_mean.index, changjia_mean['mape'].values)
    plt.title('各个场景小区平均值偏差中位数', font1)
    plt.xlabel('误差率/%', font1)
    plt.grid('on')
    plt.show()

    plt.figure(figsize=(20, 20))
    changjia_mean = pd.DataFrame(mape_df.groupby(['覆盖场景'])['mape'].mean()).sort_values(
        by='mape', ascending=False)
    plt.barh(changjia_mean.index, changjia_mean['mape'].values)
    plt.title('各个场景小区平均值偏差均值', font1)
    plt.xlabel('误差率/%', font1)
    plt.grid('on')
    plt.show()
