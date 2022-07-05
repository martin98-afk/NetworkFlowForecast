import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

font1 = {'family': 'SimHei',
         'weight': 'normal',
         'size': 20,
         }

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', font1)
plt.ylabel('True Positive Rate', font1)

# plt.title('ROC',font1)

plt.legend(loc=6, prop=font1)

plt.tick_params(labelsize=13)  # 刻度字体大小13

plt.show()


# 平均值偏差numpy版
def new_mape_numpy(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    return np.abs((mean_x - mean_y) / mean_y) * 100


files = os.listdir('../result_region')

mape_list = []
for file_name in files:
    data = pd.read_csv('../result_region/' + file_name)
    split_name = file_name.split('_')
    if len(split_name) == 3:
        true = data['true'].values
        predict = data['predict'].values
        mape_list.append([split_name[0], split_name[1], split_name[2][:-4],
                          new_mape_numpy(true, predict)])

mape_df = pd.DataFrame(mape_list, columns=['city', 'region', 'cgi', 'mape'])

plt.figure(figsize=(50, 20))
sns.violinplot("region", "mape", data=mape_df)
plt.xlabel('覆盖场景', font1)
plt.ylabel('平均值偏差', font1)
plt.xticks(rotation=90)
plt.ylim([0, 100])
plt.title('秦淮区各覆盖场景小区预测准确率分布', font1)
plt.savefig('../figures/region_result1.png')
plt.show()
#
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
plt.savefig('../figures/region_result2.png')
pl.xticks(rotation=90)
plt.show()

plt.figure(figsize=(20, 20))
changjia_mean = pd.DataFrame(mape_df.groupby(['覆盖场景'])['mape'].median()).sort_values(
    by='mape')
plt.bar(changjia_mean.index, changjia_mean['mape'].values)
pl.xticks(rotation=90)
plt.show()
