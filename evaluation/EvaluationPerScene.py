import pylab as pl

from CalculateAcc import calculate_acc
from PlotFunction import *
from config import *
from models.Losses import new_weekly_mape_numpy, new_monthly_mape_numpy


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


path = '.' + result_path + '_' + city_name
files = os.listdir(path)

mape_list = []
increase = pd.DataFrame()
fiveG_result = []
for file_name in files:
    split_name = file_name.split('_')
    if len(split_name) == 5 and mode == split_name[0]:
        print(path + '/' + file_name)
        data = pd.read_csv(path + '/' + file_name)
        # data2 = pd.read_csv(path[:-3] +'_60' + '/' + file_name)
        true = data['true'].values
        predict = data['predict'].values
        result = [split_name[0], '_'.join(split_name[1:3]),
                  new_mape_numpy(true, predict),
                  new_monthly_mape_numpy(true, predict),
                  new_weekly_mape_numpy(true, predict),
                  (np.sum(predict) - np.sum(true)) / np.sum(true),
                  mse(true, predict),
                  mae(true, predict)]
        fiveG_result.append(result)
    elif len(split_name) > 5 and '_'.join(split_name[:5]) == mode:
        print(path + '/' + file_name)
        data = pd.read_csv(path + '/' + file_name)
        # data2 = pd.read_csv(path[:-3] +'_60' + '/' + file_name)
        true = data['true'].values
        predict = data['predict'].values
        result = ['_'.join(split_name[:5]), '_'.join(split_name[5:8]),
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
fiveG_result.to_csv('../result/statistic_result_' + city_name + '_' + mode + '.csv',
                    index=False)

plt.figure(figsize=(10, 10))
binsize = np.arange(0, 60, 5)

sns.distplot(a=fiveG_result['整体平均值偏差'], label="整体平均值偏差", kde=False, bins=binsize)
sns.distplot(a=fiveG_result['月粒度平均值偏差'], label="月粒度平均值偏差", kde=False, bins=binsize)
sns.distplot(a=fiveG_result['周粒度平均值偏差'], label="周粒度平均值偏差", kde=False, bins=binsize)
plt.title('4，5g同覆盖扇区 ' + city_name + '_' + mode + '流量预测结果')
plt.xlabel('偏差率')
plt.ylabel('基站计数')
plt.legend()
plt.savefig('../figures/' + city_name + '_' + mode + '_result.png')
plt.show()
