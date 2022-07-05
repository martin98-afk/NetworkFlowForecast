import numpy as np
import pandas as pd
from models.Losses import new_weekly_mape_numpy, new_monthly_mape_numpy
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_acc(dataframe, index):
    id = ['scene_name', '< 10', '< 20', '< 30', '< 40', '> 40']
    result = []
    scene_list = dataframe[index].unique()
    for scene_name in scene_list:
        filter_data = dataframe[dataframe[index] == scene_name]
        count = [scene_name, filter_data['mape'].median(), filter_data['mape'].mean(

        ), 0, 0, 0, 0, 0]
        for mape in filter_data['mape']:
            count[1] + 1
            if mape <= 10:
                count[3] += 1
            elif mape <= 20:
                count[4] += 1
            elif mape <= 30:
                count[5] += 1
            elif mape <= 40:
                count[6] += 1
            else:
                count[7] += 1
        result.append(count)

    sum_result = []
    for i in range(len(result)):
        sum_result.append(
            [
                result[i][0], np.sum(result[i][3:]), result[i][1], result[i][2],
                dataframe[dataframe[index] == scene_list[i]]['mse'].mean(),
                dataframe[dataframe[index] == scene_list[i]]['mae'].mean(),
                str(np.round(result[i][3] / np.sum(result[i][3:]) * 100)) + '%',
                str(np.round(np.sum(result[i][3:5]) / np.sum(result[i][3:]) * 100)) + '%',
                str(np.round(np.sum(result[i][3:6]) / np.sum(result[i][3:]) * 100)) + '%',
                str(np.round(np.sum(result[i][3:7]) / np.sum(result[i][3:]) * 100)) + '%',
                str(np.round(result[i][-1] / np.sum(result[i][3:]) * 100)) + '%',

            ]
        )
    sum_result = pd.DataFrame(sum_result,
                              columns=[index, 'cgi_count', 'median', 'mean', 'mse', 'mae', '<10',
                                       '<20', '<30', '<40', '>40'])
    sum_result = sum_result.sort_values(by='median')
    return sum_result


def cal_single_acc(true, predict):
    count = 0
    mape_list = []
    mape_list2 = []
    for i in range(true.shape[0]):
        mape = new_weekly_mape_numpy(true[i, :], predict[i, :])
        mape_list.append(mape)
        mape2 = new_monthly_mape_numpy(true[i, :], predict[i, :])
        mape_list2.append(mape2)
        if mape < 20:
            count += 1
    mape = np.array(sorted(mape_list))
    mape2 = np.array(sorted(mape_list2))
    print('当前周粒度mape均值为： %.2f ' % mape[0] + '%')
    print('当前月粒度mape均值为： %.2f ' % mape2[0] + '%')
    # plt.plot(mape)
    # plt.savefig('./figures/mape_distribution.png')
    # plt.show()

    return count / true.shape[0] * 100, mape[0]
