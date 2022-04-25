import pandas as pd

from config import *
from CgiTablePreprocess import read_base_station_info


def import_data(path1, path2, not_include=['地市', '区县', '覆盖场景']):
    """
    导入小区数据和4g工参数据，同时join时间表。
    :param path1: 小区数据路径
    :param path2: 4g工参数据路径
    :return: processed_df1: 和时间表join后的小区数据集, df_dummy: 4g工参数据
    """
    # # 导入已经处理过的数据集
    data = pd.read_csv(path1, index_col='cgi')
    data.timestamp = data.timestamp.astype('datetime64[ns]')
    # 综合5g基站信息
    data5g = get_5g_data()
    data5g.index = pd.DatetimeIndex(data5g.index)

    data = pd.merge(data, data5g, how='left', left_on='timestamp', right_index=True)
    data['download5G'] = data['download5G'].fillna(0)

    # 获取南京疫情人数数据
    patient_df = get_patient_data()
    data = pd.merge(data, patient_df, how='left', left_on='timestamp', right_index=True)

    # 获取南京天气数据
    weather_df = get_weather_data()
    data = pd.merge(data, weather_df, how='left', left_on='timestamp', right_index=True)
    data = pd.get_dummies(data, columns=['weather'])

    # 南京小区流量求和用来训练直接预测总量的模型
    processed_df1 = join_date(data)
    processed_df1 = processed_df1.fillna(method='bfill')

    processed_df1 = processed_df1[
        ['timestamp', 'download5G', 'currentConfirmedCount', 'max_tempreture',
         'min_tempreture', 'weather_多云', 'weather_晴', 'weather_雨', 'weather_雪',
         'year', 'month', 'day', 'dayofweek', 'dayofyear', 'data_is_workday_True',
         'on_holiday_True', 'download']
    ]
    # 小区基站信息导入
    cgi_list = data.index.unique()
    df_dummy = read_base_station_info(path2, cgi_list, not_include=not_include)

    df_dummy['地点'] = df_dummy[not_include[0]]
    for i in range(len(not_include) - 1):
        df_dummy['地点'] = df_dummy['地点'] + '_' + df_dummy[not_include[i + 1]]
    df_dummy.drop(not_include, axis=1, inplace=True)
    processed_df1 = processed_df1.sort_values(by='timestamp')
    return processed_df1, df_dummy


def join_date(data):
    # 导入日期信息，提供时间维度特征
    day_df = pd.read_csv('./data/date.csv')
    day_df.set_index('date', inplace=True)
    day_df.index = pd.DatetimeIndex(day_df.index)
    # day_df.drop(['year', 'dayofyear'], axis=1, inplace=True)
    processed_df = pd.merge(day_df, data, how='inner', left_index=True, right_on='timestamp')
    return processed_df


def get_5g_data():
    data = pd.read_csv('./5Gprocessed_data/5G全量数据.csv', index_col='cgi')
    cgi_info = pd.read_csv('./5Gprocessed_data/处理后5G工参.csv', index_col='CGI')
    data = pd.merge(data, cgi_info, left_index=True, right_index=True)
    sum_download = data.groupby(['城市', 'timestamp'])['download'].sum()
    sum_download = pd.DataFrame(sum_download.loc['南京'], columns=['download'])
    sum_download['download5G'] = sum_download['download']
    sum_download.drop('download', axis=1, inplace=True)
    return sum_download


def get_patient_data():
    data = pd.read_csv('./data/patient_df.csv', index_col='f_date')
    data.index = pd.DatetimeIndex(data.index)
    return data


def get_weather_data():
    data = pd.read_csv('./data/weather_df.csv', index_col='f_date')
    data.index = pd.DatetimeIndex(data.index)
    return data


# def sum_result(processed_df, df_dummy):
#     dongtai_cgi = df_dummy[df_dummy['地点'] == '镇江_丹阳市'].index
#     processed_df1 = processed_df.loc[dongtai_cgi]
#     print(processed_df1.info())
#     sum_true = processed_df1.groupby(['month', 'day', 'dayofweek'])['download'].sum()
#     print(sum_true)
#     plt.plot(sum_true.values)
#     plt.show()


if __name__ == '__main__':
    processed_df1, df_dummy = import_data('./data/cgi_2year_qinghuai.csv',
                                          './data/剔除冗余特征后全量工参.csv', )
    print(processed_df1.info())
