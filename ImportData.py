import pandas as pd

from config import *
from CgiTablePreprocess import read_base_station_info

def filter_one_cgi(data):
    data = data.sort_values(by='timestamp')
    filtered_data = data.iloc[3:-3]
    download = data['download'].values
    filtered_download = []
    for i in range(3,len(download)-3):
        filtered_download.append(np.mean(download[i-3:i+3]))
    filtered_download = np.array(filtered_download)
    filtered_data['download'] = filtered_download

    return filtered_data

def filter(data):
    cgi_list = data.index.unique()
    filtered_data = pd.DataFrame()
    for cgi in cgi_list:
        filtered_data = pd.concat([filtered_data, filter_one_cgi(data.loc[cgi])])
    return filtered_data


def import_data(path1, path2, not_include=['地市', '区县', '覆盖场景']):
    """
    导入小区数据和4g工参数据，同时join时间表。
    :param path1: 小区数据路径
    :param path2: 4g工参数据路径
    :return: processed_df1: 和时间表join后的小区数据集, df_dummy: 4g工参数据
    """
    # # 导入已经处理过的数据集
    # 综合5g基站信息
    if mode == '4g':
        data = pd.read_csv(path1, index_col='cgi')
        data.timestamp = data.timestamp.astype('datetime64[ns]')
        data5g = get_5g_data()
        data5g.index = pd.DatetimeIndex(data5g.index)

        data = pd.merge(data, data5g, how='left', left_on='timestamp', right_index=True)
        data['download_another'] = data['download5G'].fillna(0)
        data.drop('download5G', axis=1, inplace=True)
        data = data[['timestamp','download_another', 'download']]
        # print('滤波前：',data)
        # data = filter(data)
        # print('滤波后：',data)
        data = merge_extra_info(data)
    else:
        # 综合4g基站信息，使用4g-5g信息匹配表
        data = pd.read_csv(path1, index_col='cgi')
        # data['cgi'] = data['index']
        # data.drop('index', axis=1, inplace=True)

        data.timestamp = data.timestamp.astype('datetime64[ns]')
        # data4g = get_4g_data()
        # data['label'] = data['4g_cgi'] + '-' + data['timestamp'].astype('str')
        #
        # data = pd.merge(data, data4g, how='inner', left_on='label', right_on='label')
        # data['timestamp'] = data['timestamp_x']
        # data.drop(['label','timestamp_x','timestamp_y'], axis=1, inplace=True)
        # data['download_another'] = data['download_another'].fillna(0)
        # data = pd.DataFrame(data.groupby(['cgi', 'timestamp']).mean())
        # data.reset_index(inplace=True)
        # data = data[['cgi','timestamp','download_another', 'download']]
        start_time = data.timestamp.unique()[-history_day]
        data = data[data['timestamp'] > start_time]
        # data.set_index('cgi', inplace = True)
        # data = filter(data)
        data = join_date(data)

    # 小区基站信息导入
    cgi_list = data.index.unique()
    df_dummy = read_base_station_info(path2, cgi_list, not_include=not_include)

    df_dummy['地点'] = df_dummy[not_include[0]]
    for i in range(len(not_include) - 1):
        df_dummy['地点'] = df_dummy['地点'] + '_' + df_dummy[not_include[i + 1]]
    df_dummy.drop(not_include, axis=1, inplace=True)
    data = data.sort_values(by='timestamp')
    return data, df_dummy


def merge_extra_info(data):
    # 添加额外信息
    # 日期信息分解
    data = join_date(data)

    # 获取南京疫情人数数据
    patient_df = get_patient_data()
    data = pd.merge(data, patient_df, how='left', left_on='timestamp', right_index=True)

    # 获取南京天气数据
    weather_df = get_weather_data()
    data = pd.merge(data, weather_df, how='left', left_on='timestamp', right_index=True)
    data = pd.get_dummies(data, columns=['weather'])


    processed_df1 = data.fillna(method='bfill')

    processed_df1 = processed_df1[
        ['timestamp', 'download_another', 'currentConfirmedCount', 'max_tempreture',
         'min_tempreture', 'weather_多云', 'weather_晴', 'weather_雨', 'weather_雪',
         'year', 'month', 'day', 'dayofweek', 'dayofyear', 'data_is_workday_True',
         'on_holiday_True', 'download']
    ]
    return processed_df1

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


def get_4g_data():
    data = pd.read_csv('./4Gprocessed_data/4g_cgi_扬州.csv', index_col='cgi')
    data['download_another'] = data['download']
    data.drop('download', axis=1, inplace=True)
    data['label'] = data.index + '-' + data['timestamp']
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
    not_include = ['城市', '覆盖场景']
    raw_data, base_station_info = import_data(
        './5Gprocessed_data/yangzhou.csv',
        './5Gprocessed_data/处理后5G工参.csv',
        not_include=not_include)
    print(raw_data.info())
    # get_4g_data()
