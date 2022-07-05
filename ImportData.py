import pandas as pd

from config import *
import pymysql
from Scrapy.GetWeather import *


def import_data(path1, path3, mode, city=city_name):
    """
    导入小区数据和4g工参数据，同时join时间表。
    :param path1: 小区数据路径
    :param path2: 工参数据路径
    :param path3: 日期数据路径
    :return: processed_df1: 和时间表join后的小区数据集, df_dummy: 4g工参数据
    """
    # # 导入已经处理过的数据集
    # 综合5g基站信息
    if mode[:10] == 'na_test_4g':
        data = pd.read_csv(path1, index_col='STATION_ID')
        data = pd.DataFrame(data.groupby([data.index, '时间'])['download'].mean())
        data.reset_index(inplace=True)
        data.set_index('STATION_ID', inplace=True)
        data.columns = ['timestamp', 'download']
        data.timestamp = data.timestamp.astype('datetime64[ns]')
        data = data.sort_values('timestamp', ascending=True)
        data = join_date(data, path3)
        weather_df = get_weather_data(city)
        weather_df = pd.get_dummies(weather_df, columns=['weather'])
        data = pd.merge(weather_df, data, how='inner', left_index=True, right_on='timestamp')

    elif mode[:10] == 'na_test_5g':
        data = pd.read_csv(path1, index_col='STATION_ID')
        data = pd.DataFrame(data.groupby([data.index, '时间'])['download'].mean())
        data.reset_index(inplace=True)
        data.set_index('STATION_ID', inplace=True)
        data.columns = ['timestamp', 'download']
        data.timestamp = data.timestamp.astype('datetime64[ns]')
        data = data.sort_values('timestamp', ascending=True)
        data = join_date(data, path3)
        weather_df = get_weather_data(city)
        weather_df = pd.get_dummies(weather_df, columns=['weather'])
        data = pd.merge(weather_df, data, how='inner', left_index=True, right_on='timestamp')


    data = data.sort_values(by='timestamp')
    return data


def merge_extra_info(data, path):
    # 添加额外信息
    # 日期信息分解
    data = join_date(data, path)

    # 获取南京疫情人数数据
    # patient_df = get_patient_data()
    # data = pd.merge(data, patient_df, how='left', left_on='timestamp', right_index=True)

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


def join_date(data, path):
    # 导入日期信息，提供时间维度特征
    day_df = pd.read_csv(path)
    day_df.set_index('date', inplace=True)
    day_df.index = pd.DatetimeIndex(day_df.index)
    # day_df.drop(['year', 'dayofyear'], axis=1, inplace=True)
    processed_df = pd.merge(day_df, data, how='inner', left_index=True, right_on='timestamp')
    return processed_df


# def get_5g_data():
#     data = pd.read_csv('./5Gprocessed_data/5G全量数据.csv', index_col='cgi')
#     cgi_info = pd.read_csv('./5Gprocessed_data/处理后5G工参.csv', index_col='CGI')
#     data = pd.merge(data, cgi_info, left_index=True, right_index=True)
#     data.timestamp = data.timestamp.astype('datetime64[ns]')
#     sum_download = data.groupby(['城市', 'timestamp'])['download'].mean()
#     sum_download = pd.DataFrame(sum_download.loc[city_name], columns=['download'])
#     sum_download['download5G'] = sum_download['download']
#     sum_download.drop('download', axis=1, inplace=True)
#     return sum_download


# def get_patient_data():
#     data = pd.read_csv('./data/patient_df.csv', index_col='f_date')
#     data.index = pd.DatetimeIndex(data.index)
#     return data


def get_weather_data(city):
    update_info('./data/', city)
    data = pd.read_csv('./data/weather_df_' + city + '.csv')
    data.f_date = data.f_date.astype('datetime64[ns]')
    data.set_index('f_date', inplace=True)
    return data


# def get_4g_data():
#     data = pd.read_csv('./4Gprocessed_data/4g_cgi_扬州.csv', index_col='cgi')
#     data['download_another'] = data['download']
#     data.drop('download', axis=1, inplace=True)
#     data['label'] = data.index + '-' + data['timestamp']
#     return data


def extract_city_data(table_name, table_name2, city_name):
    conn = pymysql.connect(**config)
    sql = f"""
            SELECT a.cgi, a.download, a.timestamp
            FROM {table_name} a
            JOIN {table_name2} b
            ON a.cgi = b.CGI
            WHERE b.city = "{city_name}"
            """
    df = pd.read_sql(sql, conn)
    return df


if __name__ == '__main__':
    get_weather_data(city_name)
    # data, _ = import_data('./4Gprocessed_data/missing_value/fillna_example1-10.csv',
    #                        './5Gprocessed_data/处理后5G工参.csv',
    #                        './data/date.csv',
    #                        not_include=['城市', '覆盖类型'])
    # print(data)
    # not_include = ['城市', '覆盖场景']
    # raw_data, base_station_info = import_data(
    #     './5Gprocessed_data/扬州SECTOR合并.csv',
    #     './5Gprocessed_data/处理后5G工参.csv',
    #     './data/date.csv',
    #     not_include=not_include)
    # print(raw_data.info())
    # print(raw_data)
    # plt.figure(figsize=(15,15))
    # raw_data.groupby(raw_data.index)['timestamp'].count().plot()
    # plt.plot(raw_data.timestamp.unique(), raw_data.loc['84AN_2ABP']['4Gdownload'])
    # plt.plot(raw_data.timestamp.unique(), raw_data.loc['84AN_2ABP']['5Gdownload'])
    # plt.xticks(rotation=90)
    # plt.show()
