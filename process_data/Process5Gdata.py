import os
from config import *

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime


#### 筛选没有缺失值的小区cgi编号
def FilterCGI(data_path='../5G', latest_date='2020-07-01'):
    '''
    过滤没有缺失值的小区cgi
    :param data_path:
    :return:
    '''

    fold_name = os.listdir(data_path)
    fold_name.sort()
    if os.path.exists(data_path + 'processed_data/missing_value/null_ct_df_5g.csv'):
        data = pd.read_csv(data_path + 'processed_data/missing_value/null_ct_df_5g.csv')
        data.sort_values('end_date', inplace=True)
        last_date = data.iloc[-1]['end_date']
        if fold_name[-1] == last_date[:7]:
            print('已统计过缺失值，跳过此步骤!')
            return
    # 获取缺失值统计表中更新的最新日期
    # null_ct_df = pd.DataFrame(
    #     columns=['cgi', 'start_date', 'null_count', 'end_date', 'null_date'])
    null_ct_df = pd.DataFrame(columns=['cgi', 'start_date', 'end_date', 'null_count'])

    null_ct_df.set_index('cgi', inplace=True)
    count = 0
    for name in fold_name:
        file_list = os.listdir(os.path.join(data_path, name))
        file_list.sort()
        for file_name in file_list:
            if file_name[-2:] == 'gz':
                filter_name = file_name.replace('5G-', '')
                date = filter_name[:10]
                if date <= latest_date:
                    print(date + '日期数据超出规定日期，跳过！')
                    continue
                print('正在处理: ', date, '日期的5G数据')
                data = pd.read_csv(os.path.join(data_path, name, file_name),
                                   compression='gzip', encoding='gbk',
                                   error_bad_lines=False)
                data.drop(data[pd.isnull(data['下行流量(TB)'])].index, inplace=True)
                data.drop(data[pd.isnull(data['上行流量(TB)'])].index, inplace=True)
                data.drop(data[pd.isnull(data['cgi'])].index, inplace=True)

                input_cgi_set = set(data.cgi.unique())
                cgi_set = set(null_ct_df.index.unique())
                old_cgi_set = cgi_set & input_cgi_set
                null_cgi_set = cgi_set - old_cgi_set
                new_cgi_set = input_cgi_set - old_cgi_set

                null_ct_df['null_count'].loc[null_cgi_set] = null_ct_df['null_count'].loc[
                                                                 null_cgi_set] + 1
                new_content = np.array(
                    [list(new_cgi_set), [date] * len(new_cgi_set), [date] * len(new_cgi_set),
                     [count] * len(new_cgi_set)]).T
                new_df = pd.DataFrame(new_content,
                                      columns=['cgi', 'start_date', 'end_date', 'null_count'])
                new_df.set_index('cgi', inplace=True)
                new_df['null_count'] = new_df['null_count'].astype('int')
                null_ct_df = pd.concat([null_ct_df, new_df])

                null_ct_df['end_date'].loc[input_cgi_set] = date

                count += 1
                print(null_ct_df)

    null_ct_df.to_csv(data_path + 'processed_data/missing_value/null_ct_df_5g.csv')


def ExtractFlowData(path='../5G'):
    '''
    提取小区流量数据，
    :param path: 全量数据路径
    :return: None
    '''
    path1 = '../5Gprocessed_data/'
    # 提取没有缺失值的小区cgi
    try:
        exist_data = pd.read_csv('../5Gprocessed_data/5G全量数据.csv', index_col='cgi')
        cgi_total_list = exist_data.index.unique()
        exist_date = read_process_date()
    except:
        with open(path1 + 'complete_data_cgi.txt', 'r') as f:
            cgi_total_list = []
            for line in f.readlines():
                cgi_total_list.append(line[:-1])
            exist_date = []

    dataframe = pd.DataFrame()
    fold_name = os.listdir(path)
    for name in fold_name:

        for file_name in os.listdir(path + '/' + name):
            filter_name = file_name.replace('5G-', '')

            if file_name[-2:] == 'gz' and filter_name[:10] not in exist_date:

                print('正在处理', filter_name[:10], '日期的数据')
                data = pd.read_csv(path + '/' + name + '/' + file_name, compression='gzip',
                                   encoding='gbk', error_bad_lines=False)
                data.set_index('cgi', inplace=True)
                if len(set(cgi_total_list) & set(data.index.unique())) == 0:
                    print('没有匹配到任何相同的cgi')
                    continue
                data = data.loc[set(cgi_total_list) & set(data.index.unique())]

                try:
                    data['download'] = data[['下行流量(TB)', '上行流量(TB)']].apply(lambda x: x.sum(),
                                                                            axis=1)
                    data.drop(['小区名称', '下行流量(TB)', '上行流量(TB)', '用户数量'], axis=1, inplace=True)
                except:
                    data['download'] = data[['PDCP_UpOctDl', 'PDCP_UpOctUl']].apply(
                        lambda x: x.sum(), axis=1)
                    data['时间'] = data['start_time']
                    data.drop(['start_time', 'EUTRANCELLTDD_NAME', 'PDCP_UpOctDl', 'PDCP_UpOctUl',
                               'RRC_CONNMEAN'], axis=1, inplace=True)
                # 如果遇到有cgi一天有多个数据，取所有download的最大值
                avg_download = pd.DataFrame(data.groupby('cgi')['download'].max())
                avg_download['时间'] = np.array([filter_name[:10]] * len(avg_download.index))
                dataframe = pd.concat([dataframe, avg_download])

    return dataframe


def TransformDateFormat(dataframe, timelabel):
    '''
    转换csv文件中timestamp列的时间数据格式，转换为yyyy-mm-dd格式的日期。
    :param path:
    :return:
    '''
    print('------------------------------------------------------')
    print('数据录入完毕， 开始处理时间格式')
    print('------------------------------------------------------')
    data = dataframe
    try:
        data['timestamp'] = data[timelabel] \
            .apply(lambda x: time.strptime(x, "%b %d %Y %H:%M:%S:%f%p")) \
            .apply(lambda x: time.strftime("%Y-%m-%d", x))
    except:
        data['timestamp'] = data[timelabel]

    data.drop(timelabel, axis=1, inplace=True)
    exist_data = pd.read_csv('../5Gprocessed_data/5G全量数据.csv', index_col='cgi')
    data = pd.concat([exist_data, data])
    data.sort_values(by='timestamp', inplace=True)
    data.to_csv('../5Gprocessed_data/5G全量数据.csv')


def read_process_date():
    '''
    读取当前处理过的最后日期，方便新数据进入后单独对新数据做处理
    :return:
    '''
    data = pd.read_csv('../5Gprocessed_data/5G全量数据.csv', index_col='cgi')
    exist_date = list(data['timestamp'].unique())
    return exist_date


def extract_city_data(city_name):
    cgi_info = pd.read_csv('../5Gprocessed_data/处理后5G工参.csv', index_col='CGI')
    cgi_info = cgi_info[cgi_info['城市'] == city_name]
    data = pd.read_csv('../5Gprocessed_data/5G全量数据.csv', index_col='cgi')
    city_cgi = set(cgi_info.index.unique())
    data = data.loc[city_cgi & set(data.index.unique())]
    data.to_csv('../5Gprocessed_data/' + city_name + '.csv')


if __name__ == '__main__':
    # CGICount()
    ...
    # dataframe = ExtractFlowData()
    # if len(dataframe.values) != 0:
    #     TransformDateFormat(dataframe, '时间')
    # else:
    #     print('没有新数据录入!!!')
    # extract_city_data('常州')
    # data = pd.read_csv('../5Gprocessed_data/5G全量数据.csv', index_col='cgi')
    # cgi_info = pd.read_csv('../5Gprocessed_data/处理后5G工参.csv', index_col='CGI')
    # data = pd.merge(data, cgi_info, left_index=True, right_index=True)
    # sum_download = data.groupby(['城市', 'timestamp'])['download'].sum()
    # print(pd.DataFrame(sum_download.loc['南京'], columns=['download']))
    # print(read_process_date())
