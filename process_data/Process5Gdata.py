import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time


#### 筛选没有缺失值的小区cgi编号

def FilterCGI(data_path='../5G'):
    dataframe = []
    fold_name = os.listdir(data_path)
    for name in fold_name:
        for file_name in os.listdir(data_path + '/' + name):
            if file_name[-2:] == 'gz':
                data = pd.read_csv(data_path + '/' + name + '/' + file_name,
                                   compression='gzip', encoding='gbk',
                                   error_bad_lines=False)
                data.drop(data[pd.isnull(data['下行流量(TB)'])].index, inplace=True)
                data.drop(data[pd.isnull(data['cgi'])].index, inplace=True)
                data.drop(data[data['下行流量(TB)'] < 1e-5].index, inplace=True)

                cgi_set = set(data.cgi.unique())
                try:
                    cgi_total_set = cgi_set & cgi_total_set
                except:
                    cgi_total_set = cgi_set
                print(len(cgi_total_set))
    print(len(cgi_total_set))
    with open('../5Gprocessed_data/complete_data_cgi.txt', 'w') as f:
        for cgi in cgi_total_set:
            f.write(cgi + '\n')


def ExtractFlowData(path='../5G'):
    '''
    提取小区流量数据，
    :param path: 全量数据路径
    :return: None
    '''
    path1 = '../5Gprocessed_data/'
    with open(path1 + 'complete_data_cgi.txt', 'r') as f:
        cgi_total_list = []
        for line in f.readlines():
            cgi_total_list.append(line[:-1])
    # cgi_total_list = list(cgi_total_set)
    # cgi_list = filter_df.index.unique()
    dataframe = pd.DataFrame()
    fold_name = os.listdir(path)
    for name in fold_name:
        print('正在处理', name, '日期的数据')
        if name[-3:] == 'txt':
            continue
        for file_name in os.listdir(path + '/' + name):

            if file_name[-2:] == 'gz':
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

                dataframe = pd.concat([dataframe, data])
    dataframe.to_csv(path1 + '/' + '5G全量数据.csv')


def TransformDateFormat(path, timelabel):
    '''
    转换csv文件中timestamp列的时间数据格式，转换为yyyy-mm-dd格式的日期。
    :param path:
    :return:
    '''
    data = pd.read_csv(path, index_col='cgi')
    data['timestamp'] = data[timelabel] \
        .apply(lambda x: time.strptime(x, "%b %d %Y %H:%M:%S:%f%p")) \
        .apply(lambda x: time.strftime("%Y-%m-%d", x))

    data.drop(timelabel, axis=1, inplace=True)
    data.sort_values(by='timestamp', inplace=True)
    data.to_csv(path)


if __name__ == '__main__':
    # FilterCGI()
    # ExtractFlowData()
    # TransformDateFormat('../5Gprocessed_data/5G全量数据.csv', '时间')
    data = pd.read_csv('../5Gprocessed_data/5G全量数据.csv', index_col='cgi')
    cgi_info = pd.read_csv('../5Gprocessed_data/处理后5G工参.csv', index_col='CGI')
    data = pd.merge(data, cgi_info, left_index=True, right_index=True)
    sum_download = data.groupby(['城市', 'timestamp'])['download'].sum()
    print(pd.DataFrame(sum_download.loc['南京'], columns=['download']))
