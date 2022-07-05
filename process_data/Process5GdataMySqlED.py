import os
from config import *

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import pymysql

# cur = conn.cursor()
table_name = 'flow_5g'


#### 筛选没有缺失值的小区cgi编号
def FilterCGI(data_path='../5G'):
    '''
    过滤没有缺失值的小区cgi
    :param data_path:
    :return:
    '''
    dataframe = []
    try:
        with open('../5Gprocessed_data/complete_data_cgi.txt', 'r') as f:
            cgi_total_set = []
            for line in f.readlines():
                cgi_total_set.append(line[:-1])
            cgi_total_set = set(cgi_total_set)
    except:
        cgi_total_set = None

    fold_name = os.listdir(data_path)
    for name in fold_name:
        for file_name in os.listdir(data_path + '/' + name):
            if file_name[-2:] == 'gz':
                data = pd.read_csv(data_path + '/' + name + '/' + file_name,
                                   compression='gzip', encoding='gbk',
                                   error_bad_lines=False)
                data.drop(data[pd.isnull(data['下行流量'])].index, inplace=True)
                data.drop(data[pd.isnull(data['cgi'])].index, inplace=True)
                data.drop(data[data['下行流量'] < 1e-5].index, inplace=True)

                cgi_set = set(data.cgi.unique())
                if cgi_total_set != None:
                    cgi_total_set = cgi_set & cgi_total_set
                else:
                    cgi_total_set = cgi_set

                print(len(cgi_total_set))

    print(len(cgi_total_set))
    with open('../5Gprocessed_data/complete_data_cgi.txt', 'w') as f:
        for cgi in cgi_total_set:
            f.write(cgi + '\n')


def write_exist_date(exist_date):
    '''
    写入已录入数据时间信息
    :param exist_date:
    :return:
    '''
    print('------------------------------------------------------')
    print(f'读取到已经录入的信息为: {exist_date}')
    print('------------------------------------------------------')
    with open('../5Gprocessed_data/exist_date.txt', 'w') as f:
        for i, date in enumerate(exist_date):
            f.write(date + '\n')


def ExtractFlowData(path='../5G/2021-01'):
    '''
    提取小区流量数据，按每月单独提取，一次性全部提取内存不够
    :param path: 全量数据路径
    :return: None
    '''
    path1 = '../5Gprocessed_data/'
    # 提取没有缺失值的小区cgi
    conn = pymysql.connect(**config)
    CreateTable()
    # 读取已录入数据的cgi信息
    with open('../5Gprocessed_data/complete_data_cgi.txt', 'r') as f:
        exist_cgi = f.read().split('\n')
    print('------------------------------------------------------')
    print(f'读取到已经录入的信息为: {len(exist_cgi)}')
    print('------------------------------------------------------')
    cgi_total_list = exist_cgi

    # 读取已录入数据的日期
    exist_date, exist_cgi = read_process_date()
    exist_date = list(exist_date)
    if exist_cgi != []:
        cgi_total_list = exist_cgi
    print('------------------------------------------------------')
    print(f'读取到已经录入的信息为: {exist_date}')
    print('------------------------------------------------------')

    sql = """SELECT * FROM stationInfo_5g"""
    cgi_info_df = pd.read_sql(sql, conn)

    for file_name in os.listdir(path):
        filter_name = file_name.replace('5G-', '')
        date = filter_name[:10]
        if file_name[-2:] == 'gz' and date not in exist_date:

            print('正在处理', date, '日期的数据')
            data = pd.read_csv(path + '/' + file_name, compression='gzip',
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
            avg_download['时间'] = np.array([date] * len(avg_download.index))
            avg_download = avg_download.reset_index()
            avg_download = pd.merge(avg_download, cgi_info_df, left_on='cgi', right_on='CGI')
            avg_download = avg_download[['cgi', 'download', '时间', 'city']]
            TransformDateFormat(avg_download, '时间')
        else:
            print(date + ' 数据已录入数据库！！')


def CreateTable():
    conn = pymysql.connect(**config)
    cursor = conn.cursor()
    # 建表操作
    try:
        sql_creat1 = f"""CREATE TABLE if not exists {table_name}(
                        cgi VARCHAR(255),
                        download FLOAT,
                        timestamps VARCHAR(255),
                        city VARCHAR(255),
                        PRIMARY KEY (cgi, timestamps)
                        ) ENGINE=InnoDB DEFAULT  CHARSET=utf8mb4
                        PARTITION BY LIST (city)"""
        cursor.execute(sql_creat1)
    except UserWarning:
        print("建表失败")


def TransformDateFormat(dataframe, timelabel):
    '''
    转换csv文件中timestamp列的时间数据格式，转换为yyyy-mm-dd格式的日期。
    :param path:
    :return:
    '''
    conn = pymysql.connect(**config)
    cursor = conn.cursor()

    print('------------------------------------------------------')
    print('数据录入完毕， 开始处理时间格式')
    print('------------------------------------------------------')
    data = dataframe
    data['timestamps'] = data[timelabel]

    data.drop(timelabel, axis=1, inplace=True)

    # 写入数据
    key_sql = ','.join(dataframe.keys())
    value_sql = ','.join(['%s'] * dataframe.shape[1])
    values = dataframe.values.tolist()

    insert_data_sql = """insert into %s (%s) values (%s)""" % (table_name, key_sql, value_sql)
    cursor.executemany(insert_data_sql, values)
    conn.commit()
    print('数据写入成功!!!')
    cursor.close()
    conn.close()


def read_process_date():
    '''
    读取当前处理过的最后日期，方便新数据进入后单独对新数据做处理
    :return:
    '''

    conn = pymysql.connect(**config)
    sql = f"SELECT distinct timestamps FROM {table_name} ORDER BY timestamps"
    df = pd.read_sql(sql, conn)

    sql = f"SELECT distinct cgi FROM {table_name}"
    cgi = pd.read_sql(sql, conn)
    conn.close()
    if df.empty:
        print('无过往数据')
        return [], []
    else:
        exist_date = df.values.squeeze()
        exist_cgi = cgi.values.squeeze()
        return exist_date, exist_cgi


def extract_city_data(city_name):
    conn = pymysql.connect(**config)
    sql = f"""
            SELECT a.cgi, a.download, a.timestamps
            FROM {table_name} a
            JOIN (SELECT * FROM stationInfo_5g WHERE city = "{city_name}") b
            ON a.cgi = b.CGI
            """
    df = pd.read_sql(sql, conn)
    conn.close()
    df.to_csv('../5Gprocessed_data/' + city_name + '.csv')


if __name__ == '__main__':
    # FilterCGI()
    folders = os.listdir('../5G')

    # for folder_name in folders:
    #     print('------------------------------------------------------')
    #     print(f'正在处理{folder_name}文件夹中的数据')
    #     print('------------------------------------------------------')
    #     folder_name = os.path.join('../5G', folder_name)
    #     ExtractFlowData(folder_name)

    extract_city_data('南京')
    # print(read_process_date())

    # sql = "SELECT a.cgi, a.download, a.timestamp " \
    #       "FROM flow_5g a " \
    #       "JOIN stationInfo_5g b " \
    #       "ON a.cgi = b.CGI " \
    #       "WHERE b.city = " + '"扬州"'
    # df = pd.read_sql(sql, conn)
    # print(df)
