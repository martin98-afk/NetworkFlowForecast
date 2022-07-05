import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time


#### 筛选没有缺失值的小区cgi编号
def FilterCGI(data_path='../4G', latest_date='2020-07-01'):
    '''
    过滤没有缺失值的小区cgi
    :param data_path:
    :return:
    '''

    fold_name = os.listdir(data_path)
    fold_name.sort()
    if os.path.exists(data_path + 'processed_data/missing_value/null_ct_df_4g.csv'):
        data = pd.read_csv(data_path + 'processed_data/missing_value/null_ct_df_4g.csv')
        data.sort_values('end_date', inplace=True)
        last_date = data.iloc[-1]['end_date']
        if fold_name[-1] == last_date[:7]:
            print('已统计过缺失值，跳过此步骤!')
            return
    # 获取缺失值统计表中更新的最新日期

    null_ct_df = pd.DataFrame(columns=['cgi', 'start_date', 'end_date', 'null_count'])
    null_ct_df.set_index('cgi', inplace=True)
    count = 0

    for name in fold_name:
        file_list = os.listdir(data_path + '/' + name)
        file_list.sort()
        for file_name in file_list:
            if file_name[-2:] == 'gz':
                if file_name[:2] == '4G':
                    date = file_name[3:13]
                else:
                    date = file_name[:7] + '-' + file_name[7:9]
                if date <= latest_date:
                    print(date + '日期数据超出规定日期，跳过！')
                    continue
                print('正在处理: ', date, '日期的4G数据')
                data = pd.read_csv(data_path + '/' + name + '/' + file_name,
                                   compression='gzip', encoding='gbk',
                                   error_bad_lines=False)
                try:
                    data.drop(data[pd.isnull(data['下行流量'])].index, inplace=True)
                    data.drop(data[pd.isnull(data['上行流量'])].index, inplace=True)
                except:
                    data.drop(data[pd.isnull(data['PDCP_UpOctDl'])].index, inplace=True)
                    data.drop(data[pd.isnull(data['PDCP_UpOctUl'])].index, inplace=True)

                data.drop(data[pd.isnull(data['cgi'])].index, inplace=True)
                # data.drop(data[data['下行流量(TB)'] < 1e-5].index, inplace=True)

                input_cgi_set = set(data.cgi.unique())
                cgi_set = set(null_ct_df.index.unique())
                old_cgi_set = cgi_set & input_cgi_set
                null_cgi_set = cgi_set - old_cgi_set
                new_cgi_set = input_cgi_set - old_cgi_set

                # 缺失的cgi 缺失数量加1
                null_ct_df['null_count'].loc[null_cgi_set] = null_ct_df['null_count'].loc[
                                                                 null_cgi_set] + 1
                ######################################
                # 新增的cgi记录信息
                new_content = np.array(
                    [list(new_cgi_set),
                     [date] * len(new_cgi_set),
                     [date] * len(new_cgi_set),
                     [count] * len(new_cgi_set)]).T
                new_df = pd.DataFrame(new_content,
                                      columns=['cgi', 'start_date', 'end_date', 'null_count'])

                new_df.set_index('cgi', inplace=True)
                new_df['null_count'] = new_df['null_count'].astype('int')
                null_ct_df = pd.concat([null_ct_df, new_df])
                ####################################
                # 旧cgi更新最后有数据日期
                null_ct_df['end_date'].loc[old_cgi_set] = date

                count += 1
                print(null_ct_df)

    null_ct_df.to_csv(data_path + 'processed_data/missing_value/null_ct_df_4g.csv')


def write_exist_date(exist_date):
    '''
    写入已录入数据时间信息
    :param exist_date:
    :return:
    '''
    print('------------------------------------------------------')
    print(f'读取到已经录入的信息为: {exist_date}')
    print('------------------------------------------------------')
    with open('../4Gprocessed_data/exist_date.txt', 'w') as f:
        for i, date in enumerate(exist_date):
            f.write(date + '\n')


def write_exist_cgi(exist_cgi):
    '''
    写入已录入数据cgi信息
    :param exist_cgi:
    :return:
    '''
    print('------------------------------------------------------')
    print(f'读取到已经录入的信息为: {len(exist_cgi)}')
    print('------------------------------------------------------')
    with open('../4Gprocessed_data/exist_cgi.txt', 'w') as f:
        for i, date in enumerate(exist_cgi):
            f.write(date + '\n')


def ExtractFlowData(path='../4G/1月'):
    '''
    提取小区流量数据，按每月单独提取，一次性全部提取内存不够
    :param path: 全量数据路径
    :return: None
    '''
    path1 = '../4Gprocessed_data/'
    # 提取没有缺失值的小区cgi
    # 读取已录入数据的cgi信息
    if os.path.exists('../4Gprocessed_data/exist_cgi.txt'):
        with open('../4Gprocessed_data/exist_cgi.txt', 'r') as f:
            exist_cgi = f.read().split('\n')
        print('------------------------------------------------------')
        print(f'读取到已经录入的信息为: {len(exist_cgi)}')
        print('------------------------------------------------------')
        cgi_total_list = exist_cgi

    else:
        exist_data = pd.read_csv('../4Gprocessed_data/4G全量数据.csv', index_col='cgi')
        cgi_total_list = exist_data.index.unique()
        print('------------------------------------------------------')
        print(f'读取到已经录入的信息为: {len(cgi_total_list)}')
        print('------------------------------------------------------')
        write_exist_cgi(cgi_total_list)

    # 读取已录入数据的日期
    if os.path.exists('../4Gprocessed_data/exist_date.txt'):
        with open('../4Gprocessed_data/exist_date.txt', 'r') as f:
            exist_date = f.read().split('\n')
        print('------------------------------------------------------')
        print(f'读取到已经录入的信息为: {exist_date}')
        print('------------------------------------------------------')
    else:
        exist_date = read_process_date()
        write_exist_date(exist_date)

    dataframe = pd.DataFrame()

    for file_name in os.listdir(path):
        if file_name[:2] == '4G':
            date = file_name[3:13]
        else:
            date = file_name[:7] + '-' + file_name[7:9]
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
                data['download'] = data[['下行流量', '上行流量']].apply(lambda x: x.sum(),
                                                                axis=1)
                data.drop(['小区名称', '下行流量', '上行流量', '用户数量'], axis=1, inplace=True)
            except:
                data['download'] = data[['PDCP_UpOctDl', 'PDCP_UpOctUl']].apply(
                    lambda x: x.sum(), axis=1)
                data['时间'] = data['start_time']
                data.drop(['start_time', 'EUTRANCELLTDD_NAME', 'PDCP_UpOctDl', 'PDCP_UpOctUl',
                           'RRC_CONNMEAN'], axis=1, inplace=True)
            # 如果遇到有cgi一天有多个数据，取所有download的最大值
            avg_download = pd.DataFrame(data.groupby('cgi')['download'].max())
            avg_download['时间'] = np.array([date] * len(avg_download.index))
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
    data['timestamp'] = data[timelabel]
    # .apply(lambda x: time.strptime(x, "%b %d %Y %H:%M:%S:%f%p")) \
    # .apply(lambda x: time.strftime("%Y-%m-%d", x))

    data.drop(timelabel, axis=1, inplace=True)

    with open('../4Gprocessed_data/exist_date.txt', 'r') as f:
        exist_date = f.read().split('\n')
    exist_date.extend(data['timestamp'].unique())
    write_exist_date(exist_date)

    if os.path.exists('../4Gprocessed_data/4G全量数据_2.csv'):
        exist_data = pd.read_csv('../4Gprocessed_data/4G全量数据_2.csv', index_col='cgi')
        data = pd.concat([exist_data, data])
    data.sort_values(by='timestamp', inplace=True)
    print('与历史数据融合完毕，准备录入结果')
    data.to_csv('../4Gprocessed_data/4G全量数据_2.csv')


def read_process_date():
    '''
    读取当前处理过的最后日期，方便新数据进入后单独对新数据做处理
    :return:
    '''
    data = pd.read_csv('../4Gprocessed_data/4G全量数据.csv', index_col='cgi')
    exist_date = list(data['timestamp'].unique())
    return exist_date


def extract_city_data(city_name):
    cgi_info = pd.read_csv('../4Gprocessed_data/剔除冗余特征后全量工参.csv', index_col='CGI')
    cgi_info = cgi_info[cgi_info['地市'] == city_name]
    city_cgi = set(cgi_info.index.unique())
    data = pd.read_csv('../4Gprocessed_data/4G全量数据.csv', index_col='cgi')
    data = data.loc[city_cgi & set(data.index.unique())]

    data2 = pd.read_csv('../4Gprocessed_data/4G全量数据_2.csv', index_col='cgi')
    data2 = data2.loc[city_cgi & set(data2.index.unique())]
    data = pd.concat([data, data2])
    data.sort_values('timestamp', inplace=True)
    data.to_csv('../4Gprocessed_data/' + city_name + '.csv')


if __name__ == '__main__':
    FilterCGI()
    # folders = os.listdir('../4G')
    #
    # # for folder_name in folders:
    # #     print('------------------------------------------------------')
    # #     print(f'正在处理{folder_name}文件夹中的数据')
    # #     print('------------------------------------------------------')
    # #     folder_name = os.path.join('../4G', folder_name)
    # #     dataframe = ExtractFlowData(folder_name)
    # #     if len(dataframe.values) != 0:
    # #         TransformDateFormat(dataframe, '时间')
    # #     else:
    # #         print('没有新数据录入!!!')
    # extract_city_data('南京')
