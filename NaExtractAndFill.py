import time

import pandas as pd

from config import *
from process_data.ExtractSingleCGI import ExtractFlowData
from NaFill.FillNaMaster import fill_na
from process_data.Process4Gdata import FilterCGI as filter4g_cgi
from process_data.Process5Gdata import FilterCGI as filter5g_cgi
import threading


def extract_date(mode, ne_type, city, start_dt='2020-07-01'):
    '''
    按照缺失值数量的不同进行分类提取数据，同时对缺失值进行填充。
    可以同时处理4,5g数据的提取，城市可以指定为江苏省任意城市。
    如果已经有过往数据，支持在过去数据基础之上新增数据。
    :param mode: 取值范围: 00_00 (无缺失值), 01_10 (缺失值1-10), 10_20 (缺失值10-20), 20_30 (缺失值20-30)
    :param ne_type: 4G  /   5G
    :param city: 样例；南京
    :return: 无返回值，最后结果存为csv文件
    '''
    # 如果已经跑过，则跳过：
    print('城市: ', city, ' 网络类型: ', ne_type, ' 缺失值范围: ', mode)
    fold_name = os.listdir('./' + ne_type)
    fold_name.sort()
    if os.path.exists(
            './' + ne_type + 'processed_data/missing_value/fillna_example' + mode + '_' + city + '.csv'):
        data = pd.read_csv(
            './' + ne_type + 'processed_data/missing_value/fillna_example' + mode + '_' + city + '.csv')
        data.sort_values('时间', inplace=True)
        last_date = data.iloc[-1]['时间']
        if fold_name[-1] == last_date[:7]:
            print('已处理过历史数据，跳过！')
            return

    null0_10 = pd.read_csv('./' + ne_type + 'processed_data/missing_value/缺失值' + mode + '.csv')
    conn_45g = pd.read_csv('./4Gprocessed_data/45G-0424_000000_tmp_hjj1_141_20220424.csv',
                           encoding='gbk')
    conn_45g.drop(['CREATE_TIME', 'CITY_ID', 'VENDOR_ID', 'VENDOR_NAME', 'CELL_NAME', 'BEARING',
                   'LONGITUDE', 'LATITUDE', 'STATION_NAME', 'STATION_LON', 'STATION_LAT',
                   'SECTOR_ID'], axis=1, inplace=True)
    conn_45g = conn_45g[conn_45g['CITY_NAME'] == city]
    null0_10 = pd.merge(null0_10, conn_45g, left_on='cgi', right_on='CGI')
    null0_10.set_index('STATION_ID', inplace=True)
    conn_45g.drop(['CITY_NAME', 'NE_TYPE'], axis=1, inplace=True)
    cgi_list = null0_10.cgi.unique().tolist()
    print('使用的cgi数量: ', len(cgi_list))
    print('使用的基站数量: ', len(null0_10.index.unique()))
    df = ExtractFlowData(start_date=start_dt, path='./' + ne_type, cgi_list=cgi_list)
    print(df)
    df = pd.merge(conn_45g, df, left_on='CGI', right_index=True)
    df.set_index('CGI', inplace=True)
    df.drop_duplicates(inplace=True)

    df.sort_values('时间', inplace=True)

    # 如果缺失值少的小区通过station id关联到缺失值大的，则剔除关联到的缺失值大的小区。
    for cgi in df.index.unique():
        if df['download'].loc[cgi].isnull().sum() > int(mode[-2:]):
            df.drop(cgi, axis=0, inplace=True)
            print(cgi, '缺失值过多，剔除！')

    # 填补缺失值
    df = fill_na(df=df)
    df.to_csv(
        './' + ne_type + 'processed_data/missing_value/fillna_example' + mode + '_' + city + '.csv')


def filter_na(ne_type):
    '''
    按照缺失值数量划分数据集
    :param ne_type: 可选：4G / 5G
    :return:
    '''
    data = pd.read_csv(
        './' + ne_type[0] + 'Gprocessed_data/missing_value/null_ct_df_' + ne_type[0] + 'g.csv')
    data.set_index('cgi', inplace=True)
    if ne_type == '4G':
        quit_4g = pd.read_csv('./data/4G_退网_20220613.csv', encoding='gbk')
        not_include = set(data.index) - (set(data.index) & set(quit_4g.ECGI))
        data = data.loc[not_include]
        data = data[data['start_date'] < '2021-01-01']
    else:
        data = data[data['start_date'] < '2021-04-01']

    data_0 = data[data['null_count'] == 0]
    data_0_10 = data[data['null_count'] > 1][data['null_count'] < 10]
    data_10_20 = data[data['null_count'] >= 10][data['null_count'] < 20]
    data_20_30 = data[data['null_count'] >= 20][data['null_count'] < 30]
    print('4g数据缺失量统计:')
    print('选择缺失值为0的数据占总数据：',
          len(data[data['null_count'] == 0].index.unique()) / len(data.index.unique()))
    print('选择缺失值<10的数据占总数据：',
          len(data[data['null_count'] < 10].index.unique()) / len(data.index.unique()))
    print('选择缺失值<20的数据占总数据：',
          len(data[data['null_count'] < 20].index.unique()) / len(data.index.unique()))
    print('选择缺失值<30的数据占总数据：',
          len(data[data['null_count'] < 30].index.unique()) / len(data.index.unique()))
    print('-------------------------------------------------------------------')
    data_0.to_csv('./' + ne_type + 'processed_data/missing_value/缺失值00_00.csv')
    data_0_10.to_csv('./' + ne_type + 'processed_data/missing_value/缺失值01_10.csv')
    data_10_20.to_csv('./' + ne_type + 'processed_data/missing_value/缺失值10_20.csv')
    data_20_30.to_csv('./' + ne_type + 'processed_data/missing_value/缺失值20_30.csv')


if __name__ == '__main__':
    '''
    提取数据注意事项:
    
    目前读取数据的逻辑：
    每次重新训练之前都要重新获取一遍全部数据，这么做是因为我们选择的是固定日期之后的数据进行训练，随着时间进展之前提取的数据会有部分超出时间范围，同时其中也会有部分cgi
    退网，同时也会有部分cgi的数据量达到了可训练的标准，如果直接在以前提取好的数据上进行新增，提取数据和筛选数据的逻辑太过复杂，所以决定每次都重新提取数据。
    
    整个提取数据的过程中需要根据时间变化的文件，需要有对应的脚本来在每次训练前更新这些文件到最新的信息：
    退网表：./data/4G_退网_20220613.csv， 
    流量数据：./4G   和   ./5G
    4-5g基站关联表：./4Gprocessed_data/45G-0424_000000_tmp_hjj1_141_20220424.csv
    
    同时流量数据文件夹命名有规范，月数据需要以： 年份-月份作为文件夹名称。
    文件名只支持：
    4G: 1. 4G-年份-月份-日_XXXXXXXXX.csv.gz 或者 2. 年份-月份日_XXXXXXXXXX.csv.gz
    5G: 1. 5G-年份-月份-日_XXXXXXXXX.csv.gz 或者 2. 年份-月份-日_XXXXXXXXXX.csv.gz
    '''
    # 将输出结果输出到日志文件
    f_handler = open('./result/out_NaExtractAndFill.log', 'w')
    __console__ = sys.stdout
    sys.stdout = f_handler

    start_time = time.time()
    # 获取当前时间
    current_dt = datetime.datetime.now()
    day_disparity = datetime.timedelta(days=history_day)
    start_dt = current_dt - day_disparity
    start_dt = start_dt.strftime('%Y-%m-%d')
    print('============================================')
    print('使用往期 ', history_day, ' 日的数据进行训练')
    print('开始数据日期为: ', start_dt)
    print('============================================')
    # 获取到开始训练时间

    # # 使用多线程统计数据缺失之情况
    t1 = threading.Thread(target=filter4g_cgi, args=('./4G', start_dt))
    t2 = threading.Thread(target=filter5g_cgi, args=('./5G', start_dt))

    t1.start()
    t2.start()

    t1.join()
    t2.join()
    # 筛选缺失值少的cgi

    filter_na('4G')
    filter_na('5G')

    # （使用多线程）根据筛选的cgi提取数据，并和4-5g station做关联
    mode_list = ['20_30', '10_20', '01_10', '00_00']
    ne_type_list = ['5G', '4G']
    # 获取江苏所有城市名称
    data = pd.read_csv('./4Gprocessed_data/45G-0424_000000_tmp_hjj1_141_20220424.csv',
                       encoding='gbk')
    city_list = data.CITY_NAME.unique()

    for city in city_list:
        for ne_type in ne_type_list:
            threading_list = []

            for mode in mode_list:
                t = threading.Thread(target=extract_date, args=(mode, ne_type, city, start_dt))
                threading_list.append(t)
                t.start()

            for thread in threading_list:
                thread.join()

            gc.collect()

    end_time = time.time()
    print('处理完所有数据共耗时: ', end_time - start_time)
