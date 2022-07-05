import pandas as pd

from config import *


def ExtractFlowData(start_date='2020-07-01',
                    path='../4G',
                    cgi_list=['460-00-10486907-2', '460-00-10493835-8', '460-00-10498652-4',
                              '460-00-10503120-2', '460-00-10513976-1', '460-00-10503120-2']):
    '''
    提取小区流量数据，
    :param path: 全量数据路径
    :return: None
    '''
    path1 = '../5Gprocessed_data/'
    # 提取没有缺失值的小区cgi

    dataframe = pd.DataFrame(columns=['cgi', '时间', 'download'])
    dataframe.set_index('cgi', inplace=True)
    fold_name = os.listdir(path)
    fold_name.sort()
    for name in fold_name:
        file_list = os.listdir(path + '/' + name)
        file_list.sort()
        for file_name in file_list:
            if path[-2:] == '5G':
                date = file_name.replace('5G-', '')[:10]
            else:
                if file_name[:2] == '4G':
                    date = file_name[3:13]
                else:
                    date = file_name[:7] + '-' + file_name[7:9]
            if date <= start_date:
                continue
            if file_name[-2:] == 'gz':

                print('正在处理', date, '日期的数据')
                data = pd.read_csv(path + '/' + name + '/' + file_name, compression='gzip',
                                   encoding='gbk', error_bad_lines=False)
                data.set_index('cgi', inplace=True)

                same_cgi = set(cgi_list) & set(data.index)
                not_in_cgi = set(cgi_list) - same_cgi

                for cgi in not_in_cgi:
                    filter_data = np.array([cgi, date, None]).reshape(1, -1)
                    df = pd.DataFrame(filter_data, columns=['cgi', '时间', 'download'])
                    df.set_index('cgi', inplace=True)
                    dataframe = pd.concat([dataframe, df])

                filter_data = data.loc[same_cgi]

                if path[-2:] == '5G':
                    filter_data['download'] = filter_data[['下行流量(TB)', '上行流量(TB)']] \
                        .apply(lambda x: x.sum(), axis=1)
                    filter_data = filter_data[['download']]
                else:
                    try:
                        filter_data['download'] = filter_data[['下行流量', '上行流量']] \
                            .apply(lambda x: x.sum(), axis=1)
                        filter_data = filter_data[['download']]
                    except:
                        filter_data['download'] = filter_data[
                            ['PDCP_UpOctDl', 'PDCP_UpOctUl']].apply(
                            lambda x: x.sum(), axis=1)
                        filter_data = filter_data[['download']]

                # 如果遇到有cgi一天有多个数据，取所有download的最大值
                avg_download = pd.DataFrame(filter_data.groupby('cgi')['download'].max())
                avg_download['时间'] = np.array([date] * len(avg_download.index))
                dataframe = pd.concat([dataframe, avg_download])

    return dataframe


if __name__ == '__main__':

    ne_type = '5G'
    data = pd.read_csv('../5Gprocessed_data/missing_value/null_ct_df.csv')
    data.fillna('_', inplace=True)
    data['null_date'] = data['null_date'].apply(lambda x: x[1:].split("_"))
    print(data)
    join_45g = pd.read_csv('../4Gprocessed_data/45G-0424_000000_tmp_hjj1_141_20220424.csv',
                           encoding='gbk')
    join_45g['STATION_ID'] = join_45g['STATION_ID'].apply(lambda x: '_'.join(x.split('_')[:2]))
    join_45g = join_45g[['CGI', 'STATION_ID', 'CITY_NAME', 'CELL_NAME', 'LONGITUDE', 'LATITUDE']]
    data = pd.merge(data, join_45g, left_on='cgi', right_on='CGI')

    explode_data = data.explode('null_date')
    count = pd.DataFrame(explode_data.groupby(['STATION_ID', 'null_date'])['cgi'].count())

    count.reset_index(inplace=True)
    print(count)

    variance = pd.DataFrame(count.groupby('STATION_ID')['cgi'].var()).sort_values('cgi',
                                                                                  ascending=False)
    first_station = variance.index.tolist()[0]
    select_cgi = data[data['STATION_ID'] == first_station].cgi.tolist()
    print(data[data['STATION_ID'] == first_station])
    df = ExtractFlowData(path='../' + ne_type, cgi_list=list(select_cgi))
    df.to_csv(first_station + '.csv')
