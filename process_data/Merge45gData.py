from config import *

# loading packages
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import pymysql


def merge_45g(city_name):
    '''
    根据45g基站关联表合并4，5g数据
    :param city_name:
    :return:
    '''
    cgi_45g = pd.read_csv('../4Gprocessed_data/45G-0424_000000_tmp_hjj1_141_20220424.csv', encoding
    ='gbk')

    cgi_45g.CITY_NAME.unique().tolist()

    cgi_45g['STATION_ID'] = cgi_45g['STATION_ID'].apply(lambda x: '_'.join(x.split('_')[:3]))
    print(cgi_45g)

    df_4g = pd.read_csv('../4Gprocessed_data/' + city_name + '.csv', index_col='cgi')
    # df_4g = extract_city_data('flow4g', 'stationInfo_4g', city_name)
    df_4g_sector = pd.merge(cgi_45g, df_4g, how='inner', left_on='CGI', right_index=True)
    print(len(df_4g_sector.CGI.unique()))
    df_4g_sector = df_4g_sector.groupby(['STATION_ID', 'timestamp'])['download'].mean()
    df_4g_sector = pd.DataFrame(df_4g_sector).reset_index()
    df_4g_sector.columns = ['STATION_ID', 'timestamp', '4g_download']
    df_4g_sector['label'] = df_4g_sector['STATION_ID'].apply(lambda x: '_'.join(x.split('_')[:2]))

    df_5g = pd.read_csv('../5Gprocessed_data/' + city_name + '.csv', index_col='cgi')
    # df_5g = extract_city_data('flow5g', 'stationInfo_5g', city_name)

    df_5g_sector = pd.merge(cgi_45g, df_5g, how='inner', left_on='CGI', right_index=True)
    print(len(df_5g_sector.CGI.unique()))
    df_5g_sector = df_5g_sector.groupby(['STATION_ID', 'timestamp'])['download'].mean()
    df_5g_sector = pd.DataFrame(df_5g_sector).reset_index()
    df_5g_sector.columns = ['STATION_ID', 'timestamp', '5g_download']
    df_5g_sector['label'] = df_5g_sector['STATION_ID'].apply(lambda x: '_'.join(x.split('_')[:2]))
    print(df_4g_sector)
    print(df_5g_sector)
    df_combined = pd.merge(df_4g_sector, df_5g_sector, how='inner', on=['label', 'timestamp'])
    df_combined.set_index('label', inplace=True)
    df_combined.to_csv('../5Gprocessed_data/' + city_name + 'SECTOR合并.csv')


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
    df.set_index('cgi', inplace=True)
    return df


def merge_4g(city_name):
    cgi_45g = pd.read_csv('./4Gprocessed_data/45G-0424_000000_tmp_hjj1_141_20220424.csv', encoding
    ='gbk')

    cgi_45g.CITY_NAME.unique().tolist()

    cgi_45g['SECTOR_ID'] = cgi_45g['SECTOR_ID'].apply(lambda x: '_'.join(x.split('_')[:2]))
    print(cgi_45g)

    df_4g = pd.read_csv('./4Gprocessed_data/' + city_name + '.csv', index_col='cgi')
    # df_4g = extract_city_data('flow4g', 'stationInfo_4g', city_name)

    cgi_45g.set_index('SECTOR_ID', inplace=True)
    cgi_45g_ct = cgi_45g.groupby(['SECTOR_ID']).agg({'NE_TYPE': 'nunique'})
    cgi_45g_ct = (cgi_45g_ct == 1)
    single_cgi = cgi_45g_ct[cgi_45g_ct['NE_TYPE'] == True].index
    cgi_45g = cgi_45g.loc[single_cgi]
    print((cgi_45g_ct == 1).index)
    print(len(single_cgi))
    print(cgi_45g['NE_TYPE'].unique())

    df_only_4g_sector = pd.merge(cgi_45g, df_4g, how='inner', left_on='CGI', right_index=True)
    print(len(df_only_4g_sector.CGI.unique()))
    df_only_4g_sector = df_only_4g_sector.groupby(['SECTOR_ID', 'timestamp'])['download'].mean()
    df_only_4g_sector = pd.DataFrame(df_only_4g_sector).reset_index()
    df_only_4g_sector.columns = ['SECTOR_ID', 'timestamp', '4g_download']

    print('只有4g的基站的4g数据')
    print('--------------------------------------')
    print(df_only_4g_sector)
    df_only_4g_sector.to_csv('./4Gprocessed_data/' + city_name + '_单4g.csv')


# 只有4g数据基站提取
if __name__ == '__main__':
    merge_45g('常州')
