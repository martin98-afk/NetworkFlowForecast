import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from models.Losses import new_weekly_mape_numpy, new_monthly_mape_numpy, new_mape_numpy

station_info = pd.read_csv('../4Gprocessed_data/常州区县信息.csv')
station_info['STATION_ID'] = station_info['STATION_ID'].apply(lambda x: '_'.join(x.split('_')[:-1]))
station_info['NE_TYPE'] = station_info['STATION_ID'].apply(lambda x: x[-2:])
result_path = '../result_region_transformer_Inp60_Out150_常州'

print(station_info)
station_info.drop_duplicates()
print(station_info)
result_4g = {}
result_5g = {}
for region in station_info['区县'].unique():
    for file_name in os.listdir(result_path):
        station_id = '_'.join(file_name.split('_')[5:8])
        if station_id in station_info[station_info['区县'] == region]['STATION_ID'].values.tolist() \
                and file_name.split('_')[2] == '4g':
            data = pd.read_csv(os.path.join(result_path, file_name))
            if region in result_4g.keys():
                result_4g[region] += data[['true', 'predict']].values
            else:
                result_4g[region] = data[['true', 'predict']].values
        elif station_id in station_info[station_info['区县'] == region]['STATION_ID'].values.tolist() \
                and file_name.split('_')[2] == '5g':
            data = pd.read_csv(os.path.join(result_path, file_name))
            if region in result_5g.keys():
                result_5g[region] += data[['true', 'predict']].values
            else:
                result_5g[region] = data[['true', 'predict']].values

acc_4g = pd.DataFrame(columns=['区县', 'cgi数量', 'station数量',
                               '整体偏差率/%', '月粒度偏差率/%',
                               '周粒度偏差率/%'])
acc_5g = acc_4g.copy()
acc_all = pd.DataFrame(columns=['区县', '整体偏差率/%', '月粒度偏差率/%',
                                '周粒度偏差率/%'])

sum_4g = np.zeros((150, 2))
sum_5g = np.zeros((150, 2))
sum_all = np.zeros((150, 2))
for region in result_4g.keys():
    week_mape = 100 - new_weekly_mape_numpy(result_4g[region][:, 0], result_4g[region][:, 1])
    month_mape = 100 - new_monthly_mape_numpy(result_4g[region][:, 0], result_4g[region][:, 1])
    mape = 100 - new_mape_numpy(result_4g[region][:, 0], result_4g[region][:, 1])
    sum_4g[:, 0] += result_4g[region][:, 0]
    sum_4g[:, 1] += result_4g[region][:, 1]
    sum_all[:, 0] += result_4g[region][:, 0]
    sum_all[:, 1] += result_4g[region][:, 1]
    cgi_count = len(station_info[station_info['区县'] == region][station_info['NE_TYPE'] ==
                                                               '4G'].CGI.unique())
    station_count = len(station_info[station_info['区县'] == region][station_info['NE_TYPE'] ==
                                                                   '4G'].STATION_ID.unique())
    acc_4g = pd.concat([acc_4g, pd.DataFrame(np.array([region, cgi_count, station_count, mape,
                                                       month_mape,
                                                       week_mape])
                                             .reshape(1, -1),
                                             columns=['区县', 'cgi数量', 'station数量',
                                                      '整体偏差率/%', '月粒度偏差率/%',
                                                      '周粒度偏差率/%'])])

    week_mape = 100 - new_weekly_mape_numpy(result_5g[region][:, 0], result_5g[region][:, 1])
    month_mape = 100 - new_monthly_mape_numpy(result_5g[region][:, 0], result_5g[region][:, 1])
    mape = 100 - new_mape_numpy(result_5g[region][:, 0], result_5g[region][:, 1])
    sum_5g[:, 0] += result_5g[region][:, 0]
    sum_5g[:, 1] += result_5g[region][:, 1]
    sum_all[:, 0] += result_5g[region][:, 0]
    sum_all[:, 1] += result_5g[region][:, 1]
    cgi_count = len(station_info[station_info['区县'] == region][station_info['NE_TYPE'] ==
                                                               '5G'].CGI.unique())
    station_count = len(station_info[station_info['区县'] == region][station_info['NE_TYPE'] ==
                                                                   '5G'].STATION_ID.unique())
    acc_5g = pd.concat([acc_5g, pd.DataFrame(np.array([region, cgi_count, station_count, mape,
                                                       month_mape, week_mape])
                                             .reshape(1, -1),
                                             columns=['区县', 'cgi数量', 'station数量',
                                                      '整体偏差率/%', '月粒度偏差率/%',
                                                      '周粒度偏差率/%'])])

    week_mape = 100 - new_weekly_mape_numpy(result_5g[region][:, 0] + result_4g[region][:, 0],
                                      result_5g[region][:, 1] + result_4g[region][:, 1])
    month_mape = 100 - new_monthly_mape_numpy(result_5g[region][:, 0] + result_4g[region][:, 0],
                                        result_5g[region][:, 1] + result_4g[region][:, 1])
    mape = 100 - new_mape_numpy(result_5g[region][:, 0] + result_4g[region][:, 0],
                          result_5g[region][:, 1] + result_4g[region][:, 1])
    acc_all = pd.concat([acc_all, pd.DataFrame(np.array([region, mape, month_mape, week_mape])
                                             .reshape(1, -1),
                                             columns=['区县', '整体偏差率/%', '月粒度偏差率/%',
                                                      '周粒度偏差率/%'])])

week_mape = new_weekly_mape_numpy(sum_4g[:, 0], sum_4g[:, 1])
month_mape = new_monthly_mape_numpy(sum_4g[:, 0], sum_4g[:, 1])
mape = new_mape_numpy(sum_4g[:, 0], sum_4g[:, 1])
acc_4g = pd.concat([acc_4g, pd.DataFrame(np.array(['总计', None, None, 100 - mape,
                                                   100 - month_mape,
                                                   100 - week_mape])
                                         .reshape(1, -1),
                                         columns=['区县', 'cgi数量', 'station数量',
                                                  '整体偏差率/%', '月粒度偏差率/%',
                                                  '周粒度偏差率/%'])])

week_mape = new_weekly_mape_numpy(sum_5g[:, 0], sum_5g[:, 1])
month_mape = new_monthly_mape_numpy(sum_5g[:, 0], sum_5g[:, 1])
mape = new_mape_numpy(sum_5g[:, 0], sum_5g[:, 1])
acc_5g = pd.concat([acc_5g, pd.DataFrame(np.array(['总计', None, None, 100 - mape,
                                                   100 - month_mape,
                                                   100 - week_mape])
                                         .reshape(1, -1),
                                         columns=['区县', 'cgi数量', 'station数量',
                                                  '整体偏差率/%', '月粒度偏差率/%',
                                                  '周粒度偏差率/%'])])

week_mape = new_weekly_mape_numpy(sum_5g[:, 0], sum_5g[:, 1])
month_mape = new_monthly_mape_numpy(sum_5g[:, 0], sum_5g[:, 1])
mape = new_mape_numpy(sum_5g[:, 0], sum_5g[:, 1])
acc_all = pd.concat([acc_all, pd.DataFrame(np.array(['总计', 100 - mape,
                                                   100 - month_mape,
                                                   100 - week_mape])
                                         .reshape(1, -1),
                                         columns=['区县', '整体偏差率/%', '月粒度偏差率/%',
                                                  '周粒度偏差率/%'])])

acc_4g.to_csv('../result/region_4g_常州.csv', index=False, encoding='utf-8')
acc_5g.to_csv('../result/region_5g_常州.csv', index=False, encoding='utf-8')
acc_all.to_csv('../result/region_all_常州.csv', index=False, encoding='utf-8')
