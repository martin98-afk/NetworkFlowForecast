import pandas as pd

data = pd.read_csv('../4Gprocessed_data/missing_value/null_ct_df_4g.csv')
data.set_index('cgi', inplace=True)
quit_4g = pd.read_csv('../data/4G_退网_20220613.csv', encoding='gbk')
not_include = set(data.index) - (set(data.index) & set(quit_4g.ECGI))
data = data.loc[not_include]

data = data[data['start_date'] < '2021-01-01']
data_0_10 = data[data['null_count'] > 1][data['null_count'] < 10]
data_10_20 = data[data['null_count'] >= 10][data['null_count'] < 20]
data_20_30 = data[data['null_count'] >= 20][data['null_count'] < 30]
print('选择缺失值为0的数据占总数据：', len(data[data['null_count'] == 0].index) / len(data.index))
print('选择缺失值<10的数据占总数据：', len(data[data['null_count'] < 10].index) / len(data.index))
print('选择缺失值<20的数据占总数据：', len(data[data['null_count'] < 20].index) / len(data.index))
print('选择缺失值<30的数据占总数据：', len(data[data['null_count'] < 30].index) / len(data.index))

data_0_10.to_csv('../4Gprocessed_data/缺失值01_10.csv')
data_10_20.to_csv('../4Gprocessed_data/缺失值10_20.csv')
data_20_30.to_csv('../4Gprocessed_data/缺失值20_30.csv')

data = pd.read_csv('../5Gprocessed_data/missing_value/null_ct_df_5g.csv')
data.set_index('cgi', inplace=True)
data = data[data['start_date'] < '2021-04-01']

data_0_10 = data[data['null_count'] > 1][data['null_count'] < 10]
data_10_20 = data[data['null_count'] >= 10][data['null_count'] < 20]
data_20_30 = data[data['null_count'] >= 20][data['null_count'] < 30]
print('选择缺失值为0的数据占总数据：', len(data[data['null_count'] == 0].index) / len(data.index))
print('选择缺失值<10的数据占总数据：', len(data[data['null_count'] < 10].index) / len(data.index))
print('选择缺失值<20的数据占总数据：', len(data[data['null_count'] < 20].index) / len(data.index))
print('选择缺失值<30的数据占总数据：', len(data[data['null_count'] < 30].index) / len(data.index))

data_0_10.to_csv('../5Gprocessed_data/缺失值01_10.csv')
data_10_20.to_csv('../5Gprocessed_data/缺失值10_20.csv')
data_20_30.to_csv('../5Gprocessed_data/缺失值20_30.csv')
