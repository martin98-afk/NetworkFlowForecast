import pandas as pd

from config import *

flow_data = pd.read_csv('../5Gprocessed_data/5G全量数据.csv', index_col='cgi')
gongcan = pd.read_csv('../5Gprocessed_data/处理后5G工参.csv', index_col='CGI')

gongcan = gongcan[gongcan['城市'] == '扬州']
flow_data = flow_data.loc[set(gongcan.index) & set(flow_data.index)]
flow_data.to_csv('../5Gprocessed_data/5G_扬州.csv')
