import matplotlib.pyplot as plt
import pandas as pd
import os
from models.Losses import new_mape_numpy
from config import *
#
path = '.'+ result_path
files = os.listdir(path)

mape_list = []
for file_name in files:
    split_name = file_name.split('_')
    if len(split_name) == 3:
        file_name = file_name

        data = pd.read_csv(path + '/' + file_name)
        true = data['true'].values
        predict = data['predict'].values
        
        data.plot()
        plt.title(file_name)
        plt.show()

        # data.to_csv(path + '/' + 'test_' + file_name[6:])
        # mape_list.append([split_name[0], split_name[1], split_name[2],split_name[3][:-4],
        #                   new_mape_numpy(true,predict)])

# data = pd.read_csv('../5Gprocessed_data/5G全量数据.csv', index_col='cgi')
# cgi_list = data.index.unique()
# flow_data = []
# for i, cgi in enumerate(cgi_list[:400]):
#     flow_data.append(data.loc[cgi].values[:400])
# flow_data = np.array(flow_data)
# plt.imshow(flow_data)