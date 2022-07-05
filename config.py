# File system manangement
import os
import sys
import gc

# numpy and pandas for data manipulation
import numpy as np
import pandas as pd

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# import timecount packages
import time
import datetime

# 使用多gpu
# import resource
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  ##表示使用GPU编号为0的GPU进行计算
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, RepeatVector, TimeDistributed, Activation
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate, Reshape, Lambda
from tensorflow.keras.losses import mape
from tensorflow.keras.utils import plot_model

# 控制图像字体大小
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=20)

font1 = {'family': 'SimHei',
         'weight': 'normal',
         'size': 20,
         }

# 创建结果存放文件夹
os.makedirs('./result', exist_ok=True)
# 连接数据库
config = {'host': '192.168.9.161',
          'port': 3306,
          'user': 'xwtech',
          'passwd': 'hwfx1234',
          'db': 'xwtech',
          'charset': 'utf8',
          'local_infile': 1
          }

# 模型部分超参数
# 滞后天数
input_len = 60
# 预测天数
output_len = 150
# 使用模型
select_model = 'transformer'
# 使用历史多长时间的数据进行训练
history_day = 700
# 使用城市名，暂时废弃，现用多线程一次性跑全部城市，但是无法删除，有部分程序使用到
city_name = '常州'
# 结果存放地址
result_path = './result_region_' + select_model + '_Inp' + str(input_len) + '_Out' + str(output_len)
