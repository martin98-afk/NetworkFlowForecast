# numpy and pandas for data manipulation
import numpy as np
import pandas as pd
# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
# File system manangement
import os
# Suppress warnings
import warnings

warnings.filterwarnings('ignore')
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
from sklearn import metrics
import time
import seaborn as sns
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import tensorflow.keras as keras
# 使用多gpu
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  ##表示使用GPU编号为0的GPU进行计算
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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

font1 = {'family': 'SimHei',
         'weight': 'normal',
         'size': 20,
         }

mode = '4g'
validate = False
label_len = 0
input_len = 60
output_len = 180
history_day = 200

select_model = 'transformer'
result_path = './result_region_' + select_model + '_Inp' + str(input_len) + '_Out' + str(output_len)
