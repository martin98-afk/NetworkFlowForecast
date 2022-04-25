from config import *


# 平均值偏差tensor版
def new_mape(x, y):
    mean_x = tf.reduce_mean(tf.abs(x))
    mean_y = tf.reduce_mean(tf.abs(y))
    return tf.abs((mean_x - mean_y) / (tf.sqrt(mean_x * mean_y) + 1e-6)) * 100


# 平均值偏差numpy版
def new_mape_numpy(x, y):
    mean_x = np.mean(np.abs(x))
    mean_y = np.mean(np.abs(y))
    return np.abs((mean_x - mean_y) / (np.sqrt(mean_x * mean_y) + 1e-6)) * 100
