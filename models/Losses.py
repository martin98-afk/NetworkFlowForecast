from config import *

# 平均值偏差tensor版
def new_mape(x, y):
    mean_x = tf.reduce_mean(x)
    mean_y = tf.reduce_mean(y)
    return tf.abs((mean_x - mean_y) / (tf.sqrt(mean_x * mean_y))) * 100
# 平均值偏差numpy版
def new_mape_numpy(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    return np.abs((mean_x - mean_y) / (np.sqrt(mean_x * mean_y))) * 100