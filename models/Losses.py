from config import *


# 平均值偏差tensor版
def new_mape(x, y):
    mean_x = tf.reduce_mean(tf.abs(x))
    mean_y = tf.reduce_mean(tf.abs(y))
    return tf.abs((mean_x - mean_y) / ((mean_x + mean_y) / 2)) * 100


# 平均值偏差numpy版
def new_mape_numpy(x, y):
    mean_x = np.mean(np.abs(x))
    mean_y = np.mean(np.abs(y))
    return np.abs((mean_x - mean_y) / ((mean_x + mean_y) / 2)) * 100


# 平均值偏差numpy版
def new_weekly_mape_numpy(x, y):
    mean_x, mean_y = [], []
    for i in range(int(np.floor(len(x) / 7) - 1)):
        mean_x.append(np.mean(x[i * 7:(i + 1) * 7]))
        mean_y.append(np.mean(y[i * 7:(i + 1) * 7]))
    mean_x.append(np.mean(x[(i + 1) * 7:]))
    mean_y.append(np.mean(y[(i + 1) * 7:]))
    mean_x = np.array(mean_x)
    mean_y = np.array(mean_y)

    return np.mean(np.abs((mean_x - mean_y) / ((mean_x + mean_y) / 2))) * 100


# 平均值偏差numpy版
def new_monthly_mape_numpy(x, y):
    mean_x, mean_y = [], []
    for i in range(int(np.floor(len(x) / 30) - 1)):
        mean_x.append(np.mean(x[i * 30:(i + 1) * 30]))
        mean_y.append(np.mean(y[i * 30:(i + 1) * 30]))
    mean_x.append(np.mean(x[(i + 1) * 30:]))
    mean_y.append(np.mean(y[(i + 1) * 30:]))
    mean_x = np.array(mean_x)
    mean_y = np.array(mean_y)

    return np.mean(np.abs((mean_x - mean_y) / ((mean_x + mean_y) / 2))) * 100

# 平均值偏差numpy版
# def new_weekly_mape(x, y):
#     mean_x, mean_y = [], []
#     for i in range(tf.floor(len(x) / 7) - 1):
#         mean_x.append(tf.reduce_mean(x[int(i * 7):int((i + 1) * 7)]))
#         mean_y.append(tf.reduce_mean(y[int(i * 7):int((i + 1) * 7)]))
#     mean_x.append(tf.reduce_mean(x[int((i + 1) * 7):]))
#     mean_y.append(tf.reduce_mean(y[int((i + 1) * 7):]))
#     # mean_x = tf.convert_to_tensor(np.array(mean_x))
#     # mean_y = tf.convert_to_tensor(np.array(mean_y))
#     return tf.reduce_mean(tf.abs([(x - y) / ((x + y) * 2) for x, y in zip(mean_x, mean_y)])) * 100
#     # return tf.reduce_mean(tf.abs((mean_x - mean_y) / ((mean_x + mean_y) / 2))) * 100
