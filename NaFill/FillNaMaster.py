import pandas as pd
import matplotlib.pyplot as plt
from NaFill.trmf import *


def get_slice(data, T_train, T_test, T_start, normalize=True):
    N = len(data)
    # split on train and test
    train = data[:, T_start:T_start + T_train].copy()
    test = data[:, T_start + T_train:T_start + T_train + T_test].copy()

    # normalize data
    if normalize:
        mean_train = np.array([])
        std_train = np.array([])
        for i in range(len(train)):
            if (~np.isnan(train[i])).sum() == 0:
                mean_train = np.append(mean_train, 0)
                std_train = np.append(std_train, 0)
            else:
                mean_train = np.append(mean_train, train[i][~np.isnan(train[i])].mean())
                std_train = np.append(std_train, train[i][~np.isnan(train[i])].std())

        std_train[std_train == 0] = 1.

        train -= mean_train.repeat(T_train).reshape(N, T_train)
        train /= std_train.repeat(T_train).reshape(N, T_train)
        test -= mean_train.repeat(T_test).reshape(N, T_test)
        test /= std_train.repeat(T_test).reshape(N, T_test)

    return train, mean_train, std_train, test


def inverse_normalize(data, mean, std):
    return data * std + mean


def fill_na(path='../5Gprocessed_data/missing_value/missing_value_example20-30.csv',
            df=pd.DataFrame()):
    if df.values.shape[0] == 0:
        df = pd.read_csv(path)
    df.reset_index(inplace=True)
    df.sort_values(['CGI', '时间'], inplace=True)
    df.drop_duplicates(inplace=True)
    df.set_index('CGI', inplace=True)
    length = len(df.loc[df.index[0]])
    data = []
    for i, cgi in enumerate(df.index.unique()):
        flow_data = df.loc[cgi]['download'].values
        data.append(flow_data)
    data = np.array(data)
    print(data.shape)
    data, mean_train, std_train, _ = get_slice(data, length, 0, 0, normalize=True)

    plt.figure(figsize=(15, 8))
    plt.plot(data[0], color='blue')

    plt.xlabel('timepoint')
    plt.ylabel('value')

    plt.show()

    model = trmf(lags, K, lambda_f, lambda_x, lambda_w, alpha, eta, max_iter)
    data_missing = data.copy()
    model.fit(data_missing)
    data_imputed = model.impute_missings()
    for i in range(data_imputed.shape[0]):
        data_imputed[i] = inverse_normalize(data_imputed[i], mean_train[i], std_train[i])

    plt.figure(figsize=(15, 8))
    plt.plot(data_imputed[0], color='blue')

    plt.xlabel('timepoint')
    plt.ylabel('value')

    plt.show()

    for i, cgi in enumerate(df.index.unique()):
        df.loc[cgi, 'download'] = data_imputed[i]
    return df


if __name__ == '__main__':
    df = fill_na(path='../5Gprocessed_data/missing_value/missing_value_example20_30_常州.csv')
