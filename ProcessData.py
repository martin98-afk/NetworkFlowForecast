from config import *
import datetime

# 使用滑窗构建多对多lstm数据集
def series_to_supervised(data, time_label, seq_len=30, pred_len=180):
    '''
    使用tf.keras的滑窗工具包划分数据集
    :param data:
    :param seq_len:
    :param pred_len:
    :param label_len:
    :return:
    '''
    X = data[:-pred_len, :]
    Y = data[(seq_len):, -1]  # .reshape(-1,1)
    Z = data[(seq_len):, -8:-1]
    time_label = time_label[seq_len:]
    inputs, targets, predicts = None, None, None
    time_dataset = []
    if X.shape[0] != 0:
        input_dataset = timeseries_dataset_from_array(
            X, None, sequence_length=seq_len, sequence_stride=1, batch_size=X.shape[0])
        target_dataset = timeseries_dataset_from_array(
            Y, None, sequence_length=pred_len, sequence_stride=1,
            batch_size=Y.shape[0])
        predict_dataset = timeseries_dataset_from_array(
            Z, None, sequence_length=pred_len, sequence_stride=1,
            batch_size=Y.shape[0])

        for i in range(Y.shape[0] - pred_len + 1):
            time_dataset.append(time_label[i:i + pred_len])

        for batch in zip(input_dataset, target_dataset, predict_dataset):
            inputs, targets, predicts = batch
            # print("input: ",inputs.shape," \n output: ", targets.shape)
            break

    return inputs, targets, predicts, time_dataset


# 划分真实数据
def split_data(data, cgi, time_label, num_in, num_out):
    '''
    所有数据一起划分，并做滑窗
    :param data:
    :param cgi:
    :param num_in:
    :param num_out:
    :return:
    '''
    raw_data = data.loc[cgi].values
    inputs, targets, predicts, time = series_to_supervised(raw_data, time_label, num_in, num_out)
    return inputs, targets, predicts, time


# 处理数据
def process_data(processed_df, time_label, process_mode, scalers, split_rate=0.8, input_len=30, \
                                                                                 output_len=180):
    '''
    滑窗划分训练集验证集，将一类小区都按照一定比例进行划分
    如果validate为true则生成数据集时考虑生成验证集，false时只有训练和测试集
    :param processed_df:
    :param split_rate:
    :param input_len:
    :param output_len:
    :return:
    '''

    inputs, targets, predicts, time = split_data(processed_df, processed_df.index[0],
                                                 time_label, input_len,
                                                 output_len)
    if process_mode == 'validate':
        train_inputs = inputs[:int(split_rate * inputs.shape[0]), ...]
        train_targets = targets[:int(split_rate * inputs.shape[0]), ...]
        train_predicts = predicts[:int(split_rate * inputs.shape[0]), ...]
        train_time_label = time[:int(split_rate * inputs.shape[0])]

        train_sample_inputs = inputs[:1, ...]
        train_sample_targets = targets[:1, ...]
        train_sample_predicts = predicts[:1, ...]
        train_sample_time_label = time[:1]

        test_inputs = inputs[-1:, ...]
        test_targets = targets[-1:, ...]
        test_predicts = predicts[-1:, ...]
        test_time_label = time[-1:]
    else:
        train_inputs = inputs
        train_targets = targets
        train_predicts = predicts
        train_time_label = time

        train_sample_inputs = inputs[:1, ...]
        train_sample_targets = targets[:1, ...]
        train_sample_predicts = predicts[:1, ...]
        train_sample_time_label = time[:1]

        test_inputs = processed_df.iloc[-input_len:].values
        test_targets = None

        cur_date = time_label[-1]
        current_dt = datetime.datetime.strptime(cur_date, "%Y-%m-%d")
        count = 1
        test_time_label = []
        while count <= output_len:
            day_disparity = datetime.timedelta(days=count)
            dt = current_dt + day_disparity
            test_time_label.append(dt.strftime("%Y-%m-%d"))
            count += 1

        # 读取节假日信息表
        # TODO 暂时只有2022年内的日期信息，如果要预测2023年的数据需要对GetDateTable进行改进
        date_df = pd.read_csv('./data/date.csv', index_col='date')

        for key in scalers.keys():
            if key in date_df.columns:
                date_df[key] = scalers[key].transform(date_df[key].values.reshape(-1, 1))

        test_predicts = date_df.loc[test_time_label].values
        test_inputs = test_inputs[tf.newaxis, :]
        test_predicts = test_predicts[tf.newaxis, :]
        test_time_label = [test_time_label]

    return train_inputs, train_targets, train_predicts, train_time_label, \
           test_inputs, test_targets, test_predicts, test_time_label, \
           train_sample_inputs, train_sample_targets, train_sample_predicts, \
           train_sample_time_label, \
           processed_df.index.unique().to_list()


def split_region(processed_df, time_label, input_len=input_len, output_len=output_len,
                 process_mode='predict'):
    '''
    按照区县提取数据，并且将小区数据和基站工参做join
    :param region_name:
    :param processed_df1:
    :param df_dummy:
    :return:
    '''

    print('预处理数据结构: ', processed_df.info())
    # 数据标准化

    columns = processed_df.columns
    uint8_cols = processed_df.select_dtypes('uint8').columns.to_list()
    scalers = {}
    for i, col in enumerate(columns):
        if col != 'data_is_workday_True' and col != 'on_holiday_True' and \
                col != '5g_download' and col != '4g_download' and col != 'download5G' and col not \
                in uint8_cols:
            scaler = MinMaxScaler()
            col_data = processed_df[col].values
            if process_mode == 'validate':
                col_data = col_data[:int((len(col_data) - output_len) * 0.8)]

            scaler.fit(col_data.reshape(-1, 1))
            processed_df[col] = scaler.transform(processed_df[col].values.reshape(-1, 1))
            scalers[col] = scaler
    print(processed_df)
    # 滑窗划分训练验证集
    return process_data(processed_df, time_label, process_mode, scalers, input_len=input_len,
                        output_len=output_len), scalers


if __name__ == '__main__':
    ...
