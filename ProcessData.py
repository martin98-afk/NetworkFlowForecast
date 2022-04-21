from config import *


# 使用滑窗构建多对多lstm数据集
def series_to_supervised(data, seq_len=30, pred_len=180, label_len=15):
    X = data[:-pred_len, :]
    Y = data[(seq_len - label_len):, -1]  # .reshape(-1,1)
    inputs, targets = None, None
    if X.shape[0] != 0:
        input_dataset = timeseries_dataset_from_array(
            X, None, sequence_length=seq_len, sequence_stride=1, batch_size=X.shape[0])
        target_dataset = timeseries_dataset_from_array(
            Y, None, sequence_length=(label_len + pred_len), sequence_stride=1,
            batch_size=Y.shape[0])

        input = []
        for batch in zip(input_dataset, target_dataset):
            inputs, targets = batch
            # print("input: ",inputs.shape," \noutput: ", targets.shape)
            break

    return inputs, targets


# 划分数据
def split_data(data, cgi, num_in, num_out):
    raw_data = data.loc[cgi].values
    inputs, targets = series_to_supervised(raw_data, num_in, num_out)
    return inputs, targets


# 处理数据
def process_data(processed_df, split_rate=0.8, input_len=30, output_len=180):
    '''
    滑窗划分训练集验证集，将一类小区都按照一定比例进行划分
    :param processed_df:
    :param split_rate:
    :param input_len:
    :param output_len:
    :return:
    '''

    inputs, targets = split_data(processed_df, processed_df.index[0], input_len, output_len)
    train_inputs = inputs[:int(split_rate * inputs.shape[0]), :, :]
    train_targets = targets[:int(split_rate * inputs.shape[0]), :]
    val_inputs = inputs[int(split_rate * inputs.shape[0]):, :, :]
    val_targets = targets[int(split_rate * inputs.shape[0]):, :]
    test_inputs = inputs[-1:, :, :]
    test_targets = targets[-1:, :]
    train_sample_inputs = inputs[:1, :, :]
    train_sample_targets = targets[:1, :]
    for i, cgi in enumerate(processed_df.index.unique()):
        if cgi == processed_df.index[0]:
            continue
        inputs, targets = split_data(processed_df, cgi, input_len, output_len)
        train_inputs = tf.concat([inputs[:int(split_rate * inputs.shape[0]), :, :], train_inputs],
                                 axis=0)
        train_targets = tf.concat([targets[:int(split_rate * inputs.shape[0]), :], train_targets],
                                  axis=0)
        val_inputs = tf.concat([inputs[int(split_rate * inputs.shape[0]):, :, :], val_inputs],
                               axis=0)
        val_targets = tf.concat([targets[int(split_rate * inputs.shape[0]):, :], val_targets],
                                axis=0)
        test_inputs = tf.concat([inputs[-1:, :, :], test_inputs], axis=0)
        test_targets = tf.concat([targets[-1:, :], test_targets], axis=0)
        train_sample_inputs = tf.concat([inputs[:1, :, :], train_sample_inputs], axis=0)
        train_sample_targets = tf.concat([targets[:1, :], train_sample_targets], axis=0)

    return train_inputs, train_targets, val_inputs, val_targets, \
           test_inputs, test_targets, \
           train_sample_inputs, train_sample_targets, \
           processed_df.index.unique().to_list()





def split_region(processed_df1, df_dummy, input_len, output_len):
    '''
    按照区县提取数据，并且将小区数据和基站工参做join
    :param region_name:
    :param processed_df1:
    :param df_dummy:
    :return:
    '''
    df_data = df_dummy.drop(['地点'], axis=1)
    processed_df = pd.merge(df_data, processed_df1, how='inner',
                            left_index=True, right_index=True)
    print(processed_df)
    # 数据标准化

    columns = processed_df.columns
    uint8_cols = processed_df.select_dtypes('uint8').columns.to_list()
    scalers = {}
    for i, col in enumerate(columns):
        if col != 'data_is_workday_True' and col != 'on_holiday_True' and col != 'download' and \
                col not in uint8_cols:
            scaler = MinMaxScaler()
            processed_df[col] = scaler.fit_transform(processed_df[col].values.reshape(-1, 1))
            scalers[col] = scaler

    # 滑窗划分训练验证集
    return process_data(processed_df, input_len=input_len, output_len=output_len), scalers


if __name__ == '__main__':
    ...