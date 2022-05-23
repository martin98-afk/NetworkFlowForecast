from config import *
import ProcessData as processer
import SaveResult as save
import models.LSTM as lstm
import models.Transformer as transformer
import models.DidpGRU as gru
import models.ConvolutionNetwork as convnet
import models.TransformerWithConv as transformer_conv
from evaluation.CalculateAcc import cal_single_acc
import models.ConvTransformer as conv_transformer
import models.Losses as loss
import models.TransformerScratch as trans_scratch

os.makedirs('./saved_model', exist_ok=True)


def predict_multi(model, input1, input2):
    predict = model.predict({'previous info': input1,
                             'predict info': input2})
    predict[predict < 0] = 0
    return predict


def get_saved_model():
    '''
    获取保存的模型
    :return:
    '''
    model_list = []
    for string in os.listdir('./saved_model'):
        position = [item for item in string.split('_')[:-3]]
        model_list.append('_'.join(position))
    return model_list


def train(region_name, raw_data,
          time_label,
          base_station_info,
          select_model='transformer',
          model=None,
          return_model=False):
    '''

    :param region_name: 区域信息
    :param raw_data: 流量数据
    :param base_station_info: 基站工参信息表
    :param select_model: 可选模型: transformer, lstm, lstmdeeper, lstmmulti
    :param model: 传递预训练的模型，默认为None
    :param return_model: 是否返回训练的模型
    :return:
    '''

    ####数据集划分#####################################
    (train_inputs, train_targets, train_predicts,
     val_inputs, val_targets, val_predicts,
     test_inputs, test_targets, test_predicts,
     train_sample_inputs, train_sample_targets, train_sample_predicts,
     cgi_list), scalers = \
        processer.split_region(raw_data, base_station_info,
                               input_len=input_len,
                               output_len=output_len)

    print('测试数据形状： ', test_inputs.shape)
    print('训练数据形状： ', train_inputs.shape)
    print('训练预测辅助信息形状： ', val_predicts.shape)
    # 选择需要训练的模型
    ##########根据选择的模型进行加载#####################################
    if model == None:
        epochs = 500
        if select_model == 'lstm':
            lstm_model = lstm.LSTMTime2Vec(train_inputs, train_targets)
            lstm_model.build_model()
            model = lstm_model.model
            if region_name in get_saved_model():

                try:
                    model.load_weights(
                        './saved_model/' + region_name + '_' + select_model + '_InpLen' +
                        str(input_len) + '_OutLen' + str(output_len))
                    print('读取到已有模型:  ', region_name, '--------------------------')
                    epochs = 30
                except:
                    ...
        elif select_model == 'transformerscratch':
            model = trans_scratch.Transformer(
                num_layers=2, d_model=68, d_model2=68, num_heads=4, dff=64,
                input_seq_len=60, target_seq_len=180
            )
            model.compile(loss='mse', optimizer='adam', metrics=[loss.new_mape])
            model([train_inputs, train_predicts], training=False)

        elif select_model == 'lstmmulti':
            lstm_model = lstm.LSTMTime2VecMultiInput(train_inputs, train_targets, train_predicts)
            lstm_model.build_model()
            model = lstm_model.model

            if region_name in get_saved_model():
                try:
                    model.load_weights(
                        './saved_model/' + region_name + '_' + select_model + '_InpLen' +
                        str(input_len) + '_OutLen' + str(output_len))
                    print('读取到已有模型:  ', region_name, '--------------------------')
                    epochs = 30
                except:
                    ...
        elif select_model == 'convtrans':
            conv_trans_model = conv_transformer.ConvTransformer(8, train_inputs.shape[1],
                                                                train_inputs.shape[2],
                                                                train_targets.shape[1])
            model = conv_trans_model
            model.compile(loss=keras.losses.Huber(),
                          optimizer=keras.optimizers.Adam(learning_rate=0.00001),
                          metrics=[loss.new_mape])

            if region_name in get_saved_model():
                try:
                    model.load_weights(
                        './saved_model/' + region_name + '_' + select_model + '_InpLen' +
                        str(input_len) + '_OutLen' + str(output_len))
                    print('读取到已有模型:  ', region_name, '--------------------------')
                    epochs = 30
                except:
                    ...
        elif select_model == 'lstmdeeper':
            lstm_model = lstm.LSTMTime2VecDeeper(train_inputs, train_targets, train_predicts)
            lstm_model.build_model()
            model = lstm_model.model
            if region_name in get_saved_model():
                try:
                    model.load_weights(
                        './saved_model/' + region_name + '_' + select_model + '_InpLen' +
                        str(input_len) + '_OutLen' + str(output_len))
                    print('读取到已有模型:  ', region_name, '--------------------------')
                    epochs = 30
                except:
                    ...

        elif select_model == 'transformer':
            transformer_model = transformer.Transformer(train_inputs, train_targets, train_predicts)
            transformer_model.build_model()
            model = transformer_model.model
            if region_name in get_saved_model():
                try:
                    model.load_weights(
                        './saved_model/' + region_name + '_' + select_model + '_InpLen' +
                        str(input_len) + '_OutLen' + str(output_len))
                    print('读取到已有模型:  ', region_name, '--------------------------')
                    epochs = 30
                except:
                    ...
        elif select_model == 'GRU':
            gru_model = gru.RNN(train_inputs.shape[1:], train_targets.shape[1])
            gru_model.build_model()
            model = gru_model.model
            if region_name in get_saved_model():
                try:
                    model.load_weights(
                        './saved_model/' + region_name + '_' + select_model + '_InpLen' +
                        str(input_len) + '_OutLen' + str(output_len))
                    print('读取到已有模型:  ', region_name, '--------------------------')
                    epochs = 30
                except:
                    ...
    else:
        epochs = 100

    # plot_model(model, to_file='./figures/model.png')
    print(model.summary())
    print('正在训练：', region_name, '的小区数据')
    print('======================================================================')
    for i in range(epochs):
        if select_model == 'lstmmulti' or select_model == 'transformer':
            history = model.fit({'previous info': train_inputs, 'predict info': train_predicts},
                                train_targets, validation_data=(
                    {'previous info': test_inputs, 'predict info': test_predicts}, test_targets),
                                batch_size=32, epochs=2, verbose=1, shuffle=True)
        else:
            history = model.fit(train_inputs, train_targets,
                                validation_data=(val_inputs, val_targets),
                                batch_size=32, epochs=2, verbose=1, shuffle=True)
        print('训练总轮数：', (i + 1) * 2)
        if i % 5 == 0:

            # 画训练集拟合效果
            select_index = np.argmax(np.random.random(train_inputs.shape[0]))
            plot_val_id = np.argmax(np.random.random(test_inputs.shape[0]))
            if select_model == 'lstmmulti' or select_model == 'transformer':
                train_predict = predict_multi(model,
                                              train_inputs[select_index:select_index + 1, :, :],
                                              train_predicts[select_index:select_index + 1, :, :])

                predictions = predict_multi(model, test_inputs, test_predicts)
                val_predict = predictions[plot_val_id:plot_val_id + 1, :]
                acc = cal_single_acc(test_targets, predictions)
                print('偏差率小于20%的小区数量占比: ', acc, '%')
            # elif select_model == 'transformer':
            #     train_predict = model.predict([train_inputs[select_index:select_index + 1, :, :],
            #                                    train_predicts[select_index:select_index + 1, :, :]])
            #     predictions = transformer_model.predict(model, test_inputs, test_predicts)
            #     val_predict = predictions[plot_val_id:plot_val_id + 1, :]
            #     acc = cal_single_acc(test_targets, predictions)
            #     print('偏差率小于20%的小区数量占比: ', acc, '%')
            else:
                train_predict = model.predict(train_inputs[select_index:select_index + 1, :, :])
                val_predict = model.predict(val_inputs[plot_val_id:plot_val_id + 1, :, :])


            fig, axes = plt.subplots(1, 2, figsize=(20,10))
            axes[0].plot(train_predict.squeeze().T, label='predict')
            axes[0].plot(train_targets[select_index:select_index + 1, :].numpy().T, label='truth')
            axes[0].fill_between(np.arange(train_targets.shape[1]),
                             train_targets[select_index, :].numpy().T * 0.8,
                             train_targets[select_index, :].numpy().T * 1.2, color='blue',
                             alpha=0.25)
            axes[0].legend()
            axes[0].set_title(region_name + ': ' + select_model + 'model, lag days:' + str(input_len)
                      + ' train_result')

            # 画验证集拟合效果
            axes[1].plot(val_predict.squeeze().T, label='predict')
            axes[1].plot(test_targets[plot_val_id:plot_val_id + 1, :].numpy().T,
                     label='truth')
            axes[1].fill_between(np.arange(train_targets.shape[1]),
                             test_targets[plot_val_id, :].numpy().T * 0.8,
                             test_targets[plot_val_id, :].numpy().T * 1.2, color='blue', alpha=0.25)
            axes[1].legend()
            axes[1].set_title(region_name + ': ' + select_model + ' model, lag days:' + str(
                input_len)
                      + ' ' + 'val_result')
            plt.savefig('./figures/val_TrainBeforeSum.png')
            plt.show()

    print('======================================================================')
    print('训练结束，正在保存结果。')
    model.save_weights('./saved_model/' + region_name + '_' + select_model + '_InpLen' +
                       str(input_len) + '_OutLen' + str(output_len))
    print('模型:  ', region_name, '  已保存！！！！！')
    if select_model == 'lstmmulti' or select_model == 'transformer':
        save.save_result_multi(model, region_name, cgi_list, time_label,
                               train_sample_inputs, train_sample_targets, train_sample_predicts,
                               test_inputs, test_targets, test_predicts, scalers)
    else:
        save.save_result(model, region_name, cgi_list, time_label,
                         train_sample_inputs, train_sample_targets, test_inputs, test_targets,
                         scalers)

    if return_model:
        return model


if __name__ == '__main__':
    print(get_saved_model())
