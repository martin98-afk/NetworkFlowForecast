from config import *
import ProcessData as processer
import SaveResult as save
import models.LSTM as lstm
import models.Transformer as transformer
import models.DidpGRU as gru
import models.ConvolutionNetwork as convnet
import models.TransformerWithConv as transformer_conv


def train(region_name, raw_data, base_station_info, select_model='transformer', model=None, \
          return_model=False):
    '''

    :param region_name: 区域信息
    :param raw_data: 流量数据
    :param base_station_info: 基站工参信息表
    :param select_model: 可选模型: transformer, lstm, lstmdeeper
    :param model: 传递预训练的模型，默认为None
    :param return_model: 是否返回训练的模型
    :return:
    '''

    select_model = select_model.split('_')

    ####数据集划分#####################################
    (train_inputs, train_targets, val_inputs, val_targets, \
     test_inputs, test_targets, \
     train_sample_inputs, train_sample_targets, \
     cgi_list), scalers = \
        processer.split_region(raw_data, base_station_info, input_len=int(select_model[1]), \
                               output_len=180)

    print('测试数据形状： ', test_inputs.shape)
    print('训练数据形状： ', train_inputs.shape)
    # 选择需要训练的模型
    ##########根据选择的模型进行加载#####################################
    if model == None:
        epochs = 50
        if select_model[0] == 'lstm':
            lstm_model = lstm.LSTMTime2Vec(train_inputs, train_targets)
            lstm_model.build_model()
            model = lstm_model.model
            if region_name in [string[:-(10 + len(select_model[0]))] for string in
                               os.listdir('./saved_model')]:

                try:
                    model.load_weights('./saved_model/' + region_name + '_' + select_model[0] + '_' +
                                       select_model[1])
                    print('读取到已有模型:  ', region_name, '--------------------------')
                    epochs = 5
                except:
                    ...
        elif select_model[0] == 'lstmdeeper':
            lstm_model = lstm.LSTMTime2VecDeeper(train_inputs, train_targets)
            lstm_model.build_model()
            model = lstm_model.model
            if region_name in [string[:-(10 + len(select_model))] for string in
                               os.listdir('./saved_model')]:
                try:
                    model.load_weights('./saved_model/' + region_name + '_' + select_model[0] + '_' +
                                       select_model[1])
                    print('读取到已有模型:  ', region_name, '--------------------------')
                    epochs = 5
                except:
                    ...

        elif select_model[0] == 'transformer':
            transformer_model = transformer.Transfomer(train_inputs, train_targets)
            transformer_model.build_model()
            model = transformer_model.model
            if region_name in [string[:-(10 + len(select_model))] for string in
                               os.listdir('./saved_model')]:
                try:
                    model.load_weights('./saved_model/' + region_name + '_' + select_model[0] + '_' +
                                       select_model[1])
                    print('读取到已有模型:  ', region_name, '--------------------------')
                    epochs = 5
                except:
                    ...
        elif select_model[0] == 'GRU':
            gru_model = gru.RNN(train_inputs.shape[1:], train_targets.shape[1])
            gru_model.build_model()
            model = gru_model.model
            if region_name in [string[:-(10 + len(select_model))] for string in
                               os.listdir('./saved_model')]:
                try:
                    model.load_weights('./saved_model/' + region_name + '_' + select_model[0] + '_' +
                                       select_model[1])
                    print('读取到已有模型:  ', region_name, '--------------------------')
                    epochs = 5
                except:
                    ...
    else:
        epochs = 30

    print(model.summary())

    print('正在训练：', region_name, '的小区数据')
    print('======================================================================')
    for i in range(epochs):
        history = model.fit(train_inputs, train_targets, validation_data=(val_inputs, val_targets),
                            batch_size=32, epochs=2, verbose=1, shuffle=True)
        print('训练总轮数：', (i + 1) * 2)
        if i % 5 == 0:
            # 画训练集拟合效果
            select_index = np.argmax(np.random.random(train_inputs.shape[0]))
            plt.plot(model.predict(train_inputs[select_index:select_index + 1, :, :]).squeeze().T,
                     label='predict')
            plt.plot(train_targets[select_index:select_index + 1, :].numpy().T, label='truth')
            plt.fill_between(np.arange(train_targets.shape[1]),
                             train_targets[select_index, :].numpy().T * 0.8,
                             train_targets[select_index, :].numpy().T * 1.2, color='blue',
                             alpha=0.25)
            plt.legend()
            plt.title(region_name + ': ' + select_model[0] + 'model, lag days:' + select_model[1]
                      + ' ' + 'train_result')
            plt.savefig('./figures/train_TrainBeforeSum.png')
            plt.show()

            # 画验证集拟合效果
            plot_val_id = np.argmax(np.random.random(val_inputs.shape[0]))
            plt.plot(model.predict(val_inputs[plot_val_id:plot_val_id + 1, :, :]).squeeze().T,
                     label='predict')
            plt.plot(val_targets[plot_val_id:plot_val_id + 1, :].numpy().T, label='truth')
            plt.fill_between(np.arange(train_targets.shape[1]),
                             val_targets[plot_val_id, :].numpy().T * 0.8,
                             val_targets[plot_val_id, :].numpy().T * 1.2, color='blue', alpha=0.25)
            plt.legend()
            plt.title(region_name + ': ' + select_model[0] + ' model, lag days:' + select_model[1]
                      + ' ' + 'val_result')
            plt.savefig('./figures/val_TrainBeforeSum.png')
            plt.show()

    print('======================================================================')
    print('训练结束，正在保存结果。')
    model.save_weights('./saved_model/' + region_name + '_' + select_model[0] + '_' +
                       select_model[1])
    print('模型:  ', region_name, '  已保存！！！！！')
    save.save_result(model, region_name, cgi_list,
                     train_sample_inputs, train_sample_targets, test_inputs, test_targets, scalers)

    if return_model:
        return model