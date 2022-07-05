import pandas as pd
import psutil

import ProcessData as processer
import SaveResult as save
import models.Transformer as transformer
from config import *
from evaluation.CalculateAcc import cal_single_acc

os.makedirs('./saved_model', exist_ok=True)


def predict_multi(model, input1, input2):
    predict = model.predict({'previous info': input1,
                             'predict info': input2})
    predict[predict < 0] = 0
    return predict


def train(region_name, raw_data,
          time_label,
          best_score,
          mode,
          gpu_index,
          city_name=city_name,
          select_model='transformer',
          process_mode='predict'):
    '''

    :param region_name:
    :param raw_data:
    :param time_label:
    :param best_score:
    :param mode:
    :param city_name:
    :param select_model:
    :param process_mode:
    :return:
    '''
    start = time.time()
    tf.keras.backend.clear_session()
    ################# 数据集划分 ####################################
    if process_mode == 'validate':
        (train_inputs, train_targets, train_predicts, train_time_label,
         test_inputs, test_targets, test_predicts, test_time_label,
         train_sample_inputs, train_sample_targets, train_sample_predicts, train_sample_time_label,
         cgi_list), scalers = \
            processer.split_region(raw_data, time_label,
                                   input_len=input_len,
                                   output_len=output_len,
                                   process_mode='validate')
    else:
        (train_inputs, train_targets, train_predicts, train_time_label,
         test_inputs, test_targets, test_predicts, test_time_label,
         train_sample_inputs, train_sample_targets, train_sample_predicts, train_sample_time_label,
         cgi_list), scalers = \
            processer.split_region(raw_data, time_label,
                                   input_len=input_len,
                                   output_len=output_len,
                                   process_mode='predict')

    print('测试数据形状： ', test_inputs.shape)
    print('训练数据形状： ', train_inputs.shape)

    # 选择需要训练的模型
    # ########## 根据选择的模型进行加载 #####################################
    # if not os.path.exists('./saved_model/' + mode + '_' + select_model +
    #                                         '_InpLen' + str(input_len) + '_OutLen' + str(
    #                                         output_len) + '_' + city_name + '.index'):

    transformer_model = transformer.Transformer(train_inputs, train_targets, train_predicts)
    transformer_model.build_model()
    model = transformer_model.model
    # elif os.path.exists('./saved_model/' + mode + '_' + select_model +
    #                                       '_InpLen' + str(input_len) + '_OutLen' + str(
    #                                         output_len) + '_' + city_name + '.index'):
    #     transformer_model = transformer.Transformer(train_inputs, train_targets, train_predicts)
    #     transformer_model.build_model()
    #     model = transformer_model.model
    #     model.load_weights(
    #         './saved_model/' + mode + '_' + select_model + '_InpLen' +
    #         str(input_len) + '_OutLen' + str(output_len) + '_' + city_name)
    #     print('已读取到模型!')

    epochs = 100
    #############################################################################
    ####################### 开始模型训练 ########################################
    ############################################################################
    # plot_model(model, to_file='./figures/model.png')
    print(model.summary())
    print('正在训练：', region_name, '的小区数据')
    print('======================================================================')
    count = 0
    week_mape = 0
    memory_list = []
    for i in range(epochs):
        if process_mode == 'validate':
            history = model.fit({'previous info': train_inputs, 'predict info': train_predicts},
                                train_targets, validation_data=(
                    {'previous info': test_inputs, 'predict info': test_predicts}, test_targets),
                                batch_size=16, epochs=2, verbose=1)
        else:
            history = model.fit({'previous info': train_inputs, 'predict info': train_predicts},
                                train_targets, batch_size=16, epochs=2, verbose=1)
        print('训练总轮数：', (i + 1) * 2)
        if i % 5 == 0:
            ################## 输出模型验证效果 ##############################
            # 画训练集拟合效果
            train_predict = predict_multi(model,
                                          train_inputs[:1, :, :],
                                          train_predicts[:1, :, :])

            predictions = predict_multi(model, test_inputs, test_predicts)
            if process_mode == 'validate':
                acc, week_mape = cal_single_acc(test_targets, predictions)
                print('偏差率小于20%的小区数量占比: ', acc, '%')

            # 画训练集拟合效果
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            x = np.arange(output_len)
            axes[0].plot(x, train_predict.squeeze().T, label='predict')
            axes[0].plot(x, train_targets[: 1, :].numpy().T,
                         label='truth')

            axes[0].set_xticks(x[::20])
            axes[0].set_xticklabels(train_time_label[0][::20])
            axes[0].legend()
            axes[0].set_title(
                city_name + ' ' + region_name + ': ' + select_model + 'model, lag days:' + str(
                    input_len) + ' train_result')
            # 画验证集拟合效果
            axes[1].plot(x, predictions.squeeze().T, label='predict')
            if process_mode == 'validate':
                axes[1].plot(x, test_targets.numpy().T,
                             label='truth')
            axes[1].legend()
            axes[1].set_xticks(x[::20])
            axes[1].set_xticklabels(test_time_label[0][::20])
            axes[1].set_title(
                city_name + ' ' + region_name + ': ' + select_model + ' model, lag days:' + str(
                    input_len) + ' val_result')
            plt.savefig('./figures/val_TrainBeforeSum_' + str(gpu_index) + '.png')

            memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_list.append(memory)
            print('目前使用内存量: %.2f MB' % memory)

    # 打印训练过程中内存占用情况
    plt.plot(memory_list)
    plt.title('训练过程中内存使用情况图')
    plt.savefig('./figures/memory_usage.png')

    print('======================================================================')
    print('训练结束，正在保存结果。')
    ################## 保存模型结果 ###################################
    # if week_mape < best_score and process_mode == 'validate':
    #     model.save_weights('./saved_model/' + mode + '_' + select_model +
    #                        '_InpLen' + str(input_len) + '_OutLen' + str(
    #         output_len) + '_' + city_name)
    #     print('模型:  ', region_name, '  已保存！！！！！')

    save.save_result_multi(model, region_name, cgi_list, city_name, mode,
                           train_sample_inputs, train_sample_targets, train_sample_predicts,
                           train_sample_time_label,
                           test_inputs, test_targets, test_predicts, test_time_label, scalers,
                           process_mode
                           )

    # 打印训练耗费时间
    end = time.time()
    print('训练一个stationid的数据花费的时间：%.2f 分钟' % ((end - start) / 60))
    # 清除内存占用和tensorflow图
    tf.keras.backend.clear_session()
    gc.collect()
    return week_mape


if __name__ == '__main__':
    ...
