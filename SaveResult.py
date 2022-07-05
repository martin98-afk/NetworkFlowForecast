import matplotlib.pyplot as plt

from config import *
from models.Losses import *


def predict_multi(model, input1, input2):
    predict = model.predict({'previous info': input1,
                             'predict info': input2})
    predict[predict < 0] = 0
    return predict


# 总量预测结果保存
def save_sum_result(model, region_name, cgi_list, test_inputs, test_targets):
    # 保存测试集
    for i in range(test_inputs.shape[0]):
        predict = model.predict(test_inputs[i:i + 1, :, :]).squeeze().T
        true = test_targets[i:i + 1, :].numpy().squeeze().T
        result = pd.DataFrame(np.vstack([true, predict]).T, columns=['true', 'predict'])
        result.to_csv(result_path + '/' + region_name + '_' + cgi_list[i] + '.csv', index=False)


def save_result_multi(
        model, region_name, cgi_list, city_name, mode,
        train_inputs, train_tagets, train_predicts, train_time_label,
        test_inputs, test_targets, test_predicts, test_time_label, scalers, process_mode):
    os.makedirs(result_path + '_' + city_name, exist_ok=True)
    scaler = scalers['download']
    # 同时保存训练集和测试集
    for i in range(test_inputs.shape[0]):
        test_predict = predict_multi(model,
                                     test_inputs[i:i + 1, :, :],
                                     test_predicts[i:i + 1, :, :])
        if process_mode == 'validate':
            predict = scaler.inverse_transform(test_predict).squeeze().T
            true = scaler.inverse_transform(test_targets[i:i + 1, :]).squeeze().T
            result = pd.DataFrame(np.vstack([true, predict]).T, columns=['true', 'predict'])
            result.set_index(test_time_label, inplace=True)
            result.to_csv(result_path + '_' + city_name + '/' + mode + '_' +
                          cgi_list[i] + '_' + process_mode + '.csv')
        else:
            predict = scaler.inverse_transform(test_predict).squeeze().T
            result = pd.DataFrame(predict.T, columns=['predict'])
            result.set_index(test_time_label, inplace=True)
            result.to_csv(result_path + '_' + city_name + '/' + mode + '_' +
                          cgi_list[i] + '_' + process_mode + '.csv')

        train_predict = predict_multi(model,
                                      train_inputs[i:i + 1, :, :],
                                      train_predicts[i:i + 1, :, :])
        predict = scaler.inverse_transform(train_predict).squeeze().T
        true = scaler.inverse_transform(train_tagets[i:i + 1, :]).squeeze().T
        result = pd.DataFrame(np.vstack([true, predict]).T, columns=['true', 'predict'])

        result.set_index(train_time_label, inplace=True)
        result.to_csv(
            result_path + '_' + city_name + '/train_' + mode + '_' +
            cgi_list[i] + '.csv')
