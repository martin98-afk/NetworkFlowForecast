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


def save_result(model, region_name, cgi_list, time_label,
                train_inputs, train_tagets, test_inputs,
                test_targets, scalers):
    os.makedirs(result_path, exist_ok=True)
    # scaler = scalers['download']
    # 同时保存训练集和测试集
    for i in range(test_inputs.shape[0]):
        predict = model.predict(test_inputs[i:i + 1, :, :]).squeeze().T
        true = test_targets[i:i + 1, :].numpy().squeeze().T
        result = pd.DataFrame(np.vstack([time_label, true, predict]).T, columns=[
            'timestamp', 'true', 'predict'])
        result.to_csv(result_path + '/' + region_name + '_' + cgi_list[i] + '.csv', index=False)

        predict = model.predict(train_inputs[i:i + 1, :, :]).squeeze().T
        true = train_tagets[i:i + 1, :].numpy().squeeze().T
        result = pd.DataFrame(np.vstack([time_label, true, predict]).T, columns=[
            'timestamp', 'true', 'predict'])
        result.to_csv(result_path + '/train_' + region_name + '_' + cgi_list[i] + '.csv',
                      index=False)


def save_result_multi(
        model, region_name, cgi_list, time_label,
        train_inputs, train_tagets, train_predicts,
        test_inputs, test_targets, test_predicts, scalers):
    os.makedirs(result_path, exist_ok=True)
    # scaler = scalers['download']
    # 同时保存训练集和测试集
    for i in range(test_inputs.shape[0]):
        test_predict = predict_multi(model,
                                     test_inputs[i:i + 1, :, :],
                                     test_predicts[i:i + 1, :, :])

        predict = test_predict.squeeze().T
        true = test_targets[i:i + 1, :].numpy().squeeze().T
        result = pd.DataFrame(np.vstack([true, predict]).T, columns=['true', 'predict'])
        result.set_index(time_label[-output_len-label_len:], inplace=True)
        result.to_csv(result_path + '/' + region_name + '_' + cgi_list[i] + '.csv')

        train_predict = predict_multi(model,
                                      train_inputs[i:i + 1, :, :],
                                      train_predicts[i:i + 1, :, :])
        predict = train_predict.squeeze().T
        true = train_tagets[i:i + 1, :].numpy().squeeze().T
        result = pd.DataFrame(np.vstack([true, predict]).T, columns=['true', 'predict'])

        result.set_index(time_label[input_len-label_len:input_len + output_len], inplace=True)
        result.to_csv(result_path + '/train_' + region_name + '_' + cgi_list[i] + '.csv')
