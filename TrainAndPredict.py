import argparse

import ImportData as data_importer
from TrainModel import train
from config import *
from process_data.Merge45gData import merge_45g, merge_4g
import threading


def start_pipeline(gpu_index, mode, city=city_name, process_mode='validate'):
    # 设置在哪个虚拟gpu上运行模型
    with tf.device('/device:GPU:' + str(gpu_index+1)):
        # 选择要细分的场景
        raw_data = data_importer.import_data(
                './' + mode[8] + 'Gprocessed_data/missing_value/fillna_example'
                + mode[-5:] + '_' + city + '.csv',
                './data/date.csv',
                mode,
                city)

        # 获取时间label
        time_label = raw_data.timestamp.unique()
        time_label = [str(date)[:10] for date in time_label]
        print('最新日期为: ', raw_data.timestamp.unique()[-1])
        raw_data.drop(['timestamp'], axis=1, inplace=True)
        print(raw_data)

        # 获取已经训练好的区县名字
        os.makedirs(result_path + '_' + city, exist_ok=True)
        files = os.listdir(result_path + '_' + city)
        cgi_list = []
        for file_name in files:
            split_name = file_name[:-4].split('_')
            if len(split_name) > 3 and split_name[0] == mode:
                if '_'.join(split_name[1:3]) in list(raw_data.index.unique()):
                    cgi_list.append('_'.join(split_name[1:3]))
            if len(split_name) > 8 and split_name[0] == 'na' and split_name[-1] == 'predict':
                if '_'.join(split_name[5:8]) in list(raw_data.index.unique()):
                    cgi_list.append('_'.join(split_name[5:8]))
        print("完整数量：", len(raw_data.index.unique()))
        raw_data = raw_data.drop(cgi_list)
        print("剔除后数量：", len(raw_data.index.unique()))

        best_score = 20
        for i, sector_id in enumerate(raw_data.index.unique()):
            filter_data = raw_data.loc[sector_id]

            score = train(sector_id,
                          filter_data,
                          time_label,
                          best_score,
                          mode,
                          gpu_index,
                          city_name=city,
                          select_model=select_model,
                          process_mode=process_mode)
            # 储存当前效果最好的模型
            if score < best_score:
                best_score = score


if __name__ == '__main__':
    # 设置模型参数
    parser = argparse.ArgumentParser()
    # parser.add_argument("-g", "--gpu", default=0, help="select which gpu to run the code", type=int)
    parser.add_argument("-m", "--mode", default='na_test_5g_20_30',
                        help='select the mode to run the code, default: na_test_4g_20_30')
    parser.add_argument("-r", "--result", default='predict',
                        help='''
                        select which result you want to get:
                        validate: 验证模型好坏，输出结果有真实值做对照，但是无法预测未来值；
                        predict: 对未来5个月的数据进行预测。
                        默认值：predict
                        ''')
    parser.add_argument("-c", "--console_output", default=True, type=bool,
                        help='''
                        是否在控制台输出结果, False 则将结果输出到日志中。
                        ''')
    args = parser.parse_args()
    mode = args.mode
    result = args.result
    console_output = args.console_output

    # 将输出结果输出到日志文件
    if not console_output:
        f_handler = open('./result/out_TrainAndPredict.log', 'w')
        __console__ = sys.stdout
        sys.stdout = f_handler
        print('输出输入到日志中：')

    # 选择要调用的gpu
    gpus = tf.config.list_physical_devices(device_type='GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 1) for i in range(4)]
                )
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(logical_gpus))
        except RuntimeError as e:
            print(e)

    # 开始每个区进行训练
    # 多少个小区的数据混合在一起训练
    city_list = ['宿迁', '苏州', '徐州', '淮安']
    threading_list = []
    for i, city in enumerate(city_list):
        # 第四个参数控制运行模式，validate为验证模型，predict为对未来进行预测
        t = threading.Thread(target=start_pipeline, args=(i, mode, city, result))
        threading_list.append(t)

    for i, t in enumerate(threading_list):
        t.start()
        print('线程', i, '已开启训练!')

    for t in threading_list:
        t.join()
