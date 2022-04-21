from config import *
import ImportData as data_importer
from TrainModel import train

if __name__ == '__main__':
    # 选择要调用的gpu
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')

    # 选择要细分的场景
    not_include = ['地市', '区县', '覆盖场景']
    raw_data, base_station_info = data_importer.import_data('./data/cgi_2year_南京.csv',
                                                            './data/剔除冗余特征后全量工参.csv',
                                                            not_include=not_include)

    # not_include = ['城市',  '覆盖场景']
    # raw_data, base_station_info = data_importer.import_data(
    #                     './5Gprocessed_data/5G全量数据.csv',
    #                     './5Gprocessed_data/处理后5G工参.csv',
    #                     not_include=not_include)

    raw_data.drop(['timestamp'], axis=1, inplace=True)
    full_cgi_list = base_station_info.index.unique()

    # 获取已经训练好的区县名字
    os.makedirs(result_path, exist_ok=True)
    files = os.listdir(result_path)
    cgi_list = []
    for file_name in files:
        data = pd.read_csv(result_path + '/' + file_name)
        split_name = file_name.split('_')
        if len(split_name) > 3:
            if split_name[3][:-4] in full_cgi_list:
                cgi_list.append(split_name[3][:-4])
    print("完整数量：", len(raw_data.index.unique()))
    raw_data = raw_data.drop(cgi_list)
    base_station_info = base_station_info.drop(cgi_list)
    print("剔除后数量：", len(raw_data.index.unique()))

    # 开始每个区进行训练
    batch_size = 100
    region_list = list(base_station_info['地点'].unique())
    region_list.sort()

    for region_name in region_list:
        cgi_list = base_station_info[base_station_info['地点'] == region_name].index.unique()

        # 如果区域为空或者在基站工参表里查询不到cgi则跳过
        if len(cgi_list) == 0 or len(set(raw_data.index.unique()) & set(cgi_list)) == 0:
            continue
        # 如果一个区域的cgi数量没有超过批训练的大小则直接训练，不需要批之间传递训练好的模型
        elif len(cgi_list) <= batch_size:
            train(region_name, raw_data, base_station_info.loc[cgi_list], select_model=select_model)
        # 如果一个区域的cgi数量超过批训练的大小则分批训练，需要批之间传递训练好的模型
        else:
            i = 0
            model = None
            for i in range(int(np.floor(len(cgi_list) / batch_size))):
                model = train(region_name,
                              raw_data,
                              base_station_info.loc[cgi_list[i * batch_size: (i + 1) * batch_size]],
                              model=model,
                              return_model=True,
                              select_model=select_model)

            train(region_name,
                  raw_data,
                  base_station_info.loc[cgi_list[(i + 1) * batch_size:]],
                  model=model,
                  select_model=select_model)
