from config import *

import re
import pymysql


# load_csv函数，参数分别为csv文件路径，表名称，数据库名称
def load_csv(csv_file_path, table_name, database, colum=None):
    conn = pymysql.connect(**config)
    cur = conn.cursor()
    # 打开csv文件
    file = open(csv_file_path, 'r', encoding='utf-8')
    # 读取csv文件第一行字段名，创建表
    reader = file.readline()
    b = reader.split(',')
    # print(b)
    if colum == None:
        colum = ''
        for a in b:
            a = re.sub('\s', '', a)
            colum = colum + a + ' varchar(255),'
        colum = colum[:-1]
    # 编写sql，create_sql负责创建表，data_sql负责导入数据
    create_sql = 'create table if not exists ' + table_name + ' ' + '(' + colum + ')' + ' DEFAULT CHARSET=utf8'
    data_sql = "LOAD DATA LOCAL INFILE '%s' INTO TABLE %s FIELDS TERMINATED BY ',' LINES TERMINATED BY '\\r\\n' IGNORE 1 LINES" % (
        csv_file_path, table_name)
    # print(data_sql)
    # 使用数据库
    cur.execute('use %s' % database)
    # 设置编码格式
    cur.execute('SET NAMES utf8;')
    cur.execute('SET character_set_connection=utf8;')
    # 执行create_sql，创建表
    cur.execute(create_sql)
    # 执行data_sql，导入数据
    cur.execute(data_sql)
    conn.commit()
    # 关闭连接
    conn.close()
    cur.close()


if __name__ == '__main__':
    data = pd.read_csv('../4Gprocessed_data/剔除冗余特征后全量工参.csv')
    columns = data.columns.to_list()
    columns[1] = 'city'
    columns[2] = 'region'
    columns = ' varchar(255),'.join(columns) + ' varchar(255)'
    load_csv('../4Gprocessed_data/剔除冗余特征后全量工参.csv', 'stationInfo_4g', 'xwtech', columns)

    data = pd.read_csv('../5Gprocessed_data/处理后5G工参.csv')
    columns = data.columns.to_list()
    columns[1] = 'city'
    columns = ' varchar(255),'.join(columns) + ' varchar(255)'
    load_csv('../5Gprocessed_data/处理后5G工参.csv', 'stationInfo_5g', 'xwtech', columns)
    # load_csv('/home/xwtech/无线网小区流量预测专用/NetworkFlowForecastLSTM/process_data/4G全量数据.csv',
    #          'flow_4g', 'xwtech')

    # load_csv('/home/xwtech/无线网小区流量预测专用/NetworkFlowForecastLSTM/process_data/5G全量数据.csv', 'flow_5g', 'xwtech')
