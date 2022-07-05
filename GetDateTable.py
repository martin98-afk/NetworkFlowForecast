import pandas as pd
import requests
import json
import numpy as np
from datetime import date
from Scrapy.GetDate import get_holiday


def is_workday(dayofweek):
    if dayofweek < 6:
        return 1
    else:
        return 0


def is_holiday(date, holidays):
    if date in holidays:
        return 1
    else:
        return 0


# 生成某一年的日历
def gen_calendar(year=2022):
    holidays = get_holiday(year)
    # 生成日历列表
    start_date = str(year) + '0101'
    end_date = str(year) + '1231'
    df = pd.DataFrame()
    dt = pd.date_range(start_date, end_date, freq='1D')
    df['date'] = dt
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    # 计算周几
    df['dayofweek'] = df['date'].dt.dayofweek + 1
    df['dayofyear'] = df['date'].dt.dayofyear
    # 获取法定节假日
    df['data_is_workday'] = df['dayofweek'].apply(lambda x: is_workday(x))

    df['on_holiday'] = df['date'].apply(lambda x: is_holiday(x.strftime('%Y-%m-%d'), holidays))

    return df


'''
分解日期信息
ver 2022-07-05
现支持日期范围到2026年，由于是网上爬取的结果，2026年后的信息需要网站更新后才能爬取，
同时如果需要更细致的调休日期需要使用chinese calendar包，但是这个包更新较慢，同时由于本项目需要预测长时间序列，可能会导致无法获取到时间特征。
'''
day_df = pd.DataFrame()
for i in range(2020, 2026):
    calendar_df = gen_calendar(i)
    day_df = pd.concat([day_df, calendar_df])
print(day_df.info())
day_df['date'] = day_df['date'].apply(lambda x: x.strftime("%Y-%m-%d"))
day_df.set_index("date", inplace=True)
bool_cols = day_df.select_dtypes('boolean').columns.to_list()
day_df = pd.get_dummies(day_df, prefix=bool_cols, columns=bool_cols, drop_first=True)
day_df.to_csv('./data/date.csv')
