import pandas as pd
import requests
import json
import numpy as np
import chinese_calendar
from datetime import date
# 生成某一年的日历
def gen_calendar(year=2022):
    # 生成日历列表
    start_date=str(year)+'0101'
    end_date=str(year)+'1231'
    df=pd.DataFrame()
    dt=pd.date_range(start_date, end_date,freq='1D')
    df['date']=dt
    df['year']=df['date'].dt.year
    df['month']=df['date'].dt.month
    df['day']=df['date'].dt.day
    # 计算周几
    df['dayofweek']=df['date'].dt.dayofweek+1
    df['dayofyear']=df['date'].dt.dayofyear
    # 获取法定节假日
    df['data_is_workday'] = df['date'].apply(lambda x:  chinese_calendar.is_workday(date.fromisoformat(x.strftime("%Y-%m-%d"))))
    df['on_holiday'] = df['date'].apply(lambda x: chinese_calendar.get_holiday_detail(date.fromisoformat(x.strftime("%Y-%m-%d")))[0])
    df['name'] = df['date'].apply(lambda x: chinese_calendar.get_holiday_detail(date.fromisoformat(x.strftime("%Y-%m-%d")))[1])

    return df
day_df = pd.DataFrame()
for i in range(2020, 2023):
    calendar_df = gen_calendar(i)
    day_df = pd.concat([day_df, calendar_df])
print(day_df.info())
day_df['date'] =  day_df['date'].apply(lambda x:x.strftime("%Y-%m-%d"))
day_df.set_index("date",inplace= True)
bool_cols = day_df.select_dtypes('boolean').columns.to_list()
day_df = pd.get_dummies(day_df,prefix=bool_cols,columns=bool_cols,drop_first=True)
day_df.drop('name',axis=1,inplace = True)
print(day_df.info())
day_df.to_csv('./data/date.csv')