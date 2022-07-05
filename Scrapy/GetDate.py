import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import datetime


def get_holiday(year=2023):
    url = 'https://publicholidays.cn/zh/' + str(year) + '-dates/'
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}  #

    html = requests.get(url, headers).content

    soup = BeautifulSoup(html, 'html.parser')
    content = soup.find_all(name='tbody')
    content1 = content[0].find_all(name='td')
    content2 = content[1].find_all(name='td')

    find_content = re.compile('>(.*?)<')
    find_month = re.compile('(\d+)月')
    find_day = re.compile('(\d+)日')
    content1 = [re.findall(find_content, str(string)) for string in content1][:-1]
    content1 = [ele[0] if len(ele) == 1 else [ele[1]] for ele in content1][::3]
    content1_month = [[int(month) for month in re.findall(find_month, string)] for string in
                      content1]
    content1_day = [[int(day) for day in re.findall(find_day, string)] for string in content1]
    content1_date = []
    for month_list, day_list in zip(content1_month, content1_day):
        if len(month_list) == 1:
            content1_date.append(datetime.datetime(year, month_list[0], day_list[0]).strftime(
                '%Y-%m-%d'))
        else:
            start_date = datetime.datetime(year, month_list[0], day_list[0])
            end_date = datetime.datetime(year, month_list[1], day_list[1])
            content1_date.extend(
                pd.date_range(start_date, end_date, freq='1D').astype('str').values.tolist())

    content2 = [re.findall(find_content, str(string)) for string in content2]
    content2 = [ele[0] if len(ele) == 1 else [ele[1]] for ele in content2][::3]

    content2_month = [[int(month) for month in re.findall(find_month, string)] for string in
                      content2]
    content2_day = [[int(day) for day in re.findall(find_day, string)] for string in content2]
    content2_date = []
    for month_list, day_list in zip(content2_month, content2_day):
        if len(month_list) == 1:
            content2_date.append(datetime.datetime(year, month_list[0], day_list[0]).strftime(
                '%Y-%m-%d'))
        else:
            start_date = datetime.datetime(year, month_list[0], day_list[0])
            end_date = datetime.datetime(year, month_list[1], day_list[1])
            content2_date.extend(
                pd.date_range(start_date, end_date, freq='1D').astype('str').values.tolist())

    holiday = content1_date + content2_date
    print(year, '年的节假日为：', holiday)
    return holiday


if __name__ == '__main__':
    get_holiday()
