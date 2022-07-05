import datetime
import json
import pandas as pd
import requests
from bs4 import BeautifulSoup


class GetPatient:
    def __init__(self):
        self.headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}  #
        self.url = 'https://ncov.dxy.cn/ncovh5/view/pneumonia?from=timeline&isappinstalled=0'

    # 得到从dt_start后的确诊人数数据
    def get_patient_df(self, dt_start='2020-01-01'):
        strhtml = requests.get(self.url, headers=self.headers)
        content = strhtml.content.decode("utf-8")
        soup = BeautifulSoup(content, 'html.parser')
        lista = soup.find_all(name="script", attrs={"id": "getAreaStat"})
        account = str(lista)
        messages = json.loads(account[52:-21])
        value_list = []
        city_list = []

        for i in range(len(messages)):
            value = (messages[i].get('provinceName'), messages[i].get('provinceShortName'),
                     messages[i].get('currentConfirmedCount'), \
                     messages[i].get('confirmedCount'), messages[i].get('suspectedCount'),
                     messages[i].get('curedCount'), messages[i].get('deadCount'), \
                     messages[i].get('locationId'), messages[i].get('statisticsData'))
            value_list.append(value)
            city_value = messages[i].get('cities')

            for j in range(len(city_value)):
                city_value_list = (
                    city_value[j].get('cityName'), city_value[j].get('currentConfirmedCount'), \
                    city_value[j].get('confirmedCount'), city_value[j].get('suspectedCount'),
                    city_value[j].get('curedCount'), city_value[j].get('deadCount'), \
                    city_value[j].get('locationId'), city_value[j].get('statisticsData'))
                city_list.append(city_value_list)

        df = pd.DataFrame(value_list,
                          columns=['provinceName', 'provinceShortName', 'currentConfirmedCount',
                                   'confirmedCount', 'suspectedCount', 'curedCount', 'deadCount',
                                   'locationId', 'statisticsData'])

        statistic_data = []
        if True:
            file = requests.get(df[df['provinceShortName'] == '江苏']['statisticsData'].values[0])
            js_file = json.loads(file.content.decode('utf-8'))
            for j in range(len(js_file['data'])):
                statistic_data.append([
                    js_file['data'][j]['currentConfirmedCount'],
                    js_file['data'][j]['dateId']
                ])
        df2 = pd.DataFrame(statistic_data, columns=['currentConfirmedCount', 'dateId'])
        df2[['f_date']] = df2['dateId'].apply(self.date_manipulate)
        df2 = df2[['f_date', 'currentConfirmedCount']]
        df2 = df2[df2['f_date'] >= dt_start]
        return df2.iloc[1:]

    @staticmethod
    def date_manipulate(x):
        x = datetime.datetime.strptime(str(x), "%Y%m%d")
        return x


if __name__ == '__main__':
    patient = GetPatient()
    current_date = datetime.datetime.now()
    delta = datetime.timedelta(days=-2)
    yesterday = (current_date + delta).strftime('%Y-%m-%d')
    patient_df = patient.get_patient_df()
    print(patient_df)
    patient_df.to_csv('../data/patient_df.csv', index=False)
