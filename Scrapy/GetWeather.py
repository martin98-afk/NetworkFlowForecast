import os.path

from config import *
import datetime
import re
import requests
from bs4 import BeautifulSoup
from xpinyin import Pinyin


class GetWeather:
    '''
    爬取·天气数据类
    '''

    def __init__(self, city_name='yangzhou', target_directory='../data/'):
        self.url_head = 'http://www.tianqihoubao.com'
        self.url = 'http://www.tianqihoubao.com/lishi/' + city_name + '/month/202112.html'
        self.headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}
        self.fw_url = 'http://www.weather.com.cn/weather/101190101.shtml'
        self.city_name = city_name
        self.target_directory = target_directory

    # 获取2020-01-01日至昨天的历史天气数据
    def get_history_weather_df(self, start_date):
        df_union_all = pd.DataFrame()
        start_date, df_union_all = self.check_history_date(start_date, df_union_all)
        current_dt = (datetime.datetime.now()).strftime('%Y-%m-%d')
        count = 1
        dt = start_date
        while dt < current_dt:

            day_disparity = datetime.timedelta(days=count)
            dt = datetime.datetime.strptime(start_date, "%Y-%m-%d") + day_disparity
            dt = dt.strftime('%Y-%m-%d')
            weather_info = self.scrap_weather_day(dt)
            print(weather_info)
            df_union_all = pd.concat([df_union_all, weather_info])
            count += 1
            if count % 10 == 0:
                df_union_all.to_csv(self.target_directory + 'weather_df_' + self.city_name + \
                                    '.csv', index=False)
        return df_union_all.iloc[1:]

    @staticmethod
    def date_manipulate(x):
        x = x.str.replace('年', '-').str.replace('月', '-').str.replace('日', '')
        x = pd.to_datetime(x, format='%Y-%m-%d', errors='coerce').dt.date
        return x

    @staticmethod
    def tempreture_manipulate(temp):
        highT = temp.split('/')[0]
        lowT = temp.split('/')[1]
        p = re.compile(r'-?[\d]+[.]?[\d]*')
        maxT = int(p.findall(highT)[0])
        minT = int(p.findall(lowT)[0])
        return (maxT, minT)

    @staticmethod
    def weather_manipulate(wea):
        if '雨' in wea:
            res = '雨'
        elif '雪' in wea:
            res = '雪'
        elif '晴' in wea:
            res = '晴'
        else:
            res = '多云'
        return res

    def check_history_date(self, start_date, df):
        print(self.target_directory + 'weather_df_' + city_name + '.csv')
        if os.path.exists(self.target_directory + 'weather_df_' + city_name + '.csv'):
            data = pd.read_csv(self.target_directory + 'weather_df_' + city_name + '.csv')
            data.sort_values('f_date', inplace=True)
            last_date = data['f_date'].unique()[-1]
            if last_date >= start_date:
                print('查到存在历史天气数据，从' + last_date + '开始加载数据')
                return last_date, data
            else:
                return start_date, df
        else:
            print('暂无历史天气数据！')
            return start_date, df

    # 预测未来7天的天气
    def scrap_future_weather(self, current_date=datetime.datetime.now(), days=7):
        # 获取日期列表
        dt_list = []
        for i in range(days):
            day_disparity = datetime.timedelta(days=i)
            dt = current_date + day_disparity
            dt_list.append(dt.strftime('%Y-%m-%d'))
        # 根据url爬取天气数据
        html_content = requests.get(self.fw_url, headers=self.headers).content
        soup = BeautifulSoup(html_content, 'html.parser')
        content = soup.find_all(name='ul', attrs={'class': 't clearfix'})[0]
        weather_condition = content.find_all(name='p', attrs={'class': 'wea'})
        high_temp = content.find_all(name='span')
        low_temp = content.find_all(name='i')
        # 解析爬取信息
        find_content = re.compile('>(.*?)<')
        find_temp = re.compile(r"[-]?\d+")
        find_temp2 = re.compile('>(.*?)℃<')
        weather_condition_content = [re.findall(find_content, str(wc))[0] for wc in
                                     weather_condition]
        high_temp_content = [re.findall(find_temp, str(ht)) for ht in high_temp]
        low_temp_content = [re.findall(find_temp2, str(lt)) for lt in low_temp]
        # 删除筛选出的空字符串
        while [] in high_temp_content: high_temp_content.remove([])
        while [] in low_temp_content: low_temp_content.remove([])

        weather = []
        for i, wea in enumerate(weather_condition_content):
            if "雨" in wea:
                weather.append("雨")
            elif "雪" in wea:
                weather.append("雪")
            elif "云" in wea:
                weather.append("多云")
            else:
                weather.append("晴")

        data = []
        if len(high_temp_content) < 7:
            data.append(
                [
                    dt_list[i],
                    int(high_temp_content[0][0]),
                    int(low_temp_content[0][0]),
                    weather[i]
                ]
            )
            for i in range(6):
                data.append(
                    [
                        dt_list[i],
                        int(high_temp_content[i][0]),
                        int(low_temp_content[i + 1][0]),
                        weather[i]
                    ]
                )
            print(1)
        else:
            for i in range(7):
                data.append(
                    [
                        dt_list[i],
                        int(high_temp_content[i][0]),
                        int(low_temp_content[i][0]),
                        weather[i]
                    ]
                )
        df = pd.DataFrame(data, columns=['f_date', 'max_tempreture', 'min_tempreture', 'weather'])
        df.set_index(['f_date'], inplace=True)
        return df

    def scrap_weather_day(self, date):
        dt = str(date).replace('-', '')
        url = 'http://www.tianqihoubao.com/lishi/' + self.city_name + '/' + dt + '.html'
        table = pd.read_html(url, encoding='gbk')[0]

        find_temp = re.compile('(.*?)℃')
        try:
            df = pd.DataFrame([[
                date,
                re.findall(find_temp, table['白天'].iloc[1])[0],
                re.findall(find_temp, table['夜晚'].iloc[1])[0],
                table['白天'].iloc[0] + '/' + table['夜晚'].iloc[0]
            ]])
        except:
            return pd.DataFrame()
        df.columns = ['f_date', 'max_tempreture', 'min_tempreture', 'weather']
        df['weather'] = df.apply(lambda row: self.weather_manipulate(row['weather']), axis=1)
        return df


def update_info(target_directory='../data/'):
    test = Pinyin()
    result = test.get_pinyin(city_name)
    city = result.replace('-', '')
    os.makedirs(target_directory, exist_ok=True)
    weather = GetWeather(city_name=city, target_directory=target_directory)
    weather_df = weather.get_history_weather_df("2020-07-01")
    print(weather_df)
    weather_df.to_csv(target_directory + 'weather_df_' + city_name + '.csv', index=False)


if __name__ == '__main__':
    update_info()
    ...
    # test = Pinyin()
    # result = test.get_pinyin(city_name)
    # city = result.replace('-', '')
    # target_directory = '../data/'
    # os.makedirs(target_directory, exist_ok=True)
    # weather = GetWeather(city)
    # weather_df = weather.get_history_weather_df("2020-07-01")
    # print(weather_df)
    # weather_df.to_csv(target_directory + 'weather_df_' + city_name + '.csv', index=False)
