from GetPatient import *
from GetWeather import *

weather = GetWeather()
weather_df = weather.get_history_weather_df("2020-07-01")
print(weather_df)
weather_df.to_csv('../data/weather_df.csv', index=False)

patient = GetPatient()
current_date = datetime.datetime.now()
delta = datetime.timedelta(days=-2)
yesterday = (current_date + delta).strftime('%Y-%m-%d')
patient_df = patient.get_patient_df()
print(patient_df)
patient_df.to_csv('../data/patient_df.csv', index=False)