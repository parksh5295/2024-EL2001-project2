import pandas as pd
import numpy as np

columns = [
    'date', 'avg (temperature)', 'max (temperature)', 'min (temperature)', 'avg (humidity)', 
    'max (humidity)', 'min (humidity)', 'power', 'label', 'avg(wind speed)', 'avg(local pressure)', 
    'avg(cloud cover)', 'avg(ground temperature)', 'avg(evaporation volume)', 'sun_rise', 'sun_max', 
    'sun_set', 'day_duration', 'night_duration', 'day_avg_temperature_app', 'night_avg_temperature_app', 
    'Nitrogen Dioxide Concentration (ppm)', 'Ozone Concentration (ppm)', 'Carbon Monoxide Concentration (ppm)', 
    'Sulfur Dioxide Concentration (ppm)', 'Particulate Matter (㎍/㎥)', 'Fine Particulate Matter (㎍/㎥)', 'AQI'
]

# 데이터 불러오기
df = pd.read_csv('./merged_training.csv', header=None, names=columns, skiprows=1)

# date를 연도-월별로 나누기
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['year_month'] = df['date'].dt.to_period('M')

# 양자화 함수 정의
def quantize_feature(df, feature, bins, labels):
    """주어진 feature를 bins에 맞춰 양자화하고 그룹명(label)을 추가"""
    df[feature + '_group'] = pd.cut(df[feature], bins=bins, labels=labels, right=False)
    return df

# 그룹명 생성용 자연수 리스트 (1부터 시작하는 자연수)
def generate_labels(n):
    return [i for i in range(1, n+1)]

# 각 feature에 대해 양자화 적용

# 1. 'avg (temperature)', 'max (temperature)', 'min (temperature)' → 5 단위
temperature_bins = np.arange(-100, 101, 5)  # 범위 설정 (-100 ~ 100)
temperature_labels = generate_labels(len(temperature_bins) - 1)

df = quantize_feature(df, 'avg (temperature)', temperature_bins, temperature_labels)
df = quantize_feature(df, 'max (temperature)', temperature_bins, temperature_labels)
df = quantize_feature(df, 'min (temperature)', temperature_bins, temperature_labels)

# 2. 'avg (humidity)', 'max (humidity)', 'min (humidity)' → 5 단위
humidity_bins = np.arange(0, 101, 5)
humidity_labels = generate_labels(len(humidity_bins) - 1)

df = quantize_feature(df, 'avg (humidity)', humidity_bins, humidity_labels)
df = quantize_feature(df, 'max (humidity)', humidity_bins, humidity_labels)
df = quantize_feature(df, 'min (humidity)', humidity_bins, humidity_labels)

# 3. 'power' → 10 단위
power_bins = np.arange(0, df['power'].max() + 10, 10)
power_labels = generate_labels(len(power_bins) - 1)

df = quantize_feature(df, 'power', power_bins, power_labels)

# 4. 'avg (wind speed)', 'avg (local pressure)', 'avg (cloud cover)' → 0.5 단위
wind_speed_bins = np.arange(0, df['avg(wind speed)'].max() + 0.5, 0.5)
pressure_bins = np.arange(900, 1050, 0.5)
cloud_cover_bins = np.arange(0, 100, 0.5)

df = quantize_feature(df, 'avg(wind speed)', wind_speed_bins, generate_labels(len(wind_speed_bins) - 1))
df = quantize_feature(df, 'avg(local pressure)', pressure_bins, generate_labels(len(pressure_bins) - 1))
df = quantize_feature(df, 'avg(cloud cover)', cloud_cover_bins, generate_labels(len(cloud_cover_bins) - 1))

# 5. 'sun_rise', 'sun_set' → 10분 단위
def time_to_minutes(time_str):
    """시간:분 형식의 시간을 분으로 변환"""
    time_obj = pd.to_datetime(time_str, format='%H:%M')
    return time_obj.hour * 60 + time_obj.minute

df['sun_rise'] = df['sun_rise'].apply(time_to_minutes)
df['sun_set'] = df['sun_set'].apply(time_to_minutes)

sun_bins = np.arange(0, 1440, 10)  # 1440분은 24시간
df = quantize_feature(df, 'sun_rise', sun_bins, generate_labels(len(sun_bins) - 1))
df = quantize_feature(df, 'sun_set', sun_bins, generate_labels(len(sun_bins) - 1))

# 6. 'sun_max' → 5분 단위
def time_to_minutes_sun_max(time_str):
    """시간:분:초 형식의 시간을 분으로 변환 (5분 단위)"""
    time_obj = pd.to_datetime(time_str, format='%H:%M:%S')
    return time_obj.hour * 60 + time_obj.minute

df['sun_max'] = df['sun_max'].apply(time_to_minutes_sun_max)

sun_max_bins = np.arange(0, 1440, 5)  # 5분 단위
df = quantize_feature(df, 'sun_max', sun_max_bins, generate_labels(len(sun_max_bins) - 1))

# '0 days 09:37:00' 형식을 분 단위로 변환하는 함수
def duration_to_minutes(duration_str):
    # 'days'와 그 앞의 '0'을 제거하고, 'HH:MM:SS' 형식만 추출
    time_str = duration_str.split(' ')[2]  # '09:37:00' 형태
    hours, minutes, seconds = map(int, time_str.split(':'))
    
    # 시간, 분, 초를 모두 분 단위로 환산
    total_minutes = hours * 60 + minutes + seconds / 60
    return total_minutes

df['day_duration'] = df['day_duration'].apply(duration_to_minutes)
df['night_duration'] = df['night_duration'].apply(duration_to_minutes)

duration_bins = np.arange(0, 1440, 5)  # 5분 단위
df = quantize_feature(df, 'day_duration', duration_bins, generate_labels(len(duration_bins) - 1))
df = quantize_feature(df, 'night_duration', duration_bins, generate_labels(len(duration_bins) - 1))

# 8. 'day_avg_temperature_app', 'night_avg_temperature_app' → 5 단위
df = quantize_feature(df, 'day_avg_temperature_app', temperature_bins, temperature_labels)
df = quantize_feature(df, 'night_avg_temperature_app', temperature_bins, temperature_labels)

# 9. 'Nitrogen Dioxide Concentration (ppm)', 'Ozone Concentration (ppm)' → 0.005 단위
nitrogen_dioxide_bins = np.arange(0, df['Nitrogen Dioxide Concentration (ppm)'].max() + 0.005, 0.005)
ozone_bins = np.arange(0, df['Ozone Concentration (ppm)'].max() + 0.005, 0.005)

df = quantize_feature(df, 'Nitrogen Dioxide Concentration (ppm)', nitrogen_dioxide_bins, generate_labels(len(nitrogen_dioxide_bins) - 1))
df = quantize_feature(df, 'Ozone Concentration (ppm)', ozone_bins, generate_labels(len(ozone_bins) - 1))

# 10. 'Carbon Monoxide Concentration (ppm)' → 0.1 단위
carbon_monoxide_bins = np.arange(0, df['Carbon Monoxide Concentration (ppm)'].max() + 0.1, 0.1)
df = quantize_feature(df, 'Carbon Monoxide Concentration (ppm)', carbon_monoxide_bins, generate_labels(len(carbon_monoxide_bins) - 1))

# 11. 'Sulfur Dioxide Concentration (ppm)' → 0.001 단위
sulfur_dioxide_bins = np.arange(0, df['Sulfur Dioxide Concentration (ppm)'].max() + 0.001, 0.001)
df = quantize_feature(df, 'Sulfur Dioxide Concentration (ppm)', sulfur_dioxide_bins, generate_labels(len(sulfur_dioxide_bins) - 1))

# 12. 'Particulate Matter (㎍/㎥)', 'Fine Particulate Matter (㎍/㎥)', 'AQI' → 10 단위
particulate_bins = np.arange(0, df['Particulate Matter (㎍/㎥)'].max() + 10, 10)
fine_particulate_bins = np.arange(0, df['Fine Particulate Matter (㎍/㎥)'].max() + 10, 10)
aqi_bins = np.arange(0, df['AQI'].max() + 10, 10)

df = quantize_feature(df, 'Particulate Matter (㎍/㎥)', particulate_bins, generate_labels(len(particulate_bins) - 1))
df = quantize_feature(df, 'Fine Particulate Matter (㎍/㎥)', fine_particulate_bins, generate_labels(len(fine_particulate_bins) - 1))
df = quantize_feature(df, 'AQI', aqi_bins, generate_labels(len(aqi_bins) - 1))

# 13. 'label' → 0(정상), 1(비정상)로 변환
df['label'] = df['label'].apply(lambda x: 0 if x == 0 else 1)

# 결과 확인
print(df.head())

# 저장 (필요한 경우)
df.to_csv('./quantized_train_data.csv', index=False)
