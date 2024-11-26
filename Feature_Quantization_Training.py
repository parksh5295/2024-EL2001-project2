import pandas as pd
import numpy as np

# 양자화 함수 정의
def quantize_feature(df, feature, bin_edges):
    """주어진 feature를 bin_edges에 맞춰 양자화하고 그룹명(label)을 추가"""
    df[feature + '_group'] = pd.cut(df[feature], bins=bin_edges, labels=False, right=False) + 1
    return df

# 그룹화할 범위 설정 함수
def get_bins_for_feature(feature, step):
    """주어진 feature의 최솟값과 최댓값에 맞춰 구간을 설정"""
    min_val = df[feature].min()
    max_val = df[feature].max()
    bins = np.arange(min_val, max_val + step, step)  # 구간에 맞게 구간을 설정
    return bins

# CSV 파일 로드 (실제 데이터를 로드하는 부분)
df = pd.read_csv('./merged_training.csv')

# 양자화 처리

# 날짜 컬럼을 datetime 형식으로 변환
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# 'date' 컬럼에서 월 정보만 추출 (정수형으로 변환)
df['month'] = df['date'].dt.month  # 1~12 사이의 정수값

# 월별로 양자화: 1~12까지 월을 그대로 사용
month_bins = np.arange(1, 14)  # 1부터 12까지의 구간
df = quantize_feature(df, 'month', month_bins)

# 1. 'avg (temperature)', 'max (temperature)', 'min (temperature)' → 5 단위
temperature_bins = get_bins_for_feature('avg (temperature)', 5)
df = quantize_feature(df, 'avg (temperature)', temperature_bins)
df = quantize_feature(df, 'max (temperature)', temperature_bins)
df = quantize_feature(df, 'min (temperature)', temperature_bins)

# 2. 'avg (humidity)', 'max (humidity)', 'min (humidity)' → 5 단위
humidity_bins = get_bins_for_feature('avg (humidity)', 5)
df = quantize_feature(df, 'avg (humidity)', humidity_bins)
df = quantize_feature(df, 'max (humidity)', humidity_bins)
df = quantize_feature(df, 'min (humidity)', humidity_bins)

# 3. 'power' → 10 단위
power_bins = get_bins_for_feature('power', 10)
df = quantize_feature(df, 'power', power_bins)

# 4. 'avg (wind speed)', 'avg (local pressure)', 'avg (cloud cover)' → 0.5 단위
wind_speed_bins = get_bins_for_feature('avg(wind speed)', 0.5)
pressure_bins = get_bins_for_feature('avg(local pressure)', 0.5)
cloud_cover_bins = get_bins_for_feature('avg(cloud cover)', 0.5)

df = quantize_feature(df, 'avg(wind speed)', wind_speed_bins)
df = quantize_feature(df, 'avg(local pressure)', pressure_bins)
df = quantize_feature(df, 'avg(cloud cover)', cloud_cover_bins)

# 5. 'avg(ground temperature)', 'avg(evaporation volume)' → 0.2 단위
ground_temp_bins = get_bins_for_feature('avg(ground temperature)', 0.2)
evaporation_bins = get_bins_for_feature('avg(evaporation volume)', 0.2)

df = quantize_feature(df, 'avg(ground temperature)', ground_temp_bins)
df = quantize_feature(df, 'avg(evaporation volume)', evaporation_bins)

# 6. 'sun_rise', 'sun_set' → 10분 단위
def time_to_minutes(time_str):
    """시간:분 형식의 시간을 분으로 변환"""
    time_obj = pd.to_datetime(time_str, format='%H:%M')
    return time_obj.hour * 60 + time_obj.minute

df['sun_rise'] = df['sun_rise'].apply(time_to_minutes)
df['sun_set'] = df['sun_set'].apply(time_to_minutes)

sun_bins = np.arange(0, 1440, 10)  # 1440분은 24시간
df = quantize_feature(df, 'sun_rise', sun_bins)
df = quantize_feature(df, 'sun_set', sun_bins)

# 7. 'sun_max' → 5분 단위
def time_to_minutes_sun_max(time_str):
    """시간:분:초 형식의 시간을 분으로 변환 (5분 단위)"""
    time_obj = pd.to_datetime(time_str, format='%H:%M:%S')
    return time_obj.hour * 60 + time_obj.minute

df['sun_max'] = df['sun_max'].apply(time_to_minutes_sun_max)

sun_max_bins = np.arange(0, 1440, 5)  # 5분 단위
df = quantize_feature(df, 'sun_max', sun_max_bins)

# 8. 'day_duration', 'night_duration' → 분 단위 (5분 단위로 양자화)
def duration_to_minutes(duration_str):
    """'0 days HH:MM:SS' 형식의 시간을 분 단위로 변환"""
    time_str = duration_str.split(' ')[2]  # '09:37:00' 형태
    hours, minutes, seconds = map(int, time_str.split(':'))
    return hours * 60 + minutes + seconds / 60

df['day_duration'] = df['day_duration'].apply(duration_to_minutes)
df['night_duration'] = df['night_duration'].apply(duration_to_minutes)

duration_bins = np.arange(0, 1440, 5)  # 5분 단위
df = quantize_feature(df, 'day_duration', duration_bins)
df = quantize_feature(df, 'night_duration', duration_bins)

# 9. 'day_avg_temperature_app', 'night_avg_temperature_app' → 5 단위
df = quantize_feature(df, 'day_avg_temperature_app', temperature_bins)
df = quantize_feature(df, 'night_avg_temperature_app', temperature_bins)

# 10. 'Nitrogen Dioxide Concentration (ppm)', 'Ozone Concentration (ppm)' → 0.005 단위
nitrogen_dioxide_bins = get_bins_for_feature('Nitrogen Dioxide Concentration (ppm)', 0.005)
ozone_bins = get_bins_for_feature('Ozone Concentration (ppm)', 0.005)

df = quantize_feature(df, 'Nitrogen Dioxide Concentration (ppm)', nitrogen_dioxide_bins)
df = quantize_feature(df, 'Ozone Concentration (ppm)', ozone_bins)

# 11. 'Carbon Monoxide Concentration (ppm)' → 0.1 단위
carbon_monoxide_bins = get_bins_for_feature('Carbon Monoxide Concentration (ppm)', 0.1)
df = quantize_feature(df, 'Carbon Monoxide Concentration (ppm)', carbon_monoxide_bins)

# 12. 'Sulfur Dioxide Concentration (ppm)' → 0.001 단위
sulfur_dioxide_bins = get_bins_for_feature('Sulfur Dioxide Concentration (ppm)', 0.001)
df = quantize_feature(df, 'Sulfur Dioxide Concentration (ppm)', sulfur_dioxide_bins)

# 13. 'Particulate Matter (micrograms per cubic meter)', 'Fine Particulate Matter (micrograms per cubic meter)', 'AQI' → 10 단위
particulate_bins = get_bins_for_feature('Particulate Matter (micrograms per cubic meter)', 10)
fine_particulate_bins = get_bins_for_feature('Fine Particulate Matter (micrograms per cubic meter)', 10)
aqi_bins = get_bins_for_feature('AQI', 10)

df = quantize_feature(df, 'Particulate Matter (micrograms per cubic meter)', particulate_bins)
df = quantize_feature(df, 'Fine Particulate Matter (micrograms per cubic meter)', fine_particulate_bins)
df = quantize_feature(df, 'AQI', aqi_bins)

# 'label' → 0(정상), 1(비정상)으로 변환
df['label'] = df['label'].apply(lambda x: 0 if x == 0 else 1)

# 양자화된 그룹만 남기기 (원본 데이터를 제외하고 그룹 컬럼만 남김)
grouped_columns = [
    'month_group', 'avg (temperature)_group', 'max (temperature)_group', 'min (temperature)_group', 'avg (humidity)_group', 
    'max (humidity)_group', 'min (humidity)_group', 'power_group', 'avg(wind speed)_group', 
    'avg(local pressure)_group', 'avg(cloud cover)_group', 'avg(ground temperature)_group', 
    'avg(evaporation volume)_group', 'sun_rise_group', 'sun_set_group', 'sun_max_group', 
    'day_duration_group', 'night_duration_group', 'day_avg_temperature_app_group', 
    'night_avg_temperature_app_group', 'Nitrogen Dioxide Concentration (ppm)_group', 
    'Ozone Concentration (ppm)_group', 'Carbon Monoxide Concentration (ppm)_group', 
    'Sulfur Dioxide Concentration (ppm)_group', 'Particulate Matter (micrograms per cubic meter)_group', 
    'Fine Particulate Matter (micrograms per cubic meter)_group', 'AQI_group', 'label'
]

df_grouped = df[grouped_columns]

# 결과 저장
df_grouped.to_csv('./quantized_train_data.csv', index=False)

# 양자화 범위 정보 CSV로 저장
quantization_ranges = {
    'avg (temperature)': temperature_bins,
    'max (temperature)': temperature_bins,
    'min (temperature)': temperature_bins,
    'avg (humidity)': humidity_bins,
    'max (humidity)': humidity_bins,
    'min (humidity)': humidity_bins,
    'power': power_bins,
    'avg(wind speed)': wind_speed_bins,
    'avg(local pressure)': pressure_bins,
    'avg(cloud cover)': cloud_cover_bins,
    'avg(ground temperature)': ground_temp_bins,
    'avg(evaporation volume)': evaporation_bins,
    'sun_rise': sun_bins,
    'sun_set': sun_bins,
    'sun_max': sun_max_bins,
    'day_duration': duration_bins,
    'night_duration': duration_bins,
    'day_avg_temperature_app': temperature_bins,
    'night_avg_temperature_app': temperature_bins,
    'Nitrogen Dioxide Concentration (ppm)': nitrogen_dioxide_bins,
    'Ozone Concentration (ppm)': ozone_bins,
    'Carbon Monoxide Concentration (ppm)': carbon_monoxide_bins,
    'Sulfur Dioxide Concentration (ppm)': sulfur_dioxide_bins,
    'Particulate Matter (micrograms per cubic meter)': particulate_bins,
    'Fine Particulate Matter (micrograms per cubic meter)': fine_particulate_bins,
    'AQI': aqi_bins
}

quantization_df = pd.DataFrame.from_dict(quantization_ranges, orient='index').transpose()
quantization_df.to_csv('./quantization_ranges.csv', index=False)