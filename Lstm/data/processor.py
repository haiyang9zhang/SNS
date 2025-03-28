import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os


class WeatherDataProcessor:
    def __init__(self):
        self.scalers = {
            'min_temp': MinMaxScaler(),
            'max_temp': MinMaxScaler(),
            'rain': MinMaxScaler(),
            'humidity': MinMaxScaler(),
            'wind_speed': MinMaxScaler(),
            'pressure': MinMaxScaler(),
            'visibility': MinMaxScaler(),
            'cloud_cover': MinMaxScaler(),
            'wind_direction_numerical': MinMaxScaler()
        }
        self.history_data = None
        self.all_weather_data = None
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
        self.location_data = {}

    def load_data(self, history_file='weatherHistory.csv', all_weather_file='all_weather_data.csv'):

        print("Load data set...")
        history_path = os.path.join(self.data_dir, history_file)
        weather_path = os.path.join(self.data_dir, all_weather_file)

        self.history_data = pd.read_csv(history_path)
        self.all_weather_data = pd.read_csv(weather_path)

        print("Processing date format...")
        self.history_data['date'] = pd.to_datetime(
            self.history_data['Formatted Date'].apply(lambda x: x.split('+')[0].strip())
        ).dt.date
        self.all_weather_data['date'] = pd.to_datetime(self.all_weather_data['date']).dt.date


        print("Organize data by location...")
        locations = self.all_weather_data['location'].unique()
        for location in locations:
            self.location_data[location] = self.all_weather_data[
                self.all_weather_data['location'] == location
            ].sort_values('date')

        print("Fit data normalizer...")

        feature_map = {
            'min_temp': ('min_temp °c', self.all_weather_data),
            'max_temp': ('max_temp °c', self.all_weather_data),
            'rain': ('rain mm', self.all_weather_data),
            'humidity': ('humidity %', self.all_weather_data),
            'cloud_cover': ('cloud_cover %', self.all_weather_data),
            'wind_speed': ('wind_speed km/h', self.all_weather_data),
            'wind_direction_numerical': ('wind_direction_numerical', self.all_weather_data),
            'pressure': ('Pressure (millibars)', self.history_data),
            'visibility': ('Visibility (km)', self.history_data)
        }

        for feature, (column, df) in feature_map.items():
            if column in df.columns:
                data = df[column].dropna().values.reshape(-1, 1)
                self.scalers[feature].fit(data)
                print(f"fitted {feature} normalizer")

        print("Data loading and preprocessing are completed")

    def get_available_locations(self):

        return sorted(list(self.location_data.keys()))

    def clean_feature_data(self, feature, data):

        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        total = len(data)
        cleaned = data[(data >= lower_bound) & (data <= upper_bound)]
        removed = total - len(cleaned)
        if removed > 0:
            print(f"trait {feature} remove {removed} outlier")

        return cleaned.values

    def prepare_feature_data(self, feature, location=None):

        try:
            feature_map = {
                'min_temp': ('min_temp °c', -30, 40),
                'max_temp': ('max_temp °c', -20, 50),
                'rain': ('rain mm', 0, 500),
                'humidity': ('humidity %', 0, 100),
                'cloud_cover': ('cloud_cover %', 0, 100),
                'wind_speed': ('wind_speed km/h', 0, 200),
                'wind_direction_numerical': ('wind_direction_numerical', 0, 360),
                'pressure': ('Pressure (millibars)', 870, 1090),
                'visibility': ('Visibility (km)', 0, 100)
            }

            column, min_val, max_val = feature_map.get(feature, (None, None, None))
            if column is None:
                raise KeyError(f"Feature not found {feature} Mapping")

            # 获取数据
            if feature in ['pressure', 'visibility']:
                data = self.history_data
            else:
                if location:
                    data = self.location_data.get(location)
                    if data is None:
                        raise ValueError(f"Location not found {location} data")
                else:
                    data = self.all_weather_data

            # 获取特征数据
            feature_data = data[column]

            # 清洗数据
            feature_data = self.clean_feature_data(feature, feature_data)

            # 应用物理约束
            feature_data = np.clip(feature_data, min_val, max_val)

            return feature_data

        except Exception as e:
            print(f"Processing characteristics {feature} error occurs: {str(e)}")
            return None

    def get_latest_sequence(self, feature, location, sequence_length=7, days_ahead=1):

        try:
            feature_map = {
                'min_temp': ('min_temp °c', self.all_weather_data),
                'max_temp': ('max_temp °c', self.all_weather_data),
                'rain': ('rain mm', self.all_weather_data),
                'humidity': ('humidity %', self.all_weather_data),
                'cloud_cover': ('cloud_cover %', self.all_weather_data),
                'wind_speed': ('wind_speed km/h', self.all_weather_data),
                'wind_direction_numerical': ('wind_direction_numerical', self.all_weather_data),
                'pressure': ('Pressure (millibars)', self.history_data),
                'visibility': ('Visibility (km)', self.history_data)
            }

            column, _ = feature_map.get(feature, (None, None))
            if column is None:
                raise KeyError(f"Feature not found {feature} Mapping")

            if feature in ['pressure', 'visibility']:
                df = self.history_data
            else:
                if location not in self.location_data:
                    raise ValueError(f"Not found location： {location} data")
                df = self.location_data[location]


            df = df.sort_values('date')


            if feature in ['pressure', 'visibility']:
                daily_data = df.groupby('date')[column].mean().values
            else:
                daily_data = df[column].values


            latest_sequence = daily_data[-sequence_length:]
            return self.scalers[feature].transform(latest_sequence.reshape(-1, 1)).ravel()

        except Exception as e:
            print(f"Acquire feature {feature} Error: {str(e)}")
            raise

    def inverse_transform(self, feature, scaled_value):

        try:
            return float(self.scalers[feature].inverse_transform([[scaled_value]])[0][0])
        except Exception as e:
            print(f"Transformation feature {feature} Error : {str(e)}")
            raise

    def get_available_features(self):

        return list(self.scalers.keys())