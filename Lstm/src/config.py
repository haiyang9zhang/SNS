import os

# 基础路径
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# 模型配置
MODEL_CONFIG = {
    'sequence_length': 7,
    'hidden_size': 64,
    'epochs': 2,
    'batch_size': 32,
    'learning_rate': 0.001,
    'early_stopping': 20,
    'validation_split': 0.2,
}

# 服务器配置
SERVER_CONFIG = {
    'host': 'localhost',
    'port': 5000,
    'debug': False
}

# 数据配置
DATA_CONFIG = {
    'history_file': 'weatherHistory.csv',
    'weather_file': 'all_weather_data.csv',
    'data_dir': os.path.join(BASE_DIR, 'data', 'raw')
}