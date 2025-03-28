from flask import Blueprint, request, jsonify
from models.predictor import MultiWeatherPredictor
from data.processor import WeatherDataProcessor
from src.config import MODEL_CONFIG, DATA_CONFIG
import os

weather_bp = Blueprint('weather', __name__)

predictor = None
data_processor = None


def get_feature_unit(feature):
    """return the unit of the given feature"""
    units = {
        'min_temp': '°C',
        'max_temp': '°C',
        'rain': 'mm',
        'humidity': '%',
        'cloud_cover': '%',
        'wind_speed': 'km/h',
        'wind_direction_numerical': 'degrees',
        'pressure': 'millibars',
        'visibility': 'km'
    }
    return units.get(feature, '')


def init_models():
    #"""**Initialize models and data processors**"""
    global predictor, data_processor

    try:
        data_processor = WeatherDataProcessor()
        print("Loading data files...")
        data_processor.load_data(DATA_CONFIG['history_file'], DATA_CONFIG['weather_file'])

        predictor = MultiWeatherPredictor(sequence_length=MODEL_CONFIG['sequence_length'])

        print("Loading models...")
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'saved')

        features = [
            'min_temp', 'max_temp', 'rain', 'humidity', 'cloud_cover',
            'wind_speed', 'wind_direction_numerical', 'pressure', 'visibility'
        ]

        models_loaded = 0
        for feature in features:
            model_path = os.path.join(model_dir, f'{feature}_model.pth')
            if os.path.exists(model_path):
                try:
                    predictor.load_saved_model(feature, f'{feature}_model.pth')
                    print(f"**Model loaded successfully**：{feature}")
                    models_loaded += 1
                except Exception as e:
                    print(f"Loading {feature} unsuccessfully: {str(e)}")

        if models_loaded == 0:
            print("**Warning: No models were successfully loaded.**")
            return False

        print(f"successfully loading {models_loaded} models")
        return True
    except Exception as e:
        print(f"**Initialization failed.**: {str(e)}")
        return False


@weather_bp.route('/', methods=['GET'])
def home():
    return '''
<html>
<head>
    <title>Weather Oracle</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #4a90e2;
            --secondary-color: #f5f6fa;
            --text-color: #2c3e50;
            --shadow-color: rgba(0, 0, 0, 0.1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: var(--text-color);
        }

        .container {
            width: 90%;
            max-width: 800px;
            background: white;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 10px 30px var(--shadow-color);
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            color: var(--primary-color);
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: #666;
            font-size: 1.1rem;
        }

        .form {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            margin: 2rem 0;
        }

        .select-group {
            display: flex;
            gap: 1rem;
        }

        .input-group {
            display: flex;
            gap: 1rem;
        }

        select, input[type="text"] {
            padding: 1rem 1.5rem;
            font-size: 1rem;
            border: 2px solid #eee;
            border-radius: 10px;
            transition: all 0.3s ease;
            background: var(--secondary-color);
        }

        select {
            flex: 0 0 200px;
        }

        input[type="text"] {
            flex: 1;
        }

        select:focus, input[type="text"]:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1);
        }

        button {
            padding: 1rem 2rem;
            background: var(--primary-color);
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        button:hover {
            background: #357abd;
            transform: translateY(-2px);
        }

        .loading {
            display: none;
            justify-content: center;
            margin: 1rem 0;
        }

        .loading .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid var(--primary-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 1rem;
            margin-top: 2rem;
        }

        .result-card {
            background: var(--secondary-color);
            padding: 1.5rem;
            border-radius: 10px;
            display: none;
        }

        .result-card.show {
            display: block;
        }

        .result-content {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .feature-icon {
            font-size: 2rem;
            color: var(--primary-color);
        }

        .feature-text {
            font-size: 1.2rem;
        }

        .error {
            color: #e74c3c;
            background: #fdeaea;
            padding: 1rem;
            border-radius: 10px;
            margin-top: 1rem;
            display: none;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 600px) {
            .container {
                padding: 1.5rem;
            }

            .select-group, .input-group {
                flex-direction: column;
            }

            select {
                width: 100%;
                flex: none;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Weather Oracle</h1>
            <p>Advanced Weather Prediction System</p>
        </div>

        <div class="form">
            <div class="select-group">
                <select id="location" class="location-select">
                    <!-- 在JavaScript中动态填充地点 -->
                </select>
                <select id="feature" class="feature-select">
                    <option value="all">All Features</option>
                    <option value="min_temp">Minimum Temperature</option>
                    <option value="max_temp">Maximum Temperature</option>
                    <option value="rain">Rainfall</option>
                    <option value="humidity">Humidity</option>
                    <option value="cloud_cover">Cloud Cover</option>
                    <option value="wind_speed">Wind Speed</option>
                    <option value="wind_direction_numerical">Wind Direction</option>
                    <option value="pressure">Pressure</option>
                    <option value="visibility">Visibility</option>
                </select>
                <select id="days" class="days-select">
                     <option value="1">Tomorrow</option>
                     <option value="2">2 Days</option>
                     <option value="3">3 Days</option>
                     <option value="4">4 Days</option>
                     <option value="5">5 Days</option>
                     <option value="6">6 Days</option>
                     <option value="7">7 Days</option>
    </select>
            </div>
            <div class="input-group">
                <input type="text" 
                       id="question" 
                       placeholder="Ask about tomorrow's weather..."
                       onkeypress="handleKeyPress(event)">
            </div>
            <button onclick="askOracle()">
                <i class="fas fa-magic"></i>
                Predict Weather
            </button>
        </div>

        <div class="loading">
            <div class="spinner"></div>
        </div>

        <div class="results-grid" id="results-grid"></div>
        <div class="error" id="error-message"></div>
    </div>

    <script>
        const featureIcons = {
            'min_temp': 'temperature-low',
            'max_temp': 'temperature-high',
            'rain': 'cloud-rain',
            'humidity': 'droplet',
            'cloud_cover': 'cloud',
            'wind_speed': 'wind',
            'wind_direction_numerical': 'compass',
            'pressure': 'tachometer-alt',
            'visibility': 'eye'
        };

        async function loadLocations() {
            try {
                const response = await fetch('/locations');
                const data = await response.json();
                const locationSelect = document.getElementById('location');

                data.locations.forEach(location => {
                    const option = document.createElement('option');
                    option.value = location;
                    option.textContent = location;
                    locationSelect.appendChild(option);
                });
            } catch (error) {
                console.error('Failed to load locations:', error);
            }
        }

        window.onload = loadLocations;

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                askOracle();
            }
        }

        function showLoading() {
            document.querySelector('.loading').style.display = 'flex';
            document.getElementById('results-grid').innerHTML = '';
            document.getElementById('error-message').style.display = 'none';
        }

        function hideLoading() {
            document.querySelector('.loading').style.display = 'none';
        }

        function showError(message) {
            const errorDiv = document.getElementById('error-message');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
            document.getElementById('results-grid').innerHTML = '';
        }

function createResultCard(feature, value, unit, dayLabel, message = null) {
    const card = document.createElement('div');
    card.className = 'result-card show';

    const content = document.createElement('div');
    content.className = 'result-content';

    const icon = document.createElement('div');
    icon.className = 'feature-icon';
    icon.innerHTML = `<i class="fas fa-${featureIcons[feature]}"></i>`;

    const text = document.createElement('div');
    text.className = 'feature-text';
    text.innerHTML = `
        <div>${feature.replace(/_/g, ' ').toUpperCase()}</div>
        <div>${dayLabel}</div>
        <div>${value} ${unit}</div>
        ${message ? `<div class="weather-message">${message}</div>` : ''}
    `;

    content.appendChild(icon);
    content.appendChild(text);
    card.appendChild(content);

    return card;
}

async function askOracle() {
    const location = document.getElementById('location').value;
    const feature = document.getElementById('feature').value;
    const days = parseInt(document.getElementById('days').value);
    const question = document.getElementById('question').value;

    if (!location) {
        showError('Please select a location');
        return;
    }

    if (!question.trim()) {
        showError('Please enter a question');
        return;
    }

    showLoading();

    try {
        const response = await fetch(feature === 'all' ? '/predict/all' : '/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                location: location,
                command: question,
                feature: feature,
                days_ahead: days
            }),
        });

        const data = await response.json();
        hideLoading();

        if (data.error) {
            showError(data.error);
        } else {
            const resultsGrid = document.getElementById('results-grid');
            resultsGrid.innerHTML = '';

            const dayLabel = days === 1 ? 'Tomorrow' : `${days} Days ahead`;

            if (feature === 'all') {
                Object.entries(data).forEach(([feat, prediction]) => {
                    const card = createResultCard(
                        feat,
                        prediction.value,
                        prediction.unit,
                        dayLabel,
                        prediction.message
                    );
                    resultsGrid.appendChild(card);
                });
            } else {
                const card = createResultCard(
                    feature,
                    data.prediction,
                    data.unit,
                    dayLabel,
                    data.message
                );
                resultsGrid.appendChild(card);
            }
        }
    } catch (error) {
        hideLoading();
        showError('Failed to connect to the server. Please try again.');
    }
}

// **Add message style**

const styleSheet = document.createElement('style');
styleSheet.textContent = `
    .weather-message {
        margin-top: 10px;
        padding: 8px;
        background: #f0f8ff;
        border-radius: 5px;
        font-style: italic;
        color: #666;
    }
`;
document.head.appendChild(styleSheet);
    </script>
</body>
</html>
    '''


@weather_bp.route('/locations', methods=['GET'])
def get_locations():
    """**Get all available prediction locations**"""
    try:
        if data_processor is None:
            if not init_models():
                return jsonify({'error': 'Failed to initialize models.'})

        locations = data_processor.get_available_locations()
        return jsonify({'locations': locations})
    except Exception as e:
        return jsonify({'error': str(e)})


@weather_bp.route('/predict', methods=['POST'])
def predict():
    try:
        if predictor is None or data_processor is None:
            if not init_models():
                return jsonify({
                    'error': 'Failed to initialize models. Please check server logs.'
                })

        data = request.json
        location = data.get('location')
        feature = data.get('feature')
        days_ahead = int(data.get('days_ahead', 1))  # 获取预测天数

        if feature == 'all':
            return predict_all(location, days_ahead)

        if feature in data_processor.get_available_features():
            sequence = data_processor.get_latest_sequence(feature, location)
            scaled_predictions = predictor.predict_multiple_days(feature, sequence, max(days_ahead, 1))

            # **Only return prediction results for the specified number of days.**
            original_pred = data_processor.inverse_transform(feature, scaled_predictions[days_ahead - 1])

            # **Add weather hint statement.**
            weather_message = get_weather_message(feature, float(original_pred))

            return jsonify({
                'feature': feature,
                'prediction': round(float(original_pred), 2),
                'unit': get_feature_unit(feature),
                'message': weather_message
            })
        else:
            return jsonify({
                'error': 'Unsupported weather feature.'
            })

    except Exception as e:
        return jsonify({'error': str(e)})


def get_weather_message(feature, value):
    """**Return corresponding hint information based on weather features and prediction values**"""
    messages = {
        'min_temp': [
            {'threshold': 5, 'messages': [
                "The temperature is very low, please keep warm and wear a thick coat when going out!",
                    "Today the temperature is very low, it is recommended to wear more clothes~",
                "The weather is cold, remember to wear gloves and a scarf.!"
            ]},
            {'threshold': 15, 'messages': [
                "The temperature difference between morning and evening is large, it is recommended to add clothing appropriately.。",
                "The weather is refreshing, remember to bring a jacket when going out.~",
                "The temperature is suitable, but pay attention to keeping warm."
            ]},
            {'threshold': float('inf'), 'messages': [
                "The temperature is comfortable, you can travel lightly.。",
                "The weather is nice, just wearing a light jacket will be enough~",
                "The temperature is pleasant, wish you a pleasant outing！"
            ]}
        ],
        'max_temp': [
            {'threshold': 20, 'messages': [
                "The temperature is moderate, very suitable for outdoor activities.",
                "The weather is comfortable, wishing you a pleasant day~",
                "The temperature is pleasant, it's a great day for an outing!"
            ]},
            {'threshold': 30, 'messages': [
                "The weather is a bit hot, remember to protect yourself from the sun!!",
                "The temperature is relatively high, remember to bring a sunshade umbrella~",
                "The temperature is on the higher side, make sure to drink plenty of water!!"
            ]},
            {'threshold': float('inf'), 'messages': [
                "The temperature is very high, please take precautions against heatstroke!",
                "It's especially hot today, it is recommended to reduce outdoor activities.",
                "High temperature weather, remember to take sun protection measures!"
            ]}
        ],
        'rain': [
            {'threshold': 0.1, 'messages': [
                "The weather is sunny, suitable for outdoor activities.",
                "There will be no rain today, perfect for drying clothes~",
                "Nice weather without rain, wish you a pleasant day!"
            ]},
            {'threshold': 10, 'messages': [
                "Light rain is possible, consider bringing an umbrella~",
                "It might drizzle today, remember to bring an umbrella!",
                "Scattered light rain, be prepared with rain gear when going out."
            ]},
            {'threshold': float('inf'), 'messages': [
                "Heavy rain is expected, please bring rain gear!",
                "There will be heavy rainfall today, take precautions against rain and slippery roads.",
                "Severe rainfall is expected, it is recommended to reduce outdoor activities!"
            ]}
        ],
        'humidity': [
            {'threshold': 40, 'messages': [
                "The air is a bit dry, remember to drink more water.",
                "The humidity is low, stay hydrated~",
                "It's dry today, consider using a humidifier."
            ]},
            {'threshold': 70, 'messages': [
                "Humidity is moderate, the weather is very comfortable.",
                "The humidity is suitable today, very pleasant~",
                "The air humidity is just right, wish you a happy mood!"
            ]},
            {'threshold': float('inf'), 'messages': [
                "High humidity, pay attention to moisture resistance.",
                "Today is quite humid, remember to use a dehumidifier~",
                "Air humidity is high, it is recommended to turn on the air conditioner for dehumidification."
            ]}
        ],
        'wind_speed': [

            {'threshold': 10, 'messages': [
                "A gentle breeze, the weather is great.",
                "There's little wind today, perfect for going out~",
                "A light breeze, how pleasant!"
            ]},
            {'threshold': 20, 'messages': [
                "A bit windy, dress warmly when going outside.",
                "The wind speed is moderate today, bring a jacket for comfort~",
                "Gentle winds, take care not to catch a chill!"
            ]},
            {'threshold': float('inf'), 'messages': [
                "Strong winds, be careful when going out!",
                "It's very windy today, remember to wear something warm.",
                "High winds, it's best to limit outdoor activities!"
            ]}
        ]
    }

    import random

    if feature not in messages:
        return None

    for threshold_data in messages[feature]:
        if value <= threshold_data['threshold']:
            return random.choice(threshold_data['messages'])

    return None


@weather_bp.route('/predict/all', methods=['POST'])
def predict_all(location=None, days_ahead=1):
    """Predict all available weather features"""
    try:
        if predictor is None or data_processor is None:
            if not init_models():
                return jsonify({
                    'error': 'Failed to initialize models. Please check server logs.'
                })

        if location is None:
            data = request.json
            location = data.get('location')
            days_ahead = int(data.get('days_ahead', 1))

        predictions = {}
        features = [
            'min_temp', 'max_temp', 'rain', 'humidity', 'cloud_cover',
            'wind_speed', 'wind_direction_numerical', 'pressure', 'visibility'
        ]

        for feature in features:
            try:
                sequence = data_processor.get_latest_sequence(feature, location)
                # Get multi-day predictions
                scaled_predictions = predictor.predict_multiple_days(feature, sequence, days_ahead)
                # Only take prediction results for the specified number of days
                original_pred = data_processor.inverse_transform(feature, scaled_predictions[days_ahead - 1])

                # Get weather hint message
                weather_message = get_weather_message(feature, float(original_pred))

                predictions[feature] = {
                    'value': round(float(original_pred), 2),
                    'unit': get_feature_unit(feature),
                    'message': weather_message
                }

            except Exception as e:
                print(f"预测特征 {feature} 时出错: {str(e)}")
                continue

        if not predictions:
            return jsonify({'error': 'No predictions available.'})

        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)})


@weather_bp.route('/features', methods=['GET'])
def get_features():
    """Get all available predictive features"""
    try:
        if data_processor is None:
            if not init_models():
                return jsonify({
                    'error': 'Failed to initialize models.'
                })

        features = data_processor.get_available_features()
        feature_info = {}

        for feature in features:
            feature_info[feature] = {
                'name': feature.replace('_', ' ').title(),
                'unit': get_feature_unit(feature)
            }

        return jsonify({
            'features': feature_info
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@weather_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    status = {
        'status': 'healthy',
        'models_loaded': predictor is not None,
        'data_processor_initialized': data_processor is not None
    }

    if predictor is not None and data_processor is not None:
        status['available_features'] = data_processor.get_available_features()

    return jsonify(status)


@weather_bp.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@weather_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500