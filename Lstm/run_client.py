import requests
import sys
from src.config import SERVER_CONFIG


class WeatherClient:
    def __init__(self):
        self.base_url = f"http://{SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}"

        self.feature_aliases = {

            'temperature': 'max_temp',
            'temp': 'max_temp',
            'maxtemp': 'max_temp',
            'max_temp': 'max_temp',
            'mintemp': 'min_temp',
            'min_temp': 'min_temp',

            'rain': 'rain',
            'rainfall': 'rain',
            'precipitation': 'rain',

            'humidity': 'humidity',
            'humid': 'humidity',

            'cloud': 'cloud_cover',
            'cloud_cover': 'cloud_cover',
            'clouds': 'cloud_cover',

            'wind': 'wind_speed',
            'windspeed': 'wind_speed',
            'wind_speed': 'wind_speed',
            'winddirection': 'wind_direction_numerical',
            'wind_direction': 'wind_direction_numerical',
            'wind_direction_numerical': 'wind_direction_numerical',

            'pressure': 'pressure',
            'atmospheric_pressure': 'pressure',

            'visibility': 'visibility',
            'visible': 'visibility',

            'all': 'all',
            'everything': 'all',
            'all_features': 'all'
        }

        self.feature_descriptions = {
            'max_temp': 'Maximum Temperature (°C)',
            'min_temp': 'Minimum Temperature (°C)',
            'rain': 'Rainfall (mm)',
            'humidity': 'Humidity (%)',
            'cloud_cover': 'Cloud Coverage (%)',
            'wind_speed': 'Wind Speed (km/h)',
            'wind_direction_numerical': 'Wind Direction (degrees)',
            'pressure': 'Atmospheric Pressure (millibars)',
            'visibility': 'Visibility (km)',
            'all': 'All Features'
        }

    def get_canonical_feature(self, feature):

        return self.feature_aliases.get(feature.lower())

    def get_locations(self):

        try:
            response = requests.get(f'{self.base_url}/locations')
            if response.status_code == 200:
                data = response.json()
                return data.get('locations', [])
            return []
        except requests.exceptions.ConnectionError:
            print("Oracle: Cannot connect to server to fetch locations.")
            return []

    def send_weather_request(self, location, feature='all', days=1):

        try:

            canonical_feature = self.get_canonical_feature(feature)
            if not canonical_feature:
                print(f"Oracle: Unknown feature '{feature}'. Type 'features' to see available options.")
                return

            request_data = {
                'location': location,
                'command': f"predict {canonical_feature} for {location}",
                'feature': canonical_feature,
                'days_ahead': days
            }

            endpoint = '/predict/all' if canonical_feature == 'all' else '/predict'
            response = requests.post(
                f'{self.base_url}{endpoint}',
                json=request_data,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if 'error' in data:
                    print(f"Oracle: I'm sorry, {data['error']}")
                else:
                    print(f"\nOracle: Weather forecast for {location}, {days} days ahead:")
                    if canonical_feature == 'all':
                        for feat, info in data.items():
                            value = info.get('value')
                            unit = info.get('unit', '')
                            message = info.get('message', '')

                            print(f"\n{self.feature_descriptions[feat]}:")
                            print(f"Value: {value} {unit}")
                            if message:
                                print(f"Note: {message}")
                    else:
                        value = data.get('prediction')
                        unit = data.get('unit', '')
                        message = data.get('message', '')
                        print(f"\n{self.feature_descriptions[canonical_feature]}:")
                        print(f"Value: {value} {unit}")
                        if message:
                            print(f"Note: {message}")
            else:
                print(f"Oracle: Server returned an error (Status code: {response.status_code})")

        except requests.exceptions.ConnectionError:
            print("Oracle: Cannot connect to the server. Please check if the server is running.")
        except requests.exceptions.Timeout:
            print("Oracle: Request timed out. The server took too long to respond.")
        except Exception as e:
            print(f"Oracle: An error occurred: {str(e)}")

    def show_help(self):

        print("\nOracle: Available commands:")
        print("1. Check weather:")
        print("   weather <location> [days] [feature]")
        print("   Example: weather Beijing 2 temperature")
        print("   Example: weather Shanghai 1 all")
        print("\n2. List commands:")
        print("   locations - Show available locations")
        print("   features - Show available features")
        print("   help - Show this help message")
        print("   quit/exit - Exit the program")
        print("\nPopular feature keywords:")
        print("- temperature (or temp, maxtemp, mintemp)")
        print("- rain (or rainfall, precipitation)")
        print("- wind (or windspeed, winddirection)")
        print("- humidity")
        print("- clouds")
        print("- pressure")
        print("- visibility")
        print("- all (show all features)")

    def show_features(self):

        print("\nOracle: Available weather features:")
        for feature, description in self.feature_descriptions.items():
            aliases = [k for k, v in self.feature_aliases.items() if v == feature and k != feature]
            if aliases:
                print(f"- {description}")
                print(f"  Alternative keywords: {', '.join(aliases)}")
            else:
                print(f"- {description}")


def parse_command(command, locations):

    parts = command.strip().lower().split()

    if not parts:
        return None, None, None, None

    cmd = parts[0]

    if cmd == 'weather' and len(parts) >= 2:
        location = parts[1].title()
        if location not in locations:
            print(f"Oracle: Sorry, {location} is not in my list of available locations")
            return None, None, None, None

        days = 1
        feature = 'all'

        if len(parts) >= 3:
            try:
                days = int(parts[2])
                if not 1 <= days <= 7:
                    print("Oracle: Please choose between 1 and 7 days")
                    return None, None, None, None

                if len(parts) >= 4:
                    feature = ' '.join(parts[3:])
            except ValueError:
                feature = ' '.join(parts[2:])

        return cmd, location, days, feature

    return cmd, None, None, None


def main():
    client = WeatherClient()
    print("Oracle: Hello, I'm the Oracle. How can I help you today?")
    print("Oracle: I'm connecting to the server and getting available locations...")

    locations = client.get_locations()
    if locations:
        print(f"Oracle: Available locations: {', '.join(locations)}")
    else:
        print("Oracle: I'm having trouble getting the location list.")
        return

    print("\nType 'help' for available commands")

    while True:
        try:
            user_input = input("\n$ ").strip().lower()

            if user_input in ['exit', 'quit', 'bye']:
                print("Oracle: Goodbye!")
                break

            elif user_input == 'help':
                client.show_help()

            elif user_input == 'locations':
                print(f"Oracle: Available locations: {', '.join(locations)}")

            elif user_input == 'features':
                client.show_features()

            else:
                cmd, location, days, feature = parse_command(user_input, locations)
                if cmd == 'weather' and location:
                    client.send_weather_request(location, feature, days)

        except KeyboardInterrupt:
            print("\nOracle: Goodbye!")
            break
        except Exception as e:
            print(f"Oracle: Sorry, something went wrong: {str(e)}")


if __name__ == '__main__':
    main()