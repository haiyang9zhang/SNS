import requests
import sys
from src.config import SERVER_CONFIG


class WeatherClient:
    def __init__(self):
        self.base_url = f"http://{SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}"

    def send_command(self, command):
        try:
            response = requests.post(f'{self.base_url}/predict',
                                     json={'command': command})

            if response.status_code == 200:
                data = response.json()
                if 'error' in data:
                    print(f"Oracle: I'm sorry, {data['error']}")
                else:
                    print(f"Oracle: Tomorrow's maximum temperature will be {data['prediction']} degrees {data['unit']}")
            else:
                print("Oracle: Sorry, I'm having trouble connecting to my prediction services.")

        except requests.exceptions.ConnectionError:
            print("Oracle: Sorry, I cannot connect to the server at the moment.")


def main():
    client = WeatherClient()
    print("Oracle: Hello, I'm the Oracle. How can I help you today?")

    while True:
        try:
            command = input("$ ")
            if command.lower() in ['exit', 'quit', 'bye']:
                print("Oracle: Goodbye!")
                break
            client.send_command(command)
        except KeyboardInterrupt:
            print("\nOracle: Goodbye!")
            break


if __name__ == '__main__':
    main()